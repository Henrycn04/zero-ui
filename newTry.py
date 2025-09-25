#!/usr/bin/env python3
# depth_pointing_system.py
# Sistema completo para Intel RealSense D455 (solo profundidad):
#  - Captura y etiquetado ligero en vivo (opcional)
#  - Modelo CNN (>3 capas) con 3 salidas (presencia, señalando, destino)
#  - Entrenamiento con pérdidas enmascaradas
#  - Inferencia en tiempo real + publicación MQTT para IoT
#
# Requisitos:
#   pip install pyrealsense2 opencv-python torch torchvision numpy pandas paho-mqtt pyyaml
#
# Uso rápido:
#   1) Crear config base de zonas/MQTT:
#      python depth_pointing_system.py configure --zones puerta,ventana,mesa --mqtt-host 192.168.1.50
#   2) Capturar datos (con rotulado interactivo opcional):
#      python depth_pointing_system.py capture --out data/s1 --label-live
#   3) (Opcional) Revisar/editar labels.csv generado.
#   4) Entrenar:
#      python depth_pointing_system.py train --data-csv data/s1/labels.csv --epochs 20 --out runs/exp1
#   5) Inferencia + IoT:
#      python depth_pointing_system.py live --weights runs/exp1/best.pt
#
# Estructura de labels.csv:
#   path,presence,pointing,target
#   data/s1/img_000123.png,1,1,2   # 2 => id de zona (1..K), 0=ninguna
#   data/s1/img_000124.png,1,0,0

import argparse, os, time, sys, json, yaml, math, random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F

# MQTT
try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

# RealSense (solo necesario para capture / live)
try:
    import pyrealsense2 as rs
except Exception:
    rs = None

# ============================ Utilidades ============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# -------- Focal Loss binaria (logits) --------
def binary_focal_with_logits(logits, targets, alpha=0.25, gamma=2.0, reduction='mean', weight=None):
    """
    logits: tensor [...], targets: {0,1} igual shape
    weight: tensor broadcastable (p.ej., peso mayor para negativos)
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p*targets + (1-p)*(1-targets)
    loss = ce * ((1 - p_t).pow(gamma))
    if alpha is not None:
        alpha_t = alpha*targets + (1-alpha)*(1-targets)
        loss = alpha_t * loss
    if weight is not None:
        loss = loss * weight
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


# --- Carga adaptativa de checkpoints (tolera cambios en #clases) ---
def _smart_init(shape, device, dtype):
    t = torch.empty(shape, dtype=dtype, device=device)
    if t.ndim >= 2:
        torch.nn.init.kaiming_normal_(t, mode='fan_out', nonlinearity='relu')
    else:
        torch.nn.init.zeros_(t)
    return t

def _adapt_and_merge_param(model_param, ckpt_param):
    mshape, kshape = tuple(model_param.shape), tuple(ckpt_param.shape)
    device, dtype = model_param.device, model_param.dtype
    if mshape == kshape:
        return ckpt_param.clone().to(dtype=dtype, device=device)
    # mismatch solo en primera dim (típico en heads de clases)
    if len(mshape) == len(kshape) and mshape[1:] == kshape[1:]:
        cm, ck = mshape[0], kshape[0]
        new_t = torch.empty(mshape, dtype=dtype, device=device)
        ncopy = min(cm, ck)
        new_t[:ncopy] = ckpt_param[:ncopy].to(device=device, dtype=dtype)
        if cm > ck:
            new_t[ncopy:] = _smart_init((cm - ncopy,) + mshape[1:], device, dtype)
        return new_t
    # shapes incompatibles -> deja el del modelo
    return model_param

def load_state_dict_adapt(model, state_dict_ckpt, verbose=True):
    model_state = model.state_dict()
    new_state, kept, adapted, skipped = {}, [], [], []
    for k_model, p_model in model_state.items():
        candidates = [k_model, f"module.{k_model}"]
        if k_model.startswith("module."):
            candidates.append(k_model[len("module."):])
        ck_key = next((c for c in candidates if c in state_dict_ckpt), None)
        if ck_key is None:
            skipped.append((k_model, 'missing_in_ckpt'))
            new_state[k_model] = p_model
            continue
        p_ck = state_dict_ckpt[ck_key]
        if tuple(p_ck.shape) == tuple(p_model.shape):
            new_state[k_model] = p_ck.clone().to(dtype=p_model.dtype, device=p_model.device); kept.append(k_model)
        else:
            p_adapt = _adapt_and_merge_param(p_model, p_ck)
            if torch.equal(p_adapt, p_model):
                skipped.append((k_model, f'shape_mismatch ckpt={tuple(p_ck.shape)} model={tuple(p_model.shape)}')); new_state[k_model] = p_model
            else:
                new_state[k_model] = p_adapt; adapted.append((k_model, f'ckpt={tuple(p_ck.shape)} model={tuple(p_model.shape)}'))
    model.load_state_dict(new_state, strict=False)
    if verbose:
        print(f"[adapt] kept={len(kept)} adapted={len(adapted)} skipped={len(skipped)}")
    return kept, adapted, skipped
import warnings
import torch
import torch.nn.functional as F

def calibrate_binary_logits(logits_np, labels_np, max_iter=100, T_min=0.3, Zmax=30.0, b_abs_limit=1e2):
    """
    Ajusta (T, b) para z = (logits - b) / T minimizando BCE-with-logits en valid.
    Evita overflow y detecta calibraciones inestables:
      - fuerza T >= T_min
      - clampa z a [-Zmax, Zmax] antes de sigmoid
      - si T termina en el límite o |b| es enorme -> devuelve None (fallback: no calibrar)
    Retorna dict {'T': float, 'b': float} o None.
    """
    # Convertir a tensores Torch
    z = torch.tensor(logits_np, dtype=torch.float32)
    y = torch.tensor(labels_np, dtype=torch.float32)

    # Parámetros óptimos iniciales
    T = torch.nn.Parameter(torch.ones(1) * 1.0)
    b = torch.nn.Parameter(torch.zeros(1))

    opt = torch.optim.LBFGS([T, b], lr=0.5, max_iter=max_iter)

    def nll_for_opt():
        # Aseguramos T mínimo antes de construir z_norm
        T_clamped = torch.clamp(T, min=T_min)
        z_norm = (z - b) / T_clamped
        # clipear z para evitar overflow numerico
        z_norm = torch.clamp(z_norm, -Zmax, Zmax)
        return F.binary_cross_entropy_with_logits(z_norm, y)

    def closure():
        opt.zero_grad()
        L = nll_for_opt()
        L.backward()
        return L

    try:
        opt.step(closure)
    except Exception as e:
        warnings.warn(f"calibrate_binary_logits: optim failed: {e}")
        return None

    # Obtener valores seguros
    with torch.no_grad():
        T_val = float(torch.clamp(T, min=T_min).item())
        b_val = float(b.item())

    # Si la optimización empujó T al límite mínimo o b es gigantesco => inestable
    if T_val <= T_min * 1.001 or abs(b_val) > b_abs_limit:
        warnings.warn(f"calibrate_binary_logits: unstable result T={T_val:.4g} b={b_val:.4g} -> ignoring calibration")
        return None

    return {'T': T_val, 'b': b_val}


def apply_calib(logit_scalar, calib):
    if calib is None:
        return logit_scalar
    T = max(1e-3, float(calib.get('T', 1.0)))
    b = float(calib.get('b', 0.0))
    return (logit_scalar - b) / T
def apply_calib_scalar(logit_scalar, calib, Zmax=30.0):
    if calib is None:
        return logit_scalar  # caller tomará sigmoid si quiere
    T = max(1e-3, float(calib.get('T', 1.0)))
    b = float(calib.get('b', 0.0))
    z = (logit_scalar - b) / T
    z = float(np.clip(z, -Zmax, Zmax))
    return z

# -------- ROI para dataset (igual idea que en live) --------
def crop_roi_np(d16: np.ndarray, pad: int = 10) -> np.ndarray:
    m = (d16 > 0).astype(np.uint8)
    ys, xs = np.where(m)
    if xs.size > 0 and ys.size > 0:
        x0 = max(0, int(xs.min()) - pad); x1 = min(d16.shape[1], int(xs.max()) + pad)
        y0 = max(0, int(ys.min()) - pad); y1 = min(d16.shape[0], int(ys.max()) + pad)
        return d16[y0:y1, x0:x1]
    return d16
import numpy as np

def _align_arrays(y_true, y_prob, name="presence"):
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape[0] != y_prob.shape[0]:
        n = min(y_true.shape[0], y_prob.shape[0])
        print(f"[WARN] {name}: mismatch y_true={y_true.shape[0]} y_prob={y_prob.shape[0]} -> trunc {n}")
        y_true = y_true[:n]
        y_prob = y_prob[:n]
    return y_true, y_prob

def best_precision_threshold(y_true, y_prob, target_prec=0.97, name="presence"):
    """
    Elige el umbral más alto que consiga al menos 'target_prec' de precisión.
    Si ninguno lo alcanza, devuelve el de mayor precisión observada.
    """
    y_true, y_prob = _align_arrays(y_true, y_prob, name)

    # ordenar por prob desc
    order = np.argsort(-y_prob)
    yt = y_true[order]
    yp = y_prob[order]

    tp = 0
    fp = 0
    best_thr = 0.5
    best_prec = 0.0

    # barrido descendente: al bajar el umbral, vas sumando positivos predichos
    last = None
    for i in range(len(yp)):
        # incluir este punto como positivo predicho
        if yt[i] == 1:
            tp += 1
        else:
            fp += 1

        # cuando cambia el valor de prob (o al final), evalúa la precisión en ese umbral
        if i == len(yp) - 1 or yp[i+1] != yp[i]:
            prec = tp / (tp + fp + 1e-9)
            thr_here = yp[i]
            if prec >= target_prec:
                best_thr = float(thr_here)
                best_prec = float(prec)
                break
            if prec > best_prec:
                best_prec = float(prec)
                best_thr = float(thr_here)

    return best_prec, best_thr

def best_f1_threshold(y_true, y_prob, name="pointing"):
    """
    Elige el umbral que maximiza F1.
    """
    y_true, y_prob = _align_arrays(y_true, y_prob, name)

    order = np.argsort(-y_prob)
    yt = y_true[order]
    yp = y_prob[order]

    tp = 0
    fp = 0
    fn = int(yt.sum())
    best_f1 = 0.0
    best_thr = 0.5

    for i in range(len(yp)):
        # al cruzar este punto, lo consideramos positivo predicho
        if yt[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1

        # evaluar en los saltos de prob
        if i == len(yp) - 1 or yp[i+1] != yp[i]:
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_thr = float(yp[i])

    return best_f1, best_thr


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def normalize_depth(d16: np.ndarray, near_mm: int = 300, far_mm: int = 4000) -> np.ndarray:
    """Recorta a [near, far] y normaliza a [0,1]. Zeros o NaNs a 0."""
    d = d16.astype(np.float32)
    d[d <= 0] = np.nan
    d = np.clip(d, near_mm, far_mm)
    # Normalizar
    d = (d - near_mm) / max(1, (far_mm - near_mm))
    d = np.nan_to_num(d, nan=0.0)
    return d


def depth_to_png_u16(d16: np.ndarray) -> np.ndarray:
    """Convierte depth en mm a PNG 16-bit (para guardar sin perder rango)."""
    d = d16.astype(np.uint16)
    return d


def png_u16_to_depth_mm(img: np.ndarray) -> np.ndarray:
    return img.astype(np.uint16)


# ======================== Modelo CNN (3 salidas) ====================
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x

# === Helper para CoordConv: concatena mapas X,Y normalizados [-1,1] ===
class AddXY(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # x: [B,1,H,W] -> [B,3,H,W]
        B, _, H, W = x.shape
        yy = torch.linspace(-1, 1, steps=H, device=x.device).view(1,1,H,1).expand(B,1,H,W)
        xx = torch.linspace(-1, 1, steps=W, device=x.device).view(1,1,1,W).expand(B,1,H,W)
        return torch.cat([x, xx, yy], dim=1)


class DepthPointingNet(nn.Module):
    """
    Backbone CNN simple (>3 capas):
      (stem1: 1->16) ó (addXY + stem3: 3->16)  -> [ConvBlock(16->32)] -> [ConvBlock(32->64)] -> [ConvBlock(64->128)]
      GAP -> heads
    Heads:
      - presence: 1 logit
      - pointing: 1 logit
      - target: (K+1) clases (0 = none)
      - row: (R+1) clases (0 = none)
      - col: (C+1) clases (0 = none)
    """
    def __init__(self, num_targets: int, grid_cols: int = 3, use_coord: bool = False):
        super().__init__()
        self.num_targets = num_targets
        self.grid_cols = max(1, int(grid_cols))
        self.num_rows = max(1, math.ceil(num_targets / self.grid_cols)) if num_targets > 0 else 1

        # --- CoordConv ---
        self.use_coord = bool(use_coord)
        self.addxy = AddXY()                 # añade canales X,Y si use_coord=True
        self.stem1 = ConvBlock(1, 16)        # entrada estándar 1 canal (depth)
        self.stem3 = ConvBlock(3, 16)        # entrada 3 canales (depth + X + Y)

        # Resto del backbone (a partir de 16 canales)
        self.block2 = ConvBlock(16, 32)
        self.block3 = ConvBlock(32, 64)
        self.block4 = ConvBlock(64, 128)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Heads
        self.presence = nn.Linear(128, 1)
        self.pointing = nn.Linear(128, 1)
        self.target   = nn.Linear(128, num_targets + 1)   # 0..K
        self.row      = nn.Linear(128, self.num_rows + 1) # 0..R
        self.col      = nn.Linear(128, self.grid_cols + 1)# 0..C

    def forward(self, x):
        # x: [B,1,H,W]
        if self.use_coord:
            x = self.addxy(x)     # [B,3,H,W]
            x = self.stem3(x)     # 3 -> 16
        else:
            x = self.stem1(x)     # 1 -> 16

        x = self.block2(x)        # 16 -> 32
        x = self.block3(x)        # 32 -> 64
        x = self.block4(x)        # 64 -> 128
        f = self.gap(x).flatten(1)

        return {
            'presence': self.presence(f),  # logits
            'pointing': self.pointing(f),  # logits
            'target':   self.target(f),    # logits
            'row':      self.row(f),       # logits
            'col':      self.col(f)        # logits
        }



# ======================== Dataset y DataLoader ======================
@dataclass
class TrainConfig:
    input_size: int = 224
    near_mm: int = 300
    far_mm: int = 4000
    grid_cols: int = 3
    aug_flip: bool = True      # flip horizontal + remapeo de target
    aug_jitter: bool = True    # gain/bias en profundidad normalizada
    aug_roi: bool = True       # recorte por masa de profundidad
    jitter_gain_min: float = 0.9
    jitter_gain_max: float = 1.1
    jitter_bias_abs: float = 0.05

class DepthPointingDataset(Dataset):
    def __init__(self, csv_path: str, cfg: TrainConfig):
        if isinstance(csv_path, (list, tuple)):
            dfs = []
            for p in csv_path:
                df_i = pd.read_csv(p)
                dfs.append(df_i)
            self.df = pd.concat(dfs, ignore_index=True).dropna()
        else:
            self.df = pd.read_csv(csv_path).dropna()
        self.cfg = cfg
        for col in ['path','presence','pointing','target']:
            assert col in self.df.columns, f"Falta columna {col} en {csv_path}"
        # Tipos
        self.df['presence'] = self.df['presence'].astype(int)
        self.df['pointing'] = self.df['pointing'].astype(int)
        self.df['target']   = self.df['target'].astype(int)
        # 'row'/'column' pueden existir, pero en train derivamos desde 'target'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        path = r['path']
        d16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d16 is None:
            raise FileNotFoundError(path)
        d16 = png_u16_to_depth_mm(d16)

        # Copia para FULL y ROI
        d_full = d16.copy()
        d_roi = d16.copy()


        # ROI SOLO en positivos (presence=1). En negativos dejamos el frame completo como ROI "negativo".
        if self.cfg.aug_roi and int(r['presence']) == 1:
            d_roi = crop_roi_np(d_roi)

        # ---------- Aumentos CONSISTENTES ----------
        # Flip horizontal + remapeo del target (aplicar IGUAL a d_full y d_roi)
        tgt = int(r['target'])
        if self.cfg.aug_flip and random.random() < 0.5:
            d_full = np.fliplr(d_full).copy()
            d_roi = np.fliplr(d_roi).copy()
            if tgt > 0:
                Kc = int(self.cfg.grid_cols)
                rr = ((tgt - 1) // Kc) + 1
                cc = ((tgt - 1) % Kc) + 1
                cc = (Kc + 1) - cc
                tgt = (rr - 1) * Kc + cc

        # Normaliza SIEMPRE
        d_full = normalize_depth(d_full, self.cfg.near_mm, self.cfg.far_mm)
        d_roi = normalize_depth(d_roi, self.cfg.near_mm, self.cfg.far_mm)

        # Jitter (ganancia/bias) opcional DESPUÉS de normalizar
        if self.cfg.aug_jitter and random.random() < 0.5:
            g = random.uniform(self.cfg.jitter_gain_min, self.cfg.jitter_gain_max)  # p.ej. 0.9..1.1
            b = random.uniform(-self.cfg.jitter_bias_abs, self.cfg.jitter_bias_abs)  # ±0.05
            d_full = np.clip(g * d_full + b, 0.0, 1.0)
            d_roi = np.clip(g * d_roi + b, 0.0, 1.0)

        if random.random() < 0.30:
            sigma = 0.03

            def _speckle(x):
                n = np.random.randn(*x.shape).astype(np.float32) * sigma
                return np.clip(x * (1.0 + n), 0.0, 1.0)

            d_full = _speckle(d_full)
            d_roi = _speckle(d_roi)

        # Resize a input_size
        d_full = cv2.resize(d_full, (self.cfg.input_size, self.cfg.input_size), interpolation=cv2.INTER_NEAREST)
        d_roi = cv2.resize(d_roi, (self.cfg.input_size, self.cfg.input_size), interpolation=cv2.INTER_NEAREST)

        x_full = torch.from_numpy(d_full).float().unsqueeze(0)  # [1,H,W]
        x_roi = torch.from_numpy(d_roi).float().unsqueeze(0)

        # Etiquetas
        y_presence = int(r['presence'])
        y_pointing = int(r['pointing'])
        y_target = int(tgt)

        # Derivar fila/col
        if y_target > 0:
            Kc = int(self.cfg.grid_cols)
            y_row = (y_target - 1) // Kc + 1
            y_col = (y_target - 1) % Kc + 1
        else:
            y_row = 0
            y_col = 0

        y_presence = torch.tensor(y_presence, dtype=torch.float32)
        y_pointing = torch.tensor(y_pointing, dtype=torch.float32)
        y_target = torch.tensor(y_target, dtype=torch.long)
        y_row = torch.tensor(y_row, dtype=torch.long)
        y_col = torch.tensor(y_col, dtype=torch.long)

        # OJO: ahora devolvemos (x_full, x_roi, labels...)
        return x_full, x_roi, y_presence, y_pointing, y_target, y_row, y_col


# =========================== Entrenamiento ==========================

def masked_losses(outputs, y_presence, y_pointing, y_target, y_row=None, y_col=None,
                  w_row=0.5, w_col=0.5, w_none=0.3):
    """
    Jerarquía:
      - presence: siempre
      - pointing: SIEMPRE (sin máscara) → penaliza falsos positivos cuando presence=0
      - target:   positivo si pointing=1; negativo empujando a clase 0 si pointing=0
      - row/col:  positivo si target>0; negativo empujando a clase 0 si pointing=0
    """
    device = outputs['target'].device
    ce  = nn.CrossEntropyLoss(reduction='none')

    # --- presence (igual) ---
    pres_loss = binary_focal_with_logits(
        outputs['presence'].view(-1), y_presence.view(-1),
        alpha=0.25, gamma=2.0, reduction='mean'
    )

    # --- pointing: sin máscara (aprende negativos con presence=0) ---
    point_all = binary_focal_with_logits(
        outputs['pointing'].view(-1), y_pointing.view(-1),
        alpha=0.25, gamma=2.0, reduction='mean'
    )

    # --- target: positivo si pointing=1; negativo (clase 0) si pointing=0 ---
    mask_point = (y_pointing.view(-1) >= 0.5).float()
    # positivo
    t_pos_all = ce(outputs['target'], y_target)
    t_pos = (t_pos_all * mask_point).sum() / (mask_point.sum() + 1e-6)
    # negativo (empujar a clase 0)
    zeros_t = torch.zeros_like(y_target)
    t_neg_all = ce(outputs['target'], zeros_t)
    t_neg = (t_neg_all * (1 - mask_point)).sum() / ((1 - mask_point).sum() + 1e-6)
    target_loss = 0.7 * t_pos + w_none * t_neg  # w_none ~ 0.3

    # --- row/col: positivo si target>0; negativo a clase 0 si no apunta ---
    row_loss = torch.tensor(0.0, device=device)
    col_loss = torch.tensor(0.0, device=device)
    if y_row is not None and 'row' in outputs:
        mask_grid = mask_point * (y_target > 0).float()
        r_pos_all = ce(outputs['row'], y_row)
        r_pos = (r_pos_all * mask_grid).sum() / (mask_grid.sum() + 1e-6)
        zeros_r = torch.zeros_like(y_row)
        r_neg_all = ce(outputs['row'], zeros_r)
        r_neg = (r_neg_all * (1 - mask_point)).sum() / ((1 - mask_point).sum() + 1e-6)
        row_loss = 0.7 * r_pos + w_none * r_neg

    if y_col is not None and 'col' in outputs:
        mask_grid = mask_point * (y_target > 0).float()
        c_pos_all = ce(outputs['col'], y_col)
        c_pos = (c_pos_all * mask_grid).sum() / (mask_grid.sum() + 1e-6)
        zeros_c = torch.zeros_like(y_col)
        c_neg_all = ce(outputs['col'], zeros_c)
        c_neg = (c_neg_all * (1 - mask_point)).sum() / ((1 - mask_point).sum() + 1e-6)
        col_loss = 0.7 * c_pos + w_none * c_neg

    total = pres_loss + point_all + target_loss + w_row * row_loss + w_col * col_loss
    return total, pres_loss.detach(), point_all.detach(), target_loss.detach(), row_loss.detach(), col_loss.detach()




def train_model(args):
    import numpy as np
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outdir = Path(args.out); ensure_dir(outdir)

    # === Config zonas/grid como ya lo tienes arriba ===
    with open('config.yaml', 'r') as f:
        cfg_all = yaml.safe_load(f)
    zones = cfg_all.get('zones', [])
    grid_cols = int(cfg_all.get('grid', {}).get('cols', 3))
    num_targets = len(zones)

    ### PATCH B1: configurar lambdas y focal/weights
    lam_pres = float(args.lambda_pres)
    lam_point = float(args.lambda_point)
    lam_col = float(args.lambda_col)
    lam_row = float(args.lambda_row)

    row_warmup_epochs = int(args.row_warmup_epochs)
    row_focal_gamma = float(args.row_focal_gamma)
    col_weight_rowB = float(args.col_weight_rowB)  # >1.0 penaliza más errores de columna cuando row=2 (fila B)

    # === Config de aumentos ===
    tcfg_train = TrainConfig(
        input_size=args.input, near_mm=args.near, far_mm=args.far,
        grid_cols=grid_cols,
        aug_flip=(not args.no_flip),
        aug_jitter=(not args.no_jitter),
        aug_roi=(not args.no_roi)
    )
    tcfg_val = TrainConfig(
        input_size=args.input, near_mm=args.near, far_mm=args.far,
        grid_cols=grid_cols,
        aug_flip=False,  # <-- sin flip en validación
        aug_jitter=False,  # <-- sin jitter en validación
        aug_roi=(not args.no_roi)  # ROI sí (consistencia con live)
    )

    # === Split 80/20 por índice (reproducible) ===
    # === Unir uno o varios CSVs ===
    if isinstance(args.data_csv, (list, tuple)):
        dfs = [pd.read_csv(p).dropna() for p in args.data_csv]
        full_df = pd.concat(dfs, ignore_index=True)
    else:
        full_df = pd.read_csv(args.data_csv).dropna()

    idx = np.arange(len(full_df));
    np.random.shuffle(idx)
    n = len(idx);
    n_val = max(1, int(0.2 * n))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # === Dos instancias de dataset (train/val) y luego Subset por índices ===
    ds_train_full = DepthPointingDataset(args.data_csv, tcfg_train)
    ds_val_full = DepthPointingDataset(args.data_csv, tcfg_val)

    ds_train = torch.utils.data.Subset(ds_train_full, train_idx)
    ds_val = torch.utils.data.Subset(ds_val_full, val_idx)

    # ---- Sampler balanceado por presencia (reduce sesgo "siempre 1") ----
    labels_train = ds_train_full.df.iloc[train_idx]['presence'].values.astype(int)
    class_counts = np.bincount(labels_train, minlength=2)  # [neg, pos]
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    dl_train = DataLoader(ds_train, batch_size=args.batch, sampler=sampler, drop_last=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=0)

    # Modelo + opt
    model = DepthPointingNet(num_targets=num_targets, grid_cols=grid_cols, use_coord=args.coordconv).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=2, min_lr=1e-5
    )

    best_val = float('inf');
    best_thr_pres, best_thr_point = 0.50, 0.50
    patience, bad = int(args.patience), 0

    for ep in range(1, args.epochs+1):
        best_path = outdir / ("best" + str(ep) + ".pt")
        model.train()
        tr_losses = []
        lam_row_eff = 0.0 if ep < args.row_warmup_epochs else args.lambda_row
        for batch in dl_train:
            xf, xr, yp, yg, yt, yrow, ycol = batch
            xf, xr = xf.to(device), xr.to(device)
            yp, yg, yt = yp.to(device), yg.to(device), yt.to(device)
            yrow, ycol = yrow.to(device), ycol.to(device)

            opt.zero_grad()

            # ---- PASO A: presence con FULL ----
            out_f = model(xf)
            pres_loss = binary_focal_with_logits(
                out_f['presence'].view(-1), yp.view(-1),
                alpha=0.10, gamma=2.0, reduction='mean'  # más conservador con FP
            )

            # ---- PASO B: pointing/target/row/col con ROI ----
            out_r = model(xr)

            # pointing (sin máscara)
            neg_w = 1.5
            w_point = yg.view(-1) * 1.0 + (1.0 - yg.view(-1)) * neg_w  # 1.0 en pos, 1.5 en neg
            point_loss = binary_focal_with_logits(
                out_r['pointing'].view(-1), yg.view(-1),
                alpha=0.25, gamma=2.0, reduction='mean', weight=w_point
            )

            ce = nn.CrossEntropyLoss(reduction='none')
            mask_point = (yg.view(-1) >= 0.5).float()

            # target positivo/negativo (clase 0 cuando NO apunta)
            t_all = ce(out_r['target'], yt)
            t_pos = (t_all * mask_point).sum() / (mask_point.sum() + 1e-6)
            zeros_t = torch.zeros_like(yt)
            t_neg_all = ce(out_r['target'], zeros_t)
            t_neg = (t_neg_all * (1 - mask_point)).sum() / ((1 - mask_point).sum() + 1e-6)
            target_loss = 0.7 * t_pos + 0.35 * t_neg  # w_none=0.35

            # row/col (si existen)
            row_loss = torch.tensor(0.0, device=device)
            col_loss = torch.tensor(0.0, device=device)
            if 'row' in out_r and 'col' in out_r:
                mask_grid = mask_point * (yt > 0).float()

                r_all = ce(out_r['row'], yrow)
                r_pos = (r_all * mask_grid).sum() / (mask_grid.sum() + 1e-6)
                zeros_r = torch.zeros_like(yrow)
                r_neg_all = ce(out_r['row'], zeros_r)
                r_neg = (r_neg_all * (1 - mask_point)).sum() / ((1 - mask_point).sum() + 1e-6)
                row_loss = 0.7 * r_pos + 0.35 * r_neg

                c_all = ce(out_r['col'], ycol)
                zeros_c = torch.zeros_like(ycol)

                # máscaras
                mask_grid = mask_point * (yt > 0).float()  # ya la tienes más arriba
                fila_B = (yrow == 2).float()  # fila central (B)

                # pesos: solo aplican cuando hay pointing y la fila real es B
                col_weights = torch.ones_like(c_all)
                col_weights = col_weights + (args.col_weight_rowB - 1.0) * (fila_B * mask_point)

                # CE ponderada en positivos (clases 1..C), normalizada por cantidad de positivos
                c_pos_w = (c_all * col_weights * mask_grid).sum() / (mask_grid.sum() + 1e-6)

                # negativos (clase 0 cuando NO apunta), sin ponderar extra
                c_neg_all = ce(out_r['col'], zeros_c)
                c_neg = (c_neg_all * (1 - mask_point)).sum() / ((1 - mask_point).sum() + 1e-6)

                col_loss = 0.7 * c_pos_w + 0.35 * c_neg
            target_aux_weight = 0.2
            total = (lam_pres * pres_loss
                     + lam_point * point_loss
                     + target_aux_weight * target_loss
                     + lam_row_eff * row_loss
                     + lam_col * col_loss)
            for name, val in [('pres', pres_loss), ('point', point_loss), ('targ', target_loss), ('row', row_loss),
                              ('col', col_loss), ('total', total)]:
                if torch.isnan(val).any():
                    raise ValueError(f"NaN en {name}_loss")
                if float(val.detach().cpu().item()) < -1e-6:
                    raise ValueError(f"{name}_loss negativa (= {float(val):.6f}). Revisa signos/targets.")
            total.backward()
            opt.step()

            tr_losses.append(total.item())

        train_loss = float(np.mean(tr_losses)) if tr_losses else 0.0

        # -------- Validación + F1 thresholds --------
        model.eval()
        with torch.no_grad():
            va_losses = []
            pres_true = []
            point_true = []
            pres_prob = []
            point_prob = []
            pres_logits, point_logits = [], []
            cols_true_B, cols_pred_B = [], []  # col | row=B
            rows_true_C, rows_pred_C = [], []
            for batch in dl_val:
                xf, xr, yp, yg, yt, yrow, ycol = batch
                xf, xr = xf.to(device), xr.to(device)
                yp, yg, yt = yp.to(device), yg.to(device), yt.to(device)
                yrow, ycol = yrow.to(device), ycol.to(device)

                # FULL → presence
                out_f = model(xf)
                lp = out_f['presence'].view(-1).detach().cpu().numpy().tolist()
                pres_loss = binary_focal_with_logits(
                    out_f['presence'].view(-1), yp.view(-1),
                    alpha=0.10, gamma=2.0, reduction='mean'
                )
                pres_p = torch.sigmoid(out_f['presence'].view(-1)).cpu().numpy().tolist()

                # ROI → pointing/target
                out_r = model(xr)
                lg = out_r['pointing'].view(-1).detach().cpu().numpy().tolist()
                neg_w = 1.5
                w_point = yg.view(-1) * 1.0 + (1.0 - yg.view(-1)) * neg_w
                point_loss = binary_focal_with_logits(
                    out_r['pointing'].view(-1), yg.view(-1),
                    alpha=0.25, gamma=2.0, reduction='mean', weight=w_point
                )
                ### PATCH D2: recolectar predicciones por cabeza
                if 'col' in out_r and 'row' in out_r:
                    p_col_val = torch.softmax(out_r['col'], dim=-1)  # [N, C+1]
                    p_row_val = torch.softmax(out_r['row'], dim=-1)  # [N, R+1]
                    col_hat = torch.argmax(p_col_val, dim=-1)  # [N]
                    row_hat = torch.argmax(p_row_val, dim=-1)  # [N]

                    # Solo evaluar cuando hay pointing (mask_point)
                    mp = (yg.view(-1) >= 0.5)

                    # 1) BalancedAcc[col | row=B]  (fila central B = 2)
                    mask_rowB = mp & (yrow.view(-1) == 2)
                    if mask_rowB.any():
                        cols_true_B.extend(ycol.view(-1)[mask_rowB].cpu().tolist())
                        cols_pred_B.extend(col_hat.view(-1)[mask_rowB].cpu().tolist())

                    # 2) BalancedAcc[row | col=C]  (columna central C = 2)
                    mask_colC = mp & (ycol.view(-1) == 2)
                    if mask_colC.any():
                        rows_true_C.extend(yrow.view(-1)[mask_colC].cpu().tolist())
                        rows_pred_C.extend(row_hat.view(-1)[mask_colC].cpu().tolist())

                ce = nn.CrossEntropyLoss(reduction='none')
                mask_point = (yg.view(-1) >= 0.5).float()

                t_all = ce(out_r['target'], yt)
                t_pos = (t_all * mask_point).sum() / (mask_point.sum() + 1e-6)
                zeros_t = torch.zeros_like(yt)
                t_neg_all = ce(out_r['target'], zeros_t)
                t_neg = (t_neg_all * (1 - mask_point)).sum() / ((1 - mask_point).sum() + 1e-6)
                target_loss = 0.7 * t_pos + 0.35 * t_neg

                row_loss = torch.tensor(0.0, device=device)
                col_loss = torch.tensor(0.0, device=device)
                if 'row' in out_r and 'col' in out_r:
                    mask_grid = mask_point * (yt > 0).float()

                    r_all = ce(out_r['row'], yrow)
                    r_pos = (r_all * mask_grid).sum() / (mask_grid.sum() + 1e-6)
                    zeros_r = torch.zeros_like(yrow)
                    r_neg_all = ce(out_r['row'], zeros_r)
                    r_neg = (r_neg_all * (1 - mask_point)).sum() / ((1 - mask_point).sum() + 1e-6)
                    row_loss = 0.7 * r_pos + 0.35 * r_neg

                    c_all = ce(out_r['col'], ycol)
                    c_pos = (c_all * mask_grid).sum() / (mask_grid.sum() + 1e-6)
                    zeros_c = torch.zeros_like(ycol)
                    c_neg_all = ce(out_r['col'], zeros_c)
                    c_neg = (c_neg_all * (1 - mask_point)).sum() / ((1 - mask_point).sum() + 1e-6)
                    col_loss = 0.7 * c_pos + 0.35 * c_neg

                total = (1.5 * pres_loss + point_loss + target_loss + 0.5 * row_loss + 0.5 * col_loss)
                va_losses.append(total.item())

                # recolecta probabilidades
                pres_true.extend(yp.view(-1).cpu().numpy().tolist())
                point_true.extend(yg.view(-1).cpu().numpy().tolist())
                point_p = torch.sigmoid(out_r['pointing'].view(-1)).cpu().numpy().tolist()
                pres_prob.extend(pres_p)
                point_prob.extend(point_p)
                pres_logits.extend(lp)
                point_logits.extend(lg)

            val_loss = float(np.mean(va_losses)) if va_losses else 0.0

            # --- Calibración segura ---
            calib_pres = calibrate_binary_logits(np.array(pres_logits), np.array(pres_true))
            calib_point = calibrate_binary_logits(np.array(point_logits), np.array(point_true))

            # Función auxiliar para probs calibradas (maneja None)
            def calibrated_probs_from_logits(logits_np, calib):
                if calib is None:
                    # fallback: probs sin calibrar
                    return 1.0 / (1.0 + np.exp(-logits_np))
                T = max(1e-3, calib.get('T', 1.0))
                b = calib.get('b', 0.0)
                z = (logits_np - b) / T
                # numeric stable clip
                z = np.clip(z, -30.0, 30.0)
                return 1.0 / (1.0 + np.exp(-z))

            # Obtener probs (calibradas si calib estable, si no probs simples)
            pres_prob_cal = calibrated_probs_from_logits(np.array(pres_logits), calib_pres)
            point_prob_cal = calibrated_probs_from_logits(np.array(point_logits), calib_point)

            # Selección de umbrales: usar por defecto F1 (equilibrio) o prec objetivo 0.98.
            # Cambia 'method' a 'prec' si prefieres prec objetivo.
            method = 'f1'  # 'f1'  o  'prec'
            target_prec = 0.98

            if method == 'f1':
                f1p, thp = best_f1_threshold(np.array(pres_true), pres_prob_cal, name="presence")
                f1g, thg = best_f1_threshold(np.array(point_true), point_prob_cal, name="pointing")
                # Log para depuración
                print(f"[INFO] thresholds by F1 -> pres: {thp:.3f} (F1={f1p:.3f}), point: {thg:.3f} (F1={f1g:.3f})")
            else:
                precp, thp = best_precision_threshold(np.array(pres_true), pres_prob_cal, target_prec=target_prec,
                                                      name="presence")
                f1g, thg = best_f1_threshold(np.array(point_true), point_prob_cal, name="pointing")
                print(
                    f"[INFO] thresholds prec>= {target_prec} -> pres_th: {thp:.3f} (Prec={precp:.3f}); point F1_th: {thg:.3f} (F1={f1g:.3f})")

            sched.step(val_loss)


        precp, thp = best_precision_threshold(np.array(pres_true), np.array(pres_prob_cal), target_prec=0.97)
        f1g, thg = best_f1_threshold(np.array(point_true), np.array(point_prob_cal))

        print(f"[Ep {ep:03d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"Prec[pres]>={precp:.3f}@{thp:.2f}  F1[point]={f1g:.3f}@{thg:.2f}")
        import numpy as np

        def balanced_accuracy(y_true, y_pred):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            classes = np.unique(y_true)
            accs = []
            for c in classes:
                tp = np.sum((y_pred == c) & (y_true == c))
                fn = np.sum((y_pred != c) & (y_true == c))
                denom = tp + fn
                accs.append((tp / denom) if denom > 0 else 0.0)
            return float(np.mean(accs)) if len(accs) else 0.0

        if cols_true_B:
            bacc_col_rowB = balanced_accuracy(cols_true_B, cols_pred_B)
            print(f"[Val] BalancedAcc[col | row=B] = {bacc_col_rowB:.3f}")

        if rows_true_C:
            bacc_row_colC = balanced_accuracy(rows_true_C, rows_pred_C)
            print(f"[Val] BalancedAcc[row | col=C] = {bacc_row_colC:.3f}")
        # Guardado always-on de 'last'
        torch.save({'model': model.state_dict(),
                    'zones': zones, 'input': args.input, 'grid_cols': grid_cols,
                    'thr_presence': float(thp), 'thr_pointing': float(thg),
                    'calib_presence': calib_pres, 'calib_pointing': calib_point,
                    'near_mm': args.near,
                    'far_mm': args.far,
                    },
                   outdir / 'last.pt')

        # Early-stopping en val_loss + best ckpt
        if val_loss < best_val:
            best_val = val_loss
            best_thr_pres, best_thr_point = float(thp), float(thg)
            torch.save({'model': model.state_dict(),
                        'zones': zones, 'input': args.input, 'grid_cols': grid_cols,
                        'thr_presence': best_thr_pres, 'thr_pointing': best_thr_point,
                        'calib_presence': calib_pres, 'calib_pointing': calib_point,
                        'near_mm': args.near,
                        'far_mm': args.far,
                        },
                       best_path)

            print(f"  → Nuevo mejor: {best_path} (val={best_val:.4f}, thP={best_thr_pres:.2f}, thG={best_thr_point:.2f})")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (paciencia {patience})")
                break

    print(f"Mejor val_loss={best_val:.4f}  Umbrales: presence={best_thr_pres:.2f}, pointing={best_thr_point:.2f}")



# ====================== Captura y etiquetado ========================

def rs_pipeline(width=640, height=480, fps=30, enable_filters=True, enable_align=False, align_to='depth', enable_color=False):
    """
    Crea y devuelve el pipeline de RealSense.
    - enable_align: si True y hay color habilitado, alinea COLOR→DEPTH o DEPTH→COLOR según align_to.
    - enable_color: habilita stream RGB (útil si planeas alinear a color). Por defecto apagado para depth-only.
    """
    if rs is None:
        raise RuntimeError('pyrealsense2 no disponible')
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    if enable_color:
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(cfg)

    sensor = profile.get_device().first_depth_sensor()
    try:
        sensor.set_option(rs.option.laser_power, 360.0)
    except Exception:
        pass

    dec = rs.decimation_filter(2) if enable_filters else None
    spat = rs.spatial_filter() if enable_filters else None
    temp = rs.temporal_filter() if enable_filters else None
    hole = rs.hole_filling_filter() if enable_filters else None

    align = None
    if enable_align:
        try:
            if align_to == 'depth':
                align = rs.align(rs.stream.depth)
            elif align_to == 'color':
                align = rs.align(rs.stream.color)
            else:
                align = None
        except Exception:
            align = None

    return pipeline, align, dec, spat, temp, hole



def capture(args):
    out = Path(args.out)
    ensure_dir(out)
    img_dir = out / 'images'
    ensure_dir(img_dir)
    csv_path = out / 'labels.csv'

    if args.label_live:
        with open('config.yaml','r') as f:
            cfg_all = yaml.safe_load(f)
        zones = cfg_all.get('zones', [])
        grid_cols = int(cfg_all.get('grid', {}).get('cols', 3))
    else:
        zones = []
        grid_cols = 3

    # depth-only por defecto, sin alineación; se puede forzar con --align
    pipeline, align, dec, spat, temp, hole = rs_pipeline(enable_align=args.align, align_to='depth', enable_color=args.align)
    print('Capturando... (q=salir, space=capturar, 0..9=zona, n=ninguna)')
    i = 0
    labels = []
    zone_sel = 0
    presence = 1
    pointing = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            # aplicar alineación solo si está disponible
            if align is not None:
                try:
                    frames = align.process(frames)
                except Exception:
                    # Fallback silencioso si la alineación falla
                    pass
            depth = frames.get_depth_frame()
            if not depth:
                continue
            d = np.asanyarray(depth.get_data()).astype(np.uint16)
            if dec: d = np.asanyarray(dec.process(depth).get_data()).astype(np.uint16)
            if spat: d = np.asanyarray(spat.process(rs.frame_from_stream(depth)).get_data()) if False else d
            # Nota: filtros sobre rs.frame pueden variar, mantenemos simple

            show = (normalize_depth(d) * 255).astype(np.uint8)
            show_c = cv2.applyColorMap(show, cv2.COLORMAP_TURBO)

            # Overlay de zona seleccionada + grid
            sel_row, sel_col = 0, 0
            if zone_sel != 0:
                sel_col = ((zone_sel-1) % grid_cols) + 1
                sel_row = ((zone_sel-1) // grid_cols) + 1
            txt = f"Z:{zone_sel} (R{sel_row} C{sel_col})  Pres:{presence}  Apuntanding:{pointing}"
            if zones:
                zname = zones[zone_sel-1] if zone_sel>0 and zone_sel<=len(zones) else 'ninguna'
                txt += f"  ({zname})"
            cv2.putText(show_c, txt, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)
            cv2.imshow('D455 depth (label-live)' if args.label_live else 'D455 depth', show_c)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                break
            if k == ord(' '):
                fn = img_dir / f"img_{i:06d}.png"
                cv2.imwrite(str(fn), depth_to_png_u16(d))
                if args.label_live:
                    row_val, col_val = 0, 0
                    if zone_sel != 0:
                        col_val = ((zone_sel-1) % grid_cols) + 1
                        row_val = ((zone_sel-1) // grid_cols) + 1
                    labels.append([str(fn).replace('\\','/'), presence, pointing, row_val, col_val, zone_sel])
                    print(f"Guardado {fn.name} con labels p={presence}, g={pointing}, row={row_val}, col={col_val}, t={zone_sel}")
                else:
                    print(f"Guardado {fn.name}")
                i += 1
            if not args.label_live:
                continue
            # Controles de etiquetado
            if k in [ord(str(x)) for x in range(10)]:
                zone_sel = int(chr(k))
                pointing = 1 if zone_sel>0 else 0
            if k == ord('n'):
                zone_sel = 0; pointing = 0
            if k == ord('p'):
                presence = 1 - presence
            if k == ord('g'):
                pointing = 1 - pointing
    finally:
        if args.label_live and labels:
            df = pd.DataFrame(labels, columns=['path','presence','pointing','row','column','target'])
            df.to_csv(csv_path, index=False)
            print(f"Etiquetas guardadas en {csv_path}")
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass


# ====================== Inferencia en vivo + IoT ====================
@dataclass
class LiveConfig:
    ema: float = 0.2
    thresh_presence: float = 0.6
    thresh_pointing: float = 0.6

class EMASmooth:
    def __init__(self, alpha):
        self.alpha = alpha
        self.state = None
    def __call__(self, x):
        if self.state is None:
            self.state = x
        else:
            self.state = self.alpha*self.state + (1-self.alpha)*x
        return self.state


def mqtt_publish(client, topic, payload):
    if client is None:
        return
    try:
        client.publish(topic, payload)
    except Exception as e:
        print('MQTT publish error:', e)

# -------------------- publisher local (detector -> toggle_daemon) --------------------
import uuid
import requests

RECEIVER_URL = "http://127.0.0.1:5000/event"
PUBLISH_TIMEOUT = 2.0
PUBLISH_RETRIES = 3
PUBLISH_BACKOFF = 0.5  # seconds base

def publish_zone_hold(zone_name, p_presence, p_pointing, hold_seconds=3.0):
    """POST a local receptor /event. Retries on failure. Retorna True/False."""
    msg = {
        "msg_id": str(uuid.uuid4()),
        "source": "detector",
        "zone": zone_name,
        "event": "zone_hold",
        "hold_seconds": float(hold_seconds),
        "p_presence": float(p_presence),
        "p_pointing": float(p_pointing),
        "ts": time.time()
    }
    backoff = PUBLISH_BACKOFF
    for attempt in range(1, PUBLISH_RETRIES + 1):
        try:
            r = requests.post(RECEIVER_URL, json=msg, timeout=PUBLISH_TIMEOUT)
            if 200 <= r.status_code < 300:
                return True
            else:
                print(f"[publisher] receiver returned {r.status_code}: {r.text}")
        except Exception as e:
            print(f"[publisher] post attempt {attempt} exception: {e}")
        time.sleep(backoff)
        backoff *= 2
    print("[publisher] failed to notify receiver after retries")
    return False
# -------------------------------------------------------------------------------------

def live(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.weights, map_location=device)

    # Config
    with open('config.yaml','r') as f:
        cfg_all = yaml.safe_load(f)
    mqtt_cfg = cfg_all.get('mqtt', {})
    zones_cfg = cfg_all.get('zones', [])
    cfg_grid_cols = int(cfg_all.get('grid', {}).get('cols', 3))

    # Zonas/grid desde el checkpoint (fallback a config si no existen)
    zones = ckpt.get('zones', zones_cfg)
    grid_cols = int(ckpt.get('grid_cols', cfg_grid_cols))
    input_sz = int(ckpt.get('input', cfg_all.get('input', 96)))

    cap = cfg_all.get('capture', {})
    near_mm = int(ckpt.get('near_mm', cap.get('near_mm', 300)))
    far_mm = int(ckpt.get('far_mm', cap.get('far_mm', 4000)))
    # Modelo
    model = DepthPointingNet(num_targets=len(zones), grid_cols=grid_cols, use_coord=args.coordconv).to(device)
    state = ckpt.get('model', ckpt)
    calib_pres = None if getattr(args, 'no_calib', False) else ckpt.get('calib_presence', None)
    calib_point = None if getattr(args, 'no_calib', False) else ckpt.get('calib_pointing', None)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        load_state_dict_adapt(model, state, verbose=True)
    model.eval()

    # MQTT (omitido si --no-mqtt)
    client = None
    if (not getattr(args, 'no_mqtt', False)) and mqtt is not None and mqtt_cfg.get('host'):
        client = mqtt.Client()
        try:
            if mqtt_cfg.get('username'):
                client.username_pw_set(mqtt_cfg['username'], mqtt_cfg.get('password',''))
            client.connect(mqtt_cfg['host'], int(mqtt_cfg.get('port',1883)), 60)
            client.loop_start()
        except Exception as e:
            print('No se pudo conectar a MQTT:', e)
            client = None

    # RealSense
    pipeline, align, dec, spat, temp, hole = rs_pipeline()

    # Suavizado y umbrales (desde args)
    ema_p = EMASmooth(alpha=args.ema)
    ema_g = EMASmooth(alpha=args.ema)
    TH_PRES = float(args.th_pres)
    TH_POINT = float(args.th_point)
    if getattr(args, 'use_ckpt_thr', False):
        TH_PRES = float(ckpt.get('thr_presence', TH_PRES))
        TH_POINT = float(ckpt.get('thr_pointing', TH_POINT))

    stable_needed = max(1, int(getattr(args, 'stable_frames', 2)))

    # Estado
    use_ema = True
    stable_cnt = 0
    last_state = False
    last_zone_stable = 0
    last_sent = 0.0
    # Parámetros tiempo-based (en segundos)
    HOLD_SECONDS = 1.5               # tiempo requerido de apuntado continuo para emitir evento
    OFF_GRACE = 0.7                  # tolerancia antes de resetear hold cuando se deja de apuntar
    MIN_SEND_INTERVAL = 3.0          # mínimo entre envíos al receptor para evitar spam

    # Estructuras por zona (clave: nombre de zona string)
    hold_start_ts = {}               # zone_name -> timestamp when hold started (or None)
    hold_sent_flag = {}              # zone_name -> bool (already sent for this hold)
    last_hold_sent_ts = {}           # zone_name -> last time we sent to receiver

    # Inicializar para cada zona conocida
    for z in zones:
        hold_start_ts[z] = None
        hold_sent_flag[z] = False
        last_hold_sent_ts[z] = 0.0

    def crop_roi(d16: np.ndarray) -> np.ndarray:
        """Recorta al blob principal de profundidad (quita fondo)"""
        m = (d16 > 0).astype(np.uint8)
        ys, xs = np.where(m)
        if xs.size > 0 and ys.size > 0:
            pad = 10
            x0 = max(0, int(xs.min()) - pad); x1 = min(d16.shape[1], int(xs.max()) + pad)
            y0 = max(0, int(ys.min()) - pad); y1 = min(d16.shape[0], int(ys.max()) + pad)
            d16 = d16[y0:y1, x0:x1]
        return d16

    try:
        last_col, last_row = 0, 0
        last_pcol = None
        last_prow = None

        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                try:
                    frames = align.process(frames)
                except Exception:
                    pass
            depth = frames.get_depth_frame()
            if dec is not None: depth = dec.process(depth)
            if spat is not None: depth = spat.process(depth)
            if temp is not None: depth = temp.process(depth)
            if hole is not None: depth = hole.process(depth)

            if not depth:
                continue
            d = np.asanyarray(depth.get_data()).astype(np.uint16)

            # --- Inferencia (con ROI) ---
            # --- Paso 1: presencia sobre frame completo (sin ROI) ---
            x_full = normalize_depth(d,   near_mm, far_mm)
            x_full = cv2.resize(x_full, (input_sz, input_sz), interpolation=cv2.INTER_NEAREST)
            xt_full = torch.from_numpy(x_full).float().unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                out_full = model(xt_full)
                logit_pres = float(out_full['presence'].item())
                logit_pres_z = apply_calib_scalar(logit_pres, calib_pres)
                p_pres = float(torch.sigmoid(torch.tensor(logit_pres_z)))
            p_pres_s = float(ema_p(p_pres)) if use_ema else p_pres

            # Si no llega al umbral de presencia → fuerza estado OFF y continúa
            if p_pres_s <= TH_PRES:
                p_point_s = 0.0
                zone_pred = 0
                row_pred = 0
                col_pred = 0

                vis = (normalize_depth(d, near_mm, far_mm) * 255).astype(np.uint8)
                vis_c = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
                cv2.putText(
                    vis_c, f"pres:{p_pres_s:.3f}  point:{p_point_s:.3f}  (NO ROI)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
                )

                # estabilización: OFF
                on_now = False
                if on_now == last_state:
                    stable_cnt += 1
                else:
                    stable_cnt = 1
                if stable_cnt >= stable_needed and last_state:
                    ts = time.strftime('%H:%M:%S')
                    prev_name = zones[last_zone_stable - 1] if (0 < last_zone_stable <= len(zones)) else 'ninguna'
                    if getattr(args, 'no_mqtt', False):
                        print(f"[{ts}] OFF → zone={prev_name} pres={p_pres_s:.2f} point={p_point_s:.2f}")
                    else:
                        if client is not None and last_zone_stable > 0:
                            topic = mqtt_cfg.get('topic_prefix', 'home/pointing') + f"/{prev_name}"
                            mqtt_publish(client, topic, 'OFF')
                    last_zone_stable = 0
                    last_state = False

                cv2.imshow('LIVE depth', vis_c)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                continue

            # --- Paso 2: ROI + pointing/target solo si hay presencia ---
            d_in = d
            x = normalize_depth(d_in, near_mm, far_mm)
            x = cv2.resize(x, (input_sz, input_sz), interpolation=cv2.INTER_NEAREST)
            xt = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)

            ### PATCH C1: decisión de zona por col+row + confianza

            with torch.no_grad():
                out = model(xt)

                # Pointing calibrado
                logit_point = float(out['pointing'].item())
                logit_point_z = apply_calib_scalar(logit_point, calib_point)
                p_point = float(torch.sigmoid(torch.tensor(logit_point_z)))

                # Distribuciones de columna y fila
                p_col = torch.softmax(out['col'][0], dim=-1) if 'col' in out else None
                p_row = torch.softmax(out['row'][0], dim=-1) if 'row' in out else None
                # --- Ensamble con target agrupado (col/row) ---
                Kc = int(grid_cols)
                pt = torch.softmax(out['target'][0], dim=-1)  # [K+1], 0=none

                # Agrupa probas de target por columna y por fila (ignorando clase 0)
                C = Kc
                R = (len(pt) - 1 + Kc - 1) // Kc  # ceil(K/Kc)
                p_col_t = torch.zeros(C + 1, device=pt.device)  # 0..C
                p_row_t = torch.zeros(R + 1, device=pt.device)  # 0..R
                for k in range(1, len(pt)):
                    r = ((k - 1) // Kc) + 1
                    c = ((k - 1) % Kc) + 1
                    p_col_t[c] += pt[k]
                    p_row_t[r] += pt[k]
                p_col_t = p_col_t / (p_col_t.sum() + 1e-6)
                p_row_t = p_row_t / (p_row_t.sum() + 1e-6)

                # Mezcla (50/50 de inicio; ajusta 0.6/0.4 si te rinde mejor)
                p_col_ens = 0.5 * p_col + 0.5 * p_col_t
                p_row_ens = 0.5 * p_row + 0.5 * p_row_t

                # Predicciones argmax (clase 0 = none)
                col_pred = int(torch.argmax(p_col_ens).item())
                row_pred = int(torch.argmax(p_row_ens).item())

            # --- Histeresis por cabeza ---
            MARGIN_COL = 0.12
            MARGIN_ROW = 0.10
            BETA = 0.6  # EMA de las distribuciones

            # Inicializa memorias si es la primera vez
            if last_pcol is None:
                last_pcol = p_col_ens.clone()
                last_prow = p_row_ens.clone()

            # Exige margen para cambiar de columna
            if col_pred != last_col and last_col > 0:
                if p_col_ens[col_pred] < p_col_ens[last_col] + MARGIN_COL:
                    col_pred = last_col

            # Exige margen para cambiar de fila
            if row_pred != last_row and last_row > 0:
                if p_row_ens[row_pred] < p_row_ens[last_row] + MARGIN_ROW:
                    row_pred = last_row

            # Actualiza memorias (EMA vectorial)
            last_pcol = BETA * last_pcol + (1 - BETA) * p_col_ens
            last_prow = BETA * last_prow + (1 - BETA) * p_row_ens
            last_col, last_row = col_pred, row_pred

            # Suavizado de pointing
            p_point_eff = p_point
            p_point_s = float(ema_g(p_point_eff)) if use_ema else p_point_eff

            # Gating suave (usa umbrales, pero no bloquea cálculo)
            if p_pres_s <= TH_PRES:
                # no presencia
                zone_pred = 0
                row_pred = 0
                col_pred = 0
                conf_zone = 0.0

            elif p_point_s < TH_POINT:
                # presencia sin pointing
                zone_pred = 0
                row_pred = 0
                col_pred = 0
                conf_zone = p_point_s

            else:
                # Zona por combinación col+row
                # Confianza final = P(pointing) * max(p_col) * max(p_row)
                mc = float(torch.max(p_col_ens).item())
                mr = float(torch.max(p_row_ens).item())
                conf_zone = p_point_s * mc * mr

                if col_pred > 0 and row_pred > 0:
                    Kc = int(grid_cols)
                    zone_pred = (row_pred - 1) * Kc + col_pred
                else:
                    zone_pred = 0

            # Manejo de incertidumbre: si conf_zone < 0.30, marca como “incierta”
            CONF_MIN = 0.50
            uncertain = (conf_zone < CONF_MIN and zone_pred > 0)
            zone_pred_action = zone_pred if not uncertain else 0


            # --- Visualización ---
            vis = (normalize_depth(d, near_mm, far_mm) * 255).astype(np.uint8)
            vis_c = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)

            display_zone = zones[zone_pred - 1] if (zone_pred > 0 and zone_pred <= len(zones)) else 'ninguna'
            unc_tag = " ?unc" if uncertain else ""
            cv2.putText(
                vis_c,
                f"pres:{p_pres_s:.3f} ({logit_pres:+.2f})  point:{p_point_s:.3f} ({logit_point:+.2f})  confZ:{conf_zone:.2f}{unc_tag}  ema:{int(use_ema)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
            cv2.putText(
                vis_c,
                f"tgt:{display_zone}  [rc:{row_pred},{col_pred}]  TH(P:{TH_PRES:.2f},G:{TH_POINT:.2f})",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
            cv2.imshow('LIVE depth', vis_c)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('e'):  # Toggle EMA
                use_ema = not use_ema

            # --- Time-based hold logic (3s hold) ---
            on_frame_valid = (p_pres_s > TH_PRES and p_point_s > TH_POINT and zone_pred_action > 0)

            display_zone = zones[zone_pred_action - 1] if (
                        zone_pred_action > 0 and zone_pred_action <= len(zones)) else 'ninguna'
            now_ts = time.time()

            INTEREST_ZONES = {"A1", "A3"}  # ejemplo; ajusta a lo que te interese realmente

            if on_frame_valid and display_zone in INTEREST_ZONES:
                if hold_start_ts.get(display_zone) is None:
                    hold_start_ts[display_zone] = now_ts
                    hold_sent_flag[display_zone] = False
                elapsed = now_ts - hold_start_ts[display_zone]
                if elapsed >= HOLD_SECONDS and not hold_sent_flag[display_zone]:
                    if now_ts - last_hold_sent_ts.get(display_zone, 0.0) >= MIN_SEND_INTERVAL:
                        import threading
                        def _bg_publish(zone_name, p_pres, p_point, hold_s):
                            publish_zone_hold(zone_name, p_pres, p_point, hold_seconds=hold_s)

                        threading.Thread(
                            target=_bg_publish,
                            args=(display_zone, p_pres_s, p_point_s, HOLD_SECONDS),
                            daemon=True
                        ).start()
                        last_hold_sent_ts[display_zone] = now_ts
                        hold_sent_flag[display_zone] = True
                        if getattr(args, 'no_mqtt', False):
                            print(
                                f"[{time.strftime('%H:%M:%S')}] HOLD -> zone={display_zone} pres={p_pres_s:.2f} point={p_point_s:.2f}")
                        else:
                            if client is not None:
                                topic = mqtt_cfg.get('topic_prefix', 'home/pointing') + f"/{display_zone}"
                                mqtt_publish(client, topic, 'ON')
            else:
                # limpieza con gracia
                for z in INTEREST_ZONES:
                    if hold_start_ts.get(z) is not None and display_zone != z:
                        last_valid_ts = hold_start_ts[z]
                        if last_valid_ts is not None and (now_ts - last_valid_ts) > OFF_GRACE:
                            hold_start_ts[z] = None
                            hold_sent_flag[z] = False



    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        if client is not None:
            client.loop_stop()
            try:
                client.disconnect()
            except Exception:
                pass

def configure(args):
    zones = [z.strip() for z in args.zones.split(',')] if args.zones else []
    cfg = {
        'zones': zones,
        'mqtt': {
            'host': args.mqtt_host,
            'port': args.mqtt_port,
            'username': args.mqtt_user,
            'password': args.mqtt_pass,
            'topic_prefix': args.mqtt_prefix,
        },
        'capture': {
            'near_mm': args.near,
            'far_mm': args.far
        },
        'grid': {
            'cols': args.grid_cols
        }
    }
    with open('config.yaml','w') as f:
        yaml.safe_dump(cfg, f)
    print('config.yaml escrito con:', cfg)


# =============================== CLI ================================

def parse_args():
    p = argparse.ArgumentParser(description='Depth Pointing System (D455, depth-only)')
    sub = p.add_subparsers(dest='cmd', required=True)

    # configure
    pc = sub.add_parser('configure')
    pc.add_argument('--zones', type=str, default='')
    pc.add_argument('--mqtt-host', type=str, default='')
    pc.add_argument('--mqtt-port', type=int, default=1883)
    pc.add_argument('--mqtt-user', type=str, default='')
    pc.add_argument('--mqtt-pass', type=str, default='')
    pc.add_argument('--mqtt-prefix', type=str, default='home/pointing')
    pc.add_argument('--near', type=int, default=300)
    pc.add_argument('--far', type=int, default=4000)
    pc.add_argument('--grid-cols', type=int, default=3, help='Número de columnas del grid (para row/column).')
    pc.set_defaults(func=configure)

    # capture
    pcap = sub.add_parser('capture')
    pcap.add_argument('--out', type=str, required=True)
    pcap.add_argument('--label-live', action='store_true')
    pcap.add_argument('--align', action='store_true', help='Habilitar alineación (abre también el stream de color).')
    pcap.set_defaults(func=capture)

    # train
    ptr = sub.add_parser('train')
    ptr.add_argument('--data-csv', type=str, nargs='+', required=True,
                     help='Ruta(s) a uno o varios labels.csv (acepta 1 o más).')
    ptr.add_argument('--epochs', type=int, default=40)  # ↑
    ptr.add_argument('--batch', type=int, default=32)  # si cabe
    ptr.add_argument('--lr', type=float, default=3e-4)  # ↑
    ptr.add_argument('--input', type=int, default=256)  # ↑
    ptr.add_argument('--near', type=int, default=300)
    ptr.add_argument('--far', type=int, default=4000)
    ptr.add_argument('--patience', type=int, default=6)  # early stopping
    # toggles aumentos
    ptr.add_argument('--no-flip', action='store_true')
    ptr.add_argument('--no-jitter', action='store_true')
    ptr.add_argument('--no-roi', action='store_true')
    ptr.add_argument('--out', type=str, required=True)
    ptr.set_defaults(func=train_model)
    ptr.add_argument('--coordconv', action='store_true', help='Añade canales XY normalizados a la entrada')

    ### PATCH A1: args de pérdidas/plan A (train)
    ptr.add_argument('--lambda-pres', type=float, default=1.0, help='Peso λ1 para BCE presencia')
    ptr.add_argument('--lambda-point', type=float, default=1.0, help='Peso λ2 para BCE pointing')
    ptr.add_argument('--lambda-col', type=float, default=1.0, help='Peso λ3 para CE columna')
    ptr.add_argument('--lambda-row', type=float, default=1.0, help='Peso λ4 para CE/Focal fila')

    # Currículum: cuántas épocas con fila desactivada (λ_row=0)
    ptr.add_argument('--row-warmup-epochs', type=int, default=6,
                     help='Épocas iniciales con λ_row=0 (solo aprende columna)')

    # Focal y pesos dirigidos (hard cases)
    ptr.add_argument('--row-focal-gamma', type=float, default=2.0,
                     help='γ de Focal para head fila')
    ptr.add_argument('--col-weight-rowB', type=float, default=1.5,
                     help='Peso extra en pérdida de columna para muestras con row==2 (fila central B)')

    # live
    plv = sub.add_parser('live')
    plv.add_argument('--weights', type=str, required=True)
    plv.add_argument('--no-mqtt', action='store_true', help='No publicar a MQTT; imprimir ON/OFF en consola')
    plv.add_argument('--align', action='store_true', help='Habilitar alineación (requiere color).')
    plv.add_argument('--th-pres', type=float, default=0.20)
    plv.add_argument('--th-point', type=float, default=0.35)
    plv.add_argument('--ema', type=float, default=0.4)
    plv.add_argument('--stable-frames', type=int, default=4) #subir a 4-5
    plv.add_argument('--use-ckpt-thr', action='store_true', help='Usar umbrales guardados en el .pt')
    plv.add_argument('--no-calib', action='store_true', help='No aplicar calibración (T,b) del checkpoint')
    plv.add_argument('--coordconv', action='store_true', help='Añade canales XY normalizados a la entrada')

    plv.set_defaults(func=live)

    return p.parse_args()


def main():
    args = parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
    #hacer cambios nuevos
