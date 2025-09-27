#!/usr/bin/env python3
# Ejemplos de uso:
#   Entrenar:
#     python presence.py train --data-csv data/s2/labels_presence.csv --epochs 30 --out runs-pre/presence_exp1
#   Live:
#     python presence.py live --weights runs-pre/presence_exp1/best.pt --use-ckpt-thr
#     python presence.py live --weights runs-pre/presence_exp2/best.pt --th-pres 0.6 --no-calib --no-roi

import argparse
import os
from pathlib import Path
import time
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import yaml

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


# ============================== Utilidades básicas ==============================

def set_seed(seed: int = 1):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_config(path='config.yaml'):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

def normalize_depth(d16: np.ndarray, near_mm: int = 300, far_mm: int = 4000) -> np.ndarray:
    """Convierte u16 (mm) a float32 [0,1] recortando a [near, far]."""
    d = d16.astype(np.float32)
    d[d <= 0] = np.nan
    d = np.clip(d, float(near_mm), float(far_mm))
    d = (d - float(near_mm)) / max(1.0, float(far_mm - near_mm))
    d = np.clip(d, 0.0, 1.0)
    d[np.isnan(d)] = 0.0
    return d

def resize_keep_ratio(img: np.ndarray, out_hw: int) -> np.ndarray:
    """Redimensiona a cuadrado out_hw x out_hw manteniendo proporción (padding negro si hace falta)."""
    h, w = img.shape[:2]
    if h == w:
        return cv2.resize(img, (out_hw, out_hw), interpolation=cv2.INTER_AREA)
    if h > w:
        new_h = out_hw
        new_w = int(round(w * out_hw / h))
    else:
        new_w = out_hw
        new_h = int(round(h * out_hw / w))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_hw, out_hw), dtype=resized.dtype)
    y0 = (out_hw - new_h) // 2
    x0 = (out_hw - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

def best_threshold_by_f1(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.uint8)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr), float(best_f1)

def best_threshold_by_precision(y_true, y_prob, min_precision=0.98):
    thresholds = np.linspace(0.05, 0.95, 19)
    ok = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.uint8)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        ok.append((t, p, r, f1))
    ok = [row for row in ok if row[1] >= min_precision]
    if not ok:
        return best_threshold_by_f1(y_true, y_prob)[0]
    ok.sort(key=lambda x: x[0], reverse=True)
    return float(ok[0][0])

def calibrate_scalar_logits(y_true, logits):
    """
    Calibración escalar simple: ajusta T (temperatura) y b (bias) minimizando BCE.
    Devuelve (T, b).  T>0.
    """
    y = torch.tensor(y_true, dtype=torch.float32)
    z = torch.tensor(logits, dtype=torch.float32)

    T = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.LBFGS([T, b], lr=0.5, max_iter=100, line_search_fn='strong_wolfe')
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        zz = z / (T.abs() + 1e-6) + b
        loss = bce(zz, y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        pass

    Tv = float(T.detach().abs().item())
    bv = float(b.detach().item())
    if not np.isfinite(Tv) or Tv < 1e-3:
        Tv = 1.0
    if not np.isfinite(bv):
        bv = 0.0
    return Tv, bv

def apply_calib(logits, T=1.0, b=0.0):
    return logits / max(T, 1e-6) + b

def device_auto():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================== Dataset ==============================

class PresenceDataset(Dataset):
    """
    Lee CSV(s) con columnas: path,presence,pointing,row,column,target
    Carga imágenes PNG u16, normaliza y redimensiona a input_size (cuadrado).
    Retorna tensores (1,H,W) y label (0/1).
    """
    def __init__(self, csv_paths, input_size=96, near_mm=300, far_mm=4000, root_dir=None):
        if isinstance(csv_paths, (list, tuple)):
            dfs = [pd.read_csv(p) for p in csv_paths]
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(csv_paths)

        if root_dir is not None:
            df['path'] = df['path'].apply(lambda p: str(Path(root_dir) / p) if not os.path.isabs(p) else p)

        needed = ['path', 'presence']
        for c in needed:
            if c not in df.columns:
                raise ValueError(f'Falta columna {c} en CSV.')

        df = df[df['path'].apply(lambda p: os.path.exists(p))].reset_index(drop=True)

        self.df = df
        self.input_size = int(input_size)
        self.near = int(near_mm)
        self.far = int(far_mm)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        y = int(row['presence'])

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f'No se pudo leer {path}')
        if img.dtype != np.uint16:
            img = img.astype(np.uint16)

        x = normalize_depth(img, self.near, self.far)
        x = resize_keep_ratio(x, self.input_size)
        x = np.expand_dims(x, axis=0)  # (1,H,W)
        x = torch.from_numpy(x).float()

        return x, torch.tensor(y, dtype=torch.float32)


# ============================== Modelo ==============================

class SmallPresenceNet(nn.Module):
    """
    CNN eficiente para binario (presencia). Entrada (B,1,inp,inp), salida logit (B,1).
    """
    def __init__(self, in_ch=1, width=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(width, width*2, 3, padding=1), nn.BatchNorm2d(width*2), nn.ReLU(inplace=True),
            nn.Conv2d(width*2, width*2, 3, padding=1), nn.BatchNorm2d(width*2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(width*2, width*4, 3, padding=1), nn.BatchNorm2d(width*4), nn.ReLU(inplace=True),
            nn.Conv2d(width*4, width*4, 3, padding=1), nn.BatchNorm2d(width*4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Linear(width*4, 1)

    def forward(self, x):
        feat = self.backbone(x)        # (B,C,1,1)
        feat = feat.flatten(1)         # (B,C)
        logit = self.head(feat)        # (B,1)
        return logit.squeeze(1)        # (B,)


# ============================== Entrenamiento ==============================

def evaluate(model, loader, device, near, far, use_auc=True):
    """
    Calcula métricas en validación. ACC se evalúa con el umbral que maximiza F1 (thr_f1).
    """
    model.eval()
    ys, ps, zs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            z = model(xb)                  # logits
            p = torch.sigmoid(z)
            ys.append(yb.cpu().numpy())
            ps.append(p.cpu().numpy())
            zs.append(z.cpu().numpy())
    y_true = np.concatenate(ys).astype(np.float32)
    y_prob = np.concatenate(ps).astype(np.float32)
    z_all  = np.concatenate(zs).astype(np.float32)

    auc = roc_auc_score(y_true, y_prob) if use_auc and (len(np.unique(y_true)) > 1) else np.nan
    thr, _ = best_threshold_by_f1(y_true, y_prob)

    y_pred = (y_prob >= thr).astype(np.uint8)
    p, r, f1x, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = float((y_pred == y_true.astype(np.uint8)).mean())

    metrics = {
        'AUC': float(auc),
        'thr_f1': float(thr),
        'F1': float(f1x),
        'P': float(p),
        'R': float(r),
        'ACC': float(acc),
    }
    return metrics, y_true, y_prob, z_all

def train(args):
    set_seed(args.seed)
    cfg = load_config('config.yaml')
    near = int(cfg.get('capture', {}).get('near_mm', args.near))
    far  = int(cfg.get('capture',  {}).get('far_mm',  args.far))

    # Dataset
    ds = PresenceDataset(args.data_csv, input_size=args.input, near_mm=near, far_mm=far, root_dir=None)

    # Split
    n = len(ds)
    val_n = max(1, int(round(n * args.val_ratio)))
    train_n = n - val_n
    ds_train, ds_val = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))

    # Sampler opcional por clase (estable en Windows con num_workers=0)
    train_labels = [int(ds_train[i][1].item()) for i in range(len(ds_train))]
    pos_t = sum(train_labels)
    neg_t = len(train_labels) - pos_t
    use_sampler = (pos_t > 0 and neg_t > 0)

    if use_sampler:
        class_counts = np.array([neg_t, pos_t], dtype=np.float32)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = np.array([class_weights[y] for y in train_labels], dtype=np.float32)
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        loader_train = DataLoader(ds_train, batch_size=args.batch, sampler=sampler,
                                  shuffle=False, num_workers=0, pin_memory=False)
    else:
        loader_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                                  num_workers=0, pin_memory=False)

    loader_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=False)

    # Modelo
    device = device_auto()
    model = SmallPresenceNet(in_ch=1, width=args.width).to(device)

    # Pérdida (pos_weight desde train)
    pos_weight = torch.tensor([(neg_t / max(pos_t, 1))], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if args.opt == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    outdir = Path(args.out)
    ensure_dir(outdir)
    best_val = 1e9
    best_path = None
    patience = args.patience
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader_train:
            xb = xb.to(device)
            yb = yb.to(device)
            z = model(xb)
            loss = bce(z, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)

        train_loss = running / max(1, len(ds_train))
        (metrics_val, y_true, y_prob, z_all) = evaluate(model, loader_val, device, near, far)
        val_loss_proxy = 1.0 - metrics_val['F1']  # proxy simple para early stopping
        scheduler.step(val_loss_proxy)

        print(f"[{epoch:03d}] "
              f"train_loss={train_loss:.4f} | "
              f"val F1={metrics_val['F1']:.4f} "
              f"P={metrics_val['P']:.4f} "
              f"R={metrics_val['R']:.4f} "
              f"ACC={metrics_val['ACC']:.4f} "
              f"AUC={metrics_val['AUC']:.4f} "
              f"thr≈{metrics_val['thr_f1']:.2f}")

        # Guardado "last"
        ckpt_last = {
            'state_dict': model.state_dict(),
            'meta': {
                'model_type': 'presence',
                'input': args.input,
                'near_mm': near,
                'far_mm': far,
                'zones': cfg.get('zones', []),
                'grid_cols': int(cfg.get('grid', {}).get('cols', 3)),
                'created_at': datetime.utcnow().isoformat() + 'Z',
            },
            'calibration': None,
            'thresholds': None,
            'norm': {'method': 'clip_minmax', 'near_mm': near, 'far_mm': far},
            'roi': None,
        }
        torch.save(ckpt_last, outdir / 'last.pt')

        # Early stopping & best
        if val_loss_proxy < best_val:
            best_val = val_loss_proxy
            bad_epochs = 0

            # Calibración y umbral con val
            try:
                T, b = calibrate_scalar_logits(y_true, np.asarray(z_all).reshape(-1))
            except Exception:
                T, b = 1.0, 0.0
            thr_f1, _ = best_threshold_by_f1(y_true, y_prob)
            if args.precision_target is not None:
                thr_prec = best_threshold_by_precision(y_true, y_prob, args.precision_target)
                chosen_thr = float(thr_prec)
            else:
                chosen_thr = float(thr_f1)

            ckpt_best = ckpt_last.copy()
            ckpt_best['calibration'] = {'T': float(T), 'b': float(b)}
            ckpt_best['thresholds']  = {'thr': float(chosen_thr)}

            # Guarda best.pt y además best{epoch}.pt
            best_path = outdir / 'best.pt'
            torch.save(ckpt_best, best_path)
            epoch_best_path = outdir / f'best{epoch}.pt'
            torch.save(ckpt_best, epoch_best_path)

            print(f"  ↳ best.pt y {epoch_best_path.name} actualizados | "
                  f"thr={chosen_thr:.3f} | calib T={T:.3f} b={b:.3f}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    print("Entrenamiento finalizado.")
    if best_path:
        print(f"Mejor checkpoint: {best_path}")
    else:
        print("No hubo mejora sobre el proxy de validación; usa last.pt.")


# ============================== Live ==============================

def rs_pipeline(width=640, height=480, fps=30):
    if rs is None:
        raise RuntimeError('pyrealsense2 no disponible')
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(cfg)
    sensor = profile.get_device().first_depth_sensor()
    try:
        sensor.set_option(rs.option.laser_power, 360.0)
    except Exception:
        pass
    # Filtros
    dec = rs.decimation_filter(2)
    spat = rs.spatial_filter()
    temp = rs.temporal_filter()
    hole = rs.hole_filling_filter()
    return pipeline, dec, spat, temp, hole

@torch.inference_mode()
def live(args):
    # Cargar ckpt
    ckpt = torch.load(args.weights, map_location='cpu')
    meta = ckpt.get('meta', {})
    near = int(meta.get('near_mm', 300))
    far  = int(meta.get('far_mm', 4000))
    input_size = int(meta.get('input', args.input))

    # Modelo
    device = device_auto()
    model = SmallPresenceNet(in_ch=1, width=args.width).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    # Umbral y calibración
    thr = args.th_pres
    if args.use_ckpt_thr and ckpt.get('thresholds'):
        thr = float(ckpt['thresholds'].get('thr', thr))
    T = 1.0
    b = 0.0
    if not args.no_calib and ckpt.get('calibration'):
        T = float(ckpt['calibration'].get('T', 1.0))
        b = float(ckpt['calibration'].get('b', 0.0))

    # Cámara
    pipeline, dec, spat, temp, hole = rs_pipeline()
    try:
        cv2.namedWindow('presence', cv2.WINDOW_NORMAL)
        while True:
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if depth is None:
                continue
            # filtros
            depth = dec.process(depth)
            depth = spat.process(depth)
            depth = temp.process(depth)
            depth = hole.process(depth)

            d = np.asanyarray(depth.get_data()).astype(np.uint16)
            x = normalize_depth(d, near, far)
            x = resize_keep_ratio(x, input_size)
            vis = (x * 255.0).astype(np.uint8)

            xt = torch.from_numpy(np.expand_dims(x, 0)[None, ...]).float().to(device)  # (1,1,H,W)
            z = model(xt).cpu().numpy().reshape(-1)[0]
            if not args.no_calib:
                z = apply_calib(z, T, b)
            p = 1.0 / (1.0 + np.exp(-z))
            flag = int(p >= thr)

            # HUD
            hud = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            cv2.putText(hud, f"P={p:.3f} thr={thr:.2f} flag={flag}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('presence', hud)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass


# ============================== CLI ==============================

def build_parser():
    p = argparse.ArgumentParser(description='Modelo de Presencia (train + live)')
    sub = p.add_subparsers(dest='cmd', required=True)

    # Train
    pt = sub.add_parser('train')
    pt.add_argument('--data-csv', type=str, nargs='+', required=True, help='Uno o varios CSVs con columnas estándar.')
    pt.add_argument('--out', type=str, required=True)
    pt.add_argument('--input', type=int, default=96)
    pt.add_argument('--width', type=int, default=32, help='Ancho base de canales en la CNN.')
    pt.add_argument('--batch', type=int, default=32)
    pt.add_argument('--epochs', type=int, default=30)
    pt.add_argument('--val-ratio', type=float, default=0.2)
    pt.add_argument('--seed', type=int, default=1)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--wd', type=float, default=1e-4)
    pt.add_argument('--opt', type=str, choices=['adam','sgd'], default='adam')
    pt.add_argument('--near', type=int, default=300, help='Usado si no está en config.yaml')
    pt.add_argument('--far',  type=int, default=4000, help='Usado si no está en config.yaml')
    pt.add_argument('--patience', type=int, default=7)
    pt.add_argument('--precision-target', type=float, default=None,
                    help='Si se define, selecciona umbral con precisión >= este valor; si no, usa mejor F1.')
    pt.set_defaults(func=train)

    # Live
    pl = sub.add_parser('live')
    pl.add_argument('--weights', type=str, required=True)
    pl.add_argument('--input', type=int, default=96, help='Se usa si el ckpt no trae meta.input.')
    pl.add_argument('--width', type=int, default=32)
    pl.add_argument('--th-pres', type=float, default=0.5)
    pl.add_argument('--use-ckpt-thr', action='store_true')
    pl.add_argument('--no-calib', action='store_true', help='Desactiva calibración (ignora T,b del ckpt).')
    pl.set_defaults(func=live)

    return p

if __name__ == '__main__':
    args = build_parser().parse_args()
    args.func(args)
