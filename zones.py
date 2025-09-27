#!/usr/bin/env python3
# Ejemplos de uso:
#   Entrenar (ROI activado por defecto, recorte centrado):
#     python zones.py train --data-csv data/z1/labels.csv data/z2/labels.csv --input 96 --out runs-z/zones_exp1
#   Entrenar (ROI por profundidad mínima):
#     python zones.py train --data-csv data/z*/labels.csv --roi-mode min --input 96 --out runs-z/zones_exp2
#   Live (usa calibración del ckpt):
#     python zones.py live --weights runs-zon/zones_exp1/best.pt --no-roi

import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score

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
    """Redimensiona a cuadrado out_hw x out_hw manteniendo proporción."""
    h, w = img.shape[:2]
    if h == w:
        return cv2.resize(img, (out_hw, out_hw), interpolation=cv2.INTER_AREA)
    if h > w:
        new_h = out_hw; new_w = int(round(w * out_hw / h))
    else:
        new_w = out_hw; new_h = int(round(h * out_hw / w))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_hw, out_hw), dtype=resized.dtype)
    y0 = (out_hw - new_h) // 2; x0 = (out_hw - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

def device_auto():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================== ROI helpers ==============================

def crop_roi_center(d16: np.ndarray, side_px: int) -> np.ndarray:
    """Recorte cuadrado centrado (en u16)."""
    h, w = d16.shape[:2]
    side = int(min(h, w, side_px))
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - side // 2); x0 = max(0, cx - side // 2)
    y1 = min(h, y0 + side);      x1 = min(w, x0 + side)
    return d16[y0:y1, x0:x1]

def crop_roi_min_depth(d16: np.ndarray, side_px: int) -> np.ndarray:
    """Recorte alrededor del píxel de menor profundidad (>0)."""
    h, w = d16.shape[:2]
    valid = d16.copy(); valid[valid == 0] = 65535
    idx = np.argmin(valid); y, x = divmod(int(idx), w)
    side = int(min(h, w, side_px))
    y0 = max(0, y - side // 2); x0 = max(0, x - side // 2)
    y1 = min(h, y0 + side);     x1 = min(w, x0 + side)
    roi = d16[y0:y1, x0:x1]
    if roi.shape[0] != side or roi.shape[1] != side:
        canvas = np.zeros((side, side), dtype=d16.dtype)
        yy = (side - roi.shape[0]) // 2; xx = (side - roi.shape[1]) // 2
        canvas[yy:yy+roi.shape[0], xx:xx+roi.shape[1]] = roi
        roi = canvas
    return roi


# ============================== Dataset ==============================

class ZonesDataset(Dataset):
    """
    CSV(s) con columnas: path, target (otras columnas se ignoran).
    Etiqueta: target (string).
    ROI opcional (por defecto activado): center o min.
    """
    def __init__(self, csv_paths, input_size=96, near_mm=300, far_mm=4000,
                 root_dir=None, use_roi=True, roi_mode='center', roi_side=None):
        if isinstance(csv_paths, (list, tuple)):
            dfs = [pd.read_csv(p) for p in csv_paths]
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(csv_paths)

        if root_dir is not None:
            df['path'] = df['path'].apply(lambda p: str(Path(root_dir) / p) if not os.path.isabs(p) else p)

        needed = ['path', 'target']
        for c in needed:
            if c not in df.columns:
                raise ValueError(f'Falta columna {c} en CSV.')

        # Mantener solo archivos existentes
        df = df[df['path'].apply(lambda p: os.path.exists(p))].reset_index(drop=True)

        # Normalizar etiqueta a string
        df['target'] = df['target'].astype(str)

        # Clases (orden alfabético estable)
        classes = sorted(df['target'].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        df = df[df['target'].isin(self.class_to_idx.keys())].reset_index(drop=True)

        self.df = df
        self.input_size = int(input_size)
        self.near = int(near_mm)
        self.far = int(far_mm)
        self.use_roi = bool(use_roi)
        self.roi_mode = roi_mode
        self.roi_side = roi_side  # px del frame original, None => 0.6*min(H,W)

    def __len__(self):
        return len(self.df)

    def _crop_roi(self, img_u16: np.ndarray) -> np.ndarray:
        if not self.use_roi:
            return img_u16
        h, w = img_u16.shape[:2]
        side = self.roi_side if self.roi_side is not None else int(0.6 * min(h, w))
        side = max(32, min(min(h, w), side))
        if self.roi_mode == 'min':
            return crop_roi_min_depth(img_u16, side)
        else:
            return crop_roi_center(img_u16, side)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        y_lbl = str(row['target'])
        y = self.class_to_idx[y_lbl]

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f'No se pudo leer {path}')
        if img.dtype != np.uint16:
            img = img.astype(np.uint16)

        roi = self._crop_roi(img)
        x = normalize_depth(roi, self.near, self.far)
        x = resize_keep_ratio(x, self.input_size)
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float()
        return x, torch.tensor(y, dtype=torch.long)


# ============================== Modelo ==============================

class SmallZonesNet(nn.Module):
    """CNN multiclass para zonas. Entrada (B,1,H,W) → logits (B,C)."""
    def __init__(self, num_classes: int, in_ch=1, width=32):
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
        self.head = nn.Linear(width*4, num_classes)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.head(feat)


# ============================== Calibración (temperatura escalar) ==============================

def calibrate_temperature_mc(logits_val: np.ndarray, y_true: np.ndarray):
    """
    Ajusta una temperatura escalar T>0 para softmax: softmax(logits/T).
    Minimiza NLL. Devuelve T (float).
    """
    z = torch.tensor(logits_val, dtype=torch.float32)
    y = torch.tensor(y_true, dtype=torch.long)

    T = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.5, max_iter=100, line_search_fn='strong_wolfe')

    def nll_with_T():
        opt.zero_grad()
        zz = z / (T.abs() + 1e-6)
        logp = F.log_softmax(zz, dim=1)
        loss = F.nll_loss(logp, y, reduction='mean')
        loss.backward()
        return loss

    try:
        opt.step(nll_with_T)
    except Exception:
        pass

    Tv = float(T.detach().abs().item())
    if not np.isfinite(Tv) or Tv < 1e-3:
        Tv = 1.0
    return Tv


# ============================== Métricas ==============================

@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    ys, zs = [], []
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        z = model(xb)                 # (B,C)
        ys.append(yb.cpu().numpy())
        zs.append(z.cpu().numpy())
    y_true = np.concatenate(ys).astype(np.int64)
    logits = np.concatenate(zs).astype(np.float32)
    probs  = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    # ACC top-1
    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == y_true).mean())

    # F1 macro
    f1_macro = float(f1_score(y_true, y_pred, average='macro'))

    # AUC macro (OVR) si hay diversidad por clase
    aucs = []
    try:
        for c in range(num_classes):
            y_bin = (y_true == c).astype(np.int32)
            if len(np.unique(y_bin)) < 2:
                continue
            aucs.append(roc_auc_score(y_bin, probs[:, c]))
        auc_macro = float(np.mean(aucs)) if aucs else float('nan')
    except Exception:
        auc_macro = float('nan')

    return {
        'ACC': acc,
        'F1m': f1_macro,
        'AUCm': auc_macro
    }, y_true, logits


# ============================== Entrenamiento ==============================

def train(args):
    set_seed(args.seed)
    cfg = load_config('config.yaml')
    near = int(cfg.get('capture', {}).get('near_mm', args.near))
    far  = int(cfg.get('capture',  {}).get('far_mm',  args.far))

    # Dataset completo para definir clases
    ds_all = ZonesDataset(
        args.data_csv, input_size=args.input, near_mm=near, far_mm=far,
        root_dir=None, use_roi=not args.no_roi, roi_mode=args.roi_mode, roi_side=args.roi_side
    )
    classes = [ds_all.idx_to_class[i] for i in range(len(ds_all.idx_to_class))]
    num_classes = len(classes)

    # Split
    n = len(ds_all)
    val_n = max(1, int(round(n * args.val_ratio)))
    train_n = n - val_n
    ds_train, ds_val = random_split(ds_all, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))

    # DataLoaders (Windows-friendly)
    loader_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False)
    loader_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)

    # Modelo
    device = device_auto()
    model = SmallZonesNet(num_classes=num_classes, in_ch=1, width=args.width).to(device)

    # Pérdida y optimizador
    ce = nn.CrossEntropyLoss()
    if args.opt == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    outdir = Path(args.out); ensure_dir(outdir)
    best_proxy = 1e9; bad_epochs = 0; best_path = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader_train:
            xb = xb.to(device); yb = yb.to(device)
            z = model(xb)
            loss = ce(z, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.item()) * xb.size(0)

        train_loss = running / max(1, len(ds_train))
        metrics_val, y_true_val, logits_val = evaluate(model, loader_val, device, num_classes)
        # Proxy para early stopping: 1 - F1 macro
        proxy = 1.0 - metrics_val['F1m']
        scheduler.step(proxy)

        print(f"[{epoch:03d}] train_loss={train_loss:.4f} | "
              f"val ACC={metrics_val['ACC']:.4f} F1m={metrics_val['F1m']:.4f} AUCm={metrics_val['AUCm']:.4f}")

        # Guardar last
        ckpt_last = {
            'state_dict': model.state_dict(),
            'meta': {
                'model_type': 'zones',
                'input': args.input,
                'near_mm': near,
                'far_mm': far,
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'classes': classes,
                'roi': {'use_roi': not args.no_roi, 'roi_mode': args.roi_mode, 'roi_side': args.roi_side},
            },
            'calibration': None,  # T
            'norm': {'method': 'clip_minmax', 'near_mm': near, 'far_mm': far},
        }
        torch.save(ckpt_last, outdir / 'last.pt')

        # Early stopping & best
        if proxy < best_proxy:
            best_proxy = proxy
            bad_epochs = 0

            # Calibración por temperatura escalar con val
            try:
                T = calibrate_temperature_mc(logits_val, y_true_val)
            except Exception:
                T = 1.0

            ckpt_best = ckpt_last.copy()
            ckpt_best['calibration'] = {'T': float(T)}

            best_path = outdir / 'best.pt'
            torch.save(ckpt_best, best_path)
            epoch_best_path = outdir / f'best{epoch}.pt'
            torch.save(ckpt_best, epoch_best_path)
            print(f"  -> best.pt y {epoch_best_path.name} actualizados | calib T={T:.3f}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    print("Entrenamiento finalizado.")
    if best_path:
        print(f"Mejor checkpoint: {best_path}")
    else:
        print("No hubo mejora; usa last.pt.")


# ============================== Live ==============================

def rs_pipeline(width=640, height=480, fps=30):
    if rs is None:
        raise RuntimeError('pyrealsense2 no disponible')
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(cfg)
    sensor = profile.get_device().first_depth_sensor()
    try: sensor.set_option(rs.option.laser_power, 360.0)
    except Exception: pass
    dec = rs.decimation_filter(2); spat = rs.spatial_filter(); temp = rs.temporal_filter(); hole = rs.hole_filling_filter()
    return pipeline, dec, spat, temp, hole

@torch.inference_mode()
def live(args):
    ckpt = torch.load(args.weights, map_location='cpu')
    meta = ckpt.get('meta', {})
    classes = meta.get('classes', [])
    if not classes:
        raise RuntimeError('El checkpoint no contiene la lista de clases.')
    num_classes = len(classes)

    near = int(meta.get('near_mm', 300)); far  = int(meta.get('far_mm', 4000))
    input_size = int(meta.get('input', args.input))
    roi_info = meta.get('roi', {'use_roi': True, 'roi_mode': 'center', 'roi_side': None})

    device = device_auto()
    model = SmallZonesNet(num_classes=num_classes, in_ch=1, width=args.width).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=False); model.eval()

    # Calibración T (si existe)
    T = 1.0
    if ckpt.get('calibration'):
        T = float(ckpt['calibration'].get('T', 1.0))

    # ROI runtime: por defecto usa lo del ckpt, pero puedes forzarlo por CLI si quisieras extender
    use_roi = roi_info.get('use_roi', True) if args.no_roi is None else (not args.no_roi)
    roi_mode = args.roi_mode or roi_info.get('roi_mode', 'center')
    roi_side = args.roi_side if args.roi_side is not None else roi_info.get('roi_side', None)

    pipeline, dec, spat, temp, hole = rs_pipeline()
    try:
        cv2.namedWindow('zones', cv2.WINDOW_NORMAL)

        while True:
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if depth is None: continue
            depth = dec.process(depth); depth = spat.process(depth); depth = temp.process(depth); depth = hole.process(depth)

            d = np.asanyarray(depth.get_data()).astype(np.uint16)
            if use_roi:
                h, w = d.shape[:2]
                side = int(0.6 * min(h, w)) if roi_side is None else int(roi_side)
                side = max(32, min(min(h, w), side))
                d_roi = crop_roi_min_depth(d, side) if roi_mode=='min' else crop_roi_center(d, side)
            else:
                d_roi = d

            x = normalize_depth(d_roi, near, far)
            x = resize_keep_ratio(x, input_size)
            vis = (x * 255.0).astype(np.uint8)

            xt = torch.from_numpy(np.expand_dims(x, 0)[None, ...]).float().to(device)
            z = model(xt) / max(T, 1e-6)
            p = torch.softmax(z, dim=1).cpu().numpy()[0]  # (C,)
            top = int(np.argmax(p))
            zone = classes[top]
            conf = float(p[top])

            hud = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            cv2.putText(hud, f"Zone={zone} P={conf:.3f} roi={roi_mode if use_roi else 'off'}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('zones', hud)
            if (cv2.waitKey(1) & 0^0xFF) == ord('q'): break  # ^ para evitar precedencia (igual a &)

    finally:
        cv2.destroyAllWindows()
        try: pipeline.stop()
        except Exception: pass


# ============================== CLI ==============================

def build_parser():
    p = argparse.ArgumentParser(description='Modelo de Zonas (multiclase): train + live')
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
    pt.add_argument('--near', type=int, default=300)
    pt.add_argument('--far',  type=int, default=4000)
    pt.add_argument('--patience', type=int, default=7)
    pt.add_argument('--no-roi', action='store_true', help='Desactiva ROI.')
    pt.add_argument('--roi-mode', type=str, default='center', choices=['center','min'])
    pt.add_argument('--roi-side', type=int, default=None, help='Lado del ROI en px del frame original (default 0.6*min(H,W)).')

    pt.set_defaults(func=train)

    # Live
    pl = sub.add_parser('live')
    pl.add_argument('--weights', type=str, required=True)
    pl.add_argument('--input', type=int, default=96)
    pl.add_argument('--width', type=int, default=32)
    pl.add_argument('--no-roi', action='store_true')
    pl.add_argument('--roi-mode', type=str, default=None, choices=['center','min'])
    pl.add_argument('--roi-side', type=int, default=None)

    pl.set_defaults(func=live)
    return p

if __name__ == '__main__':
    args = build_parser().parse_args()
    args.func(args)
