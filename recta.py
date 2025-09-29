#!/usr/bin/env python3
# Ejemplo de uso:
#   python recta.py live --weights runs-poi/pointing_exp2/best.pt
#   (opcional) cambiar input:
#   python recta.py live --weights runs-poi/pointing_exp2/best.pt --input 96

import argparse
import os
from datetime import datetime
import yaml
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


# ============================== Utilidades ==============================

def load_config(path='config.yaml'):
    if not os.path.exists(path): return {}
    with open(path, 'r') as f: return yaml.safe_load(f) or {}

def device_auto():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_depth(d16: np.ndarray, near_mm: int = 300, far_mm: int = 4000) -> np.ndarray:
    d = d16.astype(np.float32)
    d[d <= 0] = np.nan
    d = np.clip(d, float(near_mm), float(far_mm))
    d = (d - float(near_mm)) / max(1.0, float(far_mm - near_mm))
    d = np.clip(d, 0.0, 1.0)
    d[np.isnan(d)] = 0.0
    return d

def resize_keep_ratio(img: np.ndarray, out_hw: int) -> np.ndarray:
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


# ============================== Modelo pointing + Grad-CAM ==============================

class SmallPointingNet(nn.Module):
    """CNN binaria (1 canal) estilo de tus scripts."""
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
        feat = self.backbone(x).flatten(1)
        return self.head(feat).squeeze(1)

class GradCAM:
    """Grad-CAM sobre la última Conv2d del backbone."""
    def __init__(self, model, target_layer=None):
        self.model = model
        self.features = None
        self.gradients = None
        self.target_layer = target_layer or self._find_last_conv(model)
        self.fwd_hook = self.target_layer.register_forward_hook(self._save_features)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _find_last_conv(self, model):
        last = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is None:
            raise RuntimeError("No se encontró Conv2d para Grad-CAM.")
        return last

    def _save_features(self, module, inp, out):
        self.features = out.detach()  # (B,C,H,W)

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()  # (B,C,H,W)

    def generate(self, score):
        if score.dim() != 0:
            score = score.mean()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        feats = self.features      # (B,C,H,W)
        grads = self.gradients     # (B,C,H,W)
        w = grads.mean(dim=(2,3), keepdim=True)
        cam = (w * feats).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.amin(dim=(1,2), keepdim=True)
        cam = cam / (cam.amax(dim=(1,2), keepdim=True) + 1e-6)
        return cam[0].detach().cpu().numpy()

    def close(self):
        try: self.fwd_hook.remove()
        except Exception: pass
        try: self.bwd_hook.remove()
        except Exception: pass


# ============================== RealSense ==============================

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


# ============================== LIVE ==============================

def build_parser():
    p = argparse.ArgumentParser(description='Visualiza Grad-CAM del modelo de pointing (heatmap overlay).')
    sub = p.add_subparsers(dest='cmd', required=True)
    pl = sub.add_parser('live')
    pl.add_argument('--weights', type=str, required=True, help='Ruta a best.pt del modelo pointing.')
    pl.add_argument('--input', type=int, default=96, help='Tamaño de input del modelo (cuadrado).')
    pl.add_argument('--width', type=int, default=32, help='Ancho base de canales en la CNN.')
    pl.add_argument('--near', type=int, default=300)
    pl.add_argument('--far',  type=int, default=4000)
    pl.add_argument('--alpha', type=float, default=0.35, help='Transparencia del heatmap (0..1).')
    pl.set_defaults(func=live)
    return p

def live(args):
    # Cargar checkpoint
    ckpt = torch.load(args.weights, map_location='cpu')
    meta = ckpt.get('meta', {})
    inp  = int(meta.get('input', args.input))
    # Calibración (si existe)
    T = 1.0; b = 0.0
    if ckpt.get('calibration'):
        T = float(ckpt['calibration'].get('T', 1.0))
        b = float(ckpt['calibration'].get('b', 0.0))

    device = device_auto()
    model = SmallPointingNet(in_ch=1, width=args.width).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    cam = GradCAM(model)

    # Near/Far desde config.yaml si existe
    cfg = load_config('config.yaml')
    near = int(cfg.get('capture', {}).get('near_mm', args.near))
    far  = int(cfg.get('capture',  {}).get('far_mm',  args.far))

    # Cámara
    pipeline, dec, spat, temp, hole = rs_pipeline()
    try:
        cv2.namedWindow('pointing-cam', cv2.WINDOW_AUTOSIZE)

        while True:
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if depth is None: continue

            depth = dec.process(depth); depth = spat.process(depth); depth = temp.process(depth); depth = hole.process(depth)
            d = np.asanyarray(depth.get_data()).astype(np.uint16)

            # Normalización para visual y para input
            x_full = normalize_depth(d, near, far)                # (H,W) float [0,1]
            x_in   = resize_keep_ratio(x_full, inp)               # (inp, inp)
            vis    = (x_full * 255.0).astype(np.uint8)            # visual de fondo
            hud    = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            # Forward + CAM (sin inference_mode)
            xt = torch.from_numpy(np.expand_dims(x_in, 0)[None, ...]).float().to(device)
            xt.requires_grad_(True)
            z = model(xt)                           # logit
            z_cal = z / max(T, 1e-6) + b
            p = torch.sigmoid(z_cal).detach().cpu().item()

            cam_small = cam.generate(z)             # (h', w') normalizado [0,1]
            cam_inp   = cv2.resize(cam_small, (inp, inp), interpolation=cv2.INTER_LINEAR)
            cam_full  = cv2.resize(cam_inp, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_LINEAR)
            cam_u8    = (np.clip(cam_full, 0.0, 1.0) * 255).astype(np.uint8)
            cam_color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)

            # Overlay (alpha)
            alpha = float(max(0.0, min(1.0, args.alpha)))
            hud = cv2.addWeighted(hud, 1.0, cam_color, alpha, 0.0)

            # Texto
            cv2.putText(hud, f"P_point={p:.3f}  input={inp}  alpha={alpha:.2f}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow('pointing-cam', hud)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        try: cam.close()
        except Exception: pass
        cv2.destroyAllWindows()
        try: pipeline.stop()
        except Exception: pass


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
