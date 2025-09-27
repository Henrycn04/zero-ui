#!/usr/bin/env python3
# Ejemplo de uso:
#   python pipeline.py --w-pres runs-pre/presence_exp2/best.pt --w-point runs-poi/pointing_exp2/best.pt --w-zones runs-zon/zones_exp1/best.pt --no-roi
#
#   Desactivar ROI (pointing y zones):
#     python pipeline.py ... --no-roi
#
#   Cambiar inputs (por defecto 96):
#     python pipeline.py ... --inp-pres 96 --inp-point 96 --inp-zones 96

import argparse
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


# ============================== Utilidades ==============================

def set_seed(seed: int = 1):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_config(path='config.yaml'):
    if not os.path.exists(path): return {}
    with open(path, 'r') as f: return yaml.safe_load(f) or {}

def normalize_depth(d16: np.ndarray, near_mm: int = 300, far_mm: int = 4000) -> np.ndarray:
    d = d16.astype(np.float32)
    d[d <= 0] = np.nan
    d = np.clip(d, float(near_mm), float(far_mm))
    d = (d - float(near_mm)) / max(1.0, float(far_mm - near_mm))
    d = np.clip(d, 0.0, 1.0); d[np.isnan(d)] = 0.0
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

def apply_calib_bin(logit, T=1.0, b=0.0):
    return logit / max(T, 1e-6) + b

def crop_roi_center(d16: np.ndarray, side_px: int):
    h, w = d16.shape[:2]; side = int(min(h, w, side_px))
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - side // 2); x0 = max(0, cx - side // 2)
    y1 = min(h, y0 + side);      x1 = min(w, x0 + side)
    roi = d16[y0:y1, x0:x1]
    if roi.shape[0] != side or roi.shape[1] != side:
        canvas = np.zeros((side, side), dtype=d16.dtype)
        yy = (side - roi.shape[0]) // 2; xx = (side - roi.shape[1]) // 2
        canvas[yy:yy+roi.shape[0], xx:xx+roi.shape[1]] = roi
        roi = canvas
    return roi, (x0, y0, x1, y1)

def crop_roi_min_depth(d16: np.ndarray, side_px: int):
    h, w = d16.shape[:2]
    valid = d16.copy(); valid[valid == 0] = 65535
    idx = int(np.argmin(valid)); y, x = divmod(idx, w)
    side = int(min(h, w, side_px))
    y0 = max(0, y - side // 2); x0 = max(0, x - side // 2)
    y1 = min(h, y0 + side);     x1 = min(w, x0 + side)
    roi = d16[y0:y1, x0:x1]
    if roi.shape[0] != side or roi.shape[1] != side:
        canvas = np.zeros((side, side), dtype=d16.dtype)
        yy = (side - roi.shape[0]) // 2; xx = (side - roi.shape[1]) // 2
        canvas[yy:yy+roi.shape[0], xx:xx+roi.shape[1]] = roi
        roi = canvas
    return roi, (x0, y0, x1, y1)

def device_auto():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================== Modelos (mismo estilo) ==============================

class SmallPresenceNet(nn.Module):
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
    def forward(self, x): return self.head(self.backbone(x).flatten(1)).squeeze(1)

class SmallPointingNet(nn.Module):
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
    def forward(self, x): return self.head(self.backbone(x).flatten(1)).squeeze(1)

class SmallZonesNet(nn.Module):
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
    def forward(self, x): return self.head(self.backbone(x).flatten(1))


# ============================== RealSense ==============================

def rs_pipeline(width=640, height=480, fps=30):
    if rs is None: raise RuntimeError('pyrealsense2 no disponible')
    pipeline = rs.pipeline(); cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(cfg)
    sensor = profile.get_device().first_depth_sensor()
    try: sensor.set_option(rs.option.laser_power, 360.0)
    except Exception: pass
    dec = rs.decimation_filter(2); spat = rs.spatial_filter(); temp = rs.temporal_filter(); hole = rs.hole_filling_filter()
    return pipeline, dec, spat, temp, hole


# ============================== Helpers inferencia ==============================

def bin_predict(model, x_tensor, thr=0.5, calib=None):
    z = model(x_tensor)  # logits
    zv = z.detach().cpu().numpy().reshape(-1)[0]
    if calib:
        T = float(calib.get('T', 1.0)); b = float(calib.get('b', 0.0))
        zv = apply_calib_bin(zv, T, b)
    p = 1.0 / (1.0 + np.exp(-zv))
    flag = int(p >= thr)
    return float(p), int(flag)

def mc_predict(model, x_tensor, classes, calib=None):
    z = model(x_tensor)               # (1,C)
    if calib:
        T = float(calib.get('T', 1.0))
        z = z / max(T, 1e-6)
    p = torch.softmax(z, dim=1).cpu().numpy()[0]
    top = int(np.argmax(p))
    return classes[top], float(p[top])


# ============================== Live (pipeline secuencial) ==============================

@torch.inference_mode()
def live(args):
    device = device_auto()
    cfg = load_config('config.yaml')
    near = int(cfg.get('capture', {}).get('near_mm', 300))
    far  = int(cfg.get('capture',  {}).get('far_mm',  4000))

    # --------- PRESENCE ---------
    ckpt_pres = torch.load(args.w_pres, map_location='cpu')
    meta_pres = ckpt_pres.get('meta', {})
    inp_pres  = int(meta_pres.get('input', args.inp_pres))
    thr_pres  = args.th_pres
    if args.use_ckpt_thr_pres and ckpt_pres.get('thresholds'):
        thr_pres = float(ckpt_pres['thresholds'].get('thr', thr_pres))
    calib_pres = None if args.no_calib_pres else ckpt_pres.get('calibration', None)

    model_pres = SmallPresenceNet(in_ch=1, width=args.width).to(device)
    model_pres.load_state_dict(ckpt_pres['state_dict'], strict=False); model_pres.eval()

    # --------- POINTING ---------
    ckpt_point = torch.load(args.w_point, map_location='cpu')
    meta_point = ckpt_point.get('meta', {})
    roi_point  = meta_point.get('roi', {'use_roi': True, 'roi_mode': 'center', 'roi_side': None})
    inp_point  = int(meta_point.get('input', args.inp_point))
    thr_point  = args.th_point
    if args.use_ckpt_thr_point and ckpt_point.get('thresholds'):
        thr_point = float(ckpt_point['thresholds'].get('thr', thr_point))
    calib_point = None if args.no_calib_point else ckpt_point.get('calibration', None)

    model_point = SmallPointingNet(in_ch=1, width=args.width).to(device)
    model_point.load_state_dict(ckpt_point['state_dict'], strict=False); model_point.eval()

    # --------- ZONES ---------
    ckpt_zones = torch.load(args.w_zones, map_location='cpu')
    meta_zones = ckpt_zones.get('meta', {})
    classes    = meta_zones.get('classes', [])
    if not classes: raise RuntimeError("El checkpoint de zonas no trae 'classes'.")
    roi_zones  = meta_zones.get('roi', {'use_roi': True, 'roi_mode': 'center', 'roi_side': None})
    inp_zones  = int(meta_zones.get('input', args.inp_zones))
    calib_zones= None if args.no_calib_zones else ckpt_zones.get('calibration', None)

    model_zones = SmallZonesNet(num_classes=len(classes), in_ch=1, width=args.width).to(device)
    model_zones.load_state_dict(ckpt_zones['state_dict'], strict=False); model_zones.eval()

    # --------- Cámara ---------
    pipeline, dec, spat, temp, hole = rs_pipeline()
    try:
        cv2.namedWindow('presence', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('pointing', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('zones',    cv2.WINDOW_AUTOSIZE)

        last_zone = ('-', 0.0)
        while True:
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if depth is None: continue
            depth = dec.process(depth); depth = spat.process(depth); depth = temp.process(depth); depth = hole.process(depth)

            d = np.asanyarray(depth.get_data()).astype(np.uint16)
            h, w = d.shape[:2]

            vis_full = (normalize_depth(d, near, far) * 255.0).astype(np.uint8)
            # ================= Presence =================
            x_pres = resize_keep_ratio(normalize_depth(d, near, far), inp_pres)
            xt_pres = torch.from_numpy(np.expand_dims(x_pres,0)[None,...]).float().to(device)
            p_pres, flag_pres = bin_predict(model_pres, xt_pres, thr=thr_pres, calib=calib_pres)

            hud_pres = cv2.cvtColor(vis_full, cv2.COLOR_GRAY2BGR)
            cv2.putText(hud_pres, f"pre P={p_pres:.3f} thr={thr_pres:.2f} a{flag_pres}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('presence', hud_pres)

            # ================= Pointing =================
            hud_point = cv2.cvtColor(vis_full, cv2.COLOR_GRAY2BGR)
            if args.no_roi:
                d_roi_p, (x0p,y0p,x1p,y1p) = d, (0,0,w,h)
            else:
                sidep = int(0.6 * min(h, w)) if roi_point.get('roi_side') is None else int(roi_point['roi_side'])
                sidep = max(32, min(min(h, w), sidep))
                if roi_point.get('roi_mode','center') == 'min':
                    d_roi_p, (x0p,y0p,x1p,y1p) = crop_roi_min_depth(d, sidep)
                else:
                    d_roi_p, (x0p,y0p,x1p,y1p) = crop_roi_center(d, sidep)
            cv2.rectangle(hud_point, (x0p,y0p), (x1p,y1p), (255,255,255), 1)

            p_point, flag_point = 0.0, 0
            if flag_pres == 1:
                x_point = resize_keep_ratio(normalize_depth(d_roi_p, near, far), inp_point)
                xt_point = torch.from_numpy(np.expand_dims(x_point,0)[None,...]).float().to(device)
                p_point, flag_point = bin_predict(model_point, xt_point, thr=thr_point, calib=calib_point)

            cv2.putText(hud_point, f"poi P={p_point:.3f} thr={thr_point:.2f} a={flag_point}  (gp={flag_pres})",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('pointing', hud_point)

            # ================= Zones =================
            hud_z = cv2.cvtColor(vis_full, cv2.COLOR_GRAY2BGR)
            if args.no_roi:
                d_roi_z, (x0z,y0z,x1z,y1z) = d, (0,0,w,h)
            else:
                sidez = int(0.6 * min(h, w)) if roi_zones.get('roi_side') is None else int(roi_zones['roi_side'])
                sidez = max(32, min(min(h, w), sidez))
                if roi_zones.get('roi_mode','center') == 'min':
                    d_roi_z, (x0z,y0z,x1z,y1z) = crop_roi_min_depth(d, sidez)
                else:
                    d_roi_z, (x0z,y0z,x1z,y1z) = crop_roi_center(d, sidez)
            cv2.rectangle(hud_z, (x0z,y0z), (x1z,y1z), (255,255,255), 1)

            zone_lbl, zone_conf, active_zone = '-', 0.0, 0
            if flag_point == 1:
                x_zone = resize_keep_ratio(normalize_depth(d_roi_z, near, far), inp_zones)
                xt_zone = torch.from_numpy(np.expand_dims(x_zone,0)[None,...]).float().to(device)
                zone_lbl, zone_conf = mc_predict(model_zones, xt_zone, classes, calib=calib_zones)
                active_zone = 1

            cv2.putText(hud_z, f"zon z={zone_lbl} P={zone_conf:.3f} a={active_zone}  (gp={flag_point})",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('zones', hud_z)

            # Salir
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        try: pipeline.stop()
        except Exception: pass


# ============================== CLI ==============================

def build_parser():
    p = argparse.ArgumentParser(description='Pipeline: presence -> pointing -> zones (3 ventanas)')
    p.add_argument('--w-pres', type=str, required=True, help='Checkpoint presence (best.pt).')
    p.add_argument('--w-point', type=str, required=True, help='Checkpoint pointing (best.pt).')
    p.add_argument('--w-zones', type=str, required=True, help='Checkpoint zones (best.pt).')
    p.add_argument('--width', type=int, default=32, help='Ancho base de canales de los CNN.')

    # Inputs (por defecto 96; solo cambian si los pasas)
    p.add_argument('--inp-pres', type=int, default=96)
    p.add_argument('--inp-point', type=int, default=96)
    p.add_argument('--inp-zones', type=int, default=96)

    # Umbrales binarios
    p.add_argument('--th-pres', type=float, default=0.90)
    p.add_argument('--th-point', type=float, default=0.90)
    p.add_argument('--use-ckpt-thr-pres', action='store_true')
    p.add_argument('--use-ckpt-thr-point', action='store_true')

    # Calibraciones
    p.add_argument('--no-calib-pres', action='store_true')
    p.add_argument('--no-calib-point', action='store_true')
    p.add_argument('--no-calib-zones', action='store_true')

    # ROI global (afecta pointing y zones)
    p.add_argument('--no-roi', action='store_true', help='Desactiva ROI para pointing y zones (usa frame completo).')

    p.set_defaults(func=live)
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    set_seed(1)
    args.func(args)
