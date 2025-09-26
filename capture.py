#!/usr/bin/env python3
# Ejemplos de uso:
#   python capture.py --out data/s1 --label-live
#   python capture.py --out data/s1 --align

import argparse
from pathlib import Path
import time

import numpy as np
import cv2
import pandas as pd
import yaml

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

# ---------------------- Utilidades (sin cambios) ----------------------

def ensure_dir(p: Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def normalize_depth(d16: np.ndarray, near_mm: int = 300, far_mm: int = 4000) -> np.ndarray:
    """Recorta a [near, far] y normaliza a [0,1]. Zeros o NaNs a 0."""
    d = d16.astype(np.float32)
    d[d <= 0] = np.nan
    if near_mm is not None:
        d = np.maximum(d, float(near_mm))
    if far_mm is not None:
        d = np.minimum(d, float(far_mm))
    d = (d - float(near_mm)) / max(1.0, float(far_mm - near_mm))
    d = np.clip(d, 0.0, 1.0)
    d[np.isnan(d)] = 0.0
    return d

def depth_to_png_u16(d_mm: np.ndarray) -> np.ndarray:
    return d_mm.astype(np.uint16)

def png_u16_to_depth_mm(img: np.ndarray) -> np.ndarray:
    return img.astype(np.uint16)

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

# ---------------------------- Captura (igual) ----------------------------

def capture(args):
    out = Path(args.out)
    ensure_dir(out)
    img_dir = out / 'images'
    ensure_dir(img_dir)
    csv_path = out / 'labels.csv'

    # Leer zonas y grid si hay etiquetado en vivo
    if args.label_live:
        with open('config.yaml','r') as f:
            cfg_all = yaml.safe_load(f)
        zones = cfg_all.get('zones', [])
        grid_cols = int(cfg_all.get('grid', {}).get('cols', 3))
    else:
        zones = []
        grid_cols = 3

    # Near/far desde config si existen (para visual)
    try:
        near_mm = int(cfg_all.get('capture', {}).get('near_mm', 300)) if args.label_live else 300
        far_mm  = int(cfg_all.get('capture', {}).get('far_mm', 4000)) if args.label_live else 4000
    except Exception:
        near_mm, far_mm = 300, 4000

    # Abrir cámara
    pipeline, align, dec, spat, temp, hole = rs_pipeline(
        enable_filters=True,
        enable_align=bool(args.align),
        align_to='depth',
        enable_color=bool(args.align)
    )

    i = 0
    labels = []
    presence = 0
    pointing = 0
    zone_sel = 0  # 0 = none; 1..K = índice (1-based) sobre zones

    try:
        cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
        while True:
            frames = pipeline.wait_for_frames()

            # Alineación si corresponde
            if align is not None:
                try:
                    frames = align.process(frames)
                except Exception:
                    pass

            depth = frames.get_depth_frame()
            if depth is None:
                continue

            # Aplicar filtros si están disponibles
            if dec is not None:  depth = dec.process(depth)
            if spat is not None: depth = spat.process(depth)
            if temp is not None: depth = temp.process(depth)
            if hole is not None: hole.process(depth)

            d = np.asanyarray(depth.get_data()).astype(np.uint16)

            # Vista rápida (8-bit) para GUI
            vis = (normalize_depth(d, near_mm, far_mm) * 255.0).astype(np.uint8)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            # HUD simple (estado actual)
            txt = f'p={presence}  g={pointing}  zone={zone_sel if zone_sel>=0 else 0}'
            cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            if zones:
                zname = zones[zone_sel-1] if (zone_sel>0 and zone_sel<=len(zones)) else 'none'
                cv2.putText(vis, f'Z: {zname}', (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow('depth', vis)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                break

            # Guardar frame (y etiqueta si corresponde)
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

            # Etiquetado rápido (igual que tu script)
            if not args.label_live:
                continue

            # Teclas de zona: 0..9, 'n' (ninguna)
            if k in [ord(str(dig)) for dig in range(10)]:
                zone_sel = int(chr(k))  # 0..9
                if zone_sel < 0:
                    zone_sel = 0
            if k == ord('n'):
                zone_sel = 0; pointing = 0
            if k == ord('p'):
                presence = 1 - presence
            if k == ord('g'):
                pointing = 1 - pointing

    finally:
        # Guardar CSV si hubo etiquetas
        if args.label_live and labels:
            df = pd.DataFrame(labels, columns=['path','presence','pointing','row','column','target'])
            df.to_csv(csv_path, index=False)
            print(f"Etiquetas guardadas en {csv_path}")

        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass

def parse_args():
    p = argparse.ArgumentParser(description='Captura y etiquetado (depth-only)')
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--label-live', action='store_true')
    p.add_argument('--align', action='store_true', help='Habilitar alineación (abre también el stream de color).')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    capture(args)
