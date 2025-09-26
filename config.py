#!/usr/bin/env python3
# Ejemplos de uso:
#   (tipo solicitado)    python config.py --zones A1,A2,A3,B1,B2,B3,C1,C2,C3 --mqtt-host 192.168.1.50
#   (CLI real del script) python config.py configure --zones A1,A2,A3,B1,B2,B3,C1,C2,C3 --mqtt-host 192.168.1.50

import argparse
import yaml

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

def parse_args():
    p = argparse.ArgumentParser(description='Configurar zonas, MQTT y rangos de captura (depth-only)')
    sub = p.add_subparsers(dest='cmd', required=True)

    # configure (igual que en tu script original)
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

    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.func(args)
