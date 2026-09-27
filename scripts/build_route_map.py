"""Карта линии и места стоянок по GNSS обучающих прогонов (разрешено организаторами, QA 25.09).

Берёт фиксы антенны master из прогонов rosbag2, сшивает их в один замкнутый маршрут
(tram_odometry.mapping) и пишет два CSV, которые читает нода (route_map_file, stops_file):
    route.csv — s_m, lat, lon, alt: точки через 2 м по пути антенны master;
    stops.csv — s_m, sigma_m: места регулярных стоянок.

Запуск (нужен pip-пакет rosbags):
    python3 scripts/build_route_map.py <прогон> [<прогон> ...] --out-dir maps/
"""
import argparse
import json
import math
import sys
from pathlib import Path

from rosbags.highlevel import AnyReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'tram_odometry'))
from evaluate_bags import MASTER, _stamp, typestore  # noqa: E402
from tram_odometry.mapping import build  # noqa: E402
from tram_odometry.track import LocalFrame, MgrsLocal  # noqa: E402


def read_fixes(path):
    """[(метка, lat, lon, alt)] антенны master с решением (status >= 0)."""
    out = []
    with AnyReader([Path(path)], default_typestore=typestore()) as reader:
        conns = [c for c in reader.connections if c.topic == MASTER]
        for conn, _, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            if m.status.status >= 0 and all(math.isfinite(v) for v in (m.latitude, m.longitude, m.altitude)):
                out.append((_stamp(m), m.latitude, m.longitude, m.altitude))
    return sorted(out)


def build_map(bags, out_dir, step=2.0, min_runs=3):
    """-> отчёт; пишет route.csv, stops.csv и route_preview.json (для отчёта жюри)."""
    raw = [f for f in (read_fixes(b) for b in bags) if f]
    if not raw:
        raise SystemExit('ни в одном прогоне нет фиксов master')
    frame = LocalFrame(*raw[0][0][1:])
    runs = [[(t, *frame.forward(la, lo, al)) for t, la, lo, al in f] for f in raw]
    track, info = build(runs, step, min_runs=min_runs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'route.csv', 'w', encoding='utf-8') as f:
        f.write(f'# карта линии: {len(raw)} прогонов, длина {track.length:.1f} м, '
                f'замкнута: {info["closed"]}\ns_m,lat,lon,alt\n')
        for s, x, y, z in zip(track.s, track.x, track.y, track.z):
            lat, lon, alt = frame.inverse(x, y, z)
            f.write(f'{s:.3f},{lat:.9f},{lon:.9f},{alt:.3f}\n')
    with open(out / 'stops.csv', 'w', encoding='utf-8') as f:
        f.write('s_m,sigma_m\n')
        for s, sigma in track.stops:
            f.write(f'{s:.2f},{sigma:.2f}\n')
    every = max(1, len(track.s) // 1500)
    mgrs = MgrsLocal()

    def grid(x, y, z):                              # превью — в сетке судьи, как выход ноды
        return mgrs.forward(*frame.inverse(x, y, z))
    pts = [grid(x, y, z) for x, y, z in zip(track.x[::every], track.y[::every], track.z[::every])]
    stops = [grid(*track.pose(s)[:3]) for s, _ in track.stops]
    preview = {'x': [round(p[0], 1) for p in pts], 'y': [round(p[1], 1) for p in pts],
               'z': [round(p[2], 2) for p in pts], 's': [round(v, 1) for v in track.s[::every]],
               'stops': [round(s, 1) for s, _ in track.stops],
               'stops_xy': [[round(p[0], 1), round(p[1], 1)] for p in stops], **info}
    (out / 'route_preview.json').write_text(json.dumps(preview), encoding='utf-8')
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--out-dir', default='maps')
    ap.add_argument('--step', type=float, default=2.0)
    ap.add_argument('--min-runs', type=int, default=3, help='стоянка засчитывается, если стояли в N прогонах')
    args = ap.parse_args()
    info = build_map(args.bags, args.out_dir, args.step, args.min_runs)
    print(' '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in info.items()))


if __name__ == '__main__':
    main()
