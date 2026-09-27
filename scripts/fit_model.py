"""Идентификация параметров модели тяги и торможения по прогонам rosbag2.

Для каждого прогона: масштаб колёс k — медиана отношения «скорость тележки / скорость GNSS»
на ходу (> 3 м/с; нет /sensing/gnss/master/vel — k по умолчанию), отсчёты скорости и
ускорения на чистых участках (tram_odometry.identify), уклон — по карте линии, если задана.
Затем общий подбор параметров по всем прогонам. Результат — полный файл параметров ноды:
config/tram_odometry.yaml с новыми model.*, filter.k0 и путями к карте.

Запуск (нужны pip-пакеты rosbags и PyYAML):
    python3 scripts/fit_model.py <прогон> [...] [--route-map maps/route.csv --stops maps/stops.csv] \
        --out maps/fitted.yaml [--report maps/fit_report.json]
"""
import argparse
import bisect
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'tram_odometry'))
from evaluate_bags import CMD, FRONT, MASTER, MASTER_VEL, REAR, _stamp, read_bag  # noqa: E402
from tram_odometry.identify import fit, samples_from_run  # noqa: E402
from tram_odometry.mapping import Locator  # noqa: E402
from tram_odometry.model import ModelParams  # noqa: E402
from tram_odometry.track import MgrsLocal, RouteTrack  # noqa: E402

BASE_CONFIG = ROOT / 'src' / 'tram_odometry' / 'config' / 'tram_odometry.yaml'
K_DEFAULT = 3.5966


def run_series(events):
    """Ряды одного прогона по header.stamp: notch, front, rear, master [(t, x, y)], gnss_v."""
    frame = MgrsLocal()
    out = {'notch': [], 'front': [], 'rear': [], 'master': [], 'gnss_v': []}
    for _, topic, m in events:
        t = _stamp(m)
        if topic == CMD:
            out['notch'].append((t, int(m.position)))
        elif topic == FRONT:
            out['front'].append((t, float(m.velocity)))
        elif topic == REAR:
            out['rear'].append((t, float(m.velocity)))
        elif topic == MASTER and m.status.status >= 0 and math.isfinite(m.latitude):
            x, y, _ = frame.forward(m.latitude, m.longitude, m.altitude)
            out['master'].append((t, x, y))
        elif topic == MASTER_VEL:
            lin = getattr(m.twist, 'twist', m.twist).linear
            out['gnss_v'].append((t, math.hypot(lin.x, lin.y)))
    for v in out.values():
        v.sort()
    return out


def wheel_scale(series, v_min=3.0):
    """k прогона: медиана (скорость тележки, км/ч) / (скорость GNSS, м/с) на ходу, или None."""
    gv = series['gnss_v']
    if not gv:
        return None
    gt = [g[0] for g in gv]
    ratios = []
    for t, kmh in series['front']:
        i = bisect.bisect_left(gt, t)
        near = [j for j in (i - 1, i) if 0 <= j < len(gv) and abs(gt[j] - t) <= 0.05]
        if near and gv[near[0]][1] > v_min:
            ratios.append(kmh / gv[near[0]][1])
    return sorted(ratios)[len(ratios) // 2] if len(ratios) > 50 else None


def grade_fn(series, track, locator):
    if track is None:
        return None
    mt = [m[0] for m in series['master']]

    def f(t):
        i = bisect.bisect_left(mt, t)
        near = [j for j in (i - 1, i) if 0 <= j < len(mt) and abs(mt[j] - t) <= 0.5]
        if not near:
            return None
        _, x, y = series['master'][near[0]]
        s = locator.s_at(x, y)
        return None if s is None else track.grade_at(s)
    return f


def collect(bags, track=None):
    """-> ([отсчёты прогона], [(прогон, k)])."""
    locator = Locator(track) if track is not None else None
    runs, scales = [], []
    for bag in bags:
        series = run_series(read_bag(bag))
        k = wheel_scale(series)
        scales.append((Path(bag).name, k))
        runs.append(samples_from_run(series['notch'], series['front'], series['rear'], k or K_DEFAULT,
                                     grade_fn(series, track, locator)))
    return runs, scales


def write_params(path, params, k0, route_map=None, stops=None, base=BASE_CONFIG):
    cfg = yaml.safe_load(Path(base).read_text(encoding='utf-8'))
    ros = cfg['tram_odometry']['ros__parameters']
    for key, value in asdict(params).items():
        ros['model'][key] = int(value) if key == 'notch_max' else float(value)
    ros['filter']['k0'] = float(k0)
    if route_map:
        ros['route_map_file'] = str(Path(route_map).resolve())
    if stops:
        ros['stops_file'] = str(Path(stops).resolve())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Сгенерировано scripts/fit_model.py: модель и k0 идентифицированы по прогонам.\n')
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--route-map', default=None)
    ap.add_argument('--stops', default=None)
    ap.add_argument('--out', default='maps/fitted.yaml')
    ap.add_argument('--report', default=None)
    args = ap.parse_args()
    track = RouteTrack.load(args.route_map, MgrsLocal()) if args.route_map else None
    runs, scales = collect(args.bags, track)
    params, report = fit(runs, ModelParams())
    ks = sorted(k for _, k in scales if k)
    k0 = ks[len(ks) // 2] if ks else K_DEFAULT
    write_params(args.out, params, k0, args.route_map, args.stops)
    report.update(model=asdict(params), k0=k0, scales=scales)
    print(json.dumps({k: v for k, v in report.items() if k != 'scales'}, ensure_ascii=False, indent=1))
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
