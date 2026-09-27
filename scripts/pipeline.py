"""Весь цикл одной командой: карта -> подбор модели -> оценка на отложенных прогонах.

1. Прогоны делятся на обучающие и проверочные (детерминированно, по хешу имени).
2. По обучающим: карта линии и места стоянок (build_route_map), параметры модели и k0
   (fit_model) -> <out>/route.csv, stops.csv, fitted.yaml — готовый файл параметров ноды.
3. По проверочным — метрики судьи (evaluate_bags) для вариантов:
   fitted — наша нода с картой и подобранной моделью; default — модель по умолчанию;
   no_map — без карты (прямая от старта); naive — сырая одометрия: средняя скорость тележек
   без отсева сбоев, та же выставка и карта. И ошибка скорости одной модели за 20 с без колёс.
4. <out>/results.json — всё для отчёта жюри (scripts/make_report.py).

Запуск (pip-пакеты rosbags и PyYAML):
    python3 scripts/pipeline.py --dataset ../dataset/data --out-dir results/
    python3 scripts/pipeline.py <прогон> [...] --out-dir results/ --test-frac 0.2
"""
import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from build_route_map import build_map
from evaluate_bags import FRONT, MASTER, REAR, _stamp, metrics, read_bag, reference, replay
from fit_model import collect, write_params
from tram_odometry.estimator import FilterParams
from tram_odometry.identify import fit, open_loop_errors
from tram_odometry.model import ModelParams
from tram_odometry.positioning import Positioner
from tram_odometry.track import MgrsLocal, RouteTrack


def split(bags, test_frac):
    def h(b):
        return int(hashlib.sha1(Path(b).name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    test = [b for b in bags if h(b) < test_frac]
    if not test:
        test = [min(bags, key=h)]
    return [b for b in bags if b not in test], test


class _NaiveState:
    def __init__(self, s, v):
        self.s, self.v = s, v


class Naive:
    """Сырая одометрия: последняя средняя скорость тележек / k, интегрирование по времени."""

    def __init__(self, k):
        self.k, self.s, self.t, self.v = k, 0.0, None, 0.0
        self.last = {}
        self.resets = 0

    def wheel(self, which, t, kmh):
        if self.t is not None and t > self.t:
            self.s += self.v * (t - self.t)
        self.t = t if self.t is None else max(self.t, t)
        self.last[which] = kmh / self.k
        self.v = max(0.0, sum(self.last.values()) / len(self.last))

    def set_position(self, s, sigma):
        self.s = s

    def state(self):
        return _NaiveState(self.s, self.v)


def naive_replay(events, route, k):
    est, pos, out = Naive(k), Positioner(route), []
    for _, topic, m in events:
        t = _stamp(m)
        if topic == MASTER or topic.endswith('rover/fix'):
            if not pos.locked:
                pos.fix(t, m.latitude, m.longitude, m.altitude, m.status.status, topic != MASTER)
            continue
        if topic not in (FRONT, REAR):
            continue
        est.wheel('front' if topic == FRONT else 'rear', t, float(m.velocity))
        pos.update(t, est)
        if out and t < out[-1][0]:
            continue
        x, y, z, yaw, frame = pos.pose(est.s)
        out.append((t, est.v, x, y, z, yaw, frame, False))
    return out


def _median(vals):
    v = sorted(x for x in vals if isinstance(x, float) and math.isfinite(x))
    return v[len(v) // 2] if v else None


def _trace(out, ref_pos, ref_vel, naive, every_s=1.0):
    """Ряды для графиков: раз в every_s — эталон, наша оценка, сырая одометрия, ошибки."""
    def at(series, t):
        lo, hi = 0, len(series) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if series[mid][0] <= t else (lo, mid - 1)
        return series[lo]
    rv = {'t': [], 'v_ref': [], 'v_est': [], 'v_naive': [], 'err': [], 'err_naive': [], 'slip': [],
          'x_ref': [], 'y_ref': [], 'x_est': [], 'y_est': []}
    t0 = out[0][0]
    next_t = t0
    for t, v in ref_vel:
        if t < next_t or t < t0:
            continue
        next_t = t + every_s
        o, n = at(out, t), at(naive, t)
        p = at(ref_pos, t) if ref_pos else None
        rv['t'].append(round(t - t0, 1))
        rv['v_ref'].append(round(v, 3))
        rv['v_est'].append(round(o[1], 3))
        rv['v_naive'].append(round(n[1], 3))
        rv['slip'].append(1 if o[7] else 0)
        if p and abs(p[0] - t) < 0.5 and o[6] == 'map':
            rv['err'].append(round(math.dist(p[1:4], o[2:5]), 2))
            rv['err_naive'].append(round(math.dist(p[1:4], n[2:5]), 2))
            rv['x_ref'].append(round(p[1], 1))
            rv['y_ref'].append(round(p[2], 1))
            rv['x_est'].append(round(o[2], 1))
            rv['y_est'].append(round(o[3], 1))
        else:
            for key in ('err', 'err_naive', 'x_ref', 'y_ref', 'x_est', 'y_est'):
                rv[key].append(None)
    return rv


def run(bags, out_dir, test_frac=0.2):
    started = time.time()
    out = Path(out_dir)
    train, test = split(sorted(bags), test_frac)
    print(f'обучающих {len(train)}, проверочных {len(test)}')
    map_info = build_map(train, out)
    print('карта:', map_info)
    route = RouteTrack.load(out / 'route.csv', MgrsLocal(), out / 'stops.csv')
    train_runs, scales = collect(train, route)
    params, fit_rep = fit(train_runs, ModelParams())
    ks = sorted(k for _, k in scales if k)
    k0 = ks[len(ks) // 2] if ks else FilterParams().k0
    write_params(out / 'fitted.yaml', params, k0, out / 'route.csv', out / 'stops.csv')
    print('модель:', params, 'k0', round(k0, 4))

    test_runs, _ = collect(test, route)
    dropout = {}
    for name, p in (('fitted', params), ('default', ModelParams())):
        errs = [abs(e) for r in test_runs for e in open_loop_errors(r, p)]
        errs.sort()
        dropout[name] = {'n': len(errs), 'median': errs[len(errs) // 2] if errs else None,
                         'p90': errs[int(0.9 * len(errs))] if errs else None}

    fparams = FilterParams(k0=k0)
    variants = {'fitted': dict(route=route, model=params, fparams=fparams),
                'default': dict(route=route, model=ModelParams(), fparams=FilterParams()),
                'no_map': dict(route=None, model=params, fparams=fparams)}
    per = {v: [] for v in list(variants) + ['naive']}
    trace, trace_score, hours, km = None, -1, 0.0, 0.0
    for bag in test:
        events = read_bag(bag)
        ref_pos, ref_vel = reference(events)
        outs = {v: replay(events, **kw) for v, kw in variants.items()}
        outs['naive'] = naive_replay(events, route, k0)
        for v, o in outs.items():
            per[v].append({'bag': Path(bag).name, **metrics(o, ref_pos, ref_vel)})
        if outs['fitted']:
            hours += (outs['fitted'][-1][0] - outs['fitted'][0][0]) / 3600.0
        km += per['fitted'][-1].get('dist_m', 0.0) / 1000.0
        slips = sum(1 for o in outs['fitted'] if o[7])
        if ref_vel and slips > trace_score:
            trace, trace_score = _trace(outs['fitted'], ref_pos, ref_vel, outs['naive']), slips
            trace['bag'] = Path(bag).name
        print(Path(bag).name, {v: round(per[v][-1].get('p3d_mean', float('nan')), 2) for v in per})
    keys = sorted({k for rows in per.values() for r in rows for k in r if k != 'bag'})
    summary = {v: {k: _median([r.get(k) for r in rows]) for k in keys} for v, rows in per.items()}
    preview = json.loads((out / 'route_preview.json').read_text(encoding='utf-8'))
    result = {
        'generated': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), 'runtime_s': round(time.time() - started, 1),
        'dataset': {'bags': len(bags), 'train': len(train), 'test': len(test), 'test_hours': round(hours, 2),
                    'test_km': round(km, 2), 'test_bags': [Path(b).name for b in test]},
        'map': preview, 'fit': {**fit_rep, 'model': asdict(params), 'k0': k0, 'scales': scales},
        'dropout_20s': dropout, 'metrics': {'per_bag': per, 'median': summary}, 'trace': trace,
    }
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    for v, row in summary.items():
        shown = ('v_rmse', 'p3d_mean', 'p3d_max', 'drift_pct', 'along_rmse')
        print(f'{v:8s}', ' '.join(f'{k}={row[k]:.3f}' for k in shown if row.get(k) is not None))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bags', nargs='*')
    ap.add_argument('--dataset', default=None, help='папка с прогонами (каталоги rosbag2)')
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--test-frac', type=float, default=0.2)
    args = ap.parse_args()
    bags = list(args.bags)
    if args.dataset:
        bags += sorted(str(p.parent) for p in Path(args.dataset).glob('*/metadata.yaml'))
    if len(bags) < 2:
        raise SystemExit('нужно хотя бы два прогона')
    run(bags, args.out_dir, args.test_frac)


if __name__ == '__main__':
    main()
