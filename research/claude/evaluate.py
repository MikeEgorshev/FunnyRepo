"""Офлайн-оценка оценщика по метрикам судьи на всех прогонах.

Прогон воспроизводится в порядке времени записи, как ros2 bag play. Оценщик получает
входы (тележки, контроллер) и GNSS master fix, но GNSS принимает только в окне выставки.
Каждый ответ оценщика — аналог публикации /result/* с меткой входа.

Эталон: GNSS master fix в ENU с началом в первой точке прогона (гипотеза о системе судьи)
и скорость GNSS (vel, а если его нет — по координатам). Выход и эталон сопоставляются
по ближайшей метке с допуском 0,05 с.

Запуск: python evaluate.py [--estimator baseline] [--per-vehicle] [--bags N]
        -> out/eval_<имя>.csv и сводка в консоль
"""
import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bagio import CMD, FRONT, GNSS_FIX, GNSS_VEL, MAP_DIR, REAR, bag_ids, load_cached
from tram_odometry.geo import Enu

OUT = Path(__file__).parent / 'out'
MAP = MAP_DIR / 'route.csv'
MATCH_TOL = 0.05
MIN_DURATION_S = 60.0


def make_estimator(name, bag_id, per_vehicle):
    vehicle = bag_id.split('_')[0] if per_vehicle else 'default'
    if name == 'baseline':
        from baseline import BaselineEstimator
        return BaselineEstimator(MAP, vehicle_id=vehicle)
    raise ValueError(f'неизвестный оценщик {name}')


def replay(est, d):
    """События в порядке времени записи bag -> массив выходов [stamp, v, x, y, z, s]."""
    events = []
    for topic, kind in ((FRONT, 'f'), (REAR, 'r'), (CMD, 'c'), (GNSS_FIX['master'], 'g')):
        a = d.get(topic)
        if a is not None:
            events += [(row[0], kind, row) for row in a]
    events.sort(key=lambda e: e[0])
    out = []
    for _, kind, row in events:
        if kind == 'f':
            r = est.on_wheel(row[1], True, row[2])
        elif kind == 'r':
            r = est.on_wheel(row[1], False, row[2])
        elif kind == 'c':
            r = est.on_cmd(row[1], int(row[2]))
        else:
            r = est.on_gnss(row[1], row[2], row[3], row[4])
        if r is not None and (not out or r[0] >= out[-1][0]):
            out.append(r)
    return np.array(out) if out else np.zeros((0, 6))


def reference(d):
    fix = d[GNSS_FIX['master']]
    enu = Enu(*fix[0, 2:5])
    p = np.array([enu.forward(la, lo, al) for la, lo, al in fix[:, 2:5]])
    t = fix[:, 1]
    keep = [0]
    for i in range(1, len(p)):
        if math.hypot(*(p[i, :2] - p[keep[-1], :2])) / max(t[i] - t[keep[-1]], 1e-3) < 25.0:
            keep.append(i)
    t, p = t[keep], p[keep]
    vel = d.get(GNSS_VEL['master'])
    if vel is not None and len(vel) > 10:
        tv, sp = vel[:, 1], np.hypot(vel[:, 2], vel[:, 3])
    else:  # скорость по координатам: центральная разность через ±0,5 с
        lo = np.searchsorted(t, t - 0.5)
        hi = np.clip(np.searchsorted(t, t + 0.5), 0, len(t) - 1)
        dt = np.maximum(t[hi] - t[lo], 1e-3)
        tv, sp = t, np.hypot(*(p[hi, :2] - p[lo, :2]).T) / dt
    return t, p, tv, sp


def match(t_ref, t_out):
    j = np.clip(np.searchsorted(t_out, t_ref), 1, len(t_out) - 1)
    j = np.where(np.abs(t_out[j - 1] - t_ref) < np.abs(t_out[j] - t_ref), j - 1, j)
    ok = np.abs(t_out[j] - t_ref) <= MATCH_TOL
    return j, ok


def evaluate_bag(args):
    bag_id, name, per_vehicle = args
    d = load_cached(bag_id)
    if GNSS_FIX['master'] not in d or len(d[GNSS_FIX['master']]) < 50:
        return None
    t_ref, p_ref, t_vref, v_ref = reference(d)
    if t_ref[-1] - t_ref[0] < MIN_DURATION_S:
        return None
    out = replay(make_estimator(name, bag_id, per_vehicle), d)
    row = {'bag': bag_id, 'duration_s': round(t_ref[-1] - t_ref[0], 1), 'outputs': len(out)}
    if len(out) < 10:
        return row
    t_out = out[:, 0]
    row['rate_hz'] = round(len(out) / (t_out[-1] - t_out[0]), 1)

    j, ok = match(t_vref, t_out)
    ev = out[j[ok], 1] - v_ref[ok]
    row.update(v_matched=round(ok.mean(), 3), v_rmse=round(float(np.sqrt(np.mean(ev ** 2))), 3),
               v_mae=round(float(np.mean(np.abs(ev))), 3), v_bias=round(float(np.mean(ev)), 3))

    j, ok = match(t_ref, t_out)
    ep = out[j[ok], 2:5] - p_ref[ok]
    e3 = np.linalg.norm(ep, axis=1)
    # вдольпутевая составляющая: проекция ошибки на направление движения по эталону
    tang = np.gradient(p_ref[:, :2], axis=0)[ok]
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    along = np.sum(ep[:, :2] * tang, axis=1)
    dist = float(np.sum(np.hypot(*np.diff(p_ref[:, :2], axis=0).T)))
    row.update(p_matched=round(ok.mean(), 3), dist_m=round(dist, 1),
               p3d_mean=round(float(e3.mean()), 2), p3d_rmse=round(float(np.sqrt(np.mean(e3 ** 2))), 2),
               p3d_max=round(float(e3.max()), 2), p3d_final=round(float(e3[-1]), 2),
               drift_pct=round(float(e3[-1] / max(dist, 1.0) * 100), 3),
               along_rmse=round(float(np.sqrt(np.mean(along ** 2))), 2),
               along_max=round(float(np.abs(along).max()), 2),
               z_rmse=round(float(np.sqrt(np.mean(ep[:, 2] ** 2))), 2))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--estimator', default='baseline')
    ap.add_argument('--per-vehicle', action='store_true', help='коэффициент колеса по номеру трамвая')
    ap.add_argument('--bags', type=int, default=0, help='только первые N прогонов (для отладки)')
    ap.add_argument('--tag', default='', help='суффикс имени файла результатов')
    a = ap.parse_args()
    ids = bag_ids()[:a.bags] if a.bags else bag_ids()
    with ProcessPoolExecutor(10) as ex:
        rows = [r for r in ex.map(evaluate_bag, [(b, a.estimator, a.per_vehicle) for b in ids]) if r]
    OUT.mkdir(exist_ok=True)
    name = a.estimator + ('_pv' if a.per_vehicle else '') + (f'_{a.tag}' if a.tag else '')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / f'eval_{name}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    good = [r for r in rows if 'v_rmse' in r]
    print(f'{name}: оценено {len(good)} прогонов из {len(ids)}')
    for key in ('v_rmse', 'v_mae', 'v_bias', 'p3d_mean', 'p3d_rmse', 'p3d_max', 'p3d_final', 'drift_pct',
                'along_rmse', 'z_rmse', 'rate_hz', 'v_matched', 'p_matched'):
        vals = np.array([r[key] for r in good if key in r], dtype=float)
        print(f'  {key:11s} медиана {np.median(vals):9.3f}   среднее {vals.mean():9.3f}   худший {vals.max():9.3f}')


if __name__ == '__main__':
    main()
