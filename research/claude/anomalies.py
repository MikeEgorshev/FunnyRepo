"""Настоящие аномалии колёс в датасете: эпизоды, где тележка расходится со скоростью GNSS.

Для каждой тележки: скорость (км/ч / k) против скорости GNSS master/vel в ту же метку.
Эпизод — подряд идущие показания с |ошибкой| > max(0.5 м/с, 10 %) дольше 0.3 с. Для эпизода:
длительность, пик ошибки, знак, режим контроллера (тяга/выбег/тормоз), расходится ли вторая тележка.
Ещё считаем зависания: одинаковые ненулевые показания подряд.

Нужно, чтобы синтетические помехи стресс-теста были похожи на настоящие.

Запуск: python anomalies.py  ->  out/anomalies.csv и сводка в консоль
"""
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bagio import CMD, FRONT, GNSS_VEL, REAR, bag_ids, load_cached
from baseline import KMH_PER_MPS

OUT = Path(__file__).parent / 'out'
ERR_ABS, ERR_REL, MIN_DUR = 0.5, 0.10, 0.3


def episodes(t, bad):
    """Отрезки подряд идущих True: [(i0, i1)] включительно."""
    out, i0 = [], None
    for i, b in enumerate(bad):
        if b and i0 is None:
            i0 = i
        elif not b and i0 is not None:
            out.append((i0, i - 1))
            i0 = None
    if i0 is not None:
        out.append((i0, len(bad) - 1))
    return [(a, b) for a, b in out if t[b] - t[a] >= MIN_DUR]


def max_run(values):
    best, run = 1, 1
    for a, b in zip(values[:-1], values[1:]):
        run = run + 1 if a == b and a > 1.0 else 1
        best = max(best, run)
    return best


def analyse(bag_id):
    d = load_cached(bag_id)
    vel = d.get(GNSS_VEL['master'])
    if vel is None or len(vel) < 100 or CMD not in d:
        return []
    k = KMH_PER_MPS.get(bag_id.split('_')[0], KMH_PER_MPS['default'])
    tv, sp = vel[:, 1], np.hypot(vel[:, 2], vel[:, 3])
    cmd = d[CMD]
    rows = []
    wheels = {name: d[topic] for name, topic in (('front', FRONT), ('rear', REAR)) if topic in d}
    for name, w in wheels.items():
        t, z = w[:, 1], w[:, 2] / k
        inside = (t > tv[0]) & (t < tv[-1])
        g = np.interp(t, tv, sp)
        # GNSS/vel с пропусками: не сравниваем там, где ближайшая точка дальше 0.2 с
        near = np.abs(tv[np.clip(np.searchsorted(tv, t), 0, len(tv) - 1)] - t) < 0.2
        err = z - g
        bad = inside & near & (np.abs(err) > np.maximum(ERR_ABS, ERR_REL * g))
        other = wheels.get('rear' if name == 'front' else 'front')
        for a, b in episodes(t, bad):
            seg = slice(a, b + 1)
            e = err[seg]
            u = cmd[np.clip(np.searchsorted(cmd[:, 1], t[seg]) - 1, 0, len(cmd) - 1), 2]
            mode = 'traction' if np.median(u) > 0 else ('brake' if np.median(u) < 0 else 'coast')
            both = ''
            if other is not None:
                zo = np.interp(t[seg], other[:, 1], other[:, 2] / k)
                both = int(np.median(np.abs(zo - g[seg])) > np.maximum(ERR_ABS, ERR_REL * np.median(g[seg])))
            rows.append({'bag': bag_id, 'bogie': name, 't0': round(t[a] - tv[0], 1),
                         'dur_s': round(t[b] - t[a], 2), 'peak_mps': round(float(e[np.argmax(np.abs(e))]), 2),
                         'v_gnss': round(float(np.median(g[seg])), 2), 'mode': mode, 'both': both})
        rows.append({'bag': bag_id, 'bogie': name, 't0': '', 'dur_s': '', 'peak_mps': '', 'v_gnss': '',
                     'mode': 'max_repeat', 'both': max_run(w[:, 2])})
    return rows


def main():
    with ProcessPoolExecutor(10) as ex:
        rows = [r for rr in ex.map(analyse, bag_ids()) for r in rr]
    OUT.mkdir(exist_ok=True)
    with open(OUT / 'anomalies.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ep = [r for r in rows if r['mode'] != 'max_repeat']
    print(f'эпизодов расхождения тележки с GNSS: {len(ep)} в {len({r["bag"] for r in ep})} прогонах')
    for mode in ('traction', 'coast', 'brake'):
        m = [r for r in ep if r['mode'] == mode]
        if not m:
            continue
        dur = np.array([r['dur_s'] for r in m])
        pk = np.array([r['peak_mps'] for r in m])
        both = np.mean([r['both'] == 1 for r in m])
        print(f'  {mode:8s} {len(m):4d}: длит. медиана {np.median(dur):5.1f} с, 90% {np.percentile(dur, 90):5.1f} с, '
              f'макс {dur.max():5.1f};  пик: >0 {np.mean(pk > 0):.0%}, |пик| медиана {np.median(np.abs(pk)):.2f} м/с, '
              f'макс {np.abs(pk).max():.1f};  обе тележки {both:.0%}')
    rep = np.array([r['both'] for r in rows if r['mode'] == 'max_repeat'])
    print(f'зависание (одинаковые ненулевые подряд), макс по тележкам: медиана {np.median(rep):.0f}, '
          f'макс {rep.max()}, >=6 у {np.sum(rep >= 6)} из {len(rep)}')


if __name__ == '__main__':
    main()
