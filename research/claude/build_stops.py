"""Места регулярных стоянок на маршруте: остановки и стоп-линии.

По обучающим прогонам находим стоянки (скорость колёс < 0,05 м/с дольше 3 с), берём медиану
GNSS за стоянку, проецируем на карту с курсом по движению перед остановкой и получаем s стоянки.
Стоянки кластеризуются по s (разрыв > 15 м — новый кластер). Кластер считается опорной точкой,
если в нём не меньше MIN_COUNT стоянок и разброс (1,4826·MAD) не больше MAX_SPREAD м.

Запуск: python build_stops.py  (после карты)  ->  <папка карты>/route_stops.csv (s_m, sigma_m, count)
"""
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from bagio import FRONT, GNSS_FIX, MAP_DIR, REAR, bag_ids, load_cached
from calibrate_route_s import ENU, load_map, project

OUT = MAP_DIR / 'route_stops.csv'
KMH_PER_MPS = 3.5966
STOP_V = 0.05
STOP_MIN_S = 3.0
GAP_M = 15.0
MIN_COUNT = 8
MAX_SPREAD = 4.0


def stops(bag_id):
    d = load_cached(bag_id)
    fix, front, rear = d.get(GNSS_FIX['master']), d.get(FRONT), d.get(REAR)
    if fix is None or front is None or rear is None or len(fix) < 100:
        return []
    m, xy, yaw = load_map()
    tree = cKDTree(xy)
    t = front[:, 1]
    v = 0.5 * (front[:, 2] + np.interp(t, rear[:, 1], rear[:, 2])) / KMH_PER_MPS
    still = v < STOP_V
    out = []
    i = 0
    p_all = None
    while i < len(t):
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(t) and still[j + 1]:
            j += 1
        if t[j] - t[i] >= STOP_MIN_S and i > 0:
            if p_all is None:
                p_all = np.array([ENU.forward(*r) for r in fix[:, 2:5]])[:, :2]
            sel = (fix[:, 1] >= t[i]) & (fix[:, 1] <= t[j])
            before = (fix[:, 1] >= t[i] - 15.0) & (fix[:, 1] < t[i] - 2.0)
            if sel.sum() >= 5 and before.sum() >= 5:
                p = np.median(p_all[sel], axis=0)
                pb = p_all[before]
                heading = math.atan2(pb[-1, 1] - pb[0, 1], pb[-1, 0] - pb[0, 0])
                if math.hypot(*(pb[-1] - pb[0])) > 3.0:
                    s = project(m, xy, yaw, tree, p[None, :], np.array([heading]))[0]
                    if not np.isnan(s):
                        out.append(s)
        i = j + 1
    return out


def main():
    m, _, _ = load_map()
    with ProcessPoolExecutor(10) as ex:
        s_all = np.sort(np.concatenate([np.asarray(s) for s in ex.map(stops, bag_ids()) if len(s)]))
    clusters, cur = [], [s_all[0]]
    for s in s_all[1:]:
        if s - cur[-1] > GAP_M:
            clusters.append(np.array(cur))
            cur = []
        cur.append(s)
    clusters.append(np.array(cur))
    good = []
    for c in clusters:
        med = float(np.median(c))
        spread = float(1.4826 * np.median(np.abs(c - med)))
        if len(c) >= MIN_COUNT and spread <= MAX_SPREAD:
            good.append((med, max(spread, 0.5), len(c)))
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# Места регулярных стоянок: s на карте, разброс, число стоянок. research/claude/build_stops.py\n')
        f.write('s_m,sigma_m,count\n')
        for med, spread, n in good:
            f.write(f'{med:.2f},{spread:.2f},{n}\n')
    print(f'стоянок: {len(s_all)}, кластеров: {len(clusters)}, опорных точек: {len(good)} '
          f'(медианный разброс {np.median([g[1] for g in good]):.2f} м) -> {OUT}')


if __name__ == '__main__':
    main()
