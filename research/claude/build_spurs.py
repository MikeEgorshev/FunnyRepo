"""Отводы у конечных: пути от места старта прогона до выхода на кольцо карты.

У конечных несколько путей, а в кольце карты — один. Если прогон стартует на другом пути,
ближайшая точка кольца может оказаться на другом участке петли (ошибка — сотни метров по s).
Для каждого прогона, который стартует дальше 3 м от кольца, берём его GNSS-трек до места,
где он на 20 м подряд идёт по кольцу (ближе 2 м, курс совпадает), и сохраняем как отвод:
точки с дистанцией вдоль отвода по колёсам и s_join — точка выхода на кольцо.
Похожие отводы (старт ближе 3 м и тот же s_join ±10 м) объединяются.

Запуск: python build_spurs.py  (после build_route_map.py и calibrate_route_s.py)
        -> src/tram_odometry/maps/route_spurs.csv (spur_id, s_join, d_m, lat, lon, alt)
"""
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from bagio import FRONT, GNSS_FIX, REAR, REPO, bag_ids, load_cached
from calibrate_route_s import ENU, load_map, project

OUT = REPO / 'src' / 'tram_odometry' / 'maps' / 'route_spurs.csv'
KMH_PER_MPS = 3.5966
OFF_MAP_START = 3.0
ON_MAP = 2.0
ON_MAP_RUN = 20.0


def spur(bag_id):
    d = load_cached(bag_id)
    fix = d.get(GNSS_FIX['master'])
    if fix is None or len(fix) < 100:
        return None
    m, xy, yaw = load_map()
    tree = cKDTree(xy)
    p3 = np.array([ENU.forward(*r) for r in fix[:, 2:5]])
    t = fix[:, 1]
    start_off = tree.query(p3[0, :2])[0]
    if start_off < OFF_MAP_START:
        return None
    # первые 600 м трека, прореженного через 1 м, с временем каждой точки
    keep = [0]
    for i in range(1, len(p3)):
        if math.hypot(*(p3[i, :2] - p3[keep[-1], :2])) >= 1.0:
            keep.append(i)
    keep = np.array(keep)
    arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p3[keep, :2], axis=0).T))]
    keep = keep[arc < 600.0]
    pts, tk = p3[keep], t[keep]
    if len(pts) < 30:
        return None
    heading = np.arctan2(*(np.gradient(pts[:, :2], axis=0)[:, ::-1].T))
    s = project(m, xy, yaw, tree, pts[:, :2], heading)
    dist_map = tree.query(pts[:, :2])[0]
    on = (dist_map < ON_MAP) & ~np.isnan(s)
    run = 0
    join = None
    for i in range(len(pts)):
        run = run + 1 if on[i] else 0
        if run >= ON_MAP_RUN:
            join = i - run + 1
            break
    if join is None or join < 3:
        return None
    # дистанция вдоль отвода — по колёсам (как её будет интегрировать оценщик)
    wheels = [a for a in (d.get(FRONT), d.get(REAR)) if a is not None and len(a)]
    w = np.concatenate(wheels)
    w = w[np.argsort(w[:, 1])]
    cum = np.r_[0.0, np.cumsum(0.5 * (w[1:, 2] + w[:-1, 2]) / KMH_PER_MPS * np.diff(w[:, 1]))]
    dk = np.interp(tk[:join + 1], w[:, 1], cum)
    dk -= dk[0]
    geo = np.array([ENU.inverse(*q) for q in pts[:join + 1]])
    return {'bag': bag_id, 'start': pts[0, :2], 's_join': float(s[join]), 'd': dk, 'geo': geo,
            'start_off': float(start_off)}


def main():
    with ProcessPoolExecutor(10) as ex:
        spurs = [sp for sp in ex.map(spur, bag_ids()) if sp is not None]
    spurs.sort(key=lambda sp: -sp['d'][-1])
    uniq = []
    for sp in spurs:
        if any(np.hypot(*(sp['start'] - u['start'])) < 3.0 and abs(sp['s_join'] - u['s_join']) < 10.0 for u in uniq):
            continue
        uniq.append(sp)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# Отводы у конечных: от места старта до выхода на кольцо. Строится research/claude/build_spurs.py\n')
        f.write('spur_id,s_join,d_m,lat,lon,alt\n')
        for k, sp in enumerate(uniq):
            for dd, (la, lo, al) in zip(sp['d'], sp['geo']):
                f.write(f'{k},{sp["s_join"]:.2f},{dd:.2f},{la:.8f},{lo:.8f},{al:.2f}\n')
    for k, sp in enumerate(uniq):
        print(f'отвод {k}: {sp["bag"]} старт ({sp["start"][0]:.0f},{sp["start"][1]:.0f}) в {sp["start_off"]:.1f} м от кольца, '
              f'длина {sp["d"][-1]:.0f} м, выход на кольцо s={sp["s_join"]:.0f}')
    print(f'{len(uniq)} отводов из {len(spurs)} прогонов со стартом вне кольца -> {OUT}')


if __name__ == '__main__':
    main()
