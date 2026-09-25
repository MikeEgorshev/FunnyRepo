"""Строит карту линии — замкнутый маршрут — по GNSS обучающих прогонов.

1. Две опорные поездки (восток -> запад и запад -> восток) стыкуются там, где почти совпадают.
2. Каждая точка карты сдвигается по нормали к медиане точек всех прогонов того же направления
   (двухпутный участок разделяется по направлению движения), высота — медиана соседей.
3. Сглаживание, равномерный шаг 1 м, запись src/tram_odometry/maps/route.csv (s_m, lat, lon, alt).

Запуск: python build_route_map.py
"""
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from bagio import GNSS_FIX, REPO, bag_ids, load_cached
from tram_odometry.geo import Enu

MAP_ORIGIN = (55.80, 37.42, 150.0)
TEMPLATE_EW = '30639_0be558e2'  # старт у восточной остановки, конец в глубине западной петли
TEMPLATE_WE = '30618_21dd3af3'  # старт в западной петле, конец у восточной остановки
STEP = 1.0
OUT = REPO / 'src' / 'tram_odometry' / 'maps' / 'route.csv'
ENU = Enu(*MAP_ORIGIN)


def enu_track(bag_id):
    """Точки GNSS master в ENU карты без скачков (скорость между точками > 25 м/с)."""
    fix = load_cached(bag_id).get(GNSS_FIX['master'])
    if fix is None or len(fix) < 20:
        return None
    pts = np.array([ENU.forward(la, lo, al) for la, lo, al in fix[:, 2:5]])
    t = fix[:, 1]
    keep = [0]
    for i in range(1, len(pts)):
        dt = max(t[i] - t[keep[-1]], 1e-3)
        if math.hypot(*(pts[i, :2] - pts[keep[-1], :2])) / dt < 25.0:
            keep.append(i)
    return pts[keep], t[keep]


def decimate(pts, min_step=2.0):
    """Оставляет точку, только если она дальше min_step от предыдущей оставленной.

    Без этого дрожание GNSS на стоянках добавляет опорной поездке метры несуществующего пути,
    и длина карты выходит завышенной (так было: +1% к дистанции).
    """
    keep = [0]
    for i in range(1, len(pts)):
        if math.hypot(*(pts[i, :2] - pts[keep[-1], :2])) >= min_step:
            keep.append(i)
    return pts[keep]


def resample(pts, step=STEP):
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts[:, :2], axis=0).T))]
    mask = np.r_[True, np.diff(d) > 1e-6]
    d, pts = d[mask], pts[mask]
    s = np.arange(0.0, d[-1], step)
    return np.column_stack([np.interp(s, d, pts[:, k]) for k in range(3)])


def closest_pair(a, b):
    dist, j = cKDTree(b[:, :2]).query(a[:, :2])
    i = int(np.argmin(dist))
    return i, int(j[i]), float(dist[i])


def headings(p, closed=False):
    if closed:
        d = np.roll(p[:, :2], -1, axis=0) - np.roll(p[:, :2], 1, axis=0)
    else:
        d = np.gradient(p[:, :2], axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def smooth_closed(v, w):
    k = np.ones(2 * w + 1) / (2 * w + 1)
    return np.convolve(np.r_[v[-w:], v, v[:w]], k, mode='valid')


def main():
    ew = resample(decimate(enu_track(TEMPLATE_EW)[0]))
    we = resample(decimate(enu_track(TEMPLATE_WE)[0]))
    n = int(30 / STEP)  # стыкуем конец к началу: в больших окнах треки пересекаются до петли
    i, j, d_west = closest_pair(ew[-n:], we[:n])
    i += len(ew) - n
    k, m, d_east = closest_pair(we[-n:], ew[:n])
    k += len(we) - n
    circuit = resample(np.vstack([ew[m:i + 1], we[j:k + 1], ew[m:m + 1]]))
    print(f'стыки опорных поездок: запад {d_west:.2f} м, восток {d_east:.2f} м; длина {len(circuit) * STEP:.0f} м')

    with ProcessPoolExecutor(10) as ex:
        tracks = [t for t in ex.map(enu_track, bag_ids()) if t is not None]
    pts, hdg = [], []
    for p, t in tracks:
        if len(p) < 3:
            continue
        v = np.r_[0.0, np.hypot(*np.diff(p[:, :2], axis=0).T) / np.maximum(np.diff(t), 1e-3)]
        moving = v > 1.0
        pts.append(p[moving])
        hdg.append(headings(p)[moving])
    pts, hdg = np.vstack(pts), np.concatenate(hdg)
    tree = cKDTree(pts[:, :2])
    print(f'точек GNSS в движении: {len(pts)}')

    for it in range(3):
        h = headings(circuit, closed=True)
        nx, ny = -np.sin(h), np.cos(h)
        shift, used = np.zeros(len(circuit)), 0
        z = circuit[:, 2].copy()
        for q, idx in enumerate(tree.query_ball_point(circuit[:, :2], r=3.0)):
            if len(idx) < 5:
                continue
            idx = np.asarray(idx)
            idx = idx[np.abs(np.angle(np.exp(1j * (hdg[idx] - h[q])))) < math.radians(30)]
            if len(idx) < 5:
                continue
            off = (pts[idx, 0] - circuit[q, 0]) * nx[q] + (pts[idx, 1] - circuit[q, 1]) * ny[q]
            shift[q] = np.median(off)
            z[q] = np.median(pts[idx, 2])
            used += 1
        circuit[:, 0] += shift * nx
        circuit[:, 1] += shift * ny
        circuit[:, 2] = z
        print(f'итерация {it + 1}: уточнено {used}/{len(circuit)} точек, медианный сдвиг {np.median(np.abs(shift)):.2f} м')

    circuit[:, 0] = smooth_closed(circuit[:, 0], 3)
    circuit[:, 1] = smooth_closed(circuit[:, 1], 3)
    circuit[:, 2] = smooth_closed(circuit[:, 2], 10)
    circuit = resample(np.vstack([circuit, circuit[:1]]))

    # невязка: расстояние от точек прогонов до карты своего направления
    ctree = cKDTree(circuit[:, :2])
    dist, _ = ctree.query(pts[::20, :2])
    print(f'расстояние точек GNSS до карты: медиана {np.median(dist):.2f} м, 95% {np.percentile(dist, 95):.2f} м')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# Карта линии: замкнутый маршрут, шаг 1 м. Строится research/claude/build_route_map.py\n')
        f.write(f'# опорные поездки: {TEMPLATE_EW}, {TEMPLATE_WE}\n')
        f.write('s_m,lat,lon,alt\n')
        for q, (x, y, zz) in enumerate(circuit):
            la, lo, al = ENU.inverse(x, y, zz)
            f.write(f'{q * STEP:.1f},{la:.8f},{lo:.8f},{al:.2f}\n')
    print(f'{OUT}: {len(circuit)} точек, длина маршрута {len(circuit) * STEP:.0f} м')


if __name__ == '__main__':
    main()
