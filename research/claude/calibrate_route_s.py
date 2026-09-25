"""Калибровка s карты под пройденный путь: на каждом участке ~20 м s карты сравнивается
с интегралом скорости GNSS по всем прогонам, и s пересчитывается так, чтобы совпадать с путём.

Геометрическая длина ломаной по GNSS-точкам и путь по скорости расходятся (шум координат,
срезанные или лишние участки, стыки опорных поездок), а оценщик интегрирует именно скорость.
Результат перезаписывает src/tram_odometry/maps/route.csv: s_m становится одометрической дистанцией.

Запуск: python calibrate_route_s.py  (после build_route_map.py)
"""
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from bagio import GNSS_FIX, GNSS_VEL, MAP_DIR, bag_ids, load_cached
from tram_odometry.geo import Enu
from tram_odometry.route_map import RouteMap

MAP = MAP_DIR / 'route.csv'
ENU = Enu(55.80, 37.42, 150.0)
BIN_M = 20.0
MAX_OFF_MAP = 3.0


def load_map():
    m = RouteMap.load(MAP).to_frame(ENU)
    xy = np.column_stack([m.x, m.y])
    yaw = np.arctan2(np.roll(xy[:, 1], -1) - xy[:, 1], np.roll(xy[:, 0], -1) - xy[:, 0])
    return m, xy, yaw


def project(m, xy, yaw, tree, p, heading):
    """s на карте для точек p с учётом курса; nan, если дальше MAX_OFF_MAP от карты."""
    s = np.full(len(p), np.nan)
    n = len(xy)
    for q, idx in enumerate(tree.query_ball_point(p, r=10.0)):
        if not idx:
            continue
        idx = np.asarray(idx)
        good = idx[np.abs(np.angle(np.exp(1j * (yaw[idx] - heading[q])))) < math.radians(60)]
        if not len(good):
            continue
        i = good[np.argmin(np.hypot(*(xy[good] - p[q]).T))]
        j = (i + 1) % n
        seg = xy[j] - xy[i]
        t = np.clip(np.dot(p[q] - xy[i], seg) / max(np.dot(seg, seg), 1e-9), 0, 1)
        if np.hypot(*(xy[i] + t * seg - p[q])) > MAX_OFF_MAP:
            continue
        s_next = m.s[j] if j else m.length
        s[q] = m.s[i] + t * (s_next - m.s[i])
    return s


def pairs(bag_id):
    """Для прогона: (середина участка по s, приращение s карты, путь по скорости GNSS)."""
    d = load_cached(bag_id)
    fix, vel = d.get(GNSS_FIX['master']), d.get(GNSS_VEL['master'])
    if fix is None or vel is None or len(fix) < 100 or len(vel) < 100:
        return None
    m, xy, yaw = load_map()
    tree = cKDTree(xy)
    t = fix[::10, 1]  # примерно раз в секунду
    p = np.array([ENU.forward(*r) for r in fix[::10, 2:5]])[:, :2]
    sp = np.hypot(vel[:, 2], vel[:, 3])
    cum = np.r_[0.0, np.cumsum(0.5 * (sp[1:] + sp[:-1]) * np.diff(vel[:, 1]))]
    dist = np.interp(t, vel[:, 1], cum)
    heading = np.arctan2(*(np.gradient(p, axis=0)[:, ::-1].T))
    s = project(m, xy, yaw, tree, p, heading)
    out = []
    L = m.length
    for k in range(len(t) - 1):
        dd = dist[k + 1] - dist[k]
        if np.isnan(s[k]) or np.isnan(s[k + 1]) or dd < 0.5 or t[k + 1] - t[k] > 1.5:
            continue
        ds = (s[k + 1] - s[k] + L / 2) % L - L / 2
        if not 0.3 * dd < ds < 3.0 * dd:  # скачок GNSS или смена пути
            continue
        out.append(((s[k] + ds / 2) % L, ds, dd))
    return np.array(out) if out else None


def main():
    m, xy, _ = load_map()
    with ProcessPoolExecutor(10) as ex:
        chunks = [c for c in ex.map(pairs, bag_ids()) if c is not None]
    allp = np.vstack(chunks)
    nb = int(math.ceil(m.length / BIN_M))
    b = np.minimum((allp[:, 0] // BIN_M).astype(int), nb - 1)
    ds_sum = np.bincount(b, allp[:, 1], nb)
    dd_sum = np.bincount(b, allp[:, 2], nb)
    ratio = np.where(dd_sum > 50.0, ds_sum / np.maximum(dd_sum, 1e-9), np.nan)
    # пропуски — соседними значениями, затем медианное сглаживание по 3 участкам
    idx = np.arange(nb)
    good = ~np.isnan(ratio)
    ratio = np.interp(idx, idx[good], ratio[good], period=nb)
    ratio = np.array([np.median(np.take(ratio, [i - 1, i, i + 1], mode='wrap')) for i in idx])
    ratio = np.clip(ratio, 0.8, 1.25)
    print(f'пар: {len(allp)}; отношение s карты к пути: медиана {np.median(ratio):.4f}, '
          f'min {ratio.min():.3f}, max {ratio.max():.3f}; участков с данными {good.sum()}/{nb}')

    s_geo = np.array(m.s)
    step = np.diff(np.r_[s_geo, m.length])
    s_cal = np.r_[0.0, np.cumsum(step / ratio[np.minimum((s_geo // BIN_M).astype(int), nb - 1)])[:-1]]
    print(f'длина маршрута: геометрическая {m.length:.1f} м, откалиброванная {s_cal[-1] + step[-1] / ratio[-1]:.1f} м')
    header = [line for line in MAP.read_text(encoding='utf-8').splitlines() if line.startswith('#')]
    with open(MAP, 'w', encoding='utf-8', newline='\n') as f:
        for line in header:
            if 'откалибровано' not in line:
                f.write(line + '\n')
        f.write('# s_m откалибровано под путь по скорости GNSS: research/claude/calibrate_route_s.py\n')
        f.write('s_m,lat,lon,alt\n')
        for q in range(len(s_geo)):
            f.write(f'{s_cal[q]:.2f},{m.lat[q]:.8f},{m.lon[q]:.8f},{m.alt[q]:.2f}\n')


if __name__ == '__main__':
    main()
