"""Где начинается и кончается каждый прогон и сколько метров он проходит по GNSS."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bagio import GNSS_FIX, bag_ids, load_cached
from tram_odometry.geo import Enu

ENU = Enu(55.80, 37.42, 150.0).forward


def overview(bag_id):
    fix = load_cached(bag_id).get(GNSS_FIX['master'])
    if fix is None or len(fix) < 20:
        return bag_id, None
    xy = np.array([ENU(la, lo, al)[:2] for la, lo, al in fix[:, 2:5]])
    step = np.hypot(*np.diff(xy, axis=0).T)
    good = step < 5.0  # отбрасываем скачки GNSS
    return bag_id, (xy[0], xy[-1], step[good].sum(), fix[-1, 0] - fix[0, 0])


def main():
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(overview, bag_ids()))
    for bag_id, r in sorted(res, key=lambda kv: -(kv[1][2] if kv[1] else 0)):
        if r is None:
            print(f'{bag_id}: нет GNSS')
            continue
        (x0, y0), (x1, y1), length, dur = r
        print(f'{bag_id}  start ({x0:7.0f},{y0:6.0f})  end ({x1:7.0f},{y1:6.0f})  path {length:7.0f} m  {dur / 60:5.1f} min')


if __name__ == '__main__':
    main()
