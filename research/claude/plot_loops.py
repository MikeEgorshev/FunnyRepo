"""Крупный план разворотных петель: откуда прогоны стартуют и где заканчиваются."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np

from bagio import GNSS_FIX, bag_ids, load_cached
from tram_odometry.geo import Enu

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / 'out'
ENU = Enu(55.80, 37.42, 150.0).forward
BOXES = {'west': (-2100, -1650, -200, 150), 'east': (2300, 2750, 850, 1250)}


def xy(bag_id):
    fix = load_cached(bag_id).get(GNSS_FIX['master'])
    if fix is None or len(fix) < 20:
        return bag_id, None
    return bag_id, np.array([ENU(la, lo, al)[:2] for la, lo, al in fix[:, 2:5]])


def main():
    with ProcessPoolExecutor(10) as ex:
        tracks = [t for t in ex.map(xy, bag_ids()) if t[1] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, (name, (x0, x1, y0, y1)) in zip(axes, BOXES.items()):
        for _, a in tracks:
            ax.plot(a[:, 0], a[:, 1], lw=0.5, alpha=0.4, color='tab:blue')
            ax.plot(*a[0], 'g^', ms=5)
            ax.plot(*a[-1], 'rv', ms=5)
            # стрелки направления движения
            for i in range(0, len(a) - 30, 150):
                if x0 < a[i, 0] < x1 and y0 < a[i, 1] < y1:
                    d = a[i + 30] - a[i]
                    ax.annotate('', a[i] + d, a[i], arrowprops=dict(arrowstyle='->', color='k', lw=0.6))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_title(f'{name}: ▲ старт, ▼ конец прогона')
    fig.tight_layout()
    fig.savefig(OUT / 'loops.png', dpi=100)
    print('saved')


if __name__ == '__main__':
    main()
