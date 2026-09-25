"""Все треки GNSS master на одной картинке, в метрах от общей точки.

Запуск: python plot_tracks.py  ->  out/all_tracks.png (заодно заполняет кэш прогонов)
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np

from bagio import GNSS_FIX, bag_ids, load_cached

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / 'out'
LAT0, LON0 = 55.80, 37.42
KX = 111320 * np.cos(np.radians(LAT0))
KY = 110540


def track(bag_id):
    a = load_cached(bag_id).get(GNSS_FIX['master'])
    return bag_id, (None if a is None else a[:, 2:4])


def main():
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(10) as ex:
        tracks = list(ex.map(track, bag_ids()))
    fig, ax = plt.subplots(figsize=(14, 6))
    for bag_id, a in tracks:
        if a is None or not len(a):
            continue
        ax.plot((a[:, 1] - LON0) * KX, (a[:, 0] - LAT0) * KY, lw=0.4, alpha=0.5,
                color='tab:blue' if bag_id.startswith('30618') else 'tab:red')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('x, м (восток)')
    ax.set_ylabel('y, м (север)')
    ax.set_title('GNSS master fix, все прогоны (синие — 30618, красные — 30639)')
    fig.tight_layout()
    fig.savefig(OUT / 'all_tracks.png', dpi=110)
    print('saved', OUT / 'all_tracks.png')


if __name__ == '__main__':
    main()
