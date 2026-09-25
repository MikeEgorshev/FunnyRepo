"""Диагностика одного прогона: ошибка положения во времени, s оценщика против s эталона на карте.

Запуск: python diagnose.py <bag_id>  ->  out/diag_<bag_id>.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial import cKDTree

from evaluate import MAP, make_estimator, match, reference, replay
from bagio import load_cached

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / 'out'


def main(bag_id):
    d = load_cached(bag_id)
    est = make_estimator('baseline', bag_id, per_vehicle=False)
    out = replay(est, d)
    t_ref, p_ref, _, _ = reference(d)
    j, ok = match(t_ref, out[:, 0])
    m = est.map
    mx, my = np.array(m.x), np.array(m.y)
    # s эталона: ближайшая точка карты с учётом направления движения
    tang = np.gradient(p_ref[:, :2], axis=0)
    yaw_ref = np.arctan2(tang[:, 1], tang[:, 0])
    myaw = np.arctan2(np.gradient(my), np.gradient(mx))
    tree = cKDTree(np.column_stack([mx, my]))
    s_ref = np.full(len(p_ref), np.nan)
    for q in np.flatnonzero(ok):
        idx = np.array(tree.query_ball_point(p_ref[q, :2], r=15.0))
        if len(idx) == 0:
            continue
        good = idx[np.abs(np.angle(np.exp(1j * (myaw[idx] - yaw_ref[q])))) < 1.0]
        if len(good) == 0:
            good = idx
        k = good[np.argmin(np.hypot(mx[good] - p_ref[q, 0], my[good] - p_ref[q, 1]))]
        s_ref[q] = m.s[k]
    s_est = out[j, 5]
    ds = (s_est - s_ref + m.length / 2) % m.length - m.length / 2
    e3 = np.linalg.norm(out[j, 2:5] - p_ref, axis=1)
    tt = t_ref - t_ref[0]
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(tt[ok], e3[ok])
    ax[0].set_ylabel('3D ошибка, м')
    ax[1].plot(tt, ds)
    ax[1].set_ylabel('s_est − s_ref, м')
    ax[2].plot(tt, s_ref, '.', ms=1, label='s эталона')
    ax[2].plot(tt, s_est, '.', ms=1, label='s оценщика')
    ax[2].legend()
    ax[2].set_ylabel('s, м')
    ax[2].set_xlabel('время, с')
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(bag_id)
    fig.tight_layout()
    fig.savefig(OUT / f'diag_{bag_id}.png', dpi=90)
    print(f'длина карты {m.length:.0f} м; s старт est/ref: {s_est[0]:.1f}/{s_ref[np.flatnonzero(~np.isnan(s_ref))[0]]:.1f}; '
          f'ds: старт {ds[np.isfinite(ds)][0]:.1f}, конец {ds[np.isfinite(ds)][-1]:.1f}, '
          f'min {np.nanmin(ds):.1f}, max {np.nanmax(ds):.1f}')


if __name__ == '__main__':
    main(sys.argv[1])
