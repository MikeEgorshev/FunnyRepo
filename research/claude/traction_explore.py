"""Как ускорение зависит от позиции контроллера и скорости: данные для модели тяги.

Для каждого прогона на сетке 10 Гц: v — средняя скорость тележек / k, a — производная сглаженной v
(Савицкий–Голай, окно 1,1 с), u — позиция контроллера с задержкой tau (последнее значение).
Отбрасываем подозрительное: расхождение тележек > 3%, |a| > 3 м/с², пропуски колёс.
Для задержек 0…2 с подгоняем таблицу a(u, v) и смотрим долю объяснённой дисперсии.

Запуск: python traction_explore.py  ->  out/traction_samples.npz, out/traction_table.png
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import savgol_filter

from bagio import CMD, FRONT, REAR, bag_ids, load_cached

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / 'out'
KMH_PER_MPS = {'30618': 3.5953, '30639': 3.6106}
TAUS = np.arange(0.0, 2.01, 0.2)
V_BINS = np.arange(0.0, 17.0, 1.0)


def samples(bag_id):
    d = load_cached(bag_id)
    f, r, c = d.get(FRONT), d.get(REAR), d.get(CMD)
    if f is None or r is None or c is None or len(f) < 200 or len(c) < 200:
        return None
    k = KMH_PER_MPS[bag_id.split('_')[0]]
    t = np.arange(max(f[0, 1], r[0, 1], c[0, 1]) + 2.5, min(f[-1, 1], r[-1, 1], c[-1, 1]), 0.1)
    if len(t) < 100:
        return None
    vf = np.interp(t, f[:, 1], f[:, 2]) / k
    vr = np.interp(t, r[:, 1], r[:, 2]) / k
    # пропуски колёс: ближайшее сообщение дальше 0,3 с
    gap = np.zeros(len(t), bool)
    for a in (f, r):
        idx = np.clip(np.searchsorted(a[:, 1], t), 1, len(a) - 1)
        gap |= np.minimum(np.abs(a[idx, 1] - t), np.abs(a[idx - 1, 1] - t)) > 0.3
    v = 0.5 * (vf + vr)
    vs = savgol_filter(v, 11, 2)
    acc = savgol_filter(v, 11, 2, deriv=1, delta=0.1)
    ok = ~gap & (np.abs(acc) < 3.0) & (np.abs(vf - vr) <= 0.03 * np.maximum(v, 1.0))
    order = np.argsort(c[:, 1])
    tc, uc = c[order, 1], c[order, 2]
    u = np.stack([uc[np.clip(np.searchsorted(tc, t - tau, side='right') - 1, 0, len(tc) - 1)] for tau in TAUS])
    return vs[ok], acc[ok], u[:, ok]


def table_r2(u, v, a):
    """Доля объяснённой дисперсии таблицей средних a по (u, бин скорости)."""
    vb = np.clip(np.digitize(v, V_BINS) - 1, 0, len(V_BINS) - 1)
    key = (u.astype(int) + 15) * len(V_BINS) + vb
    sums = np.bincount(key, a, 31 * len(V_BINS))
    cnts = np.bincount(key, None, 31 * len(V_BINS))
    mean = sums / np.maximum(cnts, 1)
    resid = a - mean[key]
    return 1 - resid.var() / a.var(), mean.reshape(31, len(V_BINS)), cnts.reshape(31, len(V_BINS))


def main():
    with ProcessPoolExecutor(10) as ex:
        parts = [p for p in ex.map(samples, bag_ids()) if p is not None]
    v = np.concatenate([p[0] for p in parts])
    a = np.concatenate([p[1] for p in parts])
    u = np.concatenate([p[2] for p in parts], axis=1)
    moving = v > 0.3
    print(f'выборок: {len(v)}, в движении: {moving.sum()}')
    best = None
    for i, tau in enumerate(TAUS):
        r2, _, _ = table_r2(u[i][moving], v[moving], a[moving])
        print(f'  задержка {tau:.1f} с: R² = {r2:.3f}')
        if best is None or r2 > best[0]:
            best = (r2, i)
    r2, i = best
    _, mean, cnt = table_r2(u[i][moving], v[moving], a[moving])
    print(f'лучшая задержка {TAUS[i]:.1f} с, R² {r2:.3f}')
    np.savez_compressed(OUT / 'traction_samples.npz', v=v, a=a, u=u, taus=TAUS)
    fig, ax = plt.subplots(figsize=(12, 6))
    for vb in (1, 3, 5, 8, 11, 14):
        sel = cnt[:, vb] >= 30
        ax.plot(np.arange(-15, 16)[sel], mean[sel, vb], 'o-', ms=3, label=f'v ≈ {V_BINS[vb]:.0f}–{V_BINS[vb] + 1:.0f} м/с')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('позиция контроллера')
    ax.set_ylabel('ускорение, м/с²')
    ax.set_title(f'Среднее ускорение по позиции контроллера (задержка {TAUS[i]:.1f} с, R² {r2:.2f})')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'traction_table.png', dpi=90)


if __name__ == '__main__':
    main()
