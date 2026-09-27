"""Разбор плохих прогонов EKF: откуда берётся ошибка положения.

Для каждого прогона: где выбран старт (кольцо или отвод, расстояние до пути), ошибка во времени
(3D и вдоль пути), привязки к стоянкам (когда и на сколько сдвинули s), и картинка: карта и отводы
вокруг старта, эталон и оценка.

Карта — из TRAM_MAP_DIR (для честного разбора — карта той половины кросс-валидации, где прогон
был проверочным: dataset/cv/fold<k>).

Запуск: TRAM_MAP_DIR=.../cv/fold1 python worst.py 30639_9c362687 [...]  ->  out/worst_<bag>_<карта>.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

from bagio import MAP_DIR, load_cached
from evaluate import EkfAdapter, match, reference, replay

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / 'out'


def main(bag_id):
    d = load_cached(bag_id)
    ad = EkfAdapter(bag_id.split('_')[0])
    est = ad.est
    snaps = []
    orig_snap = est._snap

    def logged_snap():
        s0 = est.s
        orig_snap()
        if est.s != s0:
            snaps.append((est.t, s0, est.s - s0))
    est._snap = logged_snap
    starts = []
    orig_locate = est.map.locate_start

    def logged_locate(x, y, yaw=None, **kw):
        s, dist = orig_locate(x, y, yaw, **kw)
        starts.append((x, y, yaw, s, dist, est.map.active_spur is not None))
        return s, dist
    est.map.locate_start = logged_locate

    out = replay(ad, d)
    t_ref, p_ref, _, _ = reference(d)
    j, ok = match(t_ref, out[:, 0])
    tt, pr, po = t_ref[ok], p_ref[ok], out[j[ok], 2:5]
    e3 = np.linalg.norm(po - pr, axis=1)
    tang = np.gradient(pr[:, :2], axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    along = np.sum((po - pr)[:, :2] * tang, axis=1)
    cross = (po - pr)[:, 0] * -tang[:, 1] + (po - pr)[:, 1] * tang[:, 0]
    t0 = tt[0]

    print(f'== {bag_id}: карта {MAP_DIR.name}, {len(tt)} точек, 3D средн. {e3.mean():.1f} м, макс {e3.max():.1f} м')
    x, y, yaw, s, dist, on_spur = starts[-1]
    print(f'   старт: s={s:.1f}, до пути {dist:.1f} м, {"отвод" if on_spur else "кольцо"}, '
          f'курс {"нет (стоит)" if yaw is None else f"{np.degrees(yaw):.0f}°"}, точек выставки {len(starts)}')
    for q in (0, 30, 60, 120, 300):
        m = tt - t0 <= q if q == 0 else (tt - t0 > q - 30) & (tt - t0 <= q)
        if m.any():
            print(f'   до {q:4d} с: 3D {e3[m].mean():6.1f}  вдоль {along[m].mean():+7.1f}  поперёк {cross[m].mean():+6.1f}')
    for ts, s0, ds in snaps:
        k = min(np.searchsorted(tt, ts), len(tt) - 1)
        print(f'   привязка t={ts - t0:7.1f} с  s={s0:8.1f}  сдвиг {ds:+6.1f} м   (ошибка вдоль до неё {along[max(k - 1, 0)]:+6.1f})')
    worst = np.argsort(-e3)[:1]
    for k in worst:
        print(f'   худшая точка t={tt[k] - t0:.1f} с: 3D {e3[k]:.1f}, вдоль {along[k]:+.1f}, поперёк {cross[k]:+.1f}')

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(tt - t0, e3, label='3D')
    ax[0].plot(tt - t0, along, label='вдоль')
    ax[0].plot(tt - t0, cross, label='поперёк', lw=0.8)
    for ts, _, ds in snaps:
        ax[0].axvline(ts - t0, color='k', alpha=0.2)
    ax[0].set_xlabel('с от начала'), ax[0].set_ylabel('м'), ax[0].legend(), ax[0].grid(alpha=0.3)
    ax[0].set_title(f'{bag_id}: ошибка (вертикали — привязки к стоянкам)')
    m = est.map
    near = 400.0
    cx, cy = pr[0, 0], pr[0, 1]
    mx, my = np.array(m.x), np.array(m.y)
    sel = np.hypot(mx - cx, my - cy) < near
    ax[1].plot(mx[sel], my[sel], '.', ms=1, color='0.6', label='кольцо')
    for sp in m.spurs:
        ax[1].plot(sp['x'], sp['y'], '-', color='tab:orange', lw=1)
    first = (tt - t0) < 180
    ax[1].plot(pr[first, 0], pr[first, 1], 'g-', lw=2, label='эталон, 3 мин')
    ax[1].plot(po[first, 0], po[first, 1], 'r-', lw=1, label='оценка, 3 мин')
    ax[1].plot(cx, cy, 'go')
    ax[1].set_xlim(cx - near, cx + near), ax[1].set_ylim(cy - near, cy + near), ax[1].set_aspect('equal')
    ax[1].legend(), ax[1].grid(alpha=0.3), ax[1].set_title('старт: карта (серым), отводы (оранжевым)')
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f'worst_{bag_id}_{MAP_DIR.name}.png', dpi=80)
    plt.close(fig)


if __name__ == '__main__':
    for b in sys.argv[1:]:
        main(b)
