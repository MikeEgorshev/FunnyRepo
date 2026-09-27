"""Проверочный прогон организаторов (check-code-with-bag.zip, 27.09): наш оценщик против эталона судьи.

В прогоне есть /localization/kinematic_state — эталон, с которым сравнивает metrics.py организаторов.
Сопоставление как у них: пара «эталон — выход» по header.stamp с допуском 0,05 с (у них
ApproximateTimeSynchronizer), RMSE и максимум модуля ошибки: скорость (twist.linear.x), x, y, z и 3D.

Запуск: python check_bag.py [путь к папке bag]  ->  сводка в консоль и out/check_<bag>.png
"""
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
from rosbags.highlevel import AnyReader

from bagio import CMD, DATASET, FRONT, GNSS_FIX, REAR, typestore
from evaluate import EkfAdapter, replay

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / 'out'
REF = '/localization/kinematic_state'
TOL = 0.05


def load(path):
    out = {}
    with AnyReader([path], default_typestore=typestore()) as reader:
        for conn, t, raw in reader.messages(connections=reader.connections):
            m = reader.deserialize(raw, conn.msgtype)
            row = [t * 1e-9, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9]
            if conn.topic in (FRONT, REAR):
                row.append(m.velocity)
            elif conn.topic == CMD:
                row.append(m.position)
            elif conn.topic.endswith('/fix'):
                row += [m.latitude, m.longitude, m.altitude, m.status.status]
            elif conn.topic == REF:
                p, v = m.pose.pose.position, m.twist.twist.linear
                row += [p.x, p.y, p.z, v.x, v.y]
            else:
                continue
            out.setdefault(conn.topic, []).append(row)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def main(path):
    os.environ.setdefault('TRAM_GNSS', 'all')  # GNSS проверочного прогона — как есть, он уже пачками
    d = load(path)
    bag = path.name
    ref = d[REF]
    fix = d.get(GNSS_FIX['master'])
    print(f'{bag}: эталон {len(ref)} сообщений, {ref[-1, 1] - ref[0, 1]:.0f} с; '
          f'GNSS master {0 if fix is None else len(fix)} точек')
    if fix is not None:
        gaps = np.diff(fix[:, 1])
        print(f'  GNSS: с {fix[0, 1] - ref[0, 1]:.1f} по {fix[-1, 1] - ref[0, 1]:.1f} с от начала эталона, '
              f'разрывы > 1 с: {[round(float(g), 1) for g in gaps[gaps > 1]][:10]}')
    out = replay(EkfAdapter(bag.split('_')[0]), d)
    t_out = out[:, 0]
    j = np.clip(np.searchsorted(t_out, ref[:, 1]), 1, len(t_out) - 1)
    j = np.where(np.abs(t_out[j - 1] - ref[:, 1]) < np.abs(t_out[j] - ref[:, 1]), j - 1, j)
    ok = np.abs(t_out[j] - ref[:, 1]) <= TOL
    ev = out[j[ok], 1] - ref[ok, 5]
    ep = out[j[ok], 2:5] - ref[ok, 2:5]
    e3 = np.linalg.norm(ep, axis=1)
    print(f'  пар: {ok.sum()} из {len(ref)} ({ok.mean():.1%}); выходов {len(out)}, {len(out) / (t_out[-1] - t_out[0]):.1f} Гц')
    print(f'  скорость: RMSE {np.sqrt(np.mean(ev ** 2)):.3f} м/с, max {np.abs(ev).max():.2f}, смещение {ev.mean():+.3f}')
    for k, name in enumerate('xyz'):
        print(f'  {name}: RMSE {np.sqrt(np.mean(ep[:, k] ** 2)):7.2f} м, max {np.abs(ep[:, k]).max():7.2f}, '
              f'среднее {ep[:, k].mean():+7.2f}')
    print(f'  3D: RMSE {np.sqrt(np.mean(e3 ** 2)):.2f} м, среднее {e3.mean():.2f}, max {e3.max():.2f}, в конце {e3[-1]:.2f}')
    # скорость эталона против GNSS-скорости по колёсам — проверка знака и единиц
    t0 = ref[0, 1]
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax[0].plot(ref[:, 1] - t0, ref[:, 5], 'k-', lw=1, label='эталон twist.linear.x')
    ax[0].plot(t_out - t0, out[:, 1], 'r-', lw=0.8, label='наш /result/velocity')
    ax[0].set_ylabel('м/с'), ax[0].legend(), ax[0].grid(alpha=0.3)
    ax[1].plot(ref[ok, 1] - t0, e3, label='3D'), ax[1].plot(ref[ok, 1] - t0, ep[:, 2], label='z')
    ax[1].set_ylabel('ошибка, м'), ax[1].legend(), ax[1].grid(alpha=0.3)
    ax[2].plot(ref[ok, 1] - t0, ev, 'g-', lw=0.8), ax[2].set_ylabel('ошибка скорости, м/с'), ax[2].grid(alpha=0.3)
    ax[2].set_xlabel('с от начала эталона')
    fig.suptitle(f'{bag}: наш оценщик против /localization/kinematic_state')
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f'check_{bag}.png', dpi=75)


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET / 'check' / 'check-code' / 'bags' / '30618_88aea4d9')
