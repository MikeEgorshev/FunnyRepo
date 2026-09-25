"""Разбор симуляции в Docker (docker/sim.sh): частота, задержка, точность, ресурсы.

Задержка «вход -> результат»: выход публикуется с header.stamp входа, который его вызвал;
разница времён получения (в записи) входа и выхода с одинаковой меткой — сквозная задержка
через DDS и ноду. Точность — против GNSS исходного прогона (как в evaluate.py).

Запуск: python analyze_sim.py <каталог результата sim> <id прогона>
"""
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from bagio import load_cached, typestore
from evaluate import match, reference

INPUTS = ('/vehicle/front_bogie_velocity', '/vehicle/rear_bogie_velocity', '/vehicle/driver_position_cmd')


def read(result_dir):
    ts = typestore()
    rows = {}
    with AnyReader([Path(result_dir)], default_typestore=ts) as r:
        for conn, t, raw in r.messages():
            m = r.deserialize(raw, conn.msgtype)
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if conn.topic == '/result/velocity':
                rows.setdefault('v', []).append((t * 1e-9, stamp, m.velocity))
            elif conn.topic == '/result/position':
                p = m.pose.pose.position
                rows.setdefault('p', []).append((t * 1e-9, stamp, p.x, p.y, p.z))
            elif conn.topic in INPUTS:
                rows.setdefault('in', []).append((t * 1e-9, stamp))
    return {k: np.array(v) for k, v in rows.items()}


def main(sim_dir, bag_id):
    sim_dir = Path(sim_dir)
    d = read(sim_dir / 'result')
    v, p, inp = d['v'], d['p'], d['in']
    dur = v[-1, 0] - v[0, 0]
    print(f'/result/velocity: {len(v)} сообщ., {len(v) / dur:.1f} Гц; /result/position: {len(p)}, {len(p) / dur:.1f} Гц')
    gaps = np.diff(p[:, 0])
    print(f'интервал между /result/position: медиана {np.median(gaps) * 1e3:.0f} мс, 99% {np.percentile(gaps, 99) * 1e3:.0f} мс, '
          f'макс {gaps.max() * 1e3:.0f} мс')
    recv_in = {}
    for t, s in inp:
        recv_in.setdefault(round(s, 6), t)
    lat = np.array([t - recv_in[round(s, 6)] for t, s, *_ in p if round(s, 6) in recv_in]) * 1e3
    print(f'задержка вход -> /result/position: медиана {np.median(lat):.1f} мс, 95% {np.percentile(lat, 95):.1f} мс, '
          f'99% {np.percentile(lat, 99):.1f} мс, макс {lat.max():.1f} мс (сопоставлено {len(lat)} из {len(p)})')
    src = load_cached(bag_id)
    t_ref, p_ref, t_vref, v_ref = reference(src)
    order = np.argsort(p[:, 1])
    ps, vs = p[order], v[np.argsort(v[:, 1])]
    j, ok = match(t_vref, vs[:, 1])
    ev = vs[j[ok], 2] - v_ref[ok]
    j, ok2 = match(t_ref, ps[:, 1])
    e3 = np.linalg.norm(ps[j[ok2], 2:5] - p_ref[ok2], axis=1)
    print(f'точность: скорость RMSE {np.sqrt(np.mean(ev ** 2)):.3f} м/с (сопоставлено {ok.mean():.1%}), '
          f'3D-ошибка средняя {e3.mean():.2f} м, в конце {e3[-1]:.2f} м (сопоставлено {ok2.mean():.1%})')
    res = np.loadtxt(sim_dir / 'resources.log', ndmin=2) if (sim_dir / 'resources.log').stat().st_size else np.zeros((0, 2))
    if len(res):
        print(f'ресурсы ноды: RSS макс {res[:, 0].max() / 1024:.0f} МБ, CPU средн {res[:, 1].mean():.1f}% '
              f'(ps: среднее за жизнь процесса), макс {res[:, 1].max():.1f}%')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
