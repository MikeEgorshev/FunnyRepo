"""Синтетический датасет в формате организаторов: прогоны rosbag2 с теми же топиками и типами.

Нужен, чтобы проверить весь конвейер (карта -> подбор модели -> оценка) без настоящих
данных. Это не данные трамвая, а контролируемая модель мира:
- линия: двухпутка длиной --line-m, пути в 4 м, разворотные петли на концах; высота меняется;
- водитель: разгон до своей скорости, выбег, торможение точно к остановке, стоянка 15–30 с,
  иногда остановка у светофора; динамика — model.py с «истинными» параметрами и уклоном;
- датчики: тележки 10 Гц в км/ч со своим k у каждого прогона, контроллер 20 Гц, GNSS 10 Гц
  (master, rover в 12,436 м впереди, скорость master); метка header на 35 мс раньше записи;
- сбои: буксование, юз, пропуск обеих тележек, зависание датчика, пропуск GNSS.

Запуск (нужен pip-пакет rosbags):
    python3 scripts/make_synthetic_bags.py --out-dir /tmp/synth --runs 10
"""
import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Writer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'tram_odometry'))
from evaluate_bags import CMD, FRONT, MASTER, MASTER_VEL, REAR, ROVER, typestore  # noqa: E402
from tram_odometry.model import ModelParams, accel, lag  # noqa: E402
from tram_odometry.track import LocalFrame  # noqa: E402

TRUTH = ModelParams(a_max=1.2, v_base=6.5, b_max=1.4, v_blend=2.0, blend_floor=0.75,
                    res_a=0.018, res_b=0.001, res_c=0.0005, tau_cmd=0.7)
ORIGIN = (55.80, 37.40, 150.0)
T0 = 1_758_800_000.0
BASELINE = 12.436


class Circuit:
    """Замкнутый маршрут антенны master: s -> (x, y, z, курс) в ENU от ORIGIN."""

    def __init__(self, line_m=3500.0, r=25.0, a=1.0007, ds=0.5):
        prof = [(0.0, line_m), (-1 / r, a * r), (1 / r, (math.pi + 2 * a) * r), (-1 / r, a * r)] * 2
        x, y, h = 0.0, -2.0, 0.0
        self.pts, self.line_m = [], line_m
        for k, length in prof:
            n = int(round(length / ds))
            for _ in range(n):
                z = 20.0 * math.sin(math.pi * x / line_m) + 6.0 * math.sin(6 * math.pi * x / line_m)
                self.pts.append((x, y, z, h))
                h += k * length / n
                x += math.cos(h) * length / n
                y += math.sin(h) * length / n
        self.s = [0.0]
        for p, q in zip(self.pts, self.pts[1:] + self.pts[:1]):
            self.s.append(self.s[-1] + math.dist(p[:3], q[:3]))
        self.length = self.s[-1]
        self.ds = self.length / len(self.pts)
        loop = prof[1][1] + prof[2][1] + prof[3][1]
        # остановки: на каждом пути через ~600 м, не в петлях; конечные — в начале прямых
        self.stops = sorted([30.0 + 600.0 * i for i in range(6)] +
                            [line_m + loop + 30.0 + 600.0 * i for i in range(6)])

    def at(self, s):
        s %= self.length
        i = min(int(s / self.ds), len(self.pts) - 1)
        while self.s[i] > s:
            i -= 1
        while self.s[i + 1] < s:
            i += 1
        p, q = self.pts[i], self.pts[(i + 1) % len(self.pts)]
        t = (s - self.s[i]) / (self.s[i + 1] - self.s[i])
        dh = math.atan2(math.sin(q[3] - p[3]), math.cos(q[3] - p[3]))
        return tuple(p[k] + t * (q[k] - p[k]) for k in range(3)) + (p[3] + t * dh,)

    def grade(self, s):
        return (self.at(s + 10.0)[2] - self.at(s - 10.0)[2]) / 20.0


def drive(circ, rnd, s0, dist, dt=0.01):
    """Истинное движение: [(t, s, v, notch)] каждые dt. Водитель тормозит к остановкам."""
    stops = [x + k * circ.length for k in range(3) for x in circ.stops if s0 + 5.0 < x + k * circ.length < s0 + dist]
    extra = sorted(rnd.uniform(s0 + 200.0, s0 + dist - 100.0) for _ in range(rnd.randint(0, 2)))   # светофоры
    targets = sorted(stops + extra)
    t, s, v, u, n = 0.0, s0, 0.0, 0.0, 0
    cruise = rnd.uniform(9.0, 14.0)
    out, dwell_until, hold_until = [], rnd.uniform(3.0, 8.0), 0.0
    while s < s0 + dist and t < 4 * 3600.0:
        if not targets and v < 0.05 and s0 + dist - s < 10.0 and t > dwell_until:
            break                                                 # доехал до конца прогона
        if t < dwell_until:
            n = -4
        elif t >= hold_until:
            gap = (targets[0] if targets else s0 + dist) - s
            b = 0.9                                               # м/с², плановое замедление
            if gap < v * v / (2 * b) + 3.0 * v * TRUTH.tau_cmd + 2.0:
                need = v * v / (2 * max(gap, 0.5))
                n = -min(15, max(3, round(15 * need / TRUTH.b_max)))
            elif v < cruise - 1.0:
                n = rnd.choice([8, 10, 12, 15])
            elif v > cruise + 0.5:
                n = rnd.choice([0, 0, -2])
            else:
                n = rnd.choice([0, 0, 2, 3])
            hold_until = t + rnd.uniform(0.5, 2.0)
        if v < 0.05 and n < 0 and targets and targets[0] - s < 8.0:
            targets.pop(0)
            dwell_until = t + rnd.uniform(15.0, 30.0)
            cruise = rnd.uniform(9.0, 14.0)
        u = lag(u, n, dt, TRUTH)
        a = accel(u, v, TRUTH, circ.grade(s)) if v > 0 or u > 0 else 0.0
        v = max(0.0, v + a * dt)
        s += v * dt
        out.append((t, s, v, n))
        t += dt
    return out


def faults(rnd, duration):
    def spans(count, lo, hi):
        out = []
        for _ in range(count):
            a = rnd.uniform(30.0, max(31.0, duration - 60.0))
            out.append((a, a + rnd.uniform(lo, hi)))
        return out
    return {'slip': spans(rnd.randint(1, 3), 4.0, 12.0), 'slide': spans(rnd.randint(1, 2), 3.0, 8.0),
            'dropout': spans(rnd.randint(0, 2), 8.0, 25.0), 'frozen': spans(rnd.randint(0, 1), 5.0, 15.0),
            'gnss_gap': spans(rnd.randint(0, 2), 5.0, 30.0)}


def _in(t, spans):
    return any(a <= t < b for a, b in spans)


def write_run(path, circ, frame, seed, ts):
    rnd = random.Random(seed)
    s0 = rnd.choice(circ.stops) - 5.0
    truth = drive(circ, rnd, s0, rnd.uniform(0.45, 0.9) * circ.length)
    duration = truth[-1][0]
    k = rnd.uniform(3.585, 3.612)
    f = faults(rnd, duration)
    T = ts.types

    def hdr(t):
        stamp = T['builtin_interfaces/msg/Time'](sec=int(t), nanosec=int((t - int(t)) * 1e9))
        return T['std_msgs/msg/Header'](stamp=stamp, frame_id='')

    def speed(t, kmh):
        return T['tram_vehicle_msgs/msg/VelocitySensor'](header=hdr(t), velocity=float(max(kmh, -0.3)))
    msgs = []
    frozen_val = {'front': None}
    for i, (t, s, v, n) in enumerate(truth):
        st = T0 + seed * 10_000 + t
        if i % 5 == 0:
            msgs.append((st, CMD, T['tram_vehicle_msgs/msg/DriverControllerCommand'](header=hdr(st), position=int(n))))
        if i % 10 == 0 and not _in(t, f['dropout']):
            fv, rv = v, v
            if _in(t, f['slip']) and n > 0:
                fv, rv = v * 1.25 + 0.3, v * 1.08
            if _in(t, f['slide']) and n < 0:
                fv, rv = v * 0.7, v * 0.8
            fk = fv * k + rnd.gauss(0.0, 0.12)
            if _in(t, f['frozen']) and v > 1.0:
                frozen_val['front'] = frozen_val['front'] or round(fk, 2)
                fk = frozen_val['front']
            else:
                frozen_val['front'] = None
            msgs.append((st, FRONT, speed(st, fk)))
            msgs.append((st + 0.004, REAR, speed(st + 0.004, rv * k + rnd.gauss(0.0, 0.12))))
        if i % 10 == 5 and not _in(t, f['gnss_gap']):
            for topic, ds in ((MASTER, 0.0), (ROVER, BASELINE)):
                x, y, z, _ = circ.at(s + ds)
                lat, lon, alt = frame.inverse(x + rnd.gauss(0, 0.03), y + rnd.gauss(0, 0.03), z + rnd.gauss(0, 0.05))
                msgs.append((st, topic, T['sensor_msgs/msg/NavSatFix'](
                    header=hdr(st), status=T['sensor_msgs/msg/NavSatStatus'](status=2, service=1),
                    latitude=lat, longitude=lon, altitude=alt, position_covariance=np.zeros(9),
                    position_covariance_type=0)))
            yaw = circ.at(s)[3]
            V = T['geometry_msgs/msg/Vector3']
            twist = T['geometry_msgs/msg/Twist'](linear=V(x=v * math.cos(yaw), y=v * math.sin(yaw), z=0.0),
                                                 angular=V(x=0.0, y=0.0, z=0.0))
            msgs.append((st, MASTER_VEL, T['geometry_msgs/msg/TwistStamped'](header=hdr(st), twist=twist)))
    msgs.sort(key=lambda m: m[0])
    types = {FRONT: 'tram_vehicle_msgs/msg/VelocitySensor', REAR: 'tram_vehicle_msgs/msg/VelocitySensor',
             CMD: 'tram_vehicle_msgs/msg/DriverControllerCommand', MASTER: 'sensor_msgs/msg/NavSatFix',
             ROVER: 'sensor_msgs/msg/NavSatFix', MASTER_VEL: 'geometry_msgs/msg/TwistStamped'}
    with Writer(path, version=8) as w:
        conns = {topic: w.add_connection(topic, typ, typestore=ts) for topic, typ in types.items()}
        for st, topic, m in msgs:
            w.write(conns[topic], int((st + 0.035) * 1e9), ts.serialize_cdr(m, conns[topic].msgtype))
    return {'bag': path.name, 'duration_s': round(duration, 1), 'distance_m': round(truth[-1][1] - s0, 1),
            'k': round(k, 4), 'faults': f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--line-m', type=float, default=3500.0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    circ, frame, ts = Circuit(args.line_m), LocalFrame(*ORIGIN), typestore()
    info = {'truth_model': asdict(TRUTH), 'length_m': circ.length, 'stops': circ.stops, 'runs': []}
    for i in range(args.runs):
        r = write_run(out / f'synth_{i:02d}', circ, frame, i + 1, ts)
        info['runs'].append(r)
        print(r['bag'], r['duration_s'], 's', r['distance_m'], 'm', 'k', r['k'])
    (out / 'truth.json').write_text(json.dumps(info, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
