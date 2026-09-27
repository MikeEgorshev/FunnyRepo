"""Офлайн-оценка tram_odometry по прогонам rosbag2 без ROS, по метрикам судьи.

Прогон воспроизводится в порядке времени записи, как ros2 bag play. Оценщик получает
тележки и контроллер, GNSS — только в окне выставки (как нода). После каждого входа —
выход с меткой этого входа, как публикует нода.

Эталон — base_link по двум антеннам GNSS в сетке MGRS (QA 25.09): антенны на оси x
трамвая, master в 9,873 м позади base_link, rover в 2,563 м впереди, обе на 3 м выше
рельса. Эталон скорости — /sensing/gnss/master/vel, если есть. У судьи эталон другой
(/localization/kinematic_state, слияние с лидаром), поэтому цифры — оценка, а не балл.

Выход и эталон сопоставляются по ближайшей метке с допуском 0,05 с.

Запуск (нужен pip-пакет rosbags; нода его не использует):
    python3 scripts/evaluate_bags.py <прогон> [<прогон> ...] \
        [--route-map route.csv] [--stops stops.csv] [--k 3.5966] [--params fitted.yaml] [--out results.csv]
"""
import argparse
import bisect
import csv
import math
import sys
from dataclasses import fields
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'tram_odometry'))
from tram_odometry.estimator import Estimator, FilterParams  # noqa: E402
from tram_odometry.model import ModelParams  # noqa: E402
from tram_odometry.positioning import Positioner  # noqa: E402
from tram_odometry.stops import StopFixer  # noqa: E402
from tram_odometry.track import MgrsLocal, RouteTrack  # noqa: E402

FRONT, REAR = '/vehicle/front_bogie_velocity', '/vehicle/rear_bogie_velocity'
CMD = '/vehicle/driver_position_cmd'
MASTER, ROVER = '/sensing/gnss/master/fix', '/sensing/gnss/rover/fix'
MASTER_VEL = '/sensing/gnss/master/vel'
KIN = '/localization/kinematic_state'     # эталон судьи (есть в проверочных прогонах)
MSG_DEFS = {
    'tram_vehicle_msgs/msg/VelocitySensor': 'std_msgs/Header header\nfloat64 velocity\n',
    'tram_vehicle_msgs/msg/DriverControllerCommand': 'std_msgs/Header header\nint8 position\n',
}
LEVER_M, DZ_M = 9.873, -3.0
MATCH_TOL = 0.05


def typestore():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for name, text in MSG_DEFS.items():
        types.update(get_types_from_msg(text, name))
    ts.register(types)
    return ts


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def read_bag(path):
    """[(время записи, топик, сообщение)] по возрастанию времени записи."""
    topics = {FRONT, REAR, CMD, MASTER, ROVER, MASTER_VEL, KIN}
    out = []
    with AnyReader([Path(path)], default_typestore=typestore()) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, t, raw in reader.messages(connections=conns):
            out.append((t * 1e-9, conn.topic, reader.deserialize(raw, conn.msgtype)))
    out.sort(key=lambda e: e[0])
    return out


def reference(events, frame=MgrsLocal()):
    """Эталон: [(метка, x, y, z)] base_link и [(метка, скорость)]. Есть /localization/kinematic_state —
    берётся он, как у судьи: pose.pose.position и twist.twist.linear.x. Нет — по двум антеннам GNSS."""
    kin = [(_stamp(m), m) for _, topic, m in events if topic == KIN]
    if kin:
        pos = [(t, m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z) for t, m in kin]
        vel = [(t, m.twist.twist.linear.x) for t, m in kin]
        return sorted(pos), sorted(vel)
    rover = sorted((_stamp(m), frame.forward(m.latitude, m.longitude, m.altitude))
                   for _, topic, m in events if topic == ROVER and m.status.status >= 0)
    rover_t = [r[0] for r in rover]
    pos, vel = [], []
    for _, topic, m in events:
        if topic == MASTER and m.status.status >= 0 and math.isfinite(m.latitude):
            t = _stamp(m)
            mx, my, mz = frame.forward(m.latitude, m.longitude, m.altitude)
            i = bisect.bisect_left(rover_t, t)
            near = [j for j in (i - 1, i) if 0 <= j < len(rover) and abs(rover_t[j] - t) <= 0.1]
            if not near:
                continue
            rx, ry, _ = rover[min(near, key=lambda j: abs(rover_t[j] - t))][1]
            d = math.hypot(rx - mx, ry - my)
            if not 10.0 < d < 15.0:          # база антенн 12,436 м: иначе один из фиксов плохой
                continue
            pos.append((t, mx + LEVER_M * (rx - mx) / d, my + LEVER_M * (ry - my) / d, mz + DZ_M))
        elif topic == MASTER_VEL:
            lin = getattr(m.twist, 'twist', m.twist).linear   # TwistStamped или TwistWithCovarianceStamped
            vel.append((_stamp(m), math.hypot(lin.x, lin.y)))
    return sorted(pos), sorted(vel)


def load_params(path):
    """Модель и фильтр из файла параметров ноды (YAML), например от fit_model.py."""
    import yaml
    ros = yaml.safe_load(Path(path).read_text(encoding='utf-8'))['tram_odometry']['ros__parameters']

    def make(cls, section):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in ros.get(section, {}).items() if k in names})
    return make(ModelParams, 'model'), make(FilterParams, 'filter')


def replay(events, route=None, k=3.5966, window_s=5.0, model=None, fparams=None, corrections=False,
           timer_hz=None, max_extrapolation_s=0.2):
    """Выходы оценщика: [(метка, v, x, y, z, yaw, frame, проскальзывание)].

    timer_hz=None — выход после каждого входа с его меткой. timer_hz=25 — как в ноде: выход на
    сетке меток через state_at() (прогноз вперёд не дольше max_extrapolation_s); так офлайн-оценка
    ловит и то, что бывает только при публикации по таймеру."""
    fparams = fparams or FilterParams(k0=k)
    est = Estimator(model=model or ModelParams(), params=fparams)
    pos = Positioner(route, window_s, corrections=corrections)
    stops, resets, out, last_out = None, 0, [], None
    tick, last_in = None, None

    def emit(t, st):
        x, y, z, yaw, frame = pos.pose(st.s)
        out.append((t, st.v, x, y, z, yaw, frame, st.slip))

    def use_track():
        est.grade_at = getattr(pos.track, 'grade_at', None) or (lambda s: 0.0)
        st = getattr(pos.track, 'stops', None)
        return StopFixer(st, getattr(pos.track, 'length', None)) if st else None

    for _, topic, m in events:
        t = _stamp(m)
        if topic in (MASTER, ROVER):
            pos.fix(t, m.latitude, m.longitude, m.altitude, m.status.status, topic == ROVER)
            continue
        if topic not in (CMD, FRONT, REAR):
            continue
        if timer_hz:
            tick = t if tick is None else tick
            while tick < t:                   # тики таймера до этого входа
                if last_in is not None and tick - last_in <= max_extrapolation_s:
                    emit(tick, est.state_at(tick))
                tick += 1.0 / timer_hz
            last_in = t if last_in is None else max(last_in, t)
        if topic == CMD:
            est.set_notch(t, int(m.position))
        elif topic in (FRONT, REAR):
            est.wheel('front' if topic == FRONT else 'rear', t, float(m.velocity))
        else:
            continue
        if est.resets != resets:
            resets = est.resets
            pos.start_run()
            stops = use_track()
        if pos.update(t, est):
            stops = use_track()
        if stops is not None and topic in (FRONT, REAR):
            stops.update(t, est)
        if timer_hz or (last_out is not None and t < last_out):
            continue                          # метки выхода не убывают, как у ноды
        last_out = t
        emit(t, est.state())
    return out


def _nearest(times, t):
    i = bisect.bisect_left(times, t)
    best = min((j for j in (i - 1, i) if 0 <= j < len(times)), key=lambda j: abs(times[j] - t), default=None)
    return best if best is not None and abs(times[best] - t) <= MATCH_TOL else None


def metrics(out, ref_pos, ref_vel):
    res = {'outputs': len(out)}
    times = [o[0] for o in out]
    if ref_vel:
        ev = [out[j][1] - v for t, v in ref_vel if (j := _nearest(times, t)) is not None]
        if ev:
            res['v_rmse'] = math.sqrt(sum(e * e for e in ev) / len(ev))
            res['v_mae'] = sum(abs(e) for e in ev) / len(ev)
            res['v_bias'] = sum(ev) / len(ev)
            res['v_max'] = max(abs(e) for e in ev)
    mapped = [(t, x, y, z) for t, x, y, z in ref_pos]
    pairs = [(r, out[j]) for r in mapped if (j := _nearest(times, r[0])) is not None and out[j][6] == 'map']
    if pairs:
        e3 = [math.dist(r[1:4], o[2:5]) for r, o in pairs]
        along = [(o[2] - r[1]) * math.cos(o[5]) + (o[3] - r[2]) * math.sin(o[5]) for r, o in pairs]
        dist = sum(math.dist(a[1:3], b[1:3]) for a, b in zip(mapped, mapped[1:]))
        res.update(p3d_mean=sum(e3) / len(e3), p3d_rmse=math.sqrt(sum(e * e for e in e3) / len(e3)),
                   p3d_max=max(e3), p3d_final=e3[-1], along_rmse=math.sqrt(sum(a * a for a in along) / len(along)),
                   z_rmse=math.sqrt(sum((o[4] - r[3]) ** 2 for r, o in pairs) / len(pairs)), dist_m=dist,
                   drift_pct=100.0 * e3[-1] / dist if dist > 0 else float('nan'),
                   **{f'{a}_rmse': math.sqrt(sum((o[2 + k] - r[1 + k]) ** 2 for r, o in pairs) / len(pairs))
                      for k, a in enumerate('xy')})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--route-map', default=None)
    ap.add_argument('--stops', default=None)
    ap.add_argument('--k', type=float, default=3.5966)
    ap.add_argument('--params', default=None, help='YAML параметров ноды (fit_model.py): модель и k0')
    ap.add_argument('--gnss-corrections', action='store_true', help='фиксы после выставки -> привязки дистанции')
    ap.add_argument('--timer-hz', type=float, default=None, help='выход по таймеру, как в ноде (например, 25)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    route = RouteTrack.load(args.route_map, MgrsLocal(), args.stops) if args.route_map else None
    rows = []
    for bag in args.bags:
        events = read_bag(bag)
        ref_pos, ref_vel = reference(events)
        model, fparams = load_params(args.params) if args.params else (None, None)
        row = {'bag': Path(bag).name, **metrics(replay(events, route, args.k, model=model, fparams=fparams,
                                                        corrections=args.gnss_corrections, timer_hz=args.timer_hz),
                                                 ref_pos, ref_vel)}
        rows.append(row)
        print(' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))
    keys = sorted({k for r in rows for k in r if k != 'bag'})
    if len(rows) > 1:
        for k in keys:
            vals = sorted(r[k] for r in rows if k in r and isinstance(r[k], float))
            if vals:
                print(f'median {k:12s} {vals[len(vals) // 2]:.3f}')
    if args.out:
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['bag'] + keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
