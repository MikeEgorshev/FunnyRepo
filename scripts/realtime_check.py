"""Проверка реального времени ноды tram_odometry.

Публикует синтетический прогон (src/tram_odometry/test/synthetic.py) с реальной частотой:
контроллер 20 Гц, тележки по 10 Гц. Метка входа — текущее время. Меряет:
- задержку «вход → публикация»: время прихода выхода минус метка последнего входа,
  который в него вошёл (только пока входы идут: прогноз при молчании входов — отдельно);
- частоту /result/position и /result/velocity и самый длинный разрыв выхода;
- загрузку CPU и память процесса ноды (по /proc, если передан --pid).

Запуск (нода уже запущена):
    python3 scripts/realtime_check.py --duration 60 --pid $(pgrep -x tram_odometry_n)
    --stall-at 30 --stall-s 1.5 — все входы молчат 1,5 с (в данных так бывает: около 1 с);
    --gnss 55.80,37.42 — первые 3 с публиковать фиксы master и rover (rover на 12,436 м
    севернее): проверка выставки и привязки к карте.
Печатает сводку и строку JSON.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'tram_odometry' / 'test'))
from synthetic import Scenario, run  # noqa: E402

BASELINE_M = 12.436       # master -> rover вдоль оси трамвая (QA 25.09)


def proc_stats(pid):
    """(процессорное время, с; RSS, МБ) процесса."""
    with open(f'/proc/{pid}/stat') as f:
        fields = f.read().rsplit(')', 1)[1].split()
    ticks = os.sysconf('SC_CLK_TCK')
    cpu = (int(fields[11]) + int(fields[12])) / ticks
    with open(f'/proc/{pid}/status') as f:
        rss = next(int(line.split()[1]) for line in f if line.startswith('VmRSS')) / 1024.0
    return cpu, rss


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(p * (len(s) - 1)))] if s else float('nan')


class Checker(Node):
    def __init__(self, duration, stall_at=None, stall_s=0.0, gnss=None):
        super().__init__('realtime_check')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = {
            'front': self.create_publisher(VelocitySensor, '/vehicle/front_bogie_velocity', qos),
            'rear': self.create_publisher(VelocitySensor, '/vehicle/rear_bogie_velocity', qos),
            'cmd': self.create_publisher(DriverControllerCommand, '/vehicle/driver_position_cmd', qos),
            'master': self.create_publisher(NavSatFix, '/sensing/gnss/master/fix', qos),
            'rover': self.create_publisher(NavSatFix, '/sensing/gnss/rover/fix', qos),
        }
        self.create_subscription(Odometry, '/result/position', self._odom, qos)
        self.create_subscription(VelocitySensor, '/result/velocity', self._vel, qos)
        events, _ = run(Scenario(duration=duration, slip=[(22.0, 30.0)], dropout=[(40.0, 55.0)]))
        if stall_at is not None:        # все входы молчат, дальше прогон идёт с тем же временем
            events = [e for e in events if not stall_at <= e[0] < stall_at + stall_s]
        if gnss is not None:
            lat, lon = gnss
            dlat = BASELINE_M / 111320.0
            events += [(0.1 * i, 'master', (lat, lon)) for i in range(30)]
            events += [(0.1 * i + 0.01, 'rover', (lat + dlat, lon)) for i in range(30)]
            events.sort(key=lambda e: e[0])
        self.events = events
        self.i = 0
        self.t0 = None
        self.inputs = []          # (метка, время отправки) опубликованных входов тележек и контроллера
        self.latency = []
        self.n_odom = self.n_vel = 0
        self.max_gap = 0.0
        self.first_rx = self.last_rx = None
        self.last_odom = None
        self.create_timer(0.001, self._tick)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _tick(self):
        now = self._now()
        if self.t0 is None:
            self.t0 = now
        while self.i < len(self.events) and self.events[self.i][0] <= now - self.t0:
            _, kind, value = self.events[self.i]
            if kind in ('master', 'rover'):
                msg = NavSatFix()
                msg.latitude, msg.longitude = value
                msg.altitude = 150.0
            elif kind == 'cmd':
                msg = DriverControllerCommand()
                msg.position = int(value)
            else:
                msg = VelocitySensor()
                msg.velocity = float(value)
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub[kind].publish(msg)
            if kind not in ('master', 'rover'):
                self.inputs.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, now))
            self.i += 1

    def _odom(self, msg):
        rx = self._now()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        last = next((inp for inp in reversed(self.inputs) if inp[0] <= stamp + 1e-6), None)
        if last is not None and rx - last[1] < 0.1:     # вход ещё идёт: это ответ на него
            self.latency.append(rx - last[0])
        self.n_odom += 1
        if self.last_rx is not None:
            self.max_gap = max(self.max_gap, rx - self.last_rx)
        self.first_rx = self.first_rx or rx
        self.last_rx = rx
        self.last_odom = msg

    def _vel(self, msg):
        self.n_vel += 1

    @property
    def done(self):
        return self.i >= len(self.events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=90.0)
    ap.add_argument('--pid', type=int, default=0, help='PID ноды для замера CPU и памяти')
    ap.add_argument('--stall-at', type=float, default=None, help='с от начала: все входы замолкают')
    ap.add_argument('--stall-s', type=float, default=1.5, help='сколько секунд входы молчат')
    ap.add_argument('--gnss', default=None, help='LAT,LON точки старта для фиксов master и rover')
    args = ap.parse_args()
    gnss = tuple(float(v) for v in args.gnss.split(',')) if args.gnss else None
    rclpy.init()
    node = Checker(args.duration, args.stall_at, args.stall_s, gnss)
    cpu0 = proc_stats(args.pid) if args.pid else None
    wall0 = time.monotonic()
    rss_max = 0.0
    last_sample = 0.0
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.01)
        if args.pid and time.monotonic() - last_sample > 1.0:
            rss_max = max(rss_max, proc_stats(args.pid)[1])
            last_sample = time.monotonic()
    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.01)
    wall = time.monotonic() - wall0
    span = (node.last_rx - node.first_rx) if node.first_rx else float('nan')
    result = {
        'duration_s': round(wall, 1),
        'inputs': len(node.inputs),
        'odom_msgs': node.n_odom,
        'odom_hz': round(node.n_odom / span, 2) if span else None,
        'velocity_hz': round(node.n_vel / span, 2) if span else None,
        'max_output_gap_ms': round(1000 * node.max_gap, 1),
        'latency_ms_p50': round(1000 * pct(node.latency, 0.5), 1),
        'latency_ms_p99': round(1000 * pct(node.latency, 0.99), 1),
        'latency_ms_max': round(1000 * max(node.latency), 1) if node.latency else None,
    }
    if node.last_odom is not None:
        p = node.last_odom.pose.pose.position
        result['last_frame_id'] = node.last_odom.header.frame_id
        result['last_xy'] = [round(p.x, 1), round(p.y, 1)]
        result['last_yaw_deg'] = round(math.degrees(2 * math.atan2(node.last_odom.pose.pose.orientation.z,
                                                                   node.last_odom.pose.pose.orientation.w)), 1)
    if cpu0:
        cpu1, rss = proc_stats(args.pid)
        result['node_cpu_percent_of_one_core'] = round(100 * (cpu1 - cpu0[0]) / wall, 1)
        result['node_rss_mb_max'] = round(max(rss_max, rss), 1)
    for k, v in result.items():
        print(f'{k:32s} {v}')
    print(json.dumps(result))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
