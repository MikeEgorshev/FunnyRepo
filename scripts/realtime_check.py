"""Проверка реального времени ноды tram_odometry.

Публикует синтетический прогон (цикл разгон — торможение) с реальной частотой:
контроллер 20 Гц, тележки по 10 Гц. Метка входа — текущее время. Меряет:
- задержку «вход → публикация»: время прихода выхода с меткой входа минус эта метка
  (выходы-прогнозы, когда входы молчат, считаются отдельно);
- частоту /result/position и /result/velocity и самый длинный разрыв между выходами;
- загрузку CPU и память процесса ноды (по /proc, если передан --pid).

Запуск (нода уже запущена):
    python3 scripts/realtime_check.py --duration 90 --pid $(pgrep -x tram_odometry_n)
    --stall-at 30 --stall-s 1.5 — все входы молчат 1,5 с (в данных так бывает: около 1 с);
    критерий жюри — выход не реже 10 Гц.
Печатает сводку и строку JSON.
"""
import argparse
import json
import os
import random
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

K_WHEEL = 3.5966   # км/ч тележки на м/с


def _notch(t):
    c = t % 80.0
    return 10 if c < 14 else 3 if c < 36 else 0 if c < 52 else -8 if c < 68 else -4


def scenario(duration, seed=1):
    """Цикл «разгон — ход — выбег — торможение — стоянка» по 80 с: [(t, kind, value)].

    Грубая кинематика по позиции контроллера — для нагрузки на ноду, не для точности.
    """
    rnd = random.Random(seed)
    events, v, dt = [], 0.0, 0.01
    for i in range(int(duration / dt) + 1):
        t = i * dt
        n = _notch(t)
        a = 1.0 * min(1.0, 7.0 / max(v, 0.1)) * n / 15 if n > 0 else 1.2 * n / 15 if n < 0 else -0.02
        v = max(0.0, v + a * dt)
        if i % 5 == 0:
            events.append((t, 'cmd', n))
        if i % 10 == 0:
            events.append((t, 'front', max(0.0, v * K_WHEEL + rnd.gauss(0.0, 0.15))))
            events.append((t + 0.005, 'rear', max(0.0, v * K_WHEEL + rnd.gauss(0.0, 0.15))))
    return sorted(events)


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
    def __init__(self, duration, stall_at=None, stall_s=0.0):
        super().__init__('realtime_check')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = {
            'front': self.create_publisher(VelocitySensor, '/vehicle/front_bogie_velocity', qos),
            'rear': self.create_publisher(VelocitySensor, '/vehicle/rear_bogie_velocity', qos),
            'cmd': self.create_publisher(DriverControllerCommand, '/vehicle/driver_position_cmd', qos),
        }
        self.create_subscription(Odometry, '/result/position', self._odom, qos)
        self.create_subscription(VelocitySensor, '/result/velocity', self._vel, qos)
        events = scenario(duration)
        if stall_at is not None:        # все входы молчат, дальше прогон идёт с тем же временем
            events = [e for e in events if not stall_at <= e[0] < stall_at + stall_s]
        self.events = events
        self.i = 0
        self.t0 = None
        self.inputs = []          # метки опубликованных входов, по возрастанию
        self.latency = []          # выходы с меткой входа: задержка ответа на вход
        self.latency_any = []      # все выходы: от последнего входа (для ноды на таймере)
        self.n_predicted = 0
        self._input_set = set()
        self.n_odom = self.n_vel = 0
        self.max_gap = 0.0
        self.first_rx = self.last_rx = None
        self.create_timer(0.001, self._tick)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _tick(self):
        now = self._now()
        if self.t0 is None:
            self.t0 = now
        while self.i < len(self.events) and self.events[self.i][0] <= now - self.t0:
            _, kind, value = self.events[self.i]
            msg = DriverControllerCommand() if kind == 'cmd' else VelocitySensor()
            msg.header.stamp = self.get_clock().now().to_msg()
            if kind == 'cmd':
                msg.position = int(value)
            else:
                msg.velocity = float(value)
            self.pub[kind].publish(msg)
            self.inputs.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            self._input_set.add((msg.header.stamp.sec, msg.header.stamp.nanosec))
            self.i += 1

    def _odom(self, msg):
        rx = self._now()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if (msg.header.stamp.sec, msg.header.stamp.nanosec) in self._input_set:
            self.latency.append(rx - stamp)
        else:
            self.n_predicted += 1
        # последний вход, который уже мог войти в этот выход
        last_in = next((t for t in reversed(self.inputs) if t <= stamp + 1e-6), None)
        if last_in is not None:
            self.latency_any.append(rx - last_in)
        self.n_odom += 1
        if self.last_rx is not None:
            self.max_gap = max(self.max_gap, rx - self.last_rx)
        self.first_rx = self.first_rx or rx
        self.last_rx = rx

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
    args = ap.parse_args()
    rclpy.init()
    node = Checker(args.duration, args.stall_at, args.stall_s)
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
    lat = node.latency or node.latency_any
    span = (node.last_rx - node.first_rx) if node.first_rx else float('nan')
    result = {
        'duration_s': round(wall, 1),
        'inputs': len(node.inputs),
        'odom_msgs': node.n_odom,
        'odom_hz': round(node.n_odom / span, 2) if span else None,
        'velocity_hz': round(node.n_vel / span, 2) if span else None,
        'max_output_gap_ms': round(1000 * node.max_gap, 1),
        'predicted_outputs': node.n_predicted,
        'latency_ms_p50': round(1000 * pct(lat, 0.5), 1),
        'latency_ms_p99': round(1000 * pct(lat, 0.99), 1),
        'latency_ms_max': round(1000 * max(lat), 1) if lat else None,
    }
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
