"""ROS 2 нода резервной одометрии: входы /vehicle/*, выход /result/velocity и /result/position.

Только ROS-обвязка: вся математика — в estimator.py (TramEstimator). На каждое входное сообщение
публикуется оценка с header.stamp этого входа (так её сопоставляет судья). Если все входы молчат
дольше keepalive_s, публикуется прогноз по модели с меткой «последний вход + прошедшее время»:
выход не реже 20 Гц даже при пропуске всех входов. GNSS принимается только в окне выставки после первой точки.
"""
import math
import os
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

from .estimator import Params, TramEstimator, load_stops
from .model import TractionModel
from .route_map import RouteMap

WHEEL_KMH_PER_MPS = {'30618': 3.5953, '30639': 3.6106}


class TramOdometryNode(Node):
    def __init__(self):
        super().__init__('tram_odometry')
        share = get_package_share_directory('tram_odometry')
        p = Params()
        declared = {
            'wheel_kmh_per_mps': p.wheel_kmh_per_mps,
            'vehicle_id': '',
            'route_map_file': os.path.join(share, 'maps', 'route.csv'),
            'spurs_file': os.path.join(share, 'maps', 'route_spurs.csv'),
            'stops_file': os.path.join(share, 'maps', 'route_stops.csv'),
            'traction_table_file': os.path.join(share, 'config', 'traction_table.csv'),
            'traction_delay_s': 0.4,
            'init_window_s': p.init_window_s,
            'map_frame': 'map',
            'base_frame': 'base_link',
            'keepalive_s': 0.04,       # входы молчат дольше — публикуем прогноз по модели
            'keepalive_max_s': 2.0,    # и не дальше этого от последнего входа
        }
        for name, default in declared.items():
            self.declare_parameter(name, default)
        # параметры фильтра из Params можно переопределить одноимёнными параметрами ноды
        for name in [k for k in vars(Params) if not k.startswith('_') and k not in declared]:
            self.declare_parameter(name, getattr(Params, name))
            setattr(p, name, self.get_parameter(name).value)
        get = lambda name: self.get_parameter(name).value  # noqa: E731
        p.init_window_s = get('init_window_s')
        vehicle = get('vehicle_id')
        p.wheel_kmh_per_mps = WHEEL_KMH_PER_MPS.get(vehicle, get('wheel_kmh_per_mps'))
        route = RouteMap.load(get('route_map_file'))
        if os.path.exists(get('spurs_file')):
            route.load_spurs(get('spurs_file'))
        stops = load_stops(get('stops_file')) if os.path.exists(get('stops_file')) else []
        model = TractionModel.load(get('traction_table_file'), delay_s=get('traction_delay_s'))
        self.est = TramEstimator(route, model, stops, p)
        self.map_frame, self.base_frame = get('map_frame'), get('base_frame')
        self.keepalive_s, self.keepalive_max_s = get('keepalive_s'), get('keepalive_max_s')
        self.last_in = None          # (метка последнего входа, время его прихода по monotonic)
        self.last_pub = None         # время последней публикации по monotonic

        qos_in = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(VelocitySensor, '/vehicle/front_bogie_velocity',
                                 lambda m: self._wheel(m, True), qos_in)
        self.create_subscription(VelocitySensor, '/vehicle/rear_bogie_velocity',
                                 lambda m: self._wheel(m, False), qos_in)
        self.create_subscription(DriverControllerCommand, '/vehicle/driver_position_cmd', self._cmd, qos_in)
        self.gnss_sub = self.create_subscription(NavSatFix, '/sensing/gnss/master/fix', self._gnss, qos_in)
        self.rover_sub = self.create_subscription(NavSatFix, '/sensing/gnss/rover/fix', self._rover, qos_in)
        self.pub_v = self.create_publisher(VelocitySensor, '/result/velocity', 10)
        self.pub_p = self.create_publisher(Odometry, '/result/position', 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, '/result/diagnostics', 10)
        self.proc_ms = []
        self.last_diag = 0.0
        self.last_out = None
        self.create_timer(0.5, self._diagnostics)
        self.create_timer(self.keepalive_s / 2, self._keepalive)
        self.get_logger().info(f'tram_odometry: k={p.wheel_kmh_per_mps:.4f}, карта {route.s[-1]:.0f} м, '
                               f'отводов {len(route.spurs)}, опорных стоянок {len(stops)}')

    # --- входы -----------------------------------------------------------------------------
    @staticmethod
    def _stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _input(self, stamp):
        if self.last_in is None or stamp >= self.last_in[0]:
            self.last_in = (stamp, time.monotonic())

    def _wheel(self, msg, front):
        t0 = time.perf_counter()
        self._input(self._stamp(msg))
        self._publish(self.est.on_wheel(self._stamp(msg), front, msg.velocity), msg.header.stamp, t0)

    def _cmd(self, msg):
        t0 = time.perf_counter()
        self._input(self._stamp(msg))
        self._publish(self.est.on_cmd(self._stamp(msg), msg.position), msg.header.stamp, t0)

    def _rover(self, msg):
        """Вторая антенна — только курс на стоянке для выставки; после окна выставки не нужна."""
        if msg.status.status < 0 or math.isnan(msg.latitude) or self.rover_sub is None:
            return
        if not self.est.on_gnss_rover(self._stamp(msg), msg.latitude, msg.longitude, msg.altitude):
            self.destroy_subscription(self.rover_sub)
            self.rover_sub = None

    def _keepalive(self):
        """Входы молчат — публикуем прогноз, чтобы выход не замирал."""
        if self.last_in is None or self.last_pub is None:
            return
        now = time.monotonic()
        elapsed = now - self.last_in[1]
        if now - self.last_pub < self.keepalive_s or elapsed > self.keepalive_max_s:
            return
        t = self.last_in[0] + elapsed
        sec, nsec = divmod(int(round(t * 1e9)), 1000000000)
        t0 = time.perf_counter()
        self._publish(self.est.predict_output(t), Time(sec=sec, nanosec=nsec), t0)

    def _gnss(self, msg):
        if msg.status.status < 0 or math.isnan(msg.latitude):
            return
        t0 = time.perf_counter()
        out = self.est.on_gnss(self._stamp(msg), msg.latitude, msg.longitude, msg.altitude)
        if out is None and self.est.ready:
            # окно выставки закончилось — GNSS больше не нужен и в контуре не участвует
            self.destroy_subscription(self.gnss_sub)
            self.get_logger().info('выставка по GNSS завершена, дальше только колёса и контроллер')
            return
        self._publish(out, msg.header.stamp, t0)

    # --- выходы ------------------------------------------------------------------------------
    def _publish(self, out, stamp, t0):
        if out is None:
            return
        v = VelocitySensor()
        v.header.stamp = stamp
        v.header.frame_id = self.base_frame
        v.velocity = float(out['v'])
        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = self.map_frame if out['frame'] == 'map' else 'odom'
        o.child_frame_id = self.base_frame
        o.pose.pose.position.x = float(out['x'])
        o.pose.pose.position.y = float(out['y'])
        o.pose.pose.position.z = float(out['z'])
        o.pose.pose.orientation.z = math.sin(out['yaw'] / 2)
        o.pose.pose.orientation.w = math.cos(out['yaw'] / 2)
        var_s = float(out['var_s'])
        cov = [0.0] * 36
        cov[0] = cov[7] = var_s + 0.25   # x, y: неопределённость вдоль пути + точность карты
        cov[14] = 1.0                    # z
        cov[21] = cov[28] = 1e3          # крен, тангаж не оцениваем
        cov[35] = 0.01                   # курс по карте
        o.pose.covariance = cov
        o.twist.twist.linear.x = float(out['v'])
        tcov = [0.0] * 36
        tcov[0] = float(out['var_v'])
        tcov[7] = tcov[14] = 1e-4
        tcov[21] = tcov[28] = tcov[35] = 1e3
        o.twist.covariance = tcov
        self.pub_v.publish(v)
        self.pub_p.publish(o)
        self.last_out = out
        self.last_pub = time.monotonic()
        self.proc_ms.append((time.perf_counter() - t0) * 1e3)

    def _diagnostics(self):
        out = self.last_out
        if out is None:
            return
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus()
        st.name = 'tram_odometry'
        st.hardware_id = 'backup_odometry'
        if out['dropout']:
            st.level, st.message = DiagnosticStatus.WARN, 'нет данных колёс: работает только модель'
        elif out['slip']:
            st.level, st.message = DiagnosticStatus.WARN, 'проскальзывание/юз: колесо исключено'
        else:
            st.level, st.message = DiagnosticStatus.OK, 'ok'
        proc = sorted(self.proc_ms[-500:]) or [0.0]
        st.values = [KeyValue(key=k, value=str(v)) for k, v in (
            ('slip', out['slip']), ('wheel_dropout', out['dropout']),
            ('speed_mps', round(out['v'], 3)), ('s_m', round(out['s'], 1)),
            ('sigma_s_m', round(math.sqrt(max(out['var_s'], 0.0)), 2)),
            ('wheel_scale_correction', round(out['scale'], 5)),
            ('proc_ms_p50', round(proc[len(proc) // 2], 3)),
            ('proc_ms_p99', round(proc[int(len(proc) * 0.99) - 1 if len(proc) > 1 else 0], 3)))]
        arr.status = [st]
        self.pub_diag.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = TramOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
