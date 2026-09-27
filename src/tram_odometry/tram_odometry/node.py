"""Нода ROS 2: подписки, таймер, параметры, публикации. Вся математика — в estimator.py.

Входы: скорости тележек, позиция контроллера, GNSS master и rover (только в окне выставки).
Выходы: /result/velocity, /result/position (nav_msgs/Odometry, base_link в сетке MGRS или,
без GNSS на старте, от старта в frame odom), /result/diagnostics.
Публикация — по таймеру: выход не замирает, когда входы пропадают.
"""
import math

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

from tram_odometry.estimator import Estimator, FilterParams
from tram_odometry.model import ModelParams
from tram_odometry.outputs import StampClock, pose_covariance, twist_covariance, yaw_to_quaternion
from tram_odometry.params import NODE_DEFAULTS
from tram_odometry.positioning import Positioner
from tram_odometry.slip import SlipParams
from tram_odometry.stops import StopFixer, StopParams
from tram_odometry.track import MgrsLocal, RouteTrack


def _stamp(msg, fallback):
    header = getattr(msg, 'header', None)
    if header is None or (header.stamp.sec == 0 and header.stamp.nanosec == 0):
        return fallback
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _to_time(t):
    return divmod(int(round(t * 1e9)), 1000000000)


def _dataclass_params(node, prefix, cls):
    """Параметры dataclass из ROS: имя prefix.<поле>, значение по умолчанию — из класса."""
    values = {}
    for name, f in cls.__dataclass_fields__.items():
        values[name] = node.declare_parameter(f'{prefix}.{name}', f.default).value
    return cls(**values)


class TramOdometryNode(Node):
    def __init__(self):
        super().__init__('tram_odometry')
        self.cfg = {k: self.declare_parameter(k, v).value for k, v in NODE_DEFAULTS.items()}
        self.est = Estimator(model=_dataclass_params(self, 'model', ModelParams),
                             params=_dataclass_params(self, 'filter', FilterParams),
                             slip=_dataclass_params(self, 'slip', SlipParams))
        self.stop_params = _dataclass_params(self, 'stops', StopParams)
        self.clock_out = StampClock(self.cfg['max_extrapolation_s'])
        cfg = self.cfg
        route = None
        if cfg['route_map_file']:
            route = RouteTrack.load(cfg['route_map_file'], MgrsLocal(), cfg['stops_file'] or None)
            self.get_logger().info(f'route map {route.length:.0f} m, {len(route.stops)} stops')
        self.pos = Positioner(route, cfg['gnss_init_window_s'], cfg['output_frame'], cfg['output_lever_m'],
                              cfg['output_dz_m'], cfg['initial_yaw'], cfg['gnss_min_move_m'],
                              corrections=cfg['gnss_corrections'])
        self._start_run()

        best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(VelocitySensor, cfg['front_topic'], lambda m: self._wheel('front', m), best_effort)
        self.create_subscription(VelocitySensor, cfg['rear_topic'], lambda m: self._wheel('rear', m), best_effort)
        self.create_subscription(DriverControllerCommand, cfg['cmd_topic'], self._cmd, best_effort)
        self.create_subscription(NavSatFix, cfg['gnss_topic'], lambda m: self._gnss(m, False), best_effort)
        self.create_subscription(NavSatFix, cfg['rover_topic'], lambda m: self._gnss(m, True), best_effort)
        self.pub_v = self.create_publisher(VelocitySensor, cfg['velocity_topic'], reliable)
        self.pub_odom = self.create_publisher(Odometry, cfg['position_topic'], reliable)
        self.pub_diag = self.create_publisher(DiagnosticArray, cfg['diagnostics_topic'], reliable)
        self.create_timer(1.0 / cfg['rate_hz'], self._publish)
        self.create_timer(1.0 / cfg['diagnostics_hz'], self._diagnostics)
        self.get_logger().info('tram_odometry started')

    # --- входы -----------------------------------------------------------------

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _start_run(self):
        """Новый прогон: новая выставка по GNSS и новые часы выхода."""
        self.pos.start_run()
        self._use_track()
        self.clock_out.reset()
        self.resets_seen = self.est.resets

    def _use_track(self):
        """Если у пути есть уклон и места стоянок (карта) — фильтр их использует."""
        track = self.pos.track
        self.est.grade_at = getattr(track, 'grade_at', None) or (lambda s: 0.0)
        stops = getattr(track, 'stops', None)
        self.stops = StopFixer(stops, getattr(track, 'length', None), self.stop_params) if stops else None

    def _input(self, t):
        if self.est.resets != self.resets_seen:
            self._start_run()
        self.clock_out.input(t, self._now())
        if self.pos.update(t, self.est):
            self._use_track()
            init = self.pos.init
            if init.relative:
                self.get_logger().info('no GNSS at start: relative odometry in frame odom')
            else:
                yaw = init.heading()
                self.get_logger().info(
                    f'GNSS alignment done: heading {"unknown" if yaw is None else f"{math.degrees(yaw):.1f} deg"}, '
                    f'map {"on, %.1f m to track" % self.pos.lock_dist if self.pos.lock_dist is not None else "off"}')

    def _wheel(self, which, msg):
        t = _stamp(msg, self._now())
        self.est.wheel(which, t, float(msg.velocity))
        self._input(t)
        if self.stops is not None:
            self.stops.update(t, self.est)

    def _cmd(self, msg):
        t = _stamp(msg, self._now())
        self.est.set_notch(t, int(msg.position))
        self._input(t)

    def _gnss(self, msg, rover):
        if self.pos.locked and not self.pos.corrections:
            return                                    # после выставки GNSS не используем
        t = _stamp(msg, self._now())
        self.pos.fix(t, msg.latitude, msg.longitude, msg.altitude, msg.status.status, rover)

    # --- выходы ----------------------------------------------------------------

    def _publish(self):
        t = self.clock_out.output(self._now())
        if t is None:
            return
        st = self.est.state_at(t)
        x, y, z, yaw, frame = self.pos.pose(st.s)
        sec, nsec = _to_time(t)

        vel = VelocitySensor()
        vel.header.stamp.sec, vel.header.stamp.nanosec = sec, nsec
        vel.header.frame_id = self.cfg['child_frame_id']
        vel.velocity = float(st.v)
        self.pub_v.publish(vel)

        odom = Odometry()
        odom.header.stamp.sec, odom.header.stamp.nanosec = sec, nsec
        odom.header.frame_id = self.cfg['frame_id'] if frame == 'map' else 'odom'
        odom.child_frame_id = self.cfg['child_frame_id']
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = x, y, z
        q = yaw_to_quaternion(yaw)
        o = odom.pose.pose.orientation
        o.x, o.y, o.z, o.w = q
        cfg = self.cfg
        odom.pose.covariance = pose_covariance(st.var_s, yaw, cfg['var_cross'], cfg['var_z'], cfg['var_yaw'])
        odom.twist.twist.linear.x = float(st.v)
        odom.twist.covariance = twist_covariance(st.var_v)
        self.pub_odom.publish(odom)

    def _diagnostics(self):
        if self.est.t is None:
            return
        st = self.est.state()
        status = DiagnosticStatus()
        status.name = 'tram_odometry'
        status.hardware_id = 'tram'
        if not st.wheels_ok:
            status.level, status.message = DiagnosticStatus.WARN, 'no wheel data: model only'
        elif st.flags.frozen:
            status.level, status.message = DiagnosticStatus.WARN, 'wheel sensor frozen: ignored'
        elif st.slip:
            status.level, status.message = DiagnosticStatus.WARN, 'slip or slide: wheels not trusted'
        else:
            status.level, status.message = DiagnosticStatus.OK, 'ok'
        values = {
            'slip': st.slip, 'slip_ratio': round(st.slip_ratio, 4), 'adhesion_used': round(st.adhesion_used, 4),
            'bogies_inconsistent': st.flags.inconsistent, 'front_accel_limit': st.flags.front_accel,
            'rear_accel_limit': st.flags.rear_accel, 'front_frozen': st.flags.front_frozen,
            'rear_frozen': st.flags.rear_frozen, 'wheels_ok': st.wheels_ok,
            'wheel_scale_k': round(st.k, 5), 'disturbance_accel': round(st.d, 4),
            'sigma_s_m': round(math.sqrt(st.var_s), 3), 's_m': round(st.s, 2),
            'gnss_heading_known': self.pos.init.heading_known, 'relative_mode': self.pos.relative,
            'resets': self.est.resets,
            'stop_fixes': self.stops.fixes if self.stops else 0,
        }
        status.values = [KeyValue(key=k, value=str(v)) for k, v in values.items()]
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [status]
        self.pub_diag.publish(msg)


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
