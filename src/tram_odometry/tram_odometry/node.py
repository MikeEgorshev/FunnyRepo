"""Нода ROS 2: подписки, таймер, параметры, публикации. Вся математика — в estimator.py.

Входы: скорости тележек, позиция контроллера, GNSS master fix (только в окне выставки).
Выходы: /result/velocity, /result/position (nav_msgs/Odometry), /result/diagnostics.
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
from tram_odometry.slip import SlipParams
from tram_odometry.track import GnssInit


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
        self.clock_out = StampClock(self.cfg['max_extrapolation_s'])
        self._start_run()

        best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        cfg = self.cfg
        self.create_subscription(VelocitySensor, cfg['front_topic'], lambda m: self._wheel('front', m), best_effort)
        self.create_subscription(VelocitySensor, cfg['rear_topic'], lambda m: self._wheel('rear', m), best_effort)
        self.create_subscription(DriverControllerCommand, cfg['cmd_topic'], self._cmd, best_effort)
        self.create_subscription(NavSatFix, cfg['gnss_topic'], self._gnss, best_effort)
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
        self.init = GnssInit(self.cfg['gnss_init_window_s'], self.cfg['gnss_min_move_m'])
        self.track = self.init.track(self.cfg['initial_yaw'])
        self.clock_out.reset()
        self.resets_seen = self.est.resets

    def _input(self, t):
        if self.est.resets != self.resets_seen:
            self._start_run()
        self.init.start(t)
        self.clock_out.input(t, self._now())
        if not self.init.done and self.init.expired(t):
            self.track = self.init.track(self.cfg['initial_yaw'])
            self.get_logger().info(
                f'GNSS alignment done: heading {"known" if self.init.heading_known else "unknown"}, '
                f'yaw {math.degrees(self.track.yaw):.1f} deg')

    def _wheel(self, which, msg):
        t = _stamp(msg, self._now())
        self.est.wheel(which, t, float(msg.velocity))
        self._input(t)

    def _cmd(self, msg):
        t = _stamp(msg, self._now())
        self.est.set_notch(t, int(msg.position))
        self._input(t)

    def _gnss(self, msg):
        if self.init.done:
            return                                    # после выставки GNSS не используем
        t = _stamp(msg, self._now())
        self.init.start(t)
        self.init.fix(t, msg.latitude, msg.longitude, msg.altitude, msg.status.status)
        if self.init.frame is not None:
            self.track = self.init.track(self.cfg['initial_yaw'])

    # --- выходы ----------------------------------------------------------------

    def _publish(self):
        t = self.clock_out.output(self._now())
        if t is None:
            return
        st = self.est.state_at(t)
        x, y, z, yaw = self.track.pose(st.s)
        sec, nsec = _to_time(t)

        vel = VelocitySensor()
        vel.header.stamp.sec, vel.header.stamp.nanosec = sec, nsec
        vel.header.frame_id = self.cfg['child_frame_id']
        vel.velocity = float(st.v)
        self.pub_v.publish(vel)

        odom = Odometry()
        odom.header.stamp.sec, odom.header.stamp.nanosec = sec, nsec
        odom.header.frame_id = self.cfg['frame_id']
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
        elif st.slip:
            status.level, status.message = DiagnosticStatus.WARN, 'slip or slide: wheels not trusted'
        else:
            status.level, status.message = DiagnosticStatus.OK, 'ok'
        values = {
            'slip': st.slip, 'slip_ratio': round(st.slip_ratio, 4), 'adhesion_used': round(st.adhesion_used, 4),
            'bogies_inconsistent': st.flags.inconsistent, 'front_accel_limit': st.flags.front_accel,
            'rear_accel_limit': st.flags.rear_accel, 'wheels_ok': st.wheels_ok,
            'wheel_scale_k': round(st.k, 5), 'disturbance_accel': round(st.d, 4),
            'sigma_s_m': round(math.sqrt(st.var_s), 3), 's_m': round(st.s, 2),
            'gnss_heading_known': self.init.heading_known, 'resets': self.est.resets,
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
