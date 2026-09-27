"""Competition entry point: position-only GNSS corrections, causal heartbeat."""
import math
import time
from collections import deque

import rclpy
from builtin_interfaces.msg import Time

from .node import TramOdometryNode as LegacyNode
from .integrated import IntegratedEstimator
from .integrated_model import IntegratedModel


class TramOdometryNode(LegacyNode):
    estimator_type = IntegratedEstimator
    model_type = IntegratedModel

    def __init__(self):
        self.output_time = None
        self.output_stamp = -math.inf
        super().__init__()
        self.proc_ms = deque(maxlen=500)

    def _now(self):
        if self.get_parameter('use_sim_time').value:
            return self.get_clock().now().nanoseconds * 1e-9
        return time.monotonic()

    def _input(self, stamp):
        if math.isfinite(stamp) and (self.last_in is None or stamp > self.last_in[0]):
            self.last_in = (stamp, self._now())

    def _gnss(self, msg):
        if msg.status.status < 0:
            return
        out = self.est.on_gnss(self._stamp(msg), msg.latitude, msg.longitude, msg.altitude)
        if self.est.init_locked and not self.est.p.gnss_corrections and self.gnss_sub is not None:
            self.destroy_subscription(self.gnss_sub)
            self.gnss_sub = None
        self._publish(out, msg.header.stamp, time.perf_counter())

    def _keepalive(self):
        if self.last_in is None:
            return
        now = self._now()
        if self.output_time is not None and now - self.output_time < self.keepalive_s:
            return
        stamp = self.last_in[0] + max(0.0, now - self.last_in[1])
        sec, nsec = divmod(round(stamp * 1e9), 1_000_000_000)
        self._publish(self.est.predict_output(stamp), Time(sec=sec, nanosec=nsec), time.perf_counter())

    def _publish(self, out, stamp, t0):
        t = stamp.sec + stamp.nanosec * 1e-9
        if out is None or t <= self.output_stamp:
            return
        self.output_stamp, self.output_time = t, self._now()
        # Along-track covariance rotated into XY, plus cross-track uncertainty.
        c, s = math.cos(out['yaw']), math.sin(out['yaw'])
        cross = out.get('var_cross', 0.25)
        self.position_covariance = (out['var_s']*c*c + cross*s*s,
                                    (out['var_s']-cross)*c*s,
                                    out['var_s']*s*s + cross*c*c)
        super()._publish(out, stamp, t0)

    def _diagnostics(self):
        # Legacy diagnostic formatter slices a list; keep long-run storage bounded.
        samples = self.proc_ms
        self.proc_ms = list(samples)
        try:
            super()._diagnostics()
        finally:
            self.proc_ms = samples


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
