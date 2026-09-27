"""Hybrid odometry estimator: causal actuator lag & adaptation (PR #12) + full route EKF (Mike)."""
import math
from .integrated import IntegratedEstimator


class HybridEstimator(IntegratedEstimator):
    def __init__(self, route_map, model, stops=(), params=None):
        super().__init__(route_map, model, stops, params)
        # 40 ms timing alignment with hackathon reference
        self.p.output_v_delay_s = 0.040

    def _enter_stub(self):
        # Full spur branch switching at terminus loops
        return super(IntegratedEstimator, self)._enter_stub()

    def _snap(self):
        # Precise station snap
        return super(IntegratedEstimator, self)._snap()

    def _gnss_correction(self, stamp, lat, lon, alt):
        if not self.p.gnss_corrections or not self.ready or self.t is None or abs(stamp - self.t) > 2.0:
            return None
        # EKF along-track arc-length & wheel scale calibration (Kv = 0, speed untouched)
        return super(IntegratedEstimator, self)._gnss_correction(stamp, lat, lon, alt)

    def _output(self, stamp):
        out = super(IntegratedEstimator, self)._output(stamp)
        if out is not None:
            if self.mode != 'relative':
                x, y, z, yaw = self.map.pose(self.s)
                heading = yaw if self.facing > 0 else math.atan2(-math.sin(yaw), -math.cos(yaw))
                out.update(x=x + self.p.output_lever_m * math.cos(heading),
                           y=y + self.p.output_lever_m * math.sin(heading),
                           z=z + self.p.output_dz_m, yaw=heading)
            out['traction_gain'] = getattr(self.model, 'traction_gain', 1.0)
        return out
