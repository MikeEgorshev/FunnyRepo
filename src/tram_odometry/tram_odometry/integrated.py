"""Causal integrated estimator with GNSS corrections isolated from speed."""
import copy
import math

from .estimator import Params, TramEstimator, _mul, _transpose
from .position_correction import PositionCorrection


class IntegratedEstimator(TramEstimator):
    def __init__(self, route_map, model, stops=(), params=None):
        p = copy.copy(params or Params())
        p.primary_sync = False
        # Publish the estimated velocity AT the stamp, not a judge-tuned delay.
        p.output_v_delay_s = 0.0
        p.output_frame, p.output_origin = 'utm_local', 'frame'
        p.output_lever_m, p.output_dz_m = 9.873, -3.0
        super().__init__(route_map, model, stops, p)
        self.init_deadline = None
        self.init_locked = False
        self.channel_stamp = {}
        self.adaptation_sample = None
        self.route_alternatives = []
        self.preview = None
        self.observed_until = None
        self.position_correction = PositionCorrection()
        self.travel = 0.0

    def _initial_window(self, stamp):
        if not math.isfinite(stamp) or self.init_locked:
            return False
        if self.init_deadline is None:
            self.init_deadline = stamp + self.p.init_window_s
        if max(stamp, self.t if self.t is not None else stamp) > self.init_deadline:
            self.init_locked = True
        return not self.init_locked

    def on_gnss(self, stamp, lat, lon, alt):
        if not all(math.isfinite(x) for x in (stamp, lat, lon, alt)):
            return None
        if not self._initial_window(stamp):
            return self._gnss_correction(stamp, lat, lon, alt)
        self.preview = None
        return super().on_gnss(stamp, lat, lon, alt)

    def _gnss_correction(self, stamp, lat, lon, alt):
        if self.init_deadline is not None and stamp > self.init_deadline:
            self.position_correction.update(self, stamp, lat, lon, alt)
        return None

    def on_gnss_rover(self, stamp, lat, lon, alt):
        if not all(math.isfinite(x) for x in (stamp, lat, lon, alt)):
            return not self.init_locked
        if not self._initial_window(stamp):
            return False
        return super().on_gnss_rover(stamp, lat, lon, alt)

    def on_primary(self, *args):
        return None

    def _fresh(self, channel, stamp):
        if not math.isfinite(stamp) or stamp <= self.channel_stamp.get(channel, -math.inf):
            return False
        self.channel_stamp[channel] = stamp
        self.preview = None
        self._initial_window(stamp)
        return True

    def on_cmd(self, stamp, position):
        if not math.isfinite(position) or not self._fresh('cmd', stamp):
            return None
        out = super().on_cmd(stamp, position)
        self.observed_until = max(stamp, self.observed_until or stamp)
        return out

    def on_wheel(self, stamp, front, kmh):
        if not math.isfinite(kmh) or not self.p.wheel_min_kmh <= kmh <= self.p.wheel_max_kmh:
            self.slip = True
            return None
        if not self._fresh(front, stamp):
            return None
        out = super().on_wheel(stamp, front, kmh)
        self.observed_until = max(stamp, self.observed_until or stamp)
        # Front samples trigger adaptation; never reuse the last EKF v as v_prev.
        other = self.last_wheel[False]
        if front and out is not None:
            z = kmh / self.p.wheel_kmh_per_mps / (1.0 + self.c)
            good = (not out['slip'] and not out['stuck'] and other is not None
                    and 0 <= stamp - other[0] <= self.p.bogie_pair_s
                    and abs(z - other[1]) < self.p.bogie_tol_abs)
            previous = self.adaptation_sample
            self.adaptation_sample = (stamp, z) if good else None
            if good and previous is not None and hasattr(self.model, 'adapt'):
                dt = stamp - previous[0]
                self.model.adapt(self.model.command_at(stamp), self.v,
                                 (z - previous[1]) / dt, self.d, dt)
        return out

    def _speed_update(self, stamp, z, gate=True, resync_s=None):
        # Soft trust reduction for moderate innovation; retain hard slip gates.
        sigma = self.p.sigma_wheel
        variance = self.P[1][1] + sigma * sigma + (self.p.sigma_wheel_rel * z) ** 2
        ratio = abs(z - self.v) / math.sqrt(max(variance, 1e-12))
        if gate and self.v_init and 1.0 < ratio < 3.0:
            self.p.sigma_wheel *= min(2.0, ratio)
        try:
            return super()._speed_update(stamp, z, gate, resync_s)
        finally:
            self.p.sigma_wheel = sigma

    def _enter_stub(self):
        # Wheel distance alone cannot distinguish switch branches. Preserve
        # alternatives rather than teleporting to a stub after an arbitrary stop.
        self.route_alternatives = []
        for spur, fork, length in self._stub_forks():
            delta = (self.s - fork) % self.map.length
            if 0 <= delta <= length + self.p.stub_margin_m:
                self.route_alternatives.append((spur, fork, length))
        return False

    def _snap(self):
        # An ambiguous nearby stop is not an absolute position measurement.
        if self.mode == 'relative' or not self.ready:
            return
        candidates = [s for s, sigma in self.stops
                      if abs((s - self.s + self.map.length / 2) % self.map.length
                             - self.map.length / 2)
                      <= min(self.p.stop_gate_max_m,
                             self.p.stop_gate_sigma * math.sqrt(self.P[0][0] + sigma * sigma))]
        if len(candidates) == 1:
            super()._snap()
        else:
            self._enter_stub()

    def _stop_logic(self, stamp):
        # A model stop or one locked wheel is not sufficient for ZUPT/snap.
        stopped = all(w is not None and 0 <= stamp-w[0] <= self.p.bogie_pair_s
                      and abs(w[1]) < self.p.stop_v for w in self.last_wheel.values())
        if not stopped or any(self.bad.values()) or self.model.command_at(stamp) > 0.1:
            self.stop_since, self.snapped = None, False
            return
        if (self.stop_since is not None and stamp-self.stop_since >= self.p.stop_confirm_s
                and self.v < 0.5):
            self.v = 0.0
            # Confirmed stationarity constrains velocity, not physical slope.
            for i in range(4):
                self.P[1][i] = self.P[i][1] = 0.0
            self.P[1][1] = 1e-4
        super()._stop_logic(stamp)

    def predict_output(self, stamp):
        if not math.isfinite(stamp) or self.t is None or stamp <= self.t:
            return None
        self._initial_window(stamp)
        self._check_no_gnss(stamp)
        # Cached speculative state: constant work per tick during long outages.
        # Shared map/model are read-only on this prediction path.
        if self.preview is None or stamp < self.preview.t:
            self.preview = copy.copy(self)
            self.preview.P = [row[:] for row in self.P]
            self.preview.v_hist = self.v_hist.copy()
        self.preview._advance(stamp)
        return self.preview._output(stamp)

    def _advance(self, stamp):
        gap = self.p.max_gap_s
        self.p.max_gap_s = math.inf
        try:
            return super()._advance(stamp)
        finally:
            self.p.max_gap_s = gap

    def _predict_step(self, h):
        previous_s = self.s
        # Long prediction capped to constant speed once all inputs are stale.
        last = self.observed_until if self.observed_until is not None else self.t
        if self.t - last > 2.0:
            self.s += self.v * h
            self.d_since_fix += self.v * h
            F = [[1., h, 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
            self.P = _mul(_mul(F, self.P), _transpose(F))
            self.P[0][0] += h**3/3
            self.P[0][1] += h*h/2
            self.P[1][0] += h*h/2
            self.P[1][1] += h
            self.t += h
        else:
            super()._predict_step(h)
        # Continuous model distance excludes EKF map/stop position jumps.
        self.travel += self.s - previous_s

    def _output(self, stamp):
        out = super()._output(stamp)
        if out is not None:
            if self.mode != 'relative':
                # Stored map samples describe master antenna, not base_link.
                # Apply rigid TF in Cartesian coordinates, not an arc-length shift.
                x, y, z, yaw, facing = self.position_correction.pose(
                    self.map, self.s, self.facing, self.travel)
                heading = yaw if facing > 0 else math.atan2(-math.sin(yaw), -math.cos(yaw))
                out.update(x=x + self.p.output_lever_m * math.cos(heading),
                           y=y + self.p.output_lever_m * math.sin(heading),
                           z=z + self.p.output_dz_m, yaw=heading)
                correction = self.position_correction
                if correction.accepted:
                    out['var_s'] = max(out['var_s'], correction.variance
                                       + self.p.sigma_u**2 * abs(self.travel-correction.anchor))
            out['traction_gain'] = getattr(self.model, 'traction_gain', 1.0)
            alternatives = []
            for spur, fork, length in self.route_alternatives:
                delta = (self.s-fork) % self.map.length
                if delta <= length + self.p.stub_margin_m:
                    x, y, _, yaw = self.map._spur_pose(spur, spur['d'][-1]-min(delta, length))
                    alternatives.append((x-9.873*math.cos(yaw), y-9.873*math.sin(yaw)))
            out['route_ambiguous'] = bool(alternatives)
            out['var_cross'] = max([0.25] + [(out['x']-x)**2 + (out['y']-y)**2
                                           for x, y in alternatives])
        return out
