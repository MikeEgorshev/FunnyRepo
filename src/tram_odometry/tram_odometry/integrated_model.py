"""Train-calibrated acceleration with causal actuator lag (Gemini model idea).

The table already contains NET acceleration: do not subtract Davis drag twice.
Effective traction gain is a bounded nuisance parameter, not measured mass.
"""
import math
from collections import deque

from .model import TractionModel


class IntegratedModel(TractionModel):
    tau_s = 0.2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filtered = deque(maxlen=400)
        self.traction_gain = 1.0

    def push_command(self, stamp, u):
        if self._cmd and stamp <= self._cmd[-1][0]:
            return
        effective = self.command_at(stamp + self.delay_s)
        super().push_command(stamp, u)
        self.filtered.append((stamp, self._cmd[-1][1], effective))

    def command_at(self, t):
        t -= self.delay_s
        for stamp, raw, initial in reversed(self.filtered):
            if stamp <= t:
                return raw + (initial - raw) * math.exp(-(t - stamp) / self.tau_s)
        return 0.0

    def accel(self, u, v):
        u = max(-15.0, min(15.0, u))
        lo, hi = math.floor(u), math.ceil(u)
        a = super().accel(lo, v)
        a += (u - lo) * (super().accel(hi, v) - a)
        return a * self.traction_gain if u > 0 else a

    def adapt(self, u, v, acceleration, disturbance, dt):
        """Only called with two fresh, agreeing, healthy wheel observations."""
        if not (u >= 5 and v > 1.5 and 0.02 <= dt <= 0.3):
            return
        nominal = self.accel(u, v) / self.traction_gain
        measured = acceleration - disturbance
        if nominal < 0.2 or not 0.2 < measured < 1.2:
            return
        target = max(0.8, min(1.2, measured / nominal))
        # Slow enough not to learn a brief slip or a wheel quantization step.
        self.traction_gain += (1.0 - math.exp(-dt / 30.0)) * (target - self.traction_gain)
