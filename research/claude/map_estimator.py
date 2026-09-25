"""Базовый оценщик + привязка к местам регулярных стоянок.

Неопределённость s растёт с пройденным путём: sigma_s = sqrt(sigma0² + (REL_ERR · d)²), где d —
путь после последней привязки. Когда колёса показывают стоянку дольше STOP_CONFIRM_S, ищем
опорную точку стоянки (route_stops.csv) в пределах GATE_SIGMA · sqrt(sigma_s² + sigma_st²) и делаем
одномерную калмановскую поправку s. Одна поправка на одну стоянку.
"""
from pathlib import Path

from baseline import BaselineEstimator

REL_ERR = 0.004
STOP_V = 0.05
STOP_CONFIRM_S = 2.0
GATE_SIGMA = 3.0
GATE_MAX_M = 60.0


def load_stops(path):
    stops = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line[:1].isdigit():
                s, sigma, _ = line.split(',')
                stops.append((float(s), float(sigma)))
    return stops


class MapStopsEstimator(BaselineEstimator):
    def __init__(self, route_map_path, **kw):
        super().__init__(route_map_path, **kw)
        path = Path(route_map_path).with_name('route_stops.csv')
        self.stops = load_stops(path) if path.exists() else []
        self.sigma0_sq = 4.0
        self.d_since_fix = 0.0
        self.stop_since = None
        self.snapped_this_stop = False
        self.snaps = []

    def on_gnss(self, stamp, lat, lon, alt):
        r = super().on_gnss(stamp, lat, lon, alt)
        if r is not None:
            self.sigma0_sq, self.d_since_fix = 4.0, 0.0
        return r

    def _advance(self, stamp):
        s_before = self.s
        super()._advance(stamp)
        if s_before is not None and self.s is not None:
            self.d_since_fix += self.s - s_before

    def on_wheel(self, stamp, front, kmh):
        r = super().on_wheel(stamp, front, kmh)
        if self.s is None:
            return r
        if self.v < STOP_V:
            if self.stop_since is None:
                self.stop_since = stamp
            elif not self.snapped_this_stop and stamp - self.stop_since >= STOP_CONFIRM_S:
                self.snapped_this_stop = True
                self._snap(stamp)
                return self._output(stamp)
        else:
            self.stop_since = None
            self.snapped_this_stop = False
        return r

    def _snap(self, stamp):
        sp = self.map.active_spur
        if sp is not None and self.s < sp['s_join']:
            return
        L = self.map.length
        var_s = self.sigma0_sq + (REL_ERR * self.d_since_fix) ** 2
        best = None
        for s_st, sig in self.stops:
            nu = (s_st - self.s + L / 2) % L - L / 2
            S = var_s + sig * sig
            if abs(nu) <= min(GATE_SIGMA * S ** 0.5, GATE_MAX_M) and (best is None or abs(nu) < abs(best[0])):
                best = (nu, S)
        if best is None:
            return
        nu, S = best
        k = var_s / S
        self.s += k * nu
        self.sigma0_sq = (1 - k) * var_s
        self.d_since_fix = 0.0
        self.snaps.append((stamp, nu, k))
