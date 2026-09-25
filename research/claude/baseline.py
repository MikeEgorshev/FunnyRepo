"""Базовый оценщик для сравнения: скорость колёс / k, интегрирование вдоль карты.

Без модели тяги и без защиты от проскальзывания: это нижняя планка, которую должен побить
настоящий фильтр (задача 5). Интерфейс тот же, что будет у estimator.py в пакете.
"""
import math

from tram_odometry.geo import Enu
from tram_odometry.route_map import RouteMap

KMH_PER_MPS = {'default': 3.5966, '30618': 3.5953, '30639': 3.6106}


class BaselineEstimator:
    def __init__(self, route_map_path, vehicle_id='default', init_window_s=5.0, wheel_fresh_s=0.5):
        self.map = RouteMap.load(route_map_path)
        self.k = KMH_PER_MPS.get(vehicle_id, KMH_PER_MPS['default'])
        self.init_window_s = init_window_s
        self.wheel_fresh_s = wheel_fresh_s
        self.enu = None
        self.t_first_fix = None
        self.first_xy = None
        self.s = None
        self.v = 0.0
        self.t = None
        self.wheels = {True: None, False: None}  # front -> (stamp, м/с)

    # --- входы -----------------------------------------------------------------------------
    def on_gnss(self, stamp, lat, lon, alt):
        """GNSS принимается только в окне выставки после первой точки."""
        if self.enu is None:
            self.enu = Enu(lat, lon, alt)
            self.map.to_frame(self.enu)
            self.t_first_fix = stamp
        if stamp - self.t_first_fix > self.init_window_s:
            return None
        x, y, _ = self.enu.forward(lat, lon, alt)
        if self.first_xy is None:
            self.first_xy = (x, y)
        dx, dy = x - self.first_xy[0], y - self.first_xy[1]
        yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 3.0 else None
        self.s, _ = self.map.locate(x, y, yaw)
        self.t = stamp
        return self._output(stamp)

    def on_wheel(self, stamp, front, kmh):
        self._advance(stamp)
        self.wheels[front] = (stamp, kmh / self.k)
        fresh = [w[1] for w in self.wheels.values() if w and stamp - w[0] <= self.wheel_fresh_s]
        if fresh:
            self.v = max(0.0, sum(fresh) / len(fresh))
        return self._output(stamp)

    def on_cmd(self, stamp, position):
        self._advance(stamp)
        return self._output(stamp)

    # --- внутреннее --------------------------------------------------------------------------
    def _advance(self, stamp):
        if self.s is None or self.t is None:
            self.t = stamp if self.t is None else self.t
            return
        dt = stamp - self.t
        if dt <= 0:
            return
        self.s += self.v * min(dt, 1.0)
        self.t = stamp

    def _output(self, stamp):
        if self.s is None or stamp < (self.t or stamp):
            return None
        x, y, z, _ = self.map.pose(self.s)
        return stamp, self.v, x, y, z, self.s
