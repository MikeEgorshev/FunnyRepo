"""Положение выходной точки: выставка по GNSS, привязка к карте, переход к base_link.

Пока идёт окно выставки — запасная прямая от антенны master по курсу GNSS. В конце окна:
- есть карта — s фильтра ставится проекцией точки старта на карту, дальше координаты
  берутся с карты (и уклон, и места стоянок для привязок);
- карты нет — остаётся прямая;
- GNSS не было — относительная одометрия от старта (frame odom, x = пройденный путь).

Выходная точка — base_link: ось передней тележки на уровне рельса, на lever_m впереди
антенны master по пути и на dz_m по высоте (tf антенн от организаторов, QA 25.09).
"""
from tram_odometry.track import GnssInit, StraightTrack


class Positioner:
    def __init__(self, route=None, window_s=5.0, frame='mgrs', lever_m=9.873, dz_m=-3.0,
                 fallback_yaw=0.0, min_move_m=3.0):
        if route is not None and frame != 'mgrs':
            raise ValueError('карта задана в сетке MGRS: выход с картой — только frame=mgrs')
        self.route = route
        self.window_s, self.frame, self.min_move_m = window_s, frame, min_move_m
        self.lever_m, self.dz_m, self.fallback_yaw = lever_m, dz_m, fallback_yaw
        self.start_run()

    def start_run(self):
        self.init = GnssInit(self.window_s, self.frame, self.min_move_m)
        self.track = StraightTrack()
        self.locked = False
        self.lock_dist = None           # расстояние от точки старта до карты, м

    def fix(self, t, lat, lon, alt, status=0, rover=False):
        if self.locked:
            return
        self.init.fix(t, lat, lon, alt, status, rover)
        if not self.init.relative:
            self.track = self.init.straight_track(self.fallback_yaw)

    def update(self, t, estimator):
        """Вызывать на каждом входе. True — в этот момент закончилась выставка."""
        self.init.start(t)
        if self.locked or not self.init.expired(t):
            return False
        self.locked = True
        if self.init.relative:
            self.track = StraightTrack()
        elif self.route is not None:
            x, y, _ = self.init.start_point()
            s0, dist = self.route.locate(x, y, self.init.heading())
            self.lock_dist = dist
            if self.init.still:              # стоим с начала окна: к проекции добавляем то, что проехали
                estimator.set_position(s0 + estimator.state().s, max(dist, 1.0))
            else:                            # едем: последний фикс — это «сейчас»
                estimator.set_position(s0, max(dist, 3.0))
            self.track = self.route
        elif not self.init.still:            # без карты и на ходу: прямая идёт от последнего фикса,
            estimator.set_position(0.0, 3.0)  # значит и дистанция от него
        return True

    @property
    def relative(self):
        return self.init.relative

    def pose(self, s):
        """(x, y, z, yaw, frame): frame — 'map' или 'odom' (без GNSS — от старта)."""
        if self.init.relative:
            x, y, _, yaw = self.track.pose(s)
            return x, y, 0.0, yaw, 'odom'
        x, y, z, yaw = self.track.pose(s + self.lever_m)
        return x, y, z + self.dz_m, yaw, 'map'
