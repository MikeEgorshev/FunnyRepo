"""Положение выходной точки: выставка по GNSS, привязка к карте, переход к base_link.

Пока идёт окно выставки — запасная прямая от антенны master по курсу GNSS. В конце окна:
- есть карта — s фильтра ставится проекцией точки старта на карту, дальше координаты
  берутся с карты (и уклон, и места стоянок для привязок);
- карты нет — остаётся прямая;
- GNSS не было — относительная одометрия от старта (frame odom, x = пройденный путь).

Выходная точка — base_link: ось передней тележки на уровне рельса, на lever_m впереди
антенны master по пути и на dz_m по высоте (tf антенн от организаторов, QA 25.09).

Коррекция по GNSS после выставки (corrections=True, по умолчанию выключена): в проверочных
прогонах фиксы появляются короткими всплесками раз в 2–3 минуты; фикс master проецируется
на карту и уходит в фильтр привязкой дистанции (не чаще раза в corr_period_s). Описание задачи
разрешает GNSS «для начальной выставки и коррекции», README датасета — только для выставки:
включать ли — решение команды.
"""
import math

from tram_odometry.track import GnssInit, MgrsLocal, StraightTrack


class Positioner:
    def __init__(self, route=None, window_s=5.0, frame='mgrs', lever_m=9.873, dz_m=-3.0,
                 fallback_yaw=0.0, min_move_m=3.0, max_lock_m=50.0, corrections=False,
                 corr_sigma=2.0, corr_gate_m=60.0, corr_period_s=1.0):
        if route is not None and frame != 'mgrs':
            raise ValueError('карта задана в сетке MGRS: выход с картой — только frame=mgrs')
        self.route = route
        self.max_lock_m = max_lock_m
        self.corrections, self.corr_sigma = corrections, corr_sigma
        self.corr_gate_m, self.corr_period_s = corr_gate_m, corr_period_s
        self.window_s, self.frame, self.min_move_m = window_s, frame, min_move_m
        self.lever_m, self.dz_m, self.fallback_yaw = lever_m, dz_m, fallback_yaw
        self.start_run()

    def start_run(self):
        self.init = GnssInit(self.window_s, self.frame, self.min_move_m)
        self.track = StraightTrack()
        self.locked = False
        self.lock_dist = None           # расстояние от точки старта до карты, м
        self._pending = None            # последний фикс master после выставки
        self._last_corr = None
        self.corrections_applied = 0

    def fix(self, t, lat, lon, alt, status=0, rover=False):
        if self.locked:
            if self.corrections and not rover and status >= 0 and all(map(math.isfinite, (lat, lon, alt))):
                self._pending = (t, lat, lon, alt)
            return
        self.init.fix(t, lat, lon, alt, status, rover)
        if not self.init.relative:
            self.track = self.init.straight_track(self.fallback_yaw)

    def update(self, t, estimator):
        """Вызывать на каждом входе. True — в этот момент закончилась выставка."""
        self.init.start(t)
        if self.locked:
            if self._pending is not None:
                self._correct(estimator)
            return False
        if not self.init.expired(t):
            return False
        self.locked = True
        if self.init.relative:
            self.track = StraightTrack()
        elif self.route is not None and self._lock_route(estimator):
            self.track = self.route
        elif not self.init.still:            # без карты и на ходу: прямая идёт от последнего фикса,
            estimator.set_position(0.0, 3.0)  # значит и дистанция от него
        return True

    def _correct(self, estimator):
        """Фикс master после выставки -> привязка дистанции (только на карте)."""
        t, lat, lon, alt = self._pending
        self._pending = None
        if self.track is not self.route or (self._last_corr is not None and t - self._last_corr < self.corr_period_s):
            return
        x, y, _ = (self.init.frame or MgrsLocal()).forward(lat, lon, alt)
        s_est = estimator.state().s
        s_map, dist = self.route.locate(x, y, self.route.pose(s_est)[3])
        if dist > 10.0:
            return                                      # фикс не на пути: прыжок GNSS или отвод
        half = 0.5 * self.route.length
        delta = (s_map - s_est + half) % self.route.length - half
        if abs(delta) > self.corr_gate_m:
            return
        estimator.position_fix(t, s_est + delta, self.corr_sigma)
        self._last_corr = t
        self.corrections_applied += 1

    def _lock_route(self, estimator):
        """Проекция точки старта на карту. С курсом — свой путь двухпутки; ни одного участка
        с таким курсом рядом (старт на развороте, курс не определён точно) — без курса.
        Дальше max_lock_m от карты (депо, отвод) — карту не используем: прямая от старта."""
        x, y, _ = self.init.start_point()
        s0, dist = self.route.locate(x, y, self.init.heading())
        if not dist <= self.max_lock_m:
            s0, dist = self.route.locate(x, y)
        self.lock_dist = dist
        if not dist <= self.max_lock_m:
            return False
        if self.init.still:              # стоим с начала окна: к проекции добавляем то, что проехали
            estimator.set_position(s0 + estimator.state().s, max(dist, 1.0))
        else:                            # едем: последний фикс — это «сейчас»
            estimator.set_position(s0, max(dist, 3.0))
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
