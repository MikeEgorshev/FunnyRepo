"""Путь, по которому едет трамвай, и начальная выставка по GNSS.

Фильтр знает только дистанцию s. Координаты x, y, z и курс даёт «путь» — объект с методом
pose(s) -> (x, y, z, yaw) в системе map (ENU, метры). Здесь — запасной путь без карты:
прямая от точки старта по начальному курсу (относительная одометрия от старта, её допускает
описание задачи). Карта линии с тем же методом pose подключается вместо него.

Система map — гипотеза до ответа организаторов: начало в первой точке GNSS master прогона,
оси ENU, z — высота относительно этой точки.
"""
import math

_A = 6378137.0                  # WGS-84
_E2 = 6.69437999014e-3


def _ecef(lat, lon, alt):
    la, lo = math.radians(lat), math.radians(lon)
    n = _A / math.sqrt(1.0 - _E2 * math.sin(la) ** 2)
    return ((n + alt) * math.cos(la) * math.cos(lo),
            (n + alt) * math.cos(la) * math.sin(lo),
            (n * (1.0 - _E2) + alt) * math.sin(la))


class LocalFrame:
    """Локальная касательная система ENU с началом в (lat0, lon0, alt0)."""

    def __init__(self, lat0, lon0, alt0=0.0):
        self.origin = (lat0, lon0, alt0)
        self._o = _ecef(lat0, lon0, alt0)
        la, lo = math.radians(lat0), math.radians(lon0)
        self._r = ((-math.sin(lo), math.cos(lo), 0.0),
                   (-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)),
                   (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)))

    def forward(self, lat, lon, alt=0.0):
        p = _ecef(lat, lon, alt)
        d = (p[0] - self._o[0], p[1] - self._o[1], p[2] - self._o[2])
        return tuple(sum(row[i] * d[i] for i in range(3)) for row in self._r)


class StraightTrack:
    """Запасной путь: прямая из (x0, y0, z0) по курсу yaw (рад, от оси x = восток)."""

    def __init__(self, x0=0.0, y0=0.0, z0=0.0, yaw=0.0):
        self.x0, self.y0, self.z0, self.yaw = x0, y0, z0, yaw

    def pose(self, s):
        return (self.x0 + s * math.cos(self.yaw), self.y0 + s * math.sin(self.yaw), self.z0, self.yaw)


class GnssInit:
    """Начальная выставка: собирает фиксы GNSS в окне window_s от первого входа прогона.

    Начало системы — первый годный фикс. Курс — по смещению между фиксами, если трамвай
    сдвинулся больше min_move_m; иначе курс не определён (heading_known = False).
    """

    def __init__(self, window_s=5.0, min_move_m=3.0):
        self.window_s = window_s
        self.min_move_m = min_move_m
        self.t0 = None
        self.frame = None
        self.last_enu = None
        self.done = False

    def start(self, t):
        if self.t0 is None:
            self.t0 = t

    def fix(self, t, lat, lon, alt, status=0):
        """Фикс GNSS. status < 0 (нет решения) и NaN не используются."""
        if self.done or self.t0 is None:
            return
        if t - self.t0 > self.window_s:
            self.done = True
            return
        if status < 0 or not all(math.isfinite(v) for v in (lat, lon, alt)):
            return
        if self.frame is None:
            self.frame = LocalFrame(lat, lon, alt)
        self.last_enu = self.frame.forward(lat, lon, alt)

    def expired(self, t):
        if self.t0 is not None and t - self.t0 > self.window_s:
            self.done = True
        return self.done

    @property
    def heading_known(self):
        return self.last_enu is not None and math.hypot(self.last_enu[0], self.last_enu[1]) >= self.min_move_m

    def track(self, fallback_yaw=0.0):
        """Путь от точки старта. Выставка без GNSS — старт в (0, 0, 0) с курсом fallback_yaw."""
        if not self.heading_known:
            return StraightTrack(yaw=fallback_yaw)
        e, n, _ = self.last_enu
        return StraightTrack(yaw=math.atan2(n, e))
