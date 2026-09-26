"""Путь, по которому едет трамвай, системы координат и начальная выставка по GNSS.

Фильтр знает только дистанцию s. Координаты x, y, z и курс даёт «путь» — объект с методом
pose(s) -> (x, y, z, yaw). Два пути:
- RouteTrack — карта линии из CSV (s_m, lat, lon, alt), замкнутый маршрут; даёт ещё уклон
  grade_at(s), места стоянок stops и длину length;
- StraightTrack — запасной: прямая от точки старта по начальному курсу.

Система выхода (QA 25.09): эталон судьи — /localization/kinematic_state в сетке MGRS,
она же UTM 37N минус (300 000, 6 100 000). Высота — эллипсоидальная высота GNSS.
Без GNSS на старте — относительная одометрия от старта (frame odom).
"""
import bisect
import math

_A = 6378137.0                  # WGS-84
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


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


class MgrsLocal:
    """Сетка UTM со сдвигом начала: (E - false_e, N - false_n, высота). По умолчанию — MGRS
    квадрата судьи: UTM 37N минус (300 000, 6 100 000). UTM — ряды Крюгера (точность ~1 мм)."""

    def __init__(self, zone=37, false_e=300000.0, false_n=6100000.0):
        self.zone, self.false_e, self.false_n = zone, false_e, false_n
        n = _F / (2.0 - _F)
        self._k = 0.9996 * _A / (1.0 + n) * (1.0 + n * n / 4.0 + n ** 4 / 64.0)
        self._alpha = (n / 2 - 2 * n * n / 3 + 5 * n ** 3 / 16, 13 * n * n / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240)
        self._e = math.sqrt(_E2)
        self._lon0 = math.radians(zone * 6 - 183)

    def forward(self, lat, lon, alt=0.0):
        phi, lam = math.radians(lat), math.radians(lon) - self._lon0
        e = self._e
        t = math.sinh(math.atanh(math.sin(phi)) - e * math.atanh(e * math.sin(phi)))
        xi = math.atan2(t, math.cos(lam))
        eta = math.atanh(math.sin(lam) / math.sqrt(1.0 + t * t))
        x = eta + sum(a * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j, a in enumerate(self._alpha, 1))
        y = xi + sum(a * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j, a in enumerate(self._alpha, 1))
        return 500000.0 + self._k * x - self.false_e, self._k * y - self.false_n, alt


def make_frame(name, lat0=None, lon0=None, alt0=0.0):
    """'mgrs' — сетка судьи; 'enu' — ENU от первой точки GNSS."""
    return MgrsLocal() if name == 'mgrs' else LocalFrame(lat0, lon0, alt0)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class StraightTrack:
    """Запасной путь: прямая из (x0, y0, z0) по курсу yaw (рад, от оси x к оси y)."""

    def __init__(self, x0=0.0, y0=0.0, z0=0.0, yaw=0.0):
        self.x0, self.y0, self.z0, self.yaw = x0, y0, z0, yaw

    def pose(self, s):
        return (self.x0 + s * math.cos(self.yaw), self.y0 + s * math.sin(self.yaw), self.z0, self.yaw)


def _read_csv(path, ncols):
    cols = tuple([] for _ in range(ncols))
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not (line[0].isdigit() or line[0] == '-'):
                continue                        # комментарии и заголовок
            for col, value in zip(cols, line.split(',')):
                col.append(float(value))
    return cols


class RouteTrack:
    """Карта линии: замкнутый маршрут; за последней точкой снова идёт первая.

    s в карте — дистанция вдоль пути той точки трамвая, по которой карта построена
    (обычно антенна master). Уклон — по высоте карты на базе ±grade_base_m.
    """

    def __init__(self, s, x, y, z, stops=(), grade_base_m=10.0):
        if len(s) < 3:
            raise ValueError('в карте меньше трёх точек')
        self.s, self.x, self.y, self.z = list(s), list(x), list(y), list(z)
        self.length = self.s[-1] + math.hypot(self.x[0] - self.x[-1], self.y[0] - self.y[-1])
        self.stops = sorted(stops)
        self.grade_base_m = grade_base_m

    @classmethod
    def load(cls, route_csv, frame, stops_csv=None):
        s, lat, lon, alt = _read_csv(route_csv, 4)
        pts = [frame.forward(la, lo, al) for la, lo, al in zip(lat, lon, alt)]
        stops = []
        if stops_csv:
            st, sigma = _read_csv(stops_csv, 2)
            stops = list(zip(st, sigma))
        return cls(s, [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], stops)

    def _segment(self, s):
        s %= self.length
        i = bisect.bisect_right(self.s, s) - 1
        j = (i + 1) % len(self.s)
        s_next = self.s[j] if j else self.length
        t = (s - self.s[i]) / (s_next - self.s[i]) if s_next > self.s[i] else 0.0
        return i, j, t

    def pose(self, s):
        i, j, t = self._segment(s)
        return (self.x[i] + t * (self.x[j] - self.x[i]), self.y[i] + t * (self.y[j] - self.y[i]),
                self.z[i] + t * (self.z[j] - self.z[i]), math.atan2(self.y[j] - self.y[i], self.x[j] - self.x[i]))

    def grade_at(self, s):
        b = self.grade_base_m
        return (self.pose(s + b)[2] - self.pose(s - b)[2]) / (2.0 * b)

    def locate(self, x, y, yaw=None, max_yaw_diff=math.radians(60)):
        """Ближайшая к (x, y) точка маршрута -> (s, расстояние). С yaw — только участки своего
        направления: на двухпутном участке выбирается путь, по которому трамвай едет."""
        best = (float('inf'), 0.0)
        n = len(self.s)
        for i in range(n):
            j = (i + 1) % n
            dx, dy = self.x[j] - self.x[i], self.y[j] - self.y[i]
            seg2 = dx * dx + dy * dy
            if seg2 == 0.0 or (yaw is not None and abs(_wrap(math.atan2(dy, dx) - yaw)) > max_yaw_diff):
                continue
            t = max(0.0, min(1.0, ((x - self.x[i]) * dx + (y - self.y[i]) * dy) / seg2))
            d = math.hypot(x - self.x[i] - t * dx, y - self.y[i] - t * dy)
            if d < best[0]:
                s_next = self.s[j] if j else self.length
                best = (d, self.s[i] + t * (s_next - self.s[i]))
        return best[1], best[0]


def _median(values):
    v = sorted(values)
    return v[len(v) // 2]


class GnssInit:
    """Начальная выставка по фиксам GNSS в окне window_s от первого входа прогона.

    - Точка старта — медиана фиксов master, если трамвай стоит (разброс < still_m); одиночный
      прыжок дальше jump_m от медианы отбрасывается. Если трамвай едет — последний фикс.
    - Курс — по базе антенн master → rover (обе на оси x трамвая, QA 25.09): работает и на месте.
      Если rover нет — по смещению master больше min_move_m. Иначе курс не определён.
    - Ни одного фикса за окно — относительная одометрия (relative = True).
    """

    def __init__(self, window_s=5.0, frame='mgrs', min_move_m=3.0, still_m=3.0, jump_m=10.0,
                 baseline_m=12.436, baseline_tol_m=2.0):
        self.window_s, self.frame_name = window_s, frame
        self.min_move_m, self.still_m, self.jump_m = min_move_m, still_m, jump_m
        self.baseline_m, self.baseline_tol_m = baseline_m, baseline_tol_m
        self.t0 = None
        self.frame = None
        self.master = []                  # [(x, y, z)]
        self.rover = []
        self.done = False

    def start(self, t):
        if self.t0 is None:
            self.t0 = t

    def expired(self, t):
        if self.t0 is not None and t - self.t0 > self.window_s:
            self.done = True
        return self.done

    def fix(self, t, lat, lon, alt, status=0, rover=False):
        """Фикс GNSS. status < 0 (нет решения), NaN и фиксы после окна не используются."""
        self.start(t)
        if self.expired(t) or status < 0 or not all(math.isfinite(v) for v in (lat, lon, alt)):
            return
        if self.frame is None:
            self.frame = make_frame(self.frame_name, lat, lon, alt)
        p = self.frame.forward(lat, lon, alt)
        pts = self.rover if rover else self.master
        if len(pts) > 2:
            mx, my = _median(q[0] for q in pts), _median(q[1] for q in pts)
            if max(math.hypot(q[0] - mx, q[1] - my) for q in pts) < self.still_m \
                    and math.hypot(p[0] - mx, p[1] - my) > self.jump_m:
                return                    # трамвай стоит, а точка улетела — прыжок GNSS
        pts.append(p)

    @property
    def relative(self):
        return not self.master

    @property
    def still(self):
        """Трамвай стоял всё окно: разброс фиксов master меньше still_m."""
        if not self.master:
            return False
        mx, my = _median(q[0] for q in self.master), _median(q[1] for q in self.master)
        return max(math.hypot(q[0] - mx, q[1] - my) for q in self.master) < self.still_m

    def _point(self, pts):
        mx, my = _median(q[0] for q in pts), _median(q[1] for q in pts)
        if max(math.hypot(q[0] - mx, q[1] - my) for q in pts) < self.still_m:
            return mx, my, _median(q[2] for q in pts)
        return pts[-1]

    def start_point(self):
        """Положение антенны master на старте: (x, y, z) или None."""
        return self._point(self.master) if self.master else None

    def heading(self):
        """Курс, рад, или None, если не определён."""
        if self.master and self.rover:
            m, r = self._point(self.master), self._point(self.rover)
            if abs(math.hypot(r[0] - m[0], r[1] - m[1]) - self.baseline_m) <= self.baseline_tol_m:
                return math.atan2(r[1] - m[1], r[0] - m[0])
        if len(self.master) >= 2:
            dx, dy = self.master[-1][0] - self.master[0][0], self.master[-1][1] - self.master[0][1]
            if math.hypot(dx, dy) >= self.min_move_m:
                return math.atan2(dy, dx)
        return None

    @property
    def heading_known(self):
        return self.heading() is not None

    def straight_track(self, fallback_yaw=0.0):
        """Запасной путь от антенны master на старте; без GNSS — от нуля (относительный)."""
        p = self.start_point()
        yaw = self.heading()
        yaw = fallback_yaw if yaw is None else yaw
        return StraightTrack(yaw=yaw) if p is None else StraightTrack(p[0], p[1], p[2], yaw)
