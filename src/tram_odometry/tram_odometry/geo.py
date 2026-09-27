"""WGS-84 <-> локальная ENU (метры). Чистый Python: модуль используется и в ноде."""
import math

A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)
B = A * (1 - F)
EP2 = (A * A - B * B) / (B * B)


def to_ecef(lat, lon, alt):
    la, lo = math.radians(lat), math.radians(lon)
    n = A / math.sqrt(1 - E2 * math.sin(la) ** 2)
    return ((n + alt) * math.cos(la) * math.cos(lo),
            (n + alt) * math.cos(la) * math.sin(lo),
            (n * (1 - E2) + alt) * math.sin(la))


def from_ecef(x, y, z):
    """ECEF -> (lat, lon, alt), формула Боуринга (точность — доли миллиметра у поверхности)."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    th = math.atan2(z * A, p * B)
    lat = math.atan2(z + EP2 * B * math.sin(th) ** 3, p - E2 * A * math.cos(th) ** 3)
    n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
    return math.degrees(lat), math.degrees(lon), p / math.cos(lat) - n


def utm_ok(lat, lon, zone=37):
    """Точка, где utm_forward определена: не у полюса и не дальше 60° от осевого меридиана зоны.

    На широте ±90° (и в 90° от меридиана) формулы дают math domain error: такой фикс GNSS — мусор.
    """
    return abs(lat) < 84.0 and abs(lon - (zone * 6 - 183)) < 60.0


def utm_forward(lat, lon, zone=37):
    """WGS-84 -> UTM (северное полушарие): (E, N), м. Ряды Крюгера, точность ~1 мм в пределах зоны."""
    k0, n = 0.9996, F / (2 - F)
    a_ = A / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    alpha = (n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16, 13 * n ** 2 / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240)
    phi, lam = math.radians(lat), math.radians(lon - (zone * 6 - 183))
    e = math.sqrt(E2)
    t = math.sinh(math.atanh(math.sin(phi)) - e * math.atanh(e * math.sin(phi)))
    xi, eta = math.atan2(t, math.cos(lam)), math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    x = eta + sum(al * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j, al in enumerate(alpha, 1))
    y = xi + sum(al * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j, al in enumerate(alpha, 1))
    return 500000.0 + k0 * a_ * x, k0 * a_ * y


class UtmLocal:
    """Сетка UTM со сдвигом начала — система pathgraph организаторов: UTM 37N минус (300000, 6100000).

    Высота — как есть (эллипсоидальная высота GNSS). Интерфейс как у Enu: forward(lat, lon, alt).
    """

    def __init__(self, zone=37, false_e=300000.0, false_n=6100000.0):
        self.zone, self.false_e, self.false_n = zone, false_e, false_n

    def forward(self, lat, lon, alt):
        e, n = utm_forward(lat, lon, self.zone)
        return e - self.false_e, n - self.false_n, alt


class Enu:
    """Локальная система East-North-Up с началом в (lat0, lon0, alt0)."""

    def __init__(self, lat0, lon0, alt0):
        self.origin = (lat0, lon0, alt0)
        self._o = to_ecef(lat0, lon0, alt0)
        la, lo = math.radians(lat0), math.radians(lon0)
        self._sl, self._cl = math.sin(la), math.cos(la)
        self._so, self._co = math.sin(lo), math.cos(lo)

    def forward(self, lat, lon, alt):
        x, y, z = to_ecef(lat, lon, alt)
        dx, dy, dz = x - self._o[0], y - self._o[1], z - self._o[2]
        e = -self._so * dx + self._co * dy
        n = -self._sl * self._co * dx - self._sl * self._so * dy + self._cl * dz
        u = self._cl * self._co * dx + self._cl * self._so * dy + self._sl * dz
        return e, n, u

    def inverse(self, e, n, u):
        dx = -self._so * e - self._sl * self._co * n + self._cl * self._co * u
        dy = self._co * e - self._sl * self._so * n + self._cl * self._so * u
        dz = self._cl * n + self._sl * u
        return from_ecef(self._o[0] + dx, self._o[1] + dy, self._o[2] + dz)
