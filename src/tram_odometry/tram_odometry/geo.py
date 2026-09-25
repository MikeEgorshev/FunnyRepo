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
