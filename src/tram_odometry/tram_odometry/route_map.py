"""Карта линии: замкнутый маршрут с параметром s — дистанцией вдоль пути.

Файл карты — CSV со столбцами s_m, lat, lon, alt (точки примерно через 1 м, строки с # — комментарии).
Маршрут замкнут: за последней точкой снова идёт первая. Трамвай движется по нему только вперёд.
Перед использованием карту переводят в локальную систему прогона: to_frame(Enu(...)).
"""
import bisect
import math


class RouteMap:
    def __init__(self, s, lat, lon, alt):
        if len(s) < 3:
            raise ValueError('в карте меньше трёх точек')
        self.s, self.lat, self.lon, self.alt = list(s), list(lat), list(lon), list(alt)
        self.x = self.y = self.z = None
        self.length = None

    @classmethod
    def load(cls, path):
        cols = ([], [], [], [])
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('s_m'):
                    continue
                for col, value in zip(cols, line.split(',')):
                    col.append(float(value))
        return cls(*cols)

    def to_frame(self, enu):
        """Переводит точки карты в локальную систему enu (geo.Enu) и замыкает маршрут."""
        pts = [enu.forward(la, lo, al) for la, lo, al in zip(self.lat, self.lon, self.alt)]
        self.x = [p[0] for p in pts]
        self.y = [p[1] for p in pts]
        self.z = [p[2] for p in pts]
        closing = math.hypot(self.x[0] - self.x[-1], self.y[0] - self.y[-1])
        self.length = self.s[-1] + closing
        return self

    def _segment(self, s):
        s %= self.length
        i = bisect.bisect_right(self.s, s) - 1
        j = (i + 1) % len(self.s)
        s_next = self.s[j] if j else self.length
        t = (s - self.s[i]) / (s_next - self.s[i]) if s_next > self.s[i] else 0.0
        return i, j, t

    def pose(self, s):
        """(x, y, z, yaw) в точке маршрута s; yaw — курс по касательной, рад, от оси x к оси y."""
        i, j, t = self._segment(s)
        x = self.x[i] + t * (self.x[j] - self.x[i])
        y = self.y[i] + t * (self.y[j] - self.y[i])
        z = self.z[i] + t * (self.z[j] - self.z[i])
        return x, y, z, math.atan2(self.y[j] - self.y[i], self.x[j] - self.x[i])

    def locate(self, x, y, yaw=None, max_yaw_diff=math.radians(60)):
        """Ближайшая к (x, y) точка маршрута -> (s, расстояние до маршрута).

        Если задан yaw, отбрасываются участки с курсом, отличным больше чем на max_yaw_diff:
        так на двухпутном участке выбирается путь нужного направления.
        """
        best = (float('inf'), 0.0)
        n = len(self.s)
        for i in range(n):
            j = (i + 1) % n
            dx, dy = self.x[j] - self.x[i], self.y[j] - self.y[i]
            seg2 = dx * dx + dy * dy
            if seg2 == 0.0:
                continue
            if yaw is not None:
                diff = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
                if abs(diff) > max_yaw_diff:
                    continue
            t = max(0.0, min(1.0, ((x - self.x[i]) * dx + (y - self.y[i]) * dy) / seg2))
            px, py = self.x[i] + t * dx, self.y[i] + t * dy
            d = math.hypot(x - px, y - py)
            if d < best[0]:
                s_next = self.s[j] if j else self.length
                best = (d, self.s[i] + t * (s_next - self.s[i]))
        return best[1], best[0]
