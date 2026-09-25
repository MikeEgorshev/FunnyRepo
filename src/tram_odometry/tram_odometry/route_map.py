"""Карта линии: замкнутый маршрут с параметром s — дистанцией вдоль пути.

Файл карты — CSV со столбцами s_m, lat, lon, alt (точки примерно через 1 м, строки с # — комментарии).
Маршрут замкнут: за последней точкой снова идёт первая. Трамвай движется по нему только вперёд.
Перед использованием карту переводят в локальную систему прогона: to_frame(Enu(...)).

Отводы (необязательный файл spur_id, s_join, d_m, lat, lon, alt) — пути у конечных, которых нет
в кольце: от места стоянки до выхода на кольцо в точке s_join. Если старт выбран на отводе,
координата s «виртуальная»: пока s < s_join, положение берётся с отвода, дальше — с кольца.
"""
import bisect
import math


def _read_csv(path, ncols):
    cols = tuple([] for _ in range(ncols))
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or not (line[0].isdigit() or line[0] == '-'):
                continue
            for col, value in zip(cols, line.split(',')):
                col.append(float(value))
    return cols


class RouteMap:
    def __init__(self, s, lat, lon, alt):
        if len(s) < 3:
            raise ValueError('в карте меньше трёх точек')
        self.s, self.lat, self.lon, self.alt = list(s), list(lat), list(lon), list(alt)
        self.x = self.y = self.z = None
        self.length = None
        self.spurs = []
        self.active_spur = None

    @classmethod
    def load(cls, path):
        return cls(*_read_csv(path, 4))

    def load_spurs(self, path):
        ids, s_join, d, lat, lon, alt = _read_csv(path, 6)
        by_id = {}
        for k, sj, dd, la, lo, al in zip(ids, s_join, d, lat, lon, alt):
            sp = by_id.setdefault(int(k), {'s_join': sj, 'd': [], 'lat': [], 'lon': [], 'alt': []})
            sp['d'].append(dd)
            sp['lat'].append(la)
            sp['lon'].append(lo)
            sp['alt'].append(al)
        self.spurs = list(by_id.values())
        return self

    def to_frame(self, enu):
        """Переводит точки карты и отводов в локальную систему enu (geo.Enu) и замыкает маршрут."""
        pts = [enu.forward(la, lo, al) for la, lo, al in zip(self.lat, self.lon, self.alt)]
        self.x = [p[0] for p in pts]
        self.y = [p[1] for p in pts]
        self.z = [p[2] for p in pts]
        closing = math.hypot(self.x[0] - self.x[-1], self.y[0] - self.y[-1])
        self.length = self.s[-1] + closing
        for sp in self.spurs:
            p = [enu.forward(la, lo, al) for la, lo, al in zip(sp['lat'], sp['lon'], sp['alt'])]
            sp['x'], sp['y'], sp['z'] = [q[0] for q in p], [q[1] for q in p], [q[2] for q in p]
        self.active_spur = None
        return self

    def locate_start(self, x, y, yaw=None, max_yaw_diff=math.radians(60)):
        """Как locate(), но учитывает отводы: стартовая s и расстояние до выбранного пути.

        Отвод выбирается, если он ближе кольца больше чем на метр. Тогда возвращаемая s
        виртуальная (s_join минус остаток отвода), а pose() до s_join идёт по отводу.
        """
        s, dist = self.locate(x, y, yaw, max_yaw_diff)
        self.active_spur = None
        best = None
        for sp in self.spurs:
            n = len(sp['d'])
            for i in range(n - 1):
                dx, dy = sp['x'][i + 1] - sp['x'][i], sp['y'][i + 1] - sp['y'][i]
                seg2 = dx * dx + dy * dy
                if seg2 == 0.0:
                    continue
                if yaw is not None:
                    diff = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
                    if abs(diff) > max_yaw_diff:
                        continue
                t = max(0.0, min(1.0, ((x - sp['x'][i]) * dx + (y - sp['y'][i]) * dy) / seg2))
                d = math.hypot(x - sp['x'][i] - t * dx, y - sp['y'][i] - t * dy)
                if best is None or d < best[0]:
                    best = (d, sp, sp['d'][i] + t * (sp['d'][i + 1] - sp['d'][i]))
        if best is not None and best[0] < dist - 1.0:
            d, sp, along = best
            self.active_spur = sp
            return sp['s_join'] - (sp['d'][-1] - along), d
        return s, dist

    def _segment(self, s):
        s %= self.length
        i = bisect.bisect_right(self.s, s) - 1
        j = (i + 1) % len(self.s)
        s_next = self.s[j] if j else self.length
        t = (s - self.s[i]) / (s_next - self.s[i]) if s_next > self.s[i] else 0.0
        return i, j, t

    def pose(self, s):
        """(x, y, z, yaw) в точке маршрута s; yaw — курс по касательной, рад, от оси x к оси y."""
        sp = self.active_spur
        if sp is not None and s < sp['s_join']:
            return self._spur_pose(sp, sp['d'][-1] - (sp['s_join'] - s))
        i, j, t = self._segment(s)
        x = self.x[i] + t * (self.x[j] - self.x[i])
        y = self.y[i] + t * (self.y[j] - self.y[i])
        z = self.z[i] + t * (self.z[j] - self.z[i])
        return x, y, z, math.atan2(self.y[j] - self.y[i], self.x[j] - self.x[i])

    @staticmethod
    def _spur_pose(sp, d):
        d = max(sp['d'][0], min(d, sp['d'][-1]))
        i = max(0, min(bisect.bisect_right(sp['d'], d) - 1, len(sp['d']) - 2))
        span = sp['d'][i + 1] - sp['d'][i]
        t = (d - sp['d'][i]) / span if span > 0 else 0.0
        pos = [sp[c][i] + t * (sp[c][i + 1] - sp[c][i]) for c in ('x', 'y', 'z')]
        yaw = math.atan2(sp['y'][i + 1] - sp['y'][i], sp['x'][i + 1] - sp['x'][i])
        return pos[0], pos[1], pos[2], yaw

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
