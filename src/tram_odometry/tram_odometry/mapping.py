"""Карта линии по GNSS обучающих прогонов: офлайн, без ROS и внешних библиотек.

Карта строится по антенне master (по ней считает s фильтр и позиционер). Шаги:
1. clean_run — фиксы одного прогона -> точки на ходу через step_m: прыжки GNSS и дрожание
   на стоянках убраны;
2. stitch — основа карты — самый длинный прогон; куски других прогонов, которых в карте ещё
   нет и которые продолжают её конец или начало, дописываются. Так из прогонов «туда» и
   «обратно» собирается замкнутый круг с петлями на конечных. Соседство ищется только
   с близким курсом: на двухпутном участке у каждого направления свой путь;
3. refine — среднее поперечное смещение и высота всех прогонов по точкам карты, сглаживание;
4. find_stops — стоянки дольше dwell_s во всех прогонах -> s на карте -> места, где стоят
   в нескольких прогонах (остановки, стоп-линии).

Все расчёты — в локальной ENU, м: длина пути s — настоящая, а не длина в сетке UTM
(в сетке на этой долготе отрезки короче на 0,03 %, это 2 м на 7,5 км).
"""
import math

from tram_odometry.track import RouteTrack


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _d2(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def path_length(pts):
    return sum(_d2(a, b) for a, b in zip(pts, pts[1:]))


def resample(pts, step):
    """Ломаная [(x, y, z)] -> точки через step по длине пути."""
    if len(pts) < 2:
        return list(pts)
    out, need = [pts[0]], step
    for a, b in zip(pts, pts[1:]):
        seg, pos = _d2(a, b), 0.0
        while seg - pos >= need:
            pos += need
            t = pos / seg
            out.append(tuple(a[k] + t * (b[k] - a[k]) for k in range(3)))
            need = step
        need -= seg - pos
    return out


def headings(pts, closed=False):
    n = len(pts)
    out = []
    for i in range(n):
        if closed:
            a, b = pts[i - 1], pts[(i + 1) % n]
        else:
            a, b = pts[max(i - 1, 0)], pts[min(i + 1, n - 1)]
        out.append(math.atan2(b[1] - a[1], b[0] - a[0]))
    return out


def drop_jumps(fixes, max_speed=30.0, reset_after=20):
    """[(t, x, y, z)] -> без прыжков: точка дальше, чем проедешь на max_speed, отбрасывается.
    Если отброшено reset_after подряд — ошибкой была прошлая точка, начинаем с текущей."""
    out, bad = [], 0
    for f in fixes:
        if out:
            dt = f[0] - out[-1][0]
            if dt <= 0.0 or _d2(out[-1][1:], f[1:]) > max_speed * max(dt, 0.1):
                bad += 1
                if bad < reset_after:
                    continue
        out.append(f)
        bad = 0
    return out


def clean_run(fixes, step=2.0, min_move=3.0):
    """[(t, x, y, z)] одного прогона -> точки пути на ходу через step.

    Точки берутся не чаще чем через min_move: это в разы больше шума фиксов, поэтому
    дрожание на стоянке не рисует петель. Шаг назад (разворот больше 120°) — шум:
    движения назад в данных нет; после 5 таких шагов подряд курс сбрасывается."""
    kept, prev_h, back = [], None, 0
    for f in drop_jumps(fixes):
        p = f[1:]
        if kept:
            if _d2(kept[-1], p) < min_move:
                continue
            h = math.atan2(p[1] - kept[-1][1], p[0] - kept[-1][0])
            if prev_h is not None and abs(_wrap(h - prev_h)) > math.radians(120) and back < 5:
                back += 1
                continue
            prev_h, back = h, 0
        kept.append(p)
    return resample(kept, step)


class _Grid:
    def __init__(self, pts, cell=10.0):
        self.pts, self.cell, self.cells = pts, cell, {}
        for i, p in enumerate(pts):
            self.cells.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(i)

    def near(self, x, y, r):
        c, k = self.cell, int(r // self.cell) + 1
        cx, cy = int(x // c), int(y // c)
        for i in range(cx - k, cx + k + 1):
            for j in range(cy - k, cy + k + 1):
                yield from self.cells.get((i, j), ())


def _nearest(grid, hd, p, h, r, tol, skip=None):
    """Ближайшая точка карты в радиусе r с курсом в пределах tol -> (индекс, расстояние)."""
    best = None
    for j in grid.near(p[0], p[1], r):
        if skip is not None and skip(j):
            continue
        d = _d2(grid.pts[j], p)
        if d <= r and abs(_wrap(hd[j] - h)) <= tol and (best is None or d < best[1]):
            best = (j, d)
    return best


def _close(route, step, cover_m, tol, min_loop_m=300.0):
    """Если маршрут вернулся на своё начало (тот же путь, тот же курс), обрезает повтор.
    -> (маршрут, замкнут ли)."""
    hd = headings(route)
    gap = int(min_loop_m / step)
    grid = _Grid(route)
    for i in range(gap, len(route)):
        m = _nearest(grid, hd, route[i], hd[i], cover_m, tol, skip=lambda j, i=i: j > i - gap)
        if m is not None:
            return route[m[0]:i], True
    return route, False


def stitch(runs, step=2.0, cover_m=6.0, tol=math.radians(60), min_new_m=40.0, join_m=30.0):
    """Сшивает прогоны в один маршрут. -> (точки, замкнут ли)."""
    runs = sorted((r for r in runs if len(r) > 2), key=path_length, reverse=True)
    if not runs:
        raise ValueError('нет прогонов с движением')
    route, closed = _close(list(runs[0]), step, cover_m, tol)
    join = int(join_m / step)
    changed = True
    while changed and not closed:
        changed = False
        grid, hd = _Grid(route), headings(route)
        for run in runs[1:]:
            rh = headings(run)
            cov = [_nearest(grid, hd, p, h, cover_m, tol) for p, h in zip(run, rh)]
            n, i = len(run), 0
            while i < n and not changed:
                if cov[i] is not None:
                    i += 1
                    continue
                a = i
                while i < n and cov[i] is None:
                    i += 1
                b = i - 1
                if (b - a + 1) * step < min_new_m:
                    continue
                if a > 0 and len(route) - 1 - cov[a - 1][0] <= join:
                    route = route + list(run[a:min(b + 2, n)])
                    changed = True
                elif b < n - 1 and cov[b + 1][0] <= join:
                    route = list(run[max(a - 1, 0):b + 1]) + route
                    changed = True
            if changed:
                route, closed = _close(route, step, cover_m, tol)
                break
    return route, closed


def _smooth(values, w, closed):
    if w <= 1:
        return list(values)
    n, h = len(values), w // 2
    out = []
    for i in range(n):
        if closed:
            idx = [(i + k) % n for k in range(-h, h + 1)]
        else:
            idx = range(max(0, i - h), min(n, i + h + 1))
        vals = [values[j] for j in idx]
        out.append(sum(vals) / len(vals))
    return out


def _taubin(xs, ys, closed, passes=10, lam=0.5, mu=-0.53):
    """Сглаживание Таубина: убирает зубцы шума, но, в отличие от скользящего среднего,
    не стягивает кривые внутрь — длина петель на конечных не уменьшается."""
    xs, ys, n = list(xs), list(ys), len(xs)
    for _ in range(passes):
        for f in (lam, mu):
            nx, ny = xs[:], ys[:]
            for i in range(n):
                if not closed and i in (0, n - 1):
                    continue
                a, b = (i - 1) % n, (i + 1) % n
                nx[i] = xs[i] + f * (0.5 * (xs[a] + xs[b]) - xs[i])
                ny[i] = ys[i] + f * (0.5 * (ys[a] + ys[b]) - ys[i])
            xs, ys = nx, ny
    return xs, ys


def refine(route, runs, closed, step=2.0, cover_m=6.0, tol=math.radians(60), iters=2,
           smooth_z=15):
    """Сдвигает точки карты на среднее поперечное смещение прогонов, высота — среднее.
    -> (точки, СКО поперечного отклонения прогонов от карты, м)."""
    rms = float('nan')
    for _ in range(iters):
        n = len(route)
        hd, grid = headings(route, closed), _Grid(route)
        off, zs, cnt = [0.0] * n, [0.0] * n, [0] * n
        sq, m = 0.0, 0
        for run in runs:
            for p, h in zip(run, headings(run)):
                best = _nearest(grid, hd, p, h, cover_m, tol)
                if best is None:
                    continue
                j = best[0]
                q = route[j]
                lat = -math.sin(hd[j]) * (p[0] - q[0]) + math.cos(hd[j]) * (p[1] - q[1])
                off[j] += lat
                zs[j] += p[2]
                cnt[j] += 1
                sq += lat * lat
                m += 1
        rms = math.sqrt(sq / m) if m else float('nan')
        xs, ys, zz = [], [], []
        for j, q in enumerate(route):
            o = off[j] / cnt[j] if cnt[j] else 0.0
            xs.append(q[0] - math.sin(hd[j]) * o)
            ys.append(q[1] + math.cos(hd[j]) * o)
            zz.append(zs[j] / cnt[j] if cnt[j] else q[2])
        xs, ys = _taubin(xs, ys, closed)
        pts = list(zip(xs, ys, _smooth(zz, smooth_z, closed)))
        if closed:
            pts = resample(pts + [pts[0]], step)
            if _d2(pts[-1], pts[0]) < 0.5 * step:
                pts.pop()
        else:
            pts = resample(pts, step)
        route = pts
    return route, rms


def distances(route):
    """Длина пути от начала до каждой точки (3D), м."""
    s = [0.0]
    for a, b in zip(route, route[1:]):
        s.append(s[-1] + math.dist(a, b))
    return s


def stop_places(fixes, dwell_s=8.0, radius_m=2.0, lookback_m=10.0):
    """Стоянки одного прогона [(t, x, y, z)] -> [(x, y, курс или None)]."""
    fixes = drop_jumps(fixes)
    out, i, n = [], 0, len(fixes)
    while i < n:
        j = i
        while j + 1 < n and _d2(fixes[i][1:], fixes[j + 1][1:]) <= radius_m:
            j += 1
        if fixes[j][0] - fixes[i][0] >= dwell_s:
            span = sorted(fixes[i:j + 1], key=lambda f: f[1])
            x = span[len(span) // 2][1]
            y = sorted(f[2] for f in span)[len(span) // 2]
            out.append((x, y, _course(fixes, i, j, lookback_m)))
        i = j + 1
    return out


def _course(fixes, i, j, dist):
    """Курс движения перед стоянкой [i, j], а если прогон с неё начался — после неё."""
    for k in range(i - 1, -1, -1):
        if _d2(fixes[k][1:], fixes[i][1:]) >= dist:
            return math.atan2(fixes[i][2] - fixes[k][2], fixes[i][1] - fixes[k][1])
    for k in range(j + 1, len(fixes)):
        if _d2(fixes[k][1:], fixes[j][1:]) >= dist:
            return math.atan2(fixes[k][2] - fixes[j][2], fixes[k][1] - fixes[j][1])
    return None


def find_stops(track, runs_fixes, cluster_m=15.0, min_runs=3, max_dist_m=5.0, **kw):
    """Места регулярных стоянок на карте -> [(s, sigma)], м. runs_fixes — [(t, x, y, z)]
    каждого прогона. Место засчитывается, если там стояли не меньше чем в min_runs прогонах."""
    hits = []
    for r, fixes in enumerate(runs_fixes):
        for x, y, yaw in stop_places(fixes, **kw):
            s, d = track.locate(x, y, yaw)
            if d <= max_dist_m:
                hits.append((s, r))
    hits.sort()
    groups, cur = [], []
    for h in hits:
        if cur and h[0] - cur[-1][0] > cluster_m:
            groups.append(cur)
            cur = []
        cur.append(h)
    if cur:
        groups.append(cur)
    if len(groups) > 1 and track.length - groups[-1][-1][0] + groups[0][0][0] <= cluster_m:
        groups[0] = [(s - track.length, r) for s, r in groups.pop()] + groups[0]   # через начало круга
    stops = []
    for g in groups:
        if len({r for _, r in g}) < min_runs:
            continue
        s = sorted(v for v, _ in g)
        med = s[len(s) // 2]
        mad = sorted(abs(v - med) for v in s)[len(s) // 2]
        stops.append((med % track.length, max(1.0, 1.4826 * mad)))
    return sorted(stops)


class Locator:
    """Быстрый поиск s на карте по точке (x, y): ближайшая точка карты в радиусе r.
    Курс не учитывается — годится для уклона и высоты (у двух путей они одинаковые)."""

    def __init__(self, track, cell=20.0):
        self.track = track
        self.grid = _Grid(list(zip(track.x, track.y)), cell)

    def s_at(self, x, y, r=15.0):
        best = None
        for j in self.grid.near(x, y, r):
            d = math.hypot(self.track.x[j] - x, self.track.y[j] - y)
            if d <= r and (best is None or d < best[1]):
                best = (j, d)
        return None if best is None else self.track.s[best[0]]


def build(runs_fixes, step=2.0, **kw):
    """Полный цикл: [(t, x, y, z)] прогонов -> (RouteTrack в ENU, отчёт)."""
    runs = [clean_run(f, step) for f in runs_fixes]
    route, closed = stitch(runs, step)
    route, rms = refine(route, runs, closed, step)
    track = RouteTrack(distances(route), [p[0] for p in route], [p[1] for p in route],
                       [p[2] for p in route])
    if not closed:
        track.length = track.s[-1]
    stops = find_stops(track, runs_fixes, **kw)
    track.stops = stops
    return track, {'closed': closed, 'length_m': track.length, 'points': len(route),
                   'lateral_rms_m': rms, 'stops': len(stops), 'runs': len(runs)}
