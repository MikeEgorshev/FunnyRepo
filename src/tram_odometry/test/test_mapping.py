import math
import random

from tram_odometry.mapping import build, clean_run, resample, stitch
from tram_odometry.track import LocalFrame

L = 3000.0


def _circuit(ds=0.3, r=25.0, a=1.0007):
    """Двухпутная линия длиной L, пути в 4 м друг от друга, на концах — разворотные петли
    радиусом r (вправо a, влево pi + 2a, вправо a). Строится интегрированием кривизны."""
    prof = [(0.0, L), (-1 / r, a * r), (1 / r, (math.pi + 2 * a) * r), (-1 / r, a * r)] * 2
    x, y, h = 0.0, -2.0, 0.0
    pts = []
    for k, length in prof:
        n = int(round(length / ds))
        for _ in range(n):                  # шаг подогнан, чтобы угол поворота был точным
            pts.append((x, y, 150.0 + 10.0 * math.sin(math.pi * x / L)))
            h += k * length / n
            x += math.cos(h) * length / n
            y += math.sin(h) * length / n
    s = [0.0]
    for p, q in zip(pts, pts[1:] + pts[:1]):
        s.append(s[-1] + math.dist(p, q))
    return pts, s


TRUTH, TRUTH_S = _circuit()
LENGTH = TRUTH_S[-1]
STOPS = [500.0, 1700.0, 3100.0, 4400.0, 5600.0]   # регулярные стоянки, s по кругу


def _at(s):
    s %= LENGTH
    lo, hi = 0, len(TRUTH) - 1
    while lo < hi:                                  # последняя точка с TRUTH_S <= s
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if TRUTH_S[mid] <= s else (lo, mid - 1)
    a, b = TRUTH[lo], TRUTH[(lo + 1) % len(TRUTH)]
    t = (s - TRUTH_S[lo]) / (TRUTH_S[lo + 1] - TRUTH_S[lo])
    return tuple(a[k] + t * (b[k] - a[k]) for k in range(3))


def _run(s0, dist, rnd, extra_stop=None, v=8.0):
    """Фиксы master 10 Гц: едет от s0 на dist, стоит 20 с на регулярных стоянках."""
    bias = (rnd.gauss(0, 0.2), rnd.gauss(0, 0.2))
    fixes, t, s = [], 0.0, s0
    stops = [x for x in STOPS for k in (0, 1) if s0 < x + k * LENGTH <= s0 + dist]
    stops = sorted(x + LENGTH if x < s0 else x for x in stops)
    if extra_stop:
        stops = sorted(stops + [extra_stop])
    while s < s0 + dist:
        wait = 20.0 if stops and s >= stops[0] else 0.0
        if wait:
            stops.pop(0)
        for _ in range(int(wait * 10) + 1):
            x, y, z = _at(s)
            fixes.append((t, x + bias[0] + rnd.gauss(0, 0.3), y + bias[1] + rnd.gauss(0, 0.3), z + rnd.gauss(0, 0.3)))
            t += 0.1
        s += v * 0.1
    return fixes


def _runs():
    rnd = random.Random(3)
    runs = []
    for k, s0 in enumerate([100.0, 3300.0, 5200.0, 1500.0, 4000.0, 2600.0]):
        f = _run(s0, 0.55 * LENGTH, rnd, extra_stop=s0 + 900.0 + 37.0 * k)   # случайная стоянка
        if k == 2:
            x, y, z = f[len(f) // 2][1:]
            f[len(f) // 2] = (f[len(f) // 2][0], x + 60.0, y, z)                # прыжок GNSS
        runs.append(f)
    return runs


def _dist_to_truth(p):
    return min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in TRUTH)


def test_resample_keeps_spacing():
    pts = resample([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 5.0, 1.0)], 2.0)
    assert len(pts) == 8 and all(abs(math.dist(a[:2], b[:2]) - 2.0) < 1e-9 for a, b in zip(pts[:5], pts[1:5]))


def test_local_frame_inverse_roundtrip():
    f = LocalFrame(55.80, 37.42, 150.0)
    for e, n, u in [(0.0, 0.0, 0.0), (5000.0, -800.0, 25.0), (-3000.0, 1200.0, -10.0)]:
        lat, lon, alt = f.inverse(e, n, u)
        e2, n2, u2 = f.forward(lat, lon, alt)
        assert max(abs(e - e2), abs(n - n2), abs(u - u2)) < 1e-4


def test_stitch_closes_circuit_from_partial_runs():
    runs = [clean_run(f) for f in _runs()]
    route, closed = stitch(runs)
    assert closed
    assert abs(sum(math.dist(a[:2], b[:2]) for a, b in zip(route, route[1:] + route[:1])) - LENGTH) < 0.01 * LENGTH


def test_build_map_matches_truth_and_finds_regular_stops():
    track, info = build(_runs())
    assert info['closed'] and abs(info['length_m'] - LENGTH) < 0.003 * LENGTH
    worst = max(_dist_to_truth((x, y)) for x, y in zip(track.x[::25], track.y[::25]))
    assert worst < 0.6 and info['lateral_rms_m'] < 0.6
    # каждая регулярная стоянка найдена и лежит на своём месте; случайные — нет
    assert len(track.stops) == len(STOPS)
    for s_true in STOPS:
        tx, ty, _ = _at(s_true)
        near = [math.hypot(track.pose(s)[0] - tx, track.pose(s)[1] - ty) for s, _ in track.stops]
        assert min(near) < 3.0
    # длина между стоянками по карте совпадает с настоящей: s карты годится фильтру
    def gaps(s, length):
        return sorted((b - a) % length for a, b in zip(s, s[1:] + s[:1]))
    for g_map, g_true in zip(gaps([s for s, _ in track.stops], track.length), gaps(STOPS, LENGTH)):
        assert abs(g_map - g_true) < 3.0
