import math
from pathlib import Path

from tram_odometry.geo import Enu
from tram_odometry.route_map import RouteMap

MAP = Path(__file__).resolve().parents[1] / 'maps' / 'route.csv'


def test_enu_roundtrip():
    enu = Enu(55.80, 37.42, 150.0)
    for lat, lon, alt in ((55.80, 37.42, 150.0), (55.8107, 37.4623, 176.7), (55.7948, 37.3857, 142.1)):
        e, n, u = enu.forward(lat, lon, alt)
        la, lo, al = enu.inverse(e, n, u)
        assert abs(la - lat) < 1e-9 and abs(lo - lon) < 1e-9 and abs(al - alt) < 1e-4


def test_enu_axes():
    enu = Enu(55.80, 37.42, 150.0)
    e, n, _ = enu.forward(55.80, 37.43, 150.0)  # на восток
    assert e > 600 and abs(n) < 1
    e, n, _ = enu.forward(55.81, 37.42, 150.0)  # на север
    assert n > 1100 and abs(e) < 1


def square_map():
    enu = Enu(55.80, 37.42, 150.0)
    pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
    s, lat, lon, alt, acc = [], [], [], [], 0.0
    for k, (x, y) in enumerate(pts):
        if k:
            acc += math.hypot(x - pts[k - 1][0], y - pts[k - 1][1])
        la, lo, al = enu.inverse(x, y, 0.0)
        s.append(acc)
        lat.append(la)
        lon.append(lo)
        alt.append(al)
    return RouteMap(s, lat, lon, alt).to_frame(enu)


def test_pose_wraps_and_interpolates():
    m = square_map()
    assert abs(m.length - 400.0) < 1e-3
    x, y, _, yaw = m.pose(50.0)
    assert abs(x - 50) < 1e-3 and abs(y) < 1e-3 and abs(yaw) < 1e-6
    x, y, _, _ = m.pose(450.0)  # за концом маршрута — снова с начала
    assert abs(x - 50) < 1e-3 and abs(y) < 1e-3
    x, y, _, _ = m.pose(350.0)  # замыкающий участок (0,100) -> (0,0)
    assert abs(x) < 1e-3 and abs(y - 50) < 1e-3


def test_locate_respects_heading():
    m = square_map()
    s, d = m.locate(50.0, 2.0)
    assert abs(s - 50.0) < 1e-3 and abs(d - 2.0) < 1e-3
    # у угла (100, 0): с курсом на север выбираем участок, идущий вверх
    s, _ = m.locate(99.0, 1.0, yaw=math.pi / 2)
    assert 100.0 <= s <= 102.0


def test_real_map_is_closed_and_dense():
    m = RouteMap.load(MAP).to_frame(Enu(55.80, 37.42, 150.0))
    assert 10000 < m.length < 12000
    steps = [math.hypot(m.x[i + 1] - m.x[i], m.y[i + 1] - m.y[i]) for i in range(len(m.x) - 1)]
    assert max(steps) < 2.0
