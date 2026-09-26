import math

import pytest
from test_track import _fixes, _ring

from tram_odometry.estimator import Estimator
from tram_odometry.positioning import Positioner


def _drive(pos, est, t_end, notch=0, kmh=0.0):
    t = 0.0
    while t < t_end:
        t = round(t + 0.1, 6)
        est.set_notch(t, notch)
        est.wheel('front', t, kmh)
        est.wheel('rear', t, kmh)
        pos.update(t, est)


def test_locks_onto_map_and_outputs_base_link():
    route, x0, y0 = _ring()
    pos, est = Positioner(route, window_s=5.0), Estimator()
    # трамвай стоит на кольце при s = 300 м, антенны на оси, курс по кольцу
    xa, ya, _, yaw = route.pose(300.0)
    _fixes(pos, xa, ya)
    _fixes(pos, xa + 12.436 * math.cos(yaw), ya + 12.436 * math.sin(yaw), rover=True)
    _drive(pos, est, 6.0)
    assert pos.locked and pos.track is route and abs(est.state().s - 300.0) < 0.5
    x, y, z, _, frame = pos.pose(est.state().s)
    bx, by, bz, _ = route.pose(300.0 + 9.873)     # base_link впереди антенны master
    assert frame == 'map' and math.hypot(x - bx, y - by) < 0.5 and abs(z - (bz - 3.0)) < 0.01


def test_without_map_stays_on_straight_line_from_start():
    pos, est = Positioner(None, window_s=5.0), Estimator()
    from test_track import M
    x0, y0, _ = M.forward(55.80, 37.42)
    _fixes(pos, x0, y0)
    _fixes(pos, x0, y0 + 12.436, rover=True)      # курс на север
    _drive(pos, est, 6.0)
    x, y, _, yaw, frame = pos.pose(100.0)
    assert frame == 'map' and abs(x - x0) < 0.05 and abs(y - (y0 + 109.873)) < 0.05


def test_no_gnss_gives_relative_odometry():
    pos, est = Positioner(None, window_s=5.0), Estimator()
    _drive(pos, est, 6.0, notch=3, kmh=18.0)
    x, y, z, _, frame = pos.pose(42.0)
    assert pos.relative and frame == 'odom' and (x, y, z) == (42.0, 0.0, 0.0)


def test_map_requires_mgrs_frame():
    route, _, _ = _ring()
    with pytest.raises(ValueError):
        Positioner(route, frame='enu')


def test_moving_start_without_map_counts_distance_from_last_fix():
    from test_track import M, _latlon_of
    pos, est = Positioner(None, window_s=5.0), Estimator()
    x0, y0, _ = M.forward(55.80, 37.42)
    t = 0.0
    for i in range(50):                          # едет на восток 5 м/с всё окно, фиксы 10 Гц
        t = round(t + 0.1, 6)
        est.set_notch(t, 3)
        est.wheel('front', t, 18.0 * est.state().k / 3.6)
        est.wheel('rear', t, 18.0 * est.state().k / 3.6)
        lat, lon = _latlon_of(x0 + 5.0 * t, y0)
        pos.fix(t, lat, lon, 150.0)
        pos.update(t, est)
    pos.update(5.3, est)                          # окно 5 с от первого фикса (0,1 с) закончилось
    assert pos.locked and not pos.init.still
    x, y, _, _, _ = pos.pose(est.state().s)
    assert abs(x - (x0 + 25.0 + 9.873)) < 1.0 and abs(y - y0) < 0.5
