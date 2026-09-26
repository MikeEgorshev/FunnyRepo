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
