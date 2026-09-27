"""Position fixes must not change any subsequent wheel/model velocity."""
import copy

import pytest

from test_integrated import make, drive, ORIGIN


def fix(e, stamp, along):
    i = round(along)
    return e.on_gnss(stamp, e.map.lat[i], e.map.lon[i], e.map.alt[i])


def test_correction_improves_position_without_affecting_future_speed():
    a = make()
    a.on_gnss(0., *ORIGIN)
    drive(a)
    b = copy.deepcopy(a)
    b.p.gnss_corrections = False
    for i in range(80, 201):
        t = i/10
        for e in (a, b):
            e.on_cmd(t, 0)
            e.on_wheel(t, False, 18.)
            e.on_wheel(t, True, 18.)
        # A systematic 10 m along-track displacement; sparse fixes stop at 12 s.
        if i in (80, 90, 110, 120):
            for e in (a, b):
                fix(e, t, e.s+10)
        assert (a.s, a.v, a.c, a.d, a.P) == (b.s, b.v, b.c, b.d, b.P)
        assert a.model.traction_gain == b.model.traction_gain
        assert a.predict_output(t+.04)['v'] == b.predict_output(t+.04)['v']
    assert a.position_correction.accepted >= 2
    assert b.position_correction.accepted == 0
    target = a.map.pose(a.s+10)[0]+9.873
    assert abs(a._output(20.)['x']-target) < abs(b._output(20.)['x']-target)
    assert a.map.active_spur is b.map.active_spur is None


def test_stale_duplicate_and_outlier_fixes_do_not_move_anchor():
    e = make()
    e.on_gnss(0., *ORIGIN)
    drive(e)
    fix(e, 8., e.s+10)
    fix(e, 9., e.s+e.v+10)
    correction = e.position_correction
    assert correction.accepted == 1
    before = (correction.anchor, correction.measured, correction.last_fix)
    for t, along in [(9., 100), (0.1, 50), (99., 50), (9.1, 1000)]:
        fix(e, t, along)
    assert (correction.anchor, correction.measured, correction.last_fix) == before


def test_spur_switch_requires_two_fixes_and_does_not_mutate_ekf_map():
    e = make()
    e.on_gnss(0., *ORIGIN)
    drive(e)
    x, y, z, _ = e.map.pose(e.s)
    spur = dict(s_join=100., d=[0., 100.], x=[x-50, x+50],
                y=[y+5, y+5], z=[z, z])
    e.map.spurs = [spur]
    e.enu.forward = lambda *args: args
    e.on_gnss(8., x, y+5, z)
    assert e.position_correction.route is None
    e.on_gnss(9., x+e.v, y+5, z)
    assert e.position_correction.route.active_spur is spur
    assert e.map.active_spur is None
    assert e._output(9.)['y'] == pytest.approx(y+5)


def test_spur_reversal_keeps_body_orientation_and_position_ignores_stop_snap():
    e = make()
    e.on_gnss(0., *ORIGIN)
    drive(e)
    x, y, z, _ = e.map.pose(e.s)
    spur = dict(s_join=100., d=[0., 100.], x=[x+50, x-50],
                y=[y+5, y+5], z=[z, z])
    e.map.spurs = [spur]
    e.enu.forward = lambda *args: args
    e.on_gnss(8., x, y+5, z)
    e.on_gnss(9., x+e.v, y+5, z)
    c = e.position_correction
    assert c.direction == -1
    initial = e._output(9.)
    # A map snap in the velocity EKF cannot teleport the corrected position.
    e.s += 20.
    assert e._output(9.)['x'] == initial['x']
    # Later GNSS proves a reversal, while the body still faces the same way.
    for t, dx in [(10., 0.), (11., -5.), (12., -10.)]:
        e.t = t
        e.travel += 5.
        e.on_gnss(t, x+dx, y+5, z)
    assert c.direction == 1
    assert e._output(12.)['yaw'] == pytest.approx(initial['yaw'])
