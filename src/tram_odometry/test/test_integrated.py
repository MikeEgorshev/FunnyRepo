"""Regressions for strict inputs, causal prediction and model adaptation."""
import copy
import math

import pytest

from tram_odometry.estimator import Params
from tram_odometry.integrated import IntegratedEstimator
from tram_odometry.integrated_model import IntegratedModel
from tram_odometry.geo import UtmLocal
from test_estimator import straight_map, flat_model, ORIGIN


def make(**kwargs):
    p = Params()
    for key, value in kwargs.items():
        setattr(p, key, value)
    base = flat_model()
    model = IntegratedModel(base.v_nodes, base.rows, delay_s=0.0)
    return IntegratedEstimator(straight_map(), model, params=p)


def drive(e, end=8):
    for i in range(1, end * 10):
        t = i / 10
        e.on_cmd(t, 0)
        e.on_wheel(t, False, 18 + 0.001 * (i % 3))
        e.on_wheel(t, True, 18 + 0.001 * (i % 3))


def test_organizer_coordinate_example():
    x, y, _ = UtmLocal().forward(55.8088325462547, 37.4602768500852, 0)
    assert x == pytest.approx(103501.6309, abs=0.0001)
    assert y == pytest.approx(85876.1201, abs=0.0001)


def test_rigid_master_tf_not_distance_along_curved_map():
    e = make()
    e.on_gnss(0., *ORIGIN)
    e.s = 10.
    e.map.pose = lambda s: (s, s*s, 150., math.pi/2)
    out = e._output(1.)
    assert out['x'] == pytest.approx(10.)
    assert out['y'] == pytest.approx(109.873)
    assert out['z'] == pytest.approx(147.)


def test_no_future_command_before_delay():
    m = make().model
    m.delay_s = .4
    m.push_command(0., 0)
    m.push_command(1., 15)
    assert m.command_at(1.39) == 0.
    assert 0 < m.command_at(1.5) < 15


def test_late_gnss_and_reference_cannot_change_state_even_with_flags():
    a = make(gnss_corrections=True, primary_sync=True)
    a.on_gnss(0., *ORIGIN)
    drive(a)
    b = copy.deepcopy(a)
    for t in [8., 0.1, 90.]:
        assert a.on_gnss(t, 55.9, 37.6, 200.) is None
        assert a.on_gnss_rover(t, 55.9, 37.6, 200.) is False
        a.on_primary(t, 100., 100., 0.)
    assert (a.s, a.v, a.d, a.c, a.P) == (b.s, b.v, b.d, b.c, b.P)
    assert not a.p.primary_sync and not a.p.gnss_corrections


def test_no_gnss_start_has_relative_output():
    e = make()
    drive(e)
    assert e.mode == 'relative'
    assert e.predict_output(8.)['frame'] == 'odom'


def test_prediction_does_not_contaminate_filter_and_survives_long_gap():
    e = make()
    e.on_gnss(0., *ORIGIN)
    drive(e)
    b = copy.deepcopy(e)
    first = e.predict_output(8.)
    for i in range(1, 1501):
        out = e.predict_output(8. + i * .04)
        assert all(math.isfinite(out[k]) for k in ['v', 'x', 'y', 'z', 'var_s'])
    assert out['var_s'] > first['var_s']
    assert (e.s, e.v, e.d, e.P, e.t, e.v_hist) == (b.s, b.v, b.d, b.P, b.t, b.v_hist)
    a_out = e.on_wheel(69., True, 18.01)
    b_out = b.on_wheel(69., True, 18.01)
    assert a_out == b_out


def test_duplicate_and_nonfinite_wheels_do_not_reuse_measurement():
    e = make()
    drive(e)
    before = (e.s, e.v, copy.deepcopy(e.P))
    assert e.on_wheel(7.9, True, 30.) is None
    assert e.on_wheel(9., True, math.nan) is None
    assert (e.s, e.v, e.P) == before


def test_actuator_lag_is_causal_and_queries_are_order_independent():
    e = make()
    m = e.model
    m.push_command(1., 10)
    expected = 10 * (1 - math.exp(-.1/m.tau_s))
    assert m.command_at(1.1) == pytest.approx(expected)
    m.push_command(2., -10)
    m.command_at(20.)
    assert m.command_at(1.1) == pytest.approx(expected)
    assert m.command_at(.9) == 0.


def test_traction_gain_adapts_from_independent_acceleration_and_is_bounded():
    m = make().model
    for _ in range(3000):
        m.adapt(10, 5., .6, 0., .1)
    assert 1.1 < m.traction_gain <= 1.2
    previous = m.traction_gain
    for _ in range(100):
        m.adapt(10, 5., 20., 0., .1)
    assert m.traction_gain == previous


def test_locked_wheels_do_not_trigger_stop_snap_while_model_moves():
    e = make()
    e.on_gnss(0., *ORIGIN)
    e.v = 5.
    e.last_wheel = {True: (10., 0.), False: (10., 0.)}
    e.bad[True] = (10., 'skid')
    e.stop_since = 0.
    e._stop_logic(10.)
    assert e.stop_since is None
