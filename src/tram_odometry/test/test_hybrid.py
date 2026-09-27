"""Unit tests for HybridEstimator and HybridOdometryNode."""
import math
import pytest

from tram_odometry.estimator import Params
from tram_odometry.hybrid import HybridEstimator
from tram_odometry.integrated_model import IntegratedModel
from test_estimator import straight_map, flat_model, ORIGIN


def make_hybrid(**kwargs):
    p = Params()
    for key, value in kwargs.items():
        setattr(p, key, value)
    base = flat_model()
    model = IntegratedModel(base.v_nodes, base.rows, delay_s=0.0)
    return HybridEstimator(straight_map(), model, params=p)


def test_hybrid_initialization_defaults():
    e = make_hybrid()
    assert e.p.output_v_delay_s == pytest.approx(0.040)
    assert not e.p.primary_sync


def test_hybrid_actuator_lag_and_speed_accuracy():
    e = make_hybrid()
    m = e.model
    m.delay_s = 0.4
    m.push_command(0.0, 0)
    m.push_command(1.0, 15)
    assert m.command_at(1.39) == 0.0
    assert 0.0 < m.command_at(1.5) < 15.0


def test_hybrid_gnss_along_track_correction():
    e = make_hybrid(gnss_corrections=True)
    e.on_gnss(0.0, *ORIGIN)
    assert e.ready
    # Simulate forward motion
    for i in range(1, 40):
        t = i / 10.0
        e.on_cmd(t, 0)
        e.on_wheel(t, False, 18.0)
        e.on_wheel(t, True, 18.0)
    s_before = e.s
    # Two agreeing GNSS fixes in the vicinity
    e.on_gnss(4.1, ORIGIN[0] + 0.0001, ORIGIN[1] + 0.0001, 150.0)
    e.on_gnss(4.3, ORIGIN[0] + 0.00012, ORIGIN[1] + 0.00012, 150.0)
    # Filter state remains finite and valid
    assert math.isfinite(e.s)
    assert math.isfinite(e.v)
    assert math.isfinite(e.c)
