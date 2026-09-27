import math

from tram_odometry.model import ModelParams, accel, accel_dv, brake, resistance, traction

P = ModelParams()


def test_traction_grows_with_notch_and_is_zero_otherwise():
    values = [traction(n, 5.0, P) for n in range(0, 16)]
    assert values[0] == 0.0
    assert all(b > a for a, b in zip(values, values[1:]))
    assert traction(-5, 5.0, P) == 0.0


def test_traction_is_constant_force_then_constant_power():
    assert traction(15, 2.0, P) == traction(15, P.v_base, P) == P.a_max
    # выше базовой скорости a·v постоянно
    assert math.isclose(traction(15, 10.0, P) * 10.0, traction(15, 14.0, P) * 14.0)


def test_brake_only_in_brake_notches_and_fades_at_low_speed():
    assert brake(5, 5.0, P) == 0.0
    assert math.isclose(brake(-15, 10.0, P), P.b_max)
    assert brake(-15, 0.0, P) == P.b_max * P.blend_floor
    assert accel(-8, 5.0, P) < 0.0


def test_resistance_is_continuous_and_zero_at_standstill():
    assert resistance(0.0, P) == 0.0
    assert resistance(P.v_stop - 1e-6, P) < resistance(P.v_stop, P) + 1e-6
    assert accel(0, 0.0, P) == 0.0
    assert accel(0, 10.0, P) < 0.0


def test_grade_acts_against_uphill_motion():
    assert accel(0, 5.0, P, grade=0.01) < accel(0, 5.0, P)


def test_derivative_is_finite_everywhere():
    for n in (-15, -4, 0, 3, 15):
        for v in (0.0, 0.05, 0.3, P.v_base, 20.0):
            assert math.isfinite(accel_dv(n, v, P))
            assert abs(accel_dv(n, v, P)) < 20.0
