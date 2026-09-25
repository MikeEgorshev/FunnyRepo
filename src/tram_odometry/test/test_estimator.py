"""Оценщик на синтетике: прямой маршрут, постоянная тяга, помехи колёс."""
import math

from tram_odometry.estimator import Params, TramEstimator
from tram_odometry.geo import Enu
from tram_odometry.model import TractionModel
from tram_odometry.route_map import RouteMap

ORIGIN = (55.80, 37.42, 150.0)
K = 3.6


def straight_map(length=5000.0):
    enu = Enu(*ORIGIN)
    s, lat, lon, alt = [], [], [], []
    for i in range(int(length) + 1):
        la, lo, al = enu.inverse(float(i), 0.0, 0.0)
        s.append(float(i))
        lat.append(la)
        lon.append(lo)
        alt.append(al)
    return RouteMap(s, lat, lon, alt)


def flat_model(a_traction=0.5):
    """u > 0 — постоянное ускорение, u < 0 — торможение, 0 — выбег без сопротивления."""
    rows = {u: [a_traction if u > 0 else (-0.8 if u < 0 else 0.0)] * 2 for u in range(-15, 16)}
    return TractionModel([0.5, 15.5], rows, delay_s=0.0)


def make(**over):
    p = Params()
    p.wheel_kmh_per_mps = K
    for k, v in over.items():
        setattr(p, k, v)
    est = TramEstimator(straight_map(), flat_model(), [], p)
    enu = Enu(*ORIGIN)
    la, lo, al = enu.inverse(100.0, 0.0, 0.0)
    est.on_gnss(0.0, la, lo, al)
    return est


def drive(est, t_end, v_of_t, front=lambda t, v: v, rear=lambda t, v: v, u=10, dt=0.1):
    """Подаёт команды 20 Гц и колёса 10 Гц; front/rear могут искажать скорость (None — пропуск)."""
    t, out = 0.0, None
    while t < t_end:
        t = round(t + dt / 2, 6)
        out = est.on_cmd(t, u) or out
        t = round(t + dt / 2, 6)
        v = v_of_t(t)
        for fn, is_front in ((front, True), (rear, False)):
            z = fn(t, v)
            if z is not None:
                out = est.on_wheel(t, is_front, z * K) or out
    return out


def test_tracks_clean_acceleration():
    est = make()
    out = drive(est, 20.0, lambda t: 0.5 * t)
    assert abs(out['v'] - 10.0) < 0.1
    assert abs(out['s'] - (100.0 + 0.25 * 20 ** 2)) < 2.0


def test_rejects_slipping_front_bogie_under_traction():
    est = make()
    slip = lambda t, v: v * 1.3 if 8.0 < t < 14.0 else v  # noqa: E731 — буксование передней тележки
    out = drive(est, 20.0, lambda t: 0.5 * t, front=slip)
    assert abs(out['v'] - 10.0) < 0.15
    assert abs(out['s'] - 200.0) < 3.0


def test_survives_full_wheel_dropout_on_model():
    est = make()
    gap = lambda t, v: None if 5.0 < t < 15.0 else v  # noqa: E731 — обе тележки молчат 10 с
    out = drive(est, 20.0, lambda t: 0.5 * t, front=gap, rear=gap)
    assert abs(out['v'] - 10.0) < 0.3
    assert abs(out['s'] - 200.0) < 5.0


def test_ignores_frozen_rear_sensor():
    est = make()
    frozen = lambda t, v: 5.0 if 6.0 < t < 16.0 else v  # noqa: E731 — задний датчик застыл на 5 м/с
    out = drive(est, 20.0, lambda t: 0.5 * t, rear=frozen)
    assert abs(out['v'] - 10.0) < 0.15
    assert abs(out['s'] - 200.0) < 3.0


def test_starts_on_the_move():
    est = make()
    out = drive(est, 10.0, lambda t: 8.0, u=0)  # уже едем 8 м/с, выбег
    assert abs(out['v'] - 8.0) < 0.1


def test_relative_odometry_without_gnss():
    p = Params()
    p.wheel_kmh_per_mps = K
    est = TramEstimator(straight_map(), flat_model(), [], p)  # GNSS так и не придёт
    out = drive(est, 20.0, lambda t: 0.5 * t)
    assert out['frame'] == 'odom' and out['y'] == 0.0
    assert abs(out['x'] - (0.25 * 20 ** 2 - 0.25 * 5 ** 2)) < 3.0  # путь с момента перехода в режим


def test_gnss_jump_during_init_is_ignored():
    est = make()
    enu = Enu(*ORIGIN)
    for i, x in enumerate([100.0, 100.2, 99.9, 160.0, 100.1]):  # одна точка прыгнула на 60 м
        la, lo, al = enu.inverse(x, 0.0, 0.0)
        est.on_gnss(0.1 * (i + 1), la, lo, al)
    assert abs(est.s - 100.0) < 1.0


def test_garbage_inputs_do_not_crash():
    est = make()
    for i, z in enumerate([float('nan'), -5.0, 1e6, 0.0, 30.0]):
        est.on_wheel(1.0 + i * 0.1, True, z)
        est.on_cmd(1.0 + i * 0.1, 99)
    out = est.on_wheel(2.0, False, 36.0)
    assert out is None or (math.isfinite(out['v']) and math.isfinite(out['x']))
