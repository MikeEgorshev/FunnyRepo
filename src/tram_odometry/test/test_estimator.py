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


def drive(est, t_end, v_of_t, front=lambda t, v: v, rear=lambda t, v: v, u=10, dt=0.1, errors=None):
    """Подаёт команды 20 Гц и колёса 10 Гц; front/rear могут искажать скорость (None — пропуск).

    u — позиция контроллера или функция u(t) (None — команды нет); в errors копится |v − истина|.
    """
    t, out = 0.0, None
    while t < t_end:
        t = round(t + dt / 2, 6)
        cmd = u(t) if callable(u) else u
        if cmd is not None:
            out = est.on_cmd(t, cmd) or out
        t = round(t + dt / 2, 6)
        v = v_of_t(t)
        for fn, is_front in ((front, True), (rear, False)):
            z = fn(t, v)
            if z is not None:
                out = est.on_wheel(t, is_front, z * K) or out
        if errors is not None and out is not None:
            errors.append(abs(out['v'] - v))
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


def test_rejects_slip_of_both_bogies():
    est = make()
    slip = lambda t, v: v + 2.0 * min(1.0, (t - 8.0) / 0.3) if 8.0 < t < 13.0 else v  # noqa: E731 — буксуют обе
    errors = []
    out = drive(est, 20.0, lambda t: 0.5 * t, front=slip, rear=slip, errors=errors)
    assert max(errors) < 0.6 and out['slip'] is False
    assert abs(out['v'] - 10.0) < 0.15
    assert abs(out['s'] - 200.0) < 3.0


def test_single_spikes_after_slip_do_not_move_speed():
    est = make()
    slip = lambda t, v: v + 2.0 if 8.0 < t < 9.5 else v  # noqa: E731 — буксование обеих тележек
    junk = {10.0: 0.0, 10.5: 3.0, 11.0: 30.0 / K}           # выбросы передней: ноль, ×3, +30 км/ч

    def front(t, v):
        for tj, val in junk.items():
            if abs(t - tj) < 1e-6:
                return val if val != 3.0 else v * 3.0
        return slip(t, v)
    errors = []
    drive(est, 14.0, lambda t: 0.5 * t, front=front, rear=slip, errors=errors)
    assert max(errors[95:]) < 0.5


def test_dead_front_sensor_reading_zero_under_traction():
    est = make()
    dead = lambda t, v: 0.0 if 6.0 < t < 16.0 else v  # noqa: E731 — передний датчик на ходу показывает 0
    errors = []
    out = drive(est, 20.0, lambda t: 0.5 * t, front=dead, errors=errors)
    assert max(errors) < 0.3
    assert abs(out['s'] - 200.0) < 3.0


def test_follows_emergency_braking_not_seen_by_controller():
    est = make()
    errors = []
    v_true = lambda t: 0.5 * t if t < 16.0 else max(0.0, 8.0 - 4.0 * (t - 16.0))  # noqa: E731 — −4 м/с²
    out = drive(est, 25.0, v_true, u=lambda t: 10 if t < 16.0 else 0, errors=errors)
    assert max(errors[170:]) < 1.5 and abs(out['v']) < 0.1


def test_follows_wheels_when_commands_stop():
    est = make()
    errors = []
    # команды пропали на 5-й секунде, а с 8-й трамвай идёт выбегом: модель ручку не знает
    v_true = lambda t: 0.5 * t if t < 8.0 else 4.0 - 0.05 * (t - 8.0)  # noqa: E731
    out = drive(est, 16.0, v_true, u=lambda t: 10 if t < 5.0 else None, errors=errors)
    assert max(errors) < 0.3
    assert abs(out['v'] - 3.6) < 0.1


def test_learns_wheel_scale_from_stops():
    """Колёса врут на +1,5 %: по привязкам к стоянкам фильтр учит масштаб, ошибка к стоянке падает."""
    p = Params()
    p.wheel_kmh_per_mps = K
    p.sigma_c0, p.sigma_u = 0.007, 0.0  # сам механизм: смелый масштаб, без необъяснённой ошибки пути
    stop_s = [700.0, 1300.0, 1900.0, 2500.0, 3100.0, 3700.0]
    est = TramEstimator(straight_map(), flat_model(), [(x, 0.5) for x in stop_s], p)
    la, lo, al = Enu(*ORIGIN).inverse(100.0, 0.0, 0.0)
    est.on_gnss(0.0, la, lo, al)
    t, s, v, dt, a_max = 0.0, 100.0, 0.0, 0.05, 0.8
    before_snap = []
    for target in stop_s:
        stand = 0.0
        while stand < 4.0:
            left = target - s
            if left <= 0.05 and v < 0.05:
                v, u = 0.0, 0
                stand += dt
            elif v * v / (2 * a_max) >= left:
                v, u = max(0.0, v - a_max * dt), -10
            else:
                v, u = min(10.0, v + a_max * dt), 10 if v < 10.0 else 0
            s += v * dt
            t = round(t + dt, 6)
            est.on_cmd(t, u)
            if round(t / dt) % 2 == 0:
                for front in (True, False):
                    est.on_wheel(t, front, v * 1.015 * K + (0.01 if front else 0.0) * (round(t / dt) % 4 - 1))
            if 1.0 < stand <= 1.0 + dt:
                before_snap.append(est.s - s)
    assert abs(est.c - 0.015) < 0.004
    assert abs(before_snap[0]) > 6.0            # до первой привязки масштаб неизвестен: ~9 м на 600 м
    assert max(abs(e) for e in before_snap[2:]) < 3.0


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


def test_rover_heading_picks_track_of_travel_direction():
    """Два пути в разные стороны в 5 м друг от друга; трамвай стоит посередине лицом на восток."""
    enu = Enu(*ORIGIN)
    pts = [(float(x), 0.0) for x in range(0, 1001)]                          # на восток по y = 0
    pts += [(1000.0 + 2.5 * math.sin(a / 10 * math.pi), 2.5 - 2.5 * math.cos(a / 10 * math.pi))
            for a in range(1, 10)]                                            # разворот
    pts += [(float(x), 5.0) for x in range(1000, -1, -1)]                     # на запад по y = 5
    s, lat, lon, alt, acc = [], [], [], [], 0.0
    for i, (x, y) in enumerate(pts):
        if i:
            acc += math.hypot(x - pts[i - 1][0], y - pts[i - 1][1])
        la, lo, al = enu.inverse(x, y, 0.0)
        s.append(acc)
        lat.append(la)
        lon.append(lo)
        alt.append(al)
    p = Params()
    p.wheel_kmh_per_mps = K
    p.output_frame = 'enu'
    for rover_dx, s_expected in ((12.436, 400.0), (-12.436, None)):
        est = TramEstimator(RouteMap(s, lat, lon, alt), flat_model(), [], p)
        for i in range(10):
            est.on_gnss_rover(0.1 * i, *enu.inverse(400.0 + rover_dx, 2.6, 0.0))
            est.on_gnss(0.1 * i + 0.05, *enu.inverse(400.0, 2.6, 0.0))
        if s_expected is not None:
            assert abs(est.s - s_expected) < 1.0                              # путь на восток
        else:
            assert abs(est.s - (acc - 400.0)) < 1.0                           # лицом на запад — встречный путь


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


def test_predict_output_extrapolates_without_changing_the_filter():
    est = make()
    out = drive(est, 10.0, lambda t: 0.5 * t)
    s, v, t, last = est.s, est.v, est.t, est.last_out
    ahead = est.predict_output(t + 1.0)            # все входы молчат секунду
    assert ahead is not None and ahead['stamp'] == t + 1.0
    assert ahead['s'] > s + 0.9 * v                # прогноз едет дальше по модели
    assert (est.s, est.v, est.t) == (s, v, t)      # сам фильтр не сдвинулся
    assert est.predict_output(t - 0.5) is None     # назад не прогнозируем
    assert out['stamp'] == last and est.last_out == t + 1.0
