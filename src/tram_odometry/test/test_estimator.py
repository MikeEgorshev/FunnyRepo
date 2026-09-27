import math

from synthetic import Scenario, naive_position, replay, run

from tram_odometry.estimator import Estimator


def errors(out):
    v_rmse = math.sqrt(sum((o[4] - o[2]) ** 2 for o in out) / len(out))
    s_err = [o[3] - o[1] for o in out]
    return v_rmse, max(abs(e) for e in s_err), s_err[-1]


def test_clean_run_tracks_speed_and_distance():
    events, truth = run(Scenario())
    out = replay(Estimator(), events, truth)
    v_rmse, s_max, s_end = errors(out)
    dist = truth[-1][1]
    assert v_rmse < 0.1
    assert abs(s_end) < 0.003 * dist + 1.0      # масштаб колёс в фильтре ошибается на 0,1 %
    assert not any(o[5] for o in out)            # на чистых данных флага нет


def test_slip_is_flagged_and_does_not_drag_the_estimate():
    sc = Scenario(slip=[(82.0, 96.0)], slip_ratio=0.25)
    events, truth = run(sc)
    out = replay(Estimator(), events, truth)
    v_rmse, s_max, s_end = errors(out)
    assert any(o[5] for o in out if 82.0 <= o[0] < 96.0)
    assert abs(s_end) < abs(naive_position(events, truth, 3.5966))
    assert v_rmse < 0.3


def test_dropout_of_both_bogies_is_bridged_by_the_model():
    sc = Scenario(dropout=[(120.0, 140.0)])
    events, truth = run(sc)
    out = replay(Estimator(), events, truth)
    _, s_max, s_end = errors(out)
    naive = naive_position(events, truth, 3.5966)
    assert abs(naive) > 20.0                      # держать последнюю скорость в торможении — плохо
    assert s_max < 10.0
    assert abs(s_end) < abs(naive)


def test_slide_with_locked_wheels_does_not_stop_the_tram():
    est = Estimator()
    t = 0.0
    for _ in range(50):                           # ровный ход 10 м/с
        t += 0.1
        est.set_notch(t, 0)
        est.wheel('front', t, 36.0)
        est.wheel('rear', t, 36.0)
    for _ in range(3):                            # обе тележки встали: юз, а не остановка
        t += 0.1
        est.set_notch(t, -8)
        est.wheel('front', t, 0.0)
        est.wheel('rear', t, 0.0)
    st = est.state()
    assert st.v > 8.0 and st.slip


def test_standstill_pins_speed_to_zero():
    est = Estimator()
    for i in range(1, 30):
        t = i * 0.1
        est.set_notch(t, -4)
        est.wheel('front', t, 0.02)
        est.wheel('rear', t, 0.0)
    assert est.state().v < 0.01


def test_run_that_starts_moving_takes_speed_from_wheels():
    est = Estimator()
    est.set_notch(0.0, 3)
    est.wheel('front', 0.0, 36.0)
    assert abs(est.state().v - 36.0 / est.state().k) < 1e-9
    assert not est.state().slip


def test_time_jump_back_resets_and_old_samples_are_dropped():
    est = Estimator()
    est.wheel('front', 100.0, 36.0)
    est.wheel('front', 100.1, 36.0)
    v = est.state().v
    est.wheel('front', 99.5, 0.0)                # опоздал на 0,6 с: отбрасываем
    est.wheel('rear', 98.9, 0.0)                 # задняя на 1,2 с позади (так бывает в начале прогона)
    assert est.state().v == v and est.resets == 0
    est.wheel('front', 10.0, 0.0)                # новый прогон
    assert est.resets == 1 and est.state().t == 10.0


def test_position_fix_pulls_distance():
    est = Estimator()
    est.wheel('front', 0.0, 36.0)
    est.advance(10.0)
    s_before = est.state().s
    est.position_fix(10.0, s_before + 20.0, 2.0)
    assert est.state().s > s_before + 15.0


def test_bad_values_are_ignored():
    est = Estimator()
    est.wheel('front', 0.0, float('nan'))
    est.wheel('front', 0.1, None)
    assert est.t is None                         # мусор не запускает часы фильтра


def test_stop_fixes_teach_the_wheel_scale():
    events, truth = run(Scenario(k=3.63, duration=480.0))
    est = Estimator()
    ti, fixed = 0, set()
    for t, kind, value in events:
        if kind == 'cmd':
            est.set_notch(t, value)
        else:
            est.wheel(kind, t, value)
        cycle = int(t // 80)
        if t % 80 > 74 and cycle not in fixed:      # стоянка у «известной остановки»
            while ti + 1 < len(truth) and truth[ti + 1][0] <= t:
                ti += 1
            est.position_fix(t, truth[ti][1], 1.0)
            fixed.add(cycle)
    assert abs(est.state().k - 3.63) < 0.006        # старт 3.5966, ошибка была 0,9 %


def test_state_at_predicts_without_moving_the_filter():
    est = Estimator()
    est.set_notch(0.0, 0)
    est.wheel('front', 0.0, 36.0)
    ahead = est.state_at(0.4)
    assert est.t == 0.0 and ahead.t == 0.4 and ahead.s > 3.9
    est.wheel('front', 0.1, 36.0)                # не считается опоздавшим
    assert est.t == 0.1


def test_disturbance_does_not_carry_over_a_stop_into_the_next_start():
    est = Estimator()
    t = 0.0
    for _ in range(100):                          # торможение сильнее модели: d уходит в минус
        t += 0.1
        est.set_notch(t, 0)
        v = max(0.0, 8.0 - 1.2 * t)
        est.wheel('front', t, v * 3.5966)
        est.wheel('rear', t + 0.005, v * 3.5966)
    for _ in range(30):                           # стоянка
        t += 0.1
        est.set_notch(t, -2)
        est.wheel('front', t, 0.0)
        est.wheel('rear', t + 0.005, 0.0)
    t0 = t
    for _ in range(40):                           # трогание: 1 м/с², обе тележки согласны
        t += 0.1
        est.set_notch(t, 8)
        est.wheel('front', t, (t - t0) * 3.5966)
        est.wheel('rear', t + 0.005, (t - t0) * 3.5966)
    assert abs(est.state().v - (t - t0)) < 0.3 and not est.state().slip


def test_state_at_does_not_change_the_filter_or_the_wheel_scale_distance():
    est = Estimator()
    for i in range(20):
        t = 0.1 * i
        est.set_notch(t, 5)
        est.wheel('front', t, 36.0)
        est.wheel('rear', t + 0.005, 36.0)
    before = (list(est.x), est.t, est.odo)
    for k in range(50):                           # таймер 25 Гц: прогнозы вперёд с колёсами и без
        est.state_at(est.t + 0.04 * (k % 5 + 1))
    est.detector.front.t = est.detector.rear.t = -10.0     # колёса «пропали» — прогноз по модели
    for k in range(50):
        est.state_at(est.t + 0.04 * (k % 5 + 1))
    assert (list(est.x), est.t, est.odo) == before
