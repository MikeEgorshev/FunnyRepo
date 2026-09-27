from tram_odometry.slip import SlipDetector, SlipParams, pick_speed


def test_pick_speed_by_phase():
    assert pick_speed(10.0, 9.0, 5) == 9.0      # тяга: медленная
    assert pick_speed(10.0, 9.0, -5) == 10.0    # торможение: быстрая
    assert pick_speed(10.0, 9.0, 0) == 9.5      # выбег: среднее
    assert pick_speed(None, 9.0, 5) == 9.0
    assert pick_speed(None, None, 5) is None


def test_inconsistency_needs_consecutive_samples():
    det = SlipDetector(SlipParams(consistency_count=2))
    det.wheel('front', 0.0, 10.0)
    det.wheel('rear', 0.0, 10.0)
    assert not det.check(0.0, 3, 10.0)[1].inconsistent
    det.wheel('front', 0.1, 11.0)
    assert not det.check(0.1, 3, 10.0)[1].inconsistent
    det.wheel('rear', 0.2, 10.0)
    _, flags = det.check(0.2, 3, 10.0)
    assert flags.inconsistent and flags.any


def test_wheel_acceleration_limit_excludes_the_bogie():
    det = SlipDetector()
    det.wheel('front', 0.0, 5.0)
    det.wheel('rear', 0.0, 5.0)
    det.wheel('front', 0.3, 6.5)                # +5 м/с²: буксование
    det.wheel('rear', 0.3, 5.3)                 # +1 м/с²: норма
    z, flags = det.check(0.3, 0, 5.0)
    assert flags.front_accel and not flags.rear_accel
    assert z == 5.3


def test_burst_of_messages_does_not_fake_acceleration():
    det = SlipDetector()
    det.wheel('front', 0.0, 5.0)
    det.wheel('front', 0.001, 5.02)             # пачка: интервал 1 мс
    assert det.front.accel == 0.0


def test_stale_bogie_is_ignored():
    det = SlipDetector(SlipParams(stale_s=0.3))
    det.wheel('front', 0.0, 5.0)
    det.wheel('rear', 1.0, 6.0)
    assert det.fresh_speeds(1.0) == (None, 6.0)
    assert det.check(1.0, 3, 6.0)[0] == 6.0


def test_frozen_sensor_is_dropped_until_it_changes():
    det = SlipDetector(SlipParams(stuck_n=6))
    for i in range(10):                              # задняя тележка застыла на 5,0 м/с
        t = i * 0.1
        det.wheel('front', t, 5.0 + 0.01 * i)
        det.wheel('rear', t, 5.0)
    z, flags = det.check(0.9, 3, 5.0)
    assert flags.rear_frozen and not flags.front_frozen and not flags.any
    assert det.fresh_speeds(0.9) == (5.09, None) and z == 5.09
    det.wheel('rear', 1.0, 5.2)                      # показание сдвинулось — датчик снова в деле
    assert det.fresh_speeds(1.0)[1] == 5.2


def test_zeros_at_standstill_are_not_a_frozen_sensor():
    det = SlipDetector()
    for i in range(20):
        det.wheel('front', i * 0.1, 0.0)
    assert not det.front.frozen(det.p.stuck_n)


def test_agreeing_bogies_are_averaged_not_min_max():
    # разница в пределах шума — среднее; настоящее расхождение — по фазе
    assert abs(pick_speed(10.00, 10.06, notch=5, mean_band=0.1) - 10.03) < 1e-9
    assert pick_speed(10.0, 11.0, notch=5, mean_band=0.1) == 10.0
    assert pick_speed(10.0, 11.0, notch=-5, mean_band=0.1) == 11.0


def test_older_bogie_sample_is_brought_to_the_current_time():
    det = SlipDetector()
    for i in range(10):                       # разгон 1 м/с², задняя приходит на 0,05 с позже
        t = i * 0.1
        det.wheel('front', t, 5.0 + t)
        det.wheel('rear', t + 0.05, 5.0 + t + 0.05)
    front, rear = det.fresh_speeds(1.0)       # передняя — от 0,9 с, задняя — от 0,95 с
    assert abs(front - 6.0) < 0.01 and abs(rear - 6.0) < 0.01
