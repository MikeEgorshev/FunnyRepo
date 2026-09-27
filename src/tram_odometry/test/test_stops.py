from synthetic import Scenario, run

from tram_odometry.estimator import Estimator
from tram_odometry.stops import StopFixer, StopParams


def _replay(events, estimator, fixer=None):
    for t, kind, value in events:
        if kind == 'cmd':
            estimator.set_notch(t, value)
        else:
            estimator.wheel(kind, t, value)
            if fixer:
                fixer.update(t, estimator)


def test_stop_fixes_bound_drift_and_learn_the_wheel_scale():
    events, truth = run(Scenario(k=3.63, duration=480.0))      # k фильтра ошибается на 0,9 %
    stops = [(truth[int((80 * c + 79) / 0.01)][1], 1.0) for c in range(6)]
    plain, fixed = Estimator(), Estimator()
    fixer = StopFixer(stops, params=StopParams(min_sigma_m=1.0))   # места стоянок точные, до 1 м
    _replay(events, plain)
    _replay(events, fixed, fixer)
    s_true = truth[-1][1]
    assert abs(plain.state().s - s_true) > 20.0
    assert abs(fixed.state().s - s_true) < 2.0
    assert fixer.fixes >= 5
    assert abs(fixed.state().k - 3.63) < 0.006


def test_no_fix_when_stop_is_ambiguous_or_absent():
    fixer = StopFixer([(100.0, 1.0), (130.0, 1.0), (500.0, 1.0)])
    assert len(fixer.candidates(115.0, var_s=100.0)) == 2      # две стоянки в гейте
    assert fixer.candidates(300.0, var_s=100.0) == []           # светофор: стоянки нет
    assert len(fixer.candidates(495.0, var_s=4.0)) == 1


def test_closed_loop_wraps_distance():
    fixer = StopFixer([(10.0, 1.0)], length=1000.0)
    (_, d, _), = fixer.candidates(1995.0, var_s=25.0)            # второй круг
    assert abs(d - 15.0) < 1e-9


def test_one_fix_per_dwell_and_only_after_dwell_time():
    est = Estimator()
    fixer = StopFixer([(0.0, 1.0)], params=StopParams(dwell_s=5.0))
    applied = []
    for i in range(100):                                        # 10 с на месте
        t = i * 0.1
        est.set_notch(t, -4)
        est.wheel('front', t, 0.0)
        est.wheel('rear', t, 0.0)
        d = fixer.update(t, est)
        if d is not None:
            applied.append(t)
    assert len(applied) == 1 and applied[0] >= 5.0
