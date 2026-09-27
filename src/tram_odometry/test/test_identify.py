import random

from synthetic import Scenario, run

from tram_odometry.identify import fit, lagged, open_loop_errors, samples_from_run
from tram_odometry.model import ModelParams

TRUTH = ModelParams(a_max=1.25, v_base=6.0, b_max=1.45, v_blend=2.0, blend_floor=0.7,
                    res_a=0.02, res_b=0.001, res_c=0.0006, tau_cmd=0.75)


def _schedule(seed, duration):
    """Случайная езда: держим позицию 4–20 с; тяга, выбег и тормоз вперемешку."""
    rnd = random.Random(seed)
    out, t = [], 0.0
    while t < duration:
        phase = rnd.random()
        n = rnd.randint(1, 15) if phase < 0.45 else (0 if phase < 0.7 else -rnd.randint(1, 15))
        out.append((t, n))
        t += rnd.uniform(4.0, 20.0)
    return out


def _notch_fn(schedule):
    def f(t):
        n = 0
        for t0, v in schedule:
            if t0 > t:
                break
            n = v
        return n
    return f


def _samples(seed, duration=1800.0, k=3.6):
    sched = _schedule(seed, duration)
    events, _ = run(Scenario(duration=duration, k=k, truth=TRUTH, seed=seed, notch_fn=_notch_fn(sched)))
    notch = [(t, v) for t, kind, v in events if kind == 'cmd']
    front = [(t, v) for t, kind, v in events if kind == 'front']
    rear = [(t, v) for t, kind, v in events if kind == 'rear']
    return samples_from_run(notch, front, rear, k)


def test_lag_reaches_63_percent_after_tau():
    u = lagged([(0.0, 0), (1.0, 10)], [1.0, 1.75, 10.0], 0.75)
    assert u[0] == 0.0 and abs(u[1] - 6.32) < 0.01 and abs(u[2] - 10.0) < 1e-3


def test_fit_recovers_model_with_lag():
    runs = [_samples(1), _samples(2)]
    p, rep = fit(runs)
    assert abs(p.tau_cmd - TRUTH.tau_cmd) <= 0.25
    assert abs(p.a_max - TRUTH.a_max) / TRUTH.a_max < 0.05
    assert abs(p.b_max - TRUTH.b_max) / TRUTH.b_max < 0.05
    assert abs(p.v_base - TRUTH.v_base) <= 0.75
    assert rep['rms_accel'] < 0.08 and rep['traction'] > 100 and rep['brake'] > 100


def test_fitted_model_predicts_speed_through_20s_dropout():
    runs = [_samples(1), _samples(2)]
    p, _ = fit(runs)
    held_out = _samples(3)
    fitted = [abs(e) for e in open_loop_errors(held_out, p)]
    default = [abs(e) for e in open_loop_errors(held_out, ModelParams())]
    med = sorted(fitted)[len(fitted) // 2]
    assert len(fitted) > 20 and med < 0.3
    assert med < sorted(default)[len(default) // 2]
