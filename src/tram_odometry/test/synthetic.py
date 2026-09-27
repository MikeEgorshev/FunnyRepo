"""Синтетический прогон для тестов: истинная динамика, входы с шумом и внедрёнными сбоями.

Это не данные трамвая, а контролируемый сценарий: цикл «разгон — ход — выбег —
торможение — стоянка» по 80 с. Сбои включаются интервалами (t0, t1).
"""
import random
from dataclasses import dataclass, field

from tram_odometry.model import ModelParams, accel, lag


def notch_at(t):
    c = t % 80.0
    if c < 14:
        return 10
    if c < 36:
        return 3
    if c < 52:
        return 0
    if c < 68:
        return -8
    return -4


@dataclass
class Scenario:
    duration: float = 240.0
    k: float = 3.60                              # истинный масштаб колёс
    truth: ModelParams = ModelParams(a_max=1.0, b_max=1.2)   # модель фильтра чуть ошибается
    noise_kmh: float = 0.15
    slip: list = field(default_factory=list)     # буксование передней тележки в тяге
    slide: list = field(default_factory=list)    # юз обеих тележек в торможении
    dropout: list = field(default_factory=list)  # обе тележки молчат
    slip_ratio: float = 0.2
    seed: int = 1
    notch_fn: object = None                      # позиция контроллера от времени; None — notch_at


def _inside(t, spans):
    return any(a <= t < b for a, b in spans)


def run(sc):
    """События в порядке времени и истина: (events, truth).

    events — список (t, kind, value): kind — 'cmd', 'front' или 'rear'; value — позиция
    контроллера или скорость тележки в км/ч. truth — список (t, s, v) с шагом 0,01 с.
    """
    rnd = random.Random(sc.seed)
    dt = 0.01
    v = s = u = 0.0
    events, truth = [], []
    steps = int(sc.duration / dt)
    notch_fn = sc.notch_fn or notch_at
    for i in range(steps + 1):
        t = i * dt
        n = notch_fn(t)
        u = lag(u, n, dt, sc.truth)
        v = max(0.0, v + accel(u, v, sc.truth) * dt)
        s += v * dt
        truth.append((t, s, v))
        if i % 5 == 0:
            events.append((t, 'cmd', n))
        if i % 10 == 0 and not _inside(t, sc.dropout):
            front, rear = v, v
            if _inside(t, sc.slip) and n > 0:
                front = v * (1.0 + sc.slip_ratio)
                rear = v * (1.0 + 0.3 * sc.slip_ratio)
            if _inside(t, sc.slide) and n < 0:
                front = v * (1.0 - 1.4 * sc.slip_ratio)
                rear = v * (1.0 - 1.1 * sc.slip_ratio)
            events.append((t, 'front', max(0.0, front * sc.k + rnd.gauss(0.0, sc.noise_kmh))))
            events.append((t + 0.005, 'rear', max(0.0, rear * sc.k + rnd.gauss(0.0, sc.noise_kmh))))
    events.sort(key=lambda e: e[0])
    return events, truth


def replay(estimator, events, truth, every=0.1):
    """Прогоняет события через оценщик. Возвращает [(t, s_true, v_true, s_est, v_est, slip)]."""
    out = []
    ti = 0
    next_out = 0.0
    for t, kind, value in events:
        if kind == 'cmd':
            estimator.set_notch(t, value)
        else:
            estimator.wheel(kind, t, value)
        if t >= next_out:
            while ti + 1 < len(truth) and truth[ti + 1][0] <= t:
                ti += 1
            st = estimator.state()
            out.append((t, truth[ti][1], truth[ti][2], st.s, st.v, st.slip))
            next_out += every
    return out


def naive_position(events, truth, k):
    """Наивная одометрия: среднее тележек / k, в пропуске держим последнюю скорость."""
    last = {'front': 0.0, 'rear': 0.0}
    s, v, t_prev = 0.0, 0.0, 0.0
    for t, kind, value in events:
        s += v * (t - t_prev)
        t_prev = t
        if kind in last:
            last[kind] = value / k
            v = 0.5 * (last['front'] + last['rear'])
    return s - truth[-1][1]
