"""Идентификация параметров модели (model.py) по прогонам: офлайн, чистый Python.

Отсчёты. На сетке 10 Гц берём моменты, где обе тележки свежие и согласованы (нет
проскальзывания): скорость v = среднее двух тележек / k, ускорение — центральная разность
за ±0,5 с по сглаженной скорости. Цель — ускорение без уклона: y = a + g·θ.

Модель при известных τ, v_base и v_blend линейна по остальным параметрам:
    выбег  (u ≈ 0): -y = (A + B·v + C·v²)·w(v)
    тяга   (u > 0): y + r(v) = a_max · (u/N)·min(1, v_base/v)
    тормоз (u < 0): -(y + r(v)) = c1·|u|/N + c2·|u|/N·min(1, v/v_blend),
                    b_max = c1 + c2, blend_floor = c1 / b_max
Поэтому τ перебираем (он общий), для каждого τ сопротивление — МНК по выбегу, затем
перебор v_base и v_blend, остальное — МНК. Критерий — сумма квадратов невязок ускорения.
"""
import bisect
import math
import random
from dataclasses import replace

from tram_odometry.model import G, ModelParams, accel, lag

TAUS = [0.25 * i for i in range(13)]                  # 0 … 3 с
V_BASES = [2.0 + 0.25 * i for i in range(49)]         # 2 … 14 м/с
V_BLENDS = [0.25 + 0.25 * i for i in range(20)]       # 0,25 … 5 м/с


def _nearest_walk(samples, times, tol):
    """Для каждого момента из times — ближайшее значение из samples [(t, v)] в пределах tol."""
    out, j = [], 0
    for t in times:
        while j + 1 < len(samples) and samples[j + 1][0] <= t:
            j += 1
        best = None
        for c in (j, j + 1):
            if 0 <= c < len(samples) and abs(samples[c][0] - t) <= tol:
                if best is None or abs(samples[c][0] - t) < abs(best[0] - t):
                    best = samples[c]
        out.append(None if best is None else best[1])
    return out


def samples_from_run(notch, front, rear, k, grade_fn=None, dt=0.1, win=5, every=5,
                     cons_abs=0.2, cons_rel=0.02, a_lim=2.5, v_min=0.5):
    """Отсчёты одного прогона. notch — [(t, позиция)], front/rear — [(t, км/ч)].
    grade_fn(t) -> уклон или None (тогда отсчёт не берём); без grade_fn уклон 0.
    -> {'notch': notch, 't': [...], 'v': [...], 'y': [...], 'g': [уклон]}."""
    front, rear = sorted(front), sorted(rear)
    out = {'notch': sorted(notch), 't': [], 'v': [], 'y': [], 'g': []}
    if not front or not rear or not notch:
        return out
    t0, t1 = max(front[0][0], rear[0][0]), min(front[-1][0], rear[-1][0])
    times = [t0 + i * dt for i in range(int((t1 - t0) / dt) + 1)]
    fv, rv = _nearest_walk(front, times, 0.6 * dt), _nearest_walk(rear, times, 0.6 * dt)
    v = []
    for f, r in zip(fv, rv):
        ok = f is not None and r is not None and abs(f - r) <= max(cons_abs * k, cons_rel * 0.5 * (f + r))
        v.append(0.5 * (f + r) / k if ok else None)
    for i in range(win + 1, len(v) - win - 1, every):
        window = v[i - win - 1:i + win + 2]
        if any(x is None for x in window) or v[i] < v_min:
            continue
        lo = sum(window[:3]) / 3.0
        hi = sum(window[-3:]) / 3.0
        a = (hi - lo) / (2 * win * dt)
        if abs(a) > a_lim:
            continue
        grade = 0.0 if grade_fn is None else grade_fn(times[i])
        if grade is None:
            continue
        out['t'].append(times[i])
        out['v'].append(v[i])
        out['y'].append(a + G * grade)
        out['g'].append(grade)
    return out


def lagged(notch, times, tau):
    """Позиция контроллера после запаздывания τ в моменты times (по возрастанию)."""
    p = ModelParams(tau_cmd=tau)
    out, j, u, n, t = [], 0, 0.0, 0, None
    for tt in times:
        while j < len(notch) and notch[j][0] <= tt:
            if t is not None:
                u = lag(u, n, notch[j][0] - t, p)
            t, n = notch[j][0], notch[j][1]
            j += 1
        if t is None:
            out.append(0.0)
            continue
        u = lag(u, n, tt - t, p)
        t = tt
        out.append(u)
    return out


def _solve(a, b):
    """Гаусс с выбором главного элемента для малой системы; вырожденная — None."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            return None
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def _lsq(features, target):
    """МНК: -> (коэффициенты, сумма квадратов невязок)."""
    n = len(features)
    ata = [[sum(fi * fj for fi, fj in zip(features[i], features[j])) for j in range(n)] for i in range(n)]
    atb = [sum(fi * y for fi, y in zip(features[i], target)) for i in range(n)]
    c = _solve(ata, atb)
    if c is None:
        return None, float('inf')
    sse = sum((y - sum(c[i] * features[i][k] for i in range(n))) ** 2 for k, y in enumerate(target))
    return c, sse


def _stack(runs, tau, max_samples, seed):
    u, v, y = [], [], []
    for r in runs:
        u += lagged(r['notch'], r['t'], tau)
        v += r['v']
        y += r['y']
    idx = list(range(len(v)))
    if len(idx) > max_samples:
        idx = sorted(random.Random(seed).sample(idx, max_samples))
    return [u[i] for i in idx], [v[i] for i in idx], [y[i] for i in idx]


def fit(runs, base=ModelParams(), taus=TAUS, v_bases=V_BASES, v_blends=V_BLENDS, coast_eps=0.3,
        max_samples=40000, seed=1):
    """runs — результаты samples_from_run. -> (ModelParams, отчёт)."""
    N, vs = base.notch_max, base.v_stop
    best = None
    for tau in taus:
        u, v, y = _stack(runs, tau, max_samples, seed)
        w = [min(1.0, x / vs) for x in v]
        co = [i for i, x in enumerate(u) if abs(x) < coast_eps]
        tr = [i for i, x in enumerate(u) if x >= coast_eps]
        br = [i for i, x in enumerate(u) if x <= -coast_eps]
        if len(co) < 10 or len(tr) < 10 or len(br) < 10:
            raise ValueError(f'мало отсчётов: выбег {len(co)}, тяга {len(tr)}, тормоз {len(br)}')
        res, sse_co = _lsq([[w[i] for i in co], [v[i] * w[i] for i in co], [v[i] ** 2 * w[i] for i in co]],
                           [-y[i] for i in co])
        if res is None:
            continue
        r = [(res[0] + res[1] * x + res[2] * x * x) * wi for x, wi in zip(v, w)]
        z = [y[i] + r[i] for i in tr]
        zz = sum(q * q for q in z)
        tr_best = (float('inf'), None, None)
        for vb in v_bases:
            f = [u[i] / N * min(1.0, vb / max(v[i], 0.1)) for i in tr]
            ff, fz = sum(q * q for q in f), sum(a * b for a, b in zip(f, z))
            sse = zz - fz * fz / ff
            if sse < tr_best[0]:
                tr_best = (sse, vb, fz / ff)
        zb = [-(y[i] + r[i]) for i in br]
        br_best = (float('inf'), None, None, None)
        for vl in v_blends:
            f1 = [-u[i] / N for i in br]
            f2 = [-u[i] / N * min(1.0, v[i] / vl) for i in br]
            c, sse = _lsq([f1, f2], zb)
            if c is None:
                continue
            if c[0] < 0.0 or c[1] < 0.0:            # доля тормоза вне [0, 1]: одна составляющая
                keep = [f2] if c[0] < 0.0 else [f1]
                c1, sse = _lsq(keep, zb)
                c = [0.0, c1[0]] if c[0] < 0.0 else [c1[0], 0.0]
            if sse < br_best[0]:
                br_best = (sse, vl, c[0], c[1])
        total = sse_co + tr_best[0] + br_best[0]
        if best is None or total < best[0]:
            best = (total, tau, res, tr_best, br_best, (len(co), len(tr), len(br)), (sse_co, tr_best[0], br_best[0]))
    total, tau, res, tr_best, br_best, counts, sses = best
    b_max = br_best[2] + br_best[3]
    params = replace(base, tau_cmd=tau, a_max=tr_best[2], v_base=tr_best[1], b_max=b_max,
                     v_blend=br_best[1], blend_floor=br_best[2] / b_max if b_max > 0 else base.blend_floor,
                     res_a=res[0], res_b=res[1], res_c=res[2])
    n = sum(counts)
    report = {'samples': n, 'coast': counts[0], 'traction': counts[1], 'brake': counts[2],
              'rms_accel': math.sqrt(total / n), 'rms_coast': math.sqrt(sses[0] / counts[0]),
              'rms_traction': math.sqrt(sses[1] / counts[1]), 'rms_brake': math.sqrt(sses[2] / counts[2])}
    return params, report


def open_loop_errors(run, params, horizon_s=20.0, stride_s=30.0, dt=0.05):
    """Ошибка скорости одной модели за horizon_s без колёс (как при пропуске обеих тележек):
    старт с измеренной скорости, дальше только контроллер и уклон отсчётов.
    -> [ошибка в конце, м/с]."""
    t, v, notch = run['t'], run['v'], run['notch']
    grades = run.get('g') or [0.0] * len(t)

    def grade_fn(tt):
        return grades[min(bisect.bisect_left(t, tt), len(t) - 1)]
    errs, i = [], 0
    while i < len(t):
        t0 = t[i]
        j = i
        while j + 1 < len(t) and t[j + 1] - t0 <= horizon_s:
            j += 1
        if t[j] - t0 < 0.9 * horizon_s or (t[j] - t0) / max(j - i, 1) > 1.5:   # дыры в отсчётах
            i += 1
            continue
        steps = int((t[j] - t0) / dt)
        times = [t0 + dt * k for k in range(steps + 1)]
        u = lagged(notch, times, params.tau_cmd)
        x = v[i]
        for k in range(steps):
            x = max(0.0, x + accel(u[k], x, params, grade_fn(times[k])) * dt)
        errs.append(x - v[j])
        while i < len(t) and t[i] < t0 + stride_s:
            i += 1
    return errs
