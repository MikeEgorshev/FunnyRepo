"""Фильтр Калмана (EKF) вдоль пути: s — дистанция, м; v — скорость, м/с;
d — возмущающее ускорение, м/с² (масса, уклон, ошибка модели); k — масштаб колёс,
(км/ч по тележке) / (м/с).

Прогноз — по модели тяги и торможения (model.py). Измерение — скорость тележки в км/ч,
h(x) = k·v. По колёсам k и v неразделимы, поэтому колёса k не двигают
(k — «учитываемый» параметр фильтра Шмидта); k уточняют только абсолютные привязки дистанции.
Подозрительные отсчёты отсекаются проверкой χ²; если отсекаем дольше
reject_max_s, отсчёты снова принимаются, но с большим шумом — фильтр не «залипает» на модели.
Стоянка (обе тележки около нуля, нет тяги) — измерение v = 0.

Чистый Python без зависимостей: один и тот же код работает в ноде ROS и в офлайн-оценке.
"""
import math
from dataclasses import dataclass, field

from tram_odometry.model import G, ModelParams, accel, accel_dv
from tram_odometry.slip import SlipDetector, SlipFlags, SlipParams

N = 4
S, V, D, K = range(N)


@dataclass(frozen=True)
class FilterParams:
    k0: float = 3.5966          # начальный масштаб колёс
    sigma_k0: float = 0.01      # его неопределённость
    sigma_v0: float = 0.5       # м/с
    q_accel: float = 0.15       # м/с², шум ускорения модели
    q_d: float = 0.02           # м/с² за √с, блуждание возмущения
    q_k: float = 1e-5           # блуждание масштаба за √с
    r_wheel: float = 0.15       # км/ч, шум скорости тележки
    gate: float = 9.0           # порог χ² (1 степень свободы, 3σ)
    flag_after: int = 3         # столько отсечений подряд — флаг проскальзывания (одиночные бывают и на чистых данных)
    reject_max_s: float = 3.0   # с, дольше подряд отсекать нельзя
    r_recover: float = 3.0      # км/ч, шум при вынужденном приёме
    zupt_speed: float = 0.1     # м/с, ниже — стоянка
    zupt_max_est: float = 1.0   # м/с, при большей оценке нули колёс — юз, а не стоянка
    r_zupt: float = 0.02        # м/с
    step_s: float = 0.05        # с, шаг интегрирования прогноза
    max_gap_s: float = 120.0    # с, больший скачок времени вперёд — сброс (новый прогон)
    reset_back_s: float = 1.0   # с, скачок времени назад больше — сброс (новый прогон)
    max_lag_s: float = 0.25     # с, отсчёт старее текущего времени — отбрасывается


@dataclass
class State:
    t: float
    s: float
    v: float
    d: float
    k: float
    var_s: float
    var_v: float
    accel: float                 # оценка ускорения, м/с²
    slip: bool
    flags: SlipFlags = field(default_factory=SlipFlags)
    wheels_ok: bool = True       # есть хотя бы одна свежая тележка
    slip_ratio: float = 0.0      # (скорость колёс - оценка) / оценка

    @property
    def adhesion_used(self):
        """Доля веса, которую использует тяга или тормоз: |a| / g."""
        return abs(self.accel) / G


def _matmul(a, b):
    return [[sum(a[i][m] * b[m][j] for m in range(N)) for j in range(N)] for i in range(N)]


def _transpose(a):
    return [list(row) for row in zip(*a)]


class Estimator:
    def __init__(self, model=ModelParams(), params=FilterParams(), slip=SlipParams(), grade_at=None):
        self.model = model
        self.p = params
        self.detector = SlipDetector(slip)
        self.grade_at = grade_at or (lambda s: 0.0)
        self.resets = 0
        self._init_state(None)

    # --- состояние ---------------------------------------------------------------

    def _init_state(self, t):
        p = self.p
        self.t = t
        self.x = [0.0, 0.0, 0.0, p.k0]
        self.P = [[0.0] * N for _ in range(N)]
        self.P[V][V] = p.sigma_v0 ** 2
        self.P[D][D] = 0.05 ** 2
        self.P[K][K] = p.sigma_k0 ** 2
        self.notch = 0
        self.flags = SlipFlags()
        self.slip = False
        self.slip_ratio = 0.0
        self._reject_since = None
        self._rejects = 0
        self._need_v = True           # первую скорость берём с колёс: прогон может начаться на ходу
        self.detector.reset()

    def reset(self, t=None):
        self.resets += 1
        self._init_state(t)

    def set_initial(self, v=None, s=0.0, k=None):
        if v is not None:
            self.x[V] = max(v, 0.0)
        self.x[S] = s
        if k is not None:
            self.x[K] = k

    # --- время -------------------------------------------------------------------

    def _accept_time(self, t):
        """True — отсчёт можно использовать; при необходимости продвигает фильтр до t."""
        if self.t is None:
            self.t = t
            return True
        if t < self.t - self.p.reset_back_s or t > self.t + self.p.max_gap_s:
            self.reset(t)
            return True
        if t < self.t - self.p.max_lag_s:
            return False
        self.advance(t)
        return True

    def advance(self, t):
        """Прогноз по модели до момента t (назад не идёт)."""
        if self.t is None:
            self.t = t
            return
        while t - self.t > 1e-9:
            dt = min(self.p.step_s, t - self.t)
            self._predict(dt)
            self.t = t if t - self.t - dt < 1e-9 else self.t + dt

    def _predict(self, dt):
        x, p, m = self.x, self.p, self.model
        grade = self.grade_at(x[S])
        a = accel(self.notch, x[V], m, grade) + x[D]
        v_new = x[V] + a * dt
        if v_new < 0.0:                   # трамвай не едет назад: останавливается
            a = -x[V] / dt if dt > 0 else 0.0
            v_new = 0.0
        x[S] += x[V] * dt + 0.5 * a * dt * dt
        x[V] = v_new
        dadv = accel_dv(self.notch, x[V], m, grade)
        F = [[1.0, dt, 0.5 * dt * dt, 0.0],
             [0.0, 1.0 + dadv * dt, dt, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        P = _matmul(_matmul(F, self.P), _transpose(F))
        qa = p.q_accel ** 2
        P[S][S] += qa * dt ** 3 / 3.0
        P[S][V] += qa * dt ** 2 / 2.0
        P[V][S] += qa * dt ** 2 / 2.0
        P[V][V] += qa * dt
        P[D][D] += p.q_d ** 2 * dt
        P[K][K] += p.q_k ** 2 * dt
        self.P = P

    # --- измерения ---------------------------------------------------------------

    def _update(self, H, z, h, r2, frozen=()):
        """Скалярное обновление; состояния из frozen не меняются (фильтр Шмидта).

        Ковариация — в форме Джозефа: она верна и при усечённом усилении.
        """
        PH = [sum(self.P[i][j] * H[j] for j in range(N)) for i in range(N)]
        s = sum(H[i] * PH[i] for i in range(N)) + r2
        inn = z - h
        gain = [0.0 if i in frozen else PH[i] / s for i in range(N)]
        for i in range(N):
            self.x[i] += gain[i] * inn
        A = [[(1.0 if i == j else 0.0) - gain[i] * H[j] for j in range(N)] for i in range(N)]
        P = _matmul(_matmul(A, self.P), _transpose(A))
        for i in range(N):
            for j in range(N):
                P[i][j] += gain[i] * gain[j] * r2
        for i in range(N):
            for j in range(i + 1, N):
                P[i][j] = P[j][i] = 0.5 * (P[i][j] + P[j][i])
        self.P = P
        self.x[V] = max(self.x[V], 0.0)

    def _nis(self, H, z, h, r2):
        s = sum(H[i] * sum(self.P[i][j] * H[j] for j in range(N)) for i in range(N)) + r2
        return (z - h) ** 2 / s

    def set_notch(self, t, notch):
        if self._accept_time(t):
            self.notch = int(notch)

    def wheel(self, which, t, kmh):
        """Отсчёт тележки: which — 'front' или 'rear', kmh — скорость в км/ч."""
        if kmh is None or not math.isfinite(kmh) or not self._accept_time(t):
            return
        p, x = self.p, self.x
        self.detector.wheel(which, t, kmh / x[K])
        z_mps, self.flags = self.detector.check(t, self.notch, x[V])
        front, rear = self.detector.fresh_speeds(t)
        raw = [v for v in (front, rear) if v is not None]

        if self._need_v and z_mps is not None:
            x[V] = z_mps
            self.P[V][V] = (2.0 * p.r_wheel / x[K]) ** 2
            self._need_v = False
            return
        self.slip_ratio = (sum(raw) / len(raw) - x[V]) / max(x[V], 0.5) if raw else 0.0
        if raw and max(raw) < p.zupt_speed and self.notch <= 0 and x[V] < p.zupt_max_est:
            self._update([0.0, 1.0, 0.0, 0.0], 0.0, x[V], p.r_zupt ** 2, frozen=(K,))
            self._reject_since = None
            self._rejects = 0
            self.slip = self.flags.any
            return
        if z_mps is None:
            self.slip = self.flags.any
            return

        z = z_mps * x[K]
        H = [0.0, x[K], 0.0, x[V]]
        h = x[K] * x[V]
        rejected = self._nis(H, z, h, p.r_wheel ** 2) > p.gate
        if not rejected:
            self._update(H, z, h, p.r_wheel ** 2, frozen=(K,))
            self._reject_since = None
            self._rejects = 0
        else:
            self._rejects += 1
            if self._reject_since is None:
                self._reject_since = t
            if t - self._reject_since > p.reject_max_s:
                self._update(H, z, h, p.r_recover ** 2, frozen=(K,))
        self.slip = self._rejects >= p.flag_after or self.flags.any

    def position_fix(self, t, s, sigma):
        """Абсолютная привязка дистанции (например, стоянка у известной остановки)."""
        if self._accept_time(t):
            self._update([1.0, 0.0, 0.0, 0.0], s, self.x[S], sigma ** 2)

    # --- выход -------------------------------------------------------------------

    def state_at(self, t):
        """Состояние, спрогнозированное на момент t, без изменения самого фильтра.

        Для публикации по таймеру: прогноз выхода не должен сдвигать часы фильтра,
        иначе следующие входы окажутся «опоздавшими».
        """
        if self.t is None or t <= self.t:
            return self.state()
        saved = (list(self.x), [row[:] for row in self.P], self.t)
        self.advance(t)
        st = self.state()
        self.x, self.P, self.t = saved
        return st

    def state(self):
        x = self.x
        front, rear = self.detector.fresh_speeds(self.t) if self.t is not None else (None, None)
        return State(
            t=self.t if self.t is not None else 0.0,
            s=x[S], v=x[V], d=x[D], k=x[K],
            var_s=max(self.P[S][S], 0.0), var_v=max(self.P[V][V], 0.0),
            accel=accel(self.notch, x[V], self.model, self.grade_at(x[S])) + x[D],
            slip=self.slip, flags=self.flags,
            wheels_ok=front is not None or rear is not None,
            slip_ratio=self.slip_ratio if front is not None or rear is not None else 0.0,
        )
