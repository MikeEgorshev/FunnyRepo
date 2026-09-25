"""Проверка скорости тележек перед тем, как отдать её фильтру.

Правила из практики рельсовой одометрии:
- в тяге ближе к истинной скорости медленная тележка (быстрая буксует), в торможении —
  быстрая (медленная идёт юзом), на выбеге — среднее;
- тележки расходятся больше порога несколько отсчётов подряд — признак проскальзывания;
- ускорение колёс выше физического предела — тележка буксует или идёт юзом, её отсчёт
  не используем.

Все скорости здесь — в м/с (км/ч уже поделены на масштаб колёс k).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SlipParams:
    consistency_abs: float = 0.3    # м/с
    consistency_rel: float = 0.03   # доля скорости
    consistency_count: int = 2      # столько отсчётов подряд — расхождение признаётся
    accel_traction: float = 2.0     # м/с², предел ускорения колёс
    accel_brake: float = 3.0        # м/с², предел замедления колёс
    accel_min_dt: float = 0.3       # с, база производной: короче — шум и пачки дают ложные пики
    stale_s: float = 0.3            # с, старее — тележка считается пропавшей


@dataclass
class SlipFlags:
    inconsistent: bool = False
    front_accel: bool = False
    rear_accel: bool = False

    @property
    def any(self):
        return self.inconsistent or self.front_accel or self.rear_accel


class Bogie:
    """Последний отсчёт одной тележки и ускорение колёс по нему."""

    def __init__(self):
        self.t = None
        self.v = None
        self.accel = 0.0
        self._ref = None   # (t, v): опорная точка для производной

    def update(self, t, v, min_dt):
        if self._ref is None or t < self._ref[0]:
            self._ref = (t, v)
            self.accel = 0.0
        elif t - self._ref[0] >= min_dt:
            dt = t - self._ref[0]
            self.accel = (v - self._ref[1]) / dt if dt < 1.0 else 0.0
            self._ref = (t, v)
        self.t, self.v = t, v

    def fresh(self, t, stale_s):
        return self.t is not None and t - self.t <= stale_s


def pick_speed(front, rear, notch):
    """Скорость по фазе движения; None — тележка недоступна."""
    if front is None:
        return rear
    if rear is None:
        return front
    if notch > 0:
        return min(front, rear)
    if notch < 0:
        return max(front, rear)
    return 0.5 * (front + rear)


class SlipDetector:
    def __init__(self, params=SlipParams()):
        self.p = params
        self.front = Bogie()
        self.rear = Bogie()
        self._incons = 0

    def reset(self):
        self.__init__(self.p)

    def wheel(self, which, t, v):
        bogie = self.front if which == 'front' else self.rear
        bogie.update(t, v, self.p.accel_min_dt)

    def fresh_speeds(self, t):
        """Скорости свежих тележек: (front, rear), пропавшая — None."""
        f = self.front.v if self.front.fresh(t, self.p.stale_s) else None
        r = self.rear.v if self.rear.fresh(t, self.p.stale_s) else None
        return f, r

    def check(self, t, notch, v_est):
        """Скорость для фильтра (или None) и флаги проскальзывания."""
        p = self.p
        front, rear = self.fresh_speeds(t)
        flags = SlipFlags()
        for name, bogie, value in (('front', self.front, front), ('rear', self.rear, rear)):
            if value is None:
                continue
            if bogie.accel > p.accel_traction or bogie.accel < -p.accel_brake:
                setattr(flags, name + '_accel', True)
        if front is not None and rear is not None:
            limit = max(p.consistency_abs, p.consistency_rel * max(v_est, 0.0))
            self._incons = self._incons + 1 if abs(front - rear) > limit else 0
            flags.inconsistent = self._incons >= p.consistency_count
        else:
            self._incons = 0
        use_front = None if flags.front_accel else front
        use_rear = None if flags.rear_accel else rear
        return pick_speed(use_front, use_rear, notch), flags
