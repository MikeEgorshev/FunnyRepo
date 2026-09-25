"""Модель продольной динамики трамвая на единицу массы.

Масса неизвестна, поэтому модель считает ускорение, м/с², а не силу:

    a = f_tr(n, v) - f_br(n, v) - r(v) - g·θ

    f_tr = (n/N)·min(a_max, a_max·v_base/v),  n > 0   постоянная сила, затем постоянная мощность
    f_br = (|n|/N)·b_max·blend(v),            n < 0   электрический тормоз слабеет на малой скорости
    r(v) = A + B·v + C·v²                              сопротивление движению (формула Дэвиса)
    θ    — уклон (подъём > 0), доля: 0.01 = 1 %

Значения по умолчанию — стартовые оценки. Настоящие параметры идентифицируются по данным
и задаются в YAML-конфиге ноды.
"""
from dataclasses import dataclass

G = 9.81


@dataclass(frozen=True)
class ModelParams:
    notch_max: int = 15
    a_max: float = 1.1         # м/с², ускорение тяги на позиции notch_max ниже базовой скорости
    v_base: float = 7.0        # м/с, выше — режим постоянной мощности
    b_max: float = 1.3         # м/с², замедление на позиции -notch_max
    v_blend: float = 1.8       # м/с, ниже электрический тормоз слабеет
    blend_floor: float = 0.8   # доля тормоза, которая остаётся при v -> 0
    res_a: float = 0.015       # м/с²
    res_b: float = 0.0         # 1/с
    res_c: float = 0.0005      # 1/м
    v_stop: float = 0.3        # м/с, ниже сопротивление плавно уходит в ноль


def traction(notch, v, p):
    if notch <= 0:
        return 0.0
    v = max(v, 0.1)
    return notch / p.notch_max * min(p.a_max, p.a_max * p.v_base / v)


def brake(notch, v, p):
    if notch >= 0:
        return 0.0
    blend = p.blend_floor + (1.0 - p.blend_floor) * min(1.0, max(v, 0.0) / p.v_blend)
    return -notch / p.notch_max * p.b_max * blend


def resistance(v, p):
    # у остановки сопротивление плавно уходит в ноль: стоящий трамвай не едет назад,
    # а якобиан фильтра остаётся конечным
    v = max(v, 0.0)
    return (p.res_a + p.res_b * v + p.res_c * v * v) * min(1.0, v / p.v_stop)


def accel(notch, v, p, grade=0.0):
    """Модельное ускорение, м/с². notch — позиция контроллера, v — скорость, м/с."""
    return traction(notch, v, p) - brake(notch, v, p) - resistance(v, p) - G * grade


def accel_dv(notch, v, p, grade=0.0, h=1e-3):
    """Производная ускорения по скорости (для якобиана фильтра), численно."""
    lo = max(v - h, 0.0)
    return (accel(notch, v + h, p, grade) - accel(notch, lo, p, grade)) / (v + h - lo)
