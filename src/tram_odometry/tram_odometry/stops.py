"""Привязка дистанции к местам регулярных стоянок (остановки, стоп-линии).

Трамвай стоит дольше dwell_s — ищем стоянку из списка в пределах гейта
gate_sigma·√(σ_s² + σ_стоянки²) от текущей оценки s. Одна подходящая стоянка —
поправка s через Estimator.position_fix; через неё же фильтр узнаёт масштаб колёс k.
Две стоянки в гейте (неоднозначно) или ни одной (светофор) — поправки нет.
Одна поправка на одну стоянку.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StopParams:
    dwell_s: float = 5.0          # с, стоянка короче — не используем
    stop_speed: float = 0.1       # м/с, оценка ниже — трамвай стоит
    moving_speed: float = 0.5     # м/с, выше — стоянка закончилась
    gate_sigma: float = 3.0
    gate_min_m: float = 5.0       # м, гейт не уже: разброс места остановки
    gate_max_m: float = 80.0      # м, гейт не шире: иначе путаем соседние стоянки
    min_sigma_m: float = 3.0      # м, нижняя граница разброса места стоянки (трамвай встаёт у платформы ±3 м)


class StopFixer:
    def __init__(self, stops, length=None, params=StopParams()):
        """stops — [(s, sigma)], м; length — длина замкнутого маршрута (None — не замкнут)."""
        self.stops = sorted(stops)
        self.length = length
        self.p = params
        self.fixes = 0
        self._dwell_start = None
        self._fixed = False

    def reset(self):
        self._dwell_start = None
        self._fixed = False

    def _offset(self, s_stop, s):
        """Смещение от оценки s до стоянки с учётом замкнутости маршрута."""
        d = s_stop - s
        if self.length:
            d = (d + 0.5 * self.length) % self.length - 0.5 * self.length
        return d

    def candidates(self, s, var_s):
        """Стоянки в гейте: [(|d|, d, sigma)] по возрастанию расстояния."""
        p = self.p
        out = []
        for s_stop, sigma in self.stops:
            sigma = max(sigma, p.min_sigma_m)
            gate = min(max(p.gate_sigma * math.sqrt(var_s + sigma * sigma), p.gate_min_m), p.gate_max_m)
            d = self._offset(s_stop, s)
            if abs(d) <= gate:
                out.append((abs(d), d, sigma))
        return sorted(out)

    def update(self, t, estimator):
        """Вызывать после каждого отсчёта колёс. Возвращает поправку s, м, или None."""
        st = estimator.state()
        p = self.p
        if st.v > p.moving_speed or not st.wheels_ok:
            self.reset()
            return None
        if st.v > p.stop_speed:
            return None
        if self._dwell_start is None:
            self._dwell_start = t
        if self._fixed or t - self._dwell_start < p.dwell_s:
            return None
        self._fixed = True
        found = self.candidates(st.s, st.var_s)
        if len(found) != 1:
            return None
        _, d, sigma = found[0]
        estimator.position_fix(t, st.s + d, sigma)
        self.fixes += 1
        return d
