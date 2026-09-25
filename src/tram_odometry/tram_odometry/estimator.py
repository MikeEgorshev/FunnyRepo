"""Онлайн-оценщик скорости и положения трамвая: EKF вдоль маршрута на модели тяги.

Состояние x = [s, v, d]: s — дистанция вдоль маршрута (м), v — скорость (м/с), d — возмущающее
ускорение (уклон, масса, ошибка модели; случайное блуждание, м/с²).

Предсказание: a = модель(u(t − задержка), v) + d, где u — позиция контроллера.
Измерения: скорость каждой тележки (км/ч / k). Защита от недостоверной одометрии:
  * гейтинг по невязке (χ²): выброс или проскальзывание не проходит в фильтр;
  * согласование тележек: при тяге буксующее колесо показывает больше, при торможении в юзе — меньше;
  * пропуски колёс — работает только модель; долгое расхождение с моделью — пересинхронизация.
Положение: s -> карта линии (с отводами у конечных); на стоянке — привязка к месту регулярной
стоянки. GNSS — только выставка в первые секунды прогона. Чистый Python: нода без зависимостей.
"""
import math

from .geo import Enu, UtmLocal


class Params:
    wheel_kmh_per_mps = 3.5966   # делитель «км/ч по тележке -> м/с»
    init_window_s = 5.0          # сколько секунд принимаем GNSS для выставки
    sigma_a = 0.25               # СКО ошибки модели ускорения, м/с² (белый шум)
    sigma_d = 0.02               # блуждание возмущения d, м/с² / √с
    d_max = 0.5                  # ограничение |d|, м/с²
    wheel_min_kmh = -2.0         # правдоподобный диапазон скорости тележки, км/ч
    wheel_max_kmh = 120.0
    stuck_n = 6                  # столько одинаковых ненулевых показаний подряд — датчик завис
    stuck_min_kmh = 1.0          # на стоянке нули законно повторяются
    sigma_wheel = 0.05           # шум скорости тележки, м/с
    sigma_wheel_rel = 0.01       # и относительный, доля скорости
    gate_chi2 = 9.0              # порог невязки (3σ)
    resync_s = 3.0               # колёса расходятся с моделью дольше — доверяем колёсам
    bogie_tol_abs = 0.3          # допустимое расхождение тележек, м/с
    bogie_tol_rel = 0.05         # и доля скорости
    bogie_pair_s = 0.25          # показания тележек сравниваем, если они не дальше по времени
    stale_s = 0.3                # вход старше времени фильтра на столько — отбрасываем
    max_gap_s = 30.0             # дольше — не интегрируем (разрыв записи)
    stop_v = 0.05                # скорость «стоим», м/с
    stop_confirm_s = 2.0         # стоим дольше — привязка к месту стоянки
    stop_gate_sigma = 3.0
    stop_gate_max_m = 60.0
    rel_scale_err = 0.004        # относительная ошибка масштаба колёс для неопределённости s
    scale_adapt = 0.0            # подстройка масштаба по привязкам (0 — выкл.: невязки привязок шумные)
    scale_adapt_min_m = 300.0
    scale_max = 0.02
    # система координат выхода. Судья сравнивает с /localization/kinematic_state (QA 25.09):
    # base_link в сетке MGRS = UTM 37N минус (300000, 6100000) — как pathgraph организаторов.
    #   output_frame: 'utm_local' — сетка MGRS/pathgraph; 'enu' — ENU с началом в первой точке GNSS
    #   output_origin: 'frame' — как в рамке; 'start' — вычесть положение выходной точки на старте
    #   output_lever_m / output_dz_m — от антенны master до base_link: +9.873 м вперёд по пути,
    #   −3.0 м по высоте (tf антенн от организаторов)
    init_still_m = 3.0           # разброс точек GNSS меньше — стоим, берём медиану
    init_jump_m = 10.0           # точка дальше медианы — прыжок GNSS, если трамвай стоит
    output_frame = 'utm_local'
    output_origin = 'frame'
    output_lever_m = 9.873
    output_dz_m = -3.0


class TramEstimator:
    def __init__(self, route_map, model, stops=(), params=None):
        self.map = route_map
        self.model = model
        self.stops = list(stops)          # [(s, sigma), ...]
        self.p = params or Params()
        self.enu = None
        self.t0_gnss = None
        self.first_xy = None
        self.t = None                     # время фильтра (header.stamp последнего шага)
        self.s = self.v = self.d = 0.0
        self.P = [[25.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.01]]
        self.ready = False
        self.last_wheel = {True: None, False: None}
        self.repeat = {True: (None, 0), False: (None, 0)}   # (последнее значение, повторов подряд)
        self.stuck = {True: False, False: False}
        self.v_init = False
        self.reject_since = None
        self.stop_since = None
        self.snapped = False
        self.d_since_fix = 0.0
        self.var_fix = 4.0
        self.scale = 0.0                  # поправка масштаба колёс: v = z · (1 + scale)
        self.slip = False
        self.last_out = None
        self.origin = None                # положение выходной точки на старте (output_origin='start')
        self.init_fixes = []              # точки GNSS окна выставки в системе выхода
        self.first_input = None           # метка первого входа (для режима без GNSS)
        self.mode = None                  # None — по карте; 'relative' — GNSS на старте не было

    # --- входы -----------------------------------------------------------------------------
    def on_gnss(self, stamp, lat, lon, alt):
        if not all(math.isfinite(x) for x in (stamp, lat, lon, alt)):
            return None
        if self.enu is None:
            self.enu = UtmLocal() if self.p.output_frame == 'utm_local' else Enu(lat, lon, alt)
            self.map.to_frame(self.enu)
            self.t0_gnss = stamp
        if stamp - self.t0_gnss > self.p.init_window_s or self.mode == 'relative':
            return None
        x, y, _ = self.enu.forward(lat, lon, alt)
        if self.first_xy is None:
            self.first_xy = (x, y)
        # на старте трамвай обычно стоит: медиана точек устойчива к прыжкам GNSS;
        # если едем (точки расходятся), берём свежую точку
        self.init_fixes.append((x, y))
        mx = sorted(p[0] for p in self.init_fixes)[len(self.init_fixes) // 2]
        my = sorted(p[1] for p in self.init_fixes)[len(self.init_fixes) // 2]
        spread = max(math.hypot(p[0] - mx, p[1] - my) for p in self.init_fixes)
        if spread < self.p.init_still_m:
            x, y = mx, my
        elif math.hypot(x - mx, y - my) > self.p.init_jump_m and len(self.init_fixes) > 2:
            self.init_fixes.pop()
            return None  # одиночный прыжок GNSS — пропускаем
        dx, dy = x - self.first_xy[0], y - self.first_xy[1]
        yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 3.0 else None
        self._advance(stamp)
        self.s, dist = self.map.locate_start(x, y, yaw)
        self.P[0][0] = max(1.0, dist * dist)
        self.P[0][1] = self.P[1][0] = self.P[0][2] = self.P[2][0] = 0.0
        self.var_fix, self.d_since_fix = self.P[0][0], 0.0
        self.ready = True
        if self.t is None or stamp > self.t:
            self.t = stamp
        return self._output(stamp)

    def on_cmd(self, stamp, position):
        if not math.isfinite(stamp):
            return None
        self.model.push_command(stamp, position)
        if not self._advance(stamp):
            return None
        self._check_no_gnss(stamp)
        return self._output(stamp)

    def _check_no_gnss(self, stamp):
        """GNSS на старте так и не пришёл — относительная одометрия от старта (просили на QA)."""
        if self.first_input is None:
            self.first_input = stamp
        if self.enu is None and self.mode is None and stamp - self.first_input > self.p.init_window_s:
            self.mode, self.ready, self.s, self.d_since_fix = 'relative', True, 0.0, 0.0

    def on_wheel(self, stamp, front, kmh):
        if not (math.isfinite(stamp) and math.isfinite(kmh)) or not self.p.wheel_min_kmh <= kmh <= self.p.wheel_max_kmh:
            self.slip = True  # мусор на входе: как пропуск, в фильтр не пускаем
            return None
        if not self._advance(stamp):
            return None
        self._check_no_gnss(stamp)
        # зависание датчика: одно и то же ненулевое значение много раз подряд (в живых данных — 2–3)
        last, count = self.repeat[front]
        count = count + 1 if kmh == last and kmh > self.p.stuck_min_kmh else 1
        self.repeat[front] = (kmh, count)
        self.stuck[front] = count >= self.p.stuck_n
        self.slip = False
        if self.stuck[front]:
            self.last_wheel[front] = None
            self._stop_logic(stamp)
            return self._output(stamp)
        z = kmh / self.p.wheel_kmh_per_mps * (1.0 + self.scale)
        self.last_wheel[front] = (stamp, z)
        if self._wheel_trusted(front, stamp, z):
            self._speed_update(stamp, z)
        self._stop_logic(stamp)
        return self._output(stamp)

    # --- фильтр ------------------------------------------------------------------------------
    def _advance(self, stamp):
        """Предсказание до stamp. False — вход слишком старый (пачка), его пропускаем."""
        if self.t is None:
            self.t = stamp
            return True
        if stamp < self.t - self.p.stale_s:
            return False
        dt = stamp - self.t
        if dt <= 0:
            return True
        if dt > self.p.max_gap_s:
            self.t = stamp
            return True
        while dt > 1e-9:
            h = min(0.1, dt)
            self._predict_step(h)
            dt -= h
        self.t = stamp
        return True

    def _predict_step(self, h):
        u = self.model.command_at(self.t + h)
        a = self.model.accel(u, self.v) + self.d
        if self.v <= 1e-3 and a < 0:
            a = 0.0  # торможением назад не поехать
        ds = self.v * h + 0.5 * a * h * h
        self.v = max(0.0, self.v + a * h)
        if self.ready:
            self.s += max(0.0, ds)
            self.d_since_fix += max(0.0, ds)
        self.t += h
        F = [[1.0, h, 0.5 * h * h], [0.0, 1.0, h], [0.0, 0.0, 1.0]]
        P = _mul(_mul(F, self.P), _transpose(F))
        qa, qd = self.p.sigma_a ** 2, self.p.sigma_d ** 2
        P[0][0] += qa * h ** 3 / 3
        P[0][1] += qa * h * h / 2
        P[1][0] += qa * h * h / 2
        P[1][1] += qa * h
        P[2][2] += qd * h
        self.P = P

    def _wheel_trusted(self, front, stamp, z):
        other = self.last_wheel[not front]
        if other is None or abs(other[0] - stamp) > self.p.bogie_pair_s:
            return True
        zo = other[1]
        if abs(z - zo) <= max(self.p.bogie_tol_abs, self.p.bogie_tol_rel * max(z, zo, self.v)):
            return True
        a_cmd = self.model.accel(self.model.command_at(stamp), self.v)
        if a_cmd > 0.05:        # тяга: буксующее колесо крутится быстрее
            suspect = z > zo
        elif a_cmd < -0.05:     # торможение: колесо в юзе крутится медленнее
            suspect = z < zo
        else:
            suspect = abs(z - self.v) > abs(zo - self.v)
        if suspect:
            self.slip = True
        return not suspect

    def _speed_update(self, stamp, z):
        r = self.p.sigma_wheel ** 2 + (self.p.sigma_wheel_rel * z) ** 2
        if not self.v_init:  # первое показание колёс задаёт скорость: прогон может начаться на ходу
            self._reset_speed(z, r)
            self.v_init = True
            return
        nu = z - self.v
        S = self.P[1][1] + r
        if nu * nu > self.p.gate_chi2 * S:
            if self.reject_since is None:
                self.reject_since = stamp
            if stamp - self.reject_since < self.p.resync_s:
                self.slip = True
                return
            # колёса долго и согласованно расходятся с моделью — ошиблась модель: сброс к колёсам
            self._reset_speed(z, r)
            self.reject_since = None
            return
        self.reject_since = None
        S = self.P[1][1] + r
        K = [self.P[i][1] / S for i in range(3)]
        if self.v < 0.3:
            K[2] = 0.0  # на стоянке возмущение не учим
        self.s += K[0] * nu if self.ready else 0.0
        self.v = max(0.0, self.v + K[1] * nu)
        self.d = max(-self.p.d_max, min(self.p.d_max, self.d + K[2] * nu))
        row = list(self.P[1])
        self.P = [[self.P[i][j] - K[i] * row[j] for j in range(3)] for i in range(3)]

    def _reset_speed(self, z, r):
        self.v, self.d = max(0.0, z), 0.0
        for i in range(3):
            self.P[1][i] = self.P[i][1] = 0.0
            self.P[2][i] = self.P[i][2] = 0.0
        self.P[1][1], self.P[2][2] = r, self.p.sigma_d ** 2 * 25

    # --- привязка к местам стоянок ----------------------------------------------------------
    def _stop_logic(self, stamp):
        if self.v >= self.p.stop_v:
            self.stop_since, self.snapped = None, False
            return
        if self.stop_since is None:
            self.stop_since = stamp
        elif not self.snapped and stamp - self.stop_since >= self.p.stop_confirm_s:
            self.snapped = True
            self._snap()

    def _snap(self):
        if not self.ready or not self.stops or self.mode == 'relative':
            return
        sp = self.map.active_spur
        if sp is not None and self.s < sp['s_join']:
            return
        L = self.map.length
        var_s = self.P[0][0] + (self.p.rel_scale_err * self.d_since_fix) ** 2
        best = None
        for s_st, sig in self.stops:
            nu = (s_st - self.s + L / 2) % L - L / 2
            S = var_s + sig * sig
            if abs(nu) <= min(self.p.stop_gate_sigma * math.sqrt(S), self.p.stop_gate_max_m):
                if best is None or abs(nu) < abs(best[0]):
                    best = (nu, S)
        if best is None:
            return
        nu, S = best
        k = var_s / S
        if self.p.scale_adapt > 0 and self.d_since_fix > self.p.scale_adapt_min_m:
            err = nu / self.d_since_fix
            self.scale = max(-self.p.scale_max, min(self.p.scale_max,
                                                    self.scale + self.p.scale_adapt * k * err))
        self.s += k * nu
        self.P[0][0] = (1 - k) * var_s
        self.P[0][1] = self.P[1][0] = self.P[0][2] = self.P[2][0] = 0.0
        self.d_since_fix = 0.0

    # --- выход -------------------------------------------------------------------------------
    def predict_output(self, stamp):
        """Выход на момент stamp по модели, фильтр не меняется.

        Для публикации, когда все входы молчат: судье нужен выход не реже 10 Гц,
        а в данных все три потока пропадают вместе примерно на секунду.
        """
        if not self.ready or self.t is None or stamp <= self.t:
            return None
        saved = (self.s, self.v, self.d, [row[:] for row in self.P], self.t, self.d_since_fix)
        self._advance(stamp)
        out = self._output(stamp)
        self.s, self.v, self.d, self.P, self.t, self.d_since_fix = saved
        return out

    def _output(self, stamp):
        if not self.ready or (self.last_out is not None and stamp < self.last_out):
            return None
        self.last_out = stamp
        frame = 'map'
        if self.mode == 'relative':  # без GNSS: пройденный путь по оси x от старта
            x, y, z, yaw, frame = self.s, 0.0, 0.0, 0.0, 'odom'
        else:
            x, y, z, yaw = self.map.pose(self.s + self.p.output_lever_m)
            z += self.p.output_dz_m
        if self.p.output_origin == 'start' and self.mode != 'relative':
            if self.origin is None:
                self.origin = (x, y, z)
            x, y, z = x - self.origin[0], y - self.origin[1], z - self.origin[2]
        dropout = all(w is None or stamp - w[0] > 0.5 for w in self.last_wheel.values())
        return {'stamp': stamp, 'v': self.v, 'x': x, 'y': y, 'z': z, 'yaw': yaw, 's': self.s,
                'var_v': self.P[1][1], 'var_s': self.P[0][0] + (self.p.rel_scale_err * self.d_since_fix) ** 2,
                'slip': self.slip, 'dropout': dropout, 'stuck': any(self.stuck.values()), 'scale': self.scale,
                'frame': frame}


def _mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def load_stops(path):
    stops = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line[:1].isdigit():
                s, sigma = line.split(',')[:2]
                stops.append((float(s), float(sigma)))
    return stops
