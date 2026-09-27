"""Онлайн-оценщик скорости и положения трамвая: EKF вдоль маршрута на модели тяги.

Состояние x = [s, v, d, c]: s — дистанция вдоль маршрута (м), v — скорость (м/с), d — возмущающее
ускорение (уклон, масса, ошибка модели; случайное блуждание, м/с²), c — ошибка масштаба колёс:
тележка показывает v·(1 + c). Масштаб от прогона к прогону гуляет на ±1,5 % (износ, обточка колёс),
и c учится на привязках к стоянкам: расхождение пути с картой делится между s и c.

Предсказание: a = модель(u(t − задержка), v) + d, где u — позиция контроллера.
Измерения: скорость каждой тележки (км/ч / k). Защита от недостоверной одометрии:
  * гейтинг по невязке (χ²): выброс или проскальзывание не проходит в фильтр;
  * согласование тележек: при тяге буксующее колесо показывает больше, при торможении в юзе — меньше;
    показание, ушедшее от прогноза в невозможную для режима сторону, — отказ датчика;
  * проскальзывание по ускорению: колесо разгоняется или тормозит быстрее, чем позволяет модель, —
    тележка исключается (ловит и одновременное проскальзывание обеих тележек), но остаётся границей:
    на тяге колесо не крутится медленнее трамвая, на торможении — не быстрее;
  * пропуски колёс — работает только модель; долгое расхождение с моделью — пересинхронизация;
  * команда контроллера устарела (пропуск, пачка) — модель знает ручку хуже, колёсам доверия больше.
Положение: s -> карта линии (с отводами у конечных); на стоянке — привязка к месту регулярной
стоянки. GNSS — только выставка в первые секунды прогона: положение по антенне master, курс на
стоянке — по второй антенне rover (12,4 м впереди): так выбирается путь нужного направления.
Чистый Python: нода без зависимостей.
"""
import math
from collections import deque

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
    resync_agree_s = 0.6         # а если обе тележки согласны между собой — так быстро (экстренное торможение
                                 #   не видно по ручке контроллера: 30618_616ec56b, −4 м/с² при ручке 0)
    cmd_stale_s = 0.3            # команда старше (с учётом задержки модели) — ручка неизвестна, с
    sigma_a_stale = 1.0          # СКО ошибки модели, пока ручка неизвестна или ей не верим, м/с²
    distrust_s = 3.0             # после сброса к колёсам столько не верим модели: ручка не объясняет движение
    fail_abs = 1.0               # показание ушло от прогноза «не в ту сторону» дальше (на тяге — ниже,
    fail_rel = 0.2               #   на торможении — выше) — отказ датчика, а не проскальзывание; м/с, доля
    slip_win_s = 0.5             # окно оценки ускорения колеса, с
    slip_acc = 0.7               # колесо разгоняется (тяга) или тормозит (тормоз) быстрее модели на столько, м/с²,
    slip_dv = 0.3                #   и уже ушло от прогноза дальше, м/с — проскальзывание тележки
    slip_exit = 0.4              # вернулось к прогнозу ближе — проскальзывание кончилось, м/с
    slip_recover_s = 3.0         # столько после буксования резкий спад колеса — возврат к скорости трамвая, а не юз
    slip_max_s = 8.0             # дольше не исключаем (в данных эпизоды до 15 с, 90 % — до 9 с), с
    a_phys = 8.0                 # |ускорение колеса| больше — скачок показания, отказ датчика, м/с²
    spike_max = 10               # столько скачков подряд — не выбросы, а новый уровень показаний
    a_trac_max = 2.0             # трамвай не разгоняется быстрее, м/с² ...
    a_brake_max = 5.0            # ... и не тормозит быстрее (экстренное торможение), м/с²: согласные тележки
                                 #   с ускорением в этих пределах — правда, а не проскальзывание
    bogie_tol_abs = 0.3          # допустимое расхождение тележек, м/с
    bogie_tol_rel = 0.05         # и доля скорости
    bogie_pair_s = 0.25          # показания тележек сравниваем, если они не дальше по времени
    stale_s = 0.3                # вход старше времени фильтра на столько — отбрасываем
    max_gap_s = 30.0             # дольше — не интегрируем (разрыв записи)
    stop_v = 0.05                # скорость «стоим», м/с
    stop_confirm_s = 2.0         # стоим дольше — привязка к месту стоянки
    stop_gate_sigma = 3.0
    stop_gate_max_m = 60.0
    sigma_c0 = 0.002             # априорное СКО масштаба колёс. Разброс по прогонам 0,3 % (30618) и 0,7 % (30639),
                                 #   но смелее учить нельзя: ложная привязка (светофор у стоянки) уходит в масштаб
    sigma_c_rw = 2e-5            # его медленное блуждание, 1/√с
    sigma_u = 0.2                # необъяснённая ошибка пути (недопойманное проскальзывание, карта), м/√м:
                                 #   такие ошибки привязка исправляет в s, не списывая на масштаб
    scale_max = 0.03             # ограничение |c|
    # система координат выхода. Судья сравнивает с /localization/kinematic_state (QA 25.09):
    # base_link в сетке MGRS = UTM 37N минус (300000, 6100000) — как pathgraph организаторов.
    #   output_frame: 'utm_local' — сетка MGRS/pathgraph; 'enu' — ENU с началом в первой точке GNSS
    #   output_origin: 'frame' — как в рамке; 'start' — вычесть положение выходной точки на старте
    #   output_lever_m / output_dz_m — от антенны master до base_link: +9.873 м вперёд по пути,
    #   −3.0 м по высоте (tf антенн от организаторов)
    init_still_m = 3.0           # разброс точек GNSS меньше — стоим, берём медиану
    init_jump_m = 10.0           # точка дальше медианы — прыжок GNSS, если трамвай стоит
    rover_base_m = 12.436        # от master до rover по оси трамвая (tf антенн: −9,873 и +2,563 м)
    rover_tol_m = 1.5            # медианы антенн дальше или ближе на столько — курсу по ним не верим
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
        self.s = self.v = self.d = self.c = 0.0
        self.P = [[25.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.01, 0.0],
                  [0.0, 0.0, 0.0, self.p.sigma_c0 ** 2]]
        self.ready = False
        self.last_wheel = {True: None, False: None}
        self.repeat = {True: (None, 0), False: (None, 0)}   # (последнее значение, повторов подряд)
        self.stuck = {True: False, False: False}
        self.v_init = False
        self.last_cmd = None              # метка последней команды контроллера
        self.distrust_until = None        # до этой метки модели не верим (после сброса к колёсам)
        self.hist = {True: deque(maxlen=32), False: deque(maxlen=32)}  # (метка, скорость) по тележке
        self.bad = {True: None, False: None}  # тележка исключена: (с какого момента, 'slip' | 'skid' | 'fail')
        self.last_slip = None             # когда последний раз буксовала любая тележка
        self.spikes = {True: 0, False: 0}  # выбросов подряд по тележке
        self.reject_since = None
        self.stop_since = None
        self.snapped = False
        self.d_since_fix = 0.0            # путь с последней привязки (выставки или стоянки), м
        self.slip = False
        self.last_out = None
        self.origin = None                # положение выходной точки на старте (output_origin='start')
        self.init_fixes = []              # точки GNSS окна выставки в системе выхода
        self.rover_fixes = []             # (lat, lon, alt) антенны rover в окне выставки
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
        yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 3.0 else self._rover_yaw(x, y)
        self._advance(stamp)
        self.s, dist = self.map.locate_start(x, y, yaw)
        for i in range(1, 4):
            self.P[0][i] = self.P[i][0] = 0.0
        self.P[0][0] = max(1.0, dist * dist)
        self.d_since_fix = 0.0
        self.ready = True
        if self.t is None or stamp > self.t:
            self.t = stamp
        return self._output(stamp)

    def on_gnss_rover(self, stamp, lat, lon, alt):
        """Антенна rover в окне выставки. False — окно прошло, подписку можно снять."""
        if self.mode == 'relative' or (self.t0_gnss is not None and stamp - self.t0_gnss > self.p.init_window_s):
            return False
        if all(math.isfinite(x) for x in (stamp, lat, lon, alt)):
            self.rover_fixes.append((lat, lon, alt))
        return True

    def _rover_yaw(self, x, y):
        """Курс трамвая по антеннам: от master (x, y) к медиане rover; None — rover нет или он не сходится."""
        if not self.rover_fixes:
            return None
        pts = [self.enu.forward(*f) for f in self.rover_fixes]
        rx = sorted(p[0] for p in pts)[len(pts) // 2]
        ry = sorted(p[1] for p in pts)[len(pts) // 2]
        if abs(math.hypot(rx - x, ry - y) - self.p.rover_base_m) > self.p.rover_tol_m:
            return None
        return math.atan2(ry - y, rx - x)

    def on_cmd(self, stamp, position):
        if not math.isfinite(stamp):
            return None
        self.model.push_command(stamp, position)
        if self.last_cmd is None or stamp > self.last_cmd:
            self.last_cmd = stamp
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
        z = kmh / self.p.wheel_kmh_per_mps / (1.0 + self.c)  # скорость по тележке с поправкой масштаба
        other = self.last_wheel[not front]
        agree = (other is not None and abs(other[0] - stamp) <= self.p.bogie_pair_s and
                 abs(z - other[1]) <= max(self.p.bogie_tol_abs, self.p.bogie_tol_rel * max(z, other[1], self.v)))
        kind = self._bogie_bad(front, stamp, z, agree)
        if kind == 'spike':  # одиночный выброс: не используем и не запоминаем
            self.slip = True
            self._stop_logic(stamp)
            return self._output(stamp)
        self.last_wheel[front] = (stamp, z)
        if kind == 'recover':
            self._speed_update(stamp, z, gate=False)
        elif kind is None:
            if self._wheel_trusted(front, stamp, z):
                self._speed_update(stamp, z, resync_s=self.p.resync_agree_s if agree else self.p.resync_s)
        else:
            self.slip = True
            # буксующее колесо — верхняя граница скорости, колесо в юзе — нижняя: модель за неё не пускаем
            if (kind == 'slip' and z < self.v) or (kind == 'skid' and z > self.v):
                self._speed_update(stamp, z, gate=False)
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
        stale = (self.last_cmd is None or self.t + h - self.model.delay_s - self.last_cmd > self.p.cmd_stale_s or
                 (self.distrust_until is not None and self.t + h < self.distrust_until))
        a = self.model.accel(u, self.v) + self.d
        if self.v <= 1e-3 and a < 0:
            a = 0.0  # торможением назад не поехать
        ds = self.v * h + 0.5 * a * h * h
        self.v = max(0.0, self.v + a * h)
        if self.ready:
            self.s += max(0.0, ds)
            self.d_since_fix += max(0.0, ds)
        self.t += h
        F = [[1.0, h, 0.5 * h * h, 0.0], [0.0, 1.0, h, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        P = _mul(_mul(F, self.P), _transpose(F))
        qa, qd = (self.p.sigma_a_stale if stale else self.p.sigma_a) ** 2, self.p.sigma_d ** 2
        P[0][0] += qa * h ** 3 / 3
        P[0][1] += qa * h * h / 2
        P[1][0] += qa * h * h / 2
        P[1][1] += qa * h
        P[2][2] += qd * h
        P[3][3] += self.p.sigma_c_rw ** 2 * h
        P[0][0] += self.p.sigma_u ** 2 * max(0.0, ds)
        self.P = P

    def _wheel_trusted(self, front, stamp, z):
        other = self.last_wheel[not front]
        if other is None or abs(other[0] - stamp) > self.p.bogie_pair_s or self.bad[not front]:
            return True
        zo = other[1]
        if abs(z - zo) <= max(self.p.bogie_tol_abs, self.p.bogie_tol_rel * max(z, zo, self.v)):
            return True
        a_cmd = self.model.accel(self.model.command_at(stamp), self.v)
        # проскальзывание уводит показание только в одну сторону: на тяге вверх, на торможении вниз.
        # Ушло далеко в другую (на ходу при тяге показывает 0) — отказ этого датчика
        far = max(self.p.fail_abs, self.p.fail_rel * self.v)
        if a_cmd > 0.05:        # тяга: буксующее колесо крутится быстрее
            suspect = z > zo if self.v - min(z, zo) < far else z < zo
        elif a_cmd < -0.05:     # торможение: колесо в юзе крутится медленнее
            suspect = z < zo if max(z, zo) - self.v < far else z > zo
        else:
            suspect = abs(z - self.v) > abs(zo - self.v)
        if suspect:
            self.slip = True
        return not suspect

    def _bogie_bad(self, front, stamp, z, agree):
        """Тележка проскальзывает или отказала: None — всё в порядке, иначе 'spike', 'slip', 'skid', 'fail'.

        Выброс (spike): скачок от прошлого показания этой тележки быстрее физически возможного, и вторая
        тележка его не подтверждает. Выброс не попадает ни в фильтр, ни в историю тележки.

        Буксование (slip): колесо разгоняется быстрее, чем позволяет модель тяги, и уже ушло от
        прогноза вверх; юз (skid) — тормозит быстрее модели торможения и ушло вниз. Отказ (fail):
        скачок быстрее физически возможного. Тележка исключается, пока не вернётся к прогнозу, но не
        дольше slip_max_s. Если вторая тележка показывает то же (agree), это правда, пока ускорение
        физически возможно для трамвая; невозможное — одновременное проскальзывание обеих тележек,
        которое согласование тележек не видит.
        """
        hist = self.hist[front]
        if hist and not agree and self.spikes[front] < self.p.spike_max:
            t_prev, z_prev = hist[-1]
            if abs(z - z_prev) > self.p.a_phys * max(stamp - t_prev, 0.0) + self.p.slip_dv:
                self.spikes[front] += 1
                return 'spike'
        self.spikes[front] = 0
        hist.append((stamp, z))
        while len(hist) > 2 and stamp - hist[1][0] >= self.p.slip_win_s:
            hist.popleft()
        if self.bad[front] is not None:
            since, kind = self.bad[front]
            if kind == 'slip':
                self.last_slip = stamp
            back = max(self.p.slip_exit, 2.0 * math.sqrt(self.P[1][1]))
            if abs(z - self.v) <= back or stamp - since > self.p.slip_max_s:
                self.bad[front] = None
                return None
            return kind
        if not self.v_init:
            return None
        t_old, z_old = hist[0]
        if stamp - t_old < 0.6 * self.p.slip_win_s:
            return None
        a_w = (z - z_old) / (stamp - t_old)
        u = self.model.command_at(stamp)
        a_m = self.model.accel(u, self.v) + self.d
        dv = z - self.v
        recent_slip = self.last_slip is not None and stamp - self.last_slip < self.p.slip_recover_s
        if recent_slip and agree and a_w < 0 and dv < -self.p.slip_dv:
            # колесо после буксования падает к скорости трамвая, а прогноз утянут буксованием вверх
            return 'recover'
        # фильтр успевает подтянуться за колесом: при большом избытке ускорения отход от прогноза не нужен
        far = dv > self.p.slip_dv or a_w - a_m > 2.0 * self.p.slip_acc
        if abs(a_w) > self.p.a_phys and abs(dv) > self.p.slip_dv and not agree:
            kind = 'fail'
        elif u > 0 and a_w - a_m > self.p.slip_acc and far and (not agree or a_w > self.p.a_trac_max):
            kind = 'slip'
            self.last_slip = stamp
        elif u < 0 and a_m - a_w > self.p.slip_acc and -dv > self.p.slip_dv and (not agree or a_w < -self.p.a_brake_max):
            kind = 'skid'
        else:
            return None
        self.bad[front] = (stamp, kind)
        return kind

    def _speed_update(self, stamp, z, gate=True, resync_s=None):
        """Показание тележки z (м/с, уже с поправкой масштаба) — измерение v·(1 + c).

        gate=False — без проверки невязки: показание-граница при проскальзывании.
        resync_s — сколько ждать при расхождении с моделью до сброса к колёсам.
        """
        g = 1.0 + self.c
        r = self.p.sigma_wheel ** 2 + (self.p.sigma_wheel_rel * z * g) ** 2
        if not self.v_init:  # первое показание колёс задаёт скорость: прогон может начаться на ходу
            self._reset_speed(z, r)
            self.v_init = True
            return
        H = (0.0, g, 0.0, self.v)
        PH = [sum(self.P[i][j] * H[j] for j in range(4)) for i in range(4)]
        S = sum(H[i] * PH[i] for i in range(4)) + r
        nu = (z - self.v) * g
        if gate and nu * nu > self.p.gate_chi2 * S:
            if self.reject_since is None:
                self.reject_since = stamp
            if stamp - self.reject_since < (self.p.resync_s if resync_s is None else resync_s):
                self.slip = True
                return
            # колёса долго и согласованно расходятся с моделью — ошиблась модель: сброс к колёсам
            self._reset_speed(z, r)
            self.reject_since = None
            self.distrust_until = stamp + self.p.distrust_s
            return
        self.reject_since = None
        K = [PH[i] / S for i in range(4)]
        if self.v < 0.3:
            K[2] = 0.0  # на стоянке возмущение не учим
        K[3] = 0.0      # масштаб — только по привязкам к карте: по скорости он неотличим от ошибки модели
        self.s += K[0] * nu if self.ready else 0.0
        self.v = max(0.0, self.v + K[1] * nu)
        self.d = max(-self.p.d_max, min(self.p.d_max, self.d + K[2] * nu))
        self._joseph(K, H, r)

    def _joseph(self, K, H, r):
        """P = (I − K H) P (I − K H)ᵀ + K r Kᵀ: верно и для урезанного усиления K."""
        A = [[(1.0 if i == j else 0.0) - K[i] * H[j] for j in range(4)] for i in range(4)]
        P = _mul(_mul(A, self.P), _transpose(A))
        self.P = [[P[i][j] + K[i] * r * K[j] for j in range(4)] for i in range(4)]

    def _reset_speed(self, z, r):
        self.v, self.d = max(0.0, z), 0.0
        for i in range(4):
            self.P[1][i] = self.P[i][1] = 0.0
            self.P[2][i] = self.P[i][2] = 0.0
        self.P[1][1], self.P[2][2] = r / (1.0 + self.c) ** 2, self.p.sigma_d ** 2 * 25

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
        var_s = self.P[0][0]
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
        # s и масштаб колёс c: невязка делится между ними по корреляции, накопленной с прошлой привязки
        K = [self.P[0][0] / S, 0.0, 0.0, self.P[3][0] / S]
        self.s += K[0] * nu
        self.c = max(-self.p.scale_max, min(self.p.scale_max, self.c + K[3] * nu))
        self._joseph(K, (1.0, 0.0, 0.0, 0.0), S - var_s)
        self.d_since_fix = 0.0

    # --- выход -------------------------------------------------------------------------------
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
                'var_v': self.P[1][1], 'var_s': self.P[0][0],
                'slip': self.slip or any(v is not None for v in self.bad.values()), 'dropout': dropout,
                'stuck': any(self.stuck.values()), 'scale': self.c,
                'frame': frame}


def _mul(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def _transpose(A):
    return [list(row) for row in zip(*A)]


def load_stops(path):
    stops = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line[:1].isdigit():
                s, sigma = line.split(',')[:2]
                stops.append((float(s), float(sigma)))
    return stops
