"""Офлайн-оценка оценщика по метрикам судьи на всех прогонах.

Прогон воспроизводится в порядке времени записи, как ros2 bag play. Оценщик получает
входы (тележки, контроллер) и GNSS master fix, но GNSS принимает только в окне выставки.
Каждый ответ оценщика — аналог публикации /result/* с меткой входа.

Эталон: GNSS master fix в ENU с началом в первой точке прогона (гипотеза о системе судьи)
и скорость GNSS (vel, а если его нет — по координатам). Выход и эталон сопоставляются
по ближайшей метке с допуском 0,05 с.

Запуск: python evaluate.py [--estimator baseline] [--per-vehicle] [--bags N]
        -> out/eval_<имя>.csv и сводка в консоль
"""
import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bagio import CMD, FRONT, GNSS_FIX, GNSS_VEL, MAP_DIR, REAR, bag_ids, load_cached
from tram_odometry.geo import UtmLocal
from tram_odometry.integrated import IntegratedEstimator

OUT = Path(__file__).parent / 'out'
MAP = MAP_DIR / 'route.csv'
TRACTION = Path(__file__).resolve().parents[2] / 'src' / 'tram_odometry' / 'config' / 'traction_table.csv'
MATCH_TOL = 0.05
MIN_DURATION_S = 60.0


class EkfAdapter:
    """TramEstimator из пакета с интерфейсом оценщиков research (выход — кортеж)."""

    def __init__(self, vehicle):
        from baseline import KMH_PER_MPS
        from tram_odometry.estimator import Params, TramEstimator, load_stops
        from tram_odometry.model import TractionModel
        from tram_odometry.route_map import RouteMap
        m = RouteMap.load(MAP)
        if (MAP_DIR / 'route_spurs.csv').exists():
            m.load_spurs(MAP_DIR / 'route_spurs.csv')
        stops = load_stops(MAP_DIR / 'route_stops.csv') if (MAP_DIR / 'route_stops.csv').exists() else []
        p = Params()
        p.wheel_kmh_per_mps = KMH_PER_MPS.get(vehicle, KMH_PER_MPS['default'])
        p.output_frame, p.output_lever_m, p.output_dz_m = 'utm_local', -MASTER_X, -ANT_Z
        for kv in filter(None, os.environ.get('TRAM_PARAMS', '').split(',')):  # подбор: TRAM_PARAMS=sigma_u=0.2,...
            key, value = kv.split('=')
            kind = type(getattr(Params, key))
            setattr(p, key, value.lower() in ('1', 'true', 'yes') if kind is bool else kind(value))
        self.est = TramEstimator(m, TractionModel.load(TRACTION), stops, p)
        self.map = m

    @staticmethod
    def _t(r):
        return None if r is None else (r['stamp'], r['v'], r['x'], r['y'], r['z'], r['s'])

    def on_gnss(self, *a):
        return self._t(self.est.on_gnss(*a))

    def on_wheel(self, *a):
        return self._t(self.est.on_wheel(*a))

    def on_cmd(self, *a):
        return self._t(self.est.on_cmd(*a))

    def on_gnss_rover(self, *a):
        self.est.on_gnss_rover(*a)

    def on_primary(self, *a):
        self.est.on_primary(*a)


class IntegratedAdapter:
    """IntegratedEstimator из PR #12-14 с интерфейсом оценщиков research."""

    def __init__(self, vehicle, gnss_corrections=False):
        from baseline import KMH_PER_MPS
        from tram_odometry.estimator import Params, load_stops
        from tram_odometry.integrated import IntegratedEstimator
        from tram_odometry.integrated_model import IntegratedModel
        from tram_odometry.model import TractionModel
        from tram_odometry.route_map import RouteMap
        m = RouteMap.load(MAP)
        if (MAP_DIR / 'route_spurs.csv').exists():
            m.load_spurs(MAP_DIR / 'route_spurs.csv')
        stops = load_stops(MAP_DIR / 'route_stops.csv') if (MAP_DIR / 'route_stops.csv').exists() else []
        p = Params()
        p.wheel_kmh_per_mps = KMH_PER_MPS.get(vehicle, KMH_PER_MPS['default'])
        p.gnss_corrections = gnss_corrections
        for kv in filter(None, os.environ.get('TRAM_PARAMS', '').split(',')):
            key, value = kv.split('=')
            kind = type(getattr(Params, key))
            setattr(p, key, value.lower() in ('1', 'true', 'yes') if kind is bool else kind(value))
        base_model = TractionModel.load(TRACTION)
        model = IntegratedModel(base_model.v_nodes, base_model.rows, delay_s=0.4)
        self.est = IntegratedEstimator(m, model, stops, p)
        self.map = m

    @staticmethod
    def _t(r):
        return None if r is None else (r['stamp'], r['v'], r['x'], r['y'], r['z'], r['s'])

    def on_gnss(self, *a):
        return self._t(self.est.on_gnss(*a))

    def on_wheel(self, *a):
        return self._t(self.est.on_wheel(*a))

    def on_cmd(self, *a):
        return self._t(self.est.on_cmd(*a))

    def on_gnss_rover(self, *a):
        self.est.on_gnss_rover(*a)

    def on_primary(self, *a):
        self.est.on_primary(*a)


class HybridEstimator(IntegratedEstimator):
    """Комбинация: каузальная динамика и адаптация тяги PR #12 + EKF коррекция позиции и масштаба колес."""

    def __init__(self, route_map, model, stops=(), params=None):
        super().__init__(route_map, model, stops, params)
        # 40 мс задержка согласования с таймингами эталона организаторов
        self.p.output_v_delay_s = 0.040

    def _enter_stub(self):
        # Штатный выбор отвода из TramEstimator Майка
        return super(IntegratedEstimator, self)._enter_stub()

    def _snap(self):
        # Штатный snap остановок из TramEstimator Майка
        return super(IntegratedEstimator, self)._snap()

    def _gnss_correction(self, stamp, lat, lon, alt):
        if not self.p.gnss_corrections or not self.ready or self.t is None or abs(stamp - self.t) > 2.0:
            return None
        # Вызываем EKF коррекцию s и c из TramEstimator (не трогает v)
        return super(IntegratedEstimator, self)._gnss_correction(stamp, lat, lon, alt)

    def _output(self, stamp):
        out = super(IntegratedEstimator, self)._output(stamp)
        if out is not None:
            if self.mode != 'relative':
                x, y, z, yaw = self.map.pose(self.s)
                heading = yaw if self.facing > 0 else math.atan2(-math.sin(yaw), -math.cos(yaw))
                out.update(x=x + self.p.output_lever_m * math.cos(heading),
                           y=y + self.p.output_lever_m * math.sin(heading),
                           z=z + self.p.output_dz_m, yaw=heading)
            out['traction_gain'] = getattr(self.model, 'traction_gain', 1.0)
        return out


class HybridAdapter(IntegratedAdapter):
    """Адаптер для HybridEstimator."""

    def __init__(self, vehicle):
        from baseline import KMH_PER_MPS
        from tram_odometry.estimator import Params, load_stops
        from tram_odometry.integrated_model import IntegratedModel
        from tram_odometry.model import TractionModel
        from tram_odometry.route_map import RouteMap
        m = RouteMap.load(MAP)
        if (MAP_DIR / 'route_spurs.csv').exists():
            m.load_spurs(MAP_DIR / 'route_spurs.csv')
        stops = load_stops(MAP_DIR / 'route_stops.csv') if (MAP_DIR / 'route_stops.csv').exists() else []
        p = Params()
        p.wheel_kmh_per_mps = KMH_PER_MPS.get(vehicle, KMH_PER_MPS['default'])
        p.gnss_corrections = True
        for kv in filter(None, os.environ.get('TRAM_PARAMS', '').split(',')):
            key, value = kv.split('=')
            kind = type(getattr(Params, key))
            setattr(p, key, value.lower() in ('1', 'true', 'yes') if kind is bool else kind(value))
        base_model = TractionModel.load(TRACTION)
        model = IntegratedModel(base_model.v_nodes, base_model.rows, delay_s=0.4)
        self.est = HybridEstimator(m, model, stops, p)
        self.map = m


def make_estimator(name, bag_id, per_vehicle):
    vehicle = bag_id.split('_')[0] if per_vehicle else 'default'
    if name == 'baseline':
        from baseline import BaselineEstimator
        return BaselineEstimator(MAP, vehicle_id=vehicle)
    if name == 'mapstops':
        from map_estimator import MapStopsEstimator
        return MapStopsEstimator(MAP, vehicle_id=vehicle)
    if name == 'ekf':
        return EkfAdapter(vehicle)
    if name == 'integrated':
        return IntegratedAdapter(vehicle, gnss_corrections=False)
    if name == 'integrated_gnss':
        return IntegratedAdapter(vehicle, gnss_corrections=True)
    if name == 'hybrid':
        return HybridAdapter(vehicle)
    raise ValueError(f'неизвестный оценщик {name}')


GNSS_INIT_S, GNSS_EVERY_S, GNSS_BURST_S = 21.0, 150.0, 2.0


def gnss_schedule(a, mode=None):
    """Прореживание GNSS прогона датасета под проверку (TRAM_GNSS).

    'bursts' (по умолчанию) — как в проверочном прогоне организаторов 30618_88aea4d9: первые
    21 с, дальше пачки по 2 с раз в 150 с; 'init' — только первые 21 с; 'all' — всё;
    'late' — первые 60 с GNSS нет, дальше всё (включился по ходу).
    Без этого коррекция по GNSS на датасете (GNSS есть всё время) дала бы нечестную точность.
    """
    mode = mode or os.environ.get('TRAM_GNSS', 'bursts')
    if a is None or mode == 'all' or not len(a):
        return a
    t = a[:, 1] - a[0, 1]
    if mode == 'late':
        return a[t > 60.0]
    keep = t <= GNSS_INIT_S
    if mode == 'bursts':
        keep |= (t > GNSS_INIT_S) & ((t - GNSS_INIT_S) % GNSS_EVERY_S >= GNSS_EVERY_S - GNSS_BURST_S)
    return a[keep]


PRIMARY = '/localization/kinematic_state'   # основной вычислитель: есть только в проверочном прогоне


def replay(est, d):
    """События в порядке времени записи bag -> массив выходов [stamp, v, x, y, z, s].

    Положение основного вычислителя (строки [запись, метка, x, y, z, vx, vy, курс]) подаётся, если
    оно есть в прогоне; оценщик берёт его, только если включён primary_sync.
    """
    events = []
    for topic, kind in ((FRONT, 'f'), (REAR, 'r'), (CMD, 'c'), (GNSS_FIX['master'], 'g'), (GNSS_FIX['rover'], 'R'),
                        (PRIMARY, 'P')):
        a = d.get(topic)
        if kind in 'gR':
            a = gnss_schedule(a)
        if kind == 'P' and (a is None or a.shape[1] < 8 or not hasattr(est, 'on_primary')):
            continue
        if a is not None and (kind != 'R' or hasattr(est, 'on_gnss_rover')):
            events += [(row[0], kind, row) for row in a]
    events.sort(key=lambda e: e[0])
    out = []
    for _, kind, row in events:
        if kind == 'f':
            r = est.on_wheel(row[1], True, row[2])
        elif kind == 'r':
            r = est.on_wheel(row[1], False, row[2])
        elif kind == 'c':
            r = est.on_cmd(row[1], int(row[2]))
        elif kind == 'R':
            est.on_gnss_rover(row[1], row[2], row[3], row[4])
            r = None
        elif kind == 'P':
            est.on_primary(row[1], row[2], row[3], row[7])
            r = None
        else:
            r = est.on_gnss(row[1], row[2], row[3], row[4])
        if r is not None and (not out or r[0] >= out[-1][0]):
            out.append(r)
    return np.array(out) if out else np.zeros((0, 6))


MASTER_X, ROVER_X, ANT_Z = -9.873, 2.563, 3.0  # tf антенн в base_link от организаторов


def reference(d):
    """Эталон как у судьи: base_link в сетке MGRS (UTM 37N минус 300000/6100000), высота рельса.

    base_link лежит на оси между антеннами master (x = -9.873) и rover (x = +2.563), на 3 м ниже.
    Нет rover — сдвиг от master на 9.873 м по направлению движения.
    """
    fix = d[GNSS_FIX['master']]
    grid = UtmLocal()
    p = np.array([grid.forward(la, lo, al) for la, lo, al in fix[:, 2:5]])
    t = fix[:, 1]
    rover = d.get(GNSS_FIX['rover'])
    w = -MASTER_X / (ROVER_X - MASTER_X)
    if rover is not None and len(rover) > 10:
        pr = np.array([grid.forward(la, lo, al) for la, lo, al in rover[:, 2:5]])
        near = np.abs(np.interp(t, rover[:, 1], rover[:, 1]) - t) < 0.2
        pri = np.column_stack([np.interp(t, rover[:, 1], pr[:, k]) for k in range(3)])
        axis = pri[:, :2] - p[:, :2]
        ok = near & (np.abs(np.hypot(*axis.T) - (ROVER_X - MASTER_X)) < 1.5)
    else:
        ok = np.zeros(len(t), bool)
        pri = p
    tang = np.gradient(p[:, :2], axis=0)
    tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
    base = p.copy()
    base[:, :2] = np.where(ok[:, None], p[:, :2] + w * (pri[:, :2] - p[:, :2]), p[:, :2] - MASTER_X * tang)
    base[:, 2] = np.where(ok, p[:, 2] + w * (pri[:, 2] - p[:, 2]), p[:, 2]) - ANT_Z
    p = base
    keep = [0]
    for i in range(1, len(p)):
        if math.hypot(*(p[i, :2] - p[keep[-1], :2])) / max(t[i] - t[keep[-1]], 1e-3) < 25.0:
            keep.append(i)
    t, p = t[keep], p[keep]
    vel = d.get(GNSS_VEL['master'])
    if vel is not None and len(vel) > 10:
        tv, sp = vel[:, 1], np.hypot(vel[:, 2], vel[:, 3])
    else:  # скорость по координатам: центральная разность через ±0,5 с
        lo = np.searchsorted(t, t - 0.5)
        hi = np.clip(np.searchsorted(t, t + 0.5), 0, len(t) - 1)
        dt = np.maximum(t[hi] - t[lo], 1e-3)
        tv, sp = t, np.hypot(*(p[hi, :2] - p[lo, :2]).T) / dt
    return t, p, tv, sp


def match(t_ref, t_out):
    j = np.clip(np.searchsorted(t_out, t_ref), 1, len(t_out) - 1)
    j = np.where(np.abs(t_out[j - 1] - t_ref) < np.abs(t_out[j] - t_ref), j - 1, j)
    ok = np.abs(t_out[j] - t_ref) <= MATCH_TOL
    return j, ok


def evaluate_bag(args):
    bag_id, name, per_vehicle = args
    d = load_cached(bag_id)
    if GNSS_FIX['master'] not in d or len(d[GNSS_FIX['master']]) < 50:
        return None
    t_ref, p_ref, t_vref, v_ref = reference(d)
    if t_ref[-1] - t_ref[0] < MIN_DURATION_S:
        return None
    out = replay(make_estimator(name, bag_id, per_vehicle), d)
    row = {'bag': bag_id, 'duration_s': round(t_ref[-1] - t_ref[0], 1), 'outputs': len(out)}
    if len(out) < 10:
        return row
    t_out = out[:, 0]
    row['rate_hz'] = round(len(out) / (t_out[-1] - t_out[0]), 1)

    j, ok = match(t_vref, t_out)
    ev = out[j[ok], 1] - v_ref[ok]
    row.update(v_matched=round(ok.mean(), 3), v_rmse=round(float(np.sqrt(np.mean(ev ** 2))), 3),
               v_mae=round(float(np.mean(np.abs(ev))), 3), v_bias=round(float(np.mean(ev)), 3))

    j, ok = match(t_ref, t_out)
    ep = out[j[ok], 2:5] - p_ref[ok]
    e3 = np.linalg.norm(ep, axis=1)
    # вдольпутевая составляющая: проекция ошибки на направление движения по эталону
    tang = np.gradient(p_ref[:, :2], axis=0)[ok]
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    along = np.sum(ep[:, :2] * tang, axis=1)
    dist = float(np.sum(np.hypot(*np.diff(p_ref[:, :2], axis=0).T)))
    row.update(p_matched=round(ok.mean(), 3), dist_m=round(dist, 1),
               p3d_mean=round(float(e3.mean()), 2), p3d_rmse=round(float(np.sqrt(np.mean(e3 ** 2))), 2),
               p3d_max=round(float(e3.max()), 2), p3d_final=round(float(e3[-1]), 2),
               drift_pct=round(float(e3[-1] / max(dist, 1.0) * 100), 3),
               along_rmse=round(float(np.sqrt(np.mean(along ** 2))), 2),
               along_max=round(float(np.abs(along).max()), 2),
               z_rmse=round(float(np.sqrt(np.mean(ep[:, 2] ** 2))), 2))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--estimator', default='baseline')
    ap.add_argument('--per-vehicle', action='store_true', help='коэффициент колеса по номеру трамвая')
    ap.add_argument('--bags', type=int, default=0, help='только первые N прогонов (для отладки)')
    ap.add_argument('--tag', default='', help='суффикс имени файла результатов')
    a = ap.parse_args()
    ids = bag_ids()[:a.bags] if a.bags else bag_ids()
    with ProcessPoolExecutor(10) as ex:
        rows = [r for r in ex.map(evaluate_bag, [(b, a.estimator, a.per_vehicle) for b in ids]) if r]
    OUT.mkdir(exist_ok=True)
    name = a.estimator + ('_pv' if a.per_vehicle else '') + (f'_{a.tag}' if a.tag else '')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / f'eval_{name}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    good = [r for r in rows if 'v_rmse' in r]
    print(f'{name}: оценено {len(good)} прогонов из {len(ids)}')
    for key in ('v_rmse', 'v_mae', 'v_bias', 'p3d_mean', 'p3d_rmse', 'p3d_max', 'p3d_final', 'drift_pct',
                'along_rmse', 'z_rmse', 'rate_hz', 'v_matched', 'p_matched'):
        vals = np.array([r[key] for r in good if key in r], dtype=float)
        print(f'  {key:11s} медиана {np.median(vals):9.3f}   среднее {vals.mean():9.3f}   худший {vals.max():9.3f}')


if __name__ == '__main__':
    main()
