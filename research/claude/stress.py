"""Стресс-тест оценщика: синтетические помехи на входах поверх настоящих прогонов.

Помехи накладываются на копию прогона; эталон (GNSS) считается по чистому прогону. Каждый сценарий
сравнивается с тем же прогоном без помех: тот же оценщик, та же карта.

Сценарии (параметры помех — по настоящим эпизодам из anomalies.py: 0.3–15 с, пик 1–4 м/с,
почти всегда на обеих тележках сразу):
  slip1 / slip2   — буксование на тяге одной / обеих тележек: +1…4 м/с на 1–8 с
  skid1 / skid2   — юз на торможении одной / обеих тележек: −1…5 м/с на 1–8 с
  drop1 / drop2   — одна / обе тележки молчат 5–30 с
  stuck1          — показание тележки замирает на 5–20 с (датчик завис)
  zero1           — тележка на ходу показывает 0 на 5–20 с (отказ датчика)
  spikes          — 1 % показаний каждой тележки — мусор: NaN, 999, −50, ×3, +30 км/ч, 0
  noise           — шум σ = 1 км/ч на обеих тележках весь прогон
  burst           — пачки: входы за 1–3 с приходят разом (раз в ~30 с)
  cmd_drop        — контроллер молчит 5–30 с
  scale           — колёса врут на +2 % весь прогон (износ, дрейф параметров)
  no_gnss         — GNSS на старте нет: относительная одометрия
  gnss_jump       — на старте треть точек GNSS отскакивает на 20–200 м
  mix             — slip1, skid2, drop1, stuck1, zero1, spikes, burst, cmd_drop вместе

Запуск: python stress.py [--bags N] [--scenarios a,b,...]  ->  out/stress.csv и сводка в консоль
"""
import argparse
import csv
import math
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bagio import CMD, FRONT, GNSS_FIX, REAR, bag_ids, load_cached
from evaluate import MIN_DURATION_S, EkfAdapter, match, reference

OUT = Path(__file__).parent / 'out'
SCENARIOS = ['clean', 'slip1', 'slip2', 'skid1', 'skid2', 'drop1', 'drop2', 'stuck1', 'zero1', 'spikes',
             'noise', 'burst', 'cmd_drop', 'scale', 'no_gnss', 'gnss_jump', 'mix']
AFTER_S = 3.0          # окно помехи + столько после: там меряем всплеск ошибки


# --- помехи ------------------------------------------------------------------------------------
def trap(t, t0, t1, rise=0.3, fall=0.5):
    """Трапеция 0..1: нарастание rise, полка, спад fall."""
    return np.clip(np.minimum((t - t0) / rise, (t1 - t) / fall), 0.0, 1.0)


class Perturber:
    def __init__(self, d, k, seed):
        self.d = {key: a.copy() for key, a in d.items()}
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.windows = []                      # [(t0, t1)] по меткам header.stamp
        w = d.get(FRONT) if FRONT in d else d.get(REAR)
        self.tw, self.vw = w[:, 1], w[:, 2] / k
        c = d.get(CMD)
        self.tc, self.uc = (c[:, 1], c[:, 2]) if c is not None else (np.zeros(1), np.zeros(1))
        self.t_start, self.t_end = self.tw[0] + 10.0, self.tw[-1] - 10.0

    def _u(self, t):
        return self.uc[np.clip(np.searchsorted(self.tc, t) - 1, 0, len(self.uc) - 1)]

    def _v(self, t):
        return np.interp(t, self.tw, self.vw)

    def pick(self, n, dur, cond, gap=15.0):
        """n непересекающихся окон длительностью dur (кортеж min, max) там, где cond(t0, t1)."""
        out, tries = [], 0
        while len(out) < n and tries < 400:
            tries += 1
            T = self.rng.uniform(*dur)
            t0 = self.rng.uniform(self.t_start, max(self.t_start + 1.0, self.t_end - T))
            t1 = t0 + T
            if any(t0 < b + gap and t1 > a - gap for a, b in out + self.windows):
                continue
            if cond(t0, t1):
                out.append((t0, t1))
        self.windows += out
        return out

    def n_events(self, per_s=150.0):
        return max(1, int((self.t_end - self.t_start) / per_s))

    def moving(self, vmin):
        return lambda t0, t1: np.min(self._v(np.linspace(t0, t1, 10))) > vmin

    def traction(self, t0, t1):
        ts = np.linspace(t0, t1, 10)
        return np.median(self._u(ts)) > 0 and np.min(self._v(ts)) > 1.0

    def braking(self, t0, t1):
        ts = np.linspace(t0, t1, 10)
        return np.median(self._u(ts)) < 0 and np.min(self._v(ts)) > 3.0

    # отдельные помехи
    def slip(self, both, sign):
        cond = self.traction if sign > 0 else self.braking
        for t0, t1 in self.pick(self.n_events(), (1.0, 8.0), cond):
            amp = self.rng.uniform(1.0, 4.0 if sign > 0 else 5.0)
            topics = (FRONT, REAR) if both else (self.rng.choice([FRONT, REAR]),)
            for topic in topics:
                a = self.d.get(topic)
                if a is None:
                    continue
                t = a[:, 1]
                v = a[:, 2] / self.k
                dv = amp * self.rng.uniform(0.8, 1.2) * trap(t, t0, t1)
                dv *= 1.0 + 0.2 * self.rng.standard_normal(len(t)) * (dv > 0)
                v = v + dv if sign > 0 else v - np.minimum(dv, 0.9 * v)
                a[:, 2] = v * self.k

    def drop(self, both):
        for t0, t1 in self.pick(self.n_events(200.0), (5.0, 30.0), self.moving(0.0)):
            for topic in (FRONT, REAR) if both else (self.rng.choice([FRONT, REAR]),):
                a = self.d.get(topic)
                if a is not None:
                    self.d[topic] = a[(a[:, 1] < t0) | (a[:, 1] > t1)]

    def freeze(self, zero):
        for t0, t1 in self.pick(self.n_events(200.0), (5.0, 20.0), self.moving(3.0)):
            a = self.d.get(self.rng.choice([FRONT, REAR]))
            if a is None:
                continue
            m = (a[:, 1] >= t0) & (a[:, 1] <= t1)
            if m.any():
                a[m, 2] = 0.0 if zero else a[np.argmax(m), 2]

    def spikes(self, frac=0.01):
        junk = [math.nan, 999.0, -50.0, None, 30.0, 0.0]   # None — ×3
        for topic in (FRONT, REAR):
            a = self.d.get(topic)
            if a is None:
                continue
            idx = self.rng.choice(len(a), size=max(1, int(frac * len(a))), replace=False)
            for i in idx:
                j = junk[self.rng.integers(len(junk))]
                a[i, 2] = a[i, 2] * 3.0 if j is None else (a[i, 2] + j if j == 30.0 else j)
            self.windows += [(a[i, 1], a[i, 1]) for i in idx]

    def noise(self, sigma_kmh=1.0):
        for topic in (FRONT, REAR):
            a = self.d.get(topic)
            if a is not None:
                a[:, 2] = np.maximum(0.0, a[:, 2] + sigma_kmh * self.rng.standard_normal(len(a)))

    def burst(self):
        for t0, t1 in self.pick(max(1, int((self.t_end - self.t_start) / 30.0)), (1.0, 3.0),
                                lambda *_: True, gap=5.0):
            for topic in (FRONT, REAR, CMD):
                a = self.d.get(topic)
                if a is not None:
                    m = (a[:, 1] >= t0) & (a[:, 1] <= t1)
                    a[m, 0] = a[m, 0].max() + 0.01 if m.any() else 0.0

    def cmd_drop(self):
        a = self.d.get(CMD)
        if a is None:
            return
        for t0, t1 in self.pick(self.n_events(200.0), (5.0, 30.0), self.moving(0.0)):
            a = a[(a[:, 1] < t0) | (a[:, 1] > t1)]
        self.d[CMD] = a

    def scale(self, f=1.02):
        for topic in (FRONT, REAR):
            if topic in self.d:
                self.d[topic][:, 2] *= f
        self.windows = [(self.t_start, self.t_end)]

    def no_gnss(self):
        for topic in list(self.d):
            if topic.startswith('/sensing/gnss'):
                del self.d[topic]

    def gnss_jump(self):
        fix = self.d[GNSS_FIX['master']]
        early = np.where(fix[:, 1] < fix[0, 1] + 5.0)[0]
        idx = self.rng.choice(early, size=max(1, len(early) // 3), replace=False)
        idx = idx[idx > 0]  # первая точка — честная: иначе прыжок неотличим от старта
        for i in idx:
            r, ang = self.rng.uniform(20.0, 200.0), self.rng.uniform(0, 2 * math.pi)
            fix[i, 2] += r * math.cos(ang) / 111320.0
            fix[i, 3] += r * math.sin(ang) / (111320.0 * math.cos(math.radians(fix[i, 2])))
        self.windows = [(fix[0, 1], fix[0, 1] + 60.0)]


def perturb(d, k, scenario, seed):
    p = Perturber(d, k, seed)
    steps = {
        'clean': [],
        'slip1': [lambda: p.slip(False, +1)], 'slip2': [lambda: p.slip(True, +1)],
        'skid1': [lambda: p.slip(False, -1)], 'skid2': [lambda: p.slip(True, -1)],
        'drop1': [lambda: p.drop(False)], 'drop2': [lambda: p.drop(True)],
        'stuck1': [lambda: p.freeze(False)], 'zero1': [lambda: p.freeze(True)],
        'spikes': [p.spikes], 'noise': [p.noise], 'burst': [p.burst], 'cmd_drop': [p.cmd_drop],
        'scale': [p.scale], 'no_gnss': [p.no_gnss], 'gnss_jump': [p.gnss_jump],
        'mix': [lambda: p.slip(False, +1), lambda: p.slip(True, -1), lambda: p.drop(False),
                lambda: p.freeze(False), lambda: p.freeze(True), p.spikes, p.burst, p.cmd_drop],
    }[scenario]
    for step in steps:
        step()
    return p.d, p.windows


# --- прогон ------------------------------------------------------------------------------------
def replay(est, d):
    """Как evaluate.replay, но с флагами: [stamp, v, x, y, z, s, slip, stuck, dropout, relative]."""
    events = []
    for topic, kind in ((FRONT, 'f'), (REAR, 'r'), (CMD, 'c'), (GNSS_FIX['master'], 'g'), (GNSS_FIX['rover'], 'R')):
        a = d.get(topic)
        if a is not None:
            events += [(row[0], kind, row) for row in a]
    events.sort(key=lambda e: e[0])
    out = []
    for _, kind, row in events:
        if kind in 'fr':
            r = est.on_wheel(row[1], kind == 'f', row[2])
        elif kind == 'c':
            r = est.on_cmd(row[1], int(row[2]))
        elif kind == 'R':
            est.on_gnss_rover(row[1], row[2], row[3], row[4])
            r = None
        else:
            r = est.on_gnss(row[1], row[2], row[3], row[4])
        if r is not None:
            out.append((r['stamp'], r['v'], r['x'], r['y'], r['z'], r['s'], r['slip'], r['stuck'],
                        r['dropout'], r['frame'] == 'odom'))
    return np.array(out) if out else np.zeros((0, 10))


def wheel_distance(d, k, t0, t1):
    """Путь по чистым колёсам между t0 и t1: длина ломаной GNSS на стоянках копит шум."""
    a = d.get(FRONT) if FRONT in d else d[REAR]
    m = (a[:, 1] >= t0) & (a[:, 1] <= t1)
    return float(np.sum(np.diff(a[m, 1]) * a[m, 2][:-1] / k))


def in_windows(t, windows, after=AFTER_S):
    m = np.zeros(len(t), bool)
    for t0, t1 in windows:
        m |= (t >= t0) & (t <= t1 + after)
    return m


def run_one(args):
    bag_id, scenario = args
    d = load_cached(bag_id)
    if GNSS_FIX['master'] not in d or len(d[GNSS_FIX['master']]) < 50 or FRONT not in d and REAR not in d:
        return None
    t_ref, p_ref, t_vref, v_ref = reference(d)
    if t_ref[-1] - t_ref[0] < MIN_DURATION_S:
        return None
    adapter = EkfAdapter(bag_id.split('_')[0])
    k = adapter.est.p.wheel_kmh_per_mps
    seed = zlib.crc32(f'{bag_id}/{scenario}'.encode())
    d2, windows = perturb(d, k, scenario, seed)
    row = {'bag': bag_id, 'scenario': scenario, 'events': len(windows)}
    try:
        out = replay(adapter.est, d2)
    except Exception:  # noqa: BLE001 — падение оценщика и есть результат теста
        row['crash'] = traceback.format_exc(limit=3).strip().splitlines()[-1]
        return row
    row['crash'] = ''
    row['outputs'] = len(out)
    row['nan_out'] = int(np.sum(~np.isfinite(out[:, 1:6]).all(axis=1))) if len(out) else 0
    if len(out) < 10:
        return row
    t_out = out[:, 0]
    row['rate_hz'] = round(len(out) / (t_out[-1] - t_out[0]), 1)
    row['gap_max_s'] = round(float(np.max(np.diff(t_out))), 2)

    j, ok = match(t_vref, t_out)
    ev = out[j[ok], 1] - v_ref[ok]
    row.update(v_rmse=round(float(np.sqrt(np.mean(ev ** 2))), 3), v_max=round(float(np.abs(ev).max()), 2))
    win = in_windows(t_vref[ok], windows)
    if win.any() and scenario not in ('clean', 'scale'):
        row['v_rmse_win'] = round(float(np.sqrt(np.mean(ev[win] ** 2))), 3)
        row['v_max_win'] = round(float(np.abs(ev[win]).max()), 2)
        # сырые колёса в тех же окнах: среднее доступных тележек после помех, где они есть
        raw = []
        for topic in (FRONT, REAR):
            a = d2.get(topic)
            if a is not None and len(a) > 2:
                good = np.isfinite(a[:, 2]) & (np.abs(a[:, 2]) < 500)
                raw.append(np.interp(t_vref[ok][win], a[good, 1], a[good, 2] / k))
        if raw:
            er = np.mean(raw, axis=0) - v_ref[ok][win]
            row['raw_rmse_win'] = round(float(np.sqrt(np.mean(er ** 2))), 3)
    jo = in_windows(t_out, windows)
    if windows and scenario not in ('clean', 'scale', 'no_gnss', 'gnss_jump', 'noise', 'burst', 'cmd_drop'):
        row['flag_win'] = round(float(np.mean(out[jo, 6:9].any(axis=1))), 3) if jo.any() else ''
    row['flag_clean'] = round(float(np.mean(out[~jo, 6:9].any(axis=1))), 3) if (~jo).any() else ''

    j, ok = match(t_ref, t_out)
    relative = bool(out[-1, 9])
    if relative:
        # без GNSS выход x — путь от старта. Эталон пути — s того же прогона без помех (по карте,
        # с отводами и привязками к стоянкам): длина ломаной GNSS на стоянках копит шум, а проекция
        # на кольцо не видит отводов у конечных
        ref = replay(EkfAdapter(bag_id.split('_')[0]).est, d)
        s_ref = np.interp(t_out, ref[:, 0], ref[:, 5])
        e3 = np.abs(out[j[ok], 2] - (s_ref[j[ok]] - s_ref[0]))
    else:
        e3 = np.linalg.norm(out[j[ok], 2:5] - p_ref[ok], axis=1)
    dist = wheel_distance(d, k, t_ref[0], t_ref[-1])
    row.update(relative=int(relative), dist_m=round(dist, 1), p3d_mean=round(float(e3.mean()), 2),
               p3d_max=round(float(e3.max()), 2), p3d_final=round(float(e3[-1]), 2),
               drift_pct=round(float(e3[-1] / max(dist, 1.0) * 100), 3))
    return row


def summary(rows):
    by = {}
    for r in rows:
        by.setdefault(r['scenario'], {})[r['bag']] = r
    clean = by.get('clean', {})
    keys = ('v_rmse', 'v_max', 'v_rmse_win', 'v_max_win', 'raw_rmse_win', 'p3d_mean', 'p3d_max', 'p3d_final',
            'drift_pct', 'flag_win', 'flag_clean')
    print(f'{"сценарий":10s} {"прог":>4s} {"пад":>3s} {"NaN":>3s} ' + ' '.join(f'{k:>12s}' for k in keys)
          + f' {"Δp3d_mean":>10s} {"Δ худш":>8s}')
    for sc in SCENARIOS:
        if sc not in by:
            continue
        rr = list(by[sc].values())
        crashes = sum(1 for r in rr if r.get('crash'))
        nans = sum(1 for r in rr if r.get('nan_out'))
        cells = []
        for key in keys:
            vals = np.array([float(r[key]) for r in rr if r.get(key) not in (None, '')])
            cells.append(f'{np.median(vals):7.3f}/{vals.max():<5.4g}' if len(vals) else f'{"—":>12s}')
        dp = np.array([float(r['p3d_mean']) - float(clean[b]['p3d_mean'])
                       for b, r in by[sc].items() if 'p3d_mean' in r and b in clean and 'p3d_mean' in clean[b]])
        tail = f' {np.median(dp):10.2f} {dp.max():8.1f}' if len(dp) else ''
        print(f'{sc:10s} {len(rr):4d} {crashes:3d} {nans:3d} ' + ' '.join(f'{c:>12s}' for c in cells) + tail)
    print('ячейки: медиана/худший по прогонам; Δ — p3d_mean против того же прогона без помех, м')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bags', type=int, default=0, help='каждый N-й прогон (0 — все)')
    ap.add_argument('--scenarios', default=','.join(SCENARIOS))
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    ids = bag_ids()
    if a.bags:
        ids = ids[::max(1, len(ids) // a.bags)][:a.bags]
    scenarios = [s for s in a.scenarios.split(',') if s]
    if 'clean' not in scenarios:
        scenarios.insert(0, 'clean')
    with ProcessPoolExecutor(10) as ex:
        rows = [r for r in ex.map(run_one, [(b, s) for s in scenarios for b in ids], chunksize=1) if r]
    OUT.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / f'stress{"_" + a.tag if a.tag else ""}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary(rows)


if __name__ == '__main__':
    main()
