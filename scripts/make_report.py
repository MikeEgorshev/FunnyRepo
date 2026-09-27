"""Данные для презентации жюри: собирает их и вшивает в docs/presentation.html.

Источники:
- сценарии сбоев — src/tram_odometry/test/synthetic.py прогоняется через тот же Estimator,
  что и в ноде (истина известна точно);
- результаты конвейера — results.json от scripts/pipeline.py (карта, подобранная модель,
  метрики на отложенных прогонах). На настоящих данных запускать так же;
- реальное время — JSON от scripts/realtime_check.py (последняя строка вывода,
  --realtime "название=файл.json") или замер по умолчанию (Humble, 2 ядра, 512 МБ);
- факты о настоящем датасете — сводка по 122 прогонам (PR #2, docs/data.md);
- настоящий проверочный прогон организаторов (--real-bag, с эталоном судьи
  /localization/kinematic_state) и вывод их судьи hackathon_solution_checker (--judge).

Запуск:
    python3 scripts/make_report.py --results results/results.json [--realtime "Без карты=rt1.json" ...] \
        [--real-bag <прогон> --real-route route.csv --real-stops stops.csv --judge "Судья=metrics.log"]
Страница пересобирается на месте: меняется только блок <script id="report-data">.
"""
import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src' / 'tram_odometry'), str(ROOT / 'src' / 'tram_odometry' / 'test'),
                str(ROOT / 'scripts')]
import evaluate_bags as EB  # noqa: E402
from synthetic import Scenario, run  # noqa: E402
from tram_odometry.estimator import Estimator  # noqa: E402
from tram_odometry.model import ModelParams  # noqa: E402
from tram_odometry.track import MgrsLocal, RouteTrack  # noqa: E402

PAGE = ROOT / 'docs' / 'presentation.html'

# Замер реального времени (scripts/check_ros.sh, ros:humble-ros-base, --cpus=2 --memory=512m --network none).
REALTIME = [
    {'name': 'Без GNSS, все входы молчат 1,5 с', 'rate_hz': 25.02, 'max_gap_ms': 61.0, 'p50_ms': 21.6, 'p99_ms': 47.2,
     'max_ms': 97.0, 'cpu_pct': 4.4, 'rss_mb': 58.9, 'frame': 'odom'},
    {'name': 'GNSS на старте: выставка и курс', 'rate_hz': 25.02, 'max_gap_ms': 64.8, 'p50_ms': 22.7, 'p99_ms': 48.4,
     'max_ms': 49.6, 'cpu_pct': 4.6, 'rss_mb': 59.0, 'frame': 'map'},
]

# Настоящий датасет: сводка research/claude/out/bags_summary.csv (PR #2). k — медиана
# «скорость тележки, км/ч / скорость GNSS, м/с» на ходу по прогонам, где есть скорость GNSS.
DATASET = {
    'runs': 122, 'hours': 36.7, 'median_min': 19.8, 'line_km': 7.5, 'alt_m': [142, 177], 'v_max': 15.94,
    'no_gnss_runs': 36, 'front_gap_max_s': 19.8, 'rear_gap_max_s': 73.5, 'k_median': 3.5966,
    'vehicles': {'30618': {'runs': 103, 'k_min': 3.5438, 'k_med': 3.5966, 'k_max': 3.6119},
                 '30639': {'runs': 19, 'k_min': 3.5855, 'k_med': 3.6168, 'k_max': 3.6584}},
    'topics': [
        ['/vehicle/front_bogie_velocity', '10 Гц', 'км/ч, а не м/с; пропуски до 19,8 с; пачки до 150 Гц'],
        ['/vehicle/rear_bogie_velocity', '10 Гц', 'пропуски до 73,5 с; пачки до 190 Гц'],
        ['/vehicle/driver_position_cmd', '20 Гц', 'позиции −15…+15; пропуски до 1 с'],
        ['/sensing/gnss/*/fix', '10 Гц', 'в проверке — только первые секунды; status 0/1/2'],
    ],
}

SCENARIOS = [
    ('clean', 'Чистый ход', 'Нет сбоев. Модель фильтра нарочно ошибается на 8–10 %.', {}),
    ('slip', 'Буксование', 'Передняя тележка завышает скорость на 25 % в тяге (82–96 с).',
     {'slip': [(82.0, 96.0)], 'slip_ratio': 0.25}),
    ('slide', 'Юз', 'Обе тележки занижают скорость в торможении (52–68 и 212–228 с).',
     {'slide': [(52.0, 68.0), (212.0, 228.0)]}),
    ('dropout', 'Пропуск тележек', 'Обе тележки молчат 20 с на ходу (120–140 с).', {'dropout': [(120.0, 140.0)]}),
    ('all', 'Всё сразу', 'Буксование, юз и пропуск в одном прогоне.',
     {'slip': [(82.0, 96.0)], 'slide': [(212.0, 228.0)], 'dropout': [(120.0, 140.0)], 'slip_ratio': 0.25}),
]


def scenario(key, title, text, kw, every=0.5):
    sc = Scenario(duration=240.0, **kw)
    events, truth = run(sc)
    est = Estimator()
    k0 = est.p.k0
    last = {'front': 0.0, 'rear': 0.0}
    s_raw = v_raw = t_prev = 0.0
    ti, next_out = 0, 0.0
    rows = {'t': [], 'v_true': [], 'v_est': [], 'v_raw': [], 'e_est': [], 'e_raw': [], 'slip': [], 'sigma': []}
    for t, kind, value in events:
        s_raw += v_raw * (t - t_prev)
        t_prev = t
        if kind == 'cmd':
            est.set_notch(t, value)
        else:
            est.wheel(kind, t, value)
            last[kind] = value / k0
            v_raw = 0.5 * (last['front'] + last['rear'])
        if t >= next_out:
            while ti + 1 < len(truth) and truth[ti + 1][0] <= t:
                ti += 1
            st = est.state()
            rows['t'].append(round(t, 2))
            rows['v_true'].append(round(truth[ti][2], 3))
            rows['v_est'].append(round(st.v, 3))
            rows['v_raw'].append(round(v_raw, 3))
            rows['e_est'].append(round(st.s - truth[ti][1], 2))
            rows['e_raw'].append(round(s_raw - truth[ti][1], 2))
            rows['slip'].append(1 if st.slip else 0)
            rows['sigma'].append(round(st.var_s ** 0.5, 2))
            next_out += every
    spans = {name: kw.get(name, []) for name in ('slip', 'slide', 'dropout')}
    return {'key': key, 'title': title, 'text': text, 'spans': spans, 'distance_m': round(truth[-1][1], 1),
            'end_est': rows['e_est'][-1], 'end_raw': rows['e_raw'][-1], 'series': rows}


def realtime_entry(label, path):
    """Строка JSON от realtime_check.py -> запись для страницы."""
    r = json.loads(Path(path).read_text(encoding='utf-8').strip().splitlines()[-1])
    return {'name': label, 'rate_hz': r.get('odom_hz'), 'max_gap_ms': r.get('max_output_gap_ms'),
            'p50_ms': r.get('latency_ms_p50'), 'p99_ms': r.get('latency_ms_p99'), 'max_ms': r.get('latency_ms_max'),
            'cpu_pct': r.get('node_cpu_percent_of_one_core'), 'rss_mb': r.get('node_rss_mb_max'),
            'frame': r.get('last_frame_id')}


def judge_entry(label, path):
    """Последний отчёт судьи hackathon_solution_checker из его лога."""
    vel = pos = None
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        if 'Velocity metrics' in line:
            vel = line
        elif 'Position metrics' in line:
            pos = line
    out = {'name': label}
    for text in (vel, pos):
        for name, rmse, mx, n in re.findall(r'(\w+): RMSE=([\d.naN]+), max=([\d.naN]+), n=(\d+)', text or ''):
            out[name] = {'rmse': float(rmse), 'max': float(mx), 'n': int(n)}
    return out


def real_run(bag, route_csv=None, stops_csv=None, judges=(), every_s=1.0):
    """Настоящий прогон: метрики как у судьи и ряды для графиков по нашему оценщику."""
    events = EB.read_bag(bag)
    ref_pos, ref_vel = EB.reference(events)
    route = RouteTrack.load(route_csv, MgrsLocal(), stops_csv) if route_csv else None
    variants = {'no_map': dict(route=None)}
    if route is not None:
        variants = {'map': dict(route=route), 'map_corr': dict(route=route, corrections=True), **variants}
    outs = {k: EB.replay(events, **kw) for k, kw in variants.items()}
    metrics = {k: EB.metrics(o, ref_pos, ref_vel) for k, o in outs.items()}
    main = outs.get('map') or outs['no_map']
    corr = outs.get('map_corr')
    times = [o[0] for o in main]
    ctimes = [o[0] for o in corr] if corr else []
    vel = dict(ref_vel)
    tr = {k: [] for k in ('t', 'v_ref', 'v_est', 'err', 'err_corr', 'x_ref', 'y_ref', 'x_est', 'y_est')}
    t0, nxt = ref_pos[0][0], ref_pos[0][0]
    for t, x, y, z in ref_pos:
        if t < nxt:
            continue
        nxt = t + every_s
        j = EB._nearest(times, t)
        if j is None:
            continue
        o = main[j]
        tr['t'].append(round(t - t0, 1))
        tr['v_ref'].append(round(vel.get(t, 0.0), 3))
        tr['v_est'].append(round(o[1], 3))
        tr['err'].append(round(math.dist((x, y, z), o[2:5]), 2) if o[6] == 'map' else None)
        jc = EB._nearest(ctimes, t) if corr else None
        tr['err_corr'].append(round(math.dist((x, y, z), corr[jc][2:5]), 2) if jc is not None and corr[jc][6] == 'map'
                              else None)
        tr['x_ref'].append(round(x, 1))
        tr['y_ref'].append(round(y, 1))
        tr['x_est'].append(round(o[2], 1))
        tr['y_est'].append(round(o[3], 1))
    out = {'bag': Path(bag).name, 'duration_s': round(ref_pos[-1][0] - t0, 1), 'metrics': metrics, 'trace': tr,
           'judge': [judge_entry(*j.split('=', 1)) for j in judges]}
    if route is not None:                   # саму геометрию карты в страницу не кладём — только её размеры
        out['route_info'] = {'length_m': route.length, 'stops': len(route.stops)}
    return out


def build(results=None, realtime=None, real=None):
    res = json.loads(Path(results).read_text(encoding='utf-8')) if results else None
    source = None
    if res:
        names = res['dataset'].get('test_bags', [])
        source = 'synthetic' if names and all(n.startswith('synth_') for n in names) else 'real'
    return {
        'generated': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()),
        'source': source,
        'model_default': asdict(ModelParams()),
        'scenarios': [scenario(*s) for s in SCENARIOS],
        'pipeline': res,
        'realtime': realtime or REALTIME,
        'dataset': DATASET,
        'real': real,
    }


def inject(data, page=PAGE):
    html = Path(page).read_text(encoding='utf-8')
    blob = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    new, n = re.subn(r'(<script id="report-data" type="application/json">)(.*?)(</script>)',
                     lambda m: m.group(1) + blob + m.group(3), html, flags=re.S)
    if n != 1:
        raise SystemExit('в странице нет блока <script id="report-data" type="application/json">')
    Path(page).write_text(new, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=None, help='results.json от scripts/pipeline.py')
    ap.add_argument('--realtime', nargs='*', default=None, help='"название=файл" с выводом realtime_check.py')
    ap.add_argument('--page', default=str(PAGE))
    ap.add_argument('--real-bag', default=None, help='настоящий прогон с эталоном /localization/kinematic_state')
    ap.add_argument('--real-route', default=None)
    ap.add_argument('--real-stops', default=None)
    ap.add_argument('--judge', nargs='*', default=(), help='"название=лог" судьи hackathon_solution_checker')
    args = ap.parse_args()
    rt = None
    if args.realtime:
        rt = [realtime_entry(*item.split('=', 1)) for item in args.realtime]
    real = real_run(args.real_bag, args.real_route, args.real_stops, args.judge) if args.real_bag else None
    data = build(args.results, rt, real)
    inject(data, args.page)
    print('данные вшиты в', args.page, f'({len(json.dumps(data)) // 1024} КБ)')


if __name__ == '__main__':
    main()
