"""Честная оценка: карта и отводы строятся по одной половине прогонов, проверка — на другой.

Одинаковые прогоны (в датасете есть точные дубликаты) попадают в одну половину, иначе утечка.
Карты складываются в dataset/cv/fold<k>/ (вне репозитория).

Запуск: python cv.py [--estimator baseline|mapstops] [--per-vehicle]  ->  out/eval_<оценщик>_cvfold*.csv и сводка
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from bagio import DATASET, GNSS_FIX, bag_ids, load_cached

HERE = Path(__file__).parent


def signature(bag_id):
    fix = load_cached(bag_id).get(GNSS_FIX['master'])
    if fix is None or not len(fix):
        return bag_id
    return tuple(np.round(fix[:20, 2:4], 7).ravel()) + (len(fix),)


def run(script, env, *args):
    subprocess.run([sys.executable, str(HERE / script), *args], env=env, check=True,
                   stdout=subprocess.DEVNULL if script != 'evaluate.py' else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-vehicle', action='store_true')
    ap.add_argument('--estimator', default='baseline')
    a = ap.parse_args()
    groups = {}
    for b in bag_ids():
        groups.setdefault(signature(b), []).append(b)
    folds = ([], [])
    for k, members in enumerate(sorted(groups.values())):
        folds[k % 2].extend(members)
    print(f'{len(groups)} уникальных прогонов из {sum(map(len, folds))}; половины: {len(folds[0])} и {len(folds[1])}')
    rows = []
    for k in (0, 1):
        train, test = folds[1 - k], folds[k]
        cvdir = DATASET / 'cv' / f'fold{k}'
        cvdir.mkdir(parents=True, exist_ok=True)
        (cvdir / 'train.txt').write_text('\n'.join(train), encoding='utf-8')
        (cvdir / 'test.txt').write_text('\n'.join(test), encoding='utf-8')
        env = dict(os.environ, TRAM_MAP_DIR=str(cvdir), TRAM_BAGS=str(cvdir / 'train.txt'),
                   KMP_DUPLICATE_LIB_OK='TRUE', PYTHONIOENCODING='utf-8')
        for script in ('build_route_map.py', 'calibrate_route_s.py', 'build_spurs.py', 'build_stops.py'):
            run(script, env)
        env['TRAM_BAGS'] = str(cvdir / 'test.txt')
        tag = f'cvfold{k}'
        run('evaluate.py', env, '--estimator', a.estimator, '--tag', tag, *(['--per-vehicle'] if a.per_vehicle else []))
        name = a.estimator + ('_pv' if a.per_vehicle else '') + f'_{tag}'
        with open(HERE / 'out' / f'eval_{name}.csv', encoding='utf-8') as f:
            rows += list(csv.DictReader(f))
    good = [r for r in rows if r.get('v_rmse')]
    print(f'\nКРОСС-ВАЛИДАЦИЯ: {len(good)} прогонов')
    for key in ('v_rmse', 'v_bias', 'p3d_mean', 'p3d_rmse', 'p3d_final', 'drift_pct', 'along_rmse', 'z_rmse'):
        vals = np.array([float(r[key]) for r in good])
        print(f'  {key:11s} медиана {np.median(vals):9.3f}   среднее {vals.mean():9.3f}   худший {vals.max():9.3f}')


if __name__ == '__main__':
    main()
