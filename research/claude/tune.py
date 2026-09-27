"""Подбор параметров оценщика по кросс-валидации: карты половин уже построены cv.py.

Каждый набор параметров — строка вида 'sigma_u=0.2,sigma_c0=0.004' (поля Params); прогоняется
evaluate.py на проверочной половине каждой из двух карт, сводка — по всем проверочным прогонам.

Запуск: python tune.py 'sigma_u=0.2' 'sigma_u=0.3,sigma_c0=0.004' ...  ->  сводка в консоль
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from bagio import DATASET

HERE = Path(__file__).parent
KEYS = ('v_rmse', 'v_bias', 'p3d_mean', 'p3d_final', 'drift_pct', 'along_rmse')


def run(cfg):
    rows = []
    for k in (0, 1):
        cvdir = DATASET / 'cv' / f'fold{k}'
        env = dict(os.environ, TRAM_MAP_DIR=str(cvdir), TRAM_BAGS=str(cvdir / 'test.txt'), TRAM_PARAMS=cfg,
                   KMP_DUPLICATE_LIB_OK='TRUE', PYTHONIOENCODING='utf-8')
        subprocess.run([sys.executable, str(HERE / 'evaluate.py'), '--estimator', 'ekf', '--per-vehicle',
                        '--tag', f'tune{k}'], env=env, check=True, stdout=subprocess.DEVNULL)
        with open(HERE / 'out' / f'eval_ekf_pv_tune{k}.csv', encoding='utf-8') as f:
            rows += [r for r in csv.DictReader(f) if r.get('p3d_mean')]
    return rows


def main():
    print(f'{"параметры":36s} {"прог":>4s} ' + ' '.join(f'{k:>18s}' for k in KEYS))
    for cfg in sys.argv[1:] or ['']:
        rows = run(cfg)
        cells = []
        for key in KEYS:
            v = np.array([float(r[key]) for r in rows])
            cells.append(f'{np.median(v):6.3f}/{v.mean():5.2f}/{v.max():5.1f}')
        print(f'{cfg or "по умолчанию":36s} {len(rows):4d} ' + ' '.join(f'{c:>18s}' for c in cells), flush=True)
    print('ячейки: медиана/среднее/худший')


if __name__ == '__main__':
    main()
