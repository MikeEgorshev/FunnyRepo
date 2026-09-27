"""Поправка от основного вычислителя (primary_sync) на проверочном прогоне организаторов.

Эталон прогона /localization/kinematic_state подаётся ещё и как положение основного вычислителя: оценщик
берёт его, только пока есть GNSS. Варианты: GNSS как в прогоне или без первых 60 с (включается по ходу,
первая пачка — на 201-й секунде) × primary_sync выключен или включён. Метрики — после 202 с и на участке
202–1280 с (маршрут pathgraph, до развилки у западной конечной).

Запуск: python check_primary.py [путь к папке bag]  ->  таблица в консоль
"""
import os
import sys
from pathlib import Path

import numpy as np

import check_bag
import evaluate
from bagio import DATASET


def main(path):
    d = check_bag.load(path)
    ref = d[check_bag.REF]
    t0 = ref[0, 1]
    print(f'{path.name}: 3D-ошибка после включения GNSS (с 202 с), м')
    print(f'{"GNSS":6s} {"primary_sync":13s} {"RMSE":>10s} {"среднее":>10s} {"в конце":>8s} {"RMSE 202–1280 с":>16s}')
    for gnss, sync in (('all', False), ('all', True), ('late', False), ('late', True)):
        os.environ['TRAM_GNSS'], os.environ['TRAM_PARAMS'] = gnss, 'primary_sync=1' if sync else ''
        out = evaluate.replay(evaluate.EkfAdapter(path.name.split('_')[0]), d)
        t_out = out[:, 0]
        j = np.clip(np.searchsorted(t_out, ref[:, 1]), 1, len(t_out) - 1)
        j = np.where(np.abs(t_out[j - 1] - ref[:, 1]) < np.abs(t_out[j] - ref[:, 1]), j - 1, j)
        near = np.abs(t_out[j] - ref[:, 1]) <= check_bag.TOL
        e3 = np.linalg.norm(out[j, 2:5] - ref[:, 2:5], axis=1)
        after = near & (ref[:, 1] - t0 > 202.0)
        route = after & (ref[:, 1] - t0 < 1280.0)
        print(f'{gnss:6s} {str(sync).lower():13s} {np.sqrt(np.mean(e3[after] ** 2)):10.2f} {e3[after].mean():10.2f} '
              f'{e3[after][-1]:8.2f} {np.sqrt(np.mean(e3[route] ** 2)):16.2f}', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET / 'check' / 'check-code' / 'bags' / '30618_88aea4d9')
