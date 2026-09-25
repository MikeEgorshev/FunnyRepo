"""Сводка по всем прогонам: частоты, пропуски, диапазоны, масштаб скорости колёс к GNSS.

Запуск: python summarize_bags.py  ->  out/bags_summary.csv
"""
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bagio import CMD, FRONT, GNSS_FIX, GNSS_VEL, REAR, bag_ids, load

OUT = Path(__file__).parent / 'out'


def gnss_speed(vel):
    return np.hypot(vel[:, 2], vel[:, 3])


def summarize(bag_id):
    d = load(bag_id)
    row = {'bag': bag_id, 'vehicle': bag_id.split('_')[0]}
    t_all = np.concatenate([a[:, 0] for a in d.values()])
    row['duration_s'] = round(t_all.max() - t_all.min(), 1)
    for name, topic in (('front', FRONT), ('rear', REAR), ('cmd', CMD), ('fix_m', GNSS_FIX['master']),
                        ('vel_m', GNSS_VEL['master']), ('fix_r', GNSS_FIX['rover']), ('vel_r', GNSS_VEL['rover'])):
        a = d.get(topic)
        row[f'{name}_n'] = 0 if a is None else len(a)
        if a is not None and len(a) > 2:
            dt = np.diff(a[:, 0])
            row[f'{name}_hz'] = round(1 / np.median(dt), 1)
            row[f'{name}_maxgap_s'] = round(dt.max(), 2)
    for name, topic in (('front', FRONT), ('rear', REAR)):
        a = d.get(topic)
        if a is not None and len(a):
            row[f'{name}_max'] = round(a[:, 2].max(), 2)
            row[f'{name}_min'] = round(a[:, 2].min(), 2)
    cmd = d.get(CMD)
    if cmd is not None and len(cmd):
        row['cmd_min'], row['cmd_max'] = int(cmd[:, 2].min()), int(cmd[:, 2].max())
    vel = d.get(GNSS_VEL['master'])
    front = d.get(FRONT)
    if vel is not None and len(vel) > 10 and front is not None and len(front) > 10:
        sp = gnss_speed(vel)
        row['gnss_speed_max'] = round(sp.max(), 2)
        # масштаб: скорость колёс (как записана) / скорость GNSS на равномерном движении > 3 м/с
        fr = np.interp(vel[:, 0], front[:, 0], front[:, 2])
        mask = sp > 3.0
        if mask.sum() > 20:
            row['front_to_gnss_ratio_med'] = round(float(np.median(fr[mask] / sp[mask])), 4)
    fix = d.get(GNSS_FIX['master'])
    if fix is not None and len(fix):
        row['lat0'], row['lon0'] = round(fix[0, 2], 6), round(fix[0, 3], 6)
        row['lat1'], row['lon1'] = round(fix[-1, 2], 6), round(fix[-1, 3], 6)
        row['fix_status'] = ' '.join(str(int(s)) for s in sorted(set(fix[:, 5])))
    return row


def main():
    OUT.mkdir(exist_ok=True)
    ids = bag_ids()
    with ProcessPoolExecutor(max_workers=10) as ex:
        rows = list(ex.map(summarize, ids))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('bag', 'vehicle', 'duration_s'), k))
    with open(OUT / 'bags_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} bags -> {OUT / "bags_summary.csv"}')


if __name__ == '__main__':
    main()
