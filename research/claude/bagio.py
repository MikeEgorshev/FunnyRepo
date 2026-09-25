"""Чтение прогонов датасета в numpy без ROS (через rosbags).

Путь к датасету — переменная окружения TRAM_DATASET; если её нет, ищется папка
dataset/ с подпапкой data/ выше по дереву от этого файла (см. data/README.md).
"""
import os
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


def _find_dataset():
    if 'TRAM_DATASET' in os.environ:
        return Path(os.environ['TRAM_DATASET'])
    for parent in Path(__file__).resolve().parents:
        if (parent / 'dataset' / 'data').is_dir():
            return parent / 'dataset'
    raise FileNotFoundError('dataset/data не найден: задайте TRAM_DATASET')


DATASET = _find_dataset()
BAGS = DATASET / 'data'
MSGS = DATASET / 'tram_vehicle_msgs' / 'msg'

FRONT = '/vehicle/front_bogie_velocity'
REAR = '/vehicle/rear_bogie_velocity'
CMD = '/vehicle/driver_position_cmd'
GNSS_FIX = {'master': '/sensing/gnss/master/fix', 'rover': '/sensing/gnss/rover/fix'}
GNSS_VEL = {'master': '/sensing/gnss/master/vel', 'rover': '/sensing/gnss/rover/vel'}


def typestore():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for name in ('VelocitySensor', 'DriverControllerCommand'):
        types.update(get_types_from_msg((MSGS / f'{name}.msg').read_text(), f'tram_vehicle_msgs/msg/{name}'))
    ts.register(types)
    return ts


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def load(bag_id, topics=None):
    """Возвращает dict topic -> np.ndarray.

    Колонки: [bag_time, header_stamp, ...значения]:
      bogie: velocity (как записано, км/ч); cmd: position;
      fix: lat, lon, alt, status; vel: vx, vy, vz.
    """
    out = {}
    with AnyReader([BAGS / bag_id], default_typestore=typestore()) as reader:
        conns = [c for c in reader.connections if topics is None or c.topic in topics]
        for conn, t, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            row = [t * 1e-9, _stamp(m)]
            if conn.topic in (FRONT, REAR):
                row.append(m.velocity)
            elif conn.topic == CMD:
                row.append(m.position)
            elif conn.topic.endswith('/fix'):
                row += [m.latitude, m.longitude, m.altitude, m.status.status]
            elif conn.topic.endswith('/vel'):
                v = m.twist.linear
                row += [v.x, v.y, v.z]
            out.setdefault(conn.topic, []).append(row)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def bag_ids():
    return sorted(p.name for p in BAGS.iterdir() if p.is_dir())


CACHE = DATASET / 'cache'


def load_cached(bag_id):
    """То же, что load(), но с кэшем в dataset/cache/<bag>.npz (вне репозитория)."""
    path = CACHE / f'{bag_id}.npz'
    if path.exists():
        with np.load(path) as z:
            return {k.replace('|', '/'): z[k] for k in z.files}
    d = load(bag_id)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, **{k.replace('/', '|'): v for k, v in d.items()})
    return d
