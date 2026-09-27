"""Конфиг ноды совпадает с кодом: те же ключи и те же типы (ROS 2 не приводит int к double)."""
from dataclasses import fields
from pathlib import Path

import pytest

from tram_odometry.estimator import FilterParams
from tram_odometry.model import ModelParams
from tram_odometry.params import NODE_DEFAULTS
from tram_odometry.slip import SlipParams
from tram_odometry.stops import StopParams

yaml = pytest.importorskip('yaml')
CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'tram_odometry.yaml'


def _params():
    return yaml.safe_load(CONFIG.read_text(encoding='utf-8'))['tram_odometry']['ros__parameters']


def test_top_level_keys_and_types_match_defaults():
    cfg = {k: v for k, v in _params().items() if not isinstance(v, dict)}
    assert set(cfg) == set(NODE_DEFAULTS)
    for key, value in cfg.items():
        assert type(value) is type(NODE_DEFAULTS[key]), key


SECTIONS = [('model', ModelParams), ('filter', FilterParams), ('slip', SlipParams), ('stops', StopParams)]


@pytest.mark.parametrize('section, cls', SECTIONS)
def test_sections_match_dataclasses(section, cls):
    cfg = _params()[section]
    defaults = {f.name: f.default for f in fields(cls)}
    assert set(cfg) == set(defaults)
    for key, value in cfg.items():
        assert type(value) is type(defaults[key]), f'{section}.{key}'
