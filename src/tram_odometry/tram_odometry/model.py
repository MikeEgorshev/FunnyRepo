"""Модель тяги и торможения: ускорение трамвая по позиции контроллера и скорости.

Таблица a(u, v) откалибрована по данным (research/claude/build_traction.py): строки — позиции
контроллера -15..15, столбцы — узлы скорости. Между узлами — линейная интерполяция по скорости.
Физически это «постоянная сила до базовой скорости, дальше постоянная мощность» для тяги и
тормозная сила, растущая с номером тормозной позиции. Команда действует с задержкой delay_s.
"""
import bisect
from collections import deque


class TractionModel:
    def __init__(self, v_nodes, rows, delay_s=0.4):
        self.v_nodes = list(v_nodes)
        self.rows = {int(u): list(r) for u, r in rows.items()}
        self.delay_s = delay_s
        self._cmd = deque(maxlen=400)  # (stamp, u), ~20 с при 20 Гц

    @classmethod
    def load(cls, path, delay_s=0.4):
        v_nodes, rows = None, {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if parts[0] == 'u':
                    v_nodes = [float(x) for x in parts[1:]]
                else:
                    rows[int(parts[0])] = [float(x) for x in parts[1:]]
        if v_nodes is None or len(rows) != 31:
            raise ValueError(f'{path}: ожидалась таблица 31 × N с заголовком u,...')
        return cls(v_nodes, rows, delay_s)

    def push_command(self, stamp, u):
        if self._cmd and stamp < self._cmd[-1][0]:
            return  # старая команда из пачки — порядок важнее полноты
        self._cmd.append((stamp, max(-15, min(15, int(u)))))

    def command_at(self, t):
        """Позиция контроллера, действующая в момент t (с учётом задержки); 0, если команд нет."""
        if not self._cmd:
            return 0
        t -= self.delay_s
        u = self._cmd[0][1]
        for stamp, value in reversed(self._cmd):
            if stamp <= t:
                u = value
                break
        return u

    def accel(self, u, v):
        row = self.rows[max(-15, min(15, int(u)))]
        v = max(self.v_nodes[0], min(v, self.v_nodes[-1]))
        i = min(bisect.bisect_right(self.v_nodes, v) - 1, len(self.v_nodes) - 2)
        t = (v - self.v_nodes[i]) / (self.v_nodes[i + 1] - self.v_nodes[i])
        return row[i] + t * (row[i + 1] - row[i])
