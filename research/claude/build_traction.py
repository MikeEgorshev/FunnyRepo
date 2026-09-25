"""Таблица модели тяги и торможения a(u, v) по данным: средняя производная скорости колёс
в ячейках «позиция контроллера × скорость» с задержкой отклика TAU.

Пустые и редкие ячейки (меньше MIN_N выборок) заполняются по соседним скоростям той же позиции,
затем лёгкое сглаживание по позиции. При v = 0 торможение не может разгонять назад —
это учитывает оценщик, а не таблица.

Запуск: python build_traction.py (после traction_explore.py)
        -> src/tram_odometry/config/traction_table.csv (строки u = -15..15, столбцы — узлы скорости)
"""
import numpy as np

from bagio import REPO

OUT = REPO / 'src' / 'tram_odometry' / 'config' / 'traction_table.csv'
SAMPLES = REPO / 'research' / 'claude' / 'out' / 'traction_samples.npz'
TAU = 0.4
V_NODES = np.arange(0.5, 16.0, 1.0)  # центры ячеек скорости, м/с
MIN_N = 40


def main():
    z = np.load(SAMPLES)
    i = int(np.argmin(np.abs(z['taus'] - TAU)))
    v, a, u = z['v'], z['a'], z['u'][i].astype(int)
    sel = v > 0.3
    v, a, u = v[sel], a[sel], u[sel]
    vb = np.clip(np.round(v - 0.5).astype(int), 0, len(V_NODES) - 1)
    key = (u + 15) * len(V_NODES) + vb
    n = np.bincount(key, None, 31 * len(V_NODES)).reshape(31, -1)
    mean = (np.bincount(key, a, 31 * len(V_NODES)).reshape(31, -1) / np.maximum(n, 1))
    table = np.where(n >= MIN_N, mean, np.nan)
    for r in range(31):  # заполнение по скорости внутри строки
        row, good = table[r], ~np.isnan(table[r])
        if good.sum() == 0:
            continue
        table[r] = np.interp(np.arange(len(row)), np.flatnonzero(good), row[good])
    for c in range(table.shape[1]):  # строки без данных — по соседним позициям
        col, good = table[:, c], ~np.isnan(table[:, c])
        table[:, c] = np.interp(np.arange(31), np.flatnonzero(good), col[good])
    smooth = table.copy()
    for r in range(1, 30):
        smooth[r] = 0.25 * table[r - 1] + 0.5 * table[r] + 0.25 * table[r + 1]
    smooth[15] = table[15]  # нейтраль не сглаживаем с тягой и тормозом
    # физика: тормозные позиции не разгоняют, и более глубокая позиция тормозит не слабее;
    # тяговые позиции — наоборот (редкие экстренные торможения дают в данных шум)
    smooth[:15] = np.minimum(smooth[:15], -0.05)
    for r in range(13, -1, -1):
        smooth[r] = np.minimum(smooth[r], smooth[r + 1])
    for r in range(17, 31):
        smooth[r] = np.maximum(smooth[r], smooth[r - 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'# Модель тяги и торможения: ускорение, м/с², по позиции контроллера u (строки -15..15)\n')
        f.write(f'# и скорости (столбцы, м/с). Отклик на команду с задержкой {TAU} с. research/claude/build_traction.py\n')
        f.write('u,' + ','.join(f'{x:.1f}' for x in V_NODES) + '\n')
        for r in range(31):
            f.write(f'{r - 15},' + ','.join(f'{x:.4f}' for x in smooth[r]) + '\n')
    resid = a - smooth[u + 15, vb]
    print(f'таблица {smooth.shape}, R² на выборке {1 - resid.var() / a.var():.3f}, '
          f'СКО остатка {resid.std():.3f} м/с² -> {OUT}')


if __name__ == '__main__':
    main()
