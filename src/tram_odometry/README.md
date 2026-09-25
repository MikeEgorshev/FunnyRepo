# tram_odometry

Резервная одометрия трамвая без GNSS: модель тяги и торможения, фильтр Калмана вдоль пути, обнаружение проскальзывания. Пакет ROS 2 Humble (ament_python).

## Состав

| Модуль | Что делает | ROS |
|---|---|---|
| `model.py` | Ускорение на единицу массы: `a = f_tr(n, v) − f_br(n, v) − r(v) − g·θ` | нет |
| `slip.py` | Выбор тележки по фазе, согласованность тележек, предел ускорения колёс, устаревание | нет |
| `estimator.py` | EKF на `[s, v, d, k]`: прогноз по модели, отсев χ², стоянка, привязки дистанции, сброс на новом прогоне | нет |
| `track.py` | ENU от первой точки GNSS, выставка курса, запасной путь-прямая | нет |
| `outputs.py` | Поля `nav_msgs/Odometry`, метки выхода | нет |
| `params.py` | Параметры ноды по умолчанию | нет |
| `node.py` | Подписки, таймер 25 Гц, публикации, диагностика | да |

Вся математика — без ROS и без внешних зависимостей: тот же код можно гонять офлайн по bag.

## Сборка и запуск

Нужен пакет сообщений `tram_vehicle_msgs` из датасета в том же рабочем пространстве (кладёт лид: пакеты `*_msgs` защищены, AGENTS.md §6).

```bash
colcon build --symlink-install --packages-up-to tram_odometry
source install/setup.bash
ros2 launch tram_odometry tram_odometry.launch.py use_sim_time:=true
ros2 bag play <прогон> --clock          # в другом терминале
```

Выходы: `/result/velocity` (`tram_vehicle_msgs/VelocitySensor`, м/с), `/result/position` (`nav_msgs/Odometry`, фрейм `map`), `/result/diagnostics` (флаг проскальзывания, доля сцепления, доверие тележкам, k).

## Тесты

```bash
python3 -m pytest src/tram_odometry/test -q          # без ROS
colcon test --packages-select tram_odometry && colcon test-result --verbose
```

Тесты на синтетических прогонах (`test/synthetic.py`): чистый ход, буксование, юз, пропуск обеих тележек на 20 с, обучение k по привязкам к остановкам. Это не данные трамвая: на реальных прогонах пакет ещё не проверен.

## Что ещё не сделано

- Параметры модели в `config/tram_odometry.yaml` — стартовые оценки. Их нужно идентифицировать по данным.
- Положение пока по запасному пути: прямая от старта по курсу GNSS. Карта линии (PR #4) подключается объектом с методом `pose(s) -> (x, y, z, yaw)`, привязки к остановкам — вызовом `Estimator.position_fix`.
- Тип и единицы `/result/velocity`, начало системы `map` и приёмник эталона — открытые вопросы к организаторам.
- Сборка в ROS 2 Humble и запуск на bag ещё не проверены.
