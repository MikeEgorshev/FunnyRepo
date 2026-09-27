# tram_odometry

Резервная одометрия трамвая без GNSS: модель тяги и торможения, фильтр Калмана вдоль пути, обнаружение проскальзывания и зависаний датчиков, привязка к стоянкам. Пакет ROS 2 Humble (ament_python). Полное описание — [docs/solution.md](../../docs/solution.md).

## Состав

| Модуль | Что делает | ROS |
|---|---|---|
| `model.py` | Ускорение на единицу массы: `a = f_tr(u, v) − f_br(u, v) − r(v) − g·θ`, запаздывание привода `du/dt = (n − u)/τ` | нет |
| `slip.py` | Отсчёты тележек к одному моменту, среднее или выбор по фазе, согласованность, предел ускорения колёс, зависание, устаревание | нет |
| `estimator.py` | EKF на `[s, v, d, k]`: прогноз по модели, отсев χ², стоянка, привязки дистанции, масштаб колёс k по расстояниям между привязками, сброс на новом прогоне | нет |
| `stops.py` | Привязка к местам регулярных стоянок: стоянка дольше 5 с, одна стоянка в гейте — поправка s | нет |
| `track.py` | Сетка MGRS судьи и ENU, карта линии из CSV, запасная прямая, выставка по GNSS (медиана, прыжки, курс по базе антенн) | нет |
| `positioning.py` | Выставка → привязка к карте → `base_link`; режим без GNSS (frame `odom`) | нет |
| `outputs.py`, `params.py` | Поля `nav_msgs/Odometry`, метки выхода; параметры ноды | нет |
| `node.py` | Подписки, таймер 25 Гц, публикации, диагностика | да |
| `mapping.py` | Офлайн: карта линии и места стоянок по GNSS обучающих прогонов (`scripts/build_route_map.py`) | нет |
| `identify.py` | Офлайн: подбор параметров модели по прогонам (`scripts/fit_model.py`) | нет |

Вся математика — без ROS и без внешних зависимостей: тот же код гоняется офлайн по bag (`scripts/evaluate_bags.py`, `scripts/pipeline.py`).

## Сборка и запуск

Нужен пакет сообщений `tram_vehicle_msgs` из датасета в том же рабочем пространстве (кладёт лид: пакеты `*_msgs` защищены, AGENTS.md §6).

```bash
colcon build --packages-up-to tram_odometry
source install/setup.bash
ros2 launch tram_odometry tram_odometry.launch.py params_file:=results/fitted.yaml   # YAML от scripts/pipeline.py
ros2 bag play <прогон>          # в другом терминале
```

Выходы: `/result/velocity` (`tram_vehicle_msgs/VelocitySensor`, м/с), `/result/position` (`nav_msgs/Odometry`, `base_link` в сетке MGRS, фрейм `map`; без GNSS на старте — `odom`), `/result/diagnostics`.

## Проверка

```bash
python3 -m pytest src/tram_odometry/test -q                               # 59 тестов, без ROS
ROS_IMAGE=mirror.gcr.io/library/ros:humble-ros-base TRAM_MSGS=<пакет сообщений> \
  REALTIME_ARGS="--stall-at 30 --stall-s 1.5" scripts/check_ros.sh        # Humble: сборка без сети, тесты, реальное время
python3 scripts/pipeline.py --dataset <dataset>/data --out-dir results/    # карта, модель, метрики (pip install rosbags pyyaml)
python3 scripts/make_report.py --results results/results.json              # презентация docs/presentation.html
```

В Humble с лимитами 2 ядра и 512 МБ: 25 Гц, разрыв выхода не больше 65 мс даже при пропуске всех входов на 1,5 с, задержка p99 48 мс, CPU 4,6 % ядра, память 59 МБ.

## Что ещё не сделано

- Точность на реальных прогонах не измерена и модель по ним не подобрана: датасета в среде разработки не было. Конвейер `scripts/pipeline.py` делает это одной командой; на синтетике того же формата — средняя 3D-ошибка 1,2 м, дрейф 0,05 %.
- Карта — один замкнутый круг; отводов у конечных нет.
- Сборка проверена с заменителем `tram_vehicle_msgs` с теми же полями.
