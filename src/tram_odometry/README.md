# tram_odometry

Резервная одометрия трамвая без GNSS: модель тяги и торможения, фильтр Калмана вдоль пути, обнаружение проскальзывания и зависаний датчиков, привязка к стоянкам. Пакет ROS 2 Humble (ament_python). Полное описание — [docs/solution.md](../../docs/solution.md).

## Состав

| Модуль | Что делает | ROS |
|---|---|---|
| `model.py` | Ускорение на единицу массы: `a = f_tr(n, v) − f_br(n, v) − r(v) − g·θ` | нет |
| `slip.py` | Выбор тележки по фазе, согласованность тележек, предел ускорения колёс, зависание, устаревание | нет |
| `estimator.py` | EKF на `[s, v, d, k]`: прогноз по модели, отсев χ², стоянка, привязки дистанции, масштаб колёс k по расстояниям между привязками, сброс на новом прогоне | нет |
| `stops.py` | Привязка к местам регулярных стоянок: стоянка дольше 5 с, одна стоянка в гейте — поправка s | нет |
| `track.py` | Сетка MGRS судьи и ENU, карта линии из CSV, запасная прямая, выставка по GNSS (медиана, прыжки, курс по базе антенн) | нет |
| `positioning.py` | Выставка → привязка к карте → `base_link`; режим без GNSS (frame `odom`) | нет |
| `outputs.py`, `params.py` | Поля `nav_msgs/Odometry`, метки выхода; параметры ноды | нет |
| `node.py` | Подписки, таймер 25 Гц, публикации, диагностика | да |

Вся математика — без ROS и без внешних зависимостей: тот же код гоняется офлайн по bag (`scripts/evaluate_bags.py`).

## Сборка и запуск

Нужен пакет сообщений `tram_vehicle_msgs` из датасета в том же рабочем пространстве (кладёт лид: пакеты `*_msgs` защищены, AGENTS.md §6).

```bash
colcon build --packages-up-to tram_odometry
source install/setup.bash
ros2 launch tram_odometry tram_odometry.launch.py params_file:=<YAML с route_map_file и stops_file>
ros2 bag play <прогон>          # в другом терминале
```

Выходы: `/result/velocity` (`tram_vehicle_msgs/VelocitySensor`, м/с), `/result/position` (`nav_msgs/Odometry`, `base_link` в сетке MGRS, фрейм `map`; без GNSS на старте — `odom`), `/result/diagnostics`.

## Проверка

```bash
python3 -m pytest src/tram_odometry/test -q                               # 50 тестов, без ROS
ROS_IMAGE=mirror.gcr.io/library/ros:humble-ros-base TRAM_MSGS=<пакет сообщений> \
  REALTIME_ARGS="--stall-at 30 --stall-s 1.5" scripts/check_ros.sh        # Humble: сборка без сети, тесты, реальное время
python3 scripts/evaluate_bags.py <прогоны…> --route-map … --stops …       # точность по bag (pip install rosbags)
```

В Humble с лимитами 2 ядра и 512 МБ: 25 Гц, разрыв выхода не больше 79 мс даже при пропуске всех входов на 1,5 с, задержка p99 45 мс, CPU 6 % ядра, память 60 МБ.

## Что ещё не сделано

- Параметры модели в `config/tram_odometry.yaml` — стартовые оценки: их нужно идентифицировать по данным.
- Точность на реальных прогонах не измерена: датасета в среде разработки не было.
- Карта и места стоянок в пакет не входят; отводов у конечных нет.
- Сборка проверена с заменителем `tram_vehicle_msgs` с теми же полями.
