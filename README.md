# Резервная одометрия трамвая по модели

Решение команды для трека «Резервная одометрия по модели» [Хакатона Московского транспорта](https://mt-hackathon.ru/#task), 25 сентября — 3 октября 2026.

## Для жюри: как проверить решение

Сдаваемая версия — тег [`v1.0-submission`](https://github.com/MikeEgorshev/FunnyRepo/tree/v1.0-submission). Модель, параметры, точность и ограничения описаны в [docs/solution.md](docs/solution.md): разделы совпадают с полями формы сдачи. Независимая валидация и сравнение вариантов решения — в [docs/validation.md](docs/validation.md).

### Сборка: ROS 2 Humble, без интернета

```bash
git clone https://github.com/MikeEgorshev/FunnyRepo.git && cd FunnyRepo
git checkout v1.0-submission
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

Или в Docker: `docker build -f docker/Dockerfile -t tram-odometry .`

В рабочем пространстве должен быть один пакет `tram_vehicle_msgs` — наш `src/tram_vehicle_msgs`, из основного датасета, с `DriverControllerCommand`. Если собираете вместе с `check-code` организаторов, исключите их копию: `touch <check-code>/src/tram_vehicle_msgs/COLCON_IGNORE`.

### Запуск на rosbag

```bash
ros2 launch tram_odometry tram_odometry.launch.py    # терминал 1
ros2 bag play <каталог прогона>                       # терминал 2
```

Перед каждым прогоном ноду лучше запускать заново. При повторном проигрывании без перезапуска она сбрасывает состояние сама. Номер вагона можно задать: `ros2 launch tram_odometry tram_odometry.launch.py vehicle_id:=30618`.

| Выход | Тип | Содержимое |
|---|---|---|
| `/result/velocity` | `tram_vehicle_msgs/VelocitySensor` | продольная скорость, м/с |
| `/result/position` | `nav_msgs/Odometry` | положение `base_link` в сетке MGRS, как у `/localization/kinematic_state`; `frame_id: map`. Без GNSS на старте — `odom`, путь от старта |
| `/result/diagnostics` | `diagnostic_msgs/DiagnosticArray` | проскальзывание, пропуски и зависания датчиков, время обработки |

Лог ноды — вывод `ros2 launch`.

### Судья организаторов

Собрать в одном рабочем пространстве наш `src/` и `hackathon_solution_checker` из `check-code`, затем в трёх терминалах:

```bash
ros2 launch tram_odometry tram_odometry.launch.py
ros2 run hackathon_solution_checker metrics
ros2 bag play <check-code>/bags/30618_88aea4d9
```

Или одной командой в Docker с лимитами жюри — итог в `out/metrics.log`:

```bash
docker run --rm --cpus=2 --memory=512m -v <check-code>:/check:ro -v $PWD/out:/out tram-odometry judge_check.sh
```

На этом прогоне: скорость RMSE 0,032 м/с, положение 3D RMSE 4,9–5,2 м (2,0 м на участке pathgraph), ~29 Гц, задержка p99 < 10 мс, CPU ~15 % ядра, 61 МБ.

Режимы `integrated.launch.py` и `hybrid.launch.py` экспериментальные и в сдачу не входят: [docs/integrated-odometry.md](docs/integrated-odometry.md), [docs/hybrid-odometry.md](docs/hybrid-odometry.md).

## Задача

Беспилотный трамвай должен знать свою скорость и положение, даже когда пропал сигнал GNSS в «городских каньонах» и отказали другие навигационные системы. Доступны только внутренние данные: положение ручки контроллера водителя и одометрия (скорость вращения колёс).

Нужно разработать адаптивную нелинейную модель движения (аппроксиматор) и пакет ROS 2, которые:

- оценивают скорость трамвая по сигналу управления;
- вычисляют скорость и положение без GNSS-приёмников и инерциальных систем;
- компенсируют проскальзывание колёс в плохую погоду, отказы датчиков одометрии и меняющиеся параметры трамвая;
- работают в реальном времени: ноды ROS 2 Humble на C++ или Python публикуют координаты и оценку скорости в топики.

Организаторы уточнили, что ждут в первую очередь математическую модель. Данные, формат сдачи ROS 2-пакета, имена топиков и метрики будут в полном описании задачи на платформе хакатона. Сводка всего, что известно, — в [docs/context/](docs/context/).

## Сроки

| Когда | Что |
|---|---|
| 25.09, 17:00 МСК | QA-сессия трека с экспертами |
| **27.09, 23:59 МСК** | **Дедлайн загрузки решения** |
| 29.09 | Список полуфиналистов |
| 30.09 | Полуфинал: закрытые онлайн-питчи |
| 01.10 | Список финалистов |
| 03.10 | Финал, офлайн в Москве |

## Как мы работаем

В команде люди и несколько ИИ-агентов. Правила работы с git — в [AGENTS.md](AGENTS.md), их читают до первого коммита.
