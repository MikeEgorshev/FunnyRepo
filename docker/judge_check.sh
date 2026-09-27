#!/bin/bash
# Проверка судьёй организаторов: их hackathon_solution_checker (metrics.py из check-code-with-bag.zip,
# чат капитанов трека, 27.09) + наша нода + их проверочный bag, в реальном времени. Архив в репозиторий
# не кладём: распакуйте его и подключите томом.
#   docker run --rm --cpus=2 --memory=512m -v <check-code>:/check:ro -v <результат>:/out \
#     tram-odometry judge_check.sh [скорость воспроизведения] [vehicle_id]
# Итог — последние строки /out/metrics.log (RMSE и максимум: скорость; x, y, z, 3D), ресурсы ноды —
# /out/resources.log (время, RSS КБ, CPU %). Без set -u: setup.bash ROS его не выдерживает.
set -o pipefail
RATE=${1:-1.0}
VEHICLE=${2:-30618}
BAG=$(ls -d /check/bags/*/ 2>/dev/null | head -1)
[ -n "$BAG" ] && [ -e "$BAG/metadata.yaml" ] || { echo "нет bag в /check/bags" >&2; exit 2; }
mkdir -p /out
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-11} ROS_LOCALHOST_ONLY=1

# чекер собираем поверх нашего tram_vehicle_msgs: в копии организаторов нет DriverControllerCommand
rm -rf /tmp/chk && mkdir -p /tmp/chk/src && cp -r /check/src/checker_ros /tmp/chk/src/
(cd /tmp/chk && colcon build --packages-select hackathon_solution_checker --event-handlers console_direct-) \
    > /out/checker_build.log 2>&1 || { echo "сборка чекера упала" >&2; tail -20 /out/checker_build.log >&2; exit 1; }
source /tmp/chk/install/setup.bash

ros2 launch tram_odometry tram_odometry.launch.py vehicle_id:="$VEHICLE" > /out/node.log 2>&1 &
NODE=$!
ros2 run hackathon_solution_checker metrics --ros-args -p report_period_sec:=30.0 > /out/metrics.log 2>&1 &
METRICS=$!
( while kill -0 $NODE 2>/dev/null; do
    P=$(pgrep -f 'lib/tram_odometry/tram_odometry_node' | head -1)
    [ -n "$P" ] && ps -o rss=,pcpu= -p "$P" | awk -v t="$(date +%s)" '{print t, $1, $2}'
    sleep 5
  done ) > /out/resources.log &
SAMPLER=$!
sleep 5
ros2 bag play "$BAG" --rate "$RATE" > /out/play.log 2>&1
sleep 3
kill -INT $METRICS; sleep 3
kill -INT $NODE; sleep 5
kill -9 $NODE $METRICS $SAMPLER 2>/dev/null
pkill -f tram_odometry_node 2>/dev/null
echo "=== итог metrics.py организаторов ($(basename "$BAG"), rate $RATE)"
grep -E "Velocity metrics|Position metrics" /out/metrics.log | tail -2
awk '{ if ($2 > r) r = $2; if ($3 > c) c = $3 } END { print "нода: RSS до " r " КБ, CPU до " c " %" }' /out/resources.log
