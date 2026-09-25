#!/bin/bash
# Симуляция проверки жюри внутри контейнера: нода + ros2 bag play + запись входов и выходов.
#   sim.sh <каталог bag> <каталог результата> [скорость воспроизведения]
# В результате: result/ (rosbag с /vehicle/* и /result/*), node.log, resources.log (RSS КБ, CPU %).
set -u
BAG=$1
OUT=$2
RATE=${3:-1.0}
mkdir -p "$OUT"
rm -rf "$OUT/result"
export ROS_LOCALHOST_ONLY=1

ros2 launch tram_odometry tram_odometry.launch.py > "$OUT/node.log" 2>&1 &
LAUNCH=$!
for _ in $(seq 1 50); do
  PID=$(pgrep -f tram_odometry_node | head -1) && [ -n "$PID" ] && break
  sleep 0.2
done
sleep 2
( while kill -0 "$PID" 2>/dev/null; do ps -o rss=,pcpu= -p "$PID"; sleep 1; done ) > "$OUT/resources.log" &
ros2 bag record -o "$OUT/result" \
  /vehicle/front_bogie_velocity /vehicle/rear_bogie_velocity /vehicle/driver_position_cmd \
  /result/velocity /result/position /result/diagnostics > "$OUT/record.log" 2>&1 &
REC=$!
sleep 3
ros2 bag play "$BAG" --rate "$RATE" > "$OUT/play.log" 2>&1
sleep 2
kill -INT "$REC"; wait "$REC" 2>/dev/null
kill -INT "$LAUNCH"; wait "$LAUNCH" 2>/dev/null
echo "готово: $OUT"
