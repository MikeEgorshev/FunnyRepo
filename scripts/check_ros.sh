#!/usr/bin/env bash
# Сборка без сети, тесты и замер реального времени в чистом ROS 2 Humble (Docker)
# с лимитами жюри: 2 ядра, 512 МБ.
#
# Запуск из корня репозитория:
#   TRAM_MSGS=../dataset/tram_vehicle_msgs scripts/check_ros.sh
# TRAM_MSGS — пакет сообщений из датасета; не нужен, если он уже лежит в src/.
# ROS_IMAGE — образ; при лимите Docker Hub: mirror.gcr.io/library/ros:humble-ros-base
# DURATION  — длительность замера реального времени, с.
set -euo pipefail

IMAGE=${ROS_IMAGE:-ros:humble-ros-base}
WS=$(mktemp -d)
trap 'rm -rf "$WS"' EXIT

mkdir -p "$WS/src"
cp -r src/. "$WS/src/"
cp -r scripts "$WS/"
if [ ! -d "$WS/src/tram_vehicle_msgs" ]; then
  if [ -z "${TRAM_MSGS:-}" ]; then
    echo "Нет src/tram_vehicle_msgs: задайте TRAM_MSGS=<путь к пакету из датасета>" >&2
    exit 1
  fi
  cp -r "$TRAM_MSGS" "$WS/src/tram_vehicle_msgs"
fi
find "$WS" -name __pycache__ -prune -exec rm -rf {} +

docker run --rm --network none --cpus=2 --memory=512m -e DURATION="${DURATION:-60}" \
  -v "$WS":/ws -w /ws "$IMAGE" bash -c '
  set -e
  source /opt/ros/humble/setup.bash
  colcon build --event-handlers console_direct-
  colcon test --event-handlers console_direct-
  colcon test-result --verbose
  source install/setup.bash
  export ROS_DOMAIN_ID=11 ROS_LOCALHOST_ONLY=1
  ros2 launch tram_odometry tram_odometry.launch.py > /tmp/node.log 2>&1 &
  sleep 4
  python3 scripts/realtime_check.py --duration "$DURATION" --pid "$(pgrep -x tram_odometry_n | head -1)"'
