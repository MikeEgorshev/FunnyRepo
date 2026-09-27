# ROS 2 Humble + пакеты решения. Сборка без интернета: внешних зависимостей нет.
#   docker build -f docker/Dockerfile -t tram-odometry .
FROM ros:humble-ros-base

SHELL ["/bin/bash", "-c"]
WORKDIR /ws
COPY src/ src/
RUN source /opt/ros/humble/setup.bash \
    && colcon build --event-handlers console_direct- \
    && rm -rf build log
COPY docker/entrypoint.sh docker/sim.sh docker/judge_check.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/sim.sh /usr/local/bin/judge_check.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["ros2", "launch", "tram_odometry", "tram_odometry.launch.py"]
