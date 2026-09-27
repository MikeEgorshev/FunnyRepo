"""Запуск: ros2 launch tram_odometry tram_odometry.launch.py [use_sim_time:=true] [params_file:=...]"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('tram_odometry'), 'config', 'tram_odometry.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true — время из /clock (ros2 bag play --clock)'),
        Node(
            package='tram_odometry',
            executable='tram_odometry_node',
            name='tram_odometry',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)},
            ],
        ),
    ])
