import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params = os.path.join(get_package_share_directory('tram_odometry'), 'config', 'tram_odometry.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('vehicle_id', default_value='',
                              description='номер трамвая (30618/30639) для коэффициента колёс; пусто — общий'),
        Node(
            package='tram_odometry',
            executable='tram_odometry_node',
            name='tram_odometry',
            output='screen',
            # строкой явно: иначе vehicle_id:=30618 приходит целым числом и нода падает на типе параметра
            parameters=[params, {'vehicle_id': ParameterValue(LaunchConfiguration('vehicle_id'), value_type=str)}],
        ),
    ])
