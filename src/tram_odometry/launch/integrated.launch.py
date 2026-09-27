"""Competition profile. Play bags with --clock and use_sim_time:=true."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('tram_odometry')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('gnss_corrections', default_value='false'),
        Node(package='tram_odometry', executable='integrated_odometry_node',
             name='tram_odometry', output='screen',
             parameters=[os.path.join(share, 'config', 'tram_odometry.yaml'), {
                 'gnss_corrections': ParameterValue(LaunchConfiguration('gnss_corrections'), value_type=bool),
                 'primary_sync': False,
                 'output_v_delay_s': 0.0,
                 'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
             }]),
    ])
