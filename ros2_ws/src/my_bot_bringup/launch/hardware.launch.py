"""Hardware launch file — template for physical robot deployment.

Uncomment and configure when connecting to real hardware.
This launch file mirrors sim.launch.py but swaps the MuJoCo sim node
for real hardware drivers.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Robot description (same URDF for sim and hardware)
    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('my_bot_description'),
                'launch', 'description.launch.py'
            )
        )
    )

    # Rosbridge for browser ↔ ROS 2 communication
    rosbridge = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        parameters=[{'port': 9090}],
        output='screen',
    )

    # TODO: Add hardware driver nodes here
    # Examples:
    #   - Serial driver for Arduino/ESP32 servo controller
    #   - micro-ROS agent for embedded firmware
    #   - ros2_control hardware interface

    return LaunchDescription([
        description_launch,
        rosbridge,
        # Add hardware driver nodes above
    ])
