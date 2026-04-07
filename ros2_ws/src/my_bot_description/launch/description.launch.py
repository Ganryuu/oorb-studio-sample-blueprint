import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('my_bot_description')
    xacro_file = os.path.join(pkg_dir, 'urdf', 'robot_arm.urdf.xacro')

    robot_description = Command(['xacro ', xacro_file])

    return LaunchDescription([
    Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(robot_description, value_type=str)
        }],
        output='screen',
        )
    ])
