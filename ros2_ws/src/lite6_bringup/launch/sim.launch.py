import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    model_path = LaunchConfiguration(
        'model_path',
        default='/workspace/sim/models/mjcf/ufactory_lite6/scene.xml'
    )

    # Robot description (URDF -> robot_state_publisher)
    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('lite6_description'),
                'launch', 'description.launch.py'
            )
        )
    )

    # MuJoCo sim node (headless physics + /joint_states + /position_controller)
    mujoco_sim = Node(
        package='lite6_control',
        executable='mujoco_sim',
        name='mujoco_sim_node',
        parameters=[{
            'model_path': model_path,
        }],
        output='screen',
    )

    # Rosbridge for browser <-> ROS 2 communication
    rosbridge = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        parameters=[{'port': 9090}],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('model_path',
                              default_value='/workspace/sim/models/mjcf/ufactory_lite6/scene.xml',
                              description='Path to MuJoCo XML scene'),
        description_launch,
        mujoco_sim,
        rosbridge,
    ])
