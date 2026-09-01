"""Launch a VLA policy node.

Local inference on the first GPU:
    ros2 launch vla_ros policy.launch.py model:=openvla instruction:="pick up the red block"

Both 3090s, one replica each:
    ros2 launch vla_ros policy.launch.py model:=openvla devices:=cuda:0,cuda:1

Remote inference against a `vla serve` workstation:
    ros2 launch vla_ros policy.launch.py model:=pi0 remote_url:=ws://workstation:8000
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGUMENTS = [
    ("model", "echo", "registry name: openvla, pi0, smolvla, echo"),
    ("checkpoint", "", "override the default weights"),
    ("device", "auto", "cuda:0, cpu, or auto"),
    ("devices", "", "comma-separated replica devices, e.g. cuda:0,cuda:1"),
    ("dtype", "auto", "auto, bf16, fp16, fp32"),
    ("quantization", "", "nf4 or int8"),
    ("unnorm_key", "", "OpenVLA dataset statistics key"),
    ("instruction", "", "initial task instruction"),
    ("control_hz", "20.0", "action publish rate"),
    ("image_topic", "/vla/image", "camera topic"),
    ("remote_url", "", "ws:// or http:// URL of a vla serve instance"),
    ("strategy", "temporal_ensemble", "temporal_ensemble or sequential"),
    ("ensemble_decay", "0.1", "chunk blending decay"),
]


def generate_launch_description() -> LaunchDescription:
    declarations = [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in ARGUMENTS
    ]
    parameters = {name: LaunchConfiguration(name) for name, _, _ in ARGUMENTS}
    return LaunchDescription(
        declarations
        + [
            Node(
                package="vla_ros",
                executable="policy",
                name="vla_policy",
                output="screen",
                parameters=[parameters],
            )
        ]
    )
