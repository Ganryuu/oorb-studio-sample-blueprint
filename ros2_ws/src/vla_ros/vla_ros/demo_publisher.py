"""Publish synthetic camera frames and an instruction, to exercise the policy node.

Lets you verify the full pipeline -- images in, actions out at the control rate
-- inside OORB Studio without a camera or a robot. Run this alongside
``ros2 run vla_ros policy`` and watch ``/vla/action``.

    ros2 run vla_ros demo_publisher --ros-args -p instruction:="pick up the red block"
    ros2 topic hz /vla/action
    ros2 topic echo /vla/status
"""

from __future__ import annotations

import math

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import String
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "vla_ros requires ROS 2 (source /opt/ros/jazzy/setup.bash before running)"
    ) from exc

from vla_engine.ros.demo_scene import render_scene


class DemoPublisher(Node):
    """Publishes a moving synthetic scene, an instruction, and joint states."""

    def __init__(self) -> None:
        super().__init__("vla_demo_publisher")
        self.declare_parameter("instruction", "pick up the red block")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("width", 224)
        self.declare_parameter("height", 224)
        self.declare_parameter("num_joints", 7)

        self.image_pub = self.create_publisher(Image, "/vla/image", 1)
        self.instruction_pub = self.create_publisher(String, "/vla/instruction", 1)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 1)

        self._width = int(self.get_parameter("width").value)
        self._height = int(self.get_parameter("height").value)
        self._num_joints = int(self.get_parameter("num_joints").value)
        self._frame = 0

        rate = float(self.get_parameter("rate_hz").value)
        self.create_timer(1.0 / max(rate, 1e-3), self._tick)
        self.create_timer(1.0, self._publish_instruction)
        self._publish_instruction()
        self.get_logger().info(
            f"publishing {self._width}x{self._height} frames at {rate:.0f} Hz on /vla/image"
        )

    def _publish_instruction(self) -> None:
        message = String()
        message.data = str(self.get_parameter("instruction").value)
        self.instruction_pub.publish(message)

    def _tick(self) -> None:
        self._frame += 1
        frame = render_scene(self._height, self._width, self._frame)

        image = Image()
        image.header.stamp = self.get_clock().now().to_msg()
        image.header.frame_id = "camera"
        image.height, image.width = self._height, self._width
        image.encoding = "rgb8"
        image.is_bigendian = 0
        image.step = self._width * 3
        image.data = frame.tobytes()
        self.image_pub.publish(image)

        joints = JointState()
        joints.header.stamp = image.header.stamp
        joints.name = [f"joint_{i}" for i in range(self._num_joints)]
        joints.position = [0.4 * math.sin(0.02 * self._frame + i) for i in range(self._num_joints)]
        self.joint_pub.publish(joints)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DemoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
