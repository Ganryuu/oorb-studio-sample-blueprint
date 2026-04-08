"""
Demo joint publisher for UFactory Lite 6.

Publishes smooth sinusoidal joint motions on /joint_states at 20 Hz.
Useful for verifying the URDF viewer and MuJoCo sim are receiving data.
"""
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


class JointPublisher(Node):
    def __init__(self):
        super().__init__('joint_publisher')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.05, self.publish)  # 20 Hz
        self.t = 0.0
        self.get_logger().info('Lite 6 joint publisher started (20 Hz demo motion)')

    def publish(self):
        self.t += 0.05
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES

        # Smooth sinusoidal demo motion within safe ranges
        msg.position = [
            0.8 * math.sin(0.5 * self.t),           # joint1: base rotation
            0.4 * math.sin(0.3 * self.t + 0.5),     # joint2: shoulder
            1.2 + 0.5 * math.sin(0.4 * self.t),     # joint3: elbow (biased to stay in range)
            0.6 * math.sin(0.6 * self.t + 1.0),     # joint4: wrist 1
            0.5 * math.sin(0.35 * self.t + 1.5),    # joint5: wrist 2
            1.0 * math.sin(0.7 * self.t + 2.0),     # joint6: wrist 3
        ]
        msg.velocity = [0.0] * 6
        msg.effort = [0.0] * 6
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
