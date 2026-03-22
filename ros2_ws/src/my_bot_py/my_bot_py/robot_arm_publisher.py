import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class RobotArmPublisher(Node):
    def __init__(self):
        super().__init__("robot_arm_publisher")
        self.pub = self.create_publisher(JointState, "joint_states", 10)
        self.timer = self.create_timer(0.05, self.publish_joint_states)  # 20 Hz
        self.t = 0.0

        # Joint names matching MuJoCo robot_arm.xml model
        self.joint_names = ["joint_base", "joint_shoulder", "joint_elbow", "joint_wrist"]

        self.get_logger().info("Robot arm joint state publisher started (demo mode)")
        self.get_logger().info("Publishing to /joint_states for MuJoCo visualization")

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # Create smooth sinusoidal motion for demo (angles in radians)
        msg.position = [
            0.5 * math.sin(self.t * 0.5),           # joint_base: slow rotation
            0.6 * math.sin(self.t * 0.7 + 1.0),     # joint_shoulder: up/down wave
            -0.8 * math.sin(self.t * 0.9 + 2.0),    # joint_elbow: negative range wave
            0.4 * math.sin(self.t * 1.1 + 3.0)      # joint_wrist: wrist rotation
        ]
        msg.velocity = [0.0, 0.0, 0.0, 0.0]
        msg.effort = [0.0, 0.0, 0.0, 0.0]

        self.pub.publish(msg)
        self.t += 0.05

def main():
    rclpy.init()
    node = RobotArmPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
