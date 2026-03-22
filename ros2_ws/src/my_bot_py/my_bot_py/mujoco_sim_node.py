import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import mujoco
import numpy as np
import os

class MuJoCoSimNode(Node):
    """ROS2 node that runs actual MuJoCo physics simulation."""

    def __init__(self):
        super().__init__("mujoco_sim_node")

        # Declare parameters
        self.declare_parameter("model_path", "/workspace/sim/models/mjcf/robot_arm.xml")
        self.declare_parameter("sim_rate", 50.0)  # Hz

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        sim_rate = self.get_parameter("sim_rate").get_parameter_value().double_value

        # Load MuJoCo model
        self.model = None
        self.data = None
        self.joint_names = []

        if os.path.exists(model_path):
            try:
                self.model = mujoco.MjModel.from_xml_path(model_path)
                self.data = mujoco.MjData(self.model)
                self.get_logger().info(f"Loaded MuJoCo model: {model_path}")
                self.get_logger().info(f"  DOFs: {self.model.nq}, Actuators: {self.model.nu}")

                # Extract joint names from model
                for i in range(self.model.njnt):
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    if name:
                        self.joint_names.append(name)
                self.get_logger().info(f"  Joints: {self.joint_names}")

            except Exception as e:
                self.get_logger().error(f"Failed to load model: {e}")
        else:
            self.get_logger().warn(f"Model not found: {model_path}, using demo mode")
            self.joint_names = ["joint_base", "joint_shoulder", "joint_elbow", "joint_wrist"]

        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, "joint_states", 10)

        # Subscribers for actuator commands
        self.cmd_sub = self.create_subscription(
            Float64MultiArray,
            "position_controller/commands",
            self.cmd_callback,
            10
        )

        # Control targets (position control)
        self.target_positions = np.zeros(len(self.joint_names))

        # Simulation timer
        self.sim_timer = self.create_timer(1.0 / sim_rate, self.sim_step)

        self.get_logger().info(f"MuJoCo simulation running at {sim_rate} Hz")
        self.get_logger().info("Subscribing to /position_controller/commands for control input")

    def cmd_callback(self, msg):
        """Handle incoming position commands."""
        if len(msg.data) > 0:
            for i, val in enumerate(msg.data):
                if i < len(self.target_positions):
                    self.target_positions[i] = val

    def sim_step(self):
        """Run one simulation step and publish joint states."""
        if self.model is not None and self.data is not None:
            # Apply position control via actuators
            # Simple P control: ctrl = Kp * (target - current)
            Kp = 10.0
            for i in range(min(self.model.nu, len(self.target_positions))):
                if i < self.model.nq:
                    error = self.target_positions[i] - self.data.qpos[i]
                    self.data.ctrl[i] = np.clip(Kp * error, -1.0, 1.0)

            # Step simulation
            mujoco.mj_step(self.model, self.data)

            # Publish joint states
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names[:self.model.nq]
            msg.position = list(self.data.qpos[:len(self.joint_names)])
            msg.velocity = list(self.data.qvel[:len(self.joint_names)]) if self.model.nv >= len(self.joint_names) else [0.0] * len(self.joint_names)
            msg.effort = [0.0] * len(self.joint_names)
            self.joint_state_pub.publish(msg)
        else:
            # Demo mode - publish zeros
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            msg.position = list(self.target_positions)
            msg.velocity = [0.0] * len(self.joint_names)
            msg.effort = [0.0] * len(self.joint_names)
            self.joint_state_pub.publish(msg)

def main():
    rclpy.init()
    node = MuJoCoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
