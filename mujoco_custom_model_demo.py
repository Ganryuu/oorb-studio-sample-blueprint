#!/usr/bin/env python3
# Custom MuJoCo Model Demo (ROS2)
# Loads a user-provided MJCF and publishes joint states.
# Optionally accepts control commands.
#
# Quick start (open three terminals):
# Terminal 1 (run the demo):
#   source /opt/ros/jazzy/setup.bash
#   MODEL_PATH=/workspace/sim/models/mjcf/your_model.xml python3 mujoco_custom_model_demo.py
#
# Terminal 2 (verify it is publishing):
#   source /opt/ros/jazzy/setup.bash
#   ros2 topic hz /joint_states
#
# Terminal 3 (send commands, choose one):
#   source /opt/ros/jazzy/setup.bash
#   ros2 topic pub -r 10 /mujoco_ctrl std_msgs/msg/Float64MultiArray "{data: [0.5, -0.5]}"
#   # or
#   ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: -0.5}}'
#
# Notes:
# - If your model has no actuators (nu=0), commands will be ignored.
# - /cmd_vel maps linear.x -> actuator[0], angular.z -> actuator[1] (if present).

import os
import time
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/sim/models/mjcf/reacher.xml")
STEP_HZ = float(os.getenv("STEP_HZ", "60"))
CTRL_TOPIC = "/mujoco_ctrl"
CMD_VEL_TOPIC = "/cmd_vel"


class MujocoCustomDemo(Node):
    def __init__(self):
        super().__init__("mujoco_custom_demo")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.ctrl = np.zeros(self.model.nu, dtype=float)

        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        if not self.joint_names:
            self.joint_names = [f"joint_{i}" for i in range(self.model.njnt)]

        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.qpos_pub = self.create_publisher(Float64MultiArray, "/mujoco_qpos", 10)

        self.create_subscription(Float64MultiArray, CTRL_TOPIC, self.on_ctrl, 10)
        self.create_subscription(Twist, CMD_VEL_TOPIC, self.on_cmd_vel, 10)

        self.get_logger().info(
            f"Loaded {MODEL_PATH} nq={self.model.nq} nu={self.model.nu} njnt={self.model.njnt}"
        )
        if self.model.nu == 0:
            self.get_logger().warn("Model has no actuators (nu=0). Commands will be ignored.")

        self.timer = self.create_timer(1.0 / STEP_HZ, self.step)

    def on_ctrl(self, msg: Float64MultiArray):
        if self.model.nu == 0:
            return
        arr = np.array(msg.data, dtype=float)
        n = min(len(arr), self.model.nu)
        if n:
            self.ctrl[:n] = arr[:n]

    def on_cmd_vel(self, msg: Twist):
        if self.model.nu == 0:
            return
        if self.model.nu >= 1:
            self.ctrl[0] = float(msg.linear.x)
        if self.model.nu >= 2:
            self.ctrl[1] = float(msg.angular.z)

    def step(self):
        if self.model.nu:
            self.data.ctrl[:] = self.ctrl
        mujoco.mj_step(self.model, self.data)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        js.position = self.data.qpos.tolist()
        js.velocity = self.data.qvel.tolist() if self.data.qvel is not None else []
        self.pub.publish(js)

        qpos = Float64MultiArray()
        qpos.data = self.data.qpos.tolist()
        self.qpos_pub.publish(qpos)


def main():
    rclpy.init()
    node = MujocoCustomDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
