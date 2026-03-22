#!/usr/bin/env python3
# Reacher RL Demo - Controllable Simulator
# Modes:
# - DEMO_MODE=random (default): ignores ROS commands and runs the policy/random actions
# - DEMO_MODE=ros: waits for ROS commands and uses them to drive the sim
#
# Quick start (open three terminals):
# Terminal 1 (run the demo):
#   source /opt/ros/jazzy/setup.bash
#   DEMO_MODE=ros python3 dummy_simulator_controlled.py
#
# Terminal 2 (verify it is publishing):
#   source /opt/ros/jazzy/setup.bash
#   ros2 topic hz /joint_states
#
# Terminal 3 (send commands, choose one):
#   source /opt/ros/jazzy/setup.bash
#   ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: -0.5}}'
#   # or
#   ros2 topic pub -r 10 /reacher_action std_msgs/msg/Float64MultiArray '{data: [0.5, -0.5]}'
#
# For random mode, run:
#   DEMO_MODE=random python3 dummy_simulator_controlled.py

import os
import time
import numpy as np
import gymnasium as gym
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

EXTERNAL_ACTION_TIMEOUT_S = 0.5
ACTION_TOPIC = "/reacher_action"
CMD_VEL_TOPIC = "/cmd_vel"
DEMO_MODE = os.getenv("DEMO_MODE", "random").strip().lower()

rclpy.init()


class ReacherRL(Node):
    def __init__(self):
        super().__init__("reacher_rl_node")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.env = gym.make("Reacher-v5")
        self.model = None

        self.external_action = None
        self.last_external_time = 0.0
        self.external_received_once = False
        self.mode_warned = False

        self.create_subscription(Float64MultiArray, ACTION_TOPIC, self.on_action, 10)
        self.create_subscription(Twist, CMD_VEL_TOPIC, self.on_cmd_vel, 10)

        try:
            from stable_baselines3 import SAC

            p = "/workspace/rl_models/reacher_sac_final"
            if os.path.exists(p + ".zip"):
                self.model = SAC.load(p)
                self.get_logger().info("Loaded trained model")
        except Exception as e:
            self.get_logger().warn(f"No model: {e}")
        if not self.model:
            self.get_logger().info("Using random actions")
        if DEMO_MODE not in ("random", "ros"):
            self.get_logger().warn(f"Unknown DEMO_MODE={DEMO_MODE}, defaulting to random")
        else:
            self.get_logger().info(f"DEMO_MODE={DEMO_MODE}")

        self.obs, _ = self.env.reset()
        self.timer = self.create_timer(1.0 / 30.0, self.step)

    def on_action(self, msg: Float64MultiArray):
        if DEMO_MODE != "ros":
            return
        action = np.array(msg.data, dtype=float)
        if action.size == 0:
            return
        action = self._clip_action(action)
        self.external_action = action
        self.last_external_time = time.monotonic()
        if not self.external_received_once:
            self.external_received_once = True
            self.get_logger().info(f"External actions received on {ACTION_TOPIC}")

    def on_cmd_vel(self, msg: Twist):
        if DEMO_MODE != "ros":
            return
        action = np.array([msg.linear.x, msg.angular.z], dtype=float)
        action = self._clip_action(action)
        self.external_action = action
        self.last_external_time = time.monotonic()
        if not self.external_received_once:
            self.external_received_once = True
            self.get_logger().info(f"External actions received on {CMD_VEL_TOPIC}")

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        if hasattr(self.env.action_space, "low") and hasattr(self.env.action_space, "high"):
            low = np.array(self.env.action_space.low, dtype=float)
            high = np.array(self.env.action_space.high, dtype=float)
            size = min(action.size, low.size)
            action = action[:size]
            return np.clip(action, low[:size], high[:size])
        return action

    def step(self):
        now = time.monotonic()
        if DEMO_MODE == "ros":
            if self.external_action is not None and (now - self.last_external_time) <= EXTERNAL_ACTION_TIMEOUT_S:
                action = self.external_action
            else:
                action = np.zeros(self.env.action_space.shape, dtype=float)
                if not self.mode_warned:
                    self.get_logger().info("DEMO_MODE=ros: waiting for ROS commands")
                    self.mode_warned = True
        else:
            if self.model:
                action, _ = self.model.predict(self.obs, deterministic=True)
            else:
                action = self.env.action_space.sample()

        self.obs, reward, terminated, truncated, _ = self.env.step(action)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["joint0", "joint1"]
        qpos = self.env.unwrapped.data.qpos
        msg.position = [float(qpos[0]), float(qpos[1])]
        qvel = self.env.unwrapped.data.qvel
        msg.velocity = [float(qvel[0]), float(qvel[1])]
        msg.effort = [0.0, 0.0]
        self.pub.publish(msg)

        if terminated or truncated:
            self.obs, _ = self.env.reset()


node = ReacherRL()
rclpy.spin(node)
