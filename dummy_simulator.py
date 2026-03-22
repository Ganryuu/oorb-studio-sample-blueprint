#!/usr/bin/env python3
# Reacher RL Demo - Dummy Simulator (basic)
# A 2-DOF robotic arm (Reacher-v5) that publishes joint states via ROS2.
# This is the simple version - just runs the demo without external control.
# For ROS-controlled mode, use dummy_simulator_controlled.py instead.
#
# Run: python3 dummy_simulator.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import gymnasium as gym
import os

rclpy.init()

class ReacherRL(Node):
    def __init__(self):
        super().__init__('reacher_rl_node')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.env = gym.make('Reacher-v5')
        self.model = None
        try:
            from stable_baselines3 import SAC
            p = '/workspace/rl_models/reacher_sac_final'
            if os.path.exists(p + '.zip'):
                self.model = SAC.load(p)
                self.get_logger().info('Loaded trained model')
        except Exception as e:
            self.get_logger().warn(f'No model: {e}')
        if not self.model:
            self.get_logger().info('Using random actions')
        self.obs, _ = self.env.reset()
        self.timer = self.create_timer(1.0 / 30.0, self.step)

    def step(self):
        if self.model:
            action, _ = self.model.predict(self.obs, deterministic=True)
        else:
            action = self.env.action_space.sample()
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint0', 'joint1']
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
