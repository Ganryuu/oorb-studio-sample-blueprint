#!/usr/bin/env python3
"""
MuJoCo demo tutorial for OORB Studio

This script is a hands-on reference demo to:
1) load a MuJoCo model in OORB Studio, and
2) control it through ROS topics so motion is visible in the simulator.

This is setup to work for a Unitree A1, you can apply the same workflow to other robots
from MuJoCo Menagerie.

Menagerie repository(already cloned in the workspace):
https://github.com/google-deepmind/mujoco_menagerie

What this demo publishes:
- /joint_states (sensor_msgs/JointState)
- /mujoco_qpos (std_msgs/Float64MultiArray)
- /mujoco_ctrl_applied (std_msgs/Float64MultiArray)

What this demo subscribes to:
- /cmd_vel (geometry_msgs/Twist): high-level command mapped to a simple gait
- /mujoco_ctrl (std_msgs/Float64MultiArray): direct actuator values (A1 expects 12)

Default model path used by this script:
- /workspace/mujoco_menagerie/unitree_a1/scene.xml

===============================================================================
TUTORIAL: RUN THIS DEMO IN OORB STUDIO
===============================================================================

Step 0: Prepare files in the workspace
- Preferred path: clone Menagerie directly into /workspace or check that it's already cloned.
- Uploading model files is manual via the simulator UI in this version, upload Unitree A1 files first.
- Ensure these files resolve correctly:
  - /workspace/fab/uploads/scene.xml
  - /workspace/fab/uploads/a1.xml
  - /workspace/fab/uploads/assets/*.obj

Step 1: Confirm model renders in the MuJoCo UI
- Open MuJoCo in Studio and load via the simulator UI:
  /workspace/fab/uploads/scene.xml
- If the model renders but does not move, continue with the ROS steps below.

Step 2: Open 3 terminals in Studio

Terminal 1: Run the demo controller
  source /opt/ros/jazzy/setup.bash
  export ROS_LOCALHOST_ONLY=0
  export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
  export MODEL_PATH=/workspace/fab/uploads/scene.xml
  python3 /workspace/unitree_a1_ros_control.py

Terminal 2: Watch state and applied control
  source /opt/ros/jazzy/setup.bash
  ros2 topic hz /joint_states
  ros2 topic echo /mujoco_ctrl_applied

Terminal 3: Send commands (use one mode at a time)
  source /opt/ros/jazzy/setup.bash

  Option A: High-level command on /cmd_vel
  ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
    '{linear: {x: 0.5}, angular: {z: 0.2}}'

  Option B: Direct actuator control on /mujoco_ctrl
  ros2 topic pub -r 10 /mujoco_ctrl std_msgs/msg/Float64MultiArray \
    '{data: [0.05,0.4,-0.8,-0.05,0.4,-0.8,0.05,0.4,-0.8,-0.05,0.4,-0.8]}'

Step 3: Validate that control is truly working
- /mujoco_ctrl_applied should change while commands are being published.
- /joint_states and /mujoco_qpos should continuously change.
- The Unitree A1 model should visibly move in the MuJoCo UI.

If you stop publishing:
- The demo gradually returns to a stable home control pose.

===============================================================================
HOW THE CONTROL LOGIC WORKS
===============================================================================

Priority order:
1) Fresh /mujoco_ctrl messages (direct actuator override)
2) Fresh /cmd_vel messages (mapped gait behavior)
3) Home pose fallback when no fresh command exists

This priority keeps the demo predictable:
- Direct control is exact and best for debugging actuator-level behavior.
- /cmd_vel is convenient for quick motion checks from standard ROS tools.

===============================================================================
USING THIS FLOW FOR OTHER ROBOTS
===============================================================================

To adapt this tutorial to another Menagerie model:
1) Change MODEL_PATH to that robot's scene.xml.
2) Update actuator-name mapping logic in this script to match that model.
3) Keep the same 3-terminal run pattern for validation.

This gives a repeatable ROS-to-MuJoCo integration workflow in OORB Studio
even when UI-only controls are still evolving.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/mujoco_menagerie/unitree_a1/scene.xml")
STEP_HZ = float(os.getenv("STEP_HZ", "120"))
CMD_TIMEOUT_S = float(os.getenv("CMD_TIMEOUT_S", "0.6"))
CTRL_TIMEOUT_S = float(os.getenv("CTRL_TIMEOUT_S", "0.4"))


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


class UnitreeA1RosDemo(Node):
    def __init__(self) -> None:
        super().__init__("unitree_a1_ros_demo")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.ctrl = np.zeros(self.model.nu, dtype=float)

        self.actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.ctrl_low = np.array(self.model.actuator_ctrlrange[:, 0], dtype=float)
        self.ctrl_high = np.array(self.model.actuator_ctrlrange[:, 1], dtype=float)

        # Home pose from keyframe if available, otherwise from zeros.
        if self.model.nkey > 0 and self.model.nu > 0:
            self.home_ctrl = np.array(self.model.key_ctrl[0][: self.model.nu], dtype=float)
        else:
            self.home_ctrl = np.zeros(self.model.nu, dtype=float)
        self.home_ctrl = np.clip(self.home_ctrl, self.ctrl_low, self.ctrl_high)
        self.ctrl[:] = self.home_ctrl

        self.cmd_vel = Twist()
        self.last_cmd_vel_t = 0.0
        self.last_direct_ctrl_t = 0.0

        self.phase = 0.0
        self.last_step_t = time.monotonic()

        self.pub_joint = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_qpos = self.create_publisher(Float64MultiArray, "/mujoco_qpos", 10)
        self.pub_ctrl = self.create_publisher(Float64MultiArray, "/mujoco_ctrl_applied", 10)

        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(Float64MultiArray, "/mujoco_ctrl", self.on_direct_ctrl, 10)

        self.act_idx = self._build_actuator_index()

        self.timer = self.create_timer(1.0 / STEP_HZ, self.step)
        self.get_logger().info(
            f"Loaded {MODEL_PATH} nq={self.model.nq} nu={self.model.nu} njnt={self.model.njnt}"
        )
        self.get_logger().info(
            "Control topics: /cmd_vel (gait), /mujoco_ctrl (direct). "
            "Direct control takes priority while messages are fresh."
        )

    def _build_actuator_index(self) -> Dict[str, int]:
        index = {}
        for i, name in enumerate(self.actuator_names):
            index[name] = i
        expected = [
            "FR_hip",
            "FR_thigh",
            "FR_calf",
            "FL_hip",
            "FL_thigh",
            "FL_calf",
            "RR_hip",
            "RR_thigh",
            "RR_calf",
            "RL_hip",
            "RL_thigh",
            "RL_calf",
        ]
        missing = [name for name in expected if name not in index]
        if missing:
            self.get_logger().warn(
                f"Expected A1 actuator names missing: {missing}. "
                "cmd_vel gait mapping may be partial."
            )
        return index

    def on_cmd_vel(self, msg: Twist) -> None:
        self.cmd_vel = msg
        self.last_cmd_vel_t = time.monotonic()

    def on_direct_ctrl(self, msg: Float64MultiArray) -> None:
        arr = np.array(msg.data, dtype=float)
        if arr.size == 0:
            return
        n = min(arr.size, self.model.nu)
        self.ctrl[:n] = np.clip(arr[:n], self.ctrl_low[:n], self.ctrl_high[:n])
        if n < self.model.nu:
            self.ctrl[n:] = self.home_ctrl[n:]
        self.last_direct_ctrl_t = time.monotonic()

    def _apply_cmd_vel_gait(self, now: float) -> None:
        v = _clip(self.cmd_vel.linear.x, -1.0, 1.0)
        w = _clip(self.cmd_vel.angular.z, -1.0, 1.0)
        speed = _clip(abs(v) + 0.5 * abs(w), 0.0, 1.0)

        if speed < 0.02:
            self.ctrl[:] = self.home_ctrl
            return

        dt = max(1e-4, now - self.last_step_t)
        freq_hz = 0.8 + 2.8 * speed
        self.phase += (2.0 * math.pi * freq_hz * dt)
        if self.phase > 2.0 * math.pi:
            self.phase -= 2.0 * math.pi

        amp_thigh = 0.30 * speed
        amp_calf = 0.45 * speed
        amp_abd = 0.10 * abs(w) + 0.04 * speed

        leg_phase = {
            "FR": self.phase + 0.0,
            "FL": self.phase + math.pi,
            "RR": self.phase + math.pi,
            "RL": self.phase + 0.0,
        }
        side_sign = {"FR": -1.0, "FL": +1.0, "RR": -1.0, "RL": +1.0}

        target = np.array(self.home_ctrl, copy=True)
        for leg in ("FR", "FL", "RR", "RL"):
            phi = leg_phase[leg]
            hip_name = f"{leg}_hip"
            thigh_name = f"{leg}_thigh"
            calf_name = f"{leg}_calf"

            if hip_name in self.act_idx:
                i = self.act_idx[hip_name]
                target[i] = self.home_ctrl[i] + side_sign[leg] * amp_abd * np.sign(w)
            if thigh_name in self.act_idx:
                i = self.act_idx[thigh_name]
                target[i] = self.home_ctrl[i] + amp_thigh * math.sin(phi)
            if calf_name in self.act_idx:
                i = self.act_idx[calf_name]
                target[i] = self.home_ctrl[i] - amp_calf * math.sin(phi)

        self.ctrl[:] = np.clip(target, self.ctrl_low, self.ctrl_high)

    def step(self) -> None:
        now = time.monotonic()

        # Direct /mujoco_ctrl has priority for a short timeout window.
        direct_fresh = (now - self.last_direct_ctrl_t) <= CTRL_TIMEOUT_S
        cmd_fresh = (now - self.last_cmd_vel_t) <= CMD_TIMEOUT_S
        if direct_fresh:
            pass
        elif cmd_fresh:
            self._apply_cmd_vel_gait(now)
        else:
            self.ctrl[:] = self.home_ctrl

        self.data.ctrl[:] = self.ctrl
        mujoco.mj_step(self.model, self.data)
        self.last_step_t = now

        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_names
        joint_msg.position = self.data.qpos.tolist()
        joint_msg.velocity = self.data.qvel.tolist()
        self.pub_joint.publish(joint_msg)

        qpos_msg = Float64MultiArray()
        qpos_msg.data = self.data.qpos.tolist()
        self.pub_qpos.publish(qpos_msg)

        ctrl_msg = Float64MultiArray()
        ctrl_msg.data = self.ctrl.tolist()
        self.pub_ctrl.publish(ctrl_msg)


def main() -> None:
    rclpy.init()
    node = UnitreeA1RosDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
