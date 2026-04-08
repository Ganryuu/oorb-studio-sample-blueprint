"""
MuJoCo simulation node for UFactory Lite 6.

Loads the MuJoCo scene, runs headless physics at 50 Hz, publishes
/joint_states, and subscribes to /position_controller/commands for
position control of the 6 joints.
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
DEFAULT_MODEL = '/workspace/sim/models/mjcf/ufactory_lite6/scene.xml'


class MuJoCoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')

        self.declare_parameter('model_path', DEFAULT_MODEL)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        try:
            import mujoco
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            self.mujoco = mujoco
            self.get_logger().info(f'Loaded MuJoCo model: {model_path}')
            self.get_logger().info(f'  DOFs: {self.model.nq}, Actuators: {self.model.nu}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model {model_path}: {e}')
            raise

        self.e_stopped = False
        self.target_positions = np.zeros(self.model.nu)

        # Apply home keyframe if available
        for i in range(self.model.nkey):
            key_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_KEY, i)
            if key_name == 'home':
                self.mujoco.mj_resetDataKeyframe(self.model, self.data, i)
                self.target_positions[:] = self.data.ctrl[:]
                self.get_logger().info(f'Applied home keyframe: {self.target_positions}')
                break

        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.create_subscription(
            Float64MultiArray, '/position_controller/commands',
            self.cmd_callback, 10,
        )
        self.create_subscription(Bool, '/e_stop', self.estop_callback, 10)

        self.timer = self.create_timer(0.02, self.step_callback)  # 50 Hz
        self.get_logger().info('Lite 6 MuJoCo sim running at 50 Hz')

    def cmd_callback(self, msg: Float64MultiArray):
        if len(msg.data) >= self.model.nu:
            self.target_positions[:] = msg.data[:self.model.nu]

    def estop_callback(self, msg: Bool):
        self.e_stopped = msg.data
        if self.e_stopped:
            self.get_logger().warn('E-STOP activated')

    def step_callback(self):
        if not self.e_stopped:
            self.data.ctrl[:] = self.target_positions
        else:
            self.data.ctrl[:] = self.data.qpos[:self.model.nu]  # hold position

        self.mujoco.mj_step(self.model, self.data)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES[:self.model.nq]
        msg.position = list(self.data.qpos[:self.model.nq])
        msg.velocity = list(self.data.qvel[:self.model.nq])
        msg.effort = list(self.data.qfrc_actuator[:self.model.nu]) if self.model.nu <= self.model.nq else []
        self.js_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MuJoCoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
