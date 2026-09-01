"""ROS 2 node that drives a robot with a VLA policy.

Wiring:

    /vla/image       sensor_msgs/Image            camera frames
    /vla/instruction std_msgs/String              task, latched-style (last wins)
    /joint_states    sensor_msgs/JointState       proprioception (pi0, SmolVLA)
        ->
    /vla/action      std_msgs/Float64MultiArray   one action per control tick
    /vla/chunk       std_msgs/Float64MultiArray   the full predicted chunk
    /vla/status      std_msgs/String              JSON diagnostics

The important design point is that **inference and control run at different
rates on different threads**. A VLA takes 30-200 ms to produce a chunk, while a
manipulator wants commands every 10-30 ms. Running inference inside the control
timer would make the control period equal to the inference latency, which is
exactly the stutter this engine exists to avoid.

So inference runs in a background worker on the latest available observation,
and the control timer publishes from a
:class:`~vla_engine.control.chunker.ChunkExecutor` at a fixed rate, blending
overlapping chunks. If inference falls behind, the executor holds its last
command and reports a rising starvation rate on ``/vla/status`` rather than
letting the arm jump.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np

try:
    import rclpy
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float64MultiArray, String
except ImportError as exc:  # pragma: no cover - only meaningful inside ROS
    raise SystemExit(
        "vla_ros requires ROS 2 (source /opt/ros/jazzy/setup.bash before running)"
    ) from exc

from vla_engine import EngineConfig, Observation, VLAEngine
from vla_engine.control.chunker import ChunkExecutor, ChunkPolicy
from vla_engine.ros.conversions import image_msg_to_array, joint_state_to_array

# Best-effort, depth 1: for a live camera the newest frame is the only one that
# matters, and a reliable queue would deliver stale frames after a hiccup.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

# The instruction changes rarely and late subscribers must still receive it,
# so it is reliable and transient-local (ROS's equivalent of latched).
COMMAND_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class VLAPolicyNode(Node):
    """Runs a VLA policy against ROS topics."""

    def __init__(self) -> None:
        super().__init__("vla_policy")

        self.declare_parameter("model", "echo")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("device", "auto")
        self.declare_parameter("devices", "")
        self.declare_parameter("dtype", "auto")
        self.declare_parameter("quantization", "")
        self.declare_parameter("unnorm_key", "")
        self.declare_parameter("instruction", "")
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("compile", True)
        self.declare_parameter("ensemble_decay", 0.1)
        self.declare_parameter("strategy", "temporal_ensemble")
        self.declare_parameter("image_topic", "/vla/image")
        self.declare_parameter("autostart", True)
        # Remote inference: point this at a `vla serve` instance to run the
        # policy on a GPU workstation while the node runs on the robot or in a
        # GPU-less container. This is the OORB Studio path.
        self.declare_parameter("remote_url", "")
        self.declare_parameter("joint_names", [""])

        def param(name: str):
            return self.get_parameter(name).value

        self._instruction: str = str(param("instruction") or "")
        self._latest_image: np.ndarray | None = None
        self._latest_state: np.ndarray | None = None
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._inference_count = 0
        self._inference_ms = 0.0
        self._last_error: str | None = None

        self._joint_names = [n for n in (param("joint_names") or []) if n] or None

        remote_url = str(param("remote_url") or "").strip()
        if remote_url:
            from vla_engine.registry import get_card
            from vla_engine.serve.client import VLAClient

            self.get_logger().info(f"using remote inference at {remote_url}")
            self.engine = VLAClient(remote_url)
            self.card = get_card(str(param("model")))
            self.spec = self.card.spec
            self.devices = [remote_url]
        else:
            devices = [d.strip() for d in str(param("devices") or "").split(",") if d.strip()]
            config = EngineConfig(
                model=str(param("model")),
                checkpoint=str(param("checkpoint")) or None,
                device=str(param("device")),
                devices=devices,
                precision={
                    "dtype": str(param("dtype")),
                    "quantization": str(param("quantization")) or None,
                },
                compile={"enabled": bool(param("compile"))},
                unnorm_key=str(param("unnorm_key")) or None,
            )
            self.get_logger().info(f"loading policy {config.model} ...")
            engine = VLAEngine(config).load()
            self.engine = engine
            self.card = engine.card
            self.spec = engine.spec
            self.devices = engine.devices
            self.get_logger().info(f"policy ready on {self.devices}")

        self.executor_policy = ChunkPolicy(
            strategy=str(param("strategy")),
            ensemble_decay=float(param("ensemble_decay")),
            discrete_dims=(self.spec.action_dim - 1,),  # gripper
        )
        self.chunk_executor = ChunkExecutor(
            action_dim=self.spec.action_dim, policy=self.executor_policy
        )

        sensors = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            Image, str(param("image_topic")), self._on_image, SENSOR_QOS, callback_group=sensors
        )
        self.create_subscription(
            String, "/vla/instruction", self._on_instruction, COMMAND_QOS, callback_group=sensors
        )
        self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, SENSOR_QOS, callback_group=sensors
        )

        self.action_pub = self.create_publisher(Float64MultiArray, "/vla/action", 10)
        self.chunk_pub = self.create_publisher(Float64MultiArray, "/vla/chunk", 10)
        self.status_pub = self.create_publisher(String, "/vla/status", 10)

        control_hz = float(param("control_hz"))
        self.create_timer(1.0 / max(control_hz, 1e-3), self._on_control_tick)
        self.create_timer(1.0, self._publish_status)

        self._worker: threading.Thread | None = None
        if bool(param("autostart")):
            self.start_inference()
        self.get_logger().info(
            f"control loop at {control_hz:.1f} Hz, "
            f"{self.spec.action_dim}-dim actions, "
            f"strategy={self.executor_policy.strategy}"
        )

    # -- subscriptions -------------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        try:
            frame = image_msg_to_array(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
            return
        with self._lock:
            self._latest_image = frame

    def _on_instruction(self, msg: String) -> None:
        instruction = msg.data.strip()
        with self._lock:
            changed = instruction != self._instruction
            self._instruction = instruction
        if changed:
            # A new task invalidates in-flight chunks: blending actions for the
            # old task into the new one would fight the policy.
            self.chunk_executor.reset()
            self.get_logger().info(f"instruction: {instruction!r} (chunk buffer reset)")

    def _on_joint_state(self, msg: JointState) -> None:
        if not msg.position:
            return
        try:
            state = joint_state_to_array(msg, self._joint_names)
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
            return
        with self._lock:
            self._latest_state = state

    # -- inference worker ----------------------------------------------------

    def start_inference(self) -> None:
        """Start the background inference thread."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._shutdown.clear()
        self._worker = threading.Thread(
            target=self._inference_loop, name="vla-inference", daemon=True
        )
        self._worker.start()

    def stop_inference(self) -> None:
        self._shutdown.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None

    def _snapshot(self) -> Observation | None:
        """Take the most recent observation, or ``None`` if not ready."""
        with self._lock:
            image = self._latest_image
            state = self._latest_state
            instruction = self._instruction
        if image is None:
            return None
        if self.spec.language_conditioned and not instruction:
            return None
        if self.spec.requires_state and state is None:
            return None
        return Observation.single(image, instruction, state)

    def _inference_loop(self) -> None:
        """Predict continuously on the newest observation.

        Deliberately always uses the *latest* frame rather than a queue: an
        action computed from a stale frame is worse than no action, and queuing
        would guarantee the policy falls further behind over time.
        """
        while not self._shutdown.is_set():
            if not self.chunk_executor.needs_replan():
                time.sleep(0.002)
                continue
            observation = self._snapshot()
            if observation is None:
                time.sleep(0.05)
                continue
            try:
                started = time.perf_counter()
                chunk = self.engine.predict(observation)
                elapsed = (time.perf_counter() - started) * 1000.0
            except Exception as exc:
                self._last_error = str(exc)
                self.get_logger().error(f"inference failed: {exc}")
                time.sleep(0.5)
                continue

            self._last_error = None
            self._inference_count += 1
            self._inference_ms = elapsed
            self.chunk_executor.submit(chunk)

            message = Float64MultiArray()
            message.data = [float(v) for v in chunk.actions.reshape(-1)]
            self.chunk_pub.publish(message)

    # -- control loop --------------------------------------------------------

    def _on_control_tick(self) -> None:
        """Publish one action at the control rate."""
        action = self.chunk_executor.step()
        if action is None:
            return
        message = Float64MultiArray()
        message.data = [float(v) for v in action]
        self.action_pub.publish(message)

    def _publish_status(self) -> None:
        stats = self.chunk_executor.stats()
        payload = {
            "model": self.card.key,
            "devices": self.devices,
            "inferences": self._inference_count,
            "last_inference_ms": round(self._inference_ms, 2),
            "instruction": self._instruction,
            "has_image": self._latest_image is not None,
            **stats,
        }
        if self._last_error:
            payload["error"] = self._last_error
        message = String()
        message.data = json.dumps(payload)
        self.status_pub.publish(message)

        if stats["starvation_rate"] > 0.25 and stats["steps"] > 50:
            self.get_logger().warning(
                f"control loop starved on {stats['starvation_rate']:.0%} of ticks: "
                "inference cannot keep up. Lower control_hz, shorten the horizon, "
                "or use a smaller policy.",
                throttle_duration_sec=10.0,
            )

    def destroy_node(self) -> None:
        self.stop_inference()
        closer = getattr(self.engine, "unload", None) or getattr(self.engine, "close", None)
        if callable(closer):
            closer()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = VLAPolicyNode()
        # Multi-threaded so the control timer keeps firing while a slow
        # subscription callback or status publish is in flight.
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
