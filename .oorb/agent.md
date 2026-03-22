# Project: 4-DOF Robot Arm

A 4-DOF robot arm running in OORB Studio with MuJoCo simulation
and ROS 2 Jazzy. URDF-first design — the robot model is defined in XACRO,
with MuJoCo MJCF for physics simulation. Supports sim and hardware profiles.

## Architecture

Three ROS 2 packages in `ros2_ws/src/`:

### my_bot_description (ament_cmake)
Robot model package. Contains URDF/XACRO source of truth and meshes.
- `urdf/robot_arm.urdf.xacro` — Parametric robot model (4 revolute joints, inertials, collision)
- `meshes/` — Visual/collision meshes (STL/DAE)
- `launch/description.launch.py` — Starts `robot_state_publisher` with the URDF

### my_bot_bringup (ament_python)
Launch and configuration package. Defines sim and hardware profiles.
- `launch/sim.launch.py` — Full sim stack (description + MuJoCo + rosbridge)
- `launch/hardware.launch.py` — Hardware stack template (description + drivers + rosbridge)
- `config/sim_params.yaml` — Sim-tuned parameters (50 Hz, Kp=10, Kd=1)
- `config/hardware_params.yaml` — Hardware-tuned parameters (template)

### my_bot_py (ament_python)
Application nodes:
- **talker** — Hello world publisher on `/chatter` (verifying ROS is alive)
- **robot_arm** — Publishes sinusoidal `/joint_states` at 20 Hz for 4 joints. Drives the MuJoCo browser sim.
- **mujoco_sim** — Loads MuJoCo MJCF model, runs headless physics, publishes `/joint_states`, subscribes to `/position_controller/commands` for position control (Kp=10).

The MuJoCo scene is at `sim/models/mjcf/robot_arm.xml`. It defines a
4-DOF arm (base rotation, shoulder, elbow, wrist) with position actuators
and PD control. The URDF is the source of truth — MJCF can be auto-generated.

## ROS Interface

| Topic | Type | Dir | Hz | Purpose |
|---|---|---|---|---|
| /joint_states | sensor_msgs/JointState | PUB | 20-50 | Joint positions for 4 joints |
| /position_controller/commands | std_msgs/Float64MultiArray | SUB | — | Target joint positions (4 floats) |
| /chatter | std_msgs/String | PUB | 2 | Hello world heartbeat |
| /robot_description | std_msgs/String | PUB | latch | URDF XML (from robot_state_publisher) |
| /tf | tf2_msgs/TFMessage | PUB | 20-50 | Transform tree |
| /e_stop | std_msgs/Bool | SUB | — | Emergency stop (safety) |

Nodes:
- `/robot_arm_publisher` — demo sinusoidal motion (no control input)
- `/mujoco_sim_node` — actual physics sim with position control input
- `/robot_state_publisher` — publishes URDF and TF tree
- `/rosbridge_websocket` — browser ↔ ROS 2 bridge on port 9090
- `/talker` — heartbeat

## Launch Files

### sim.launch.py
Full simulation stack. Use this for development.
```
ros2 launch my_bot_bringup sim.launch.py
ros2 launch my_bot_bringup sim.launch.py model_path:=/workspace/sim/models/mjcf/custom.xml
```
Arguments:
- `use_sim` (default: true) — Use simulation time
- `model_path` (default: /workspace/sim/models/mjcf/robot_arm.xml) — MuJoCo model path

Launches: robot_state_publisher + mujoco_sim_node + rosbridge

### hardware.launch.py
Template for physical robot. Uncomment and configure hardware driver nodes.
Launches: robot_state_publisher + rosbridge (+ hardware drivers when configured)

## Sim / Hardware Profiles

- **sim** — MuJoCo simulation, no hardware needed. Config: `config/sim_params.yaml`
- **hardware** — Physical robot template. Config: `config/hardware_params.yaml`

Switch between profiles using the Studio profile selector or launch the appropriate file.

## Robot Model (URDF)

`ros2_ws/src/my_bot_description/urdf/robot_arm.urdf.xacro`:
- 4 revolute joints + 1 fixed end-effector frame
- Links: base_link → link1 → link2 → link3 → link4 → ee_link
- Joints: joint_base (yaw ±180°), joint_shoulder (pitch ±90°), joint_elbow (pitch -135° to 0°), joint_wrist (pitch ±90°)
- Full inertials and collision geometry for physics sim
- Parametric (xacro properties for link dimensions)

## MuJoCo Model

`sim/models/mjcf/robot_arm.xml`:
- 4 position actuators with PD control (kp=100-200, kv=5-20)
- RK4 integrator at 0.002s timestep
- Keyframes: home, extended, folded, demo

## Safety Constraints

Operational limits declared in `blueprint.yaml > safety`:
- Joint position and velocity limits per joint
- Cartesian workspace bounds: x±0.6m, y±0.6m, z 0-0.9m
- Emergency stop on `/e_stop` topic
- These are tighter than URDF mechanical limits for operational safety

## Known Gotchas

- Joint names must match exactly across URDF, MJCF, and ROS nodes:
  `joint_base`, `joint_shoulder`, `joint_elbow`, `joint_wrist`
- The mujoco_sim_node defaults to model path `/workspace/sim/models/mjcf/robot_arm.xml`.
  Override with launch argument or `--ros-args -p model_path:=/your/path.xml`
- `colcon build` must run from `/workspace/ros2_ws`, not `/workspace`
- After build, always `source install/setup.bash` before `ros2 run` or `ros2 launch`
- The MuJoCo scene breaks if timestep > 0.005s (arm flies apart)
- When adding meshes to URDF, use package:// URIs for portability

## Debugging Tips

- No `/joint_states`? → Check node is running: `ros2 node list`
- Arm doesn't move in sim? → Check the MuJoCo tab loaded the right XML
- `colcon build` says "no packages"? → You're in the wrong directory, cd to `ros2_ws/`
- Position commands ignored? → Make sure you're publishing to `/position_controller/commands` not `/cmd_vel`
- URDF doesn't load? → Run `xacro` manually to see parse errors: `xacro ros2_ws/src/my_bot_description/urdf/robot_arm.urdf.xacro`
- TF tree broken? → Check `ros2 run tf2_tools view_frames` to visualize

## Workspace State

This Blueprint includes workspace state from the original session in `.oorb/state/`:
- **Conversations** (`.oorb/state/conversations/`) — Agent chat threads (JSONL)
- **Logs** (`.oorb/state/logs/`) — Build output, terminal output, ROS node logs
- **Outputs** (`.oorb/state/outputs/`) — Generated artifacts
- **Bags** (`.oorb/state/bags/`) — Recorded rosbags (mcap format)

## Marketplace Listing

`.oorb/listing.yaml` is the single source of truth for the marketplace page.

## Assets & Uploads

- `fab/uploads/` — Studio upload landing zone for user-uploaded files
- `sim/models/mjcf/` — MuJoCo scene directory
- `fab/` — CAD root (STEP, STL, URDF source files)

## File Map

- `ros2_ws/src/my_bot_description/urdf/robot_arm.urdf.xacro` — URDF source of truth
- `ros2_ws/src/my_bot_description/launch/description.launch.py` — Robot state publisher
- `ros2_ws/src/my_bot_description/meshes/` — Visual/collision meshes
- `ros2_ws/src/my_bot_bringup/launch/sim.launch.py` — Sim stack launch
- `ros2_ws/src/my_bot_bringup/launch/hardware.launch.py` — Hardware stack template
- `ros2_ws/src/my_bot_bringup/config/sim_params.yaml` — Sim parameters
- `ros2_ws/src/my_bot_bringup/config/hardware_params.yaml` — Hardware parameters
- `ros2_ws/src/my_bot_py/my_bot_py/robot_arm_publisher.py` — Demo sinusoidal publisher
- `ros2_ws/src/my_bot_py/my_bot_py/mujoco_sim_node.py` — Physics sim + position control
- `ros2_ws/src/my_bot_py/my_bot_py/talker.py` — Hello world node
- `ros2_ws/src/my_bot_py/setup.py` — Entry points: talker, robot_arm, mujoco_sim
- `sim/models/mjcf/robot_arm.xml` — MuJoCo arm model (4 DOF, PD actuators)
- `scripts/benchmark_sim.py` — Sim step rate benchmark
- `scripts/benchmark_control.py` — Control loop latency benchmark
- `.github/workflows/blueprint-ci.yaml` — CI pipeline
- `fab/uploads/` — Studio upload landing zone
