# Project: 4-DOF Robot Arm

A simple 4-DOF robot arm running in OORB Studio with MuJoCo simulation
and ROS 2 Jazzy. Publishes joint states at 20 Hz, accepts position commands.

## Architecture

One ROS 2 package in `ros2_ws/src/my_bot_py/` with three nodes:

- **talker** — Hello world publisher on `/chatter` (for verifying ROS is alive)
- **robot_arm** — Publishes sinusoidal `/joint_states` at 20 Hz for 4 joints. This drives the MuJoCo browser sim.
- **mujoco_sim** — Loads a MuJoCo MJCF model, runs headless physics, publishes `/joint_states`, subscribes to `/position_controller/commands` for position control with P controller (Kp=10).

The MuJoCo scene is at `sim/models/mjcf/robot_arm.xml`. It defines a
4-DOF arm (base rotation, shoulder, elbow, wrist) with position actuators
and PD control built into the MJCF.

## ROS Interface

| Topic | Type | Dir | Hz | Purpose |
|---|---|---|---|---|
| /joint_states | sensor_msgs/JointState | PUB | 20-50 | Joint positions for 4 joints |
| /position_controller/commands | std_msgs/Float64MultiArray | SUB | — | Target joint positions (4 floats) |
| /chatter | std_msgs/String | PUB | 2 | Hello world heartbeat |

Nodes:
- `/robot_arm_publisher` — demo sinusoidal motion (no control input)
- `/mujoco_sim_node` — actual physics sim with position control input
- `/talker` — heartbeat

## MuJoCo Model

`sim/models/mjcf/robot_arm.xml`:
- 4 joints: joint_base (yaw ±180°), joint_shoulder (pitch ±90°), joint_elbow (pitch -135° to 0°), joint_wrist (pitch ±90°)
- 4 position actuators with PD control (kp=100-200, kv=5-20)
- RK4 integrator at 0.002s timestep
- Keyframes: home, extended, folded, demo

## Known Gotchas

- Joint names in the ROS node must match joint names in the MJCF exactly:
  `joint_base`, `joint_shoulder`, `joint_elbow`, `joint_wrist`
- The mujoco_sim_node defaults to model path `/workspace/sim/models/mjcf/robot_arm.xml`.
  Override with `--ros-args -p model_path:=/your/path.xml`
- `colcon build` must run from `/workspace/ros2_ws`, not `/workspace`
- After build, always `source install/setup.bash` before `ros2 run`
- The MuJoCo scene breaks if timestep > 0.005s (arm flies apart)

## Debugging Tips

- No `/joint_states`? → Check node is running: `ros2 node list`
- Arm doesn't move in sim? → Check the MuJoCo tab loaded the right XML
- `colcon build` says "no packages"? → You're in the wrong directory, cd to `ros2_ws/`
- Position commands ignored? → Make sure you're publishing to `/position_controller/commands` not `/cmd_vel`

## Workspace State

This Blueprint includes workspace state from the original session in `.oorb/state/`:

- **Conversations** (`.oorb/state/conversations/`) — Agent chat threads as JSONL. Each file is one conversation. On import, Studio loads these into the chat UI as previous sessions so the agent has full context of what was already done, discussed, and debugged.
- **Logs** (`.oorb/state/logs/`) — Build output, terminal output, ROS node logs. On import, Studio populates the logs UI so users can see what happened in the original workspace.
- **Outputs** (`.oorb/state/outputs/`) — Generated artifacts (plots, rosbags, exported data). On import, these appear in the file tree.

When the agent sees a workspace with conversation history, it should read the thread summaries to understand what the user has already tried, what worked, and what didn't — avoiding repeated suggestions and building on prior progress.

## Assets & Uploads

- `fab/uploads/` — Studio upload landing zone. When users upload files (URDF, meshes, STEP, MJCF folders) via the Studio UI, they land here as `fab/uploads/<foldername>/`. The import/publish flow reads from this directory.
- `sim/models/mjcf/` — MuJoCo scene directory. Scenes here can reference meshes from `fab/uploads/` using relative paths.
- `fab/` — CAD root (STEP, STL, URDF source files). Subdirectories: `fab/step/`, `fab/stl/`, `fab/uploads/`.

## File Map

- `ros2_ws/src/my_bot_py/my_bot_py/robot_arm_publisher.py` — Demo sinusoidal publisher
- `ros2_ws/src/my_bot_py/my_bot_py/mujoco_sim_node.py` — Physics sim + position control
- `ros2_ws/src/my_bot_py/my_bot_py/talker.py` — Hello world node
- `ros2_ws/src/my_bot_py/setup.py` — Entry points: talker, robot_arm, mujoco_sim
- `sim/models/mjcf/robot_arm.xml` — MuJoCo arm model (4 DOF, PD actuators)
- `fab/uploads/` — Studio upload landing zone (user-uploaded files)
- `fab/step/` — STEP files for manufacturing
- `fab/stl/` — STL files for 3D printing
