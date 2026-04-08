# Project: UFactory Lite 6

A 6-DOF robot arm (UFactory Lite 6) running in OORB Studio with MuJoCo
simulation and ROS 2 Jazzy. Uses real STL meshes from the MuJoCo Menagerie
model -- both visual and collision geometry. URDF-first design with MJCF
scene for physics simulation. Supports sim and hardware profiles.

## Architecture

Three ROS 2 packages in `ros2_ws/src/`:

### lite6_description (ament_cmake)
Robot model package. Contains URDF/XACRO source of truth and STL meshes.
- `urdf/lite6.urdf.xacro` -- 6-DOF arm model (6 revolute joints, real inertials from manufacturer data, STL mesh references)
- `meshes/visual/` -- 14 STL files for visual rendering (link_base, link1-6, gripper parts)
- `meshes/collision/` -- 13 STL files for collision geometry (simplified meshes)
- `launch/description.launch.py` -- Starts `robot_state_publisher` with the URDF

### lite6_bringup (ament_python)
Launch and configuration package. Defines sim and hardware profiles.
- `launch/sim.launch.py` -- Full sim stack (description + MuJoCo sim node + rosbridge)
- `config/sim_params.yaml` -- Sim-tuned parameters (50 Hz, PD gains per joint)
- `config/hardware_params.yaml` -- Hardware template (xArm SDK connection)

### lite6_control (ament_python)
Application nodes:
- **talker** -- Heartbeat publisher on `/chatter`
- **joint_publisher** -- Demo sinusoidal motion on `/joint_states` at 20 Hz for 6 joints
- **mujoco_sim** -- Loads MuJoCo scene, runs headless physics at 50 Hz, publishes `/joint_states`, subscribes to `/position_controller/commands` for position control. Applies home keyframe on start.

The MuJoCo scene is at `sim/models/mjcf/ufactory_lite6/scene.xml`. It includes
`lite6.xml` which defines the 6-DOF arm with position actuators (kp=2000, kv=200
for large joints; kp=800 for wrist). Meshes are in `assets/visual/` and
`assets/collision/`. The model comes from MuJoCo Menagerie (BSD-3 license).

## ROS Interface

| Topic | Type | Dir | Hz | Purpose |
|---|---|---|---|---|
| /joint_states | sensor_msgs/JointState | PUB | 50 | Joint positions for 6 joints |
| /position_controller/commands | std_msgs/Float64MultiArray | SUB | -- | Target positions (6 floats, radians) |
| /chatter | std_msgs/String | PUB | 2 | Heartbeat |
| /robot_description | std_msgs/String | PUB | latch | URDF XML |
| /tf | tf2_msgs/TFMessage | PUB | 50 | Transform tree |
| /e_stop | std_msgs/Bool | SUB | -- | Emergency stop |

Nodes:
- `/joint_publisher` -- demo sinusoidal motion (no control input needed)
- `/mujoco_sim_node` -- physics sim with position control input
- `/robot_state_publisher` -- publishes URDF and TF tree
- `/rosbridge_websocket` -- browser <-> ROS 2 bridge on port 9090
- `/talker` -- heartbeat

## Joint Specifications

| Joint | Axis | Range (rad) | Range (deg) | Actuator Force |
|---|---|---|---|---|
| joint1 (base) | Z | [-6.28, 6.28] | +/-360 | 50 N |
| joint2 (shoulder) | Z | [-2.62, 2.62] | +/-150 | 50 N |
| joint3 (elbow) | Z | [-0.06, 5.24] | -3.5 to 300 | 32 N |
| joint4 (wrist 1) | Z | [-6.28, 6.28] | +/-360 | 32 N |
| joint5 (wrist 2) | Z | [-2.16, 2.16] | +/-124 | 20 N |
| joint6 (flange) | Z | [-6.28, 6.28] | +/-360 | 20 N |

Home keyframe: `[0, 0, 1.57, 0, 1.57, 0]` (elbow and wrist 2 at 90 deg)

## Launch Files

### sim.launch.py
Full simulation stack. Use this for development.
```
ros2 launch lite6_bringup sim.launch.py
ros2 launch lite6_bringup sim.launch.py model_path:=/workspace/sim/models/mjcf/ufactory_lite6/lite6_gripper_wide.xml
```
Arguments:
- `model_path` (default: /workspace/sim/models/mjcf/ufactory_lite6/scene.xml) -- MuJoCo scene path

Launches: robot_state_publisher + mujoco_sim_node + rosbridge

## MuJoCo Model Variants

Three MJCF configurations available in `sim/models/mjcf/ufactory_lite6/`:
- `scene.xml` -- Base arm, no gripper (includes `lite6.xml`)
- `lite6_gripper_wide.xml` -- With wide gripper (wider objects, limited close)
- `lite6_gripper_narrow.xml` -- With narrow gripper (full close, narrower objects)

All use the same STL meshes in `assets/visual/` and `assets/collision/`.

## Safety Constraints

Declared in `blueprint.yaml > safety`:
- Joint position and velocity limits per joint (from manufacturer specs)
- Cartesian workspace bounds: +/-0.6m XY, 0-0.7m Z
- Emergency stop on `/e_stop` topic

## Known Gotchas

- Joint names must match exactly across URDF, MJCF, and ROS nodes: `joint1` through `joint6`
- The mujoco_sim_node defaults to scene.xml. Override with launch argument `model_path:=...`
- `colcon build` must run from `/workspace/ros2_ws`, not `/workspace`
- After build, always `source install/setup.bash` before `ros2 run` or `ros2 launch`
- MuJoCo requires version >= 3.1.0 for this model
- The URDF uses `package://lite6_description/meshes/` URIs -- the package must be built and sourced first

## Debugging Tips

- No `/joint_states`? -- Check node is running: `ros2 node list`
- Arm doesn't move? -- Check MuJoCo loaded the right XML and actuator count matches (6)
- `colcon build` says "no packages"? -- You're in the wrong directory, cd to `ros2_ws/`
- URDF viewer blank? -- Ensure meshes are installed: `ls install/lite6_description/share/lite6_description/meshes/visual/`
- Position commands ignored? -- Publish to `/position_controller/commands` with exactly 6 floats
- TF tree broken? -- `ros2 run tf2_tools view_frames`

## File Map

- `ros2_ws/src/lite6_description/urdf/lite6.urdf.xacro` -- URDF source of truth
- `ros2_ws/src/lite6_description/meshes/visual/*.stl` -- 14 visual meshes
- `ros2_ws/src/lite6_description/meshes/collision/*.stl` -- 13 collision meshes
- `ros2_ws/src/lite6_description/launch/description.launch.py` -- Robot state publisher
- `ros2_ws/src/lite6_bringup/launch/sim.launch.py` -- Sim stack launch
- `ros2_ws/src/lite6_bringup/config/sim_params.yaml` -- Sim parameters
- `ros2_ws/src/lite6_control/lite6_control/mujoco_sim_node.py` -- Physics sim + control
- `ros2_ws/src/lite6_control/lite6_control/joint_publisher.py` -- Demo publisher
- `sim/models/mjcf/ufactory_lite6/scene.xml` -- MuJoCo world scene
- `sim/models/mjcf/ufactory_lite6/lite6.xml` -- MuJoCo robot model
- `sim/models/mjcf/ufactory_lite6/assets/` -- STL meshes for MuJoCo
