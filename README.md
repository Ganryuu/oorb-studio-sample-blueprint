# Welcome to OORB Studio

OORB Studio is a browser-based robotics development environment: ROS 2 + code editor + terminal + simulation + a ROS bridge for connecting the UI and external machines.

Getting started guide:
```text
https://oorb.io/guide
```

## What you can do here
- Build and run ROS 2 Jazzy packages in `ros2_ws/`
- Run MuJoCo demos and publish/subscribe ROS topics
- Connect external ROS machines (robots, VMs, laptops) through the Studio rosbridge WebSocket for live experiments and data
- Iterate quickly with code + simulation in one place

## What's included in the Studio UI
When you open a workspace, you land in the default exploded layout. It is meant to feel like a familiar robotics workflow: code, terminals, logs, and simulation side by side, with an agent that can help you move faster.

Default layout (exploded):
- Agent and plans: describe what you want to build or fix in plain language. The agent can propose a plan, apply changes, and you can review diffs before keeping them.
- Code editor and files: a VS Code-like editor in the browser for editing your repo directly.
- Terminals: run ROS nodes, build `ros2_ws/`, install dependencies, and iterate quickly.
- Logs and history: see what ran, what failed, and what changed across sessions.
- MuJoCo simulation: run the demo scene in the browser and connect it to ROS topics through the ROS Bridge.

ROS bridge:
- The MuJoCo tab can publish and subscribe to ROS topics via rosbridge WebSocket.
- You can also connect external machines (robot, VM, laptop) to the same workspace, so you can run experiments with live data and control loops without local setup.

Layouts and extra tools:
- You can switch layouts at any time from the layout controls in the UI.
- Expanded view: more space and access to additional tools, including:
  - CAD and 3D viewer, plus text-to-CAD for quick model iteration
  - Resource utilization, so you can see when a build or sim is saturating CPU or memory

GitHub and collaboration:
- Connect GitHub to import a repo into a workspace or push your workspace to GitHub when you are ready.
- Invite teammates to the workspace (team plan) to review changes and debug together.

## Workspace layout
- `README.md`: this guide (recommended to keep at workspace root)
- `ros2_ws/`: ROS 2 workspace (build with `colcon`)
- `sim/`: MuJoCo assets and models (`sim/models/mjcf/` is the default MJCF directory)
- `fab/`: CAD-related assets (if present in your workspace)
- `rl_models/`: training artifacts and logs
- `mujoco_min/`: minimal MuJoCo sanity checks
- `MUJOCO_LOG.TXT`: MuJoCo runtime logs



## Quick start
Each of these quick start scenarios and demo scripts is further detailed within each script

To get going:
Open 3 terminals in Studio.

Terminal 1 (ROS env):
```bash
source /opt/ros/jazzy/setup.bash
```

Terminal 2 (confirm ROS is alive):
```bash
ros2 node list
ros2 topic list | head
```

Terminal 3 (optional, for external publishing tests):
```bash
ros2 topic echo --once /client_count || true
```

## Build ROS packages (optional)
If you are working in `ros2_ws/`:
```bash
source /opt/ros/jazzy/setup.bash
cd /workspace/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## A sample ROS 2 package is already created for you with two demo nodes:

### 1. Simple Talker (Hello World)
```bash
# Source ROS 2 environment
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash

# Run the demo talker node
ros2 run my_bot_py talker
```

### 2. Robot Arm Simulator (MuJoCo Visualization)
This publishes joint states that drive the MuJoCo visualization in your browser:

```bash
# Source ROS 2 environment
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash

# Run the robot arm joint state publisher
ros2 run my_bot_py robot_arm

# The robot arm will now animate in the MuJoCo viewer in your browser!
# Joint states are published to /joint_states at 20 Hz
```

You can check the published topics:
```bash
ros2 topic list
ros2 topic echo /joint_states
```

## Environment Verification

Verify your ROS 2 and build tools are correctly installed:

```bash
# Check ROS 2 version
ros2 --version

# Check colcon version (note: colcon doesn't support --version flag)
pip3 show colcon-core | grep Version
# Or use the convenient alias:
colcon-version

# Check build tools
make --version
gcc --version
g++ --version
cmake --version

# List installed ROS 2 packages
ros2 pkg list
```

## Building Custom Packages

To build your ROS 2 workspace:

```bash
cd /workspace/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

Tips:
- Use `--symlink-install` to avoid rebuilding Python packages on every change
- Add `--cmake-args -DCMAKE_BUILD_TYPE=Release` for optimized builds
- Use `--packages-select <package_name>` to build specific packages only


## Demo scripts

### 1) `dummy_simulator.py`
What it does:
- Runs the Reacher task and publishes `/joint_states`
- Does not accept external control

Run:
```bash
source /opt/ros/jazzy/setup.bash
python3 dummy_simulator.py
```

Verify:
```bash
ros2 topic hz /joint_states
```

### 2) `dummy_simulator_controlled.py`
What it does:
- Same Reacher demo, but with two modes:
  - `DEMO_MODE=random`: policy/random actions
  - `DEMO_MODE=ros`: controlled by ROS topics
- Publishes `/joint_states`
- Subscribes to:
  - `/cmd_vel` (`geometry_msgs/Twist`) mapped to 2 action dims
  - `/reacher_action` (`std_msgs/Float64MultiArray`) for direct actions

Run (ROS-controlled mode):
```bash
source /opt/ros/jazzy/setup.bash
DEMO_MODE=ros python3 dummy_simulator_controlled.py
```

Send commands:
```bash
source /opt/ros/jazzy/setup.bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: -0.5}}'
```

### 3) `mujoco_custom_model_demo.py`
What it does:
- Loads a user-provided MJCF via `MODEL_PATH`
- Publishes `/joint_states` and `/mujoco_qpos`
- Subscribes to:
  - `/mujoco_ctrl` (`std_msgs/Float64MultiArray`) for direct actuator control
  - `/cmd_vel` (`geometry_msgs/Twist`) as a convenience mapping

Run with the default model:
```bash
source /opt/ros/jazzy/setup.bash
python3 mujoco_custom_model_demo.py
```

Run with your own model:
```bash
source /opt/ros/jazzy/setup.bash
MODEL_PATH=/workspace/sim/models/mjcf/your_model.xml python3 mujoco_custom_model_demo.py
```

Verify state:
```bash
ros2 topic echo /mujoco_qpos
```

### 4) `unitree_a1_ros_control.py` (recommended end-to-end ROS control demo)
What it does:
- Demonstrates a reliable ROS-to-MuJoCo control loop on Unitree A1 in OORB Studio
- Confirms model motion is driven by ROS commands, not just static model loading
- Publishes:
  - `/joint_states`
  - `/mujoco_qpos`
  - `/mujoco_ctrl_applied`
- Subscribes to:
  - `/cmd_vel` for high-level motion commands
  - `/mujoco_ctrl` for direct actuator commands

Where to find full instructions:
- Open the top section of `unitree_a1_ros_control.py`
- The script header contains the complete tutorial:
  - manual model upload checklist (for early Studio workflow)
  - exact 3-terminal commands to run
  - validation criteria to confirm control is working
  - adaptation notes for other MuJoCo Menagerie robots

Quick launch:
```bash
source /opt/ros/jazzy/setup.bash
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export MODEL_PATH=/workspace/mujoco_menagerie/unitree_a1/scene.xml
python3 /workspace/unitree_a1_ros_control.py
```

Note:
- MuJoCo Menagerie is expected to be available in workspaces so this flow can be reused for other robots beyond Unitree A1.


## Connect a real robot or external ROS machine (manual for now)
Today, you can connect any external ROS machine (robot, VM, laptop) to your Studio workspace using the built-in rosbridge WebSocket. For now this is manual. A first-class UI flow is planned.

Step 1 (get the rosbridge URL from the browser):
1) Open DevTools in the Studio workspace page
2) Network -> WS
3) Refresh the page
4) Click the WS request named `rosbridge-proxy`
5) Copy the Request URL, which looks like:
   `wss://api.oorb.io/api/v1/projects/<project_id>/container/rosbridge-proxy?token=<JWT>`

Important:
- Treat the token as sensitive
- Do not paste the token into tickets, docs, or commits

Step 2 (publish from the external machine):
On your external machine, export the URL:
```bash
export ROSBRIDGE_WSS='wss://api.oorb.io/api/v1/projects/<project_id>/container/rosbridge-proxy?token=<JWT>'
```
If you need a quick Python environment (common on Ubuntu due to PEP 668):
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install websockets
```
Then run a rosbridge publisher client (example: `rosbridge_external_publisher.py`):
```bash
python3 rosbridge_external_publisher.py --topic /cmd_vel --type geometry_msgs/Twist --rate 10 \
  --msg '{"linear":{"x":0.5,"y":0.0,"z":0.0},"angular":{"x":0.0,"y":0.0,"z":-0.5}}'
```

Step 3 (verify inside Studio):
```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo --once /client_count
ros2 topic echo --once /cmd_vel
```

Step 4 (end-to-end check with a sim):
- Run a node that subscribes to `/cmd_vel` and changes simulation state (for example `mujoco_custom_model_demo.py` or `dummy_simulator_controlled.py` in `DEMO_MODE=ros`)
- Keep the external publisher running
- Watch for state changes in:
  - the MuJoCo UI
  - `/mujoco_qpos` (if your node publishes it)

## Troubleshooting
- If ROS commands behave inconsistently across terminals, re-source:
  - `source /opt/ros/jazzy/setup.bash`
- If `/client_count` stays at 0, check that rosbridge is up:
  - `ss -ltnp | grep 9090 || true`
- If a model does not react to commands, check `nu` (number of actuators) in the script logs.
- If external rosbridge connects from the browser but fails from your robot/VM, it is often an Origin check. Use an Origin header of `https://oorb.io`.
- Can't find a ROS 2 package?
  Make sure you've sourced the environment: `source /opt/ros/jazzy/setup.bash`
  This is automatically added to your .bashrc



### If you run into any bugs, please report them [here](https://oorb.io/feedback), your feedback is more than essential, or reach out via: contact@oorb.io

### Join our [discord](https://discord.gg/7k5rarybq8): for news, feature discussions, and hanging out with fellow robotics nerds.

### Thank you!
