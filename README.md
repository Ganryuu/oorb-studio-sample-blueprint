# 4-DOF Robot Arm — OORB Blueprint

A 4-joint robot arm with ROS 2 position control and MuJoCo simulation.

## Open in OORB Studio

[![Open in OORB Studio](https://oorb.io/badge/open-in-studio.svg)](https://oorb.io/blueprints/4dof-robot-arm)

Click the button above, or go to [oorb.io/blueprints/4dof-robot-arm](https://oorb.io/blueprints/4dof-robot-arm).

## Run locally

```bash
git clone https://github.com/oorb-studio/blueprint-4dof-robot-arm.git
cd blueprint-4dof-robot-arm

# Option A: VS Code devcontainer (recommended)
code .
# → "Reopen in Container" → builds automatically

# Option B: Native ROS 2 Jazzy
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 run my_bot_py robot_arm
```

## Quick start

**Terminal 1** — run the demo:
```bash
ros2 run my_bot_py robot_arm
# Publishes /joint_states at 20 Hz with sinusoidal motion
```

**Terminal 2** — verify:
```bash
ros2 topic hz /joint_states     # expect ~20 Hz
ros2 topic echo /joint_states   # see the values
```

**Terminal 3** — run with physics (position control):
```bash
ros2 run my_bot_py mujoco_sim
# Then send commands:
ros2 topic pub /position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.5, 0.3, -0.8, 0.2]}"
```

## Project structure

```
.oorb/                      ← OORB Studio integration
├── blueprint.yaml          ← Manifest (deps, resources, sim, runbook)
├── listing.yaml            ← Marketplace page content (single source of truth)
├── agent.md                ← Agent context (the agent reads this automatically)
├── validate.yaml           ← Health checks
└── state/                  ← Workspace state (restored on import)
    ├── conversations/      ← Agent chat threads (JSONL, one file per thread)
    ├── logs/               ← Build/terminal/ROS output logs
    └── outputs/            ← Generated artifacts (plots, rosbags, data)
.devcontainer/              ← Local dev (VS Code)
├── Dockerfile              ← Replicates Studio's container
└── devcontainer.json
ros2_ws/src/
└── my_bot_py/              ← ROS 2 package
    ├── robot_arm_publisher.py  ← Demo sinusoidal motion
    ├── mujoco_sim_node.py      ← Physics sim + position control
    └── talker.py               ← Hello world
sim/models/mjcf/
├── robot_arm.xml           ← MuJoCo model (4 joints, PD actuators)
├── pendulum.xml            ← Simple pendulum demo
└── reacher.xml             ← 2-DOF reacher demo
docs/                       ← Marketplace content
├── ABOUT.md                ← Parsed for the catalog detail page
├── bom.csv                 ← Bill of materials
└── images/                 ← Screenshots, photos, diagrams
fab/                        ← CAD assets & uploads
├── uploads/                ← Studio upload landing zone (user-uploaded files)
├── step/                   ← STEP files for manufacturing
└── stl/                    ← STL files for 3D printing
```

## ROS topics

| Topic | Type | Dir | Purpose |
|---|---|---|---|
| `/joint_states` | sensor_msgs/JointState | PUB | Joint positions (4 joints @ 20 Hz) |
| `/position_controller/commands` | std_msgs/Float64MultiArray | SUB | Target positions (4 floats) |
| `/chatter` | std_msgs/String | PUB | Heartbeat |

## License

MIT
