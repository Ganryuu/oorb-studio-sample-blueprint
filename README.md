# 4-DOF Robot Arm — OORB Blueprint

A 4-joint robot arm with ROS 2 position control and MuJoCo simulation.
URDF-first design with sim/hardware profiles, safety constraints, and CI.

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
ros2 launch my_bot_bringup sim.launch.py
```

## Quick start

**Terminal 1** — launch the sim stack:
```bash
ros2 launch my_bot_bringup sim.launch.py
# Starts: robot_state_publisher + mujoco_sim + rosbridge
```

**Terminal 2** — verify:
```bash
ros2 topic hz /joint_states     # expect ~50 Hz
ros2 topic echo /joint_states   # see the values
```

**Terminal 3** — send position commands:
```bash
ros2 topic pub /position_controller/commands \
  std_msgs/msg/Float64MultiArray "{data: [0.5, 0.3, -0.8, 0.2]}"
```

## Project structure

```
.oorb/                          ← OORB Studio integration
├── blueprint.yaml              ← Manifest (packages, interfaces, profiles, safety, CI)
├── listing.yaml                ← Marketplace page content (single source of truth)
├── agent.md                    ← Agent context (injected into agent on workspace open)
├── validate.yaml               ← Health checks (build, URDF, MuJoCo, ROS nodes)
└── state/                      ← Workspace state (restored on import)
    ├── conversations/          ← Agent chat threads (JSONL)
    ├── logs/                   ← Build/terminal/ROS output logs
    ├── outputs/                ← Generated artifacts
    └── bags/                   ← Recorded rosbags (mcap)
.devcontainer/                  ← Local dev (VS Code)
├── Dockerfile                  ← Replicates Studio's container
└── devcontainer.json
.github/workflows/
└── blueprint-ci.yaml           ← CI pipeline (build + test + validate)
ros2_ws/src/
├── my_bot_description/         ← Robot model (URDF source of truth)
│   ├── urdf/robot_arm.urdf.xacro  ← 4-DOF arm (parametric XACRO)
│   ├── meshes/                 ← Visual/collision meshes
│   └── launch/                 ← robot_state_publisher launch
├── my_bot_bringup/             ← Launch files + parameter configs
│   ├── launch/
│   │   ├── sim.launch.py       ← Sim profile (MuJoCo + rosbridge)
│   │   └── hardware.launch.py  ← Hardware profile (template)
│   └── config/
│       ├── sim_params.yaml     ← Sim-tuned parameters
│       └── hardware_params.yaml← Hardware parameters (template)
└── my_bot_py/                  ← Application nodes
    ├── robot_arm_publisher.py  ← Demo sinusoidal motion
    ├── mujoco_sim_node.py      ← Physics sim + position control
    └── talker.py               ← Hello world
sim/models/mjcf/
├── robot_arm.xml               ← MuJoCo model (4 joints, PD actuators)
├── pendulum.xml                ← Simple pendulum demo
└── reacher.xml                 ← 2-DOF reacher demo
scripts/
├── benchmark_sim.py            ← Sim step rate benchmark
└── benchmark_control.py        ← Control loop latency benchmark
docs/                           ← Marketplace content
├── ABOUT.md                    ← Parsed for the catalog detail page
├── bom.csv                     ← Bill of materials
└── images/                     ← Screenshots, photos, diagrams
fab/                            ← CAD assets & uploads
├── uploads/                    ← Studio upload landing zone
├── step/                       ← STEP files for manufacturing
└── stl/                        ← STL files for 3D printing
```

## ROS interfaces

| Topic | Type | Dir | Hz | Purpose |
|---|---|---|---|---|
| `/joint_states` | sensor_msgs/JointState | PUB | 20-50 | Joint positions (4 joints) |
| `/position_controller/commands` | std_msgs/Float64MultiArray | SUB | — | Target positions (4 floats) |
| `/robot_description` | std_msgs/String | PUB | latch | URDF XML |
| `/tf` | tf2_msgs/TFMessage | PUB | 20-50 | Transform tree |
| `/e_stop` | std_msgs/Bool | SUB | — | Emergency stop |
| `/chatter` | std_msgs/String | PUB | 2 | Heartbeat |

## Profiles

| Profile | Launch | Hardware | Description |
|---|---|---|---|
| **sim** | `sim.launch.py` | No | MuJoCo simulation — default |
| **hardware** | `hardware.launch.py` | Yes | Physical robot (template) |

## Safety

- Joint position/velocity limits enforced per joint (see `blueprint.yaml > safety`)
- Cartesian workspace bounds: ±0.6m XY, 0-0.9m Z
- Emergency stop: publish `True` to `/e_stop`

## Benchmarks

| Metric | Command | Expected |
|---|---|---|
| Sim step rate | `python3 scripts/benchmark_sim.py` | >500 steps/sec |
| Control latency (p99) | `python3 scripts/benchmark_control.py` | <20ms |

## License

MIT
