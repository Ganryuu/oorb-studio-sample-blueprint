# vla-engine

An inference engine for vision-language-action models: **OpenVLA**, **pi0**, and **SmolVLA**
behind one interface, tuned for low-latency robot control.

Three ways to use it — a Python API, an HTTP/websocket server, and a ROS 2 node — over one
optimized core. Built for a dual RTX 3090 workstation, and honest about what that hardware
can and cannot do.

```python
from vla_engine import VLAEngine

engine = VLAEngine.from_pretrained("openvla", unnorm_key="bridge_orig")
chunk = engine.act(camera_frame, "pick up the red block")
print(chunk.actions)          # (1, 7) — 6-DoF delta + gripper, in robot units
```

---

## Why this exists

Running a VLA is easy. Running one *fast enough to close a control loop* is not, and the
gap is mostly plumbing:

- OpenVLA-7B emits an action as **seven sequential forward passes** over a 7B model. It is
  bound by kernel-launch overhead, not arithmetic — so the wins come from CUDA graphs and a
  static KV cache, not from a bigger GPU.
- pi0 and SmolVLA predict a **50-step action chunk** by integrating a flow-matching ODE.
  Latency scales with the denoising step count, which almost nobody exposes as a tunable.
- Every VLA is far slower than a robot's control rate. Without chunk scheduling, your
  control period *equals* your inference latency, and the arm stutters.
- The GPU is usually not on the robot, so observations cross a network every control step.
  Sent naively, pixels alone saturate a wifi link.

This engine addresses each of those directly, and ships a benchmark harness so the claims
are checkable on your own hardware.

---

## Supported models

| Model | Params | bf16 | NF4 | Action | Horizon | Cameras | State |
|---|---:|---:|---:|---:|---:|---|---|
| `openvla` | 7.6B | 17.5 GB | 4.8 GB | 7 | 1 | 1 | no |
| `pi0` | 3.3B | 7.6 GB | 2.1 GB | 32 | 50 | 2 | yes |
| `smolvla` | 0.45B | 1.0 GB | 0.3 GB | 32 | 50 | 2 | yes |
| `echo` | — | — | — | 7 | 1 | 1 | no |

Memory figures include a 15% allowance for activations, KV cache, and CUDA context. All
three fit on **one** 24 GB 3090 at bf16.

`echo` is a weightless reference policy with deterministic, observation-dependent output and
configurable simulated latency. It exists so the server, batcher, replica router, chunk
executor, and ROS 2 node can be exercised end to end — in CI or on a laptop — with no GPU and
no checkpoint. The test suite runs entirely on it.

```bash
vla models          # the table above, from the registry
vla info openvla    # inputs, outputs, memory, and deployment notes
```

---

## Install

```bash
git clone <this repo> && cd <this repo>
pip install -e .                    # core: numpy only, imports anywhere
```

Add what you need:

```bash
pip install -e '.[cuda]'      # torch + transformers + accelerate  → OpenVLA
pip install -e '.[lerobot]'   # LeRobot policies                   → pi0, SmolVLA
pip install -e '.[quant]'     # bitsandbytes                       → nf4 / int8
pip install -e '.[serve]'     # FastAPI + uvicorn                  → inference server
pip install -e '.[dev]'       # pytest + ruff
```

FlashAttention-2 needs a compiler and is worth installing on Ampere:

```bash
pip install flash-attn --no-build-isolation
```

Check what the machine can actually do:

```bash
vla doctor
```

```
GPU 0      cuda:0 NVIDIA GeForce RTX 3090 (Ampere, sm_86, 24 GB)
           bf16=True fp8=False tf32=True flash-attn-2=True
GPU 1      cuda:1 NVIDIA GeForce RTX 3090 (Ampere, sm_86, 24 GB)
           2 GPUs, homogeneous
```

---

## Quick start

### Python

```python
from vla_engine import VLAEngine, Observation

engine = VLAEngine.from_pretrained("pi0", dtype="bf16")

obs = Observation(
    images={"primary": scene_frame, "wrist": wrist_frame},
    instruction="put the spoon in the drawer",
    state=joint_positions,          # 7 values; padded to 32 internally
)
chunk = engine.predict(obs)         # (50, 32) — one second of motion at 50 Hz
```

### Driving a control loop

A VLA cannot keep up with a robot. Feed its chunks to a `ChunkExecutor`, which serves one
action per control tick and blends overlapping predictions so replans do not produce jumps:

```python
executor = engine.make_executor()   # temporal ensembling, gripper dim kept discrete

while running:
    if executor.needs_replan():
        executor.submit(engine.predict(current_observation()))
    action = executor.step()        # call at your control rate, not the policy's
    robot.command(action)
```

`executor.stats()` reports `starvation_rate` — the fraction of ticks served by holding the
last command. Anything above zero under steady load means inference is behind.

### Server

On the GPU workstation:

```bash
vla serve openvla --devices cuda:0,cuda:1 --unnorm-key bridge_orig --port 8000
```

On the robot:

```python
from vla_engine.serve.client import VLAClient

with VLAClient("ws://workstation:8000") as client:   # ws:// keeps one connection open
    chunk = client.act(frame, "pick up the red block")
```

`VLAClient` mirrors `VLAEngine`'s interface, so code moves between local and remote
inference by swapping the object.

Endpoints: `POST /predict`, `WS /stream`, `GET /healthz`, `GET /info`, `GET /metrics`.

### ROS 2

```bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash

ros2 launch vla_ros policy.launch.py \
    model:=openvla instruction:="pick up the red block" control_hz:=20.0
```

| Direction | Topic | Type |
|---|---|---|
| in | `/vla/image` | `sensor_msgs/Image` |
| in | `/vla/instruction` | `std_msgs/String` |
| in | `/joint_states` | `sensor_msgs/JointState` |
| out | `/vla/action` | `std_msgs/Float64MultiArray` |
| out | `/vla/chunk` | `std_msgs/Float64MultiArray` |
| out | `/vla/status` | `std_msgs/String` (JSON) |

Inference runs on a background thread against the newest frame; a timer publishes blended
actions at `control_hz`. The two rates are independent by design — running inference inside
the control timer would make the control period equal the inference latency.

No GPU in your container? Point the node at a remote server:

```bash
ros2 launch vla_ros policy.launch.py model:=pi0 remote_url:=ws://workstation:8000
```

Try it with no camera and no robot:

```bash
ros2 run vla_ros demo_publisher     # synthetic scene + instruction + joint states
ros2 run vla_ros policy --ros-args -p model:=echo
ros2 topic hz /vla/action
ros2 topic echo /vla/status
```

---

## Optimization guide

### What a 3090 can and cannot do

The RTX 3090 is Ampere, **sm_86**. That fixes several choices, and the engine encodes them
rather than leaving them to guesswork:

| Feature | Available | Consequence |
|---|---|---|
| bf16 tensor cores | yes (sm_80+) | default compute dtype |
| TF32 | yes | enabled by default; free speedup on residual fp32 matmuls |
| FlashAttention-2 | yes (sm_80+) | preferred attention backend |
| **FP8 tensor cores** | **no** (needs sm_89+) | fp8 is not an option on this card |
| P2P / NVLink DMA | disabled on GeForce | **do not** split one model across both cards |

`plan_precision()` resolves your request against the actual device and records *why* in
`info()["precision_notes"]`, so a downgrade is visible instead of silent.

### The optimizations, in order of impact

**1. Replace `generate()` with a fixed-trip static-cache decoder** (OpenVLA)

OpenVLA emits exactly 7 tokens greedily. `generate()` pays for machinery this problem does
not have: logit processors over a 32k vocabulary, a growing KV cache whose changing shapes
force recompilation and prevent CUDA graph capture, and a **device-to-host sync per token**
to test stopping conditions — seven full pipeline stalls. `runtime/decode.py` uses a
preallocated `StaticCache`, a fixed trip count, and keeps sampled ids on the GPU until the
loop ends.

**2. CUDA graphs via `torch.compile(mode="reduce-overhead")`**

A VLA control step is launch-bound, not compute-bound. Collapsing thousands of small kernel
launches into one graph replay is typically the single largest latency win here. It needs
static shapes — which is exactly what (1) provides. Enabled by default; warmup runs at load
so the robot's first control step does not pay for compilation.

**3. bf16 + TF32 + FlashAttention-2** — all native on sm_86.

**4. Denoising steps** (pi0, SmolVLA)

The VLM backbone runs once per observation; the action expert runs once per denoising step.
Halving the steps nearly halves latency. This is a bigger dial than precision:

```python
VLAEngine.from_pretrained("pi0", extra={"num_denoise_steps": 5})
```

**5. NF4 quantization** — opt-in, and a trade

Cuts OpenVLA from ~17.5 GB to ~4.8 GB, which is what lets one 3090 host it alongside another
policy. Dequantization is not free, so it *costs* batch-1 latency. Use it for capacity, not
for speed.

**6. Upload images as uint8** — cast and permute on the GPU, not in numpy. A 3-camera 512×512
observation is 2.3 MB instead of 9 MB across PCIe, every single inference.

### Using both 3090s

The instinct is to split one model across both cards. **Don't.** GeForce drivers disable
peer-to-peer DMA, so tensor- or pipeline-parallel traffic round-trips through host memory —
strictly worse than not splitting. Every supported policy fits on one card, so the engine
runs **one full replica per GPU** and routes each request to an idle one:

```bash
vla serve openvla --devices cuda:0,cuda:1
```

Near-linear throughput scaling, and single-request latency untouched, because no request
ever crosses a device boundary.

### Batching

Off by default: it trades latency for throughput, and one robot wants latency. Turn it on
when several robots or environments share a GPU:

```bash
vla serve smolvla --max-batch-size 8
```

The batcher waits up to `max_wait_ms` (default 2 ms, noise against a 30-100 ms forward) for
companions before firing.

### Measuring

Do not trust any of the above without measuring it on your hardware:

```bash
vla bench openvla --compare --batch-sizes 1,2,4
```

```
eager bf16                   p50   ...  p95   ...  p99   ...    ... Hz
compiled bf16                p50   ...  p95   ...  p99   ...    ... Hz
compiled nf4                 p50   ...  p95   ...  p99   ...    ... Hz
```

The harness reports **p50/p95/p99, not the mean** — a control loop is hurt by its worst
steps, not its average one. It discards warmup iterations (a first compiled call can be 100×
slower than steady state), synchronizes CUDA before stopping the clock, and pauses GC during
measurement.

**This repository ships no GPU benchmark numbers.** It was built and tested on a machine
without a CUDA device, and quoting latencies measured elsewhere — or estimated — would be
worse than quoting none. Run `vla bench` and get numbers for your own cards.

What *was* measured here, on CPU, is in the test suite: JPEG transport is 36× smaller than
raw for a two-camera stream (0.88 vs 32 Mbit/s at 10 Hz), and eight concurrent requests
coalesce into a single batched forward pass.

---

## Architecture

```
              VLAEngine ── ReplicaPool ── one adapter per GPU
                  │
   ┌──────────────┼──────────────┐
Python API    serve/ (HTTP+WS)  ros/ (ROS 2 node)
                  │
              adapters/
     openvla ──────┴────── lerobot_base ── pi0, smolvla
        │                       │
   runtime/decode          runtime/imageproc
   (static KV cache)       (GPU preprocessing)
        └──────── runtime/ (device, precision, compile) ───────┘
                            │
                        control/ (chunk scheduling, normalization)
```

| Module | Responsibility |
|---|---|
| `types.py` | `Observation`, `ActionChunk` — numpy only, never torch |
| `config.py` | Validated configuration; rejects inconsistent combinations up front |
| `registry.py` | Model cards: specs and memory estimates *without* loading weights |
| `runtime/device.py` | GPU probing, capability gates, replica placement |
| `runtime/precision.py` | Resolves dtype/attention/quantization against real hardware |
| `runtime/decode.py` | Fixed-trip greedy decode over a static KV cache |
| `control/chunker.py` | Decouples control rate from inference rate |
| `control/normalize.py` | q01/q99 and mean/std action statistics |
| `optim/replicas.py` | One replica per GPU, least-loaded routing |
| `optim/batching.py` | Continuous batching for shared GPUs |
| `serve/protocol.py` | JPEG-based wire format |

Two invariants hold throughout:

- **`import vla_engine` never imports torch.** The types, config, registry, control loop,
  and wire protocol work on a robot or CI runner with no GPU. Torch is imported when an
  adapter actually loads.
- **Optimizations never become correctness dependencies.** If `torch.compile` fails,
  FlashAttention-2 is missing, or `StaticCache` is unavailable, the engine logs the reason
  and falls back. A slow policy beats a policy that will not run.

---

## Adding a model

```python
from vla_engine.registry import ModelCard, register
from vla_engine.types import PolicySpec

register(ModelCard(
    key="my_vla",
    adapter="my_pkg.adapter:MyVLAAdapter",
    default_checkpoint="my-org/my-vla",
    params_b=2.0,
    spec=PolicySpec(name="my_vla", action_dim=7, horizon=16, requires_state=True),
))
```

The adapter implements two methods — `_load()` and `_predict_batch()` returning
`[B, horizon, action_dim]` in robot units. `VLAAdapter` handles validation, timing, warmup,
device placement, and horizon truncation. For a LeRobot-style flow-matching policy, subclass
`LeRobotFlowAdapter` and set `policy_import`.

---

## Development

```bash
pip install -e '.[dev,serve,client]'
pytest              # 139 tests, ~1s, no GPU required
ruff check src tests
```

The suite covers action detokenization against the reference bin mapping, KV-cache
scheduling, chunk blending, ROS message conversion (including channel order and padded row
stride), the wire protocol, and a live uvicorn server driven by the real client. Tests
needing hardware or weights are marked `gpu` and `weights` and are not part of the default
run.

---

## Notes and limitations

- **`unnorm_key` matters.** OpenVLA un-normalizes actions with per-dataset statistics. The
  wrong key yields well-formed actions at the wrong scale — the arm moves smoothly and
  incorrectly. If a checkpoint offers several and you name none, the engine refuses rather
  than guessing.
- **Model weights carry their own licenses.** Check each model card before deploying.
- **The real adapters are untested against live weights here** — this machine has no GPU and
  no checkpoints. Their structure is exercised by the test suite; their numerics are not.
  Validate against your robot in a safe configuration before trusting them.
- **Actions are commands, not guarantees.** Nothing here enforces joint limits, velocity
  limits, or workspace bounds. Keep whatever safety layer you already have between this and
  the hardware.
