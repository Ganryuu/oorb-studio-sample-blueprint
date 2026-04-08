#!/usr/bin/env python3
"""Benchmark: MuJoCo simulation step rate for UFactory Lite 6."""
import time
import mujoco

MODEL_PATH = "/workspace/sim/models/mjcf/ufactory_lite6/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

N = 10000
t0 = time.perf_counter()
for _ in range(N):
    mujoco.mj_step(model, data)
elapsed = time.perf_counter() - t0

rate = N / elapsed
print(f"steps_per_second: {rate:.0f}")
print(f"total_time: {elapsed:.3f}s for {N} steps")
