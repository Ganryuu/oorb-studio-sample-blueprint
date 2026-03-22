#!/usr/bin/env python3
"""Benchmark: MuJoCo simulation step rate.

Measures how many physics steps per second the model can sustain.
Used by Blueprint CI and displayed as a badge on the listing page.
"""
import time
import json
import mujoco

MODEL_PATH = "sim/models/mjcf/robot_arm.xml"
NUM_STEPS = 10_000


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Warm up
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Benchmark
    mujoco.mj_resetData(model, data)
    start = time.perf_counter()
    for _ in range(NUM_STEPS):
        mujoco.mj_step(model, data)
    elapsed = time.perf_counter() - start

    steps_per_second = NUM_STEPS / elapsed

    result = {
        "metric": "steps_per_second",
        "value": round(steps_per_second, 1),
        "num_steps": NUM_STEPS,
        "elapsed_seconds": round(elapsed, 3),
        "model": MODEL_PATH,
        "timestep": model.opt.timestep,
        "realtime_factor": round(NUM_STEPS * model.opt.timestep / elapsed, 2),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
