#!/usr/bin/env python3
"""Benchmark: Control loop latency.

Measures the time from setting a position command to completing one
physics step + reading back joint states. Simulates the real control
loop latency users would experience.

Used by Blueprint CI and displayed as a badge on the listing page.
"""
import time
import json
import numpy as np
import mujoco

MODEL_PATH = "sim/models/mjcf/robot_arm.xml"
NUM_ITERATIONS = 1000


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    latencies_ms = []
    rng = np.random.default_rng(42)

    for _ in range(NUM_ITERATIONS):
        # Generate random target positions within joint limits
        target = rng.uniform(-1.0, 1.0, size=model.nu)

        start = time.perf_counter()

        # Set actuator controls (position commands)
        data.ctrl[:] = target

        # Step physics
        mujoco.mj_step(model, data)

        # Read back joint positions
        _ = data.qpos[:model.nq].copy()

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    latencies = np.array(latencies_ms)

    result = {
        "metric": "p99_ms",
        "value": round(float(np.percentile(latencies, 99)), 3),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "mean_ms": round(float(np.mean(latencies)), 3),
        "max_ms": round(float(np.max(latencies)), 3),
        "num_iterations": NUM_ITERATIONS,
        "model": MODEL_PATH,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
