"""pi0 adapter (Physical Intelligence pi0, via LeRobot's PyTorch port).

pi0 pairs a PaliGemma-3B VLM with a ~300M flow-matching action expert. Given an
observation it integrates a learned velocity field from noise to a 50-step
action chunk over ``num_denoise_steps`` iterations.

Deployment notes for a 3090:

* ~7 GB at bf16 leaves plenty of headroom, so NF4 is rarely worth its latency
  cost here -- unlike OpenVLA, memory is not the binding constraint.
* The action expert is small and runs ``num_denoise_steps`` times, so its
  per-step launch overhead is amplified. This is where CUDA graphs pay off.
* A 50-step chunk at 50 Hz control covers a full second, which means a single
  inference can drive the arm for far longer than it takes to compute -- pair it
  with :class:`~vla_engine.control.chunker.ChunkExecutor` and the effective
  control rate is decoupled from policy latency entirely.
"""

from __future__ import annotations

from .lerobot_base import LeRobotFlowAdapter

__all__ = ["Pi0Adapter"]


class Pi0Adapter(LeRobotFlowAdapter):
    """Adapter for ``lerobot/pi0`` and pi0 fine-tunes.

    Recognized ``EngineConfig.extra`` keys:
        ``num_denoise_steps``: Flow-matching integration steps (default 10).
            The primary latency/quality dial. Values below ~5 visibly degrade
            action quality; above ~10 gains are marginal.
    """

    policy_import = "lerobot.common.policies.pi0.modeling_pi0:PI0Policy"
