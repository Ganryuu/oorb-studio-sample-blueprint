"""SmolVLA adapter (HuggingFace LeRobot).

SmolVLA is the small end of the family: a SmolVLM-2 backbone plus a
flow-matching action expert, ~450M parameters total. It was designed for
consumer hardware and community robot data, and it is the policy to reach for
when you want a VLA that leaves the GPU mostly free.

Deployment notes for a 3090:

* ~1 GB at bf16. Quantization is pointless; several replicas fit on one card.
* At this size the forward pass is dominated by kernel-launch overhead rather
  than arithmetic, so ``torch.compile(mode="reduce-overhead")`` and its CUDA
  graphs give a proportionally larger win here than on OpenVLA.
* The backbone expects square 512x512 inputs, so frames are letterboxed rather
  than stretched (see :func:`~vla_engine.runtime.imageproc.resize_with_padding`).
"""

from __future__ import annotations

from .lerobot_base import LeRobotFlowAdapter

__all__ = ["SmolVLAAdapter"]


class SmolVLAAdapter(LeRobotFlowAdapter):
    """Adapter for ``lerobot/smolvla_base`` and SmolVLA fine-tunes.

    Recognized ``EngineConfig.extra`` keys:
        ``num_denoise_steps``: Flow-matching integration steps (default 10).
    """

    policy_import = "lerobot.common.policies.smolvla.modeling_smolvla:SmolVLAPolicy"
