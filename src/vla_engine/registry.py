"""Model registry: aliases, default checkpoints, and I/O specs.

The registry answers questions *before* any weights are downloaded -- what
cameras does pi0 want, does OpenVLA need proprioception, will a 7B policy fit
on a 24 GB card at bf16. That lets the CLI, the ROS 2 node, and the server
validate a deployment without a GPU present.

Adapter classes are referenced by import path and resolved lazily, so importing
this module never pulls in torch.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass, field

from .errors import ModelNotFoundError
from .types import PolicySpec

__all__ = ["ModelCard", "register", "get_card", "list_models", "resolve", "REGISTRY"]


@dataclass(frozen=True)
class ModelCard:
    """Everything known about a policy family without loading it.

    Attributes:
        key: Canonical registry name.
        adapter: ``"module:ClassName"`` import path, resolved on demand.
        default_checkpoint: HuggingFace repo id used when none is given.
        spec: Input/output contract.
        params_b: Parameter count in billions, for memory estimates.
        default_unnorm_key: Dataset statistics key, where the family uses one.
        notes: Human-readable deployment guidance.
    """

    key: str
    adapter: str
    spec: PolicySpec
    default_checkpoint: str | None = None
    aliases: tuple[str, ...] = ()
    params_b: float = 0.0
    default_unnorm_key: str | None = None
    #: False for weightless policies (the echo reference) that load nothing.
    requires_checkpoint: bool = True
    license: str = "see model card"
    notes: str = ""
    hf_families: tuple[str, ...] = field(default_factory=tuple)

    def memory_gb(self, dtype: str = "bf16", quantization: str | None = None) -> float:
        """Rough weight-memory estimate in GB.

        Weights only, plus a 15% allowance for activations, the KV cache, and
        the CUDA context. Good enough to answer "does this fit on a 3090",
        which is the only question it is used for.
        """
        bytes_per_param = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0}.get(dtype, 2.0)
        if quantization == "nf4":
            bytes_per_param = 0.55  # 4-bit weights + per-block scales
        elif quantization in ("int8", "awq", "gptq"):
            bytes_per_param = 1.05
        return self.params_b * bytes_per_param * 1.15

    def fits_on(
        self, memory_gb: float, dtype: str = "bf16", quantization: str | None = None
    ) -> bool:
        return self.memory_gb(dtype, quantization) <= memory_gb

    def load_adapter_class(self) -> type:
        """Import and return the adapter class (imports torch transitively)."""
        module_path, _, class_name = self.adapter.partition(":")
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


REGISTRY: dict[str, ModelCard] = {}
_ALIASES: dict[str, str] = {}


def register(card: ModelCard) -> ModelCard:
    """Add a card to the registry. Later registrations override earlier ones."""
    REGISTRY[card.key] = card
    _ALIASES[card.key] = card.key
    for alias in card.aliases:
        _ALIASES[alias.lower()] = card.key
    return card


def get_card(name: str) -> ModelCard:
    """Resolve an alias, HuggingFace repo id, or canonical key to a card.

    Raises:
        ModelNotFoundError: with the list of known models, since a typo here is
            otherwise reported as a confusing download failure.
    """
    if not name:
        raise ModelNotFoundError("no model name given")
    key = name.strip()
    lowered = key.lower()
    if lowered in _ALIASES:
        return REGISTRY[_ALIASES[lowered]]
    # HuggingFace repo id -> match against known families, e.g.
    # "openvla/openvla-7b-finetuned-libero" -> the openvla card.
    for card in REGISTRY.values():
        for family in card.hf_families:
            if family in lowered:
                return card
    raise ModelNotFoundError(
        f"unknown model {name!r}. Known models: {sorted(REGISTRY)}. "
        "For a fine-tuned checkpoint, pass the base family as `model` and the "
        "checkpoint path as `checkpoint`."
    )


def resolve(name: str, checkpoint: str | None = None) -> tuple[ModelCard, str | None]:
    """Return the card plus the concrete checkpoint to load.

    The checkpoint is ``None`` only for weightless policies. Real policies
    without a resolvable checkpoint raise, since the failure would otherwise
    surface much later as a confusing download error.
    """
    card = get_card(name)
    resolved = checkpoint or (name if "/" in name else None) or card.default_checkpoint
    if not resolved and card.requires_checkpoint:
        raise ModelNotFoundError(
            f"model {card.key!r} has no default checkpoint; pass one explicitly"
        )
    return card, resolved


def list_models() -> list[ModelCard]:
    return sorted(REGISTRY.values(), key=lambda c: c.key)


def __iter__() -> Iterator[ModelCard]:  # pragma: no cover
    return iter(list_models())


# --------------------------------------------------------------------------
# Built-in policies
# --------------------------------------------------------------------------

register(
    ModelCard(
        key="openvla",
        adapter="vla_engine.adapters.openvla:OpenVLAAdapter",
        default_checkpoint="openvla/openvla-7b",
        aliases=("openvla-7b", "openvla_7b"),
        hf_families=("openvla",),
        params_b=7.6,
        default_unnorm_key="bridge_orig",
        license="MIT (weights: see model card)",
        spec=PolicySpec(
            name="openvla",
            action_dim=7,  # 6-DoF end-effector delta + gripper
            horizon=1,  # single-step: no action chunking
            cameras=("primary",),
            requires_state=False,
            language_conditioned=True,
            image_size=(224, 224),
            control_hz=5.0,
        ),
        notes=(
            "Prismatic VLM: Llama-2-7B decoder over a fused DINOv2+SigLIP vision "
            "tower. Actions are emitted as 7 discrete tokens drawn from the 256 "
            "least-used entries of the Llama vocabulary, then un-normalized with "
            "per-dataset q01/q99 statistics selected by `unnorm_key`. At bf16 the "
            "weights are ~15 GB, ~17.5 GB once activations and the KV cache are "
            "counted, so it fits a 24 GB 3090 with room to spare; nf4 brings that "
            "to ~4.8 GB. Single-step output means the control loop leans on the "
            "chunk executor to hold between predictions."
        ),
    )
)

register(
    ModelCard(
        key="pi0",
        adapter="vla_engine.adapters.pi0:Pi0Adapter",
        default_checkpoint="lerobot/pi0",
        aliases=("pi-0", "pi_0", "pizero", "pi0-base"),
        hf_families=("pi0", "pi-0"),
        params_b=3.3,
        license="Apache-2.0 (weights: see model card)",
        spec=PolicySpec(
            name="pi0",
            action_dim=32,  # padded action space; slice to the robot's DoF
            horizon=50,
            cameras=("primary", "wrist"),
            state_dim=32,
            requires_state=True,
            language_conditioned=True,
            image_size=(224, 224),
            control_hz=50.0,
        ),
        notes=(
            "PaliGemma-3B backbone with a ~300M flow-matching action expert. "
            "Predicts a 50-step chunk by integrating an ODE over N denoising "
            "steps, so latency scales with `num_denoise_steps` (10 by default) -- "
            "that knob is the main latency/quality dial, more so than precision. "
            "Comfortable on one 3090 at bf16 (~7.6 GB all-in)."
        ),
    )
)

register(
    ModelCard(
        key="smolvla",
        adapter="vla_engine.adapters.smolvla:SmolVLAAdapter",
        default_checkpoint="lerobot/smolvla_base",
        aliases=("smolvla-base", "smol-vla"),
        hf_families=("smolvla",),
        params_b=0.45,
        license="Apache-2.0",
        spec=PolicySpec(
            name="smolvla",
            action_dim=32,
            horizon=50,
            cameras=("primary", "wrist"),
            state_dim=32,
            requires_state=True,
            language_conditioned=True,
            image_size=(512, 512),
            control_hz=30.0,
        ),
        notes=(
            "SmolVLM-2 backbone plus a flow-matching action expert, ~450M "
            "parameters total. Small enough that latency is dominated by kernel "
            "launches rather than FLOPs, which makes it the biggest beneficiary "
            "of CUDA graphs here. Leaves most of a 3090 free, so it is the natural "
            "choice when running several policies or environments per card."
        ),
    )
)

register(
    ModelCard(
        key="echo",
        adapter="vla_engine.adapters.echo:EchoAdapter",
        default_checkpoint=None,
        aliases=("mock", "dummy", "test"),
        params_b=0.0,
        requires_checkpoint=False,
        license="n/a",
        spec=PolicySpec(
            name="echo",
            action_dim=7,
            horizon=1,
            cameras=("primary",),
            requires_state=False,
            language_conditioned=False,
            image_size=(224, 224),
            control_hz=10.0,
        ),
        notes=(
            "Weightless reference policy. Emits deterministic actions derived "
            "from a hash of the observation and can simulate a configurable "
            "forward-pass latency, so the server, batcher, replica router, chunk "
            "executor, and ROS 2 node can all be exercised end to end -- in CI or "
            "on a laptop -- without a GPU or a checkpoint."
        ),
    )
)
