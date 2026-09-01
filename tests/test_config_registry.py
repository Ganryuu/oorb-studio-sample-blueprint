"""Configuration validation, hardware planning, and the model registry."""

from __future__ import annotations

import pytest

from vla_engine import EngineConfig
from vla_engine.config import PrecisionConfig
from vla_engine.errors import ConfigError, ModelNotFoundError
from vla_engine.registry import get_card, list_models, resolve
from vla_engine.runtime.device import DeviceInfo, GpuTopology
from vla_engine.runtime.precision import plan_precision

RTX_3090 = DeviceInfo(0, "NVIDIA GeForce RTX 3090", "cuda", 24.0, (8, 6), 82)
RTX_2080TI = DeviceInfo(0, "NVIDIA GeForce RTX 2080 Ti", "cuda", 11.0, (7, 5), 68)
CPU = DeviceInfo(0, "cpu", "cpu")


class TestConfig:
    def test_rejects_unknown_keys(self):
        with pytest.raises(ConfigError, match="unknown"):
            EngineConfig.from_dict({"model": "echo", "typo": 1})

    def test_rejects_quantization_with_fp32(self):
        with pytest.raises(ConfigError, match="half-precision"):
            EngineConfig(model="x", precision={"dtype": "fp32", "quantization": "nf4"})

    def test_accepts_nested_mappings(self):
        config = EngineConfig(model="x", precision={"dtype": "bf16"})
        assert isinstance(config.precision, PrecisionConfig)

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("VLA_MODEL", "pi0")
        monkeypatch.setenv("VLA_DEVICES", "cuda:0,cuda:1")
        monkeypatch.setenv("VLA_HORIZON", "25")
        monkeypatch.setenv("VLA_COMPILE", "0")
        config = EngineConfig.from_env()
        assert config.model == "pi0"
        assert config.devices == ["cuda:0", "cuda:1"]
        assert config.horizon == 25
        assert config.compile.enabled is False

    def test_cudagraphs_only_with_reduce_overhead(self):
        assert EngineConfig(model="x", compile={"mode": "reduce-overhead"}).compile.uses_cudagraphs
        assert not EngineConfig(model="x", compile={"mode": "default"}).compile.uses_cudagraphs


class TestPrecisionPlanning:
    def test_ampere_gets_bf16_and_tf32(self):
        plan = plan_precision(PrecisionConfig(dtype="auto"), RTX_3090)
        assert plan.dtype == "bf16"
        assert plan.tf32 is True

    def test_turing_downgrades_bf16_to_fp16_with_a_reason(self):
        plan = plan_precision(PrecisionConfig(dtype="bf16"), RTX_2080TI)
        assert plan.dtype == "fp16"
        assert any("bf16" in note for note in plan.notes)

    def test_turing_does_not_enable_tf32(self):
        assert plan_precision(PrecisionConfig(dtype="auto"), RTX_2080TI).tf32 is False

    def test_cpu_falls_back_to_fp32(self):
        assert plan_precision(PrecisionConfig(dtype="bf16"), CPU).dtype == "fp32"

    def test_strict_mode_raises_instead_of_downgrading(self):
        from vla_engine.errors import DependencyError

        with pytest.raises(DependencyError):
            plan_precision(PrecisionConfig(dtype="bf16"), RTX_2080TI, strict=True)

    def test_quantization_dropped_without_cuda(self):
        plan = plan_precision(PrecisionConfig(dtype="auto", quantization="nf4"), CPU)
        assert plan.quantization is None

    def test_3090_has_no_fp8(self):
        """sm_86 predates FP8 tensor cores; claiming otherwise would mislead tuning."""
        assert RTX_3090.supports_fp8 is False
        assert RTX_3090.supports_bf16 is True
        assert RTX_3090.supports_flash_attention_2 is True


class TestTopology:
    def test_dual_3090_yields_two_replicas(self):
        topology = GpuTopology(
            [RTX_3090, DeviceInfo(1, "NVIDIA GeForce RTX 3090", "cuda", 24.0, (8, 6), 82)]
        )
        assert topology.replica_devices() == ["cuda:0", "cuda:1"]
        assert topology.homogeneous is True

    def test_mixed_gpus_are_flagged(self):
        topology = GpuTopology([RTX_3090, RTX_2080TI])
        assert topology.homogeneous is False

    def test_capacity_check(self):
        topology = GpuTopology([RTX_3090])
        assert topology.fits(17.5) is True
        assert topology.fits(30.0) is False

    def test_unknown_requested_device_raises(self):
        with pytest.raises(ConfigError, match="not visible"):
            GpuTopology([RTX_3090]).replica_devices(["cuda:7"])


class TestRegistry:
    def test_all_supported_policies_fit_one_3090_at_bf16(self):
        for card in list_models():
            assert card.fits_on(24.0, "bf16"), f"{card.key} does not fit"

    def test_openvla_needs_quantization_to_fit_12gb(self):
        card = get_card("openvla")
        assert card.fits_on(12.0, "bf16") is False
        assert card.fits_on(12.0, "bf16", "nf4") is True

    def test_resolves_aliases(self):
        assert get_card("pi-0").key == "pi0"
        assert get_card("MOCK").key == "echo"

    def test_resolves_finetuned_hf_repo_ids(self):
        assert get_card("openvla/openvla-7b-finetuned-libero-spatial").key == "openvla"

    def test_explicit_checkpoint_overrides_default(self):
        _, checkpoint = resolve("openvla", "my-org/my-openvla")
        assert checkpoint == "my-org/my-openvla"

    def test_unknown_model_lists_alternatives(self):
        with pytest.raises(ModelNotFoundError, match="Known models"):
            get_card("rt-2")

    def test_weightless_model_needs_no_checkpoint(self):
        assert resolve("echo")[1] is None
