"""Command-line interface: ``vla``.

Subcommands:
    ``models``   List supported policies and their memory footprints.
    ``doctor``   Check the machine: GPUs, torch, kernels, quantization backends.
    ``info``     Show what a policy expects and whether it fits this hardware.
    ``predict``  Run one observation through a policy from an image file.
    ``serve``    Start the inference server.
    ``bench``    Measure latency across precisions, compilation, and batch sizes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .config import EngineConfig
from .errors import VLAEngineError

__all__ = ["main"]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", help="cuda:0, cpu, or auto")
    parser.add_argument(
        "--devices", default="", help="comma-separated replica devices, e.g. cuda:0,cuda:1"
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--quantization", default=None, choices=["nf4", "int8", "awq", "gptq"])
    parser.add_argument(
        "--attention", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"]
    )
    parser.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    parser.add_argument("--checkpoint", default=None, help="override the default weights")
    parser.add_argument("--unnorm-key", default=None, help="OpenVLA dataset statistics key")
    parser.add_argument("--horizon", type=int, default=None, help="truncate the action chunk")


def _config_from_args(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        model=args.model,
        checkpoint=args.checkpoint,
        device=args.device,
        devices=[d.strip() for d in args.devices.split(",") if d.strip()],
        precision={
            "dtype": args.dtype,
            "quantization": args.quantization,
            "attention": args.attention,
        },
        compile={"enabled": not args.no_compile, "mode": args.compile_mode},
        batch={"max_batch_size": getattr(args, "max_batch_size", 1)},
        unnorm_key=args.unnorm_key,
        horizon=args.horizon,
    )


# -- commands ---------------------------------------------------------------


def cmd_models(args: argparse.Namespace) -> int:
    from .registry import list_models

    cards = list_models()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "key": c.key,
                        "checkpoint": c.default_checkpoint,
                        "params_b": c.params_b,
                        "bf16_gb": round(c.memory_gb("bf16"), 2),
                        "nf4_gb": round(c.memory_gb("bf16", "nf4"), 2),
                        "action_dim": c.spec.action_dim,
                        "horizon": c.spec.horizon,
                    }
                    for c in cards
                ],
                indent=2,
            )
        )
        return 0
    print(
        f"{'MODEL':<10} {'PARAMS':>7} {'BF16':>8} {'NF4':>7} {'DIM':>4} {'HORIZON':>8}  CHECKPOINT"
    )
    for card in cards:
        print(
            f"{card.key:<10} {card.params_b:>6.2f}B {card.memory_gb('bf16'):>7.1f}G "
            f"{card.memory_gb('bf16', 'nf4'):>6.1f}G {card.spec.action_dim:>4} "
            f"{card.spec.horizon:>8}  {card.default_checkpoint or '-'}"
        )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what this machine can actually do, and what is missing."""
    from .runtime.device import probe_devices, torch_available
    from .runtime.precision import bitsandbytes_available, flash_attention_available

    print(f"vla-engine {__version__}")
    print(f"python     {sys.version.split()[0]}")

    if torch_available():
        import torch

        print(f"torch      {torch.__version__} (cuda {torch.version.cuda or 'n/a'})")
    else:
        print("torch      NOT INSTALLED  -> pip install 'vla-engine[cuda]'")

    topology = probe_devices(refresh=True)
    print()
    if topology.num_gpus == 0:
        print("GPUs       none detected; inference will run on CPU and be very slow")
    else:
        for device in topology.cuda_devices:
            print(f"GPU {device.index}      {device.summary()}")
            print(
                f"           bf16={device.supports_bf16} fp8={device.supports_fp8} "
                f"tf32={device.supports_tf32} flash-attn-2={device.supports_flash_attention_2}"
            )
        if topology.num_gpus > 1:
            note = (
                "homogeneous"
                if topology.homogeneous
                else "MIXED MODELS (replica scheduling assumes matched GPUs)"
            )
            print(f"           {topology.num_gpus} GPUs, {note}")

    print()
    flash = flash_attention_available()
    bnb = bitsandbytes_available()
    print(f"flash-attn {'yes' if flash else 'no   -> pip install flash-attn --no-build-isolation'}")
    print(
        f"bitsandbytes {'yes' if bnb else 'no -> pip install bitsandbytes  (needed for nf4/int8)'}"
    )

    if topology.num_gpus:
        print()
        print(f"Fit on the smallest visible GPU ({topology.min_memory_gb():.0f} GB):")
        from .registry import list_models

        for card in list_models():
            if not card.params_b:
                continue
            bf16 = "yes" if card.fits_on(topology.min_memory_gb(), "bf16") else "no"
            nf4 = "yes" if card.fits_on(topology.min_memory_gb(), "bf16", "nf4") else "no"
            print(
                f"  {card.key:<10} bf16 {card.memory_gb('bf16'):>5.1f} GB [{bf16}]   "
                f"nf4 {card.memory_gb('bf16', 'nf4'):>5.1f} GB [{nf4}]"
            )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .engine import VLAEngine
    from .registry import get_card

    card = get_card(args.model)
    engine = VLAEngine(_config_from_args(args))
    if args.json:
        print(json.dumps(engine.info(), indent=2))
        return 0
    print(f"model        {card.key}")
    print(f"checkpoint   {card.default_checkpoint or '-'}")
    print(f"parameters   {card.params_b:.2f}B")
    print(f"license      {card.license}")
    print(f"action_dim   {card.spec.action_dim}")
    print(f"horizon      {card.spec.horizon}")
    print(f"cameras      {', '.join(card.spec.cameras)}")
    print(f"state        {'required' if card.spec.requires_state else 'not used'}")
    print(f"image_size   {card.spec.image_size[0]}x{card.spec.image_size[1]}")
    print(f"devices      {engine.devices}")
    print(f"est. memory  {engine.info()['estimated_memory_gb']} GB")
    print()
    print(card.notes)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    import numpy as np

    from .engine import VLAEngine
    from .types import Observation

    try:
        from PIL import Image
    except ImportError:
        print("error: predict needs Pillow (pip install pillow)", file=sys.stderr)
        return 1

    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    state = (
        np.array([float(x) for x in args.state.split(",")], dtype=np.float32)
        if args.state
        else None
    )
    engine = VLAEngine(_config_from_args(args)).load()
    chunk = engine.predict(Observation.single(image, args.instruction, state))

    if args.json:
        print(
            json.dumps(
                {
                    "actions": chunk.actions.tolist(),
                    "stats": chunk.stats.as_dict() if chunk.stats else None,
                },
                indent=2,
            )
        )
    else:
        print(f"# {engine.card.key}: {chunk.horizon} x {chunk.action_dim} actions")
        for i, action in enumerate(chunk.actions[: args.show]):
            print(f"[{i:>3}] " + "  ".join(f"{v:+.4f}" for v in action))
        if chunk.horizon > args.show:
            print(f"... {chunk.horizon - args.show} more steps")
        if chunk.stats:
            print(f"\nlatency {chunk.stats.total_ms:.1f} ms on {chunk.stats.device}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .serve.server import serve

    config = _config_from_args(args)
    serve(config, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench.latency import sweep
    from .engine import VLAEngine

    base = _config_from_args(args).to_dict()
    configurations: list[dict] = []
    if args.compare:
        # The comparisons people actually want: what compilation buys, and
        # what quantization costs.
        configurations = [
            {"_label": "eager bf16", "compile": False, "dtype": "bf16"},
            {"_label": "compiled bf16", "compile": True, "dtype": "bf16"},
            {"_label": "compiled nf4", "compile": True, "dtype": "bf16", "quantization": "nf4"},
        ]
    else:
        configurations = [{"_label": "current"}]

    def factory(overrides: dict) -> Any:
        config = dict(base)
        precision = dict(config["precision"])
        if "dtype" in overrides:
            precision["dtype"] = overrides["dtype"]
        if "quantization" in overrides:
            precision["quantization"] = overrides["quantization"]
        config["precision"] = precision
        if "compile" in overrides:
            compile_cfg = dict(config["compile"])
            compile_cfg["enabled"] = bool(overrides["compile"])
            config["compile"] = compile_cfg
        return VLAEngine(EngineConfig.from_dict(config)).load()

    batch_sizes = tuple(int(b) for b in args.batch_sizes.split(","))
    results = sweep(
        factory,
        configurations,
        iterations=args.iterations,
        warmup=args.warmup,
        batch_sizes=batch_sizes,
    )
    print()
    for result in results:
        print(result.summary())
    if args.json:
        print()
        print(json.dumps([r.as_dict() for r in results], indent=2))
    return 0


# -- entry point ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vla", description="Inference engine for vision-language-action models"
    )
    parser.add_argument("--version", action="version", version=f"vla-engine {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="list supported policies")
    models.add_argument("--json", action="store_true")
    models.set_defaults(func=cmd_models)

    doctor = subparsers.add_parser("doctor", help="check GPUs and optional backends")
    doctor.set_defaults(func=cmd_doctor)

    info = subparsers.add_parser("info", help="describe a policy")
    info.add_argument("model")
    info.add_argument("--json", action="store_true")
    _add_common(info)
    info.set_defaults(func=cmd_info)

    predict = subparsers.add_parser("predict", help="run one observation")
    predict.add_argument("model")
    predict.add_argument("--image", required=True, help="path to an image file")
    predict.add_argument("--instruction", required=True)
    predict.add_argument("--state", default=None, help="comma-separated proprioception")
    predict.add_argument("--show", type=int, default=5, help="action steps to print")
    predict.add_argument("--json", action="store_true")
    _add_common(predict)
    predict.set_defaults(func=cmd_predict)

    serve_parser = subparsers.add_parser("serve", help="start the inference server")
    serve_parser.add_argument("model")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--max-batch-size", type=int, default=1)
    serve_parser.add_argument("--log-level", default="info")
    _add_common(serve_parser)
    serve_parser.set_defaults(func=cmd_serve)

    bench = subparsers.add_parser("bench", help="measure latency")
    bench.add_argument("model")
    bench.add_argument("--iterations", type=int, default=30)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--batch-sizes", default="1")
    bench.add_argument(
        "--compare", action="store_true", help="sweep eager/compiled/nf4 configurations"
    )
    bench.add_argument("--json", action="store_true")
    _add_common(bench)
    bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except VLAEngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
