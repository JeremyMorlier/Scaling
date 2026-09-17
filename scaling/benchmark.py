"""Measure inference time, training time and training memory on a synthetic
ImageNet-like batch.

Timings use CUDA events (device-side, so they are immune to the asynchronous
dispatch that makes ``time.perf_counter`` around a CUDA call meaningless) and
are reported as the median over ``--iters`` steps after ``--warmup`` discarded
steps.  Memory is ``torch.cuda.max_memory_allocated`` measured over a training
step with the peak counter reset first, i.e. weights + gradients + optimizer
state + activations.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn

# Running a package module straight from an editor ("python scaling/sweep.py")
# leaves it without a parent package, so the relative imports below fail. Put
# the repo root on the path and name the package, so both that and the
# supported "python -m scaling.sweep" work.
if __package__ in (None, ""):
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "scaling"

from .models import MODEL_AXES, build_model

AMP_DTYPES = {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass
class RunSpec:
    """Everything that defines one measurement."""

    model: str
    params: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 128
    num_classes: int = 1000
    iters: int = 30
    warmup: int = 10
    amp: str = "none"
    channels_last: bool = False
    device: str = "cuda"
    run_id: str = ""
    axis: str = ""
    measure_flops: bool = False
    seed: int = 0


# --------------------------------------------------------------------------- #
# timing helpers
# --------------------------------------------------------------------------- #

def _time_loop(step: Callable[[], None], iters: int, warmup: int,
               device: torch.device) -> list[float]:
    """Run ``step`` and return per-iteration wall times in milliseconds."""
    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            step()
            ends[i].record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in zip(starts, ends)]

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _summarize(times: Iterable[float], batch_size: int) -> dict[str, float]:
    times = sorted(times)
    median = statistics.median(times)
    return {
        "ms_median": median,
        "ms_mean": statistics.fmean(times),
        "ms_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "ms_p10": times[int(0.10 * (len(times) - 1))],
        "ms_p90": times[int(0.90 * (len(times) - 1))],
        "throughput_img_per_s": batch_size / (median / 1e3),
    }


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #

def measure_inference(model: nn.Module, inputs: torch.Tensor, spec: RunSpec,
                      device: torch.device) -> dict[str, float]:
    model.eval()
    dtype = AMP_DTYPES[spec.amp]
    autocast = (torch.autocast(device.type, dtype=dtype) if dtype is not None
                else nullcontext())

    def step() -> None:
        with torch.inference_mode(), autocast:
            model(inputs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = _time_loop(step, spec.iters, spec.warmup, device)
    out = _summarize(times, spec.batch_size)
    if device.type == "cuda":
        out["peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    return out


def _grad_scaler(enabled: bool):
    """GradScaler across torch versions (``torch.cuda.amp`` is deprecated in 2.4+)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def measure_training(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
                     spec: RunSpec, device: torch.device) -> dict[str, float]:
    """One optimizer step = forward + loss + backward + SGD update."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    dtype = AMP_DTYPES[spec.amp]
    autocast = (torch.autocast(device.type, dtype=dtype) if dtype is not None
                else nullcontext())
    # GradScaler is only needed for fp16; bf16 has fp32 dynamic range.
    scaler = _grad_scaler(enabled=(spec.amp == "fp16" and device.type == "cuda"))

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            loss = criterion(model(inputs), targets)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    # Warm up first so that the peak-memory reading excludes allocator growth
    # from cuDNN autotuning, then reset the counter and measure a clean window.
    times = _time_loop(step, spec.iters, spec.warmup, device)
    out = _summarize(times, spec.batch_size)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step()
        torch.cuda.synchronize()
        out["peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        out["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20
    return out


def measure_flops(model: nn.Module, inputs: torch.Tensor) -> dict[str, float] | None:
    """Forward FLOPs for one batch, counted analytically by torch's dispatcher.

    ``FlopCounterMode`` counts the multiply and the add of a MAC separately, so
    ``fwd_macs_per_image`` is the figure papers usually quote as "GFLOPs"
    (4.1 G for ResNet-50, 4.6 G for ViT-S/16).
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    model.eval()
    counter = FlopCounterMode(display=False)
    # no_grad rather than inference_mode: inference-mode tensors bypass the
    # TorchDispatch layer the counter hooks into and it would report zero.
    with torch.no_grad(), counter:
        model(inputs)
    total = counter.get_total_flops()
    per_image = float(total) / inputs.shape[0]
    return {"fwd_flops_per_batch": float(total),
            "fwd_flops_per_image": per_image,
            "fwd_macs_per_image": per_image / 2}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def _device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "hostname": platform.node(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info |= {
            "gpu_name": props.name,
            "gpu_total_mem_mib": props.total_memory / 2 ** 20,
            "gpu_capability": f"{props.major}.{props.minor}",
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    if job := os.environ.get("SLURM_JOB_ID"):
        info["slurm_job_id"] = job
        info["slurm_array_task_id"] = os.environ.get("SLURM_ARRAY_TASK_ID")
    return info


def run(spec: RunSpec) -> dict[str, Any]:
    """Execute one measurement and return a JSON-serializable record."""
    torch.manual_seed(spec.seed)
    device = torch.device(spec.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if device.type == "cuda":
        # cuDNN autotuning: shapes are static here, so this is a pure win and
        # reflects how these models are actually trained.
        torch.backends.cudnn.benchmark = True

    record: dict[str, Any] = {
        "run_id": spec.run_id or f"{spec.model}_default",
        "axis": spec.axis,
        "spec": asdict(spec),
        "env": _device_info(device),
        "status": "ok",
    }

    try:
        model = build_model(spec.model, num_classes=spec.num_classes, **spec.params)
        model = model.to(device)
        if spec.channels_last:
            model = model.to(memory_format=torch.channels_last)

        cfg = model.config
        shape = (spec.batch_size, *cfg.input_shape)
        inputs = torch.randn(*shape, device=device)
        if spec.channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        targets = torch.randint(0, spec.num_classes, (spec.batch_size,), device=device)

        record["model_info"] = {
            "params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
            "input_shape": list(shape),
            "resolution": cfg.resolution,
            "seq_len": getattr(cfg, "seq_len", None),
            "num_heads": getattr(cfg, "num_heads", None),
        }

        if spec.measure_flops:
            record["flops"] = measure_flops(model, inputs[:1])

        record["inference"] = measure_inference(model, inputs, spec, device)
        record["training"] = measure_training(model, inputs, targets, spec, device)

    except torch.cuda.OutOfMemoryError as exc:
        # A sweep should report the OOM boundary, not abort at it.
        record["status"] = "oom"
        record["error"] = str(exc).splitlines()[0]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - one bad config must not kill a sweep
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def _parse_kv(items: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` overrides, coercing value to int/float when possible."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected key=value, got {item!r}")
        key, _, value = item.partition("=")
        for cast in (int, float):
            try:
                out[key] = cast(value)
                break
            except ValueError:
                continue
        else:
            out[key] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scaling.benchmark",
        description="Benchmark inference time, training time and training memory.")
    src = p.add_argument_group("what to run")
    src.add_argument("--model", choices=sorted(MODEL_AXES),
                     help="architecture to benchmark (ignored with --config-file)")
    src.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                     help="scaling-axis overrides, e.g. --set width_mult=0.5 "
                          "resolution=160")
    src.add_argument("--config-file", type=str,
                     help="JSONL file of run specs (see scaling.sweep)")
    src.add_argument("--index", type=int,
                     help="0-based line of --config-file to run "
                          "(defaults to $SLURM_ARRAY_TASK_ID)")

    knobs = p.add_argument_group("measurement")
    knobs.add_argument("--batch-size", type=int, default=128)
    knobs.add_argument("--num-classes", type=int, default=1000)
    knobs.add_argument("--iters", type=int, default=30)
    knobs.add_argument("--warmup", type=int, default=10)
    knobs.add_argument("--amp", choices=sorted(AMP_DTYPES), default="none")
    knobs.add_argument("--channels-last", action="store_true")
    knobs.add_argument("--device", default="cuda")
    knobs.add_argument("--flops", action="store_true",
                      help="also count forward FLOPs for a single image")
    knobs.add_argument("--seed", type=int, default=0)

    out = p.add_argument_group("output")
    out.add_argument("--out", type=str,
                     help="append the JSON record to this file (JSONL)")
    return p


def spec_from_args(args: argparse.Namespace) -> RunSpec:
    """Build a RunSpec from CLI args, optionally seeded by a sweep config line."""
    overrides: dict[str, Any] = {}
    if args.config_file:
        index = args.index
        if index is None:
            env = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env is None:
                raise SystemExit(
                    "--config-file needs --index or $SLURM_ARRAY_TASK_ID")
            index = int(env)
        with open(args.config_file) as fh:
            lines = [line for line in fh if line.strip()]
        if not 0 <= index < len(lines):
            raise SystemExit(
                f"index {index} out of range for {args.config_file} "
                f"({len(lines)} configs)")
        overrides = json.loads(lines[index])

    # CLI flags explicitly given on the command line win over the config file.
    given = {a.lstrip("-").replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    spec_fields = {f for f in RunSpec.__dataclass_fields__}
    merged = {k: v for k, v in overrides.items() if k in spec_fields}
    for name in spec_fields:
        cli_name = {"measure_flops": "flops"}.get(name, name)
        if cli_name in given and hasattr(args, cli_name):
            merged[name] = getattr(args, cli_name)
        elif name not in merged and hasattr(args, cli_name):
            merged[name] = getattr(args, cli_name)

    if args.set:
        merged.setdefault("params", {})
        merged["params"] = {**merged["params"], **_parse_kv(args.set)}
    if not merged.get("model"):
        raise SystemExit("--model is required (or provide it via --config-file)")
    return RunSpec(**merged)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = spec_from_args(args)
    record = run(spec)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
