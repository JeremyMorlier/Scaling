"""Generate the one-factor-at-a-time (OFAT) sweep grid.

Each architecture has a base configuration; every axis is swept while the other
axes stay at their base value.  The base point itself appears exactly once per
architecture, so a sweep over ``k`` axes with ``n_i`` values each costs
``1 + sum(n_i - 1)`` runs rather than the ``prod(n_i)`` of a full grid.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .models import MODEL_AXES

#: Values probed along each axis.  The base value must be present in each list.
SWEEP_VALUES: dict[str, dict[str, list[Any]]] = {
    "resnet50": {
        # Uniform channel multiplier: params scale ~quadratically.
        "width_mult": [0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        # Input resolution: activations (and so time/memory) scale ~quadratically.
        "resolution": [96, 128, 160, 192, 224, 288, 320, 384],
    },
    "vit_small": {
        "embed_dim": [128, 192, 256, 384, 512, 640, 768, 1024],
        "depth": [2, 4, 6, 8, 12, 16, 20, 24],
        "mlp_dim": [384, 768, 1152, 1536, 2304, 3072, 4096],
        # Sequence length must be a perfect square (patch grid); 196 = 14x14 is
        # the ViT-S/16 base at 224px.
        "num_patches": [36, 64, 100, 144, 196, 256, 324, 400],
    },
}


def generate(models: list[str] | None = None, batch_size: int = 128,
             **spec_overrides: Any) -> list[dict[str, Any]]:
    """Return the OFAT run specs as plain dicts, ready to serialise as JSONL."""
    models = models or sorted(SWEEP_VALUES)
    runs: list[dict[str, Any]] = []

    for model in models:
        if model not in SWEEP_VALUES:
            raise ValueError(f"no sweep defined for {model!r}")
        base = dict(MODEL_AXES[model])
        axes = SWEEP_VALUES[model]

        def emit(axis: str, params: dict[str, Any], tag: str) -> None:
            runs.append({
                "run_id": f"{model}__{tag}",
                "model": model,
                "axis": axis,
                "params": params,
                "batch_size": batch_size,
                **spec_overrides,
            })

        emit("base", dict(base), "base")
        for axis, values in axes.items():
            if base[axis] not in values:
                raise ValueError(
                    f"{model}.{axis}: base value {base[axis]} missing from "
                    f"sweep values {values}")
            for value in values:
                if value == base[axis]:
                    continue  # already covered by the base run
                emit(axis, {**base, axis: value}, f"{axis}={value}")

    return runs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scaling.sweep",
        description="Emit the OFAT sweep as JSONL (one run spec per line).")
    p.add_argument("--models", nargs="*", choices=sorted(SWEEP_VALUES),
                   help="restrict the sweep to these architectures")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="none")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--flops", action="store_true", dest="measure_flops")
    p.add_argument("--out", default="configs/sweep.jsonl")
    p.add_argument("--count", action="store_true",
                   help="print only the number of runs (for --array sizing)")
    return p


def main(argv: list[str] | None = None) -> int:
    import os

    args = build_parser().parse_args(argv)
    runs = generate(
        models=args.models,
        batch_size=args.batch_size,
        iters=args.iters,
        warmup=args.warmup,
        amp=args.amp,
        channels_last=args.channels_last,
        measure_flops=args.measure_flops,
    )
    if args.count:
        print(len(runs))
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        for run in runs:
            fh.write(json.dumps(run) + "\n")

    print(f"wrote {len(runs)} runs to {args.out}")
    print(f"slurm array range: 0-{len(runs) - 1}")
    for model in sorted({r['model'] for r in runs}):
        per_axis = {}
        for r in runs:
            if r["model"] == model:
                per_axis[r["axis"]] = per_axis.get(r["axis"], 0) + 1
        detail = ", ".join(f"{a}:{n}" for a, n in per_axis.items())
        print(f"  {model}: {sum(per_axis.values())} runs ({detail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
