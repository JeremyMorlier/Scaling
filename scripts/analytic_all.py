#!/usr/bin/env python3
"""The whole study again, analytically -- no GPU, no measurement, a second.

    python scripts/analytic_all.py 64

This is the counterpart of scripts/run_all.py. It walks the same
one-factor-at-a-time grid, but instead of timing each configuration it costs it
in closed form (scaling.analytic), then draws the figures:

    configs/sweep.jsonl (the same grid)  ->  results/analytic.csv
                                         ->  results/figures/*.pdf

The batch size is the only thing you have to supply, since memory is the half
of the model that depends on it. Nothing is executed, so a configuration that
would OOM on the measurement node costs exactly as much to plot as one that
fits -- widen SWEEP_VALUES in scaling/sweep.py and the extra points are free.

The figures land beside the measured ones and never overwrite them: the file
stems differ (flops_vs_flops, flops_vs_memory, params_vs_memory,
<model>_axes_analytic). When results/sweep.csv exists, the run ends with the
two tables side by side, which is the check that the model is worth trusting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scaling.analytic import (OPTIMIZER_STATES, compare_csv,  # noqa: E402
                             sweep_rows, write_sweep_csv)

CSV_FILE = ROOT / "results" / "analytic.csv"
MEASURED_CSV = ROOT / "results" / "sweep.csv"
FIG_DIR = ROOT / "results" / "figures"


def print_table(rows: list[dict]) -> None:
    """One line per configuration, grouped by model and axis."""
    head = (f"{'run_id':<34}{'params':>13}{'GFLOPs/img':>12}{'GFLOPs/step':>13}"
            f"{'act MiB':>10}{'train MiB':>11}")
    print(head)
    print("-" * len(head))
    axis = None
    for row in rows:
        if (row["model"], row["axis"]) != axis:
            axis = (row["model"], row["axis"])
            print()
        print(f"{row['run_id']:<34}{row['params']:>13,}"
              f"{row['fwd_gflops_per_image']:>12.3f}"
              f"{row['train_gflops_per_step']:>13.1f}"
              f"{row['act_mem_mib']:>10.0f}{row['train_mem_mib']:>11.0f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python scripts/analytic_all.py",
        description="Cost the whole sweep in closed form and plot it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:  python scripts/analytic_all.py 64")
    p.add_argument("batch_size", type=int,
                   help="batch size the memory figures assume")
    p.add_argument("--models", nargs="*",
                   help="restrict to these architectures (default: all)")
    p.add_argument("--optimizer", choices=sorted(OPTIMIZER_STATES),
                   default="sgd_momentum",
                   help="what the measured sweep uses (default: SGD + momentum)")
    p.add_argument("--param-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--act-dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                   help="bf16 to model an autocast run")
    p.add_argument("--attn-impl", choices=["flash", "math"], default="flash",
                   help="ViT only: what the attention backward keeps")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--csv", default=str(CSV_FILE))
    p.add_argument("--out-dir", default=str(FIG_DIR))
    p.add_argument("--theme", default="light", choices=["light", "dark"],
                   help="light for print/LaTeX, dark for slides")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--compare", metavar="CSV", default=str(MEASURED_CSV),
                   help="measured sweep to tabulate against, when it exists")
    p.add_argument("--no-compare", action="store_true")
    p.add_argument("--quiet", action="store_true",
                   help="skip the per-configuration table")
    args = p.parse_args(argv)

    # Keep our progress interleaved correctly with the plotter's own output
    # when this is redirected to a log.
    sys.stdout.reconfigure(line_buffering=True)

    rows = sweep_rows(models=args.models, batch_size=args.batch_size,
                      optimizer=args.optimizer, param_dtype=args.param_dtype,
                      act_dtype=args.act_dtype, attn_impl=args.attn_impl,
                      num_classes=args.num_classes)
    write_sweep_csv(args.csv, rows)

    print(f"analytic sweep: {len(rows)} configurations at batch size "
          f"{args.batch_size}, {args.optimizer}, activations {args.act_dtype}\n")
    if not args.quiet:
        print_table(rows)
    print(f"\nwrote {Path(args.csv).resolve().relative_to(ROOT)}")

    if not args.no_compare and Path(args.compare).exists():
        print(f"\nagainst {args.compare} (params and MACs are exact by "
              f"construction; memory is modelled):\n")
        print(compare_csv(args.compare, optimizer=args.optimizer))

    try:
        from scaling.plot import ANALYTIC, main as plot_main
    except ImportError as exc:
        print(f"\nskipping figures ({exc}). Install with: pip install -e '.[viz]'",
              file=sys.stderr)
        return 0

    print()
    # Name the analytic figures explicitly: asking for the measured ones as
    # well would only print two "no such column" notes.
    return plot_main(["--csv", args.csv, "--out-dir", args.out_dir,
                      "--theme", args.theme, "--formats", *args.formats,
                      "--figures", "overlay", *ANALYTIC])


if __name__ == "__main__":
    raise SystemExit(main())
