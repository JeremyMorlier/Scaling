#!/usr/bin/env python3
"""Run the whole study end to end on one machine.

    python scripts/run_all.py 128

The batch size is the only thing you have to supply. The script then generates
the sweep, measures every configuration, merges the results and writes the
figures:

    configs/sweep.jsonl  ->  results/raw/local.jsonl
                         ->  results/sweep.csv
                         ->  results/figures/*.pdf

Each configuration is measured in its own subprocess, exactly as the SLURM
array does -- one code path for both, a fresh CUDA context per measurement, and
a run that segfaults or gets OOM-killed costs one data point instead of the
whole sweep. It resumes by default, so re-running after an interruption only
measures what is missing; pass --fresh to start over.

For the cluster, use scripts/submit_sweep.sh instead: same work, in parallel.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scaling.sweep import generate  # noqa: E402

CONFIG_FILE = ROOT / "configs" / "sweep.jsonl"
RAW_FILE = ROOT / "results" / "raw" / "local.jsonl"
CSV_FILE = ROOT / "results" / "sweep.csv"
FIG_DIR = ROOT / "results" / "figures"


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60:d}m{seconds % 60:02d}s"
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m"


def preflight(device: str) -> str:
    """Fail fast on a dead device, and name the GPU for the log.

    Without this, a login node with no GPU would spawn one doomed subprocess per
    configuration and report the same traceback dozens of times.
    """
    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "error: --device cuda but torch.cuda.is_available() is False.\n"
                "       Run this on a GPU node (srun/sbatch), or pass "
                "--device cpu for a smoke test.")
        name = torch.cuda.get_device_properties(torch.device(device)).name
        return f"{name} · torch {torch.__version__}"
    return f"{device} · torch {torch.__version__}"


def write_configs(batch_size: int, models: list[str] | None) -> list[dict]:
    runs = generate(models=models, batch_size=batch_size, measure_flops=True)
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as fh:
        for run in runs:
            fh.write(json.dumps(run) + "\n")
    return runs


def already_done(path: Path) -> set[str]:
    """run_ids already present, so an interrupted sweep can pick up where it left off."""
    if not path.exists():
        return set()
    done = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["run_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # a torn final line from a hard kill
    return done


def measure(runs: list[dict], device: str, done: set[str]) -> dict[str, int]:
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        filter(None, [str(ROOT), os.environ.get("PYTHONPATH", "")])))

    tally = {"ok": 0, "oom": 0, "error": 0, "skipped": 0}
    started = time.time()
    measured = 0

    for index, run in enumerate(runs):
        tag = f"[{index + 1:>2}/{len(runs)}]"
        if run["run_id"] in done:
            tally["skipped"] += 1
            print(f"{tag} {run['run_id']:<38} skipped (already measured)")
            continue

        proc = subprocess.run(
            [sys.executable, "-m", "scaling.benchmark",
             "--config-file", str(CONFIG_FILE), "--index", str(index),
             "--device", device, "--out", str(RAW_FILE)],
            cwd=ROOT, env=env, capture_output=True, text=True)
        measured += 1

        try:
            record = json.loads(proc.stdout)
            status = record["status"]
        except (json.JSONDecodeError, KeyError):
            # The child died before it could report; benchmark.py handles its own
            # exceptions, so this is a segfault, an OOM kill or a bad environment.
            status = "error"
            record = {}
            print(f"{tag} {run['run_id']:<38} CRASHED (exit {proc.returncode})")
            print((proc.stderr or "").strip()[-500:], file=sys.stderr)

        tally[status] = tally.get(status, 0) + 1
        if record:
            if status == "ok":
                detail = (f"infer {record['inference']['ms_median']:7.1f} ms   "
                          f"train {record['training']['ms_median']:7.1f} ms")
                mem = record["training"].get("peak_mem_mib")
                detail += f"   mem {mem:8.0f} MiB" if mem else "   mem       n/a"
            else:
                detail = record.get("error", "")[:60]
            print(f"{tag} {run['run_id']:<38} {status:<5} {detail}")

        elapsed = time.time() - started
        remaining = len(runs) - index - 1
        if measured and remaining:
            eta = elapsed / measured * remaining
            print(f"{'':>6} elapsed {_hms(elapsed)} · ~{_hms(eta)} left")

    return tally


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python scripts/run_all.py",
        description="Sweep, measure, aggregate and plot, on this machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:  python scripts/run_all.py 128")
    p.add_argument("batch_size", type=int,
                   help="batch size used for every measurement")
    p.add_argument("--device", default="cuda",
                   help="cuda (default); cpu is only useful as a smoke test")
    p.add_argument("--models", nargs="*",
                   help="restrict to these architectures (default: all)")
    p.add_argument("--fresh", action="store_true",
                   help="discard previous results instead of resuming")
    args = p.parse_args(argv)

    # A sweep is long and usually redirected to a log; without this, progress
    # sits in the block buffer and arrives out of order with the child
    # processes' own output.
    sys.stdout.reconfigure(line_buffering=True)

    if args.fresh and RAW_FILE.exists():
        RAW_FILE.unlink()

    where = preflight(args.device)
    runs = write_configs(args.batch_size, args.models)
    done = already_done(RAW_FILE)
    print(f"sweep: {len(runs)} configurations at batch size {args.batch_size}"
          + (f", {len(done)} already measured" if done else ""))
    print(f"device: {where}")
    print(f"config: {CONFIG_FILE.relative_to(ROOT)}\n")

    started = time.time()
    tally = measure(runs, args.device, done)

    print(f"\nmeasured in {_hms(time.time() - started)}: "
          + ", ".join(f"{n} {name}" for name, n in tally.items() if n))

    from scaling.aggregate import main as aggregate_main
    if aggregate_main([str(RAW_FILE), "--csv", str(CSV_FILE)]) != 0:
        print("nothing to aggregate -- no successful runs", file=sys.stderr)
        return 1

    try:
        from scaling.plot import main as plot_main
    except ImportError as exc:
        print(f"\nresults are in {CSV_FILE.relative_to(ROOT)}; "
              f"skipping figures ({exc}). Install with: pip install -e '.[viz]'",
              file=sys.stderr)
        return 0

    print()
    plot_main(["--csv", str(CSV_FILE), "--out-dir", str(FIG_DIR)])
    # An error or OOM anywhere is worth a non-zero exit, but only after the
    # figures are written -- partial results are still worth having.
    return 1 if tally.get("error") or tally.get("oom") else 0


if __name__ == "__main__":
    raise SystemExit(main())
