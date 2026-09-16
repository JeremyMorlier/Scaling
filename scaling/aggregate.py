"""Merge the per-task JSONL records of a sweep into one flat CSV / table."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from typing import Any

COLUMNS = [
    "run_id", "model", "axis", "swept_value", "status",
    "batch_size", "amp", "params", "resolution", "seq_len", "num_heads",
    "fwd_macs_per_image",
    "infer_ms_median", "infer_img_per_s", "infer_peak_mem_mib",
    "train_ms_median", "train_img_per_s", "train_peak_mem_mib",
    "train_peak_reserved_mib",
    "gpu_name", "hostname", "slurm_job_id",
]


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    spec = record.get("spec", {})
    params = spec.get("params", {})
    axis = record.get("axis", "")
    info = record.get("model_info") or {}
    flops = record.get("flops") or {}
    inf = record.get("inference") or {}
    train = record.get("training") or {}
    env = record.get("env") or {}

    return {
        "run_id": record.get("run_id"),
        "model": spec.get("model"),
        "axis": axis,
        # For a base run every axis is at its base value, so there is no single
        # swept value to report.
        "swept_value": params.get(axis) if axis not in ("", "base") else "",
        "status": record.get("status"),
        "batch_size": spec.get("batch_size"),
        "amp": spec.get("amp"),
        "params": info.get("params"),
        "resolution": info.get("resolution"),
        "seq_len": info.get("seq_len"),
        "num_heads": info.get("num_heads"),
        "fwd_macs_per_image": flops.get("fwd_macs_per_image"),
        "infer_ms_median": inf.get("ms_median"),
        "infer_img_per_s": inf.get("throughput_img_per_s"),
        "infer_peak_mem_mib": inf.get("peak_mem_mib"),
        "train_ms_median": train.get("ms_median"),
        "train_img_per_s": train.get("throughput_img_per_s"),
        "train_peak_mem_mib": train.get("peak_mem_mib"),
        "train_peak_reserved_mib": train.get("peak_reserved_mib"),
        "gpu_name": env.get("gpu_name"),
        "hostname": env.get("hostname"),
        "slurm_job_id": env.get("slurm_job_id"),
    }


def load(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(flatten(json.loads(line)))
                    except json.JSONDecodeError as exc:
                        print(f"warning: {path}:{lineno}: {exc}", file=sys.stderr)
    return rows


def _sort_key(row: dict[str, Any]) -> tuple:
    value = row.get("swept_value")
    numeric = isinstance(value, (int, float))
    return (row.get("model") or "", row.get("axis") or "",
            not numeric, value if numeric else str(value))


def print_table(rows: list[dict[str, Any]]) -> None:
    cols = ["run_id", "status", "params", "fwd_macs_per_image",
            "infer_ms_median", "train_ms_median", "train_peak_mem_mib"]
    header = ["run_id", "status", "params(M)", "GMACs",
              "infer(ms)", "train(ms)", "train mem(MiB)"]

    def fmt(row: dict[str, Any], col: str) -> str:
        v = row.get(col)
        if v is None or v == "":
            return "-"
        if col == "params":
            return f"{v / 1e6:.2f}"
        if col == "fwd_macs_per_image":
            return f"{v / 1e9:.2f}"
        if isinstance(v, float):
            return f"{v:.1f}"
        return str(v)

    table = [header] + [[fmt(r, c) for c in cols] for r in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(cols))]
    for i, row in enumerate(table):
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m scaling.aggregate",
        description="Merge sweep results into a CSV / printed table.")
    p.add_argument("inputs", nargs="*", default=["results/raw/*.jsonl"],
                   help="JSONL files or globs (default: results/raw/*.jsonl)")
    p.add_argument("--csv", default="results/sweep.csv",
                   help="path of the merged CSV ('-' to skip)")
    p.add_argument("--table", action="store_true", help="also print a summary table")
    args = p.parse_args(argv)

    rows = sorted(load(args.inputs), key=_sort_key)
    if not rows:
        print(f"no records found in {args.inputs}", file=sys.stderr)
        return 1

    if args.csv != "-":
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}", file=sys.stderr)

    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        print(f"{len(failed)} run(s) not ok: "
              f"{', '.join(r['run_id'] for r in failed)}", file=sys.stderr)
    if args.table:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
