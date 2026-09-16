# Scaling

Cost measurements (inference time, training time, training memory) for a
width-scalable ResNet-50 and a ViT-S whose width, depth, MLP size and sequence
length vary independently, plus a SLURM sweep that walks one axis at a time.

## Layout

| Path | What it is |
| --- | --- |
| [scaling/models/resnet.py](scaling/models/resnet.py) | ResNet-50, uniform channel multiplier + configurable input resolution |
| [scaling/models/vit.py](scaling/models/vit.py) | ViT-S, configurable `embed_dim` / `depth` / `mlp_dim` / `num_patches` |
| [scaling/benchmark.py](scaling/benchmark.py) | One measurement: inference ms, training ms, peak training memory |
| [scaling/sweep.py](scaling/sweep.py) | Emits the one-factor-at-a-time grid as JSONL |
| [scaling/aggregate.py](scaling/aggregate.py) | Merges per-task results into a CSV / table |
| [scripts/sweep.slurm](scripts/sweep.slurm) | SLURM array job, one measurement per task |
| [scripts/submit_sweep.sh](scripts/submit_sweep.sh) | Regenerates the config, sizes `--array`, submits |

## Install

```bash
uv sync                     # or: pip install -e .
```

`requires-python` is `>=3.10,<3.13`: torch publishes no wheels for 3.14. On a
cluster you will usually load the site torch module instead and just put the
repo root on `PYTHONPATH`.

## Models

Both architectures reproduce their reference parameter counts at the base
configuration (asserted in [tests/test_models.py](tests/test_models.py)):
ResNet-50 25.56 M / 4.09 GMACs, ViT-S/16 22.05 M / 4.60 GMACs.

**ResNet-50** — `width_mult` scales *every* channel count (stem, both
bottleneck convs, the 4x expansion and the classifier input), rounded to a
multiple of 8. This differs from torchvision's `width_per_group`, which only
touches the inner 3x3. `resolution` sets the square input side; the network is
fully convolutional with a global pool, so any value works.

**ViT-S** — the four axes are decoupled. `num_heads` defaults to
`embed_dim // 64`, keeping head dimension fixed at the usual 64 as width grows.
`num_patches` *is* the sequence length (the class token adds one more), and the
image resolution is derived from it as `sqrt(num_patches) * patch_size`, so the
model still consumes a real image tensor rather than synthetic tokens. It must
be a perfect square; the error message suggests the nearest valid values.

## Measuring one configuration

```bash
python -m scaling.benchmark --model resnet50 --set width_mult=0.5 resolution=160 \
    --batch-size 128 --flops
python -m scaling.benchmark --model vit_small --set depth=6 embed_dim=512 \
    --batch-size 128 --amp bf16
```

Useful flags: `--iters/--warmup` (default 30/10), `--amp none|fp16|bf16`,
`--channels-last`, `--device cpu` (for a laptop smoke test), `--out FILE` to
append the JSON record, `--flops` for an analytic forward FLOP count.

What the numbers mean:

- **Timing** uses CUDA events, not `time.perf_counter`, which would only measure
  kernel-launch dispatch. Reported as median over `--iters` steps after
  `--warmup` discarded ones, with mean/std/p10/p90 alongside.
- **Training step** = forward + cross-entropy + backward + SGD-momentum update.
- **Training memory** is `max_memory_allocated` over a step measured *after* the
  warm-up loop, with the peak counter reset first, so cuDNN autotuning workspace
  does not pollute it. It covers weights + gradients + optimizer state +
  activations. `peak_reserved_mib` is what the caching allocator held from the
  driver.
- `cudnn.benchmark` is on: shapes are static here, which is how these models are
  really trained.
- An OOM is recorded as `"status": "oom"` instead of killing the sweep, so the
  memory ceiling of an axis shows up as data.

## Sweeping on SLURM

The sweep is **one-factor-at-a-time**: every axis is walked while the others sit
at their base value, and the base point appears once per architecture. That is
43 runs (ResNet-50 15, ViT-S 28) instead of the 3 648 of a full grid.

```bash
scripts/submit_sweep.sh                       # everything, batch size 128
scripts/submit_sweep.sh --models resnet50     # one architecture
scripts/submit_sweep.sh --batch-size 64 --amp bf16 --flops
```

The wrapper regenerates `configs/sweep.jsonl`, sets `--array` to match its line
count and submits [scripts/sweep.slurm](scripts/sweep.slurm). Each array task
runs one config and writes `results/raw/<jobid>_<taskid>.jsonl` — one file per
task on purpose, since concurrent appends to a shared JSONL interleave and
corrupt lines on NFS.

Knobs: `MAX_CONCURRENT` (default 4, becomes the `%n` array throttle),
`SBATCH_ARGS` for extra sbatch flags (partition, account, `--exclusive`),
`PYTHON`, `CONFIG_FILE`.

Adjust the `#SBATCH` header in `sweep.slurm` for your site — partition/account
are intentionally absent. **Timings are only meaningful on an uncontended GPU**;
if your cluster packs jobs onto shared devices, add `#SBATCH --exclusive`.

To sweep different values, edit `SWEEP_VALUES` in
[scaling/sweep.py](scaling/sweep.py); the generator checks that each axis's base
value is present in its list.

## Collecting results

```bash
python -m scaling.aggregate --table          # results/raw/*.jsonl -> results/sweep.csv
```

One CSV row per run with the swept axis, its value, parameter count, GMACs, the
three measurements and the GPU it ran on — ready to plot straight into the
manuscript. Runs that OOM'd or errored are listed on stderr and kept in the CSV
with their status.

## Tests

```bash
python -m pytest tests -q
```
