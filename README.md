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
| [scaling/analytic.py](scaling/analytic.py) | Closed-form FLOPs and training memory, no forward pass, no GPU |
| [scaling/sweep.py](scaling/sweep.py) | Emits the one-factor-at-a-time grid as JSONL |
| [scaling/aggregate.py](scaling/aggregate.py) | Merges per-task results into a CSV / table |
| [scaling/plot.py](scaling/plot.py) | The two trade-off figures, as PDF for LaTeX |
| [scripts/run_all.py](scripts/run_all.py) | **Everything on one machine**, batch size is the only argument |
| [scripts/analytic_all.py](scripts/analytic_all.py) | **The same study in closed form**, no GPU, one second |
| [scripts/sweep.slurm](scripts/sweep.slurm) | SLURM array job, one measurement per task |
| [scripts/submit_sweep.sh](scripts/submit_sweep.sh) | Regenerates the config, sizes `--array`, submits |

## Install

```bash
uv sync --extra viz         # or: pip install -e '.[viz]'
```

`viz` pulls in matplotlib, which only the machine that makes figures needs --
the compute nodes just need torch, so plain `uv sync` is enough there.

`requires-python` is `>=3.10,<3.13`: torch publishes no wheels for 3.14. On a
cluster you will usually load the site torch module instead and just put the
repo root on `PYTHONPATH`.

Every entry point runs three ways, so an editor's run button works as well as
the command line:

```bash
python -m scaling.sweep --count     # as a module (what the SLURM script uses)
python main.py sweep --count        # via the dispatcher
python scaling/sweep.py --count     # straight at the file
```

The last one needs no `PYTHONPATH`: each runnable module puts the repo root on
the path itself when it finds it has no parent package. Relative output paths
are still resolved against the current directory, so run from the repo root.

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

## Cost without running anything

[scaling/analytic.py](scaling/analytic.py) computes the same quantities in
closed form, straight from the configuration. A point that would OOM costs no
more to evaluate than one that fits, which is what extrapolating a scaling law
needs.

```bash
python -m scaling.analytic --model resnet50 --batch-size 64
python -m scaling.analytic --model vit_small --set depth=24 --batch-size 256 --group
python -m scaling.analytic --model resnet50 --set width_mult=4 --json
```

```
resnet50  width_mult=1.0 resolution=224
input 64x3x224x224   optimizer=sgd_momentum  params=fp32 activations=fp32

  parameters               25,557,032      97.49 MiB
  buffers (BN)                 53,173       0.20 MiB

compute, per image
  forward                4.089 GMACs      8.178 GFLOPs
  backward               8.060 GMACs     16.121 GFLOPs
  training step         12.150 GMACs     24.299 GFLOPs   (2.97x forward)

training memory, batch 64
  parameters               97.49 MiB
  gradients                97.49 MiB
  optimizer state          97.49 MiB   (1 x params)
  BN buffers                0.20 MiB
  saved activations      5109.20 MiB   (20,926,440 elems/img)
  argmax indices           98.00 MiB
  input batch              36.75 MiB
  ----------------------------------
  total                  5536.63 MiB
```

Three conventions, all per image:

- **FLOPs** counts matmul-class MACs only (conv, linear, attention) — the
  convention behind "4.1 GMACs for ResNet-50" and the one `FlopCounterMode`
  uses. Normalisations, GELU and residual adds are reported separately as
  `elementwise`: they cost bandwidth, not multipliers.
- **Backward** is twice the forward for a weighted op (one matmul for the input
  gradient, one for the weight gradient), except at the layer that touches the
  image, whose input gradient is never needed. A training step is therefore
  2.97x the forward, not exactly 3x.
- **Saved activations** are the tensors autograd keeps from the forward until
  the backward consumes them, counted per *distinct storage*. This is where an
  analytic model usually goes wrong and the reason it is worth having:
  `relu(inplace=True)` saves the batch-norm buffer it overwrote rather than a
  new one, `q/k/v` are strided views of one `qkv` tensor, a `reshape` of the
  transposed attention output is free, and `bn3` and the downsample branch feed
  only the residual add, so their outputs die immediately. Flash attention
  keeps a per-row log-sum-exp and recomputes the scores, so ViT memory is
  linear in the sequence length — `--attn-impl math` shows the quadratic
  alternative.

Flags: `--optimizer sgd|sgd_momentum|adam|adamw` (how many parameter-sized
buffers to charge), `--act-dtype bf16` for an autocast run (activations halve,
weights and optimizer state do not), `--images N` to turn the per-step cost
into a training budget, `--table`/`--group` for the per-op breakdown, `--json`.

### Is it right?

Every term has a ground truth, and `--verify` builds the model and checks
against it: parameters from the module, MACs from `FlopCounterMode`, saved
activations from `saved_tensors_hooks` deduplicated by storage. It agrees
**exactly**, on CPU, at any batch size — that is what
[tests/test_analytic.py](tests/test_analytic.py) asserts.

```bash
python -m scaling.analytic --model vit_small --verify
```

Against the real thing, `--compare` puts the model next to a measured sweep:

```bash
python -m scaling.analytic --compare results/sweep.csv
```

Parameters and MACs match to the digit. Predicted training memory lands within
a few percent of `max_memory_allocated` on an A100 across all 43 runs, low only
where the configuration is small enough for the CUDA context to dominate. The
sum of the terms *models* the peak rather than bounding it: the backward also
holds activation gradients and cuDNN workspaces, while
`zero_grad(set_to_none=True)` means the gradients are not live during the
forward.

The inference figure is deliberately weaker — weights plus the largest live
pair of tensors, a floor. Measured inference peak is set by the workspaces
`cudnn.benchmark` picks, which no closed form predicts.

### The whole sweep, analytically

```bash
python scripts/analytic_all.py 64
```

The counterpart of `run_all.py`: same one-factor-at-a-time grid, same run ids,
but costed instead of measured, then plotted. No GPU, no subprocesses, about a
second.

```
configs/sweep.jsonl (the same grid) -> results/analytic.csv -> results/figures/*.pdf
```

The figures land beside the measured ones and never overwrite them — the stems
differ (`flops_vs_flops`, `flops_vs_memory`, `params_vs_memory`,
`<model>_axes_analytic`). When `results/sweep.csv` exists the run ends with the
two tables side by side, so you can see how far the model is from the machine.

| Figure | What it shows |
| --- | --- |
| `<model>_axes_analytic.pdf` | One model, every axis superposed, three panels |
| `flops_vs_flops.pdf` | Training arithmetic against inference arithmetic |
| `flops_vs_memory.pdf` | Training memory against training arithmetic |
| `params_vs_memory.pdf` | **Training memory against parameter count** |

The first figure is the null result worth having: training FLOPs sit at
`k = 1.00` above inference FLOPs on every axis of both models, because the
backward is 2x the forward whatever you scale. All the curvature in the
*measured* time-vs-time figure is therefore hardware — arithmetic intensity,
kernel choice, occupancy — and none of it is arithmetic.

The last one is the other side of that coin: along `resolution` (ResNet) and
`num_patches` (ViT) the trajectory is a **vertical line**, an order of magnitude
of training memory at a parameter count that does not move. Those panels carry
no fitted exponent, because a power law across a 0.6% x-range is a number about
the range, not about the model.

Flags mirror the cost command — `--optimizer`, `--act-dtype bf16`,
`--attn-impl`, `--models`, `--theme dark`, `--formats png` — plus `--csv` and
`--out-dir`. Since nothing is executed, the natural next step is to widen
`SWEEP_VALUES` in [scaling/sweep.py](scaling/sweep.py) past what any GPU here
could hold: the extra points cost nothing and the figures extend straight
through them.

To drive the two halves by hand, the CSV is plain:

```bash
python -m scaling.analytic --sweep-csv results/analytic.csv --batch-size 64
python -m scaling.plot --csv results/analytic.csv --figures overlay flops_memory
```


## Everything at once

On a single GPU (an interactive `srun`, a workstation, one node), this is the
whole study in one command -- the batch size is the only thing you supply:

```bash
python scripts/run_all.py 128
```

It generates the sweep, measures every configuration, merges the results and
writes the figures:

```
configs/sweep.jsonl -> results/raw/local.jsonl -> results/sweep.csv -> results/figures/*.pdf
```

Each configuration runs in its own subprocess, exactly as the SLURM array does:
one code path for both, a fresh CUDA context per measurement, and a run that
segfaults or gets OOM-killed costs one data point instead of the whole sweep.
It **resumes by default**, so re-running after an interruption only measures
what is missing; `--fresh` starts over. `--models resnet50` restricts the sweep
and `--device cpu` gives a (slow, memory-free) smoke test.

For the cluster, use `scripts/submit_sweep.sh` instead -- same work, in
parallel. The sections below are the individual steps, should you want them.

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

## Figures

```bash
python -m scaling.plot                       # results/sweep.csv -> results/figures/
python -m scaling.plot --raw 'results/raw/*.jsonl'   # skip the CSV step
python -m scaling.plot --theme dark --formats png    # for slides
```

Four figures, as PDF for `\includegraphics` plus PNG for a quick look:

| File | What it shows |
| --- | --- |
| `<model>_axes.pdf` | **One model, every axis superposed** -- both relationships side by side |
| `time_vs_time.pdf` | Inference time against training time, one panel per axis |
| `time_vs_memory.pdf` | Training time against peak training memory, one panel per axis |

The `<model>_axes` pair is the comparison figure: all of a model's scaling axes
drawn on the same pair of panels, so you can see directly where they diverge.
The other two are the per-axis detail views -- **small multiples**, one panel per
axis, that axis highlighted in blue against every other run in gray.

Use `--figures overlay` (or `time` / `memory`) to render a subset. The analytic
kinds (`flops`, `flops_memory`, `params_memory`) go through the same command and
are described under [Cost without running anything](#cost-without-running-anything);
whichever kinds the CSV has no columns for are skipped.

How to read a panel:

- **Log-log**, so a power law `y = a x^k` is a straight line; the fitted `k` is
  printed in each panel's corner.
- The **dashed slope-1 guide** through the base configuration is the null
  hypothesis: if a trajectory tracks it, that axis buys training cost strictly
  in proportion to inference cost. Where it bends away, the axis is changing the
  compute/memory balance -- which is the interesting part.
- The **neutral ring** marks the base configuration. It is shared by every axis,
  so every trajectory passes through it.
- In the superposed figures each trajectory is **labelled at its end** and the
  legend carries that panel's fitted `k`; in the small multiples the two
  **endpoints are labelled** with their swept value. The rest is on the axis and
  in `results/sweep.csv`, which doubles as the table view.

In the small multiples panels share both scales, so a trajectory steeper than its
neighbour really is steeper.

On colour: superposing ViT-S needs four distinguishable series, and in a form
where any two marks can sit side by side the default slot order only carries
three. Enumerating all 70 four-hue subsets of the palette against that test in
both themes leaves exactly two survivors; `scaling/plot.py` uses the one that
keeps the leading blue. Its two conditional warnings -- dark-mode CVD Delta-E 6.9,
and yellow/magenta under 3:1 on the light surface -- are legal only with
secondary encoding, which is why every trajectory also carries its own marker
shape and a direct end label. Identity never rests on colour alone, so the
figures survive greyscale printing and colour-vision deficiency.

## Tests

```bash
python -m pytest tests -q
```
