"""Figures for the sweep: inference vs training time, training time vs memory.

Design notes
------------
Both figures are **small multiples**: one panel per scaling axis, each panel
showing that axis' trajectory highlighted against every other run in gray.
The alternative -- one panel with six coloured series -- is not available: in a
scatter/small-multiple form any two marks can sit side by side, and the
categorical palette only carries three series under that all-pairs test.
Emphasis-plus-context says the same thing with one hue and no legend ambiguity.

Axes are log-log, so a power law ``y = a * x^k`` is a straight line of slope
``k``; each panel is annotated with the fitted ``k`` and carries a slope-1
guide through the base configuration, which is the "y is simply proportional
to x" null hypothesis these plots exist to test.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Any, Iterable, NamedTuple

import matplotlib

matplotlib.use("Agg")  # importable on a headless login node
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

# Running a package module straight from an editor ("python scaling/sweep.py")
# leaves it without a parent package, so the relative imports below fail. Put
# the repo root on the path and name the package, so both that and the
# supported "python -m scaling.sweep" work.
if __package__ in (None, ""):
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "scaling"

from .models import MODEL_AXES  # noqa: E402

# --------------------------------------------------------------------------- #
# theme -- values taken from the validated reference palette; slots 1 and 2
# pass all six checks all-pairs in both modes (CVD dE 24.7 light / 26.8 dark).
# --------------------------------------------------------------------------- #

class Theme(NamedTuple):
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    context: str            # de-emphasised "every other run" marks
    ramp: tuple[str, ...]   # up to four superposable series (see below)


#: Superposing every axis of one model needs four distinguishable series, and in
#: a form where any two marks can sit side by side the default slot order only
#: carries three.  Enumerating all 70 four-hue subsets of the documented palette
#: against the all-pairs test in both modes leaves exactly two survivors; this is
#: the one that keeps the default's leading blue (slots 1, 4, 5, 6).  Both of its
#: warnings -- dark-mode CVD dE 6.9, and yellow/magenta under 3:1 on the light
#: surface -- are conditionally legal *only* with secondary encoding, which is
#: why every trajectory also carries its own marker shape and a direct end label,
#: and why results/sweep.csv is cited as the table view.
THEMES = {
    "light": Theme(surface="#fcfcfb", text_primary="#0b0b0b",
                   text_secondary="#52514e", text_muted="#87867f",
                   grid="#e6e5e1", context="#c9c8c2",
                   ramp=("#2a78d6", "#eda100", "#e87ba4", "#008300")),
    "dark": Theme(surface="#1a1a19", text_primary="#ffffff",
                  text_secondary="#c3c2b7", text_muted="#8f8e85",
                  grid="#333331", context="#55554f",
                  ramp=("#3987e5", "#c98500", "#d55181", "#008300")),
}

#: Shape is the second identity channel, carried alongside the ramp above.
MARKERS = ("o", "s", "^", "D")

AXIS_LABELS = {
    "width_mult": "channel width multiplier",
    "resolution": "input resolution (px)",
    "embed_dim": "embedding dimension",
    "depth": "depth (blocks)",
    "mlp_dim": "MLP hidden dimension",
    "num_patches": "sequence length (patches)",
}

#: Terse forms for the direct labels drawn at each trajectory's end.
AXIS_SHORT = {
    "width_mult": "width", "resolution": "resolution", "embed_dim": "embed dim",
    "depth": "depth", "mlp_dim": "MLP dim", "num_patches": "seq len",
}

MODEL_LABELS = {"resnet50": "ResNet-50", "vit_small": "ViT-S"}

#: (x, y) pairs each figure plots, with labels and output stem.
FIGURES = {
    "time": {
        "x": ("infer_ms_median", "inference time per batch (ms)"),
        "y": ("train_ms_median", "training time per step (ms)"),
        "title": "Training cost against inference cost",
        "stem": "time_vs_time",
    },
    "memory": {
        "x": ("train_ms_median", "training time per step (ms)"),
        "y": ("train_peak_mem_mib", "peak training memory (MiB)"),
        "title": "Training memory against training time",
        "stem": "time_vs_memory",
    },
}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def load_rows(path: str, raw: bool = False) -> list[dict[str, Any]]:
    """Read the aggregated CSV, or raw per-task JSONL when ``raw`` is set."""
    if raw:
        from .aggregate import load
        rows = load([path])
    else:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))

    out = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        row = dict(row)
        for key in ("swept_value", "infer_ms_median", "train_ms_median",
                    "train_peak_mem_mib", "params", "fwd_macs_per_image"):
            row[key] = _num(row.get(key))
        out.append(row)
    return out


def panels(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(model, axis) pairs present in the data, in a stable reading order."""
    present = {(r["model"], r["axis"]) for r in rows if r["axis"] not in ("", "base")}
    ordered = [(m, a) for m in MODEL_AXES for a in MODEL_AXES[m]]
    return [p for p in ordered if p in present]


def trajectory(rows: list[dict[str, Any]], model: str, axis: str,
               xk: str, yk: str) -> list[tuple[float, float, float]]:
    """(swept_value, x, y) along one axis, base configuration included.

    The base run is stored with ``axis == "base"``, but it lies on *every* axis'
    trajectory at that axis' base value -- without it each curve has a hole in
    the middle.
    """
    points = []
    for row in rows:
        if row["model"] != model:
            continue
        if row["axis"] == axis:
            value = row["swept_value"]
        elif row["axis"] == "base":
            value = float(MODEL_AXES[model][axis])
        else:
            continue
        # .get, not [] -- a results file written before a column existed should
        # cost that panel, not crash the whole run.
        x, y = row.get(xk), row.get(yk)
        if None in (value, x, y) or x <= 0 or y <= 0:
            continue
        points.append((value, x, y))
    return sorted(points)


def fit_exponent(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    """Slope of log(y) against log(x): the power-law exponent k in y ~ x^k."""
    xs, ys = np.asarray(list(xs), float), np.asarray(list(ys), float)
    if len(xs) < 3 or np.ptp(np.log(xs)) < 1e-6:
        return None
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def _fmt_value(value: float) -> str:
    return f"{value:g}"


def _minor_tick(value: float, _pos: int) -> str:
    """Plain-number labels for the 2x/5x minor ticks between decades."""
    return f"{value:g}"


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _style(theme: Theme) -> None:
    plt.rcParams.update({
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.edgecolor": theme.grid,
        "axes.linewidth": 0.6,
        "axes.labelcolor": theme.text_secondary,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "axes.titlecolor": theme.text_primary,
        "xtick.color": theme.text_muted,
        "ytick.color": theme.text_muted,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # Solid hairline grid, one shade off the surface -- never dashed.
        "grid.color": theme.grid,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.fontsize": 8.0,
        "figure.dpi": 140,
    })


def _draw_panel(ax, rows, model, axis, xk, yk, theme) -> None:
    # Context: every other successful run, recessive.
    ctx = [(r.get(xk), r.get(yk)) for r in rows
           if r.get(xk) and r.get(yk) and (r["model"], r["axis"]) != (model, axis)]
    if ctx:
        ax.scatter(*zip(*ctx), s=7, c=theme.context, linewidths=0, zorder=1)

    points = trajectory(rows, model, axis, xk, yk)
    if not points:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color=theme.text_muted)
        return

    values, xs, ys = zip(*points)
    base = float(MODEL_AXES[model][axis])

    # Slope-1 guide through the base point: the "strictly proportional" null.
    bx, by = next(((x, y) for v, x, y in points if v == base), (xs[0], ys[0]))
    span = [min(xs) * 0.75, max(xs) * 1.35]
    ax.plot(span, [by * s / bx for s in span], color=theme.text_muted,
            lw=0.7, ls=(0, (4, 3)), zorder=2)

    ax.plot(xs, ys, color=theme.ramp[0], lw=1.6, zorder=3, solid_capstyle="round")
    # A surface-coloured ring keeps overlapping markers separable.
    ax.plot(xs, ys, "o", ms=4.6, mfc=theme.ramp[0], mec=theme.surface,
            mew=1.0, ls="none", zorder=4)
    ax.plot([bx], [by], "o", ms=7.0, mfc="none", mec=theme.text_primary,
            mew=1.3, zorder=5)

    # Direct-label the two ends only -- a value on every point would be noise.
    # The low end goes below the curve and the high end above it, away from the
    # rising trajectory, on a surface-coloured pad so the guide line cannot
    # strike through the digits.
    pad = dict(facecolor=theme.surface, edgecolor="none", pad=0.9, alpha=0.9)
    for (value, x, y), dy, va in (((values[0], xs[0], ys[0]), -9, "top"),
                                  ((values[-1], xs[-1], ys[-1]), 9, "bottom")):
        ax.annotate(_fmt_value(value), (x, y), textcoords="offset points",
                    xytext=(0, dy), ha="center", va=va, fontsize=7,
                    color=theme.text_secondary, zorder=7, bbox=pad)

    k = fit_exponent(xs, ys)
    label = f"{MODEL_LABELS.get(model, model)} · {AXIS_LABELS.get(axis, axis)}"
    ax.set_title(label, loc="left", pad=6)
    if k is not None:
        ax.text(0.03, 0.94, f"$y \\propto x^{{{k:.2f}}}$", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color=theme.text_muted)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Decade gridlines plus a fainter 2x/5x subdivision: on a log axis a reader
    # cannot interpolate between decades without them.
    ax.grid(True, which="major", zorder=0)
    ax.grid(True, which="minor", zorder=0, alpha=0.45, linewidth=0.4)
    for sub_axis in (ax.xaxis, ax.yaxis):
        sub_axis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0), numticks=20))
        sub_axis.set_minor_formatter(FuncFormatter(_minor_tick))
    ax.tick_params(which="minor", labelsize=6, length=2)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def model_axes(rows: list[dict[str, Any]], model: str) -> list[str]:
    """Axes of one model present in the data, in the declared order."""
    present = {r["axis"] for r in rows if r["model"] == model}
    return [a for a in MODEL_AXES[model] if a in present]


def _label_ends(ax, ends, theme, min_gap_px: float = 11.0) -> None:
    """Direct-label each trajectory's end, nudged apart so the labels stay legible.

    Trajectories that finish close together would otherwise overprint each
    other's labels, and these labels are the identity channel that does not
    depend on colour -- they have to survive. Must run after the layout is
    final, since the nudge is computed in display pixels.
    """
    if not ends:
        return
    placed = sorted(((ax.transData.transform((x, y))[1], axis, x, y, colour)
                     for axis, x, y, colour in ends))
    pad = dict(facecolor=theme.surface, edgecolor="none", pad=0.9, alpha=0.9)
    dpi_to_points = 72.0 / ax.figure.dpi
    cursor = -1e9
    for py, axis, x, y, colour in placed:
        target = max(py, cursor + min_gap_px)
        cursor = target
        # A label pushed clear of its neighbour needs a leader, or it is no
        # longer obvious which trajectory it names.
        nudged = abs(target - py) > 3.0
        ax.annotate(
            AXIS_SHORT.get(axis, axis), (x, y), textcoords="offset points",
            xytext=(8 if nudged else 6, (target - py) * dpi_to_points),
            ha="left", va="center", fontsize=7, color=theme.text_secondary,
            zorder=7, bbox=pad, annotation_clip=False,
            arrowprops=(dict(arrowstyle="-", color=colour, lw=0.6,
                             shrinkA=1.0, shrinkB=2.0) if nudged else None))


def _draw_overlay_panel(ax, rows, model, axes, xk, yk, theme):
    """All of one model's axes superposed.

    Returns ``(drew_anything, legend handles, trajectory end points)``; the end
    points are labelled later, once the layout is final.
    """
    from matplotlib.lines import Line2D

    handles, base_xy = [], None
    drawn = []
    for i, axis in enumerate(axes):
        points = trajectory(rows, model, axis, xk, yk)
        if len(points) < 2:
            continue
        values, xs, ys = zip(*points)
        colour = theme.ramp[i % len(theme.ramp)]
        marker = MARKERS[i % len(MARKERS)]
        base = float(MODEL_AXES[model][axis])
        hit = next(((x, y) for v, x, y in points if v == base), None)
        if hit:
            base_xy = hit

        ax.plot(xs, ys, color=colour, lw=1.5, zorder=3, solid_capstyle="round")
        ax.plot(xs, ys, marker, ms=4.4, mfc=colour, mec=theme.surface, mew=0.9,
                ls="none", zorder=4)
        drawn.append((axis, colour, marker, xs[-1], ys[-1]))

        k = fit_exponent(xs, ys)
        label = AXIS_LABELS.get(axis, axis)
        handles.append(Line2D([], [], color=colour, lw=1.5, marker=marker, ms=4.4,
                              mfc=colour, mec=theme.surface, mew=0.9,
                              label=f"{label}" + (f"   $k$={k:.2f}" if k else "")))

    if not drawn:
        return False, [], []

    # Every trajectory runs through the base configuration; ring it once.
    if base_xy:
        ax.plot([base_xy[0]], [base_xy[1]], "o", ms=9.0, mfc="none",
                mec=theme.text_primary, mew=1.2, zorder=6)
        xmin = min(x for *_, x, _ in drawn)
        span = [xmin * 0.7, max(x for *_, x, _ in drawn) * 1.5]
        ax.plot(span, [base_xy[1] * t / base_xy[0] for t in span],
                color=theme.text_muted, lw=0.7, ls=(0, (4, 3)), zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", zorder=0)
    ax.grid(True, which="minor", zorder=0, alpha=0.45, linewidth=0.4)
    for sub_axis in (ax.xaxis, ax.yaxis):
        sub_axis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0), numticks=20))
        sub_axis.set_minor_formatter(FuncFormatter(_minor_tick))
    ax.tick_params(which="minor", labelsize=6, length=2)
    ax.set_axisbelow(True)
    # Room on the right for the direct end labels.
    ax.set_xmargin(0.18)
    ax.set_ymargin(0.10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return True, handles, [(a, x, y, c) for a, c, _m, x, y in drawn]


def make_overlay_figure(rows: list[dict[str, Any]], model: str, theme: Theme):
    """One model, every scaling axis superposed, one panel per relationship."""
    from matplotlib.lines import Line2D

    axes_names = model_axes(rows, model)
    if not axes_names:
        return None

    kinds = [k for k in ("time", "memory")
             if any(trajectory(rows, model, a, FIGURES[k]["x"][0],
                               FIGURES[k]["y"][0]) for a in axes_names)]
    if not kinds:
        return None

    fig, panels_ = plt.subplots(1, len(kinds), figsize=(5.0 * len(kinds), 4.3),
                                squeeze=False)
    pending = []
    for ax, kind in zip(panels_[0], kinds):
        spec = FIGURES[kind]
        ok, handles, ends = _draw_overlay_panel(ax, rows, model, axes_names,
                                                spec["x"][0], spec["y"][0], theme)
        pending.append((ax, ends))
        ax.set_xlabel(spec["x"][1])
        ax.set_ylabel(spec["y"][1])
        ax.set_title(spec["title"], loc="left", pad=6, fontsize=9)
        if ok:
            # Legend inside the empty lower-right corner, carrying the fitted
            # exponent for THIS panel -- k differs between the two panels.
            ax.legend(handles=handles, loc="lower right", fontsize=7.5,
                      labelcolor=theme.text_secondary, handlelength=1.9,
                      borderpad=0.6, labelspacing=0.5)

    fig.suptitle(f"{MODEL_LABELS.get(model, model)}: every scaling axis superposed",
                 x=0.008, y=0.995, ha="left", fontsize=11.5,
                 color=theme.text_primary)
    sub = ("log-log · $k$ is the fitted exponent of $y \\propto x^{k}$ · "
           f"batch size {rows[0].get('batch_size', '?')}"
           + (f" · {rows[0]['gpu_name']}" if rows[0].get("gpu_name") else ""))
    fig.text(0.008, 0.945, sub, ha="left", fontsize=8, color=theme.text_muted)

    furniture = [
        Line2D([], [], color="none", marker="o", ms=9.0, mfc="none",
               mec=theme.text_primary, mew=1.2,
               label="base configuration (shared by every axis)"),
        Line2D([], [], color=theme.text_muted, lw=0.7, ls=(0, (4, 3)),
               label="slope 1 (strict proportionality)"),
    ]
    fig.legend(handles=furniture, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.2,
               labelcolor=theme.text_secondary, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    # Labels go on last: their placement is computed in display pixels, which
    # tight_layout would otherwise invalidate.
    fig.canvas.draw()
    for ax, ends in pending:
        _label_ends(ax, ends, theme)
    return fig


def _legend(fig, theme: Theme) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=theme.ramp[0], lw=1.6, marker="o", ms=4.6,
               mfc=theme.ramp[0], mec=theme.surface, mew=1.0,
               label="swept axis (labelled at both ends)"),
        Line2D([], [], color="none", marker="o", ms=7.0, mfc="none",
               mec=theme.text_primary, mew=1.3, label="base configuration"),
        Line2D([], [], color="none", marker="o", ms=3.4, mfc=theme.context,
               mec="none", label="all other runs"),
        Line2D([], [], color=theme.text_muted, lw=0.7, ls=(0, (4, 3)),
               label="slope 1 (strict proportionality)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.2,
               labelcolor=theme.text_secondary, columnspacing=1.6)


def make_figure(rows: list[dict[str, Any]], kind: str, theme: Theme,
                ncols: int = 3):
    spec = FIGURES[kind]
    xk, xlabel = spec["x"]
    yk, ylabel = spec["y"]

    usable = [p for p in panels(rows) if trajectory(rows, *p, xk, yk)]
    if not usable:
        return None

    ncols = min(ncols, len(usable))
    nrows = math.ceil(len(usable) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.1 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()

    for ax, (model, axis) in zip(flat, usable):
        _draw_panel(ax, rows, model, axis, xk, yk, theme)
    # Headroom so the endpoint labels do not crowd the frame.
    flat[0].set_xmargin(0.06)
    flat[0].set_ymargin(0.10)
    flat[0].autoscale_view()
    for ax in flat[len(usable):]:
        ax.set_visible(False)

    # Shared scales, so one axis label per side rather than per panel.
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel(xlabel)
    for row in axes:
        row[0].set_ylabel(ylabel)
    # A column whose bottom panel is hidden needs the label on the one above.
    for col in range(ncols):
        column = [axes[r][col] for r in range(nrows)]
        visible = [ax for ax in column if ax.get_visible()]
        if visible and not column[-1].get_visible():
            visible[-1].set_xlabel(xlabel)
            visible[-1].tick_params(labelbottom=True)

    fig.suptitle(spec["title"], x=0.008, y=0.995, ha="left",
                 fontsize=11.5, color=theme.text_primary)
    sub = (f"one panel per scaling axis · log-log · "
           f"batch size {rows[0].get('batch_size', '?')}"
           + (f" · {rows[0]['gpu_name']}" if rows[0].get("gpu_name") else ""))
    fig.text(0.008, 0.958, sub, ha="left", fontsize=8, color=theme.text_muted)

    _legend(fig, theme)
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m scaling.plot",
        description="Plot the sweep: inference vs training time, time vs memory.")
    p.add_argument("--csv", default="results/sweep.csv",
                   help="aggregated CSV from scaling.aggregate")
    p.add_argument("--raw", metavar="GLOB",
                   help="read raw per-task JSONL instead of the CSV")
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--theme", choices=sorted(THEMES), default="light",
                   help="light for print/LaTeX, dark for slides")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"],
                   help="pdf for LaTeX inclusion, png for a quick look")
    p.add_argument("--figures", nargs="+",
                   choices=sorted(FIGURES) + ["overlay"],
                   default=sorted(FIGURES) + ["overlay"],
                   help="'time'/'memory' are the per-axis small multiples; "
                        "'overlay' superposes every axis of one model")
    p.add_argument("--ncols", type=int, default=3)
    args = p.parse_args(argv)

    source = args.raw or args.csv
    rows = load_rows(source, raw=bool(args.raw))
    if not rows:
        print(f"no successful runs in {source}", file=sys.stderr)
        return 1

    theme = THEMES[args.theme]
    _style(theme)
    os.makedirs(args.out_dir, exist_ok=True)

    def save(fig, stem: str) -> int:
        if args.theme != "light":
            stem += f"_{args.theme}"
        for ext in args.formats:
            path = os.path.join(args.out_dir, f"{stem}.{ext}")
            fig.savefig(path, bbox_inches="tight")
            print(f"wrote {path}")
        plt.close(fig)
        return len(args.formats)

    written = 0
    if "overlay" in args.figures:
        for model in sorted({r["model"] for r in rows}):
            fig = make_overlay_figure(rows, model, theme)
            if fig is None:
                print(f"skipping overlay for {model}: no usable trajectory",
                      file=sys.stderr)
                continue
            written += save(fig, f"{model}_axes")

    for kind in [k for k in args.figures if k != "overlay"]:
        fig = make_figure(rows, kind, theme, ncols=args.ncols)
        if fig is None:
            missing = FIGURES[kind]["y"][0]
            print(f"skipping '{kind}' figure: no run has {missing} "
                  "(memory is only recorded on CUDA)", file=sys.stderr)
            continue
        written += save(fig, FIGURES[kind]["stem"])

    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
