"""Checks for the figure data plumbing (not the rendered pixels)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaling.plot import (  # noqa: E402
    ANALYTIC, MARKERS, MEASURED, THEMES, fit_exponent, make_overlay_figure,
    model_axes, panels, provenance, trajectory,
)

ROWS = [
    {"model": "resnet50", "axis": "base", "swept_value": None,
     "infer_ms_median": 10.0, "train_ms_median": 30.0, "status": "ok"},
    {"model": "resnet50", "axis": "width_mult", "swept_value": 0.5,
     "infer_ms_median": 4.0, "train_ms_median": 12.0, "status": "ok"},
    {"model": "resnet50", "axis": "width_mult", "swept_value": 2.0,
     "infer_ms_median": 36.0, "train_ms_median": 108.0, "status": "ok"},
    {"model": "resnet50", "axis": "resolution", "swept_value": 128.0,
     "infer_ms_median": 3.5, "train_ms_median": 11.0, "status": "ok"},
]


def test_base_run_is_on_every_axis_trajectory():
    # The base run is stored once under axis="base" but belongs to each curve.
    traj = trajectory(ROWS, "resnet50", "width_mult",
                      "infer_ms_median", "train_ms_median")
    assert [v for v, _, _ in traj] == [0.5, 1.0, 2.0]  # base width_mult is 1.0

    traj = trajectory(ROWS, "resnet50", "resolution",
                      "infer_ms_median", "train_ms_median")
    assert [v for v, _, _ in traj] == [128.0, 224.0]  # base resolution is 224


def test_trajectory_is_sorted_and_excludes_other_axes():
    traj = trajectory(ROWS, "resnet50", "width_mult",
                      "infer_ms_median", "train_ms_median")
    assert traj == sorted(traj)
    assert 3.5 not in [x for _, x, _ in traj]  # the resolution run stays out


def test_trajectory_drops_unusable_points():
    rows = ROWS + [
        {"model": "resnet50", "axis": "width_mult", "swept_value": 4.0,
         "infer_ms_median": None, "train_ms_median": 5.0, "status": "ok"},
        {"model": "resnet50", "axis": "width_mult", "swept_value": 8.0,
         "infer_ms_median": 0.0, "train_ms_median": 5.0, "status": "ok"},
    ]
    traj = trajectory(rows, "resnet50", "width_mult",
                      "infer_ms_median", "train_ms_median")
    assert [v for v, _, _ in traj] == [0.5, 1.0, 2.0]


def test_fit_exponent_recovers_a_known_power_law():
    xs = [1.0, 2.0, 4.0, 8.0]
    assert abs(fit_exponent(xs, [x ** 1.5 for x in xs]) - 1.5) < 1e-9
    assert fit_exponent([1.0, 2.0], [1.0, 2.0]) is None       # too few points
    assert fit_exponent([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) is None  # no x spread


def test_fit_exponent_refuses_a_hair_thin_x_range():
    # A ViT's parameter count barely moves with sequence length while its
    # memory triples; fitting the two together would report k in the hundreds.
    xs = [22.0e6, 22.03e6, 22.07e6]
    assert fit_exponent(xs, [900.0, 3000.0, 7600.0]) is None
    assert fit_exponent([1.0, 1.5, 2.0], [1.0, 1.5, 2.0]) is not None


def test_panels_are_in_reading_order_and_skip_absent_axes():
    assert panels(ROWS) == [("resnet50", "width_mult"), ("resnet50", "resolution")]


def test_model_axes_follows_the_declared_order():
    assert model_axes(ROWS, "resnet50") == ["width_mult", "resolution"]
    assert model_axes(ROWS, "vit_small") == []


def test_ramp_covers_the_widest_model_and_matches_the_markers():
    # ViT-S has four axes; superposing them needs four validated slots, each
    # with its own marker shape as the colour-independent identity channel.
    from scaling.models import MODEL_AXES
    widest = max(len(a) for a in MODEL_AXES.values())
    for name, theme in THEMES.items():
        assert len(theme.ramp) >= widest, name
        assert len(set(theme.ramp)) == len(theme.ramp), f"{name} repeats a hue"
    assert len(MARKERS) >= widest


ANALYTIC_ROWS = [
    {"model": "resnet50", "axis": "base", "swept_value": None, "status": "ok",
     "source": "analytic", "optimizer": "sgd_momentum", "act_dtype": "fp32",
     "params": 25.6e6, "fwd_gflops_per_image": 8.2,
     "train_gflops_per_image": 24.3, "train_gflops_per_step": 1555.0,
     "train_mem_mib": 5536.0},
    {"model": "resnet50", "axis": "width_mult", "swept_value": 0.5,
     "status": "ok", "source": "analytic", "params": 6.9e6,
     "fwd_gflops_per_image": 2.1, "train_gflops_per_image": 6.3,
     "train_gflops_per_step": 396.0, "train_mem_mib": 2720.0},
    {"model": "resnet50", "axis": "width_mult", "swept_value": 2.0,
     "status": "ok", "source": "analytic", "params": 98.0e6,
     "fwd_gflops_per_image": 32.2, "train_gflops_per_image": 96.2,
     "train_gflops_per_step": 6158.0, "train_mem_mib": 11573.0},
]


def test_the_two_figure_groups_do_not_overlap():
    # Each table feeds one group; nothing may be drawn from both.
    assert not set(MEASURED) & set(ANALYTIC)


def test_analytic_rows_draw_the_analytic_panels_only():
    fig = make_overlay_figure(ANALYTIC_ROWS, "resnet50", THEMES["light"],
                              kinds=ANALYTIC)
    assert fig is not None
    # Three analytic relationships, wrapped two per row: 3 panels + 1 hidden.
    assert sum(ax.get_visible() for ax in fig.axes) == 3
    # The same rows carry no milliseconds, so the measured group draws nothing.
    assert make_overlay_figure(ANALYTIC_ROWS, "resnet50", THEMES["light"],
                               kinds=MEASURED) is None


def test_provenance_names_the_source():
    assert "analytic" in provenance(ANALYTIC_ROWS)
    assert "sgd_momentum" in provenance(ANALYTIC_ROWS)
    assert provenance(ROWS) == ""          # no gpu_name recorded
    assert "A100" in provenance([dict(ROWS[0], gpu_name="NVIDIA A100")])


def test_overlay_figure_builds_and_skips_absent_models():
    fig = make_overlay_figure(ROWS, "resnet50", THEMES["light"])
    assert fig is not None
    # Only the time panel is possible: these rows carry no memory column.
    assert len(fig.axes) == 1
    assert make_overlay_figure(ROWS, "vit_small", THEMES["light"]) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
