"""Checks for the figure data plumbing (not the rendered pixels)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaling.plot import fit_exponent, panels, trajectory  # noqa: E402

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


def test_panels_are_in_reading_order_and_skip_absent_axes():
    assert panels(ROWS) == [("resnet50", "width_mult"), ("resnet50", "resolution")]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
