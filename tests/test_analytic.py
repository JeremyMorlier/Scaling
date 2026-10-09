"""The analytic cost model must agree with torch, exactly.

Every quantity in :mod:`scaling.analytic` has a ground truth that torch can
produce -- parameters from the module, MACs from ``FlopCounterMode``, saved
activations from ``saved_tensors_hooks`` -- so these tests compare the two on
small configurations rather than asserting hand-copied constants.

Run with: python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaling.analytic import (DTYPE_BYTES, OPTIMIZER_STATES, SWEEP_COLUMNS,
                              model_cost, resnet50_cost, sweep_rows, verify,
                              vit_cost, write_sweep_csv)

# Batch 2 at least: batch-norm refuses a single sample per channel in training,
# and the activation terms have to be linear in the batch to be worth checking.
VERIFY_CONFIGS = [
    ("resnet50", {"width_mult": 1.0, "resolution": 64}),
    ("resnet50", {"width_mult": 0.375, "resolution": 96}),
    ("resnet50", {"width_mult": 2.0, "resolution": 32}),
    ("vit_small", {"embed_dim": 384, "depth": 2, "mlp_dim": 1536, "num_patches": 16}),
    ("vit_small", {"embed_dim": 128, "depth": 3, "mlp_dim": 512, "num_patches": 36}),
    ("vit_small", {"embed_dim": 64, "depth": 4, "mlp_dim": 128, "num_patches": 9}),
]


@pytest.mark.parametrize("name,params", VERIFY_CONFIGS)
@pytest.mark.parametrize("batch_size", [2, 3])
def test_matches_torch_exactly(name, params, batch_size):
    report = verify(name, batch_size=batch_size, num_classes=17, **params)
    mismatched = {k: v for k, v in report.items()
                  if k != "notes" and not v["exact"]}
    assert not mismatched


def test_base_configurations_match_the_published_figures():
    resnet = resnet50_cost()
    assert resnet.params == 25_557_032
    assert resnet.fwd_macs == 4_089_184_256          # 4.1 GMACs
    vit = vit_cost()
    assert vit.params == 22_050_664
    assert vit.fwd_macs == 4_598_882_304             # 4.6 GMACs
    assert vit.config["num_heads"] == 6 and vit.config["seq_len"] == 197


def test_training_step_is_just_under_three_forwards():
    # Exactly 3x minus the input gradient the first layer never computes.
    for cost in (resnet50_cost(), vit_cost()):
        assert 2.9 < cost.backward_ratio < 3.0
        assert cost.train_flops == 2 * cost.train_macs


def test_optimizer_state_is_a_multiple_of_the_parameter_vector():
    cost = resnet50_cost()
    for name, copies in OPTIMIZER_STATES.items():
        mem = cost.memory(32, optimizer=name)
        assert mem.optimizer == copies * mem.params
    with pytest.raises(ValueError):
        cost.memory(32, optimizer="rmsprop")


def test_activations_are_linear_in_the_batch_and_halve_under_autocast():
    cost = vit_cost()
    one, two, three = (cost.memory(b) for b in (1, 2, 3))
    assert three.activations - two.activations == two.activations - one.activations
    assert two.params == one.params                  # weights do not scale
    half = cost.memory(2, act_dtype="bf16")
    assert half.activations == two.activations // 2
    assert half.params == two.params                 # autocast keeps fp32 weights


def test_activations_dominate_a_resnet_training_step():
    # The point of the exercise: a parameter count says nothing about memory.
    mem = resnet50_cost().memory(64)
    assert mem.activations_total > 10 * mem.weights_total


def test_attention_memory_is_linear_only_with_flash():
    flash = [vit_cost(num_patches=n, attn_impl="flash").saved_elems
             for n in (196, 784)]
    math = [vit_cost(num_patches=n, attn_impl="math").saved_elems
            for n in (196, 784)]
    # 4x the tokens costs 4x with flash, strictly more once the heads x N x N
    # probability matrix is kept instead.
    assert flash[1] / flash[0] == pytest.approx(4, rel=0.01)
    assert math[1] / math[0] > 1.4 * flash[1] / flash[0]
    assert vit_cost(attn_impl="math").saved_elems > vit_cost().saved_elems
    with pytest.raises(ValueError):
        vit_cost(attn_impl="xformers")


def test_resolution_and_width_move_the_two_terms_apart():
    base = resnet50_cost()
    wider = resnet50_cost(width_mult=2.0)
    bigger = resnet50_cost(resolution=448)
    # Width grows parameters quadratically, resolution leaves them untouched.
    assert wider.params / base.params == pytest.approx(4, rel=0.05)
    assert bigger.params == base.params
    # Both grow compute; only resolution grows it without any memory for weights.
    assert bigger.fwd_macs / base.fwd_macs == pytest.approx(4, rel=0.05)
    assert bigger.saved_elems / base.saved_elems == pytest.approx(4, rel=0.05)


def test_model_cost_rejects_axes_the_model_does_not_have():
    with pytest.raises(ValueError):
        model_cost("resnet50", depth=12)
    with pytest.raises(ValueError):
        model_cost("resnet101")
    assert model_cost("vit_small", depth=6).config["depth"] == 6


def test_the_analytic_sweep_covers_the_measured_grid_run_for_run():
    from scaling.sweep import generate

    rows = sweep_rows(batch_size=8)
    assert [r["run_id"] for r in rows] == [r["run_id"] for r in generate()]
    assert all(r["status"] == "ok" for r in rows)
    # Base rows report no swept value, exactly as scaling.aggregate writes them.
    assert all((r["swept_value"] == "") == (r["axis"] == "base") for r in rows)
    base = next(r for r in rows if r["run_id"] == "resnet50__base")
    assert base["params"] == 25_557_032
    assert base["fwd_macs_per_image"] == 4_089_184_256
    assert base["train_gflops_per_step"] == pytest.approx(
        base["train_gflops_per_image"] * 8)


def test_the_sweep_csv_is_readable_by_the_plotter(tmp_path):
    import csv

    path = tmp_path / "analytic.csv"
    write_sweep_csv(str(path), sweep_rows(models=["resnet50"], batch_size=4))
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [*rows[0]] == SWEEP_COLUMNS
    assert all(float(r["train_mem_mib"]) > 0 for r in rows)

    from scaling.plot import ANALYTIC, FIGURES, load_rows
    loaded = load_rows(str(path))
    assert len(loaded) == len(rows)
    # Every analytic figure must find both of its columns as numbers.
    for kind in ANALYTIC:
        for side in ("x", "y"):
            assert isinstance(loaded[0][FIGURES[kind][side][0]], float)


def test_dtype_table_is_consistent():
    assert DTYPE_BYTES["fp32"] == 4 and DTYPE_BYTES["bf16"] == 2
    cost = resnet50_cost()
    # Max-pool indices are int64 whatever the activation dtype is.
    assert (cost.saved_bytes(8, "fp32") - cost.saved_bytes(8, "bf16")
            == 8 * cost.saved_elems * 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
