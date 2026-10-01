from __future__ import annotations

import builtins
import copy
import io
import json
from dataclasses import replace

import numpy as np
import pytest

from scripts import analyze_hltd_prefix_nulls_v2 as v2
from scripts import compare_hltd_prefix_nulls as comparison
from scripts import diagnose_hltd_prefix_calibration as cal
from scripts.hltd_prefix_interpolation import InterpolatedField
from scripts.hltd_prefix_transfer import FrozenPrefixField


@pytest.fixture
def field():
    points = np.zeros((12, 4))
    points[:, 0] = 1
    points[:, 1] = np.arange(12) * .01
    vectors = np.random.default_rng(12).normal(size=(12, 4)) * .1
    atlas = FrozenPrefixField(np.zeros(4), np.eye(4), points, vectors,
        tuple(["a"] * 4 + ["b"] * 4 + ["c"] * 4), tuple([1, 2, 3, 4] * 3),
        tuple((p, p * 64) for p in ("a", "b", "c")), 3., 1., 1e-6)
    return InterpolatedField(atlas, 8, .2, 1.)


@pytest.fixture
def prompts():
    return {p: {"token_count": 6, "family": "test"} for p in ("a", "b", "c")}


@pytest.fixture
def plan():
    return {"random_seeds": [0, 1], "shuffle_draws": 4, "shuffle_seed": 6043, "scope": "test-only"}


def test_position_strata_preserve_whole_vectors_norms_and_zeros(field, prompts):
    atlas = field.atlas
    owners = np.array(atlas.node_prompts)
    bins = comparison.position_bins(atlas, prompts)
    assert bins[:4].tolist() == [0, 1, 1, 2]
    permutations = comparison.stratified_indices(owners, bins, 20, 6043)
    np.testing.assert_array_equal(permutations, comparison.stratified_indices(owners, bins, 20, 6043))
    vectors = atlas.coexact.copy()
    vectors[1] = 0
    for row in permutations:
        np.testing.assert_array_equal(owners[row], owners)
        np.testing.assert_array_equal(bins[row], bins)
        for owner, b in set(zip(owners, bins, strict=True)):
            group = np.flatnonzero((owners == owner) & (bins == b))
            assert sorted(row[group]) == group.tolist()
            if len(group) == 1:
                assert row[group[0]] == group[0]
        assert sorted(np.linalg.norm(vectors[row], axis=1)) == sorted(np.linalg.norm(vectors, axis=1))
    assert np.any(permutations[:, 1] != 1) and np.any(permutations[:, 1] == 1)


@pytest.mark.parametrize("bins,draws", [([0, 1], 0), ([0, 3], 1), ([0], 1)])
def test_invalid_strata_rejected(bins, draws):
    with pytest.raises(ValueError, match="strata"):
        comparison.stratified_indices(["a", "b"], bins, draws, 0)


def test_common_mask_intersects_every_draw_and_does_not_zero_fill():
    actual = np.array([.8, .4, .3, np.nan, .6])
    controls = {"x": np.array([[0., .1, .1, 1., .1], [0., .1, np.nan, 1., .2]]),
                "y": np.array([[0., np.nan, .1, 1., .4]])}
    owners = np.array(["a", "a", "b", "b", "c"])
    supported = np.array([True, True, True, True, False])
    common, draws, comparisons, prompts = comparison.compare_scores(actual, controls, owners, supported)
    assert common.tolist() == [True, False, False, False, True]
    for row in comparisons:
        assert row["paired_nodes"] == row["paired_prompts"] == (2 if row["scope"] == "all" else 1)
    real = next(r for r in comparisons if r["scope"] == "all" and r["comparator"] == "actual")
    assert real["mean_cosine"] == pytest.approx(.7)
    assert any(r["matching"] == "per_draw" and r["paired_nodes"] > 2 for r in draws)
    assert all(r["mean"] is None and r["n_valid"] == 0 for r in prompts if r["prompt_id"] == "b")


def test_empty_common_mask_has_null_statistics():
    result = comparison.compare_scores(np.array([1.]), {"x": np.array([[np.nan]])}, ["a"], np.array([True]))
    assert not result[0].any()
    assert all(r["mean_cosine"] is None and r["paired_nodes"] == 0 for r in result[2])
    json.dumps(result[2], allow_nan=False)


def test_original_geometry_random_and_shuffle_are_unchanged(field, prompts, plan):
    summary, tables, arrays = comparison.analyze(field, prompts, plan)
    prior, _, _, _, old_controls, old_arrays = cal.analyze(field, prompts, plan)
    for key in old_arrays:
        np.testing.assert_array_equal(arrays[key], old_arrays[key])
    for old in old_controls:
        new = next(r for r in tables["draws.csv"] if r["matching"] == "per_draw"
                   and all(r[k] == old[k] for k in ("comparator", "draw", "scope")))
        for key in ("paired_nodes", "paired_prompts", "actual_cosine", "control_cosine", "paired_gap"):
            assert new[key] == old[key]
    assert summary["unrestricted_actual_mean"] == prior["statistics"]["recovery_cosine"]["equal_prompt"]["mean"]
    assert summary["model_forward_calls"] == summary["evaluation_prefix_queries"] == 0
    assert summary["strata"] == 9 and summary["singleton_strata"] == 6
    json.dumps(summary, allow_nan=False)


def test_inactive_targets_not_rescued_and_support_not_discarded(field, prompts, plan):
    atlas = replace(field.atlas, coexact=np.zeros_like(field.atlas.coexact), support_limit=0.)
    summary, tables, _ = comparison.analyze(replace(field, atlas=atlas), prompts, plan)
    assert summary["actual_defined_nodes"] == summary["common_nodes"] == 0
    assert all(r["mean_cosine"] is None for r in tables["comparison.csv"])
    assert len(tables["masks.csv"]) == 12


def test_stage_a_contract_never_opens_v2_responses_or_model_assets(monkeypatch):
    real_open = builtins.open
    def guarded_open(file, *args, **kwargs):
        path = str(file)
        assert "spiral_out_hltd_prefix_transfer_v2" not in path
        assert not path.endswith((".safetensors", "pytorch_model.bin"))
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    contract = comparison.contract()
    assert contract["plan"]["stage_order"] == ["calibration_null_comparison", "saved_v2_analysis"]


def test_freeze_is_exclusive_and_source_tampering_fails(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("original")
    def contract(root):
        return {"frozen_files": [comparison.file_receipt(source)], "setting": 1}
    monkeypatch.setattr(comparison, "contract", contract)
    path = tmp_path / "protocol.json"
    protocol = comparison.freeze(path, tmp_path)
    comparison.validate(protocol, tmp_path)
    with pytest.raises(FileExistsError):
        comparison.freeze(path, tmp_path)
    source.write_text("changed")
    with pytest.raises(ValueError):
        comparison.validate(protocol, tmp_path)


def test_stage_b_stops_before_importing_replay_without_stage_a(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "validate", lambda *args: None)
    path = tmp_path / "protocol.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Stage A"):
        v2.run(path, tmp_path)
    assert not (tmp_path / v2.OUTPUT).exists()


def test_v2_positions_use_prefix_minus_one_and_only_calibration_lengths(field, prompts):
    indices, distances, weights = cal.donor_lookup(field)
    vector = weights[0] @ field.atlas.coexact[indices[0]]
    cell = {"prompt_id": "evaluation_without_full_length", "family": "test", "prefix_length": 8,
            "neighbor_indices": indices[0].tolist(), "neighbor_distances": distances[0].tolist(),
            "weights": weights[0].tolist(), "coexact_chart_norm": float(np.linalg.norm(vector))}
    row = v2.cell_descriptors(field, prompts, {"cells": [cell]})[0]
    assert row["query_token_index"] == 7
    assert row["weighted_token_gap"] == pytest.approx(weights[0] @ abs(np.array(field.atlas.node_tokens)[indices[0]] - 7))
    assert sum(row["donor_mass_" + name] for name in cal.BIN_NAMES) == pytest.approx(1)
    cell["coexact_chart_norm"] += .01
    with pytest.raises(ValueError, match="norm"):
        v2.cell_descriptors(field, prompts, {"cells": [cell]})


def test_coefficients_average_seeds_before_cell_analysis_and_reject_incomplete_grid():
    cells = [{"prompt_id": "a", "family": "f", "prefix_length": 8}]
    coefficients = [{**cells[0], "seed": i, "coexact_odd": i, "random_odd": 0, "odd_gap": i} for i in range(8)]
    result = v2.attach_coefficients(cells, coefficients, list(range(8)))
    assert len(result) == 1 and result[0]["odd_gap"] == 3.5
    for invalid in (coefficients[:-1], coefficients + [coefficients[0]]):
        with pytest.raises(ValueError, match="grid"):
            v2.attach_coefficients(cells, invalid, list(range(8)))


@pytest.fixture
def dose_fixture():
    spec = {"evaluation": {"prefix_lengths": [8, 16, 24], "seeds": [0, 1], "components": ["coexact", "random_tangent"],
                           "alphas": [-1., -.5, -.25, .25, .5, 1.]}, "analysis": {"positive_magnitudes": [.25, .5, 1.]}}
    rows = []
    for prompt in ("a", "b"):
        for length in spec["evaluation"]["prefix_lengths"]:
            for seed in (0, 1):
                for component in spec["evaluation"]["components"]:
                    slope = 2 if component == "coexact" else -1
                    for alpha in spec["evaluation"]["alphas"]:
                        rows.append({"prompt_id": prompt, "prefix_length": length, "seed": seed, "component": component,
                                     "alpha": alpha, "next_token_logprob_base": -10.,
                                     "next_token_logprob_steered": -10 + slope * alpha - 3 * alpha**2})
    return rows, spec


def test_finite_odd_even_responses_have_correct_sign_and_baseline(dose_fixture):
    cells, summary = v2.dose_rows(*dose_fixture)
    for row in [*cells, *summary]:
        slope = 2 if row["component"] == "coexact" else -1
        assert row["odd"] == slope * row["magnitude"]
        assert row["even"] == -3 * row["magnitude"]**2
        assert row["delta_plus"] == row["even"] + row["odd"]
        assert row["delta_minus"] == row["even"] - row["odd"]
    assert len(cells) == 36 and len(summary) == 24


def test_dose_grid_rejects_missing_duplicate_or_mismatched_baseline(dose_fixture):
    rows, spec = dose_fixture
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError, match="grid"):
            v2.dose_rows(bad, spec)
    changed = copy.deepcopy(rows)
    changed[0]["next_token_logprob_base"] = -11
    with pytest.raises(ValueError, match="baseline"):
        v2.dose_rows(changed, spec)


def test_associations_report_all_scopes_and_constant_predictors_as_undefined():
    cells = [{"prompt_id": str(p), "prefix_length": n, "x": p, "constant": 1., "y": -p}
             for p in range(4) for n in (8, 16, 24)]
    plan = {"predictors": ["x", "constant"], "outcomes": ["y"],
            "association_scopes": ["pooled_cells", "prefix_8", "prefix_16", "prefix_24", "prompt_means"]}
    result = v2.association_rows(cells, plan)
    assert len(result) == 10
    for row in result:
        assert row["paired_prompts"] == 4
        assert row["paired_rows"] == (12 if row["scope"] == "pooled_cells" else 4)
        if row["predictor"] == "constant":
            assert row["spearman_rho"] is None
        else:
            assert row["spearman_rho"] == pytest.approx(-1)
    cells[0]["y"] = None
    result = v2.association_rows(cells, plan)
    assert result[0]["paired_rows"] == 11


def test_stage_a_missing_manifest_members_rejected(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    output = tmp_path / comparison.OUTPUT
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps({"protocol": comparison.file_receipt(protocol), "outputs": []}))
    with pytest.raises(ValueError, match="inventory"):
        v2.require_calibration(protocol, {}, tmp_path)


def test_prior_array_replay_rejects_signed_zero_change(tmp_path):
    output = tmp_path / comparison.OLD / "result"
    output.mkdir(parents=True)
    np.savez(output / "replay_arrays.npz", prediction=np.array([0.]))
    with pytest.raises(ValueError, match="array differs"):
        comparison.verify_old_replay({}, {"prediction": np.array([-0.])}, tmp_path)
