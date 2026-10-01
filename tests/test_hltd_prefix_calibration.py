from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts import diagnose_hltd_prefix_calibration as diagnostic
from scripts import hltd_prefix_interpolation as interpolation
from scripts.hltd_prefix_transfer import FrozenPrefixField


@pytest.fixture
def field():
    points = np.zeros((12, 4))
    points[:, 0] = 1
    points[:, 1] = np.arange(12) * .01
    vectors = np.tile([0, 0, .1, 0], (12, 1))
    atlas = FrozenPrefixField(np.zeros(4), np.eye(4), points, vectors,
        tuple(["a"] * 4 + ["b"] * 4 + ["c"] * 4), tuple([1, 2, 3, 4] * 3),
        tuple((p, p * 64) for p in ("a", "b", "c")), 3., 1., 1e-6)
    return interpolation.InterpolatedField(atlas, 8, .2, 1.)


@pytest.fixture
def prompts():
    return {p: {"token_count": 6, "family": "test"} for p in ("a", "b", "c")}


@pytest.fixture
def method():
    return {"random_seeds": [0, 1], "shuffle_draws": 4, "shuffle_seed": 6043, "scope": "test-only"}


def test_lookup_excludes_whole_prompt_and_keeps_frozen_weights(field):
    index, distance, weight = diagnostic.donor_lookup(field)
    owners = np.array(field.atlas.node_prompts)
    assert np.all(owners[index] != owners[:, None])
    assert index.shape == (12, 8)
    assert index[0].tolist() == list(range(4, 12))
    expected = np.exp(-.5 * (distance / field.bandwidth)**2)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(weight, expected, atol=1e-15)


def test_tie_breaks_by_original_index(field):
    atlas = replace(field.atlas, points=np.tile([1., 0, 0, 0], (12, 1)))
    indices, _, weights = diagnostic.donor_lookup(replace(field, atlas=atlas))
    assert indices[0].tolist() == list(range(4, 12))
    np.testing.assert_array_equal(weights, np.full((12, 8), .125))


def test_no_fallback_when_donor_inventory_is_too_small(field):
    with pytest.raises(ValueError, match="not enough"):
        diagnostic.donor_lookup(replace(field, neighbors=9))


def test_diagnostic_aggregate_matches_original_query_math(field):
    index, _, weight = diagnostic.donor_lookup(field)
    values = diagnostic.aggregate(field.atlas.coexact, index, weight)
    # At the first query, removing all of prompt a leaves the same eight donors.
    selected = index[0]
    atlas = replace(field.atlas, points=field.atlas.points[selected], coexact=field.atlas.coexact[selected],
                    node_prompts=tuple(field.atlas.node_prompts[i] for i in selected),
                    node_tokens=tuple(field.atlas.node_tokens[i] for i in selected))
    direction = interpolation.query_hidden(replace(field, atlas=atlas), [1, 0, 0, 0], seed=0, alpha=1)
    assert direction.coexact_chart_norm == pytest.approx(np.linalg.norm(values[0]), abs=1e-14)
    np.testing.assert_allclose(direction.coexact_delta, 3 * values[0] / np.linalg.norm(values[0]), atol=1e-14)


@pytest.mark.parametrize("vectors,fraction,pairwise,effective", [
    ([[1., 0], [1., 0]], 1., 1., 2.),
    ([[1., 0], [-1., 0]], 0., -1., 2.),
    ([[1., 0], [0., 1]], np.sqrt(.5), 0., 2.),
    ([[1., 0], [0., 0]], 1., None, 1.),
    ([[0., 0], [0., 0]], None, None, None),
])
def test_resultant_and_pairwise_metrics_handle_cancellation_and_zeros(vectors, fraction, pairwise, effective):
    actual = diagnostic.donor_metrics(np.array(vectors), np.array([[0, 1]]), np.array([[.5, .5]]))
    for key, value in (("resultant_fraction", fraction), ("pairwise_agreement", pairwise), ("effective_contributors", effective)):
        if value is None:
            assert np.isnan(actual[key][0])
        else:
            assert actual[key][0] == pytest.approx(value)
    np.testing.assert_allclose(actual["contribution_shares"].sum(), 0 if fraction is None else 1)


def test_threshold_is_used_for_cosines_but_tiny_vectors_stay_in_sum():
    target = np.array([[1., 0], [0., 0], [1e-6, 0], [1., 0]])
    prediction = np.array([[1., 0], [1., 0], [1e-6, 0], [1e-7, 0]])
    values = diagnostic.cosines(target, prediction, 1e-6)
    assert values[0] == values[2] == 1 and np.isnan(values[[1, 3]]).all()
    aggregate = diagnostic.aggregate(prediction, np.array([[0, 3]]), np.array([[.5, .5]]))
    assert aggregate[0, 0] == .5 + .5e-7


def test_prompt_balancing_does_not_weight_longer_prompts_more():
    summary = diagnostic.balanced_mean([1., 1., -1., np.nan], ["a", "a", "b", "c"])
    assert summary["mean"] == 0 and summary["n_valid"] == 2 and summary["n_total"] == 3
    means = diagnostic.prompt_means([1., 1., -1., np.nan], ["a", "a", "b", "c"])
    assert means[-1] == {"prompt_id": "c", "n_valid": 0, "mean": None}


def test_comparison_uses_identical_nodes_and_reports_missing_pairs():
    result = diagnostic.paired_comparison(np.array([1., .2, np.nan]), np.array([0., np.nan, 1.]),
        np.array(["a", "b", "c"]), np.array([True, True, True]))
    assert result == {"paired_nodes": 1, "paired_prompts": 1, "actual_cosine": 1., "control_cosine": 0., "paired_gap": 1.}
    result = diagnostic.paired_comparison(np.array([np.nan]), np.array([1.]), ["a"], np.array([True]))
    assert result["paired_nodes"] == result["paired_prompts"] == 0 and result["paired_gap"] is None


def test_positions_have_explicit_zero_based_index_and_prefix_length(field, prompts):
    index, distances, weights = diagnostic.donor_lookup(field)
    metrics = diagnostic.donor_metrics(field.atlas.coexact, index, weights)
    rows, _, _, _ = diagnostic.node_rows(field, prompts, index, distances, weights, metrics)
    assert [r["prefix_length"] for r in rows[:4]] == [2, 3, 4, 5]
    assert [r["position_bin"] for r in rows[:4]] == ["early", "middle", "middle", "late"]
    assert rows[0]["relative_position"] == .2
    for row in rows:
        assert sum(row[f"geometric_mass_{name}"] for name in diagnostic.BIN_NAMES) == pytest.approx(1)
        assert sum(row[f"contribution_mass_{name}"] for name in diagnostic.BIN_NAMES) == pytest.approx(1)


def test_shuffle_preserves_each_prompt_inventory_and_original_indexing(field):
    owners = np.array(field.atlas.node_prompts)
    draws = diagnostic.shuffled_indices(owners, 4, 6043)
    np.testing.assert_array_equal(draws, diagnostic.shuffled_indices(owners, 4, 6043))
    assert any(not np.array_equal(row, np.arange(len(owners))) for row in draws)
    for row in draws:
        np.testing.assert_array_equal(owners[row], owners)
        for owner in set(owners):
            assert sorted(row[owners == owner]) == np.flatnonzero(owners == owner).tolist()


def test_random_controls_keep_node_norms_zeros_and_frozen_seed_recipe(field):
    values = field.atlas.coexact.copy()
    values[0] = 0
    actual = diagnostic.random_vectors(values, 7)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), np.linalg.norm(values, axis=1), atol=1e-15)
    raw = np.random.default_rng(np.random.SeedSequence([7, 4])).normal(size=4)
    np.testing.assert_array_equal(actual[4], np.linalg.norm(values[4]) * raw / np.linalg.norm(raw))
    assert not actual[0].any()


def test_analysis_handles_zero_target_and_preserves_out_of_support_diagnostics(field, prompts, method):
    vectors = field.atlas.coexact.copy()
    vectors[0] = 0
    local = replace(field, atlas=replace(field.atlas, coexact=vectors, support_limit=0.))
    summary, rows, prompt_rows, positions, controls, arrays = diagnostic.analyze(local, prompts, method)
    assert summary["recovery_status_counts"] == {"INACTIVE_TARGET": 1, "DEFINED": 11}
    assert summary["distance_supported_nodes"] == 0
    assert summary["defined_recovery_nodes"] == 11 and summary["supported_recovery_nodes"] == 0
    assert summary["statistics"]["recovery_cosine"]["equal_prompt"]["mean"] == pytest.approx(1)
    assert rows[0]["recovery_cosine"] is None
    assert len(prompt_rows) == 3 and len(positions) == 9 and len(controls) == 12
    assert arrays["neighbor_indices"].shape == (12, 8)
    assert summary["model_forward_calls"] == 0
    json.dumps(summary, allow_nan=False)


def test_zero_prediction_is_not_rescued_by_nearest_or_random(field, prompts, method):
    atlas = replace(field.atlas, coexact=np.zeros_like(field.atlas.coexact))
    summary, rows, _, _, _, _ = diagnostic.analyze(replace(field, atlas=atlas), prompts, method)
    assert summary["defined_recovery_nodes"] == 0
    assert summary["statistics"]["recovery_cosine"]["equal_prompt"]["mean"] is None
    assert all(r["resultant_fraction"] is None for r in rows)
    json.dumps(summary, allow_nan=False)


def test_active_targets_with_cancelled_donors_remain_undefined(field, prompts, method):
    values = field.atlas.coexact.copy()
    values[8:] *= -1
    atlas = replace(field.atlas, points=np.tile([1., 0, 0, 0], (12, 1)), coexact=values)
    summary, rows, _, _, _, _ = diagnostic.analyze(replace(field, atlas=atlas), prompts, method)
    assert all(r["recovery_status"] == "INACTIVE_PREDICTION" for r in rows[:4])
    assert all(r["recovery_cosine"] is None and r["resultant_fraction"] == pytest.approx(0, abs=1e-15) for r in rows[:4])
    assert summary["defined_recovery_nodes"] < 12


def test_rotation_preserves_direction_recovery_and_cancellation(field, prompts, method):
    rotation, _ = np.linalg.qr(np.random.default_rng(10).normal(size=(4, 4)))
    atlas = field.atlas
    changed = replace(atlas, points=atlas.points @ rotation, coexact=atlas.coexact @ rotation)
    original = diagnostic.analyze(field, prompts, method)[0]
    rotated = diagnostic.analyze(replace(field, atlas=changed), prompts, method)[0]
    for key in ("recovery_cosine", "resultant_fraction", "pairwise_agreement", "effective_contributors"):
        assert original["statistics"][key]["equal_prompt"]["mean"] == pytest.approx(rotated["statistics"][key]["equal_prompt"]["mean"], abs=1e-12)


@pytest.mark.parametrize("what", ["count", "tokens", "owners"])
def test_node_inventory_rejects_missing_or_reordered_calibration(field, prompts, what):
    atlas = field.atlas
    if what == "count":
        prompts["a"]["token_count"] += 1
    elif what == "tokens":
        atlas = replace(atlas, node_tokens=atlas.node_tokens[::-1])
    else:
        atlas = replace(atlas, node_prompts=atlas.node_prompts[::-1])
    with pytest.raises(ValueError):
        diagnostic.validate_inventory(atlas, prompts)


def test_real_contract_only_opens_pinned_calibration_inputs():
    opened = []
    original = Path.open
    def tracked(path, *args, **kwargs):
        opened.append(str(path))
        assert "evaluation_observations" not in str(path) and "raw_treatments" not in str(path)
        assert "calibration_hidden" not in str(path) and "model.safetensors" not in str(path)
        return original(path, *args, **kwargs)
    with patch.object(Path, "open", tracked):
        expected = diagnostic.contract()
    assert expected["readout"]["neighbors"] == 8 and len(expected["prompts"]) == 40
    assert any("atlas.npz" in path for path in opened)


@pytest.mark.parametrize("key", ["readout", "method", "prompts", "frozen_files"])
def test_mutated_contract_fails_before_any_analysis(key, tmp_path):
    expected = {"readout": 1, "method": 2, "prompts": 3, "frozen_files": []}
    changed = {**copy.deepcopy(expected), "frozen_utc": "test"}
    changed[key] = None
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(changed))
    with patch.object(diagnostic, "contract", return_value=expected), patch.object(diagnostic, "analyze") as analyze:
        with pytest.raises(ValueError, match="changed"):
            diagnostic.run(path, tmp_path)
        analyze.assert_not_called()


def test_existing_paths_are_not_rewritten(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("preserve")
    with patch.object(diagnostic, "contract") as contract:
        with pytest.raises(FileExistsError):
            diagnostic.freeze(path, tmp_path)
        contract.assert_not_called()
    assert path.read_text() == "preserve"


def test_canonical_input_tampering_is_rejected(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text('{}')
    with pytest.raises(ValueError, match="canonical input changed"):
        diagnostic.pinned_json(path, "0" * 64)


def test_small_end_to_end_records_inventory_and_output_receipts(field, prompts, method, tmp_path):
    inputs = {"diagnostic_id": "test", "frozen_files": [], "output": diagnostic.OUTPUT}
    path = tmp_path / "protocol.json"
    with patch.object(diagnostic, "contract", return_value=inputs), \
            patch.object(diagnostic, "load_inputs", return_value=(field, prompts, method, [])):
        diagnostic.freeze(path, tmp_path)
        summary = diagnostic.run(path, tmp_path)
        with pytest.raises(FileExistsError):
            diagnostic.run(path, tmp_path)
    output = tmp_path / diagnostic.OUTPUT
    manifest = json.loads((output / "manifest.json").read_text())
    for record in manifest["outputs"]:
        assert diagnostic.file_receipt(Path(record["path"])) == record
    assert summary["nodes"] == 12 and len(manifest["outputs"]) == 6
