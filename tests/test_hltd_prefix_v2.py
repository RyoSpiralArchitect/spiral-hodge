from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from scripts import analyze_hltd_prefix_v2 as analysis
from scripts import hltd_prefix_interpolation as interpolation
from scripts import hltd_prefix_transfer as base
from scripts import run_hltd_prefix_v2 as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def atlas():
    points = np.zeros((16, 4))
    points[:, 0] = 1
    points[:, 1] = np.arange(16) * .01
    vectors = np.zeros_like(points)
    vectors[1:, 2] = .1
    return base.FrozenPrefixField(np.zeros(4), np.eye(4), points, vectors,
        tuple(["a"] * 8 + ["b"] * 8), tuple(range(16)), (("a", "a" * 64), ("b", "b" * 64)), 3., 10., 1e-6)


@pytest.fixture(scope="module")
def field(atlas):
    return interpolation.calibrate_readout(atlas)


@pytest.fixture(scope="module")
def protocol():
    # Real pinned input validation and tokenizer only, never a model load.
    return {**runner.contract(), "frozen_utc": "2026-10-01T00:00:00+00:00"}


def test_bandwidth_and_support_exclude_whole_calibration_prompt(atlas, field):
    owners = np.array(atlas.node_prompts)
    expected = np.concatenate([np.partition(cdist(atlas.points[owners == owner], atlas.points[owners != owner]), 7, axis=1)[:, 7]
                               for owner in ("a", "b")])
    assert field.bandwidth == np.median(expected)
    assert field.kth_support_limit == np.quantile(expected, .99, method="higher")


def test_zero_nearest_node_remains_in_the_weighted_sum(atlas, field):
    direction = interpolation.query_hidden(field, [1, 0, 0, 0], seed=0, alpha=1)
    assert direction.status == "SUPPORTED"
    assert direction.neighbor_indices == tuple(range(8)) and direction.inactive_neighbors == 1
    assert direction.weights[0] > 0 and sum(direction.weights) == pytest.approx(1.)
    assert direction.coexact_chart_norm == pytest.approx((1 - direction.weights[0]) * .1)
    assert direction.coexact_chart_norm < .1
    assert np.linalg.norm(direction.coexact_delta) == pytest.approx(atlas.dose_scale)
    assert np.linalg.norm(direction.random_delta) == pytest.approx(atlas.dose_scale)
    # The random field uses the same weights, zero mask and node magnitudes.
    nodes = []
    for index in direction.neighbor_indices:
        raw = np.random.default_rng(np.random.SeedSequence([0, index])).normal(size=4)
        nodes.append(np.linalg.norm(atlas.coexact[index]) * raw / np.linalg.norm(raw))
    expected = np.asarray(direction.weights) @ np.asarray(nodes)
    assert direction.random_chart_norm == pytest.approx(np.linalg.norm(expected))
    np.testing.assert_allclose(direction.random_delta, 3 * expected / np.linalg.norm(expected), atol=1e-14)


def test_cancellation_does_not_select_a_different_neighbor(atlas):
    points, vectors = atlas.points.copy(), atlas.coexact.copy()
    points[1] = points[0]
    vectors[0], vectors[1] = [0, 1, 0, 0], [0, -1, 0, 0]
    field = interpolation.InterpolatedField(replace(atlas, points=points, coexact=vectors), 2, 1., 10.)
    result = interpolation.query_hidden(field, [1, 0, 0, 0], seed=0, alpha=1)
    assert result.neighbor_indices == (0, 1) and result.weights == (.5, .5)
    assert result.status == "INACTIVE_COEXACT" and result.coexact_chart_norm == 0
    assert not result.coexact_delta.any() and not result.random_delta.any()


@pytest.mark.parametrize("boundary", ["nearest", "eighth"])
def test_both_distance_boundaries_are_enforced(atlas, field, boundary):
    if boundary == "eighth":
        field = replace(field, kth_support_limit=.001)
        hidden = [1, 0, 0, 0]
    else:
        field = replace(field, atlas=replace(atlas, support_limit=0.))
        hidden = [1, .0001, 0, 0]
    result = interpolation.query_hidden(field, hidden, seed=0, alpha=1)
    assert result.status == "OUT_OF_SUPPORT"
    assert not result.coexact_delta.any() and not result.random_delta.any()


@pytest.mark.parametrize("length", [8, 16, 24])
@pytest.mark.parametrize("seed,alpha", [(0, 1.), (7, -.25)])
def test_suffix_replacement_is_byte_invariant(field, length, seed, alpha):
    prefix = tuple(range(length))
    calls = []
    def provider(ids):
        assert ids == prefix and isinstance(ids, tuple)
        calls.append(ids)
        return np.array([1., 0, 0, 0])
    variants = [prefix + (20, 30), prefix + (80, 90, 100), prefix + (20, 30) + (999,) * 100, prefix]
    with patch.object(interpolation, "calibrate_readout", side_effect=AssertionError("no evaluation calibration")):
        results = [interpolation.direction_at_prefix(field, ids, length, provider, seed=seed, alpha=alpha) for ids in variants]
    assert len(calls) == 4
    assert all(r.receipt() == results[0].receipt() for r in results)


def test_sign_norm_scale_and_seed_boundaries(field):
    positive = interpolation.query_hidden(field, [1, 0, 0, 0], seed=0, alpha=.5)
    negative = interpolation.query_hidden(field, [1, 0, 0, 0], seed=0, alpha=-.5)
    scaled = interpolation.query_hidden(field, [2, 0, 0, 0], seed=0, alpha=.5)
    other = interpolation.query_hidden(field, [1, 0, 0, 0], seed=7, alpha=.5)
    assert positive.receipt() == scaled.receipt()
    np.testing.assert_array_equal(positive.coexact_delta, -negative.coexact_delta)
    np.testing.assert_array_equal(positive.random_delta, -negative.random_delta)
    np.testing.assert_array_equal(positive.coexact_delta, other.coexact_delta)
    assert not np.array_equal(positive.random_delta, other.random_delta)
    with pytest.raises(ValueError):
        positive.coexact_delta.setflags(write=True)


@pytest.mark.parametrize("neighbors,quantile", [(0, .99), (True, .99), (9, .99), (8, -1), (8, np.nan)])
def test_invalid_calibration_configuration_fails(atlas, neighbors, quantile):
    with pytest.raises(ValueError):
        interpolation.calibrate_readout(atlas, neighbors=neighbors, quantile=quantile)


@pytest.mark.parametrize("hidden,seed,alpha", [([0]*4, 0, 1), ([np.nan]*4, 0, 1), ([1]*3, 0, 1), ([1]*4, -1, 1), ([1]*4, 0, np.inf)])
def test_invalid_queries_fail(field, hidden, seed, alpha):
    with pytest.raises(ValueError):
        interpolation.query_hidden(field, hidden, seed=seed, alpha=alpha)


def test_any_seed_with_inactive_control_invalidates_cell(field, protocol):
    original = runner.delta_grid
    def changed(field, hidden, spec, seed):
        jobs, deltas = original(field, hidden, spec, seed)
        if seed == 7:
            jobs[0]["status"] = "INACTIVE_RANDOM"
        return jobs, deltas
    with patch.object(runner, "delta_grid", side_effect=changed):
        receipt = runner.direction_receipt(field, [1, 0, 0, 0], protocol["spec"])
    assert receipt["seed0_status"] == "SUPPORTED" and receipt["status"] == "INACTIVE_RANDOM"


class FakeRuntime:
    def __init__(self):
        self.observed = []
        self.nonzero = 0
    def capture(self, ids):
        self.observed.append(tuple(ids))
        return {"hidden": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(ids), 1)),
                "logits": np.tile(np.linspace(-.1, .1, 10, dtype=np.float32), (12, 1))}
    def steer(self, ids, deltas):
        self.observed.append(tuple(ids))
        if deltas.any():
            self.nonzero += len(deltas)
        return np.tile(np.linspace(-.1, .1, 10, dtype=np.float32), (len(deltas), 1)) + deltas[:, :1] * .01


def test_real_adapter_and_treatment_only_pass_observed_ids(field, protocol, tmp_path):
    runtime = FakeRuntime()
    p = {"prompt_id": "test", "family": "literal_stable", "input_ids": [3] * 40}
    local = copy.deepcopy(protocol)
    local["spec"]["evaluation"]["planned_treatment_rows"] = 96
    observation = runner.observe(runtime, field, p["input_ids"], 8, local["spec"])
    ledger = {"nonzero_rows": 0, "nonzero_forward_rows_attempted": 0}
    runner.treat(runtime, field, local, [(p, 8, observation)], tmp_path, ledger)
    assert ledger == {"nonzero_rows": 96, "nonzero_forward_rows_attempted": 96}
    assert runtime.nonzero == 96
    assert all(ids == (3,) * 8 for ids in runtime.observed)


def test_complete_pilot_uses_only_prefixes_and_zero_hooks(field, protocol, tmp_path):
    runtime = FakeRuntime()
    result = runner.pilot(runtime, field, protocol, tmp_path)
    expected = [tuple(p["input_ids"][:n]) for p in protocol["prompts"] if p["split"] == "pilot"
                for n in (8, 16, 24) for _ in range(8)]
    assert runtime.observed == expected
    assert result["passed"] and result["cells"] == 12 and runtime.nonzero == 0
    assert all(len(c["variants"]) == 4 for c in result["records"])


@pytest.mark.parametrize("failed_gate", ["pilot", "support"])
def test_gate_failure_prevents_nonzero_forwards(field, protocol, tmp_path, failed_gate):
    ledger = {"protocol": {}, "nonzero_rows": 0, "nonzero_forward_rows_attempted": 0}
    with patch.object(runner, "pilot", return_value={"passed": failed_gate != "pilot"}), \
            patch.object(runner, "preflight", return_value=({"passed": False}, [])) as preflight, \
            patch.object(runner, "treat") as treat:
        status = runner.stages(FakeRuntime(), field, protocol, tmp_path, ledger)
    treat.assert_not_called()
    assert ledger["nonzero_forward_rows_attempted"] == 0
    assert status == ("INVALID_FUTURE_DEPENDENCE" if failed_gate == "pilot" else "INSUFFICIENT_COVERAGE")
    if failed_gate == "pilot":
        preflight.assert_not_called()


def test_v2_freeze_binds_new_data_and_unchanged_prior(protocol):
    runner.validate(protocol)
    rows, audit, _ = runner.inventory()
    assert len(rows) == 24 and audit["historical_rows"] == 93 and audit["exact_duplicate_count"] == 0
    assert protocol["readout"]["neighbors"] == 8
    old, field, _ = runner.load_prior()
    assert protocol["readout"]["atlas_fingerprint"] == field.fingerprint()
    assert protocol["spec"]["analysis"] == old["spec"]["analysis"]
    assert protocol["spec"]["evaluation"] == old["spec"]["evaluation"]


@pytest.mark.parametrize("key", ["prompts", "spec", "readout", "design", "frozen_files", "runtime", "run_root"])
def test_protocol_changes_fail_before_model_load(protocol, key, tmp_path):
    expected = {k: copy.deepcopy(v) for k, v in protocol.items() if k != "frozen_utc"}
    candidate = copy.deepcopy(protocol)
    candidate[key] = None
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(candidate))
    with patch.object(runner, "contract", return_value=expected), patch.object(runner.v1, "_load_model_and_tokenizer") as load:
        with pytest.raises(ValueError):
            runner.run(path)
        load.assert_not_called()


def test_existing_output_is_not_overwritten(protocol, tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("preserve")
    with patch.object(runner, "contract") as contract:
        with pytest.raises(FileExistsError):
            runner.freeze(path)
        contract.assert_not_called()
    assert path.read_text() == "preserve"


def test_query_replay_detects_changed_neighbor_weights(field, protocol):
    p = {"input_ids": [3] * 40}
    runtime = FakeRuntime()
    obs = runner.observe(runtime, field, p["input_ids"], 8, protocol["spec"])
    logits = np.repeat(obs["logits"][:, :1], 50257, axis=1)
    arrays = {"hidden": obs["hidden"], "logits": logits, "zero_logits": logits.copy()}
    record = obs["receipt"]
    record["baseline_logits_sha256"] = base.array_digest(logits)
    record["zero_logits_sha256"] = base.array_digest(logits)
    analysis.verify_observation(record, arrays, "", p, 8, field, protocol["spec"])
    record["weights"][0] += .01
    with pytest.raises(ValueError, match="disagrees with replay"):
        analysis.verify_observation(record, arrays, "", p, 8, field, protocol["spec"])
