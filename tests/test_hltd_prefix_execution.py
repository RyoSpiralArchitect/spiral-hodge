from __future__ import annotations

import copy
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts import analyze_hltd_prefix_transfer_gate as analysis
from scripts import audit_hltd_prefix_transfer as stdlib_audit
from scripts import hltd_prefix_transfer as transfer
from scripts import prepare_hltd_prefix_gate as preparation
from scripts import run_hltd_prefix_transfer_gate as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def protocol():
    return json.loads((ROOT / runner.PREPARATION).read_text())


@pytest.fixture(scope="module")
def calibration():
    rng = np.random.default_rng(413)
    return {name: rng.normal(size=(23, 12)).astype(np.float32) for name in ("cal_a", "cal_b", "cal_c")}


@pytest.fixture(scope="module")
def field(calibration):
    return transfer.build_frozen_field(calibration, n_components=6, k=4, min_chart_norm=1e-12)


class FakeRuntime:
    def __init__(self, hidden):
        self.hidden = hidden
        self.observed = []
        self.nonzero_calls = 0

    def capture(self, ids, batch=12):
        self.observed.append(tuple(ids))
        return {"hidden": np.repeat(self.hidden[None], len(ids), axis=0),
                "logits": np.repeat(np.arange(10, dtype=np.float32)[None], batch, axis=0)}

    def steer(self, ids, deltas):
        if deltas.any():
            self.nonzero_calls += 1
            raise AssertionError("no nonzero intervention allowed in this test")
        return np.repeat(np.arange(10, dtype=np.float32)[None], len(deltas), axis=0)


def test_execution_plan_reuses_immutable_preparation():
    before = (ROOT / runner.PREPARATION).read_bytes()
    prepared = runner.load_preparation()
    contract = runner.execution_contract()
    candidate = {**contract, "frozen_utc": "2026-10-01T00:00:00+00:00"}
    runner.validate_execution(candidate)
    assert prepared["execution_allowed"] is False
    assert contract["spec"] == prepared["spec"]
    assert (ROOT / runner.PREPARATION).read_bytes() == before
    paths = {r["path"] for r in contract["frozen_files"]}
    assert set(runner.EXECUTION_SOURCES) <= paths


@pytest.mark.parametrize("change", ["spec", "prompts", "runtime", "details", "receipts", "root", "extra_key"])
def test_execution_mutations_fail_before_model_loading(protocol, change, tmp_path):
    with patch.object(runner, "load_preparation", return_value=protocol):
        candidate = {**runner.execution_contract(), "frozen_utc": "2026-10-01T00:00:00+00:00"}
        candidate = copy.deepcopy(candidate)
        if change == "spec":
            candidate["spec"]["geometry"]["layer"] = 8
        elif change == "prompts":
            candidate["prompts"][-1]["input_ids"][0] += 1
        elif change == "runtime":
            candidate["runtime"]["packages"]["torch"] = "different"
        elif change == "details":
            candidate["execution_details"]["prefix_batch_size"] = 1
        elif change == "receipts":
            candidate["frozen_files"] = []
        elif change == "root":
            candidate["run_root"] = "spiral_out_retry"
        else:
            candidate["fallback"] = True
        path = tmp_path / "candidate.json"
        path.write_text(json.dumps(candidate))
        with patch.object(runner, "_load_model_and_tokenizer") as load:
            with pytest.raises(ValueError):
                runner.run(path)
            load.assert_not_called()


def test_field_serialization_is_byte_stable_and_exclusive(field, tmp_path):
    receipt = runner.persist_field(field, tmp_path)
    restored = runner.read_field(tmp_path / "atlas.npz", tmp_path / "atlas_metadata.json")
    assert restored.fingerprint() == receipt["fingerprint"] == field.fingerprint()
    with pytest.raises(FileExistsError):
        runner.persist_field(field, tmp_path)


def test_integrated_pilot_passes_only_observed_prefixes(protocol, calibration, field, tmp_path):
    runtime = FakeRuntime(calibration["cal_a"][7])
    result = runner.run_pilot(runtime, field, protocol, tmp_path)
    assert result["passed"] and result["cells"] == 12 and result["nonzero_treatments"] == 0
    assert runtime.nonzero_calls == 0
    expected = [tuple(p["input_ids"][:n]) for p in protocol["prompts"] if p["split"] == "pilot"
                for n in (8, 16, 24) for _ in range(4)]
    assert runtime.observed == expected
    assert all(len(row["variants"][0]["all_direction_receipts"]) == 96 for row in result["records"])


def test_unstable_forward_fails_future_audit(protocol, calibration, field, tmp_path):
    class UnstableRuntime(FakeRuntime):
        def capture(self, ids, batch=12):
            result = super().capture(ids, batch)
            result["hidden"] += np.float32(len(self.observed) * 1e-4)
            return result
    runtime = UnstableRuntime(calibration["cal_a"][7])
    result = runner.run_pilot(runtime, field, protocol, tmp_path)
    assert not result["passed"]
    assert runtime.nonzero_calls == 0


def test_zero_hook_failure_precedes_nonzero_treatment(protocol, calibration, field):
    class BrokenZero(FakeRuntime):
        def steer(self, ids, deltas):
            return super().steer(ids, deltas) + np.float32(0.1)
    runtime = BrokenZero(calibration["cal_a"][7])
    with pytest.raises(ValueError, match="zero/batch"):
        runner.observe_prefix(runtime, field, [1] * 40, 8, protocol["spec"])
    assert runtime.nonzero_calls == 0


def test_all_support_cells_recorded_without_treatment_on_inactivity(protocol, calibration, field, tmp_path):
    runtime = FakeRuntime(calibration["cal_a"][7])
    inactive = replace(field, coexact=np.zeros_like(field.coexact))
    ledger = {"nonzero_rows": 0, "protocol": {"sha256": "synthetic"}}
    with patch.object(transfer, "build_frozen_field", return_value=inactive), \
            patch.object(runner, "run_pilot", return_value={"passed": True}), \
            patch.object(runner, "write_treatments") as treatments:
        status = runner.execute_stages(runtime, protocol, tmp_path, ledger)
    assert status == "INSUFFICIENT_COVERAGE"
    treatments.assert_not_called()
    assert ledger["nonzero_rows"] == 0
    support = json.loads((tmp_path / "support_preflight.json").read_text())
    assert support["observed_cells"] == 60
    assert support["status_counts"] == {"INACTIVE_COEXACT": 60}
    assert not (tmp_path / "raw_treatments.csv").exists()


def test_pilot_failure_stops_before_evaluation(protocol, calibration, field, tmp_path):
    runtime = FakeRuntime(calibration["cal_a"][7])
    ledger = {"nonzero_rows": 0, "protocol": {"sha256": "synthetic"}}
    with patch.object(transfer, "build_frozen_field", return_value=field), \
            patch.object(runner, "run_pilot", return_value={"passed": False}), \
            patch.object(runner, "preflight") as preflight:
        status = runner.execute_stages(runtime, protocol, tmp_path, ledger)
    assert status == "INVALID_FUTURE_DEPENDENCE"
    preflight.assert_not_called()
    assert ledger["nonzero_rows"] == 0


@pytest.fixture(scope="module")
def synthetic_rows(protocol):
    cells, rows = [], []
    spec = protocol["spec"]
    for prompt in [p for p in protocol["prompts"] if p["split"] == "evaluation"]:
        for length in (8, 16, 24):
            cells.append({"prompt_id": prompt["prompt_id"], "prefix_length": length, "status": "SUPPORTED",
                          "node_index": 1, "distance": 0.1, "dose_scale": 1.})
            for seed, component, alpha in itertools.product(spec["evaluation"]["seeds"], spec["evaluation"]["components"], spec["evaluation"]["alphas"]):
                effect = alpha * (0.6 if component == "coexact" else 0.2)
                rows.append({"prompt_id": prompt["prompt_id"], "family": prompt["family"], "prefix_length": length,
                             "seed": seed, "component": component, "alpha": alpha, "layer": 7,
                             "target_id": prompt["input_ids"][length], "node_index": 1, "distance": 0.1,
                             "dose_scale": 1., "nominal_delta_norm": abs(alpha), "next_token_logprob_base": -3.,
                             "next_token_logprob_steered": -3 + effect, "next_token_prob_base": math.exp(-3),
                             "next_token_prob_steered": math.exp(-3 + effect), "next_token_logprob_delta": effect})
    return rows, {"passed": True, "cells": cells, "planned_cells": 60, "observed_cells": 60, "status_counts": {"SUPPORTED": 60}}


def test_primary_is_coexact_minus_random_not_absolute_improvement(protocol, synthetic_rows):
    rows, support = synthetic_rows
    analysis.validate_rows(rows, protocol, support)
    coefficients = analysis.primary_coefficients(rows, protocol["spec"])
    result, indices = analysis.summarize_primary(coefficients, protocol["spec"])
    assert len(coefficients) == 480 and indices.shape == (5000, 20)
    assert all(row["coexact_odd"] == pytest.approx(.6) for row in coefficients)
    assert all(row["random_odd"] == pytest.approx(.2) for row in coefficients)
    assert result["mean"] == pytest.approx(.4)
    assert result["status"] == "SUPPORTED_WITHIN_SAMPLE_PREFIX_TRANSFER"


@pytest.mark.parametrize("change", ["missing", "duplicate", "target", "dose", "base", "nan", "probability", "cell_missing", "coverage"])
def test_analysis_rejects_changed_grid_or_responses(protocol, synthetic_rows, change):
    rows, support = copy.deepcopy(synthetic_rows)
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "target":
        rows[0]["target_id"] += 1
    elif change == "dose":
        rows[0]["nominal_delta_norm"] *= 2
    elif change == "base":
        rows[0]["next_token_logprob_base"] = -2
    elif change == "nan":
        rows[0]["next_token_logprob_delta"] = math.nan
    elif change == "probability":
        rows[0]["next_token_prob_steered"] = 0
    elif change == "cell_missing":
        support["cells"].pop()
    else:
        support["passed"] = False
        support["cells"][0]["status"] = "OUT_OF_SUPPORT"
        support["status_counts"] = {"SUPPORTED": 59, "OUT_OF_SUPPORT": 1}
    with pytest.raises(ValueError):
        analysis.validate_rows(rows, protocol, support)


def test_spec_and_preparation_sources_were_not_rewritten(protocol):
    assert preparation.load_spec() == protocol["spec"]
    assert protocol["status"] == "PREPARED_NOT_EXECUTED"
    assert protocol["atlas"] is None and protocol["results"] is None


def test_treatment_uses_the_exact_preflight_float32_deltas(protocol, calibration, field, tmp_path):
    class TreatmentRuntime(FakeRuntime):
        def steer(self, ids, deltas):
            if deltas.any():
                self.nonzero_calls += 1
            self.observed.append(tuple(ids))
            return (np.arange(10, dtype=np.float32)[None, :] + deltas[:, :1] * np.linspace(0, .1, 10, dtype=np.float32)[None, :])
    local = copy.deepcopy(protocol)
    local["spec"]["evaluation"]["planned_treatment_rows"] = 96
    prompt = {"prompt_id": "synthetic", "family": "literal_stable", "input_ids": [3] * 40}
    runtime = TreatmentRuntime(calibration["cal_a"][7])
    observation = runner.observe_prefix(runtime, field, prompt["input_ids"], 8, local["spec"])
    ledger = {"nonzero_rows": 0}
    runner.write_treatments(runtime, field, local, [(prompt, 8, observation)], tmp_path, ledger)
    assert ledger["nonzero_rows"] == 96 and runtime.nonzero_calls == 8
    assert all(ids == (3,) * 8 for ids in runtime.observed)
    assert (tmp_path / "raw_treatments.csv").exists()


def test_preflight_delta_mismatch_stops_before_injection(protocol, calibration, field, tmp_path):
    runtime = FakeRuntime(calibration["cal_a"][7])
    prompt = {"prompt_id": "synthetic", "family": "literal_stable", "input_ids": [3] * 40}
    observation = runner.observe_prefix(runtime, field, prompt["input_ids"], 8, protocol["spec"])
    observation["receipt"]["nominal_float32_delta_sha256"][0] = "changed"
    ledger = {"nonzero_rows": 0}
    with pytest.raises(ValueError, match="differ from preflight"):
        runner.write_treatments(runtime, field, protocol, [(prompt, 8, observation)], tmp_path, ledger)
    assert ledger["nonzero_rows"] == 0 and runtime.nonzero_calls == 0


def test_primary_zero_does_not_pass(protocol, synthetic_rows):
    rows, _ = synthetic_rows
    coefficients = analysis.primary_coefficients(rows, protocol["spec"])
    for row in coefficients:
        row["odd_gap"] = 0.0
    result, _ = analysis.summarize_primary(coefficients, protocol["spec"])
    assert result["status"] == "NOT_SUPPORTED" and result["ci_low"] == 0.0


def test_stdlib_probability_audit_recomputes_primary(protocol, synthetic_rows, tmp_path):
    rows, support = synthetic_rows
    coefficients = analysis.primary_coefficients(rows, protocol["spec"])
    primary, draws = analysis.summarize_primary(coefficients, protocol["spec"])
    run, output = tmp_path / "run", tmp_path / "result"
    run.mkdir()
    output.mkdir()
    analysis.write_csv(run / "raw_treatments.csv", rows)
    runner.save_json(run / "support_preflight.json", support)
    runner.save_json(run / "execution_receipt.json", {"nonzero_rows": 5760})
    path = tmp_path / "protocol.json"
    runner.save_json(path, protocol)
    runner.save_json(output / "verdict.json", {"primary": primary, "status": primary["status"], "supported_cells": 60})
    with (output / "bootstrap_indices.csv").open("w", newline="") as handle:
        import csv
        csv.writer(handle).writerows(draws.tolist())
    runner.save_json(output / "manifest.json", {"protocol": runner.file_receipt(path),
        "raw_files": [runner.file_receipt(p) for p in sorted(run.iterdir())],
        "derived_files": [runner.file_receipt(p) for p in sorted(output.iterdir())]})
    result = stdlib_audit.audit(output)
    assert result["passed"] and result["primary_recomputed"]
    assert result["mean"] == pytest.approx(.4)
    assert result["max_primary_error"] < 1e-12
    (run / "raw_treatments.csv").write_text("changed")
    with pytest.raises(ValueError, match="changed artifact"):
        stdlib_audit.audit(output)
