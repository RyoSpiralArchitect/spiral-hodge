#!/usr/bin/env python3
"""Replay every v2 query and dose, then apply the unchanged primary estimator."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_hltd_prefix_transfer_gate as analysis
from scripts import hltd_prefix_transfer as base
from scripts import run_hltd_prefix_v2 as runner
from scripts.evaluate_hltd_signed_layer_gate import file_receipt
from scripts.run_hltd_precision_gate import save_json, utc_now


def read_json(path):
    return json.loads(path.read_text())


def verify_observation(record, arrays, key, prompt, length, field, spec):
    hidden, logits, zero = [arrays[key + name] for name in ("hidden", "logits", "zero_logits")]
    if (hidden.shape != (length, len(field.atlas.mean)) or logits.shape != (12, 50257) or zero.shape != logits.shape
            or any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (hidden, logits, zero))):
        raise ValueError("invalid saved v2 observation shapes/dtypes")
    if record["observed_ids"] != prompt["input_ids"][:length]:
        raise ValueError("saved query did not use the frozen prefix")
    expected = runner.direction_receipt(field, hidden[-1], spec)
    if any(record[k] != v for k, v in expected.items()):
        raise ValueError("saved v2 query or delta grid disagrees with replay")
    for name, array in (("prefix_hidden_sha256", hidden), ("baseline_logits_sha256", logits), ("zero_logits_sha256", zero)):
        if record[name] != base.array_digest(array):
            raise ValueError("saved observation bytes changed")
    errors = {"zero_hook_max_abs": float(np.abs(zero - logits).max()),
              "row_spread_max_abs": float(np.abs(logits - logits[0]).max())}
    if (any(record[k] != v for k, v in errors.items()) or errors["row_spread_max_abs"] != 0
            or errors["zero_hook_max_abs"] > spec["runtime_gate"]["zero_hook_logit_max_abs"]
            or record["support_limit"] != field.atlas.support_limit or record["kth_support_limit"] != field.kth_support_limit):
        raise ValueError("zero/support audit changed")


def verify_query_replay(protocol, run, pilot, support):
    field = runner.load_readout(protocol)
    prompts = {p["prompt_id"]: p for p in protocol["prompts"]}
    expected_pilot = [(p["prompt_id"], n) for p in protocol["prompts"] if p["split"] == "pilot"
                      for n in protocol["spec"]["evaluation"]["prefix_lengths"]]
    if (pilot["cells"] != 12 or not pilot["passed"] or pilot["nonzero_treatments"] != 0
            or [(c["prompt_id"], c["prefix_length"]) for c in pilot["records"]] != expected_pilot):
        raise ValueError("invalid v2 pilot grid")
    with np.load(run / "pilot_observations.npz", allow_pickle=False) as arrays:
        expected_keys = {f"{i}_{variant}_{name}" for i in range(12) for variant in range(4)
                         for name in ("hidden", "logits", "zero_logits")}
        if set(arrays.files) != expected_keys:
            raise ValueError("pilot array inventory changed")
        for i, cell in enumerate(pilot["records"]):
            if (not cell["byte_identical"] or len(cell["variants"]) != 4
                    or any(v != cell["variants"][0] for v in cell["variants"])):
                raise ValueError("future-invariance failed")
            for variant, record in enumerate(cell["variants"]):
                verify_observation(record, arrays, f"{i}_{variant}_", prompts[cell["prompt_id"]],
                                   cell["prefix_length"], field, protocol["spec"])
    analysis.require_support(protocol, support)
    with np.load(run / "evaluation_observations.npz", allow_pickle=False) as arrays:
        expected_keys = {f"{i}_{name}" for i in range(60) for name in ("hidden", "logits", "zero_logits")}
        if set(arrays.files) != expected_keys:
            raise ValueError("evaluation array inventory changed")
        for i, cell in enumerate(support["cells"]):
            if cell["family"] != prompts[cell["prompt_id"]]["family"]:
                raise ValueError("changed evaluation family")
            verify_observation(cell, arrays, f"{i}_", prompts[cell["prompt_id"]], cell["prefix_length"], field, protocol["spec"])
    return {"passed": True, "pilot_variant_observations": 48, "evaluation_observations": 60,
            "readout_fingerprint": field.fingerprint()}


def verify_treatment_inputs(rows, protocol, run, support):
    field = runner.load_readout(protocol)
    prompts = {p["prompt_id"]: p for p in protocol["prompts"]}
    indexed = {(r["prompt_id"], int(r["prefix_length"]), int(r["seed"]), r["component"], float(r["alpha"])): r for r in rows}
    with np.load(run / "evaluation_observations.npz", allow_pickle=False) as data:
        for i, cell in enumerate(support["cells"]):
            key = (cell["prompt_id"], cell["prefix_length"])
            target = prompts[key[0]]["input_ids"][key[1]]
            baseline = float(runner.v1._log_softmax(data[f"{i}_logits"][0])[target])
            for seed in protocol["spec"]["evaluation"]["seeds"]:
                jobs, deltas = runner.delta_grid(field, data[f"{i}_hidden"][-1], protocol["spec"], seed)
                for job, delta in zip(jobs, deltas, strict=True):
                    row = indexed[(*key, seed, job["component"], job["alpha"])]
                    if row["nominal_delta_sha256"] != base.array_digest(delta):
                        raise ValueError("v2 injected delta bytes disagree with replay")
                    if abs(float(row["next_token_logprob_base"]) - baseline) > 1e-12:
                        raise ValueError("unmatched v2 baseline")


def analyze(path, output):
    protocol = read_json(path)
    runner.validate(protocol)
    if output.exists():
        raise FileExistsError(output)
    run = ROOT / protocol["run_root"]
    receipt = read_json(run / "execution_receipt.json")
    if receipt["status"] != "COMPLETED" or receipt["protocol"] != file_receipt(path):
        raise ValueError("v2 execution incomplete or protocol changed")
    load, final = [read_json(run / name) for name in ("load_audit.json", "final_weight_audit.json")]
    if (not final["passed"] or final["model"] != load["model"] or not load["model"]["all_bytes_equal"]
            or load["model"]["parameter_count"] != 124439808 or load["model"]["parameter_tensors"] != 148
            or load["runtime"] != protocol["runtime"] or load["model_load_count"] != 1):
        raise ValueError("v2 model/runtime audit failed")
    frozen = read_json(run / "readout_freeze.json")
    if (frozen["protocol"] != file_receipt(path) or frozen["readout"] != protocol["readout"]
            or frozen["fingerprint"] != protocol["readout_fingerprint"] or frozen["nonzero_rows"] != 0):
        raise ValueError("changed readout freeze")
    pilot, support = [read_json(run / name) for name in ("pilot_audit.json", "support_preflight.json")]
    times = [protocol["frozen_utc"], receipt["started_utc"], frozen["frozen_utc"], pilot["completed_utc"],
             support["completed_utc"], final["completed_utc"], receipt["completed_utc"]]
    parsed = [datetime.fromisoformat(t) for t in times]
    if any(t.tzinfo is None for t in parsed) or parsed != sorted(parsed):
        raise ValueError("invalid v2 chronology")
    replay = verify_query_replay(protocol, run, pilot, support)
    result = {"protocol_id": protocol["protocol_id"], "completed_utc": utc_now(), "planned_cells": 60,
              "supported_cells": support["status_counts"].get("SUPPORTED", 0), "status_counts": support["status_counts"],
              "nonzero_rows": receipt["nonzero_rows"], "nonzero_forward_rows_attempted": receipt["nonzero_forward_rows_attempted"],
              "primary": None, "query_replay": replay,
              "scope": "Fixed authored new-text sample, interpolated calibration-coexact versus matched interpolated random field; no semantic or fluency claim."}
    if not support["passed"]:
        if (receipt["gate_status"] != "INSUFFICIENT_COVERAGE" or receipt["nonzero_rows"] != 0
                or receipt["nonzero_forward_rows_attempted"] != 0 or (run / "raw_treatments.csv").exists()):
            raise ValueError("v2 treatment despite incomplete coverage")
        result["status"] = "INSUFFICIENT_COVERAGE"
    else:
        if (receipt["gate_status"] != "TREATMENTS_COMPLETE_PENDING_ANALYSIS" or receipt["nonzero_rows"] != 5760
                or receipt["nonzero_forward_rows_attempted"] != 5760):
            raise ValueError("v2 treatment grid not complete")
        gate = read_json(run / "pre_treatment_gate.json")
        expected = [file_receipt(run / n) for n in ("readout_freeze.json", "pilot_audit.json", "support_preflight.json")]
        if (not gate["passed"] or gate["files"] != expected or gate["nonzero_rows_at_gate"] != 0
                or gate["nonzero_forward_rows_attempted"] != 0 or gate["readout_fingerprint"] != protocol["readout_fingerprint"]
                or not parsed[4] <= datetime.fromisoformat(gate["completed_utc"]) <= parsed[5]):
            raise ValueError("v2 pre-treatment gate invalid")
        with (run / "raw_treatments.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        analysis.validate_rows(rows, protocol, support)
        verify_treatment_inputs(rows, protocol, run, support)
        coefficients = analysis.primary_coefficients(rows, protocol["spec"])
        primary, draws = analysis.summarize_primary(coefficients, protocol["spec"])
        result.update({"status": primary["status"], "primary": primary})
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "verdict.json", result)
    keys = ["prompt_id", "family", "prefix_length", "status", "distance", "node_index", "node_prompt", "node_token",
            "coexact_chart_norm", "random_chart_norm", "inactive_neighbors", "dose_scale", "support_limit", "kth_support_limit",
            "zero_hook_max_abs", "row_spread_max_abs"]
    compact = [{**{k: c[k] for k in keys}, "eighth_distance": c["neighbor_distances"][-1]} for c in support["cells"]]
    analysis.write_csv(output / "support.csv", compact)
    if support["passed"]:
        analysis.write_csv(output / "coefficients.csv", coefficients)
        with (output / "bootstrap_indices.csv").open("x", newline="") as handle:
            csv.writer(handle).writerows(draws.tolist())
    save_json(output / "manifest.json", {"protocol": file_receipt(path),
              "raw_files": [file_receipt(p) for p in sorted(run.iterdir()) if p.is_file()],
              "derived_files": [file_receipt(p) for p in sorted(output.iterdir()) if p.is_file()]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.protocol.absolute(), args.output.absolute()), indent=2))


if __name__ == "__main__":
    main()
