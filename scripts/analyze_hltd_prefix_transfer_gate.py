#!/usr/bin/env python3
"""Validate the entire prefix gate before exporting a bounded result."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hltd_prefix_transfer as transfer
from scripts import run_hltd_prefix_transfer_gate as runner
from scripts.evaluate_hltd_signed_layer_gate import file_receipt
from scripts.run_hltd_precision_gate import save_json, utc_now


def read_json(path):
    return json.loads(path.read_text())


def require_support(protocol: dict, support: dict) -> list[tuple]:
    expected = [(p["prompt_id"], n) for p in protocol["prompts"] if p["split"] == "evaluation"
                for n in protocol["spec"]["evaluation"]["prefix_lengths"]]
    cells = support["cells"]
    if [(c["prompt_id"], c["prefix_length"]) for c in cells] != expected:
        raise ValueError("incomplete or reordered support grid")
    passed = all(c["status"] == "SUPPORTED" for c in cells)
    if (support["passed"] != passed or support["planned_cells"] != 60 or support["observed_cells"] != 60
            or support["status_counts"] != dict(Counter(c["status"] for c in cells))):
        raise ValueError("inconsistent support verdict")
    return expected


def verify_observations(protocol, run: Path, support: dict) -> dict:
    spec = protocol["spec"]
    require_support(protocol, support)
    field = runner.read_field(run / "atlas.npz", run / "atlas_metadata.json")
    atlas = read_json(run / "atlas_freeze.json")
    if field.fingerprint() != atlas["fingerprint"]:
        raise ValueError("changed atlas fingerprint")
    for record in [*atlas["files"], atlas["hidden_receipt"]]:
        if file_receipt(Path(record["path"])) != record:
            raise ValueError("changed calibration artifact")
    calibration_ids = sorted(p["prompt_id"] for p in protocol["prompts"] if p["split"] == "calibration")
    with np.load(run / "calibration_hidden.npz", allow_pickle=False) as raw:
        if sorted(raw.files) != calibration_ids:
            raise ValueError("calibration membership changed")
        if tuple((name, transfer.array_digest(raw[name])) for name in calibration_ids) != field.calibration_hashes:
            raise ValueError("calibration hidden hashes changed")
    prompts = {p["prompt_id"]: p for p in protocol["prompts"]}
    with np.load(run / "evaluation_prefixes.npz", allow_pickle=False) as raw:
        if raw["hidden"].shape != (60, 768) or raw["baseline_logits"].shape != (60, 50257):
            raise ValueError("unexpected saved prefix shapes")
        for i, cell in enumerate(support["cells"]):
            observed = prompts[cell["prompt_id"]]["input_ids"][:cell["prefix_length"]]
            if cell["observed_ids"] != observed or cell["family"] != prompts[cell["prompt_id"]]["family"]:
                raise ValueError("prefix/family changed")
            direction = transfer.query_hidden(field, raw["hidden"][i], seed=0, alpha=1)
            if any(cell[key] != value for key, value in direction.receipt().items()):
                raise ValueError("saved support disagrees with frozen field query")
            if (cell["support_limit"] != field.support_limit
                    or cell["coexact_chart_norm"] != float(np.linalg.norm(field.coexact[direction.node_index]))):
                raise ValueError("changed support/activity threshold")
            jobs, hashes = [], []
            for seed in spec["evaluation"]["seeds"]:
                grid, deltas = runner.delta_grid(field, raw["hidden"][i], spec, seed)
                jobs.extend(grid)
                hashes.append(transfer.array_digest(deltas))
            if cell["all_direction_receipts"] != jobs or cell["nominal_float32_delta_sha256"] != hashes:
                raise ValueError("preflight delta grid differs")
            if (not 0 <= cell["zero_hook_max_abs"] <= spec["runtime_gate"]["zero_hook_logit_max_abs"]
                    or cell["row_spread_max_abs"] != 0):
                raise ValueError("invalid evaluation zero audit")
            repeated = np.repeat(raw["baseline_logits"][i:i + 1], 12, axis=0)
            if transfer.array_digest(repeated) != cell["baseline_logits_sha256"]:
                raise ValueError("saved baseline bytes differ")
    return {"passed": True, "recomputed_cells": 60, "atlas_fingerprint": field.fingerprint()}


def validate_rows(rows: list[dict], protocol: dict, support: dict) -> dict:
    spec = protocol["spec"]
    require_support(protocol, support)
    if not support["passed"]:
        raise ValueError("cannot estimate a reduced-coverage primary endpoint")
    prompts = {p["prompt_id"]: p for p in protocol["prompts"] if p["split"] == "evaluation"}
    cells = {(c["prompt_id"], c["prefix_length"]): c for c in support["cells"]}
    expected = set(itertools.product(prompts, spec["evaluation"]["prefix_lengths"],
                                    spec["evaluation"]["seeds"], spec["evaluation"]["components"], spec["evaluation"]["alphas"]))
    seen, baselines = set(), {}
    for row in rows:
        key = (row["prompt_id"], int(row["prefix_length"]), int(row["seed"]), row["component"], float(row["alpha"]))
        if key not in expected or key in seen:
            raise ValueError("unexpected or duplicate treatment unit")
        seen.add(key)
        prompt, cell = prompts[key[0]], cells[key[:2]]
        if (row["family"] != prompt["family"] or int(row["layer"]) != 7
                or int(row["target_id"]) != prompt["input_ids"][key[1]] or int(row["node_index"]) != cell["node_index"]):
            raise ValueError("changed target or direction identity")
        for name in ("dose_scale", "distance"):
            if float(row[name]) != cell[name]:
                raise ValueError("changed direction metadata")
        base, steered = float(row["next_token_logprob_base"]), float(row["next_token_logprob_steered"])
        probability = float(row["next_token_prob_steered"])
        base_probability = float(row["next_token_prob_base"])
        if not all(math.isfinite(x) for x in (base, steered, probability, base_probability)):
            raise ValueError("nonfinite response")
        if not 0 < probability <= 1 or not 0 < base_probability <= 1:
            raise ValueError("invalid response probability")
        if (abs(math.log(probability) - steered) > 1e-12 or abs(math.log(base_probability) - base) > 1e-12
                or not math.isclose(float(row["next_token_logprob_delta"]), steered - base, abs_tol=1e-12, rel_tol=0)):
            raise ValueError("probability/log-probability disagreement")
        if key[:2] in baselines and baselines[key[:2]] != base:
            raise ValueError("unmatched baselines")
        baselines[key[:2]] = base
        if not math.isclose(float(row["nominal_delta_norm"]), abs(key[4]) * cell["dose_scale"], rel_tol=2e-7, abs_tol=1e-10):
            raise ValueError("incorrect dose")
    if seen != expected or len(rows) != 5760:
        raise ValueError("incomplete treatment grid")
    return {"passed": True, "rows": len(rows), "cells": len(cells)}


def primary_coefficients(rows: list[dict], spec: dict) -> list[dict]:
    units = defaultdict(dict)
    families = {}
    for row in rows:
        key = (row["prompt_id"], int(row["prefix_length"]), int(row["seed"]))
        units[key][row["component"], float(row["alpha"])] = float(row["next_token_logprob_steered"])
        families[key[0]] = row["family"]
    magnitudes = spec["analysis"]["positive_magnitudes"]
    denominator = sum(a * a for a in magnitudes)
    coefficients = []
    for (prompt, length, seed), values in sorted(units.items()):
        coexact = sum(a * (values["coexact", a] - values["coexact", -a]) / 2 for a in magnitudes) / denominator
        random = sum(a * (values["random_tangent", a] - values["random_tangent", -a]) / 2 for a in magnitudes) / denominator
        coefficients.append({"prompt_id": prompt, "family": families[prompt], "prefix_length": length,
                             "seed": seed, "coexact_odd": coexact, "random_odd": random, "odd_gap": coexact - random})
    return coefficients


def verify_treatment_inputs(rows, protocol, run, support):
    field = runner.read_field(run / "atlas.npz", run / "atlas_metadata.json")
    indexed = {(r["prompt_id"], int(r["prefix_length"]), int(r["seed"]), r["component"], float(r["alpha"])): r for r in rows}
    prompts = {p["prompt_id"]: p for p in protocol["prompts"]}
    with np.load(run / "evaluation_prefixes.npz", allow_pickle=False) as data:
        for i, cell in enumerate(support["cells"]):
            prefix = (cell["prompt_id"], cell["prefix_length"])
            target = prompts[prefix[0]]["input_ids"][prefix[1]]
            base = float(runner._log_softmax(data["baseline_logits"][i])[target])
            for seed in protocol["spec"]["evaluation"]["seeds"]:
                jobs, deltas = runner.delta_grid(field, data["hidden"][i], protocol["spec"], seed)
                for job, delta in zip(jobs, deltas, strict=True):
                    row = indexed[(*prefix, seed, job["component"], job["alpha"])]
                    if row["nominal_delta_sha256"] != transfer.array_digest(delta):
                        raise ValueError("raw intervention delta bytes differ from frozen query")
                    if abs(float(row["next_token_logprob_base"]) - base) > 1e-12:
                        raise ValueError("raw baseline differs from saved prefix logits")


def summarize_primary(coefficients: list[dict], spec: dict) -> tuple[dict, np.ndarray]:
    by_prompt = defaultdict(list)
    for row in coefficients:
        by_prompt[row["prompt_id"]].append(row["odd_gap"])
    if len(by_prompt) != 20 or any(len(values) != 24 for values in by_prompt.values()):
        raise ValueError("incomplete primary coefficient grid")
    prompt_values = {key: float(np.mean(values)) for key, values in sorted(by_prompt.items())}
    values = np.array(list(prompt_values.values()))
    analysis = spec["analysis"]
    indices = np.random.default_rng(analysis["bootstrap_seed"]).integers(0, len(values), size=(analysis["bootstrap_samples"], len(values)))
    estimates = values[indices].mean(axis=1)
    interval = np.quantile(estimates, analysis["quantiles"])
    return {"mean": float(values.mean()), "ci_low": float(interval[0]), "ci_high": float(interval[1]),
            "positive_prompts": int((values > 0).sum()), "prompt_coefficients": prompt_values,
            "status": "SUPPORTED_WITHIN_SAMPLE_PREFIX_TRANSFER" if interval[0] > 0 else "NOT_SUPPORTED"}, indices


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(protocol_path: Path, output: Path) -> dict:
    protocol = read_json(protocol_path)
    runner.validate_execution(protocol)
    if output.exists():
        raise FileExistsError(output)
    run = ROOT / protocol["run_root"]
    receipt = read_json(run / "execution_receipt.json")
    if receipt["status"] != "COMPLETED" or receipt["protocol"] != file_receipt(protocol_path):
        raise ValueError("execution was not completed against this protocol")
    load = read_json(run / "load_audit.json")
    final = read_json(run / "final_weight_audit.json")
    if not final["passed"] or load["model"] != final["model"]:
        raise ValueError("weight audit failed")
    if (not load["model"]["all_bytes_equal"] or load["model"]["parameter_tensors"] != 148
            or load["model"]["parameter_count"] != 124439808 or load["runtime"] != protocol["runtime"]
            or load["model_load_count"] != 1):
        raise ValueError("unexpected model/runtime audit")
    pilot = read_json(run / "pilot_audit.json")
    expected = [(p["prompt_id"], n) for p in protocol["prompts"] if p["split"] == "pilot"
                for n in protocol["spec"]["evaluation"]["prefix_lengths"]]
    if [(r["prompt_id"], r["prefix_length"]) for r in pilot["records"]] != expected:
        raise ValueError("pilot grid changed")
    if not pilot["passed"] or pilot["nonzero_treatments"] != 0:
        raise ValueError("pilot invalid")
    for row in pilot["records"]:
        if len(row["variants"]) != 4 or not row["byte_identical"] or any(v != row["variants"][0] for v in row["variants"]):
            raise ValueError("suffix invariance failed")
        source = next(p for p in protocol["prompts"] if p["prompt_id"] == row["prompt_id"])
        for variant in row["variants"]:
            if (variant["observed_ids"] != source["input_ids"][:row["prefix_length"]]
                    or not 0 <= variant["zero_hook_max_abs"] <= protocol["spec"]["runtime_gate"]["zero_hook_logit_max_abs"]
                    or variant["row_spread_max_abs"] != 0 or len(variant["all_direction_receipts"]) != 96
                    or len(variant["nominal_float32_delta_sha256"]) != 8):
                raise ValueError("pilot prefix or zero audit changed")
    support = read_json(run / "support_preflight.json")
    atlas = read_json(run / "atlas_freeze.json")
    if (atlas["execution_protocol"] != file_receipt(protocol_path) or atlas["calibration_count"] != 40
            or atlas["preparation_sha256"] != runner.PREPARATION_SHA256
            or atlas["geometry"] != protocol["spec"]["geometry"]):
        raise ValueError("atlas was not frozen against this execution")
    times = [protocol["frozen_utc"], receipt["started_utc"], atlas["frozen_utc"],
             pilot["completed_utc"], support["completed_utc"], final["completed_utc"], receipt["completed_utc"]]
    parsed = [datetime.fromisoformat(t) for t in times]
    if any(t.tzinfo is None for t in parsed) or parsed != sorted(parsed):
        raise ValueError("invalid execution chronology")
    query_audit = verify_observations(protocol, run, support)
    verdict = {"completed_utc": utc_now(), "planned_cells": 60, "supported_cells": support["status_counts"].get("SUPPORTED", 0),
               "status_counts": support["status_counts"], "nonzero_rows": receipt["nonzero_rows"], "primary": None,
               "pilot_cells": 12, "suffix_variants_per_cell": 4, "query_replay": query_audit,
               "scope": "Prefix-available calibration-field transfer in the fixed authored sample; no semantic or fluency claim."}
    if not support["passed"]:
        if receipt["gate_status"] != "INSUFFICIENT_COVERAGE" or receipt["nonzero_rows"] != 0 or (run / "raw_treatments.csv").exists():
            raise ValueError("treatment occurred despite incomplete coverage")
        verdict["status"] = "INSUFFICIENT_COVERAGE"
    else:
        if receipt["gate_status"] != "TREATMENTS_COMPLETE_PENDING_ANALYSIS" or receipt["nonzero_rows"] != 5760:
            raise ValueError("treatment grid did not finish")
        with (run / "raw_treatments.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        validate_rows(rows, protocol, support)
        verify_treatment_inputs(rows, protocol, run, support)
        gate = read_json(run / "pre_treatment_gate.json")
        expected_receipts = [file_receipt(run / name) for name in ("atlas_freeze.json", "pilot_audit.json", "support_preflight.json")]
        if (not gate["passed"] or gate["nonzero_rows_at_gate"] != 0 or gate["files"] != expected_receipts
                or gate["atlas_fingerprint"] != atlas["fingerprint"]
                or not parsed[4] <= datetime.fromisoformat(gate["completed_utc"]) <= parsed[5]):
            raise ValueError("invalid pre-treatment gate")
        coefficients = primary_coefficients(rows, protocol["spec"])
        primary, indices = summarize_primary(coefficients, protocol["spec"])
        verdict.update({"status": primary["status"], "primary": primary})
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "verdict.json", verdict)
    keys = ["prompt_id", "family", "prefix_length", "status", "node_index", "node_prompt", "node_token", "distance",
            "support_limit", "coexact_chart_norm", "dose_scale", "zero_hook_max_abs", "row_spread_max_abs"]
    write_csv(output / "support.csv", [{key: cell[key] for key in keys} for cell in support["cells"]])
    if support["passed"]:
        write_csv(output / "coefficients.csv", coefficients)
        with (output / "bootstrap_indices.csv").open("x", newline="") as handle:
            csv.writer(handle).writerows(indices.tolist())
    save_json(output / "manifest.json", {"protocol": file_receipt(protocol_path),
              "raw_files": [file_receipt(p) for p in sorted(run.iterdir()) if p.is_file()],
              "derived_files": [file_receipt(p) for p in sorted(output.iterdir()) if p.is_file()]})
    return verdict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.protocol.absolute(), args.output.absolute()), indent=2))


if __name__ == "__main__":
    main()
