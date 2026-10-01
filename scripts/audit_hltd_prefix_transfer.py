#!/usr/bin/env python3
"""Standard-library audit of recorded probabilities, coverage, and estimates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def audit(result: Path) -> dict:
    manifest = read_json(result / "manifest.json")
    for record in [manifest["protocol"], *manifest["raw_files"], *manifest["derived_files"]]:
        path = Path(record["path"])
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if path.stat().st_size != record["bytes"] or digest != record["sha256"]:
            raise ValueError("changed artifact: " + str(path))
    protocol = read_json(Path(manifest["protocol"]["path"]))
    spec = protocol["spec"]
    raw_paths = {Path(r["path"]).name: Path(r["path"]) for r in manifest["raw_files"]}
    support = read_json(raw_paths["support_preflight.json"])
    receipt = read_json(raw_paths["execution_receipt.json"])
    verdict = read_json(result / "verdict.json")
    prompts = {p["prompt_id"]: p for p in protocol["prompts"] if p["split"] == "evaluation"}
    expected_cells = set(itertools.product(prompts, spec["evaluation"]["prefix_lengths"]))
    if len(support["cells"]) != 60 or {(c["prompt_id"], c["prefix_length"]) for c in support["cells"]} != expected_cells:
        raise ValueError("incomplete support grid")
    supported = sum(c["status"] == "SUPPORTED" for c in support["cells"])
    if supported != verdict["supported_cells"] or support["passed"] != (supported == 60):
        raise ValueError("support status mismatch")
    if supported < 60:
        if (verdict["status"] != "INSUFFICIENT_COVERAGE" or verdict["primary"] is not None
                or receipt["nonzero_rows"] != 0 or "raw_treatments.csv" in raw_paths):
            raise ValueError("invalid partial-coverage inference")
        return {"passed": True, "status": verdict["status"], "supported_cells": supported,
                "planned_cells": 60, "raw_treatment_rows": 0, "primary_recomputed": False,
                "scope": "Artifact hashes and stop-before-treatment boundary; not a model execution replay."}
    with raw_paths["raw_treatments.csv"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = set(itertools.product(prompts, spec["evaluation"]["prefix_lengths"], spec["evaluation"]["seeds"],
                                    spec["evaluation"]["components"], spec["evaluation"]["alphas"]))
    values = {}
    max_log_error = 0.0
    for row in rows:
        key = (row["prompt_id"], int(row["prefix_length"]), int(row["seed"]), row["component"], float(row["alpha"]))
        if key not in expected or key in values:
            raise ValueError("changed raw grid")
        value = math.log(float(row["next_token_prob_steered"]))
        max_log_error = max(max_log_error, abs(value - float(row["next_token_logprob_steered"])))
        values[key] = value
    if set(values) != expected or len(rows) != 5760 or max_log_error > 1e-12:
        raise ValueError("incomplete or inconsistent raw responses")
    grouped = defaultdict(list)
    for prompt, length, seed in itertools.product(sorted(prompts), spec["evaluation"]["prefix_lengths"], spec["evaluation"]["seeds"]):
        slope = 0.0
        for magnitude in spec["analysis"]["positive_magnitudes"]:
            prefix = (prompt, length, seed)
            slope += magnitude * (values[*prefix, "coexact", magnitude] - values[*prefix, "coexact", -magnitude]
                                  - values[*prefix, "random_tangent", magnitude] + values[*prefix, "random_tangent", -magnitude])
        slope /= 2 * sum(a * a for a in spec["analysis"]["positive_magnitudes"])
        grouped[prompt].append(slope)
    means = {prompt: math.fsum(grouped[prompt]) / len(grouped[prompt]) for prompt in sorted(prompts)}
    vector = list(means.values())
    with (result / "bootstrap_indices.csv").open(newline="") as handle:
        draws = [[int(index) for index in row] for row in csv.reader(handle)]
    if len(draws) != 5000 or any(len(row) != 20 or any(i not in range(20) for i in row) for row in draws):
        raise ValueError("invalid bootstrap grid")
    estimates = [math.fsum(vector[index] for index in row) / 20 for row in draws]
    mean = math.fsum(vector) / 20
    low, high = [quantile(estimates, q) for q in spec["analysis"]["quantiles"]]
    primary = verdict["primary"]
    errors = [abs(primary["mean"] - mean), abs(primary["ci_low"] - low), abs(primary["ci_high"] - high),
              *[abs(means[key] - primary["prompt_coefficients"][key]) for key in means]]
    status = "SUPPORTED_WITHIN_SAMPLE_PREFIX_TRANSFER" if low > 0 else "NOT_SUPPORTED"
    if max(errors) > 1e-12 or status != verdict["status"]:
        raise ValueError("primary inference differs from raw probabilities")
    return {"passed": True, "status": status, "raw_treatment_rows": 5760, "primary_recomputed": True,
            "mean": mean, "ci_low": low, "ci_high": high, "max_log_error": max_log_error, "max_primary_error": max(errors),
            "scope": "Independent probability-to-coefficient arithmetic; saved bootstrap draws, not a model execution replay."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.result.absolute())
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
