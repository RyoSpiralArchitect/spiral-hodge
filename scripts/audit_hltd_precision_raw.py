#!/usr/bin/env python3
"""Independently check precision coefficients using only standard-library CSVs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def receipt(path: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def audit_arm(raw_path: Path, coefficient_path: Path, protocol: dict, tolerance: float = 1e-10) -> dict:
    require(math.isfinite(tolerance) and tolerance >= 0, "invalid audit tolerance")
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    design = protocol["design"]
    pairs = defaultdict(dict)
    for row in rows:
        key = (row["family"], row["prompt_id"], int(row["token_index"]), int(row["token_count"]), int(row["seed"]), float(row["alpha"]))
        require(row["component"] not in pairs[key], f"duplicate raw unit: {key}")
        pairs[key][row["component"]] = row
    expected = {(p["family"], p["prompt_id"], token, p["token_count"], seed, alpha)
                for p in protocol["prompts"] for token, seed, alpha in itertools.product(
                    range(1, p["token_count"] - 1), design["seeds"], design["alphas"])}
    require(set(pairs) == expected, "raw seed/sign/token grid mismatch")
    seeded = defaultdict(dict)
    active_units = set()
    for key, branches in pairs.items():
        require(set(branches) == {"coexact", "random_tangent"}, "missing or unexpected branch")
        coexact, null = branches["coexact"], branches["random_tangent"]
        activity = [float(coexact["component_active"]), float(null["component_active"])]
        require(all(a in {0, 1} for a in activity), "invalid activity flag")
        if activity[0] == 0:
            continue
        require(activity[1] == 1, "active coexact is missing a random control")
        family, prompt, token, count, seed, alpha = key
        active_units.add((prompt, token))
        next_gap = float(coexact["next_token_logprob_steered"]) - float(null["next_token_logprob_steered"])
        semantic_gap = ((float(coexact["target_logprob_mass_steered"]) - float(coexact["control_logprob_mass_steered"]))
                        - (float(null["target_logprob_mass_steered"]) - float(null["control_logprob_mass_steered"])))
        for metric, gap in [("next_token_logprob_delta", next_gap), ("semantic_margin_delta", semantic_gap)]:
            require(math.isfinite(gap), "nonfinite raw gap")
            seeded[(family, prompt, token, count, alpha, metric)][seed] = gap
    token_gaps = defaultdict(dict)
    for key, values in seeded.items():
        require(set(values) == set(design["seeds"]), "incomplete active seed pairs")
        family, prompt, token, count, alpha, metric = key
        token_gaps[(family, prompt, token, count, metric)][alpha] = math.fsum(values.values()) / len(values)
    coefficients = defaultdict(list)
    for key, gaps in token_gaps.items():
        require(set(gaps) == set(design["alphas"]), "incomplete active signed strengths")
        family, prompt, token, count, metric = key
        bins = protocol["analysis"]["bins"]
        position_bin = min(bins - 1, math.floor(bins * (token / (count - 1))))
        magnitudes = sorted(a for a in gaps if a > 0)
        require(bool(magnitudes) and set(gaps) == {a * s for a in magnitudes for s in [-1, 1]}, "unpaired strengths")
        odd = math.fsum(a * (gaps[a] - gaps[-a]) / 2 for a in magnitudes) / math.fsum(a ** 2 for a in magnitudes)
        even = math.fsum(a ** 2 * (gaps[a] + gaps[-a]) / 2 for a in magnitudes) / math.fsum(a ** 4 for a in magnitudes)
        for contrast, value in [("odd", odd), ("even", even)]:
            coefficients[(family, prompt, position_bin, metric, contrast)].append(value)
    rebuilt = {key: math.fsum(values) / len(values) for key, values in coefficients.items()}
    with coefficient_path.open(newline="", encoding="utf-8") as handle:
        saved_rows = list(csv.DictReader(handle))
    saved = {(r["family"], r["prompt_id"], int(r["position_bin"]), r["metric"], r["contrast_type"]): float(r["response_coefficient"])
             for r in saved_rows}
    require(len(saved) == len(saved_rows) and bool(saved), "empty or duplicate saved coefficients")
    require(set(saved) == set(rebuilt), "saved coefficient grid mismatch")
    require(all(math.isfinite(value) for value in saved.values()), "nonfinite saved coefficient")
    max_error = max(abs(saved[key] - rebuilt[key]) for key in saved)
    require(max_error <= tolerance, f"raw coefficient mismatch: {max_error}")
    return {"raw": receipt(raw_path), "coefficients": receipt(coefficient_path), "raw_rows": len(rows),
            "active_token_units": len(active_units), "coefficient_count": len(saved), "max_abs_error": max_error, "passed": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    root = ROOT / protocol["run_root"]
    execution = json.loads((root / "execution_receipt.json").read_text())
    require(execution["status"] == "COMPLETED", "run is incomplete")
    require(execution["protocol"]["sha256"] == receipt(args.protocol)["sha256"], "run protocol differs")
    output = root / "independent_raw_audit.json"
    require(not output.exists(), "existing raw audit must not be overwritten")
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol": receipt(args.protocol),
              "audit_source": receipt(Path(__file__).resolve()), "tolerance": 1e-10,
              "method": "Independent standard-library reduction: direct steered log-probability differences, exact seed pairs, per-token signed fits, then equal token means within prompt bins. No production estimator or saved baseline deltas.",
              "arms": {}}
    for arm in protocol["arms"]:
        result["arms"][arm] = audit_arm(root / arm / "summary.csv",
            root / "analysis" / arm / "summary_prompt_bin_response_coefficients.csv", protocol, result["tolerance"])
    result["passed"] = True
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
