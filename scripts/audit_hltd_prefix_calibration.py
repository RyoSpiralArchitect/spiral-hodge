#!/usr/bin/env python3
"""Independent stdlib checks of saved diagnostic tables, not vector replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

PROTOCOL_SHA = "a58a24fad8b05015cfebf9a88a5bf6f49c2f405e3add042bbe80bae1ff1ce681"


def receipt(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def check_receipt(record):
    if receipt(Path(record["path"])) != record:
        raise ValueError("changed artifact: " + record["path"])


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def distribution(values):
    clean = sorted(float(x) for x in values if x not in (None, ""))
    if any(not math.isfinite(x) for x in clean):
        raise ValueError("nonfinite diagnostic table")
    def quantile(q):
        if not clean:
            return None
        position = (len(clean) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return clean[low] + (clean[high] - clean[low]) * (position - low)
    return {"n_total": len(values), "n_valid": len(clean),
            "mean": math.fsum(clean) / len(clean) if clean else None,
            "q10": quantile(.1), "median": quantile(.5), "q90": quantile(.9)}


def grouped_means(rows, metric, owners):
    grouped = defaultdict(list)
    for row in rows:
        if row[metric] != "":
            grouped[row["prompt_id"]].append(float(row[metric]))
    return {owner: math.fsum(grouped[owner]) / len(grouped[owner]) if grouped[owner] else None for owner in owners}


def audit(result):
    manifest = json.loads((result / "manifest.json").read_text())
    if manifest["protocol"]["sha256"] != PROTOCOL_SHA:
        raise ValueError("unexpected diagnostic protocol")
    for record in [manifest["protocol"], *manifest["outputs"]]:
        check_receipt(record)
    protocol = json.loads(Path(manifest["protocol"]["path"]).read_text())
    for record in protocol["frozen_files"]:
        check_receipt(record)
    summary = json.loads((result / "summary.json").read_text())
    nodes, prompts, controls, positions = [read_csv(result / name) for name in
        ("nodes.csv", "prompts.csv", "controls.csv", "position_mix.csv")]
    owners = sorted(protocol["prompts"])
    expected = [(p, t) for p in owners for t in range(1, protocol["prompts"][p]["token_count"] - 1)]
    if ([(r["prompt_id"], int(r["token_index"])) for r in nodes] != expected
            or [int(r["node_index"]) for r in nodes] != list(range(len(expected)))
            or [r["prompt_id"] for r in prompts] != owners
            or any(int(r["prefix_length"]) != int(r["token_index"]) + 1 for r in nodes)):
        raise ValueError("changed calibration inventory")
    if dict(Counter(r["recovery_status"] for r in nodes)) != summary["recovery_status_counts"]:
        raise ValueError("changed recovery counts")
    if any((r["recovery_cosine"] != "") != (r["recovery_status"] == "DEFINED") for r in nodes):
        raise ValueError("undefined cosine was scored")
    errors = []

    def compare(expected, recorded):
        if expected is None or recorded is None:
            if expected is not recorded:
                raise ValueError("changed missing-value mask")
        else:
            if not math.isfinite(float(recorded)):
                raise ValueError("nonfinite summary")
            errors.append(abs(expected - recorded))

    def check_distribution(values, recorded):
        for key, value in distribution(values).items():
            compare(value, recorded[key])

    for metric, stats in summary["statistics"].items():
        check_distribution([r[metric] for r in nodes], stats["nodes"])
        means = grouped_means(nodes, metric, owners)
        check_distribution(list(means.values()), stats["equal_prompt"])
        for row in prompts:
            compare(means[row["prompt_id"]], float(row[metric]) if row[metric] else None)
    supported = [r for r in nodes if r["distance_supported"] == "True"]
    compare(len(nodes), summary["nodes"])
    compare(len(owners), summary["prompts"])
    compare(len(supported), summary["distance_supported_nodes"])
    compare(sum(r["recovery_cosine"] != "" for r in nodes), summary["defined_recovery_nodes"])
    compare(sum(r["recovery_cosine"] != "" for r in supported), summary["supported_recovery_nodes"])
    check_distribution(list(grouped_means(supported, "recovery_cosine", owners).values()), summary["supported_recovery_equal_prompt"])
    expected_controls = {(name, draw, scope) for name, draws in
        (("v2_random", protocol["method"]["random_seeds"]), ("within_prompt_shuffle", range(protocol["method"]["shuffle_draws"])))
        for draw, scope in itertools.product(draws, ("all", "supported"))}
    actual_controls = [(r["comparator"], int(r["draw"]), r["scope"]) for r in controls]
    if len(actual_controls) != len(expected_controls) or set(actual_controls) != expected_controls:
        raise ValueError("changed control inventory")
    for name, by_scope in summary["controls"].items():
        for scope, fields in by_scope.items():
            subset = [r for r in controls if r["comparator"] == name and r["scope"] == scope]
            for key, stats in fields.items():
                check_distribution([r[key] for r in subset], stats)
    for source in ("early", "middle", "late"):
        subset = [r for r in positions if r["query_bin"] == source]
        if sorted(r["donor_bin"] for r in subset) != ["early", "late", "middle"]:
            raise ValueError("changed position inventory")
        for metric in ("geometric_mass", "contribution_mass"):
            compare(1., math.fsum(float(r[metric]) for r in subset))
    if max(errors) > 1e-12:
        raise ValueError("saved diagnostic arithmetic differs")
    return {"passed": True, "nodes": len(nodes), "prompts": len(prompts), "control_rows": len(controls),
            "max_arithmetic_error": max(errors), "manifest": receipt(result / "manifest.json"),
            "auditor": receipt(Path(__file__).resolve()),
            "scope": "Independent stdlib hash, inventory, distribution, and equal-prompt aggregation checks; not independent vector, neighbor, or model execution replay."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.result.absolute())
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
