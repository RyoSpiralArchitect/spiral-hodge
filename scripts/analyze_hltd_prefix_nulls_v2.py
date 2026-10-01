#!/usr/bin/env python3
"""Describe saved v2 responses only after the calibration comparison completes."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import compare_hltd_prefix_nulls as comparison
from scripts import diagnose_hltd_prefix_calibration as cal
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, verify_frozen_files
from scripts.run_hltd_precision_gate import save_json, utc_now

OUTPUT = comparison.BASE + "/v2"
RESULT = "docs/data/hltd_prefix_transfer_v2/result"


def require_calibration(path, protocol, root=ROOT):
    output = root / comparison.OUTPUT
    if not (output / "manifest.json").exists():
        raise ValueError("Stage A calibration comparison is not complete")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["protocol"] != file_receipt(path):
        raise ValueError("Stage A protocol mismatch")
    expected = {"summary.json", "draws.csv", "comparison.csv", "prompts.csv", "masks.csv", "strata.csv", "replay_arrays.npz"}
    if ({Path(r["path"]).resolve() for r in manifest["outputs"]} != {output.resolve() / p for p in expected}
            or len(manifest["outputs"]) != len(expected)):
        raise ValueError("Stage A output inventory changed")
    verify_frozen_files({"frozen_files": manifest["outputs"]}, root)
    summary = json.loads((output / "summary.json").read_text())
    times = [protocol["frozen_utc"], summary["started_utc"], summary["completed_utc"], utc_now()]
    times = [datetime.fromisoformat(t) for t in times]
    if (any(t.tzinfo is None for t in times) or times != sorted(times)
            or summary["status"] != "DESCRIPTIVE_CALIBRATION_NULL_COMPARISON"
            or summary["protocol"] != file_receipt(path) or not summary["prior_replay"]["passed"]):
        raise ValueError("Stage A status or chronology invalid")
    return {"passed": True, "calibration_manifest": file_receipt(output / "manifest.json"),
            "calibration_completed_utc": summary["completed_utc"], "verified_utc": utc_now()}


def cell_descriptors(field, prompts, support):
    cells, atlas = support["cells"], field.atlas
    indices = np.asarray([c["neighbor_indices"] for c in cells], dtype=int)
    weights = np.asarray([c["weights"] for c in cells], dtype=float)
    metrics = cal.donor_metrics(atlas.coexact, indices, weights)
    tokens, bins = np.asarray(atlas.node_tokens), comparison.position_bins(atlas, prompts)
    rows = []
    for i, cell in enumerate(cells):
        if abs(metrics["prediction_norm"][i] - cell["coexact_chart_norm"]) > 1e-12:
            raise ValueError("saved aggregate norm disagrees with donors")
        query_token = cell["prefix_length"] - 1
        row = {"prompt_id": cell["prompt_id"], "family": cell["family"], "prefix_length": cell["prefix_length"],
               "query_token_index": query_token, "resultant_fraction": cal.finite(metrics["resultant_fraction"][i]),
               "cancellation_fraction": cal.finite(1 - metrics["resultant_fraction"][i]),
               "effective_contributors": cal.finite(metrics["effective_contributors"][i]),
               "pairwise_agreement": cal.finite(metrics["pairwise_agreement"][i]),
               "prediction_norm": float(metrics["prediction_norm"][i]),
               "nearest_distance": cell["neighbor_distances"][0], "eighth_distance": cell["neighbor_distances"][-1],
               "inactive_donors": int((metrics["donor_norms"][i] < atlas.min_chart_norm).sum()),
               "weighted_token_gap": float(weights[i] @ np.abs(tokens[indices[i]] - query_token))}
        for b, name in enumerate(cal.BIN_NAMES):
            row["donor_mass_" + name] = float(weights[i][bins[indices[i]] == b].sum())
        rows.append(row)
    return rows


def attach_coefficients(cells, coefficients, seeds):
    expected = {(r["prompt_id"], r["prefix_length"], seed) for r in cells for seed in seeds}
    indexed = {(r["prompt_id"], int(r["prefix_length"]), int(r["seed"])): r for r in coefficients}
    if len(indexed) != len(coefficients) or set(indexed) != expected:
        raise ValueError("coefficient grid incomplete or duplicated")
    result = []
    for cell in cells:
        records = [indexed[cell["prompt_id"], cell["prefix_length"], seed] for seed in seeds]
        if any(r["family"] != cell["family"] for r in records):
            raise ValueError("coefficient family mismatch")
        result.append({**cell, "seeds": len(seeds),
                       **{key: float(np.mean([float(r[key]) for r in records]))
                          for key in ("coexact_odd", "random_odd", "odd_gap")}})
    return result


def association_rows(cells, plan):
    numeric = [*plan["predictors"], *plan["outcomes"]]
    averaged = []
    for prompt in sorted({r["prompt_id"] for r in cells}):
        selected = [r for r in cells if r["prompt_id"] == prompt]
        averaged.append({"prompt_id": prompt, **{key: cal.describe([r[key] for r in selected])["mean"] for key in numeric}})
    scopes = {"pooled_cells": cells, **{f"prefix_{n}": [r for r in cells if r["prefix_length"] == n] for n in (8, 16, 24)},
              "prompt_means": averaged}
    result = []
    for scope in plan["association_scopes"]:
        rows = scopes[scope]
        for predictor, outcome in itertools.product(plan["predictors"], plan["outcomes"]):
            x, y = [np.array([r[key] for r in rows], dtype=float) for key in (predictor, outcome)]
            valid = np.isfinite(x) & np.isfinite(y)
            rho = (spearmanr(x[valid], y[valid]).statistic
                   if valid.sum() > 2 and np.ptp(x[valid]) > 0 and np.ptp(y[valid]) > 0 else np.nan)
            result.append({"scope": scope, "predictor": predictor, "outcome": outcome, "rows": len(rows),
                           "paired_rows": int(valid.sum()), "paired_prompts": len({r["prompt_id"] for r, v in zip(rows, valid, strict=True) if v}),
                           "spearman_rho": cal.finite(rho)})
    return result


def dose_rows(rows, spec):
    prompts = sorted({r["prompt_id"] for r in rows})
    settings = spec["evaluation"]
    indexed = {(r["prompt_id"], int(r["prefix_length"]), int(r["seed"]), r["component"], float(r["alpha"])): r for r in rows}
    expected = set(itertools.product(prompts, settings["prefix_lengths"], settings["seeds"], settings["components"], settings["alphas"]))
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("dose grid incomplete or duplicated")
    cells = []
    for prompt, length, component, a in itertools.product(prompts, settings["prefix_lengths"], settings["components"], spec["analysis"]["positive_magnitudes"]):
        plus, minus = [], []
        for seed in settings["seeds"]:
            pos, neg = indexed[prompt, length, seed, component, a], indexed[prompt, length, seed, component, -a]
            baseline = float(pos["next_token_logprob_base"])
            if baseline != float(neg["next_token_logprob_base"]):
                raise ValueError("signed pair baseline mismatch")
            plus.append(float(pos["next_token_logprob_steered"]) - baseline)
            minus.append(float(neg["next_token_logprob_steered"]) - baseline)
        dp, dm = float(np.mean(plus)), float(np.mean(minus))
        cells.append({"prompt_id": prompt, "prefix_length": length, "component": component, "magnitude": a,
                      "delta_plus": dp, "delta_minus": dm, "odd": (dp - dm) / 2, "even": (dp + dm) / 2})
    summary = []
    keys = ("delta_plus", "delta_minus", "odd", "even")
    for scope, lengths in [("all", settings["prefix_lengths"]), *[(f"prefix_{n}", [n]) for n in settings["prefix_lengths"]]]:
        for component, a in itertools.product(settings["components"], spec["analysis"]["positive_magnitudes"]):
            group = [r for r in cells if r["component"] == component and r["magnitude"] == a and r["prefix_length"] in lengths]
            summary.append({"scope": scope, "component": component, "magnitude": a, "cells": len(group), "prompts": len(prompts),
                            **{key: cal.balanced_mean([r[key] for r in group], [r["prompt_id"] for r in group])["mean"] for key in keys}})
    return cells, summary


def run(path, root=ROOT):
    protocol = json.loads(path.read_text())
    comparison.validate(protocol, root)
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(output)
    gate = require_calibration(path, protocol, root)
    started = utc_now()
    # These replay helpers operate on saved arrays. Never call runner.validate(),
    # which rebuilds a runtime/tokenizer contract, or any model runtime method.
    from scripts import analyze_hltd_prefix_v2 as replay
    from scripts import audit_hltd_prefix_transfer as independent

    v2 = cal.pinned_json(root / cal.REFERENCE, cal.REFERENCE_SHA)
    independent_audit = independent.audit(root / RESULT)
    source = root / v2["run_root"]
    support = json.loads((source / "support_preflight.json").read_text())
    pilot = json.loads((source / "pilot_audit.json").read_text())
    raw = comparison.read_csv(source / "raw_treatments.csv")
    replay.analysis.validate_rows(raw, v2, support)
    queries = replay.verify_query_replay(v2, source, pilot, support)
    replay.verify_treatment_inputs(raw, v2, source, support)
    coefficients = replay.analysis.primary_coefficients(raw, v2["spec"])
    primary, bootstrap = replay.analysis.summarize_primary(coefficients, v2["spec"])
    verdict = json.loads((root / RESULT / "verdict.json").read_text())
    with (root / RESULT / "bootstrap_indices.csv").open(newline="") as handle:
        saved_bootstrap = np.array([[int(i) for i in row] for row in csv.reader(handle)])
    if primary != verdict["primary"] or not np.array_equal(bootstrap, saved_bootstrap) or verdict["status"] != "NOT_SUPPORTED":
        raise ValueError("original v2 primary inference changed")
    field, prompts, _, _ = cal.load_inputs(root)
    cells = attach_coefficients(cell_descriptors(field, prompts, support), coefficients, v2["spec"]["evaluation"]["seeds"])
    plan = protocol["plan"]["v2"]
    associations = association_rows(cells, plan)
    dose_cells, doses = dose_rows(raw, v2["spec"])
    numeric = [*plan["predictors"], "cancellation_fraction", "prediction_norm", "inactive_donors", "eighth_distance",
               "coexact_odd", "random_odd", "odd_gap"]
    descriptors = {key: {"cells": cal.describe([r[key] for r in cells]),
                        "equal_prompt": cal.balanced_mean([r[key] for r in cells], [r["prompt_id"] for r in cells])} for key in numeric}
    comparison.validate(protocol, root)
    if require_calibration(path, protocol, root)["calibration_manifest"] != gate["calibration_manifest"]:
        raise ValueError("Stage A changed during Stage B")
    # Recheck raw and derived receipts after analysis, without changing old files.
    manifest = json.loads((root / RESULT / "manifest.json").read_text())
    verify_frozen_files({"frozen_files": [manifest["protocol"], *manifest["raw_files"], *manifest["derived_files"]]}, root)
    summary = {"status": "DESCRIPTIVE_SAVED_V2_ANALYSIS", "primary_status_unchanged": verdict["status"], "primary": primary,
               "cells": len(cells), "prompts": len({r["prompt_id"] for r in cells}), "raw_treatment_rows": len(raw),
               "model_loads": 0, "model_forward_calls": 0, "new_treatments": 0, "query_replay": queries,
               "independent_primary_audit": independent_audit, "stage_gate": gate, "descriptors": descriptors,
               "started_utc": started, "completed_utc": utc_now(), "scope": plan["scope"], "protocol": file_receipt(path)}
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("cells.csv", cells), ("associations.csv", associations), ("dose_cells.csv", dose_cells), ("dose_summary.csv", doses)):
        cal.write_csv(output / name, rows)
    save_json(output / "summary.json", summary)
    save_json(output / "manifest.json", {"protocol": file_receipt(path), "calibration_manifest": gate["calibration_manifest"],
              "v2_manifest": file_receipt(root / RESULT / "manifest.json"),
              "outputs": [file_receipt(p) for p in sorted(output.iterdir())]})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.protocol.absolute())
    print(json.dumps({key: result[key] for key in ("status", "primary_status_unchanged", "primary", "cells", "prompts",
                                                  "raw_treatment_rows", "model_forward_calls", "query_replay")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
