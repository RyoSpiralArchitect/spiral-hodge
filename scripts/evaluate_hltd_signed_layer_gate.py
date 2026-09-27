#!/usr/bin/env python3
"""Check a frozen signed layer design and evaluate its single early endpoint."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_hltd_signed_position_gate import render_all


def file_receipt(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def verify_frozen_files(protocol: dict[str, Any], root: Path = ROOT) -> None:
    for expected in protocol["frozen_files"]:
        path = root / expected["path"]
        actual = file_receipt(path)
        if any(actual[key] != expected[key] for key in ("sha256", "bytes")):
            raise ValueError(f"frozen file changed: {path}")


def validate_protocol_rows(rows: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    """Reject omissions, replacements, duplicates and changed conditions before inference."""
    design = protocol["design"]
    keys = ["family", "prompt_id", "node_index", "token_index", "token_count", "seed", "component", "alpha"]
    metrics = ["next_token_logprob_delta", "semantic_margin_delta"]
    required = set(keys + metrics + ["component_active", "next_token_logprob_base", "next_token_logprob_steered"])
    required.update(design["row_constants"])
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"missing raw columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError("empty signed layer run")
    for key, value in design["row_constants"].items():
        if not bool((rows[key] == value).all()):
            raise ValueError(f"frozen condition mismatch: {key}")
    numeric = ["node_index", "token_index", "token_count", "seed", "alpha", "component_active", *metrics,
               "next_token_logprob_base", "next_token_logprob_steered"]
    if not bool(np.isfinite(rows[numeric].to_numpy(dtype=float)).all()):
        raise ValueError("raw gate values must be finite")
    for key in ["node_index", "token_index", "token_count", "seed"]:
        if not bool((rows[key] == np.floor(rows[key])).all()):
            raise ValueError(f"noninteger design value: {key}")
    if not bool(rows["component_active"].isin([0, 1]).all()):
        raise ValueError("component_active must be 0 or 1")
    if bool(rows.duplicated(keys).any()):
        raise ValueError("duplicate raw design units")
    expected = set()
    for prompt in protocol["prompts"]:
        for token, seed, component, alpha in itertools.product(
            range(1, prompt["token_count"] - 1), design["seeds"], design["components"], design["alphas"],
        ):
            expected.add((prompt["family"], prompt["prompt_id"], token - 1, token,
                          prompt["token_count"], seed, component, alpha))
    actual = set(rows[keys].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"raw grid mismatch: {len(expected - actual)} missing, {len(actual - expected)} unexpected")
    baseline_spread = rows.groupby(["prompt_id", "token_index"])["next_token_logprob_base"].agg(["min", "max"])
    if not bool(((baseline_spread["max"] - baseline_spread["min"]).abs() <= 1e-12).all()):
        raise ValueError("coexact and random must share the same unsteered baseline")
    if not np.allclose(
        rows["next_token_logprob_delta"],
        rows["next_token_logprob_steered"] - rows["next_token_logprob_base"], rtol=0, atol=1e-10,
    ):
        raise ValueError("next-token delta does not match saved probabilities")
    activity = rows.groupby(["prompt_id", "token_index", "component", "seed"])["component_active"].nunique()
    if not bool((activity == 1).all()):
        raise ValueError("activity must not depend on alpha/sign")
    coexact = rows[rows["component"] == "coexact"]
    coexact_activity = coexact.groupby(["prompt_id", "token_index"])["component_active"].nunique()
    if not bool((coexact_activity == 1).all()):
        raise ValueError("deterministic coexact activity must not depend on random seed")
    random_activity = rows[rows["component"] == "random_tangent"].groupby(["prompt_id", "token_index"])["component_active"].min()
    active_coexact = coexact.groupby(["prompt_id", "token_index"])["component_active"].max()
    if bool(((active_coexact > 0) & (random_activity < 1)).any()):
        raise ValueError("active coexact requires all frozen random control seeds")
    return {"raw_rows": len(rows), "candidate_token_units": sum(p["token_count"] - 2 for p in protocol["prompts"]),
            "active_coexact_token_units": len(coexact[coexact["component_active"] == 1][["prompt_id", "token_index"]].drop_duplicates()),
            "n_prompts": len(protocol["prompts"]), "exact_grid_passed": True}


def decide_primary(
    coefficients: pd.DataFrame, phases: pd.DataFrame, protocol: dict[str, Any],
) -> dict[str, Any]:
    primary = protocol["primary"]
    expected = set(itertools.product([p["prompt_id"] for p in protocol["prompts"]], primary["bins"]))
    selected = coefficients[
        (coefficients["metric"] == primary["metric"])
        & (coefficients["contrast_type"] == primary["contrast_type"])
        & coefficients["position_bin"].isin(primary["bins"])
    ]
    actual = set(selected[["prompt_id", "position_bin"]].itertuples(index=False, name=None))
    coverage = actual == expected and len(selected) == len(expected)
    coverage = coverage and bool(np.isfinite(selected["response_coefficient"]).all())
    result: dict[str, Any] = {
        "status": "INSUFFICIENT_COVERAGE", "complete_early_coverage": bool(coverage),
        "expected_prompt_bins": len(expected), "observed_prompt_bins": len(actual),
        "missing_prompt_bins": sorted(expected - actual), "primary": primary,
        "interpretation_boundary": "Within-sample layer generalization only; not independent prompt replication or semantic control.",
    }
    if not coverage:
        return result
    estimates = phases[
        (phases["metric"] == primary["metric"])
        & (phases["contrast_type"] == primary["contrast_type"])
        & (phases["position_phase"] == "early")
    ]
    if len(estimates) != 1:
        raise ValueError("expected exactly one primary estimate")
    row = estimates.iloc[0]
    if int(row["n_prompts"]) != len(protocol["prompts"]):
        raise ValueError("primary prompt count mismatch")
    if not bool(np.isfinite(row[["mean_response_coefficient", "bootstrap_ci_lower", "bootstrap_ci_upper"]].to_numpy(dtype=float)).all()):
        raise ValueError("nonfinite primary estimate")
    result.update({
        "status": "SUPPORTED_WITHIN_SAMPLE" if float(row["bootstrap_ci_lower"]) > 0 else "NOT_SUPPORTED",
        "mean": float(row["mean_response_coefficient"]),
        "ci_lower": float(row["bootstrap_ci_lower"]), "ci_upper": float(row["bootstrap_ci_upper"]),
        "n_prompts": int(row["n_prompts"]),
    })
    return result


def early_coverage(rows: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    coexact = rows[(rows["component"] == "coexact") & (rows["component_active"] == 1)]
    tokens = coexact[["prompt_id", "token_index", "token_count"]].drop_duplicates().copy()
    tokens["position_bin"] = np.floor(protocol["analysis"]["bins"] * tokens["token_index"] / (tokens["token_count"] - 1)).astype(int)
    counts = tokens.groupby(["prompt_id", "position_bin"]).size()
    return pd.DataFrame([
        {"prompt_id": prompt["prompt_id"], "position_bin": b,
         "n_active_tokens": int(counts.get((prompt["prompt_id"], b), 0))}
        for prompt in protocol["prompts"] for b in protocol["primary"]["bins"]
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    verify_frozen_files(protocol)
    if args.summary.resolve() != (ROOT / protocol["run_root"] / "summary.csv").resolve():
        raise ValueError("summary is outside the frozen run root")
    rows = pd.read_csv(args.summary)
    design_receipt = validate_protocol_rows(rows, protocol)
    analysis = protocol["analysis"]
    coverage = early_coverage(rows, protocol)
    if bool((coverage["n_active_tokens"] > 0).all()):
        render_all(
            summary_path=args.summary, output_root=args.output_root, bins=analysis["bins"],
            n_bootstrap=analysis["bootstrap_samples"], seed=analysis["bootstrap_seed"],
            expected_prompts=len(protocol["prompts"]), expected_null_seeds=len(protocol["design"]["seeds"]),
            expected_alpha_magnitudes=len(protocol["design"]["alphas"]) // 2,
        )
        result = decide_primary(
            pd.read_csv(args.output_root / "summary_prompt_bin_response_coefficients.csv"),
            pd.read_csv(args.output_root / "summary_position_phase_bootstrap.csv"), protocol,
        )
    else:
        args.output_root.mkdir(parents=True)
        result = {"status": "INSUFFICIENT_COVERAGE", "complete_early_coverage": False,
                  "missing_prompt_bins": coverage[coverage["n_active_tokens"] == 0][["prompt_id", "position_bin"]].to_dict("records"),
                  "primary": protocol["primary"], "reason": "No imputation or reduced-prompt primary estimate."}
    coverage.to_csv(args.output_root / "primary_early_coverage.csv", index=False)
    result.update({"evaluated_utc": datetime.now(timezone.utc).isoformat(), "design_validation": design_receipt,
                   "protocol_receipt": file_receipt(args.protocol), "raw_receipt": file_receipt(args.summary)})
    (args.output_root / "gate_verdict.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
