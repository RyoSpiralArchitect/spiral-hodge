#!/usr/bin/env python3
"""Compare precision arms with a prespecified per-prompt sensitivity tolerance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_hltd_signed_layer_gate import early_coverage, file_receipt, validate_protocol_rows, verify_frozen_files
from scripts.plot_hltd_signed_position_gate import render_all
from scripts.run_hltd_precision_gate import ARMS, require_fixed_field, save_json


def activity_comparison(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    activity = None
    for arm, rows in frames.items():
        selected = rows[rows["component"] == "coexact"][["prompt_id", "token_index", "component_active"]].drop_duplicates()
        selected = selected.rename(columns={"component_active": arm})
        activity = selected if activity is None else activity.merge(selected, on=["prompt_id", "token_index"], validate="one_to_one")
    return activity


def save_insufficient_coverage(
    frames: dict[str, pd.DataFrame], protocol: dict, protocol_path: Path, output: Path,
) -> dict | None:
    coverage = pd.concat([early_coverage(frames[arm], protocol).assign(precision_arm=arm) for arm in ARMS], ignore_index=True)
    missing = coverage[coverage["n_active_tokens"] == 0]
    if missing.empty:
        return None
    output.mkdir(parents=True, exist_ok=False)
    coverage.to_csv(output / "primary_early_coverage.csv", index=False)
    activity = activity_comparison(frames)
    activity.to_csv(output / "activity_comparison.csv", index=False)
    result = {
        "status": "INSUFFICIENT_COVERAGE", "complete_early_coverage": False,
        "missing_prompt_bins": missing[["precision_arm", "prompt_id", "position_bin"]].to_dict("records"),
        "missing_early_prompt_arms": missing[["precision_arm", "prompt_id"]].drop_duplicates().rename(
            columns={"precision_arm": "arm"}).to_dict("records"),
        "active_tokens": {arm: int(activity[arm].sum()) for arm in ARMS},
        "activity_changes_vs_fp16": {arm: int((activity[arm] != activity["fp16_replay"]).sum()) for arm in ARMS if arm != "fp16_replay"},
        "protocol": file_receipt(protocol_path), "primary": protocol["primary"],
        "reason": "No imputation or reduced-prompt estimate. All inference and rendering stopped before fitting; all arm coverage and activity retained.",
    }
    save_json(output / "precision_verdict.json", result)
    return result


def compare_early(coefficients: dict[str, pd.DataFrame], protocol: dict) -> tuple[pd.DataFrame, dict]:
    records = []
    missing = []
    for prompt in protocol["prompts"]:
        record = {"family": prompt["family"], "prompt_id": prompt["prompt_id"]}
        for arm in ARMS:
            data = coefficients[arm]
            selected = data[(data["prompt_id"] == prompt["prompt_id"])
                            & (data["metric"] == "next_token_logprob_delta")
                            & (data["contrast_type"] == "odd")
                            & data["position_bin"].isin(protocol["primary"]["bins"])]
            complete = set(selected["position_bin"]) == set(protocol["primary"]["bins"]) and len(selected) == 4
            if not complete or not np.isfinite(selected["response_coefficient"]).all():
                missing.append({"arm": arm, "prompt_id": prompt["prompt_id"]})
                record[arm] = np.nan
            else:
                record[arm] = float(selected["response_coefficient"].mean())
        records.append(record)
    table = pd.DataFrame(records)
    if missing:
        return table, {"status": "INSUFFICIENT_COVERAGE", "missing_early_prompt_arms": missing}
    result = {"n_prompts": len(table), "absolute_tolerance": protocol["tolerances"]["early_coefficient_max_abs_change"],
              "means": {}, "positive_prompts": {}, "max_abs_change": {}, "magnitude_stable": {}}
    for arm in ARMS:
        result["means"][arm] = float(table[arm].mean())
        result["positive_prompts"][arm] = int((table[arm] > 0).sum())
        if arm != "fp16_replay":
            difference = table[arm] - table["fp16_replay"]
            table[f"{arm}_minus_fp16"] = difference
            result["max_abs_change"][arm] = float(difference.abs().max())
            result["magnitude_stable"][arm] = bool(difference.abs().max() <= result["absolute_tolerance"])
    positive = all(count == len(table) for count in result["positive_prompts"].values())
    stable = all(result["magnitude_stable"].values())
    result["status"] = "PRECISION_STABLE_ON_PILOT" if positive and stable else "PRECISION_SENSITIVE_ON_PILOT"
    result["interpretation_boundary"] = "Four seen prompts; numerical sensitivity, not population equivalence, a new significance gate, or a full 20-prompt FP32 replication."
    return table, result


def plot_comparison(table: pd.DataFrame, tolerance: float, path: Path) -> None:
    labels = {"fp16_replay": "FP16 replay", "fp32_fixed_field": "FP32, fixed field", "fp32_rebuilt_field": "FP32, rebuilt field"}
    colors = {"fp16_replay": "#657080", "fp32_fixed_field": "#147D73", "fp32_rebuilt_field": "#BC445F"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    x = np.arange(len(table))
    for i, arm in enumerate(ARMS):
        axes[0].bar(x + (i - 1) * 0.24, table[arm], width=0.23, color=colors[arm], label=labels[arm])
        if arm != "fp16_replay":
            axes[1].plot(x, table[f"{arm}_minus_fp16"], marker="o", color=colors[arm], label=labels[arm])
    axes[1].axhspan(-tolerance, tolerance, color="#DDEBE5", alpha=0.65, label=f"Preset +/-{tolerance:g} tolerance")
    axes[0].set_title("Early odd next-token coefficient")
    axes[1].set_title("Change from FP16 (not confidence intervals)")
    axes[0].set_ylabel("Coexact minus random, nats / alpha")
    axes[1].set_ylabel("FP32 coefficient minus FP16 coefficient")
    for ax in axes:
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_xticks(x, table["prompt_id"], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("L7 precision sensitivity: four prespecified, previously seen prompts", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    verify_frozen_files(protocol)
    root = ROOT / protocol["run_root"]
    output = root / "analysis"
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((root / "execution_receipt.json").read_text())
    if receipt["status"] != "COMPLETED" or receipt["protocol"]["sha256"] != file_receipt(args.protocol)["sha256"]:
        raise ValueError("run is incomplete or bound to a different protocol")
    for name in ["zero_calibration.json", "replay_audit.json", "fixed_field_audit.json"]:
        if not json.loads((root / name).read_text())["passed"]:
            raise ValueError(f"failed prerequisite: {name}")
    require_fixed_field(json.loads((root / "fp16_replay/delta_trace.json").read_text()),
                        json.loads((root / "fp32_fixed_field/delta_trace.json").read_text()))
    frames = {}
    for arm, (model_dtype, field_dtype) in ARMS.items():
        rows = pd.read_csv(root / arm / "summary.csv")
        validate_protocol_rows(rows, protocol)
        for key, value in {"precision_arm": arm, "model_dtype": model_dtype, "field_dtype": field_dtype,
                           "baseline_mode": "matched_batch", "baseline_batch_size": 12}.items():
            if not bool((rows[key] == value).all()):
                raise ValueError(f"changed precision condition: {arm}/{key}")
        frames[arm] = rows
    missing = save_insufficient_coverage(frames, protocol, args.protocol, output)
    if missing is not None:
        print(json.dumps(missing, indent=2))
        return 0
    coefficients = {}
    for arm in ARMS:
        arm_output = output / arm
        render_all(summary_path=root / arm / "summary.csv", output_root=arm_output,
                   bins=protocol["analysis"]["bins"], n_bootstrap=protocol["analysis"]["bootstrap_samples"],
                   seed=protocol["analysis"]["bootstrap_seed"], expected_prompts=len(protocol["prompts"]),
                   expected_null_seeds=len(protocol["design"]["seeds"]), expected_alpha_magnitudes=3)
        coefficients[arm] = pd.read_csv(arm_output / "summary_prompt_bin_response_coefficients.csv")
    comparison, result = compare_early(coefficients, protocol)
    comparison.to_csv(output / "early_precision_comparison.csv", index=False)
    activity = activity_comparison(frames)
    activity.to_csv(output / "activity_comparison.csv", index=False)
    result["active_tokens"] = {arm: int(activity[arm].sum()) for arm in ARMS}
    result["activity_changes_vs_fp16"] = {arm: int((activity[arm] != activity["fp16_replay"]).sum()) for arm in ARMS if arm != "fp16_replay"}
    result["protocol"] = file_receipt(args.protocol)
    save_json(output / "precision_verdict.json", result)
    if result["status"] != "INSUFFICIENT_COVERAGE":
        plot_comparison(comparison, result["absolute_tolerance"], output / "precision_comparison.png")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
