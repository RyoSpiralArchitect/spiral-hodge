#!/usr/bin/env python3
"""Evaluate full-suite precision sensitivity separately from response support."""
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

from scripts.analyze_hltd_precision_gate import compare_early
from scripts.evaluate_hltd_signed_layer_gate import decide_primary, file_receipt, validate_protocol_rows, verify_frozen_files
from scripts.plot_hltd_signed_position_gate import render_all
from scripts.run_hltd_precision_gate import ARMS, require_fixed_field, save_json
from scripts.run_hltd_precision_full_gate import validate_full_protocol


def compare_full_early(coefficients: dict[str, pd.DataFrame], protocol: dict) -> tuple[pd.DataFrame, dict]:
    table, result = compare_early(coefficients, protocol)
    table["precision_subset"] = np.where(table["prompt_id"].isin(protocol["pilot_prompt_ids"]), "pilot4", "remaining16")
    result["interpretation_boundary"] = (
        "All 20 previously seen texts, including four with known FP32 pilot results. "
        "Engineering sensitivity only, not population equivalence, independent prompt replication, or semantic control."
    )
    if result["status"] == "INSUFFICIENT_COVERAGE":
        return table, result
    result["status"] = ("PRECISION_STABLE_ON_FULL_SUITE" if result["status"] == "PRECISION_STABLE_ON_PILOT"
                        else "PRECISION_SENSITIVE_ON_FULL_SUITE")
    result["failing_prompts"] = {}
    for arm in ARMS:
        fail = table[arm] <= 0
        if arm != "fp16_replay":
            fail = fail | (table[f"{arm}_minus_fp16"].abs() > result["absolute_tolerance"])
        result["failing_prompts"][arm] = table.loc[fail, "prompt_id"].tolist()
    result["subsets"] = {}
    for name, subset in table.groupby("precision_subset", sort=True):
        result["subsets"][name] = {
            "n_prompts": len(subset), "means": {arm: float(subset[arm].mean()) for arm in ARMS},
            "positive_prompts": {arm: int((subset[arm] > 0).sum()) for arm in ARMS},
            "max_abs_change": {arm: float(subset[f"{arm}_minus_fp16"].abs().max()) for arm in ARMS if arm != "fp16_replay"},
        }
    return table, result


def plot_full_comparison(table: pd.DataFrame, tolerance: float, path: Path) -> None:
    labels = {"fp16_replay": "FP16 replay", "fp32_fixed_field": "FP32, fixed field", "fp32_rebuilt_field": "FP32, rebuilt field"}
    colors = {"fp16_replay": "#657080", "fp32_fixed_field": "#147D73", "fp32_rebuilt_field": "#BC445F"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.4), sharey=True, constrained_layout=True)
    y = np.arange(len(table))
    for i, arm in enumerate(ARMS):
        axes[0].plot(table[arm], y + (i - 1) * 0.18, linestyle="none", marker="o", markersize=4,
                     color=colors[arm], label=labels[arm])
        if arm != "fp16_replay":
            axes[1].plot(table[f"{arm}_minus_fp16"], y + (i - 1.5) * 0.18, linestyle="none", marker="o",
                         markersize=4, color=colors[arm], label=labels[arm])
    axes[1].axvspan(-tolerance, tolerance, color="#DDEBE5", alpha=0.7, label=f"Preset +/-{tolerance:g} tolerance")
    axes[0].set_yticks(y, [f"{p}{' *' if p in set(table.loc[table.precision_subset == 'pilot4', 'prompt_id']) else ''}" for p in table.prompt_id], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("Early odd next-token response")
    axes[0].set_xlabel("Coexact minus random, nats / alpha")
    axes[1].set_title("Per-prompt change from FP16")
    axes[1].set_xlabel("FP32 coefficient minus FP16 coefficient")
    for ax in axes:
        ax.axvline(0, color="#444444", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=8)
    fig.suptitle("L7 native-FP32 precision: all 20 previously seen prompts\n* Previously checked in the FP32 pilot; shaded range is not a confidence interval", fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    verify_frozen_files(protocol)
    validate_full_protocol(protocol)
    root = ROOT / protocol["run_root"]
    output = root / "analysis"
    if output.exists():
        raise FileExistsError(output)
    execution = json.loads((root / "execution_receipt.json").read_text())
    if execution["status"] != "COMPLETED" or execution["protocol"]["sha256"] != file_receipt(protocol_path)["sha256"]:
        raise ValueError("run is incomplete or bound to a different protocol")
    for name in ["zero_calibration.json", "replay_audit.json", "fixed_field_audit.json"]:
        if not json.loads((root / name).read_text())["passed"]:
            raise ValueError(f"failed prerequisite: {name}")
    load = json.loads((root / "load_audit.json").read_text())
    for dtype in ["float16", "float32"]:
        audit = load["models"][dtype]
        if not audit.get("all_bytes_equal") or audit["parameter_tensors"] != 148 or audit["parameter_count"] != 124439808:
            raise ValueError(f"missing in-run source-byte audit: {dtype}")
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
    coefficients = {}
    response_verdicts = {}
    for arm in ARMS:
        arm_output = output / arm
        render_all(summary_path=root / arm / "summary.csv", output_root=arm_output,
                   bins=protocol["analysis"]["bins"], n_bootstrap=protocol["analysis"]["bootstrap_samples"],
                   seed=protocol["analysis"]["bootstrap_seed"], expected_prompts=20,
                   expected_null_seeds=8, expected_alpha_magnitudes=3)
        coefficients[arm] = pd.read_csv(arm_output / "summary_prompt_bin_response_coefficients.csv")
        response_verdicts[arm] = decide_primary(
            coefficients[arm], pd.read_csv(arm_output / "summary_position_phase_bootstrap.csv"), protocol,
        )
    comparison, result = compare_full_early(coefficients, protocol)
    comparison.to_csv(output / "early_precision_comparison.csv", index=False)
    activity = None
    for arm, rows in frames.items():
        selected = rows[rows["component"] == "coexact"][["prompt_id", "token_index", "component_active"]].drop_duplicates()
        selected = selected.rename(columns={"component_active": arm})
        activity = selected if activity is None else activity.merge(selected, on=["prompt_id", "token_index"], validate="one_to_one")
    activity.to_csv(output / "activity_comparison.csv", index=False)
    result["active_tokens"] = {arm: int(activity[arm].sum()) for arm in ARMS}
    result["activity_changes_vs_fp16"] = {arm: int((activity[arm] != activity["fp16_replay"]).sum()) for arm in ARMS if arm != "fp16_replay"}
    result["response_support"] = response_verdicts
    result["protocol"] = file_receipt(protocol_path)
    save_json(output / "precision_verdict.json", result)
    if result["status"] != "INSUFFICIENT_COVERAGE":
        plot_full_comparison(comparison, result["absolute_tolerance"], output / "precision_comparison.png")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.protocol), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
