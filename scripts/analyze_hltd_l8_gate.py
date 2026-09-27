#!/usr/bin/env python3
"""Evaluate the fixed L8 endpoint after a separately gated L7 runtime bridge."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_hltd_precision_raw import audit_arm
from scripts.evaluate_hltd_signed_layer_gate import (
    decide_primary, early_coverage, file_receipt, validate_protocol_rows, verify_frozen_files,
)
from scripts.plot_hltd_signed_position_gate import render_all
from scripts.run_hltd_precision_gate import save_json


def early_values(coefficients: pd.DataFrame, protocol: dict) -> pd.Series:
    primary = protocol["primary"]
    selected = coefficients[
        (coefficients.metric == primary["metric"]) & (coefficients.contrast_type == primary["contrast_type"])
        & coefficients.position_bin.isin(primary["bins"])
    ]
    expected = {(p["prompt_id"], b) for p in protocol["prompts"] for b in primary["bins"]}
    actual = set(selected[["prompt_id", "position_bin"]].itertuples(index=False, name=None))
    if actual != expected or len(selected) != len(expected) or not np.isfinite(selected.response_coefficient).all():
        raise ValueError("incomplete or duplicate early coefficients")
    return selected.groupby("prompt_id").response_coefficient.mean()


def activity_pairs(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    keys = ["prompt_id", "token_index"]
    frames = []
    for name, frame in [("current", current), ("previous", previous)]:
        selected = frame[frame.component == "coexact"][keys + ["component_active"]].drop_duplicates()
        frames.append(selected.rename(columns={"component_active": name}))
    result = frames[0].merge(frames[1], on=keys, how="outer", validate="one_to_one", indicator=True)
    if not (result["_merge"] == "both").all():
        raise ValueError("activity token grid differs")
    return result.drop(columns="_merge")


def compare_bridge(current: pd.DataFrame, previous: pd.DataFrame, current_rows: pd.DataFrame,
                   previous_rows: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, dict]:
    table = pd.DataFrame({"current": early_values(current, protocol), "previous": early_values(previous, protocol)})
    table["change"] = table.current - table.previous
    table["passed"] = (table.current > 0) & (table.change.abs() <= protocol["bridge"]["max_abs_early_change"])
    activity = activity_pairs(current_rows, previous_rows)
    changes = int((activity.current != activity.previous).sum())
    passed = bool(table.passed.all() and changes == 0)
    return table.reset_index(), {
        "status": "BRIDGE_PASSED" if passed else "BRIDGE_SENSITIVE", "passed": passed,
        "max_abs_early_change": float(table.change.abs().max()), "activity_changes": changes,
        "failing_prompts": table.index[~table.passed].tolist(), "prompt_count": len(table),
        "tolerance": protocol["bridge"]["max_abs_early_change"],
        "boundary": "Five known texts only; engineering continuity, not exact runtime equivalence on all 20 texts.",
    }


def analyze_stage(root: Path, protocol: dict) -> dict:
    output = root / "analysis"
    if output.exists():
        raise FileExistsError(output)
    rows = pd.read_csv(root / "summary.csv")
    grid = validate_protocol_rows(rows, protocol)
    for key, value in {"model_dtype": "float32", "field_dtype": "float32",
                       "baseline_mode": "matched_batch", "baseline_batch_size": 12}.items():
        if not (rows[key] == value).all():
            raise ValueError(f"changed native FP32 condition: {key}")
    coverage = early_coverage(rows, protocol)
    if not (coverage.n_active_tokens > 0).all():
        output.mkdir()
        result = {"status": "INSUFFICIENT_COVERAGE", "complete_early_coverage": False,
                  "missing_prompt_bins": coverage.loc[coverage.n_active_tokens == 0,
                                                      ["prompt_id", "position_bin"]].to_dict("records"),
                  "reason": "No imputation or reduced-prompt primary estimate."}
    else:
        analysis = protocol["analysis"]
        render_all(summary_path=root / "summary.csv", output_root=output, bins=analysis["bins"],
                   n_bootstrap=analysis["bootstrap_samples"], seed=analysis["bootstrap_seed"],
                   expected_prompts=len(protocol["prompts"]), expected_null_seeds=len(protocol["design"]["seeds"]),
                   expected_alpha_magnitudes=len(protocol["design"]["alphas"]) // 2)
        coefficients = output / "summary_prompt_bin_response_coefficients.csv"
        result = decide_primary(pd.read_csv(coefficients),
                                pd.read_csv(output / "summary_position_phase_bootstrap.csv"), protocol)
        result["positive_prompts"] = int((early_values(pd.read_csv(coefficients), protocol) > 0).sum())
        save_json(output / "independent_raw_audit.json", audit_arm(root / "summary.csv", coefficients, protocol))
    coverage.to_csv(output / "primary_early_coverage.csv", index=False)
    result.update({"design_validation": grid, "raw_receipt": file_receipt(root / "summary.csv")})
    save_json(output / "response_verdict.json", result)
    return result


def plot_layer_comparison(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True, constrained_layout=True)
    y = np.arange(len(table))
    axes[0].plot(table.l7_previous, y - 0.12, "o", markersize=4, color="#657080", label="L7, previous runtime")
    axes[0].plot(table.l8_current, y + 0.12, "o", markersize=4, color="#147D73", label="L8, current runtime")
    axes[1].plot(table.l8_minus_l7, y, "o", markersize=4, color="#BC445F")
    axes[0].set_yticks(y, table.prompt_id, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("Early odd next-token coefficient")
    axes[1].set_title("Descriptive L8 minus historical L7")
    for ax in axes:
        ax.axvline(0, color="#444444", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_xlabel("Coexact minus random, nats / alpha")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.suptitle("Native-FP32 layer extension: the same 20 previously seen texts\nDifferent runtimes; layer-relative dose scales. Not a pure layer-effect estimate.", fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(protocol_path: Path) -> dict:
    from scripts.run_hltd_l8_gate import validate_protocol

    protocol = json.loads(protocol_path.read_text())
    verify_frozen_files(protocol)
    validate_protocol(protocol)
    root = ROOT / protocol["run_root"]
    receipt = json.loads((root / "execution_receipt.json").read_text())
    if receipt["status"] != "COMPLETED" or receipt["protocol"]["sha256"] != file_receipt(protocol_path)["sha256"]:
        raise ValueError("incomplete execution or different protocol")
    bridge = json.loads((root / "bridge_verdict.json").read_text())
    if not bridge["passed"]:
        raise ValueError("L7 runtime bridge did not pass")
    for stage in ["bridge", "l8"]:
        if not json.loads((root / stage / "zero_calibration.json").read_text())["passed"]:
            raise ValueError(f"failed zero calibration: {stage}")
    load = json.loads((root / "load_audit.json").read_text())["model"]
    if not load["all_bytes_equal"] or load["parameter_tensors"] != 148 or load["parameter_count"] != 124439808:
        raise ValueError("incomplete native-F32 source audit")
    result = analyze_stage(root / "l8", protocol)
    result["protocol"] = file_receipt(protocol_path)
    result["runtime_bridge"] = bridge
    result["interpretation_boundary"] = (
        "Same-text native-FP32 L8 endpoint only. Not unseen-prompt replication, a causal online field, "
        "semantic control, fluency preservation, or proof of a layer difference."
    )
    if result["complete_early_coverage"]:
        old = pd.read_csv(ROOT / protocol["bridge"]["reference_coefficients"])
        new = pd.read_csv(root / "l8/analysis/summary_prompt_bin_response_coefficients.csv")
        table = pd.DataFrame({"l7_previous": early_values(old, protocol), "l8_current": early_values(new, protocol)})
        table["l8_minus_l7"] = table.l8_current - table.l7_previous
        table = table.reindex([p["prompt_id"] for p in protocol["prompts"]]).reset_index()
        table.to_csv(root / "l8/analysis/early_layer_comparison.csv", index=False)
        plot_layer_comparison(table, root / "l8/analysis/early_layer_comparison.png")
        result["descriptive_l8_minus_historical_l7_mean"] = float(table.l8_minus_l7.mean())
    save_json(root / "gate_verdict.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.protocol), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
