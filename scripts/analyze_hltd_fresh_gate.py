#!/usr/bin/env python3
"""Evaluate both fresh-text endpoints and the prespecified paired contrast."""
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

from scripts.analyze_hltd_l8_gate import activity_pairs, analyze_stage, early_values
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, verify_frozen_files
from scripts.run_hltd_fresh_gate import compare_charts, layer_protocol, validate_protocol
from scripts.run_hltd_precision_gate import save_json


def joint_decision(results: dict[int, dict]) -> str:
    if set(results) != {7, 8}:
        raise ValueError("both frozen layers are required")
    allowed = {"SUPPORTED_WITHIN_SAMPLE", "NOT_SUPPORTED", "INSUFFICIENT_COVERAGE"}
    if any(r["status"] not in allowed for r in results.values()):
        raise ValueError("unknown layer status")
    if any(r["status"] == "INSUFFICIENT_COVERAGE" for r in results.values()):
        return "INSUFFICIENT_COVERAGE"
    if all(r["status"] == "SUPPORTED_WITHIN_SAMPLE" for r in results.values()):
        return "BOTH_LAYERS_SUPPORTED_ON_FRESH_TEXTS"
    return "NOT_SUPPORTED_AT_BOTH_LAYERS"


def paired_contrast(coefficients: dict[int, pd.DataFrame], protocol: dict) -> tuple[pd.DataFrame, dict]:
    table = pd.DataFrame({f"l{layer}": early_values(coefficients[layer], protocol) for layer in [7, 8]}).sort_index()
    table["l8_minus_l7"] = table.l8 - table.l7
    values = table.l8_minus_l7.to_numpy()
    spec = protocol["paired_bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    draws = values[rng.integers(0, len(values), size=(spec["samples"], len(values)))].mean(axis=1)
    lower, upper = np.quantile(draws, spec["quantiles"])
    result = {"mean": float(values.mean()), "ci_lower": float(lower), "ci_upper": float(upper),
              "n_prompts": len(values), "positive_prompt_differences": int((values > 0).sum()),
              "bootstrap": spec, "boundary": "Secondary, unadjusted paired contrast with layer-relative dose norms, not a fixed-magnitude layer-only effect."}
    table["family"] = table.index.map({p["prompt_id"]: p["family"] for p in protocol["prompts"]})
    return table.reset_index(), result


def validate_zero_controls(zero: dict, protocol: dict, layer: int) -> None:
    records = zero["token_units"]
    expected = {(p["prompt_id"], t) for p in protocol["prompts"] for t in range(1, p["token_count"] - 1)}
    actual = {(r["prompt_id"], r["token_index"]) for r in records}
    if actual != expected or len(records) != len(expected) or not zero["passed"]:
        raise ValueError("incomplete, duplicate or failed zero controls")
    for row in records:
        if row["layer"] != layer or row["dtype"] != "float32":
            raise ValueError("changed zero-control layer or dtype")
        for name in ["zero_hook_max_abs", "row_spread_max_abs", "single_batch_logit_max_abs", "single_batch_next_logprob_delta"]:
            if not np.isfinite(row[name]):
                raise ValueError("nonfinite zero control")
        if not 0 <= row["zero_hook_max_abs"] <= protocol["tolerances"]["zero_hook_max_abs"] or row["row_spread_max_abs"] != 0:
            raise ValueError("zero-control tolerance failed")


def plot_paired(table: pd.DataFrame, contrast: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 8.5), sharey=True, constrained_layout=True)
    y = np.arange(len(table))
    axes[0].plot(table.l7, y - .12, "o", markersize=4, color="#426C8B", label="L7")
    axes[0].plot(table.l8, y + .12, "o", markersize=4, color="#147D73", label="L8")
    axes[1].plot(table.l8_minus_l7, y, "o", markersize=4, color="#BC445F")
    axes[0].set_yticks(y, [p.removeprefix("fresh_") for p in table.prompt_id], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("Early odd next-token coefficient")
    axes[1].set_title("Paired L8 minus L7")
    for ax in axes:
        ax.axvline(0, color="#444444", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_xlabel("Coexact minus random, nats / alpha")
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.055), ncol=2)
    axes[1].text(.5, -.09, f"Mean {contrast['mean']:+.3f}; paired 95% CI [{contrast['ci_lower']:+.3f}, {contrast['ci_upper']:+.3f}]",
                 transform=axes[1].transAxes, ha="center", fontsize=9)
    fig.suptitle("Twenty fresh authored texts; one native-FP32 runtime\nSame full-text charts, layer-relative dose norms; secondary contrast is unadjusted", fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    verify_frozen_files(protocol)
    validate_protocol(protocol)
    root = ROOT / protocol["run_root"]
    if (root / "gate_verdict.json").exists() or any((root / f"l{layer}/analysis").exists() for layer in [7, 8]):
        raise FileExistsError("existing analysis must not be overwritten")
    execution = json.loads((root / "execution_receipt.json").read_text())
    if execution["status"] != "COMPLETED" or execution["protocol"]["sha256"] != file_receipt(path)["sha256"]:
        raise ValueError("incomplete run or changed protocol")
    load = json.loads((root / "load_audit.json").read_text())
    if not load["model"]["all_bytes_equal"] or load["model"]["parameter_tensors"] != 148 or load["model_load_count"] != 1:
        raise ValueError("invalid one-model native-F32 source audit")
    if load["runtime"] != protocol["runtime"] or load["autocast"] or load["device_fallback"]:
        raise ValueError("changed runtime or precision route")
    if not json.loads((root / "final_weight_audit.json").read_text())["passed"]:
        raise ValueError("final weights did not match source")
    compare_charts(root, protocol)
    for layer in protocol["layers"]:
        validate_zero_controls(json.loads((root / f"l{layer}/zero_calibration.json").read_text()), protocol, layer)
    results, coefficients = {}, {}
    for layer in protocol["layers"]:
        results[layer] = analyze_stage(root / f"l{layer}", layer_protocol(protocol, layer))
        if results[layer]["complete_early_coverage"]:
            coefficients[layer] = pd.read_csv(root / f"l{layer}/analysis/summary_prompt_bin_response_coefficients.csv")
    result = {"status": joint_decision(results), "layers": results, "protocol": file_receipt(path),
              "interpretation_boundary": "New authored text strings, same families and offline full-text chart method. "
                  "Not external/blinded sampling, independent style, population replication, or semantic control."}
    if len(coefficients) == 2:
        table, contrast = paired_contrast(coefficients, protocol)
        table = table.set_index("prompt_id").reindex([p["prompt_id"] for p in protocol["prompts"]]).reset_index()
        table.to_csv(root / "early_paired_comparison.csv", index=False)
        result["paired_secondary"] = contrast
        plot_paired(table, contrast, root / "early_paired_comparison.png")
    frames = {layer: pd.read_csv(root / f"l{layer}/summary.csv") for layer in [7, 8]}
    activity = activity_pairs(frames[8], frames[7]).rename(columns={"current": "l8", "previous": "l7"})
    activity.to_csv(root / "activity_comparison.csv", index=False)
    result["activity"] = {"l7_active": int(activity.l7.sum()), "l8_active": int(activity.l8.sum()),
                          "different_masks": int((activity.l7 != activity.l8).sum())}
    norms = []
    for prompt in protocol["prompts"]:
        record = {"prompt_id": prompt["prompt_id"]}
        for layer in [7, 8]:
            values = frames[layer].loc[frames[layer].prompt_id == prompt["prompt_id"], "natural_step_norm"].unique()
            if len(values) != 1 or not np.isfinite(values[0]) or values[0] <= 0:
                raise ValueError("invalid per-prompt natural step norm")
            record[f"l{layer}_natural_step_norm"] = float(values[0])
        record["l8_to_l7_scale_ratio"] = record["l8_natural_step_norm"] / record["l7_natural_step_norm"]
        norms.append(record)
    pd.DataFrame(norms).to_csv(root / "dose_scale_comparison.csv", index=False)
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
