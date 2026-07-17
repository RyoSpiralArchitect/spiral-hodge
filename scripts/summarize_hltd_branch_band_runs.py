#!/usr/bin/env python3
"""Summarize executed HLTD branch-band follow-up runs."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def weighted_mean(rows: Sequence[Dict[str, Any]], value_key: str, weight_key: str = "matched_random_rows") -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = finite_float(row.get(value_key))
        weight = finite_float(row.get(weight_key))
        if not math.isfinite(value) or not math.isfinite(weight) or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator > 0 else float("nan")


def sum_int(rows: Iterable[Dict[str, Any]], key: str) -> int:
    total = 0
    for row in rows:
        value = finite_float(row.get(key))
        if math.isfinite(value):
            total += int(value)
    return total


def result_label(*, status: str, gate: float, target_minus_random: float) -> str:
    if status != "complete":
        return status
    if math.isfinite(gate) and gate >= 0.25 and math.isfinite(target_minus_random) and target_minus_random > 0.0:
        return "causal_support_confirmed"
    if math.isfinite(gate) and gate >= 0.25:
        return "gate_without_target_advantage"
    if math.isfinite(target_minus_random) and target_minus_random > 0.0:
        return "target_advantage_without_gate"
    return "not_confirmed"


def run_status(root: Path) -> str:
    if not root.exists():
        return "missing_run"
    if not (root / "closed_loop_prompt_layer_k_summary.csv").exists():
        return "missing_summary"
    return "complete"


def rows_for_plan(plan_row: Dict[str, Any]) -> List[Dict[str, str]]:
    root = Path(str(plan_row.get("output_root", "")))
    rows = read_csv(root / "closed_loop_prompt_layer_k_summary.csv")
    family = str(plan_row.get("family", ""))
    component = str(plan_row.get("component", ""))
    allowed_layers = {
        int(finite_float(value))
        for value in str(plan_row.get("layers", "")).split()
        if math.isfinite(finite_float(value))
    }
    out = []
    for row in rows:
        if str(row.get("family", "")) != family:
            continue
        if str(row.get("component", "")) != component:
            continue
        layer = int(finite_float(row.get("layer", -1)))
        if allowed_layers and layer not in allowed_layers:
            continue
        out.append(row)
    return out


def build_result_rows(plan_rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for plan_row in plan_rows:
        root = Path(str(plan_row.get("output_root", "")))
        status = run_status(root)
        rows = rows_for_plan(plan_row) if status == "complete" else []
        gate = weighted_mean(rows, "branch_specific_gate_rate")
        target = weighted_mean(rows, "mean_target_margin_delta_minus_random_mean")
        drift = weighted_mean(rows, "token_drift_rate_mean")
        random_gate = weighted_mean(rows, "random_branch_gate_rate")
        branch_gate = weighted_mean(rows, "branch_gate_rate")
        out.append(
            {
                "rank": plan_row.get("rank", ""),
                "family": plan_row.get("family", ""),
                "component": plan_row.get("component", ""),
                "candidate_label": plan_row.get("candidate_label", ""),
                "result_label": result_label(status=status, gate=gate, target_minus_random=target),
                "run_status": status,
                "priority_score": plan_row.get("priority_score", ""),
                "recommended_layers": plan_row.get("recommended_layers", ""),
                "layers": plan_row.get("layers", ""),
                "k_values": plan_row.get("k_values", ""),
                "alphas": plan_row.get("alphas", ""),
                "seeds": plan_row.get("seeds", ""),
                "planned_closed_loop_gate": plan_row.get("closed_loop_gate", ""),
                "planned_closed_loop_target_minus_random": plan_row.get("closed_loop_target_minus_random", ""),
                "result_branch_specific_gate_rate": gate,
                "result_branch_gate_rate": branch_gate,
                "result_random_branch_gate_rate": random_gate,
                "result_token_drift_rate_mean": drift,
                "result_target_margin_delta_minus_random_mean": target,
                "n_prompt_layer_k_rows": len(rows),
                "matched_random_rows": sum_int(rows, "matched_random_rows"),
                "output_root": str(root),
            }
        )
    return out


def build_layer_rows(plan_rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for plan_row in plan_rows:
        status = run_status(Path(str(plan_row.get("output_root", ""))))
        if status != "complete":
            continue
        buckets: Dict[int, List[Dict[str, str]]] = defaultdict(list)
        for row in rows_for_plan(plan_row):
            layer = int(finite_float(row.get("layer", -1)))
            buckets[layer].append(row)
        for layer, rows in sorted(buckets.items()):
            gate = weighted_mean(rows, "branch_specific_gate_rate")
            target = weighted_mean(rows, "mean_target_margin_delta_minus_random_mean")
            out.append(
                {
                    "rank": plan_row.get("rank", ""),
                    "family": plan_row.get("family", ""),
                    "component": plan_row.get("component", ""),
                    "layer": layer,
                    "result_label": result_label(status="complete", gate=gate, target_minus_random=target),
                    "branch_specific_gate_rate": gate,
                    "branch_gate_rate": weighted_mean(rows, "branch_gate_rate"),
                    "random_branch_gate_rate": weighted_mean(rows, "random_branch_gate_rate"),
                    "token_drift_rate_mean": weighted_mean(rows, "token_drift_rate_mean"),
                    "target_margin_delta_minus_random_mean": target,
                    "n_prompt_layer_k_rows": len(rows),
                    "matched_random_rows": sum_int(rows, "matched_random_rows"),
                    "output_root": plan_row.get("output_root", ""),
                }
            )
    return out


def fmt(value: Any) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.4f}"


def write_report(*, result_rows: Sequence[Dict[str, Any]], layer_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# HLTD Branch-Band Run Results",
        "",
        "This report maps executed closed-loop branch-band follow-ups back onto",
        "the candidate queue. Missing run roots stay visible as pending rows.",
        "",
        "## Branch-Band Results",
        "",
        "| rank | family | component | candidate | result | layers | gate | target-random | matched random |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in result_rows:
        lines.append(
            "| {rank} | {family} | {component} | {candidate} | {result} | {layers} | {gate} | {target} | {n} |".format(
                rank=row.get("rank", ""),
                family=row.get("family", ""),
                component=row.get("component", ""),
                candidate=row.get("candidate_label", ""),
                result=row.get("result_label", ""),
                layers=row.get("recommended_layers", ""),
                gate=fmt(row.get("result_branch_specific_gate_rate")),
                target=fmt(row.get("result_target_margin_delta_minus_random_mean")),
                n=row.get("matched_random_rows", 0),
            )
        )
    if layer_rows:
        lines.extend(
            [
                "",
                "## Layer Results",
                "",
                "| rank | family | component | layer | result | gate | target-random | matched random |",
                "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in layer_rows:
            lines.append(
                "| {rank} | {family} | {component} | L{layer} | {result} | {gate} | {target} | {n} |".format(
                    rank=row.get("rank", ""),
                    family=row.get("family", ""),
                    component=row.get("component", ""),
                    layer=row.get("layer", ""),
                    result=row.get("result_label", ""),
                    gate=fmt(row.get("branch_specific_gate_rate")),
                    target=fmt(row.get("target_margin_delta_minus_random_mean")),
                    n=row.get("matched_random_rows", 0),
                )
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-csv", default="spiral_out_hltd_branch_band_plan/branch_band_run_plan.csv")
    parser.add_argument("--output-root", default="spiral_out_hltd_branch_band_results")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    plan_rows = read_csv(Path(args.plan_csv))
    result_rows = build_result_rows(plan_rows)
    layer_rows = build_layer_rows(plan_rows)
    write_csv(result_rows, output_root / "branch_band_result_scoreboard.csv")
    write_csv(layer_rows, output_root / "branch_band_layer_result_summary.csv")
    write_report(
        result_rows=result_rows,
        layer_rows=layer_rows,
        output_path=output_root / "branch_band_result_report.md",
    )
    print(f"wrote branch-band result summaries -> {output_root}")
    print(f"branch-band result rows: {len(result_rows)}")
    print(f"branch-band layer rows: {len(layer_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
