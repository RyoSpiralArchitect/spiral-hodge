#!/usr/bin/env python3
"""Summarize HLTD closed-loop branch steering outputs."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BASELINE_COMPONENT = "baseline"
RANDOM_COMPONENT = "random_tangent"


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def mean_finite(values: Iterable[Any]) -> float:
    vals = [finite_float(value) for value in values]
    vals = [value for value in vals if math.isfinite(value)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def rate_true(values: Iterable[bool]) -> float:
    vals = [bool(value) for value in values]
    if not vals:
        return float("nan")
    return float(sum(1 for value in vals if value) / len(vals))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
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


def row_target_set(row: Dict[str, Any]) -> str:
    return str(row.get("target_set", ""))


def run_key(row: Dict[str, Any]) -> Tuple[str, str, int, int, int, str]:
    return (
        str(row.get("family", "")),
        str(row.get("prompt_id", "")),
        int(finite_float(row.get("layer", -1))),
        int(finite_float(row.get("k", -1))),
        int(finite_float(row.get("seed", -1))),
        row_target_set(row),
    )


def random_match_key(row: Dict[str, Any]) -> Tuple[str, str, int, int, int, str, float]:
    family, prompt_id, layer, k, seed, target_set = run_key(row)
    return (family, prompt_id, layer, k, seed, target_set, finite_float(row.get("alpha")))


def branch_gate(row: Dict[str, Any]) -> bool:
    drift = finite_float(row.get("token_drift_rate"))
    target = finite_float(row.get("mean_target_margin_delta"))
    return math.isfinite(drift) and math.isfinite(target) and drift >= 0.5 and target > 0.0


def branch_specific_gate(row: Dict[str, Any], random_row: Dict[str, Any]) -> bool:
    drift = finite_float(row.get("token_drift_rate"))
    random_drift = finite_float(random_row.get("token_drift_rate"))
    target = finite_float(row.get("mean_target_margin_delta"))
    random_target = finite_float(random_row.get("mean_target_margin_delta"))
    return (
        branch_gate(row)
        and math.isfinite(drift)
        and math.isfinite(random_drift)
        and math.isfinite(target)
        and math.isfinite(random_target)
        and drift >= random_drift
        and target > random_target
    )


def contrast_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    baselines: Dict[Tuple[str, str, int, int, int], Dict[str, str]] = {}
    for row in rows:
        if str(row.get("component", "")) == BASELINE_COMPONENT:
            baselines[run_key(row)] = row

    out: List[Dict[str, Any]] = []
    for row in rows:
        component = str(row.get("component", ""))
        if component == BASELINE_COMPONENT:
            continue
        baseline = baselines.get(run_key(row), {})
        overlap = finite_float(row.get("baseline_token_overlap"))
        baseline_logp = finite_float(baseline.get("mean_selected_base_logprob"))
        branch_logp = finite_float(row.get("mean_selected_base_logprob"))
        out.append(
            {
                "family": row.get("family", ""),
                "prompt_id": row.get("prompt_id", ""),
                "layer": int(finite_float(row.get("layer", -1))),
                "k": int(finite_float(row.get("k", -1))),
                "seed": int(finite_float(row.get("seed", -1))),
                "component": component,
                "alpha": finite_float(row.get("alpha")),
                "target_set": row_target_set(row),
                "generated_steps": int(finite_float(row.get("generated_steps", 0))),
                "baseline_token_overlap": overlap,
                "token_drift_rate": 1.0 - overlap if math.isfinite(overlap) else float("nan"),
                "mean_selected_base_logprob": branch_logp,
                "mean_selected_base_logprob_delta": (
                    branch_logp - baseline_logp if math.isfinite(branch_logp) and math.isfinite(baseline_logp) else float("nan")
                ),
                "mean_selected_logprob_gain": finite_float(row.get("mean_selected_logprob_gain")),
                "mean_kl_base_to_steered": finite_float(row.get("mean_kl_base_to_steered")),
                "mean_entropy_delta": finite_float(row.get("mean_entropy_delta")),
                "mean_target_margin_delta": finite_float(row.get("mean_target_margin_delta")),
                "mean_nearest_distance": finite_float(row.get("mean_nearest_distance")),
                "unique_nodes": int(finite_float(row.get("unique_nodes", 0))),
                "top_changed_rate": finite_float(row.get("top_changed_rate")),
                "baseline_generated_text": baseline.get("generated_text", ""),
                "generated_text": row.get("generated_text", ""),
            }
        )
    return out


def component_summary_rows(contrasts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in contrasts:
        groups[(str(row["component"]), finite_float(row["alpha"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (component, alpha), group in sorted(groups.items()):
        out.append(
            {
                "component": component,
                "alpha": alpha,
                "n_rows": len(group),
                "token_drift_rate_mean": mean_finite(row.get("token_drift_rate") for row in group),
                "baseline_token_overlap_mean": mean_finite(row.get("baseline_token_overlap") for row in group),
                "mean_selected_base_logprob_mean": mean_finite(row.get("mean_selected_base_logprob") for row in group),
                "mean_selected_base_logprob_delta_mean": mean_finite(
                    row.get("mean_selected_base_logprob_delta") for row in group
                ),
                "mean_selected_logprob_gain_mean": mean_finite(row.get("mean_selected_logprob_gain") for row in group),
                "mean_kl_base_to_steered_mean": mean_finite(row.get("mean_kl_base_to_steered") for row in group),
                "mean_entropy_delta_mean": mean_finite(row.get("mean_entropy_delta") for row in group),
                "mean_target_margin_delta_mean": mean_finite(row.get("mean_target_margin_delta") for row in group),
                "mean_nearest_distance_mean": mean_finite(row.get("mean_nearest_distance") for row in group),
                "top_changed_rate_mean": mean_finite(row.get("top_changed_rate") for row in group),
            }
        )
    return out


def family_summary_rows(contrasts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in contrasts:
        groups[(str(row["family"]), str(row["component"]), finite_float(row["alpha"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (family, component, alpha), group in sorted(groups.items()):
        out.append(
            {
                "family": family,
                "component": component,
                "alpha": alpha,
                "n_rows": len(group),
                "token_drift_rate_mean": mean_finite(row.get("token_drift_rate") for row in group),
                "mean_selected_logprob_gain_mean": mean_finite(row.get("mean_selected_logprob_gain") for row in group),
                "mean_kl_base_to_steered_mean": mean_finite(row.get("mean_kl_base_to_steered") for row in group),
                "mean_target_margin_delta_mean": mean_finite(row.get("mean_target_margin_delta") for row in group),
                "mean_nearest_distance_mean": mean_finite(row.get("mean_nearest_distance") for row in group),
            }
        )
    return out


def layer_summary_rows(contrasts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in contrasts:
        groups[(int(finite_float(row["layer"])), str(row["component"]), finite_float(row["alpha"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (layer, component, alpha), group in sorted(groups.items()):
        out.append(
            {
                "layer": layer,
                "component": component,
                "alpha": alpha,
                "n_rows": len(group),
                "token_drift_rate_mean": mean_finite(row.get("token_drift_rate") for row in group),
                "baseline_token_overlap_mean": mean_finite(row.get("baseline_token_overlap") for row in group),
                "mean_selected_base_logprob_mean": mean_finite(row.get("mean_selected_base_logprob") for row in group),
                "mean_selected_base_logprob_delta_mean": mean_finite(
                    row.get("mean_selected_base_logprob_delta") for row in group
                ),
                "mean_selected_logprob_gain_mean": mean_finite(row.get("mean_selected_logprob_gain") for row in group),
                "mean_kl_base_to_steered_mean": mean_finite(row.get("mean_kl_base_to_steered") for row in group),
                "mean_entropy_delta_mean": mean_finite(row.get("mean_entropy_delta") for row in group),
                "mean_target_margin_delta_mean": mean_finite(row.get("mean_target_margin_delta") for row in group),
                "mean_nearest_distance_mean": mean_finite(row.get("mean_nearest_distance") for row in group),
                "top_changed_rate_mean": mean_finite(row.get("top_changed_rate") for row in group),
            }
        )
    return out


def k_summary_rows(contrasts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in contrasts:
        groups[(int(finite_float(row["k"])), str(row["component"]), finite_float(row["alpha"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (k, component, alpha), group in sorted(groups.items()):
        out.append(
            {
                "k": k,
                "component": component,
                "alpha": alpha,
                "n_rows": len(group),
                "token_drift_rate_mean": mean_finite(row.get("token_drift_rate") for row in group),
                "baseline_token_overlap_mean": mean_finite(row.get("baseline_token_overlap") for row in group),
                "mean_selected_base_logprob_mean": mean_finite(row.get("mean_selected_base_logprob") for row in group),
                "mean_selected_base_logprob_delta_mean": mean_finite(
                    row.get("mean_selected_base_logprob_delta") for row in group
                ),
                "mean_selected_logprob_gain_mean": mean_finite(row.get("mean_selected_logprob_gain") for row in group),
                "mean_kl_base_to_steered_mean": mean_finite(row.get("mean_kl_base_to_steered") for row in group),
                "mean_entropy_delta_mean": mean_finite(row.get("mean_entropy_delta") for row in group),
                "mean_target_margin_delta_mean": mean_finite(row.get("mean_target_margin_delta") for row in group),
                "mean_nearest_distance_mean": mean_finite(row.get("mean_nearest_distance") for row in group),
                "top_changed_rate_mean": mean_finite(row.get("top_changed_rate") for row in group),
            }
        )
    return out


def prompt_summary_rows(contrasts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    random_rows: Dict[Tuple[str, str, int, int, int, float], Dict[str, Any]] = {}
    for row in contrasts:
        if str(row.get("component", "")) == RANDOM_COMPONENT:
            random_rows[random_match_key(row)] = row

    groups: Dict[Tuple[str, str, str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in contrasts:
        groups[
            (
                str(row.get("family", "")),
                str(row.get("prompt_id", "")),
                row_target_set(row),
                str(row.get("component", "")),
                finite_float(row.get("alpha")),
            )
        ].append(row)

    out: List[Dict[str, Any]] = []
    for (family, prompt_id, target_set, component, alpha), group in sorted(groups.items()):
        drift_values = [finite_float(row.get("token_drift_rate")) for row in group]
        target_values = [finite_float(row.get("mean_target_margin_delta")) for row in group]
        gate_values = [branch_gate(row) for row in group]
        random_for_group = [random_rows.get(random_match_key(row)) for row in group]
        matched_pairs = [(row, random_row) for row, random_row in zip(group, random_for_group) if random_row is not None]
        random_gate_values = [branch_gate(row) for row in random_for_group if row is not None]
        gate_minus_random = [
            float(branch_gate(row)) - float(branch_gate(random_row))
            for row, random_row in matched_pairs
        ]
        branch_specific_values = [
            branch_specific_gate(row, random_row)
            for row, random_row in matched_pairs
        ]
        drift_minus_random = [
            finite_float(row.get("token_drift_rate")) - finite_float(random_row.get("token_drift_rate"))
            for row, random_row in matched_pairs
        ]
        target_minus_random = [
            finite_float(row.get("mean_target_margin_delta")) - finite_float(random_row.get("mean_target_margin_delta"))
            for row, random_row in matched_pairs
        ]
        kl_minus_random = [
            finite_float(row.get("mean_kl_base_to_steered")) - finite_float(random_row.get("mean_kl_base_to_steered"))
            for row, random_row in matched_pairs
        ]
        out.append(
            {
                "family": family,
                "prompt_id": prompt_id,
                "target_set": target_set,
                "component": component,
                "alpha": alpha,
                "n_rows": len(group),
                "matched_random_rows": len(matched_pairs),
                "branch_gate_rate": rate_true(gate_values),
                "branch_specific_gate_rate": rate_true(branch_specific_values),
                "target_positive_rate": rate_true(math.isfinite(value) and value > 0.0 for value in target_values),
                "drift_ge50_rate": rate_true(math.isfinite(value) and value >= 0.5 for value in drift_values),
                "random_branch_gate_rate": rate_true(random_gate_values),
                "branch_gate_minus_random_rate": mean_finite(gate_minus_random),
                "token_drift_rate_mean": mean_finite(drift_values),
                "token_drift_rate_minus_random_mean": mean_finite(drift_minus_random),
                "token_drift_ge_random_rate": rate_true(
                    math.isfinite(value) and value >= 0.0 for value in drift_minus_random
                ),
                "baseline_token_overlap_mean": mean_finite(row.get("baseline_token_overlap") for row in group),
                "mean_selected_logprob_gain_mean": mean_finite(row.get("mean_selected_logprob_gain") for row in group),
                "mean_kl_base_to_steered_mean": mean_finite(row.get("mean_kl_base_to_steered") for row in group),
                "mean_kl_base_to_steered_minus_random_mean": mean_finite(kl_minus_random),
                "mean_target_margin_delta_mean": mean_finite(target_values),
                "mean_target_margin_delta_minus_random_mean": mean_finite(target_minus_random),
                "target_margin_gt_random_rate": rate_true(
                    math.isfinite(value) and value > 0.0 for value in target_minus_random
                ),
                "mean_nearest_distance_mean": mean_finite(row.get("mean_nearest_distance") for row in group),
            }
        )
    return out


def prompt_layer_k_summary_rows(contrasts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    random_rows: Dict[Tuple[str, str, int, int, int, float], Dict[str, Any]] = {}
    for row in contrasts:
        if str(row.get("component", "")) == RANDOM_COMPONENT:
            random_rows[random_match_key(row)] = row

    groups: Dict[Tuple[str, str, int, int, str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in contrasts:
        groups[
            (
                str(row.get("family", "")),
                str(row.get("prompt_id", "")),
                int(finite_float(row.get("layer", -1))),
                int(finite_float(row.get("k", -1))),
                row_target_set(row),
                str(row.get("component", "")),
                finite_float(row.get("alpha")),
            )
        ].append(row)

    out: List[Dict[str, Any]] = []
    for (family, prompt_id, layer, k, target_set, component, alpha), group in sorted(groups.items()):
        drift_values = [finite_float(row.get("token_drift_rate")) for row in group]
        target_values = [finite_float(row.get("mean_target_margin_delta")) for row in group]
        gate_values = [branch_gate(row) for row in group]
        random_for_group = [random_rows.get(random_match_key(row)) for row in group]
        matched_pairs = [(row, random_row) for row, random_row in zip(group, random_for_group) if random_row is not None]
        random_gate_values = [branch_gate(row) for row in random_for_group if row is not None]
        gate_minus_random = [
            float(branch_gate(row)) - float(branch_gate(random_row))
            for row, random_row in matched_pairs
        ]
        branch_specific_values = [
            branch_specific_gate(row, random_row)
            for row, random_row in matched_pairs
        ]
        drift_minus_random = [
            finite_float(row.get("token_drift_rate")) - finite_float(random_row.get("token_drift_rate"))
            for row, random_row in matched_pairs
        ]
        target_minus_random = [
            finite_float(row.get("mean_target_margin_delta")) - finite_float(random_row.get("mean_target_margin_delta"))
            for row, random_row in matched_pairs
        ]
        out.append(
            {
                "family": family,
                "prompt_id": prompt_id,
                "layer": layer,
                "k": k,
                "target_set": target_set,
                "component": component,
                "alpha": alpha,
                "n_rows": len(group),
                "matched_random_rows": len(matched_pairs),
                "branch_gate_rate": rate_true(gate_values),
                "branch_specific_gate_rate": rate_true(branch_specific_values),
                "target_positive_rate": rate_true(math.isfinite(value) and value > 0.0 for value in target_values),
                "drift_ge50_rate": rate_true(math.isfinite(value) and value >= 0.5 for value in drift_values),
                "random_branch_gate_rate": rate_true(random_gate_values),
                "branch_gate_minus_random_rate": mean_finite(gate_minus_random),
                "token_drift_rate_mean": mean_finite(drift_values),
                "token_drift_rate_minus_random_mean": mean_finite(drift_minus_random),
                "mean_target_margin_delta_mean": mean_finite(target_values),
                "mean_target_margin_delta_minus_random_mean": mean_finite(target_minus_random),
                "mean_nearest_distance_mean": mean_finite(row.get("mean_nearest_distance") for row in group),
            }
        )
    return out


def fmt(value: Any) -> str:
    x = finite_float(value)
    if math.isnan(x):
        return "nan"
    return f"{x:.4g}"


def write_report(
    *,
    contrasts: Sequence[Dict[str, Any]],
    component_summary: Sequence[Dict[str, Any]],
    family_summary: Sequence[Dict[str, Any]],
    layer_summary: Sequence[Dict[str, Any]],
    k_summary: Sequence[Dict[str, Any]],
    prompt_summary: Sequence[Dict[str, Any]],
    prompt_layer_k_summary: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "# HLTD Closed-Loop Summary",
        "",
        "This summary compares each closed-loop branch against the greedy baseline",
        "recorded in the same prompt/layer/k/seed run.",
        "",
        "## Component Summary",
        "",
        "| component | alpha | n | drift | overlap | base logp | gain | KL | target margin | nearest dist |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in component_summary:
        lines.append(
            "| {component} | {alpha} | {n} | {drift} | {overlap} | {base_logp} | {gain} | {kl} | {target} | {dist} |".format(
                component=row["component"],
                alpha=fmt(row["alpha"]),
                n=row["n_rows"],
                drift=fmt(row["token_drift_rate_mean"]),
                overlap=fmt(row["baseline_token_overlap_mean"]),
                base_logp=fmt(row["mean_selected_base_logprob_mean"]),
                gain=fmt(row["mean_selected_logprob_gain_mean"]),
                kl=fmt(row["mean_kl_base_to_steered_mean"]),
                target=fmt(row["mean_target_margin_delta_mean"]),
                dist=fmt(row["mean_nearest_distance_mean"]),
            )
            )

    if k_summary:
        lines.extend(
            [
                "",
                "## k Summary",
                "",
                "| k | component | alpha | n | drift | gain | KL | target margin | nearest dist |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in k_summary:
            lines.append(
                "| {k} | {component} | {alpha} | {n} | {drift} | {gain} | {kl} | {target} | {dist} |".format(
                    k=row["k"],
                    component=row["component"],
                    alpha=fmt(row["alpha"]),
                    n=row["n_rows"],
                    drift=fmt(row["token_drift_rate_mean"]),
                    gain=fmt(row["mean_selected_logprob_gain_mean"]),
                    kl=fmt(row["mean_kl_base_to_steered_mean"]),
                    target=fmt(row["mean_target_margin_delta_mean"]),
                    dist=fmt(row["mean_nearest_distance_mean"]),
                )
            )

    if layer_summary:
        lines.extend(
            [
                "",
                "## Layer Summary",
                "",
                "| layer | component | alpha | n | drift | gain | KL | target margin | nearest dist |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in layer_summary:
            lines.append(
                "| L{layer} | {component} | {alpha} | {n} | {drift} | {gain} | {kl} | {target} | {dist} |".format(
                    layer=row["layer"],
                    component=row["component"],
                    alpha=fmt(row["alpha"]),
                    n=row["n_rows"],
                    drift=fmt(row["token_drift_rate_mean"]),
                    gain=fmt(row["mean_selected_logprob_gain_mean"]),
                    kl=fmt(row["mean_kl_base_to_steered_mean"]),
                    target=fmt(row["mean_target_margin_delta_mean"]),
                    dist=fmt(row["mean_nearest_distance_mean"]),
                )
            )

    if family_summary:
        lines.extend(
            [
                "",
                "## Family Summary",
                "",
                "| family | component | alpha | n | drift | gain | KL | target margin | nearest dist |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in family_summary:
            lines.append(
                "| {family} | {component} | {alpha} | {n} | {drift} | {gain} | {kl} | {target} | {dist} |".format(
                    family=row["family"],
                    component=row["component"],
                    alpha=fmt(row["alpha"]),
                    n=row["n_rows"],
                    drift=fmt(row["token_drift_rate_mean"]),
                    gain=fmt(row["mean_selected_logprob_gain_mean"]),
                    kl=fmt(row["mean_kl_base_to_steered_mean"]),
                    target=fmt(row["mean_target_margin_delta_mean"]),
                    dist=fmt(row["mean_nearest_distance_mean"]),
                )
            )

    if prompt_summary:
        lines.extend(
            [
                "",
                "## Prompt Robustness",
                "",
                "| family | prompt | target set | component | alpha | n | matched random | gate | specific gate | rand gate | gate-rand | drift | target | target-rand |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in prompt_summary:
            lines.append(
                "| {family} | {prompt} | {target_set} | {component} | {alpha} | {n} | {matched} | {gate} | {specific_gate} | {random_gate} | {gate_adv} | {drift} | {target} | {target_adv} |".format(
                    family=row["family"],
                    prompt=row["prompt_id"],
                    target_set=row.get("target_set", ""),
                    component=row["component"],
                    alpha=fmt(row["alpha"]),
                    n=row["n_rows"],
                    matched=row.get("matched_random_rows", 0),
                    gate=fmt(row["branch_gate_rate"]),
                    specific_gate=fmt(row.get("branch_specific_gate_rate")),
                    random_gate=fmt(row["random_branch_gate_rate"]),
                    gate_adv=fmt(row["branch_gate_minus_random_rate"]),
                    drift=fmt(row["token_drift_rate_mean"]),
                    target=fmt(row["mean_target_margin_delta_mean"]),
                    target_adv=fmt(row["mean_target_margin_delta_minus_random_mean"]),
                )
            )

    if prompt_layer_k_summary:
        lines.extend(
            [
                "",
                "## Prompt Layer-k Robustness",
                "",
                "| family | prompt | layer | k | target set | component | alpha | n | matched random | gate | specific gate | rand gate | target-rand |",
                "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in prompt_layer_k_summary:
            lines.append(
                "| {family} | {prompt} | L{layer} | {k} | {target_set} | {component} | {alpha} | {n} | {matched} | {gate} | {specific_gate} | {random_gate} | {target_adv} |".format(
                    family=row["family"],
                    prompt=row["prompt_id"],
                    layer=row["layer"],
                    k=row["k"],
                    target_set=row.get("target_set", ""),
                    component=row["component"],
                    alpha=fmt(row["alpha"]),
                    n=row["n_rows"],
                    matched=row.get("matched_random_rows", 0),
                    gate=fmt(row["branch_gate_rate"]),
                    specific_gate=fmt(row["branch_specific_gate_rate"]),
                    random_gate=fmt(row["random_branch_gate_rate"]),
                    target_adv=fmt(row["mean_target_margin_delta_minus_random_mean"]),
                )
            )

    lines.extend(
        [
            "",
            "## Generated Text Contrasts",
            "",
            "| family | prompt | target set | component | alpha | overlap | baseline | generated |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in contrasts:
        lines.append(
            "| {family} | {prompt_id} | {target_set} | {component} | {alpha} | {overlap} | `{base}` | `{generated}` |".format(
                family=row["family"],
                prompt_id=row["prompt_id"],
                target_set=row.get("target_set", ""),
                component=row["component"],
                alpha=fmt(row["alpha"]),
                overlap=fmt(row["baseline_token_overlap"]),
                base=str(row.get("baseline_generated_text", ""))[:80],
                generated=str(row.get("generated_text", ""))[:80],
            )
        )

    lines.extend(
        [
            "",
            "## Read Guard",
            "",
            "High token drift with very low base logprob should be treated as",
            "fluency collapse. A useful branch should move away from the baseline",
            "while preserving reasonable base-model support and keeping nearest-node",
            "distance bounded.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    output_root = Path(args.output_root) if args.output_root else run_root
    metrics = read_csv(run_root / "closed_loop_metrics.csv")
    contrasts = contrast_rows(metrics)
    component_summary = component_summary_rows(contrasts)
    family_summary = family_summary_rows(contrasts)
    layer_summary = layer_summary_rows(contrasts)
    k_summary = k_summary_rows(contrasts)
    prompt_summary = prompt_summary_rows(contrasts)
    prompt_layer_k_summary = prompt_layer_k_summary_rows(contrasts)

    write_csv(contrasts, output_root / "closed_loop_contrasts.csv")
    write_csv(component_summary, output_root / "closed_loop_component_summary.csv")
    write_csv(family_summary, output_root / "closed_loop_family_summary.csv")
    write_csv(layer_summary, output_root / "closed_loop_layer_summary.csv")
    write_csv(k_summary, output_root / "closed_loop_k_summary.csv")
    write_csv(prompt_summary, output_root / "closed_loop_prompt_summary.csv")
    write_csv(prompt_layer_k_summary, output_root / "closed_loop_prompt_layer_k_summary.csv")
    write_report(
        contrasts=contrasts,
        component_summary=component_summary,
        family_summary=family_summary,
        layer_summary=layer_summary,
        k_summary=k_summary,
        prompt_summary=prompt_summary,
        prompt_layer_k_summary=prompt_layer_k_summary,
        output_path=output_root / "closed_loop_summary_report.md",
    )

    print(f"wrote closed-loop contrasts -> {output_root / 'closed_loop_contrasts.csv'}")
    print(f"wrote closed-loop summary -> {output_root / 'closed_loop_summary_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
