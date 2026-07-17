#!/usr/bin/env python3
"""Join structural HLTD Hodge summaries with steering/probe branch gates."""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


COMPONENTS = [
    "presence",
    "coexact",
    "presence_plus_coexact",
    "coexact_minus_presence",
    "negative_coexact",
]

PROBES = ["identity_stress", "ontology_collapse", "affordance_stress"]

EXPECTED_BRANCH_ROLES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "coexact": ("traversal", ("next_positive",)),
    "coexact_minus_presence": (
        "traversal_without_stabilization",
        ("next_positive", "probe_not_positive"),
    ),
    "presence": ("stabilization", ("next_not_positive", "probe_positive")),
    "presence_plus_coexact": ("combined", ("next_positive", "probe_positive")),
    "negative_coexact": ("reverse_control", ("next_negative",)),
}


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean_or_nan(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return mean(vals) if vals else float("nan")


def weighted_mean_or_nan(rows: Sequence[Dict[str, Any]], value_key: str, weight_key: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = finite_float(row.get(value_key))
        weight = finite_float(row.get(weight_key))
        if value is None or weight is None or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def fmt(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return "nan"
    return f"{number:.{digits}f}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def selected_layers(rows: Iterable[Dict[str, str]], layers: set[int]) -> List[Dict[str, str]]:
    return [row for row in rows if int(float(row.get("layer", "0"))) in layers]


def layer_hodge_rows(
    *,
    hodge_root: Path,
    topology: str,
    k: int,
    layers: set[int],
) -> List[Dict[str, Any]]:
    path = hodge_root / "summary_layer.csv"
    rows = selected_layers(read_csv(path), layers)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("topology")) != topology or int(float(row.get("k", 0))) != int(k):
            continue
        out.append(
            {
                "topology": topology,
                "family": row["family"],
                "k": int(k),
                "layer": int(float(row["layer"])),
                "n_runs": int(float(row.get("n_runs", 0) or 0)),
                "real_exact": finite_float(row.get("real_hltd_exact_ratio")),
                "real_coexact": finite_float(row.get("real_hltd_coexact_ratio")),
                "real_harmonic": finite_float(row.get("real_hltd_harmonic_ratio")),
                "real_semantic_flow": finite_float(row.get("real_hltd_semantic_flow_ratio")),
                "real_minus_shuffle_coexact": finite_float(row.get("real_minus_shuffle_hltd_coexact_ratio")),
                "real_minus_random_coexact": finite_float(row.get("real_minus_random_hltd_coexact_ratio")),
                "real_graph_high_freq": finite_float(row.get("real_graph_high_freq_ratio")),
                "real_hodge_curl": finite_float(row.get("real_hodge_curl_ratio")),
                "same_graph_reverse_coexact_gap": finite_float(
                    row.get("real_hltd_same_graph_reverse_coexact_ratio_gap")
                ),
            }
        )
    return out


def aggregate_layer_hodge(family_layer_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in family_layer_rows:
        buckets[(int(row["k"]), int(row["layer"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (k, layer), rows in sorted(buckets.items()):
        item: Dict[str, Any] = {
            "k": k,
            "layer": layer,
            "n_family_rows": len(rows),
        }
        for metric in [
            "real_exact",
            "real_coexact",
            "real_harmonic",
            "real_semantic_flow",
            "real_minus_shuffle_coexact",
            "real_minus_random_coexact",
            "real_graph_high_freq",
            "real_hodge_curl",
            "same_graph_reverse_coexact_gap",
        ]:
            vals = [value for value in (finite_float(row.get(metric)) for row in rows) if value is not None]
            item[f"{metric}_mean"] = mean_or_nan(vals)
            item[f"{metric}_min"] = min(vals) if vals else float("nan")
            item[f"{metric}_max"] = max(vals) if vals else float("nan")
        out.append(item)
    return out


def family_k_rows(
    *,
    hodge_root: Path,
    topology: str,
    k: int,
) -> List[Dict[str, Any]]:
    rows = read_csv(hodge_root / "summary_family_k.csv")
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("topology")) != topology or int(float(row.get("k", 0))) != int(k):
            continue
        out.append(
            {
                "topology": topology,
                "family": row["family"],
                "k": int(k),
                "real_coexact_l5_l8": finite_float(row.get("real_coexact_l5_l8_mean")),
                "real_exact_l5_l8": finite_float(row.get("real_exact_l5_l8_mean")),
                "real_harmonic_max": finite_float(row.get("real_harmonic_max_mean")),
                "real_minus_shuffle_coexact_l5_l8": finite_float(
                    row.get("real_minus_shuffle_coexact_l5_l8_mean")
                ),
                "real_minus_random_coexact_l5_l8": finite_float(
                    row.get("real_minus_random_coexact_l5_l8_mean")
                ),
                "real_graph_high_freq_l5_l8": finite_float(row.get("real_graph_high_freq_l5_l8_mean")),
                "real_hodge_curl_l5_l8": finite_float(row.get("real_hodge_curl_l5_l8_mean")),
                "max_same_graph_reverse_coexact_gap": finite_float(
                    row.get("max_same_graph_reverse_coexact_gap_mean")
                ),
            }
        )
    return out


def topology_family_k_rows(
    *,
    hodge_root: Path,
    k: int,
    topologies: set[str],
) -> List[Dict[str, Any]]:
    rows = read_csv(hodge_root / "summary_family_k.csv")
    out: List[Dict[str, Any]] = []
    for row in rows:
        if int(float(row.get("k", 0))) != int(k):
            continue
        if str(row.get("topology")) not in topologies:
            continue
        out.append(
            {
                "topology": row["topology"],
                "family": row["family"],
                "k": int(k),
                "real_exact_l5_l8": finite_float(row.get("real_exact_l5_l8_mean")),
                "real_coexact_l5_l8": finite_float(row.get("real_coexact_l5_l8_mean")),
                "real_harmonic_max": finite_float(row.get("real_harmonic_max_mean")),
                "real_minus_shuffle_coexact_l5_l8": finite_float(
                    row.get("real_minus_shuffle_coexact_l5_l8_mean")
                ),
                "real_minus_random_coexact_l5_l8": finite_float(
                    row.get("real_minus_random_coexact_l5_l8_mean")
                ),
                "max_same_graph_reverse_coexact_gap": finite_float(
                    row.get("max_same_graph_reverse_coexact_gap_mean")
                ),
            }
        )
    return out


def best_metric_value(row: Dict[str, Any], metric: str) -> float:
    value = finite_float(row.get(metric))
    return value if value is not None else float("-inf")


def mean_and_best(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    *,
    layer_key: str = "layer",
) -> Dict[str, Any]:
    vals = [value for value in (finite_float(row.get(metric)) for row in rows) if value is not None]
    if not vals:
        return {
            f"{metric}_mean": float("nan"),
            f"{metric}_max": float("nan"),
            f"{metric}_best_layer": "",
        }
    best_row = max(rows, key=lambda row: best_metric_value(row, metric))
    return {
        f"{metric}_mean": mean_or_nan(vals),
        f"{metric}_max": max(vals),
        f"{metric}_best_layer": best_row.get(layer_key, ""),
    }


def family_k_sweep_rows(
    *,
    hodge_root: Path,
    topology: str,
    ks: set[int],
) -> List[Dict[str, Any]]:
    rows = read_csv(hodge_root / "summary_family_k.csv")
    out: List[Dict[str, Any]] = []
    for row in rows:
        row_k = int(float(row.get("k", 0)))
        if str(row.get("topology")) != topology or row_k not in ks:
            continue
        out.append(
            {
                "topology": topology,
                "family": row["family"],
                "k": row_k,
                "real_coexact_l5_l8": finite_float(row.get("real_coexact_l5_l8_mean")),
                "real_exact_l5_l8": finite_float(row.get("real_exact_l5_l8_mean")),
                "real_harmonic_max": finite_float(row.get("real_harmonic_max_mean")),
                "real_minus_shuffle_coexact_l5_l8": finite_float(
                    row.get("real_minus_shuffle_coexact_l5_l8_mean")
                ),
                "real_minus_random_coexact_l5_l8": finite_float(
                    row.get("real_minus_random_coexact_l5_l8_mean")
                ),
                "real_graph_high_freq_l5_l8": finite_float(row.get("real_graph_high_freq_l5_l8_mean")),
                "real_hodge_curl_l5_l8": finite_float(row.get("real_hodge_curl_l5_l8_mean")),
                "max_same_graph_reverse_coexact_gap": finite_float(
                    row.get("max_same_graph_reverse_coexact_gap_mean")
                ),
            }
        )
    return out


def aggregate_k_sweep(family_k_sweep: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in family_k_sweep:
        buckets[int(row["k"])].append(row)

    out: List[Dict[str, Any]] = []
    for k, rows in sorted(buckets.items()):
        item: Dict[str, Any] = {
            "k": k,
            "n_family_rows": len(rows),
        }
        for metric in [
            "real_exact_l5_l8",
            "real_coexact_l5_l8",
            "real_harmonic_max",
            "real_minus_shuffle_coexact_l5_l8",
            "real_minus_random_coexact_l5_l8",
            "real_graph_high_freq_l5_l8",
            "real_hodge_curl_l5_l8",
            "max_same_graph_reverse_coexact_gap",
        ]:
            vals = [value for value in (finite_float(row.get(metric)) for row in rows) if value is not None]
            item[f"{metric}_mean"] = mean_or_nan(vals)
            item[f"{metric}_min"] = min(vals) if vals else float("nan")
            item[f"{metric}_max"] = max(vals) if vals else float("nan")
        out.append(item)
    return out


def steering_layer_rows(
    *,
    steering_root: Path,
    k: int,
    layers: set[int],
    selector: str,
    components: set[str],
) -> List[Dict[str, Any]]:
    return steering_layer_rows_for_ks(
        steering_root=steering_root,
        ks={int(k)},
        layers=layers,
        selector=selector,
        components=components,
    )


def steering_layer_rows_for_ks(
    *,
    steering_root: Path,
    ks: set[int],
    layers: set[int],
    selector: str,
    components: set[str],
) -> List[Dict[str, Any]]:
    return steering_layer_rows_for_ks_and_selectors(
        steering_root=steering_root,
        ks=ks,
        layers=layers,
        selectors={selector},
        components=components,
    )


def steering_layer_rows_for_ks_and_selectors(
    *,
    steering_root: Path,
    ks: set[int],
    layers: set[int],
    selectors: set[str],
    components: set[str],
) -> List[Dict[str, Any]]:
    rows = selected_layers(read_csv(steering_root / "summary_layer_pairwise.csv"), layers)
    out: List[Dict[str, Any]] = []
    for row in rows:
        row_k = int(float(row.get("k", 0)))
        if row_k not in ks:
            continue
        selector = str(row.get("token_selector"))
        if selector not in selectors:
            continue
        if str(row.get("component")) not in components:
            continue
        out.append(
            {
                "k": row_k,
                "layer": int(float(row["layer"])),
                "selector": selector,
                "component": row["component"],
                "n_pairs": int(float(row.get("n_pairs", 0) or 0)),
                "next_token_delta": finite_float(
                    row.get("next_token_logprob_delta_minus_random_tangent_mean")
                ),
                "semantic_margin_delta": finite_float(
                    row.get("semantic_margin_delta_minus_random_tangent_mean")
                ),
                "kl_delta": finite_float(row.get("kl_base_to_steered_minus_random_tangent_mean")),
                "entropy_delta": finite_float(row.get("entropy_delta_minus_random_tangent_mean")),
            }
        )
    return out


def steering_family_rows(
    *,
    steering_root: Path,
    k: int,
    layers: set[int],
    selector: str,
    components: set[str],
) -> List[Dict[str, Any]]:
    rows = selected_layers(read_csv(steering_root / "summary_pairwise.csv"), layers)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if int(float(row.get("k", 0))) != int(k):
            continue
        if str(row.get("token_selector")) != selector:
            continue
        if str(row.get("component")) not in components:
            continue
        out.append(
            {
                "family": row["family"],
                "k": int(k),
                "layer": int(float(row["layer"])),
                "selector": selector,
                "component": row["component"],
                "n_pairs": int(float(row.get("n_pairs", 0) or 0)),
                "next_token_delta": finite_float(
                    row.get("next_token_logprob_delta_minus_random_tangent_mean")
                ),
                "semantic_margin_delta": finite_float(
                    row.get("semantic_margin_delta_minus_random_tangent_mean")
                ),
                "kl_delta": finite_float(row.get("kl_base_to_steered_minus_random_tangent_mean")),
            }
        )
    return out


def probe_layer_rows(
    *,
    probe_root: Path,
    k: int,
    layers: set[int],
    selector: str,
    components: set[str],
    probes: set[str],
) -> List[Dict[str, Any]]:
    return probe_layer_rows_for_ks(
        probe_root=probe_root,
        ks={int(k)},
        layers=layers,
        selector=selector,
        components=components,
        probes=probes,
    )


def probe_layer_rows_for_ks(
    *,
    probe_root: Path,
    ks: set[int],
    layers: set[int],
    selector: str,
    components: set[str],
    probes: set[str],
) -> List[Dict[str, Any]]:
    return probe_layer_rows_for_ks_and_selectors(
        probe_root=probe_root,
        ks=ks,
        layers=layers,
        selectors={selector},
        components=components,
        probes=probes,
    )


def probe_layer_rows_for_ks_and_selectors(
    *,
    probe_root: Path,
    ks: set[int],
    layers: set[int],
    selectors: set[str],
    components: set[str],
    probes: set[str],
) -> List[Dict[str, Any]]:
    rows = selected_layers(read_csv(probe_root / "summary_layer_pairwise.csv"), layers)
    out: List[Dict[str, Any]] = []
    for row in rows:
        row_k = int(float(row.get("k", 0)))
        if row_k not in ks:
            continue
        selector = str(row.get("token_selector"))
        if selector not in selectors:
            continue
        if str(row.get("component")) not in components or str(row.get("probe")) not in probes:
            continue
        out.append(
            {
                "k": row_k,
                "layer": int(float(row["layer"])),
                "selector": selector,
                "component": row["component"],
                "probe": row["probe"],
                "n_pairs": int(float(row.get("n_pairs", 0) or 0)),
                "probe_label_margin_delta": finite_float(
                    row.get("label_margin_delta_minus_random_tangent_mean")
                ),
                "probe_positive_prob_delta": finite_float(
                    row.get("positive_prob_delta_minus_random_tangent_mean")
                ),
                "probe_entropy_delta": finite_float(
                    row.get("probe_entropy_delta_minus_random_tangent_mean")
                ),
            }
        )
    return out


def probe_family_rows(
    *,
    probe_root: Path,
    k: int,
    layers: set[int],
    selector: str,
    components: set[str],
    probes: set[str],
) -> List[Dict[str, Any]]:
    rows = selected_layers(read_csv(probe_root / "summary_pairwise.csv"), layers)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if int(float(row.get("k", 0))) != int(k):
            continue
        if str(row.get("token_selector")) != selector:
            continue
        if str(row.get("component")) not in components or str(row.get("probe")) not in probes:
            continue
        out.append(
            {
                "family": row["family"],
                "k": int(k),
                "layer": int(float(row["layer"])),
                "selector": selector,
                "component": row["component"],
                "probe": row["probe"],
                "n_pairs": int(float(row.get("n_pairs", 0) or 0)),
                "probe_label_margin_delta": finite_float(
                    row.get("label_margin_delta_minus_random_tangent_mean")
                ),
                "probe_positive_prob_delta": finite_float(
                    row.get("positive_prob_delta_minus_random_tangent_mean")
                ),
            }
        )
    return out


def closed_loop_prompt_rows(
    *,
    closed_loop_roots: Sequence[Path],
    components: set[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for root in closed_loop_roots:
        path = root / "closed_loop_prompt_summary.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            component = str(row.get("component", ""))
            if component not in components:
                continue
            out.append(
                {
                    "source": root.name,
                    "family": row.get("family", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "component": component,
                    "alpha": finite_float(row.get("alpha")),
                    "n_rows": int(float(row.get("n_rows", 0) or 0)),
                    "matched_random_rows": int(float(row.get("matched_random_rows", 0) or 0)),
                    "branch_gate_rate": finite_float(row.get("branch_gate_rate")),
                    "branch_specific_gate_rate": finite_float(row.get("branch_specific_gate_rate")),
                    "random_branch_gate_rate": finite_float(row.get("random_branch_gate_rate")),
                    "branch_gate_minus_random_rate": finite_float(row.get("branch_gate_minus_random_rate")),
                    "token_drift_rate_mean": finite_float(row.get("token_drift_rate_mean")),
                    "token_drift_rate_minus_random_mean": finite_float(row.get("token_drift_rate_minus_random_mean")),
                    "mean_target_margin_delta_mean": finite_float(row.get("mean_target_margin_delta_mean")),
                    "mean_target_margin_delta_minus_random_mean": finite_float(
                        row.get("mean_target_margin_delta_minus_random_mean")
                    ),
                    "mean_kl_base_to_steered_mean": finite_float(row.get("mean_kl_base_to_steered_mean")),
                    "mean_kl_base_to_steered_minus_random_mean": finite_float(
                        row.get("mean_kl_base_to_steered_minus_random_mean")
                    ),
                }
            )
    return out


def build_closed_loop_branch_score_rows(
    *,
    closed_loop_rows: Sequence[Dict[str, Any]],
    family_k: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    hodge_by_family = {str(row["family"]): row for row in family_k}
    buckets: Dict[Tuple[str, str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in closed_loop_rows:
        alpha = finite_float(row.get("alpha"))
        buckets[
            (
                str(row.get("source", "")),
                str(row.get("family", "")),
                str(row.get("component", "")),
                alpha if alpha is not None else float("nan"),
            )
        ].append(row)

    out: List[Dict[str, Any]] = []
    for (source, family, component, alpha), rows in sorted(buckets.items()):
        hodge = hodge_by_family.get(family, {})
        prompt_specific_values = [
            value
            for value in (finite_float(row.get("branch_specific_gate_rate")) for row in rows)
            if value is not None
        ]
        target_adv_values = [
            value
            for value in (finite_float(row.get("mean_target_margin_delta_minus_random_mean")) for row in rows)
            if value is not None
        ]
        item: Dict[str, Any] = {
            "source": source,
            "family": family,
            "component": component,
            "alpha": alpha,
            "n_prompt_rows": len(rows),
            "n_rows": sum(int(row.get("n_rows", 0) or 0) for row in rows),
            "matched_random_rows": sum(int(row.get("matched_random_rows", 0) or 0) for row in rows),
            "branch_gate_rate": weighted_mean_or_nan(rows, "branch_gate_rate", "n_rows"),
            "branch_specific_gate_rate": weighted_mean_or_nan(
                rows,
                "branch_specific_gate_rate",
                "matched_random_rows",
            ),
            "branch_specific_prompt_rate": (
                sum(1 for value in prompt_specific_values if value > 0.0) / len(prompt_specific_values)
                if prompt_specific_values
                else float("nan")
            ),
            "random_branch_gate_rate": weighted_mean_or_nan(rows, "random_branch_gate_rate", "matched_random_rows"),
            "branch_gate_minus_random_rate": weighted_mean_or_nan(
                rows,
                "branch_gate_minus_random_rate",
                "matched_random_rows",
            ),
            "token_drift_rate_mean": weighted_mean_or_nan(rows, "token_drift_rate_mean", "n_rows"),
            "token_drift_rate_minus_random_mean": weighted_mean_or_nan(
                rows,
                "token_drift_rate_minus_random_mean",
                "matched_random_rows",
            ),
            "mean_target_margin_delta_mean": weighted_mean_or_nan(
                rows,
                "mean_target_margin_delta_mean",
                "n_rows",
            ),
            "mean_target_margin_delta_minus_random_mean": weighted_mean_or_nan(
                rows,
                "mean_target_margin_delta_minus_random_mean",
                "matched_random_rows",
            ),
            "target_advantage_prompt_rate": (
                sum(1 for value in target_adv_values if value > 0.0) / len(target_adv_values)
                if target_adv_values
                else float("nan")
            ),
            "mean_kl_base_to_steered_mean": weighted_mean_or_nan(
                rows,
                "mean_kl_base_to_steered_mean",
                "n_rows",
            ),
            "mean_kl_base_to_steered_minus_random_mean": weighted_mean_or_nan(
                rows,
                "mean_kl_base_to_steered_minus_random_mean",
                "matched_random_rows",
            ),
            "hodge_coexact_l5_l8": hodge.get("real_coexact_l5_l8", float("nan")),
            "hodge_exact_l5_l8": hodge.get("real_exact_l5_l8", float("nan")),
            "hodge_harmonic_max": hodge.get("real_harmonic_max", float("nan")),
            "hodge_real_minus_shuffle_coexact_l5_l8": hodge.get(
                "real_minus_shuffle_coexact_l5_l8",
                float("nan"),
            ),
            "hodge_real_minus_random_coexact_l5_l8": hodge.get(
                "real_minus_random_coexact_l5_l8",
                float("nan"),
            ),
        }
        out.append(item)
    return out


def join_layer_rows(
    *,
    hodge_rows: Sequence[Dict[str, Any]],
    steering_rows: Sequence[Dict[str, Any]],
    probe_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    hodge_by_layer = {(int(row["k"]), int(row["layer"])): row for row in hodge_rows}
    steering_by_layer = {
        (int(row["k"]), int(row["layer"]), str(row["selector"]), str(row["component"])): row
        for row in steering_rows
    }
    out: List[Dict[str, Any]] = []
    for probe_row in probe_rows:
        key = (
            int(probe_row["k"]),
            int(probe_row["layer"]),
            str(probe_row["selector"]),
            str(probe_row["component"]),
        )
        hodge = hodge_by_layer.get((int(probe_row["k"]), int(probe_row["layer"])), {})
        steering = steering_by_layer.get(key, {})
        out.append(
            {
                "k": probe_row["k"],
                "layer": probe_row["layer"],
                "selector": probe_row["selector"],
                "component": probe_row["component"],
                "probe": probe_row["probe"],
                "hodge_coexact": hodge.get("real_coexact_mean", float("nan")),
                "hodge_exact": hodge.get("real_exact_mean", float("nan")),
                "hodge_harmonic": hodge.get("real_harmonic_mean", float("nan")),
                "hodge_real_minus_shuffle_coexact": hodge.get(
                    "real_minus_shuffle_coexact_mean",
                    float("nan"),
                ),
                "hodge_real_minus_random_coexact": hodge.get(
                    "real_minus_random_coexact_mean",
                    float("nan"),
                ),
                "graph_high_freq": hodge.get("real_graph_high_freq_mean", float("nan")),
                "hodge_curl": hodge.get("real_hodge_curl_mean", float("nan")),
                "next_token_delta": steering.get("next_token_delta", float("nan")),
                "semantic_margin_delta": steering.get("semantic_margin_delta", float("nan")),
                "kl_delta": steering.get("kl_delta", float("nan")),
                "probe_label_margin_delta": probe_row.get("probe_label_margin_delta", float("nan")),
                "probe_positive_prob_delta": probe_row.get("probe_positive_prob_delta", float("nan")),
            }
        )
    return out


def join_family_rows(
    *,
    hodge_rows: Sequence[Dict[str, Any]],
    steering_rows: Sequence[Dict[str, Any]],
    probe_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    hodge_by_family_layer = {
        (str(row["family"]), int(row["k"]), int(row["layer"])): row for row in hodge_rows
    }
    steering_by_family = {
        (
            str(row["family"]),
            int(row["k"]),
            int(row["layer"]),
            str(row["selector"]),
            str(row["component"]),
        ): row
        for row in steering_rows
    }
    out: List[Dict[str, Any]] = []
    for probe_row in probe_rows:
        key = (
            str(probe_row["family"]),
            int(probe_row["k"]),
            int(probe_row["layer"]),
            str(probe_row["selector"]),
            str(probe_row["component"]),
        )
        hodge = hodge_by_family_layer.get((str(probe_row["family"]), int(probe_row["k"]), int(probe_row["layer"])), {})
        steering = steering_by_family.get(key, {})
        out.append(
            {
                "family": probe_row["family"],
                "k": probe_row["k"],
                "layer": probe_row["layer"],
                "selector": probe_row["selector"],
                "component": probe_row["component"],
                "probe": probe_row["probe"],
                "hodge_coexact": hodge.get("real_coexact", float("nan")),
                "hodge_exact": hodge.get("real_exact", float("nan")),
                "hodge_harmonic": hodge.get("real_harmonic", float("nan")),
                "hodge_real_minus_shuffle_coexact": hodge.get(
                    "real_minus_shuffle_coexact",
                    float("nan"),
                ),
                "hodge_real_minus_random_coexact": hodge.get(
                    "real_minus_random_coexact",
                    float("nan"),
                ),
                "next_token_delta": steering.get("next_token_delta", float("nan")),
                "semantic_margin_delta": steering.get("semantic_margin_delta", float("nan")),
                "probe_label_margin_delta": probe_row.get("probe_label_margin_delta", float("nan")),
                "probe_positive_prob_delta": probe_row.get("probe_positive_prob_delta", float("nan")),
            }
        )
    return out


def build_branch_score_rows(joined_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        buckets[(str(row["selector"]), str(row["component"]), str(row["probe"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (selector, component, probe), rows in sorted(buckets.items()):
        item: Dict[str, Any] = {
            "selector": selector,
            "component": component,
            "probe": probe,
            "n_rows": len(rows),
        }
        for metric in [
            "hodge_coexact",
            "hodge_exact",
            "hodge_harmonic",
            "hodge_real_minus_shuffle_coexact",
            "hodge_real_minus_random_coexact",
            "next_token_delta",
            "semantic_margin_delta",
            "probe_label_margin_delta",
        ]:
            item.update(mean_and_best(rows, metric))
        out.append(item)
    return out


def build_causal_k_score_rows(
    *,
    k_sweep: Sequence[Dict[str, Any]],
    steering_rows: Sequence[Dict[str, Any]],
    probe_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    structural_by_k = {int(row["k"]): row for row in k_sweep}

    steering_by_component: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in steering_rows:
        steering_by_component[
            (int(row["k"]), str(row["selector"]), str(row["component"]))
        ].append(row)

    probe_by_component: Dict[Tuple[int, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in probe_rows:
        probe_by_component[
            (int(row["k"]), str(row["selector"]), str(row["component"]), str(row["probe"]))
        ].append(row)

    out: List[Dict[str, Any]] = []
    for (k, selector, component, probe), rows in sorted(probe_by_component.items()):
        structural = structural_by_k.get(k, {})
        steering = steering_by_component.get((k, selector, component), [])
        item: Dict[str, Any] = {
            "k": k,
            "selector": selector,
            "component": component,
            "probe": probe,
            "n_probe_rows": len(rows),
            "n_steering_rows": len(steering),
            "hodge_exact_mean": structural.get("real_exact_l5_l8_mean", float("nan")),
            "hodge_coexact_mean": structural.get("real_coexact_l5_l8_mean", float("nan")),
            "hodge_harmonic_max": structural.get("real_harmonic_max_max", float("nan")),
            "hodge_coexact_minus_shuffle_mean": structural.get(
                "real_minus_shuffle_coexact_l5_l8_mean",
                float("nan"),
            ),
            "hodge_same_graph_reverse_gap_max": structural.get(
                "max_same_graph_reverse_coexact_gap_max",
                float("nan"),
            ),
        }
        for metric in ["next_token_delta", "semantic_margin_delta", "kl_delta", "entropy_delta"]:
            item.update(mean_and_best(steering, metric))
        for metric in ["probe_label_margin_delta", "probe_positive_prob_delta", "probe_entropy_delta"]:
            item.update(mean_and_best(rows, metric))
        out.append(item)
    return out


def subtract_metric(row: Dict[str, Any], baseline: Dict[str, Any], metric: str) -> float:
    value = finite_float(row.get(metric))
    base_value = finite_float(baseline.get(metric))
    if value is None or base_value is None:
        return float("nan")
    return value - base_value


def build_selector_delta_rows(
    causal_k_scores: Sequence[Dict[str, Any]],
    *,
    baseline_selector: str,
) -> List[Dict[str, Any]]:
    by_key = {
        (int(row["k"]), str(row["selector"]), str(row["component"]), str(row["probe"])): row
        for row in causal_k_scores
    }
    out: List[Dict[str, Any]] = []
    for row in causal_k_scores:
        selector = str(row["selector"])
        if selector == baseline_selector:
            continue
        key = (int(row["k"]), baseline_selector, str(row["component"]), str(row["probe"]))
        baseline = by_key.get(key)
        if baseline is None:
            continue
        item: Dict[str, Any] = {
            "k": row["k"],
            "component": row["component"],
            "probe": row["probe"],
            "baseline_selector": baseline_selector,
            "compare_selector": selector,
            "baseline_next_token_delta_mean": baseline.get("next_token_delta_mean", float("nan")),
            "compare_next_token_delta_mean": row.get("next_token_delta_mean", float("nan")),
            "next_token_delta_mean_diff": subtract_metric(row, baseline, "next_token_delta_mean"),
            "baseline_semantic_margin_delta_mean": baseline.get("semantic_margin_delta_mean", float("nan")),
            "compare_semantic_margin_delta_mean": row.get("semantic_margin_delta_mean", float("nan")),
            "semantic_margin_delta_mean_diff": subtract_metric(row, baseline, "semantic_margin_delta_mean"),
            "baseline_probe_label_margin_delta_mean": baseline.get(
                "probe_label_margin_delta_mean",
                float("nan"),
            ),
            "compare_probe_label_margin_delta_mean": row.get(
                "probe_label_margin_delta_mean",
                float("nan"),
            ),
            "probe_label_margin_delta_mean_diff": subtract_metric(
                row,
                baseline,
                "probe_label_margin_delta_mean",
            ),
        }
        out.append(item)
    return sorted(
        out,
        key=lambda item: (
            int(item["k"]),
            str(item["compare_selector"]),
            str(item["component"]),
            str(item["probe"]),
        ),
    )


def classify_branch_role(
    *,
    next_token_delta: float,
    probe_label_margin_delta: float,
    closed_loop_gate: float,
    closed_loop_threshold: float = 0.25,
) -> str:
    has_next = math.isfinite(next_token_delta) and next_token_delta > 0.0
    has_probe = math.isfinite(probe_label_margin_delta) and probe_label_margin_delta > 0.0
    has_closed_loop = math.isfinite(closed_loop_gate) and closed_loop_gate >= closed_loop_threshold
    if has_next and has_probe and has_closed_loop:
        return "combined_closed_loop"
    if has_next and has_probe:
        return "combined"
    if has_next and has_closed_loop:
        return "closed_loop_traversal"
    if has_next:
        return "traversal"
    if has_probe:
        return "stabilization"
    if has_closed_loop:
        return "closed_loop_specific"
    return "weak_or_control"


def branch_role_criterion_passed(
    criterion: str,
    *,
    next_token_delta: float,
    probe_label_margin_delta: float,
) -> bool:
    if criterion == "next_positive":
        return math.isfinite(next_token_delta) and next_token_delta > 0.0
    if criterion == "next_negative":
        return math.isfinite(next_token_delta) and next_token_delta < 0.0
    if criterion == "next_not_positive":
        return math.isfinite(next_token_delta) and next_token_delta <= 0.0
    if criterion == "probe_positive":
        return math.isfinite(probe_label_margin_delta) and probe_label_margin_delta > 0.0
    if criterion == "probe_not_positive":
        return math.isfinite(probe_label_margin_delta) and probe_label_margin_delta <= 0.0
    raise ValueError(f"Unknown branch role criterion: {criterion}")


def best_layer_for_metric(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    *,
    maximize: bool,
) -> Any:
    candidates: List[Tuple[float, Any]] = []
    for row in rows:
        value = finite_float(row.get(metric))
        if value is None:
            continue
        candidates.append((value, row.get("layer")))
    if not candidates:
        return ""
    value, layer = max(candidates, key=lambda item: item[0]) if maximize else min(candidates, key=lambda item: item[0])
    return layer


def observed_branch_role(*, next_token_delta: float, probe_label_margin_delta: float) -> str:
    has_next = math.isfinite(next_token_delta) and next_token_delta > 0.0
    has_reverse = math.isfinite(next_token_delta) and next_token_delta < 0.0
    has_probe = math.isfinite(probe_label_margin_delta) and probe_label_margin_delta > 0.0
    if has_next and has_probe:
        return "combined"
    if has_next:
        return "traversal"
    if has_reverse and has_probe:
        return "reverse_stabilization"
    if has_reverse:
        return "reverse_control"
    if has_probe:
        return "stabilization"
    return "weak_or_flat"


def build_branch_role_diagnostic_rows(
    joined_family: Sequence[Dict[str, Any]],
    *,
    selector: str,
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in joined_family:
        if str(row.get("selector", "")) != selector:
            continue
        key = (str(row.get("family", "")), str(row.get("probe", "")), str(row.get("component", "")))
        buckets[key].append(row)

    out: List[Dict[str, Any]] = []
    for (family, probe, component), rows in sorted(buckets.items()):
        expected_role, criteria = EXPECTED_BRANCH_ROLES.get(component, ("unclassified", tuple()))
        next_delta = mean_or_nan(
            [
                value
                for value in (finite_float(row.get("next_token_delta")) for row in rows)
                if value is not None
            ]
        )
        probe_margin = mean_or_nan(
            [
                value
                for value in (finite_float(row.get("probe_label_margin_delta")) for row in rows)
                if value is not None
            ]
        )
        passed = [
            criterion
            for criterion in criteria
            if branch_role_criterion_passed(
                criterion,
                next_token_delta=next_delta,
                probe_label_margin_delta=probe_margin,
            )
        ]
        failed = [criterion for criterion in criteria if criterion not in set(passed)]
        score = float(len(passed) / len(criteria)) if criteria else float("nan")
        out.append(
            {
                "selector": selector,
                "family": family,
                "probe": probe,
                "component": component,
                "expected_role": expected_role,
                "observed_role": observed_branch_role(
                    next_token_delta=next_delta,
                    probe_label_margin_delta=probe_margin,
                ),
                "role_score": score,
                "role_pass": int(len(failed) == 0) if criteria else "",
                "criteria_passed": " ".join(passed),
                "criteria_failed": " ".join(failed),
                "n_layer_rows": len(rows),
                "layer_values": " ".join(str(int(float(row["layer"]))) for row in sorted(rows, key=lambda item: int(float(item["layer"])))),
                "hodge_coexact_mean": mean_or_nan(
                    [
                        value
                        for value in (finite_float(row.get("hodge_coexact")) for row in rows)
                        if value is not None
                    ]
                ),
                "hodge_exact_mean": mean_or_nan(
                    [value for value in (finite_float(row.get("hodge_exact")) for row in rows) if value is not None]
                ),
                "hodge_real_minus_shuffle_coexact_mean": mean_or_nan(
                    [
                        value
                        for value in (
                            finite_float(row.get("hodge_real_minus_shuffle_coexact")) for row in rows
                        )
                        if value is not None
                    ]
                ),
                "next_token_delta_mean": next_delta,
                "next_token_delta_max": max(
                    [value for value in (finite_float(row.get("next_token_delta")) for row in rows) if value is not None],
                    default=float("nan"),
                ),
                "next_token_delta_min": min(
                    [value for value in (finite_float(row.get("next_token_delta")) for row in rows) if value is not None],
                    default=float("nan"),
                ),
                "next_token_delta_best_layer": best_layer_for_metric(
                    rows,
                    "next_token_delta",
                    maximize=True,
                ),
                "next_token_delta_min_layer": best_layer_for_metric(
                    rows,
                    "next_token_delta",
                    maximize=False,
                ),
                "semantic_margin_delta_mean": mean_or_nan(
                    [
                        value
                        for value in (finite_float(row.get("semantic_margin_delta")) for row in rows)
                        if value is not None
                    ]
                ),
                "probe_label_margin_delta_mean": probe_margin,
                "probe_label_margin_delta_max": max(
                    [
                        value
                        for value in (finite_float(row.get("probe_label_margin_delta")) for row in rows)
                        if value is not None
                    ],
                    default=float("nan"),
                ),
                "probe_label_margin_delta_min": min(
                    [
                        value
                        for value in (finite_float(row.get("probe_label_margin_delta")) for row in rows)
                        if value is not None
                    ],
                    default=float("nan"),
                ),
                "probe_label_margin_delta_best_layer": best_layer_for_metric(
                    rows,
                    "probe_label_margin_delta",
                    maximize=True,
                ),
            }
        )
    return out


def branch_role_score_for_values(
    *,
    component: str,
    next_token_delta: float,
    probe_label_margin_delta: float,
) -> Tuple[str, Tuple[str, ...], List[str], List[str], float, int]:
    expected_role, criteria = EXPECTED_BRANCH_ROLES.get(component, ("unclassified", tuple()))
    passed = [
        criterion
        for criterion in criteria
        if branch_role_criterion_passed(
            criterion,
            next_token_delta=next_token_delta,
            probe_label_margin_delta=probe_label_margin_delta,
        )
    ]
    failed = [criterion for criterion in criteria if criterion not in set(passed)]
    score = float(len(passed) / len(criteria)) if criteria else float("nan")
    role_pass = int(len(failed) == 0) if criteria else 0
    return expected_role, criteria, passed, failed, score, role_pass


def build_branch_layer_condition_rows(
    joined_family: Sequence[Dict[str, Any]],
    *,
    selector: str,
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in joined_family:
        if str(row.get("selector", "")) != selector:
            continue
        layer = int(float(row.get("layer", 0)))
        buckets[(str(row.get("family", "")), str(row.get("component", "")), layer)].append(row)

    out: List[Dict[str, Any]] = []
    for (family, component, layer), rows in sorted(buckets.items()):
        expected_role, _ = EXPECTED_BRANCH_ROLES.get(component, ("unclassified", tuple()))
        pass_count = 0
        scores: List[float] = []
        failed_criteria: List[str] = []
        observed_roles: List[str] = []
        for row in rows:
            next_delta = finite_float(row.get("next_token_delta"))
            probe_margin = finite_float(row.get("probe_label_margin_delta"))
            if next_delta is None or probe_margin is None:
                continue
            _, _, _, failed, score, role_pass = branch_role_score_for_values(
                component=component,
                next_token_delta=next_delta,
                probe_label_margin_delta=probe_margin,
            )
            pass_count += int(role_pass)
            scores.append(score)
            failed_criteria.extend(failed)
            observed_roles.append(
                observed_branch_role(
                    next_token_delta=next_delta,
                    probe_label_margin_delta=probe_margin,
                )
            )
        n_cells = len(scores)
        pass_rate = float(pass_count / n_cells) if n_cells else float("nan")
        mean_score = mean_or_nan(scores)
        out.append(
            {
                "selector": selector,
                "family": family,
                "component": component,
                "layer": layer,
                "expected_role": expected_role,
                "condition_label": branch_condition_label(
                    pass_rate=pass_rate,
                    mean_role_score=mean_score,
                ),
                "n_probe_cells": n_cells,
                "role_pass_count": pass_count,
                "role_pass_rate": pass_rate,
                "mean_role_score": mean_score,
                "observed_role_counts": counted_labels(observed_roles),
                "failed_criteria_counts": counted_labels(failed_criteria),
                "mean_next_token_delta": mean_or_nan(
                    [
                        value
                        for value in (finite_float(row.get("next_token_delta")) for row in rows)
                        if value is not None
                    ]
                ),
                "mean_probe_label_margin_delta": mean_or_nan(
                    [
                        value
                        for value in (finite_float(row.get("probe_label_margin_delta")) for row in rows)
                        if value is not None
                    ]
                ),
                "mean_hodge_coexact": mean_or_nan(
                    [
                        value
                        for value in (finite_float(row.get("hodge_coexact")) for row in rows)
                        if value is not None
                    ]
                ),
                "mean_hodge_exact": mean_or_nan(
                    [value for value in (finite_float(row.get("hodge_exact")) for row in rows) if value is not None]
                ),
            }
        )
    return out


def counted_labels(values: Sequence[str]) -> str:
    counts = Counter(value for value in values if value)
    return " ".join(f"{label}:{count}" for label, count in sorted(counts.items()))


def branch_condition_label(*, pass_rate: float, mean_role_score: float) -> str:
    if math.isfinite(pass_rate) and pass_rate >= 1.0:
        return "stable_expected"
    if math.isfinite(pass_rate) and pass_rate >= 2.0 / 3.0:
        return "mostly_expected"
    if math.isfinite(pass_rate) and pass_rate > 0.0:
        return "mixed_condition"
    if math.isfinite(mean_role_score) and mean_role_score > 0.0:
        return "systematic_partial_break"
    return "systematic_break"


def format_layer_spans(layers: Sequence[int]) -> str:
    ordered = sorted({int(layer) for layer in layers})
    if not ordered:
        return ""
    spans: List[Tuple[int, int]] = []
    start = previous = ordered[0]
    for layer in ordered[1:]:
        if layer == previous + 1:
            previous = layer
            continue
        spans.append((start, previous))
        start = previous = layer
    spans.append((start, previous))
    return " ".join(f"L{a}" if a == b else f"L{a}-L{b}" for a, b in spans)


def longest_layer_run(layers: Sequence[int]) -> Tuple[int, str]:
    ordered = sorted({int(layer) for layer in layers})
    if not ordered:
        return 0, ""
    best: Tuple[int, int] = (ordered[0], ordered[0])
    start = previous = ordered[0]
    for layer in ordered[1:]:
        if layer == previous + 1:
            previous = layer
            continue
        if previous - start > best[1] - best[0]:
            best = (start, previous)
        start = previous = layer
    if previous - start > best[1] - best[0]:
        best = (start, previous)
    length = best[1] - best[0] + 1
    span = f"L{best[0]}" if best[0] == best[1] else f"L{best[0]}-L{best[1]}"
    return length, span


def layer_transition_label(*, stable_count: int, longest_stable_run: int, n_layers: int) -> str:
    if n_layers <= 0:
        return "no_layers"
    if stable_count == n_layers:
        return "all_layer_stable"
    if stable_count == 0:
        return "no_stable_layers"
    if longest_stable_run >= 3:
        return "stable_band"
    if stable_count >= 3:
        return "fragmented_stability"
    return "sparse_stability"


def clipped_unit(value: Any) -> float:
    number = finite_float(value)
    if number is None:
        return 0.0
    return max(0.0, min(1.0, number))


def branch_band_candidate_label(
    *,
    stable_layer_rate: float,
    closed_loop_gate: float,
    closed_loop_target_advantage: float,
    has_layer_support: bool,
) -> str:
    stable = clipped_unit(stable_layer_rate)
    gate = clipped_unit(closed_loop_gate)
    target = finite_float(closed_loop_target_advantage)
    target = target if target is not None else float("nan")
    if stable >= 0.8 and gate >= 0.25 and (not math.isfinite(target) or target >= 0.0):
        return "causal_band_ready"
    if stable >= 0.8:
        return "structural_band_ready"
    if gate >= 0.25 and has_layer_support:
        return "causal_exception_band"
    if has_layer_support:
        return "narrow_layer_probe"
    return "deprioritize_or_control"


def recommended_layer_text(row: Dict[str, Any]) -> str:
    stable = str(row.get("stable_layers", "") or "").strip()
    if stable and stable.lower() != "nan":
        return stable
    mixed = str(row.get("mostly_or_mixed_layers", "") or "").strip()
    if mixed and mixed.lower() != "nan":
        return mixed
    return "control"


def build_branch_layer_transition_rows(
    branch_layer_conditions: Sequence[Dict[str, Any]],
    *,
    selector: str,
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in branch_layer_conditions:
        if str(row.get("selector", "")) != selector:
            continue
        buckets[(str(row.get("family", "")), str(row.get("component", "")))].append(row)

    out: List[Dict[str, Any]] = []
    for (family, component), rows in sorted(buckets.items()):
        rows = sorted(rows, key=lambda row: int(float(row.get("layer", 0))))
        layers = [int(float(row.get("layer", 0))) for row in rows]
        stable_layers = [
            int(float(row.get("layer", 0)))
            for row in rows
            if (finite_float(row.get("role_pass_rate")) or 0.0) >= 1.0
        ]
        mostly_layers = [
            int(float(row.get("layer", 0)))
            for row in rows
            if 0.0 < (finite_float(row.get("role_pass_rate")) or 0.0) < 1.0
        ]
        break_layers = [
            int(float(row.get("layer", 0)))
            for row in rows
            if (finite_float(row.get("role_pass_rate")) or 0.0) <= 0.0
        ]
        longest_len, longest_span = longest_layer_run(stable_layers)
        pass_rates = [
            value for value in (finite_float(row.get("role_pass_rate")) for row in rows) if value is not None
        ]
        scores = [value for value in (finite_float(row.get("mean_role_score")) for row in rows) if value is not None]
        failed_criteria: List[str] = []
        for row in rows:
            failed_criteria.extend(str(row.get("failed_criteria_counts", "")).replace(":", " ").split()[::2])
        expected_role, _ = EXPECTED_BRANCH_ROLES.get(component, ("unclassified", tuple()))
        out.append(
            {
                "selector": selector,
                "family": family,
                "component": component,
                "expected_role": expected_role,
                "transition_label": layer_transition_label(
                    stable_count=len(stable_layers),
                    longest_stable_run=longest_len,
                    n_layers=len(layers),
                ),
                "n_layers": len(layers),
                "stable_layer_count": len(stable_layers),
                "stable_layer_rate": float(len(stable_layers) / len(layers)) if layers else float("nan"),
                "stable_layers": format_layer_spans(stable_layers),
                "mostly_or_mixed_layers": format_layer_spans(mostly_layers),
                "break_layers": format_layer_spans(break_layers),
                "first_stable_layer": f"L{min(stable_layers)}" if stable_layers else "",
                "last_stable_layer": f"L{max(stable_layers)}" if stable_layers else "",
                "longest_stable_run": longest_len,
                "longest_stable_span": longest_span,
                "mean_layer_pass_rate": mean_or_nan(pass_rates),
                "mean_layer_role_score": mean_or_nan(scores),
                "failed_criteria_counts": counted_labels(failed_criteria),
            }
        )
    return out


def build_branch_band_candidate_rows(
    *,
    branch_layer_transition_summary: Sequence[Dict[str, Any]],
    closed_loop_branch_scores: Sequence[Dict[str, Any]],
    selector: str,
) -> List[Dict[str, Any]]:
    closed_by_family_component: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in closed_loop_branch_scores:
        family = str(row.get("family", ""))
        component = str(row.get("component", ""))
        closed_by_family_component[(family, component)].append(row)

    out: List[Dict[str, Any]] = []
    for row in branch_layer_transition_summary:
        if str(row.get("selector", "")) != selector:
            continue
        family = str(row.get("family", ""))
        component = str(row.get("component", ""))
        closed_rows = closed_by_family_component.get((family, component), [])
        closed_gate = weighted_mean_or_nan(closed_rows, "branch_specific_gate_rate", "matched_random_rows")
        closed_target = weighted_mean_or_nan(
            closed_rows,
            "mean_target_margin_delta_minus_random_mean",
            "matched_random_rows",
        )
        closed_branch_gate = weighted_mean_or_nan(closed_rows, "branch_gate_rate", "matched_random_rows")
        closed_drift = weighted_mean_or_nan(closed_rows, "token_drift_rate_mean", "matched_random_rows")
        matched_random_rows = int(
            sum(
                int(weight)
                for weight in (
                    finite_float(closed.get("matched_random_rows"))
                    for closed in closed_rows
                )
                if weight is not None and weight > 0
            )
        )
        stable_rate = clipped_unit(row.get("stable_layer_rate"))
        role_support = clipped_unit(row.get("mean_layer_role_score"))
        gate_support = clipped_unit(closed_gate)
        target_support = clipped_unit(closed_target)
        structural_support = 0.6 * stable_rate + 0.4 * role_support
        causal_support = 0.75 * gate_support + 0.25 * target_support
        priority_score = 0.45 * stable_rate + 0.30 * role_support + 0.20 * gate_support + 0.05 * target_support
        recommended_layers = recommended_layer_text(row)
        has_layer_support = recommended_layers != "control"
        out.append(
            {
                "selector": selector,
                "family": family,
                "component": component,
                "expected_role": row.get("expected_role", ""),
                "candidate_label": branch_band_candidate_label(
                    stable_layer_rate=stable_rate,
                    closed_loop_gate=closed_gate,
                    closed_loop_target_advantage=closed_target,
                    has_layer_support=has_layer_support,
                ),
                "priority_score": priority_score,
                "structural_support": structural_support,
                "causal_support": causal_support,
                "recommended_layers": recommended_layers,
                "transition_label": row.get("transition_label", ""),
                "stable_layer_rate": stable_rate,
                "mean_layer_pass_rate": row.get("mean_layer_pass_rate", float("nan")),
                "mean_layer_role_score": row.get("mean_layer_role_score", float("nan")),
                "stable_layers": row.get("stable_layers", ""),
                "mostly_or_mixed_layers": row.get("mostly_or_mixed_layers", ""),
                "break_layers": row.get("break_layers", ""),
                "longest_stable_span": row.get("longest_stable_span", ""),
                "closed_loop_sources": len(closed_rows),
                "closed_loop_matched_random_rows": matched_random_rows,
                "closed_loop_branch_specific_gate_rate_mean": closed_gate,
                "closed_loop_branch_gate_rate_mean": closed_branch_gate,
                "closed_loop_token_drift_rate_mean": closed_drift,
                "closed_loop_target_margin_delta_minus_random_mean": closed_target,
            }
        )
    out.sort(
        key=lambda item: (
            -float(item["priority_score"]),
            str(item["family"]),
            str(item["component"]),
        )
    )
    return out


def build_branch_condition_summary_rows(
    branch_role_diagnostics: Sequence[Dict[str, Any]],
    *,
    selector: str,
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in branch_role_diagnostics:
        if str(row.get("selector", "")) != selector:
            continue
        buckets[(str(row.get("family", "")), str(row.get("component", "")))].append(row)

    out: List[Dict[str, Any]] = []
    for (family, component), rows in sorted(buckets.items()):
        expected_role, _ = EXPECTED_BRANCH_ROLES.get(component, ("unclassified", tuple()))
        pass_values = [
            value
            for value in (finite_float(row.get("role_pass")) for row in rows)
            if value is not None
        ]
        score_values = [
            value
            for value in (finite_float(row.get("role_score")) for row in rows)
            if value is not None
        ]
        pass_count = int(sum(1 for value in pass_values if value >= 1.0))
        pass_rate = float(pass_count / len(pass_values)) if pass_values else float("nan")
        mean_score = mean_or_nan(score_values)
        failed_criteria: List[str] = []
        for row in rows:
            failed_criteria.extend(str(row.get("criteria_failed", "")).split())
        observed_roles = [str(row.get("observed_role", "")) for row in rows]
        out.append(
            {
                "selector": selector,
                "family": family,
                "component": component,
                "expected_role": expected_role,
                "condition_label": branch_condition_label(
                    pass_rate=pass_rate,
                    mean_role_score=mean_score,
                ),
                "n_probe_cells": len(rows),
                "role_pass_count": pass_count,
                "role_pass_rate": pass_rate,
                "mean_role_score": mean_score,
                "observed_role_counts": counted_labels(observed_roles),
                "failed_criteria_counts": counted_labels(failed_criteria),
                "mean_next_token_delta": mean_or_nan(
                    [
                        value
                        for value in (finite_float(row.get("next_token_delta_mean")) for row in rows)
                        if value is not None
                    ]
                ),
                "mean_probe_label_margin_delta": mean_or_nan(
                    [
                        value
                        for value in (finite_float(row.get("probe_label_margin_delta_mean")) for row in rows)
                        if value is not None
                    ]
                ),
                "mean_hodge_coexact": mean_or_nan(
                    [
                        value
                        for value in (finite_float(row.get("hodge_coexact_mean")) for row in rows)
                        if value is not None
                    ]
                ),
            }
        )
    return out


def build_branch_role_summary_rows(
    *,
    causal_k_scores: Sequence[Dict[str, Any]],
    closed_loop_branch_scores: Sequence[Dict[str, Any]],
    selector: str,
) -> List[Dict[str, Any]]:
    closed_by_family_component: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in closed_loop_branch_scores:
        closed_by_family_component[(str(row.get("family", "")), str(row.get("component", "")))].append(row)

    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in causal_k_scores:
        if str(row.get("selector", "")) != selector:
            continue
        buckets[(selector, str(row.get("component", "")), str(row.get("probe", "")))].append(row)

    out: List[Dict[str, Any]] = []
    for (selector_name, component, probe), rows in sorted(buckets.items()):
        closed = closed_by_family_component.get((probe, component), [])
        closed_gate = weighted_mean_or_nan(closed, "branch_specific_gate_rate", "matched_random_rows")
        closed_target = weighted_mean_or_nan(
            closed,
            "mean_target_margin_delta_minus_random_mean",
            "matched_random_rows",
        )
        closed_gate_values = [
            value
            for value in (finite_float(row.get("branch_specific_gate_rate")) for row in closed)
            if value is not None
        ]
        closed_target_values = [
            value
            for value in (finite_float(row.get("mean_target_margin_delta_minus_random_mean")) for row in closed)
            if value is not None
        ]
        next_delta = mean_or_nan(
            [value for value in (finite_float(row.get("next_token_delta_mean")) for row in rows) if value is not None]
        )
        probe_margin = mean_or_nan(
            [
                value
                for value in (finite_float(row.get("probe_label_margin_delta_mean")) for row in rows)
                if value is not None
            ]
        )
        item: Dict[str, Any] = {
            "selector": selector_name,
            "component": component,
            "probe": probe,
            "n_k_rows": len(rows),
            "k_values": " ".join(str(int(row["k"])) for row in sorted(rows, key=lambda item: int(item["k"]))),
            "hodge_coexact_mean": mean_or_nan(
                [
                    value
                    for value in (finite_float(row.get("hodge_coexact_mean")) for row in rows)
                    if value is not None
                ]
            ),
            "hodge_exact_mean": mean_or_nan(
                [value for value in (finite_float(row.get("hodge_exact_mean")) for row in rows) if value is not None]
            ),
            "hodge_harmonic_max": mean_or_nan(
                [value for value in (finite_float(row.get("hodge_harmonic_max")) for row in rows) if value is not None]
            ),
            "next_token_delta_mean": next_delta,
            "semantic_margin_delta_mean": mean_or_nan(
                [
                    value
                    for value in (finite_float(row.get("semantic_margin_delta_mean")) for row in rows)
                    if value is not None
                ]
            ),
            "probe_label_margin_delta_mean": probe_margin,
            "probe_positive_prob_delta_mean": mean_or_nan(
                [
                    value
                    for value in (finite_float(row.get("probe_positive_prob_delta_mean")) for row in rows)
                    if value is not None
                ]
            ),
            "closed_loop_sources": len(closed),
            "closed_loop_branch_specific_gate_rate_mean": closed_gate,
            "closed_loop_branch_specific_gate_rate_max": max(closed_gate_values) if closed_gate_values else float("nan"),
            "closed_loop_target_margin_delta_minus_random_mean": closed_target,
            "closed_loop_target_margin_delta_minus_random_max": (
                max(closed_target_values) if closed_target_values else float("nan")
            ),
        }
        item["role_label"] = classify_branch_role(
            next_token_delta=next_delta,
            probe_label_margin_delta=probe_margin,
            closed_loop_gate=closed_gate,
        )
        out.append(item)
    return out


def reverse_specificity_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_csv(path):
            item: Dict[str, Any] = dict(row)
            item["reverse_specificity_source"] = str(path)
            out.append(item)
    return out


def write_report(
    *,
    path: Path,
    family_k: Sequence[Dict[str, Any]],
    k_sweep: Sequence[Dict[str, Any]],
    topology_family_k: Sequence[Dict[str, Any]],
    hodge_layer: Sequence[Dict[str, Any]],
    joined_layer: Sequence[Dict[str, Any]],
    branch_scores: Sequence[Dict[str, Any]],
    causal_k_scores: Sequence[Dict[str, Any]],
    selector_deltas: Sequence[Dict[str, Any]],
    closed_loop_branch_scores: Sequence[Dict[str, Any]],
    branch_role_summary: Sequence[Dict[str, Any]],
    branch_role_diagnostics: Sequence[Dict[str, Any]],
    branch_layer_condition_summary: Sequence[Dict[str, Any]],
    branch_layer_transition_summary: Sequence[Dict[str, Any]],
    branch_condition_summary: Sequence[Dict[str, Any]],
    branch_band_candidate_scoreboard: Sequence[Dict[str, Any]],
    reverse_specificity: Sequence[Dict[str, Any]],
    selector: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# HLTD Branch Hodge Summary",
        "",
        "This report reconnects structural graph-Hodge evidence with the later",
        "one-step steering and learned-probe branch gates.",
        "",
        "## Family Hodge Branches",
        "",
    ]
    family_rows = [
        [
            row["family"],
            fmt(row.get("real_exact_l5_l8")),
            fmt(row.get("real_coexact_l5_l8")),
            fmt(row.get("real_harmonic_max")),
            fmt(row.get("real_minus_shuffle_coexact_l5_l8")),
            fmt(row.get("real_minus_random_coexact_l5_l8")),
        ]
        for row in family_k
    ]
    lines.append(
        markdown_table(
            [
                "family",
                "presence/exact L5-L8",
                "coexact L5-L8",
                "harmonic max",
                "coexact - shuffle",
                "coexact - random",
            ],
            family_rows,
        )
    )
    if k_sweep:
        lines.extend(["", "## Structural k-Sweep Branches", ""])
        k_rows = [
            [
                f"k={row['k']}",
                fmt(row.get("real_exact_l5_l8_mean")),
                fmt(row.get("real_coexact_l5_l8_mean")),
                fmt(row.get("real_harmonic_max_max")),
                fmt(row.get("real_minus_shuffle_coexact_l5_l8_mean")),
                fmt(row.get("real_minus_random_coexact_l5_l8_mean")),
                fmt(row.get("max_same_graph_reverse_coexact_gap_max")),
            ]
            for row in k_sweep
        ]
        lines.append(
            markdown_table(
                [
                    "k",
                    "presence/exact mean",
                    "coexact mean",
                    "harmonic max",
                    "coexact - shuffle mean",
                    "coexact - random mean",
                    "same-graph reverse gap max",
                ],
                k_rows,
            )
        )
        lines.extend(
            [
                "",
                "The structural branch read is expected to survive k-neighbor",
                "changes before it is treated as a stable Hodge signal. This table",
                "tracks the exact/presence, coexact, harmonic, and reversal branches",
                "across the available structural k-sweep.",
                "",
            ]
        )
    topology_rows = [
        [
            row["topology"],
            row["family"],
            fmt(row.get("real_exact_l5_l8")),
            fmt(row.get("real_coexact_l5_l8")),
            fmt(row.get("real_harmonic_max")),
            fmt(row.get("real_minus_shuffle_coexact_l5_l8")),
        ]
        for row in topology_family_k
    ]
    if topology_rows:
        lines.extend(["", "## Topology Branch Contrast", ""])
        lines.append(
            markdown_table(
                ["topology", "family", "presence/exact", "coexact", "harmonic max", "coexact-shuffle"],
                topology_rows,
            )
        )
        lines.extend(
            [
                "",
                "With triangles, non-exact energy is resolved into the coexact branch.",
                "Without triangles, the coexact branch is structurally unavailable and",
                "the residual moves into the harmonic column. This keeps harmonic/global",
                "concept-ring claims separate from the local-swirl coexact branch.",
                "",
            ]
        )
    lines.extend(["", "## Structural Layer Spine", ""])
    spine_rows = [
        [
            f"L{row['layer']}",
            fmt(row.get("real_exact_mean")),
            fmt(row.get("real_coexact_mean")),
            fmt(row.get("real_harmonic_mean")),
            fmt(row.get("real_minus_shuffle_coexact_mean")),
            fmt(row.get("real_minus_random_coexact_mean")),
        ]
        for row in hodge_layer
    ]
    lines.append(
        markdown_table(
            ["layer", "presence/exact", "coexact", "harmonic", "coexact-shuffle", "coexact-random"],
            spine_rows,
        )
    )
    lines.extend(["", f"## Causal Branch Join (`{selector}` selector)", ""])
    join_rows = [
        [
            f"L{row['layer']}",
            row["component"],
            row["probe"],
            fmt(row.get("hodge_coexact")),
            fmt(row.get("next_token_delta")),
            fmt(row.get("semantic_margin_delta")),
            fmt(row.get("probe_label_margin_delta")),
        ]
        for row in joined_layer
        if row.get("probe") == "ontology_collapse"
    ]
    lines.append(
        markdown_table(
            ["layer", "component", "probe", "Hodge coexact", "next", "semantic margin", "probe margin"],
            join_rows,
        )
    )
    lines.extend(["", "## Branch Scoreboard", ""])
    score_rows = [
        [
            row["component"],
            row["probe"],
            fmt(row.get("next_token_delta_mean")),
            f"L{row.get('next_token_delta_best_layer')}",
            fmt(row.get("probe_label_margin_delta_mean")),
            f"L{row.get('probe_label_margin_delta_best_layer')}",
            fmt(row.get("hodge_real_minus_shuffle_coexact_mean")),
        ]
        for row in branch_scores
        if row["selector"] == selector
    ]
    lines.append(
        markdown_table(
            [
                "component",
                "probe",
                "mean next",
                "best next layer",
                "mean probe margin",
                "best probe layer",
                "mean Hodge coexact-shuffle",
            ],
            score_rows,
        )
    )
    causal_k_rows = [
        [
            f"k={row['k']}",
            row["component"],
            fmt(row.get("hodge_coexact_mean")),
            fmt(row.get("next_token_delta_mean")),
            f"L{row.get('next_token_delta_best_layer')}",
            fmt(row.get("semantic_margin_delta_mean")),
            fmt(row.get("probe_label_margin_delta_mean")),
            f"L{row.get('probe_label_margin_delta_best_layer')}",
        ]
        for row in causal_k_scores
        if row["selector"] == selector and row["probe"] == "ontology_collapse"
    ]
    if causal_k_rows:
        lines.extend(["", "## Causal k-Sweep Scoreboard", ""])
        lines.append(
            markdown_table(
                [
                    "k",
                    "component",
                    "Hodge coexact",
                    "mean next",
                    "best next layer",
                    "semantic margin",
                    "ontology probe",
                    "best probe layer",
                ],
                causal_k_rows,
            )
        )
        lines.extend(
            [
                "",
                "This table keeps the structural k-sweep and the causal/probe k-sweep",
                "on the same row. It is the current branch-level check for whether",
                "presence/coexact causal roles track the graph-Hodge neighborhood",
                "scale.",
                "",
            ]
        )
    selector_delta_rows = [
        [
            f"k={row['k']}",
            row["component"],
            row["compare_selector"],
            fmt(row.get("baseline_next_token_delta_mean")),
            fmt(row.get("compare_next_token_delta_mean")),
            fmt(row.get("next_token_delta_mean_diff")),
            fmt(row.get("baseline_probe_label_margin_delta_mean")),
            fmt(row.get("compare_probe_label_margin_delta_mean")),
            fmt(row.get("probe_label_margin_delta_mean_diff")),
        ]
        for row in selector_deltas
        if row["probe"] == "ontology_collapse"
    ]
    if selector_delta_rows:
        lines.extend(["", f"## Selector Delta Scoreboard (baseline `{selector}`)", ""])
        lines.append(
            markdown_table(
                [
                    "k",
                    "component",
                    "compare selector",
                    "baseline next",
                    "compare next",
                    "next diff",
                    "baseline ontology probe",
                    "compare ontology probe",
                    "probe diff",
                ],
                selector_delta_rows,
            )
        )
        lines.extend(
            [
                "",
                "Selector deltas show whether a branch is robust to token-position",
                "choice or depends on selecting the largest local component norm.",
                "Positive diffs mean the compare selector exceeds the baseline",
                "selector on that metric.",
                "",
            ]
        )
    closed_loop_rows = [
        [
            row["source"],
            row["family"],
            row["component"],
            fmt(row.get("alpha")),
            row.get("n_prompt_rows", 0),
            row.get("n_rows", 0),
            fmt(row.get("hodge_coexact_l5_l8")),
            fmt(row.get("branch_gate_rate")),
            fmt(row.get("branch_specific_gate_rate")),
            fmt(row.get("branch_specific_prompt_rate")),
            fmt(row.get("mean_target_margin_delta_minus_random_mean")),
        ]
        for row in closed_loop_branch_scores
        if row.get("family") == "ontology_collapse"
    ]
    if closed_loop_rows:
        lines.extend(["", "## Closed-Loop Branch-Specific Scoreboard", ""])
        lines.append(
            markdown_table(
                [
                    "source",
                    "family",
                    "component",
                    "alpha",
                    "prompts",
                    "runs",
                    "Hodge coexact",
                    "raw gate",
                    "specific gate",
                    "specific prompts",
                    "target-random",
                ],
                closed_loop_rows,
            )
        )
        lines.extend(
            [
                "",
                "`specific gate` requires the closed-loop branch gate and matched",
                "random-tangent drift/target advantage at the same",
                "prompt/layer/k/seed/alpha. This table connects that causal",
                "closed-loop read back to the structural family-level Hodge branch.",
                "",
            ]
        )
    role_rows = [
        [
            row["component"],
            row["role_label"],
            fmt(row.get("hodge_coexact_mean")),
            fmt(row.get("next_token_delta_mean")),
            fmt(row.get("probe_label_margin_delta_mean")),
            fmt(row.get("closed_loop_branch_specific_gate_rate_mean")),
            fmt(row.get("closed_loop_target_margin_delta_minus_random_mean")),
        ]
        for row in branch_role_summary
        if row.get("probe") == "ontology_collapse" and row.get("selector") == selector
    ]
    if role_rows:
        lines.extend(["", "## Branch Role Summary", ""])
        lines.append(
            markdown_table(
                [
                    "component",
                    "role",
                    "Hodge coexact",
                    "mean next",
                    "ontology probe",
                    "closed-loop specific",
                    "closed-loop target-random",
                ],
                role_rows,
            )
        )
        lines.extend(
            [
                "",
                "The role label is only a compact readout. It is assigned from the",
                "signs of k-sweep next-token traversal, ontology-probe stabilization,",
                "and closed-loop branch-specific gate support; the numeric columns",
                "remain the evidence.",
                "",
            ]
        )
    diagnostic_rows = [
        [
            row["family"],
            row["probe"],
            row["component"],
            row["expected_role"],
            row["observed_role"],
            fmt(row.get("role_score")),
            row.get("criteria_failed", ""),
            fmt(row.get("next_token_delta_mean")),
            fmt(row.get("probe_label_margin_delta_mean")),
        ]
        for row in branch_role_diagnostics
        if row.get("role_pass") != 1
    ]
    if diagnostic_rows:
        lines.extend(["", "## Family-Local Branch Role Breaks", ""])
        lines.append(
            markdown_table(
                [
                    "family",
                    "probe",
                    "component",
                    "expected",
                    "observed",
                    "score",
                    "failed criteria",
                    "mean next",
                    "mean probe",
                ],
                diagnostic_rows,
            )
        )
        lines.extend(
            [
                "",
                "This table is deliberately stricter than the role summary. It",
                "checks the expected branch role separately for each prompt family",
                "and probe, using layer-averaged signs from `family_branch_join.csv`.",
                "Rows here are not failures of the whole hypothesis; they are the",
                "places where the family-local geometry bends the branch role.",
                "",
            ]
        )
    condition_rows = [
        [
            row["family"],
            row["component"],
            row["expected_role"],
            row["condition_label"],
            f"{row.get('role_pass_count', 0)}/{row.get('n_probe_cells', 0)}",
            fmt(row.get("mean_role_score")),
            row.get("observed_role_counts", ""),
            row.get("failed_criteria_counts", ""),
            fmt(row.get("mean_next_token_delta")),
            fmt(row.get("mean_probe_label_margin_delta")),
        ]
        for row in branch_condition_summary
    ]
    if condition_rows:
        lines.extend(["", "## Family-Component Branch Conditions", ""])
        lines.append(
            markdown_table(
                [
                    "family",
                    "component",
                    "expected",
                    "condition",
                    "passes",
                    "mean score",
                    "observed roles",
                    "failed criteria",
                    "mean next",
                    "mean probe",
                ],
                condition_rows,
            )
        )
        lines.extend(
            [
                "",
                "This table compresses the branch-role diagnostics over probes.",
                "`stable_expected` means the expected role holds in every probed",
                "family/probe cell; `systematic_partial_break` means no cell fully",
                "passes, but part of the expected role remains visible.",
                "",
            ]
        )
    layer_condition_rows = [
        [
            row["family"],
            row["component"],
            f"L{row['layer']}",
            row["condition_label"],
            f"{row.get('role_pass_count', 0)}/{row.get('n_probe_cells', 0)}",
            fmt(row.get("mean_role_score")),
            row.get("failed_criteria_counts", ""),
            fmt(row.get("mean_next_token_delta")),
            fmt(row.get("mean_probe_label_margin_delta")),
        ]
        for row in branch_layer_condition_summary
        if finite_float(row.get("role_pass_rate")) is not None and float(row.get("role_pass_rate")) < 1.0
    ]
    if layer_condition_rows:
        lines.extend(["", "## Layer-Resolved Branch Condition Breaks", ""])
        lines.append(
            markdown_table(
                [
                    "family",
                    "component",
                    "layer",
                    "condition",
                    "passes",
                    "mean score",
                    "failed criteria",
                    "mean next",
                    "mean probe",
                ],
                layer_condition_rows,
            )
        )
        lines.extend(
            [
                "",
                "This is the layer-level version of the family-component condition",
                "read. It points to the layers where a branch's expected role bends",
                "before the family-level average is formed.",
                "",
            ]
        )
    transition_rows = [
        [
            row["family"],
            row["component"],
            row["transition_label"],
            row.get("stable_layers", ""),
            row.get("mostly_or_mixed_layers", ""),
            row.get("break_layers", ""),
            row.get("longest_stable_span", ""),
            fmt(row.get("mean_layer_pass_rate")),
        ]
        for row in branch_layer_transition_summary
    ]
    if transition_rows:
        lines.extend(["", "## Branch Layer Transitions", ""])
        lines.append(
            markdown_table(
                [
                    "family",
                    "component",
                    "transition",
                    "stable layers",
                    "mixed layers",
                    "break layers",
                    "longest stable",
                    "mean pass",
                ],
                transition_rows,
            )
        )
        lines.extend(
            [
                "",
                "This table compresses the layer spine into stable and broken",
                "layer spans. It is useful for finding turn-on bands, fragmented",
                "stability, and branches that only work in a narrow depth range.",
                "",
            ]
        )
    candidate_rows = [
        [
            row["family"],
            row["component"],
            row["candidate_label"],
            fmt(row.get("priority_score")),
            row.get("recommended_layers", ""),
            fmt(row.get("stable_layer_rate")),
            fmt(row.get("closed_loop_branch_specific_gate_rate_mean")),
            fmt(row.get("closed_loop_target_margin_delta_minus_random_mean")),
            row.get("closed_loop_matched_random_rows", 0),
        ]
        for row in branch_band_candidate_scoreboard[:20]
    ]
    if candidate_rows:
        lines.extend(["", "## Branch-Band Candidate Scoreboard", ""])
        lines.append(
            markdown_table(
                [
                    "family",
                    "component",
                    "candidate",
                    "priority",
                    "recommended layers",
                    "stable rate",
                    "closed-loop gate",
                    "target-random",
                    "matched random rows",
                ],
                candidate_rows,
            )
        )
        lines.extend(
            [
                "",
                "This table joins the layer-transition spine with closed-loop",
                "branch-specific support. It is a next-experiment queue rather",
                "than a claim table: high structural support without closed-loop",
                "coverage becomes a structural candidate, while high closed-loop",
                "support with narrower layer stability becomes an exception probe.",
                "",
            ]
        )
    reverse_rows = [
        [
            row.get("panel_label", ""),
            row.get("target_set", ""),
            fmt(row.get("branch_specific_gate_rate")),
            fmt(row.get("random_branch_gate_rate")),
            fmt(row.get("mean_target_margin_delta_minus_random_mean")),
            fmt(row.get("token_drift_rate_mean")),
        ]
        for row in reverse_specificity
        if row.get("component") == "negative_coexact"
    ]
    if reverse_rows:
        lines.extend(["", "## Reverse Exception Specificity", ""])
        lines.append(
            markdown_table(
                [
                    "panel",
                    "target set",
                    "specific gate",
                    "random gate",
                    "target-random",
                    "drift",
                ],
                reverse_rows,
            )
        )
        lines.extend(
            [
                "",
                "Reverse exception specificity keeps the prompt-local negative-coexact",
                "exceptions separate from the family-level sign-control read. The",
                "control rows check whether decoded drift remains while semantic",
                "target advantage disappears.",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "The structural Hodge branch remains coexact-dominant when triangles are",
            "present, with harmonic energy near zero. The causal branch split is",
            "not identical to the structural split: coexact and coexact-derived",
            "directions carry next-token traversal, while presence-derived",
            "directions carry learned-probe stabilization. When causal k-sweep",
            "rows are present, the same split is checked against neighborhood",
            "scale. This report is the ledger for tracking those branches side by",
            "side.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hodge-root", default="spiral_out_hltd_invariance")
    parser.add_argument("--steering-root", default="spiral_out_hltd_dissociation_steering_full_mps")
    parser.add_argument("--probe-root", default="spiral_out_hltd_dissociation_probe_full_mps")
    parser.add_argument(
        "--closed-loop-roots",
        nargs="*",
        default=[],
        help="Optional closed-loop run roots containing closed_loop_prompt_summary.csv.",
    )
    parser.add_argument("--output-root", default="spiral_out_hltd_branch_hodge")
    parser.add_argument("--topology", default="triangles")
    parser.add_argument("--compare-topologies", nargs="+", default=["triangles", "no_triangles"])
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument(
        "--structural-ks",
        type=int,
        nargs="+",
        default=None,
        help="k values for structural Hodge branch robustness tables; defaults to --k.",
    )
    parser.add_argument(
        "--causal-ks",
        type=int,
        nargs="+",
        default=None,
        help="k values for steering/probe branch scoreboards; defaults to --k.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--selector", default="middle")
    parser.add_argument(
        "--compare-selectors",
        nargs="+",
        default=None,
        help="Token selectors to compare in causal k-scoreboards; defaults to --selector.",
    )
    parser.add_argument("--components", nargs="+", default=COMPONENTS)
    parser.add_argument("--probes", nargs="+", default=PROBES)
    parser.add_argument(
        "--reverse-specificity-csv",
        nargs="*",
        default=[],
        help="Optional reverse-exception target-specificity CSVs to copy into the branch ledger.",
    )
    args = parser.parse_args(argv)

    hodge_root = Path(args.hodge_root)
    steering_root = Path(args.steering_root)
    probe_root = Path(args.probe_root)
    closed_loop_roots = [Path(root) for root in args.closed_loop_roots]
    reverse_specificity_paths = [Path(path) for path in args.reverse_specificity_csv]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    layers = {int(layer) for layer in args.layers}
    components = {str(component) for component in args.components}
    probes = {str(probe) for probe in args.probes}
    structural_ks = {int(k) for k in (args.structural_ks or [args.k])}
    causal_ks = {int(k) for k in (args.causal_ks or [args.k])}
    compare_selectors = {str(selector) for selector in (args.compare_selectors or [args.selector])}
    compare_selectors.add(str(args.selector))

    family_hodge = layer_hodge_rows(hodge_root=hodge_root, topology=args.topology, k=args.k, layers=layers)
    hodge_layer = aggregate_layer_hodge(family_hodge)
    family_k = family_k_rows(hodge_root=hodge_root, topology=args.topology, k=args.k)
    family_k_sweep = family_k_sweep_rows(
        hodge_root=hodge_root,
        topology=args.topology,
        ks=structural_ks,
    )
    k_sweep = aggregate_k_sweep(family_k_sweep)
    topology_family_k = topology_family_k_rows(
        hodge_root=hodge_root,
        k=args.k,
        topologies={str(x) for x in args.compare_topologies},
    )
    steering_layer = steering_layer_rows(
        steering_root=steering_root,
        k=args.k,
        layers=layers,
        selector=args.selector,
        components=components,
    )
    steering_family = steering_family_rows(
        steering_root=steering_root,
        k=args.k,
        layers=layers,
        selector=args.selector,
        components=components,
    )
    probe_layer = probe_layer_rows(
        probe_root=probe_root,
        k=args.k,
        layers=layers,
        selector=args.selector,
        components=components,
        probes=probes,
    )
    probe_family = probe_family_rows(
        probe_root=probe_root,
        k=args.k,
        layers=layers,
        selector=args.selector,
        components=components,
        probes=probes,
    )
    steering_layer_causal = steering_layer_rows_for_ks_and_selectors(
        steering_root=steering_root,
        ks=causal_ks,
        layers=layers,
        selectors=compare_selectors,
        components=components,
    )
    probe_layer_causal = probe_layer_rows_for_ks_and_selectors(
        probe_root=probe_root,
        ks=causal_ks,
        layers=layers,
        selectors=compare_selectors,
        components=components,
        probes=probes,
    )
    joined_layer = join_layer_rows(hodge_rows=hodge_layer, steering_rows=steering_layer, probe_rows=probe_layer)
    joined_family = join_family_rows(
        hodge_rows=family_hodge,
        steering_rows=steering_family,
        probe_rows=probe_family,
    )
    branch_scores = build_branch_score_rows(joined_layer)
    causal_k_scores = build_causal_k_score_rows(
        k_sweep=k_sweep,
        steering_rows=steering_layer_causal,
        probe_rows=probe_layer_causal,
    )
    selector_deltas = build_selector_delta_rows(causal_k_scores, baseline_selector=args.selector)
    closed_loop_prompts = closed_loop_prompt_rows(
        closed_loop_roots=closed_loop_roots,
        components=components,
    )
    closed_loop_branch_scores = build_closed_loop_branch_score_rows(
        closed_loop_rows=closed_loop_prompts,
        family_k=family_k,
    )
    branch_role_summary = build_branch_role_summary_rows(
        causal_k_scores=causal_k_scores,
        closed_loop_branch_scores=closed_loop_branch_scores,
        selector=args.selector,
    )
    branch_role_diagnostics = build_branch_role_diagnostic_rows(
        joined_family,
        selector=args.selector,
    )
    branch_layer_condition_summary = build_branch_layer_condition_rows(
        joined_family,
        selector=args.selector,
    )
    branch_layer_transition_summary = build_branch_layer_transition_rows(
        branch_layer_condition_summary,
        selector=args.selector,
    )
    branch_condition_summary = build_branch_condition_summary_rows(
        branch_role_diagnostics,
        selector=args.selector,
    )
    branch_band_candidate_scoreboard = build_branch_band_candidate_rows(
        branch_layer_transition_summary=branch_layer_transition_summary,
        closed_loop_branch_scores=closed_loop_branch_scores,
        selector=args.selector,
    )
    reverse_specificity = reverse_specificity_rows(reverse_specificity_paths)

    write_csv(family_k, output_root / "hodge_family_k.csv")
    write_csv(family_k_sweep, output_root / "hodge_family_k_sweep.csv")
    write_csv(k_sweep, output_root / "hodge_k_sweep.csv")
    write_csv(topology_family_k, output_root / "hodge_topology_family_k.csv")
    write_csv(family_hodge, output_root / "hodge_family_layer.csv")
    write_csv(hodge_layer, output_root / "hodge_layer.csv")
    write_csv(joined_layer, output_root / "causal_hodge_join.csv")
    write_csv(joined_family, output_root / "family_branch_join.csv")
    write_csv(branch_scores, output_root / "branch_scoreboard.csv")
    write_csv(causal_k_scores, output_root / "causal_k_scoreboard.csv")
    write_csv(selector_deltas, output_root / "selector_delta_scoreboard.csv")
    write_csv(closed_loop_prompts, output_root / "closed_loop_prompt_join.csv")
    write_csv(closed_loop_branch_scores, output_root / "closed_loop_branch_scoreboard.csv")
    write_csv(branch_role_summary, output_root / "branch_role_summary.csv")
    write_csv(branch_role_diagnostics, output_root / "branch_role_diagnostics.csv")
    write_csv(branch_layer_condition_summary, output_root / "branch_layer_condition_summary.csv")
    write_csv(branch_layer_transition_summary, output_root / "branch_layer_transition_summary.csv")
    write_csv(branch_condition_summary, output_root / "branch_condition_summary.csv")
    write_csv(branch_band_candidate_scoreboard, output_root / "branch_band_candidate_scoreboard.csv")
    write_csv(reverse_specificity, output_root / "reverse_exception_specificity.csv")
    write_report(
        path=output_root / "summary_report.md",
        family_k=family_k,
        k_sweep=k_sweep,
        topology_family_k=topology_family_k,
        hodge_layer=hodge_layer,
        joined_layer=joined_layer,
        branch_scores=branch_scores,
        causal_k_scores=causal_k_scores,
        selector_deltas=selector_deltas,
        closed_loop_branch_scores=closed_loop_branch_scores,
        branch_role_summary=branch_role_summary,
        branch_role_diagnostics=branch_role_diagnostics,
        branch_layer_condition_summary=branch_layer_condition_summary,
        branch_layer_transition_summary=branch_layer_transition_summary,
        branch_condition_summary=branch_condition_summary,
        branch_band_candidate_scoreboard=branch_band_candidate_scoreboard,
        reverse_specificity=reverse_specificity,
        selector=args.selector,
    )

    print(f"wrote branch Hodge summaries -> {output_root}")
    print(f"hodge family/layer rows: {len(family_hodge)}")
    print(f"structural k-sweep rows: {len(family_k_sweep)}")
    print(f"joined layer rows: {len(joined_layer)}")
    print(f"joined family rows: {len(joined_family)}")
    print(f"causal k-score rows: {len(causal_k_scores)}")
    print(f"selector delta rows: {len(selector_deltas)}")
    print(f"closed-loop prompt rows: {len(closed_loop_prompts)}")
    print(f"closed-loop branch-score rows: {len(closed_loop_branch_scores)}")
    print(f"branch role summary rows: {len(branch_role_summary)}")
    print(f"branch role diagnostic rows: {len(branch_role_diagnostics)}")
    print(f"branch layer condition rows: {len(branch_layer_condition_summary)}")
    print(f"branch layer transition rows: {len(branch_layer_transition_summary)}")
    print(f"branch condition summary rows: {len(branch_condition_summary)}")
    print(f"branch-band candidate rows: {len(branch_band_candidate_scoreboard)}")
    print(f"reverse specificity rows: {len(reverse_specificity)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
