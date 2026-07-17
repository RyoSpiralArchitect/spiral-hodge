#!/usr/bin/env python3
"""Summarize all-interior HLTD steering/probe gates by token-position bins."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import summarize_hltd_steering


STEERING_METRICS = [
    "kl_base_to_steered",
    "entropy_delta",
    "next_token_logprob_delta",
    "target_logprob_mass_delta",
    "control_logprob_mass_delta",
    "semantic_margin_delta",
]

PROBE_METRICS = [
    "positive_prob_delta",
    "positive_logit_delta",
    "label_margin_delta",
    "probe_entropy_delta",
]


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean_or_nan(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return mean(vals) if vals else float("nan")


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


def position_bin(position_frac: float, bins: int) -> int:
    if bins <= 0:
        raise ValueError("bins must be positive")
    clipped = min(max(float(position_frac), 0.0), 1.0)
    return min(int(clipped * bins), bins - 1)


def token_count_lookup(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, int, int], int]:
    max_token: Dict[Tuple[str, str, int, int], int] = defaultdict(int)
    explicit_counts: Dict[Tuple[str, str, int, int], int] = {}
    for row in rows:
        key = (
            str(row["family"]),
            str(row["prompt_id"]),
            int(float(row["layer"])),
            int(float(row["k"])),
        )
        token_count = finite_float(row.get("token_count"))
        if token_count is not None and int(token_count) > 0:
            explicit_counts[key] = max(explicit_counts.get(key, 0), int(token_count))
            continue
        token_index = finite_float(row.get("token_index"))
        if token_index is not None:
            max_token[key] = max(max_token[key], int(token_index))
    out = {key: value + 2 for key, value in max_token.items()}
    out.update(explicit_counts)
    return out


def attach_position(row: Dict[str, Any], token_counts: Dict[Tuple[str, str, int, int], int], bins: int) -> Dict[str, Any]:
    out = dict(row)
    key = (
        str(out["family"]),
        str(out["prompt_id"]),
        int(float(out["layer"])),
        int(float(out["k"])),
    )
    token_count = int(token_counts.get(key, 0))
    token_index = int(float(out.get("token_index", 0) or 0))
    denom = max(token_count - 1, 1)
    frac = float(token_index / denom)
    out["token_count_inferred"] = token_count
    out["position_frac"] = frac
    out["position_bin"] = position_bin(frac, bins)
    return out


def load_steering_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for csv_path in sorted(root.glob("*__*__L*__k*/steering_metrics.csv")):
        rows.extend(summarize_hltd_steering.build_run_rows(csv_path))
    return rows


def steering_pairwise_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    bins: int,
    baseline_component: str = "random_tangent",
    token_selector: str = "all_interior",
) -> List[Dict[str, Any]]:
    token_counts = token_count_lookup(rows)
    groups: Dict[Tuple[str, str, int, int, str, str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for raw in rows:
        row = attach_position(raw, token_counts, bins)
        if str(row.get("token_selector")) != token_selector:
            continue
        key = (
            str(row["family"]),
            str(row["prompt_id"]),
            int(float(row["layer"])),
            int(float(row["k"])),
            str(row.get("seed", "0")),
            str(row.get("token_index", "")),
            str(row.get("alpha", "1.0")),
            str(row.get("token_selector", "")),
        )
        groups[key][str(row["component"])] = row

    out: List[Dict[str, Any]] = []
    for (_family, _prompt_id, _layer, _k, _seed, _token_index, _alpha, _selector), by_component in sorted(groups.items()):
        baseline = by_component.get(baseline_component)
        if baseline is None or (finite_float(baseline.get("component_active")) or 0.0) <= 0.0:
            continue
        for component, row in sorted(by_component.items()):
            if component == baseline_component:
                continue
            if (finite_float(row.get("component_active")) or 0.0) <= 0.0:
                continue
            item: Dict[str, Any] = {
                "family": row["family"],
                "prompt_id": row["prompt_id"],
                "layer": int(float(row["layer"])),
                "k": int(float(row["k"])),
                "seed": row.get("seed", "0"),
                "token_selector": row.get("token_selector", token_selector),
                "token_index": int(float(row.get("token_index", 0) or 0)),
                "token": row.get("token", ""),
                "position_frac": row["position_frac"],
                "position_bin": row["position_bin"],
                "component": component,
                "alpha": float(row.get("alpha", 1.0) or 1.0),
            }
            for metric in STEERING_METRICS:
                a = finite_float(row.get(metric))
                b = finite_float(baseline.get(metric))
                item[f"{metric}_minus_{baseline_component}"] = (
                    a - b if a is not None and b is not None else float("nan")
                )
            out.append(item)
    return out


def load_probe_rows(root: Path) -> List[Dict[str, Any]]:
    path = root / "probe_metrics.csv"
    return [dict(row) for row in read_csv(path)]


def probe_pairwise_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    bins: int,
    baseline_component: str = "random_tangent",
    token_selector: str = "all_interior",
) -> List[Dict[str, Any]]:
    token_counts = token_count_lookup(rows)
    groups: Dict[Tuple[str, str, int, int, str, str, str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for raw in rows:
        row = attach_position(raw, token_counts, bins)
        if str(row.get("token_selector")) != token_selector:
            continue
        key = (
            str(row["family"]),
            str(row["prompt_id"]),
            int(float(row["layer"])),
            int(float(row["k"])),
            str(row.get("seed", "0")),
            str(row.get("token_index", "")),
            str(row.get("alpha", "1.0")),
            str(row.get("token_selector", "")),
            str(row.get("probe", "")),
        )
        groups[key][str(row["component"])] = row

    out: List[Dict[str, Any]] = []
    for (_family, _prompt_id, _layer, _k, _seed, _token_index, _alpha, _selector, _probe), by_component in sorted(groups.items()):
        baseline = by_component.get(baseline_component)
        if baseline is None or (finite_float(baseline.get("component_active")) or 0.0) <= 0.0:
            continue
        for component, row in sorted(by_component.items()):
            if component == baseline_component:
                continue
            if (finite_float(row.get("component_active")) or 0.0) <= 0.0:
                continue
            item: Dict[str, Any] = {
                "family": row["family"],
                "prompt_id": row["prompt_id"],
                "layer": int(float(row["layer"])),
                "k": int(float(row["k"])),
                "seed": row.get("seed", "0"),
                "token_selector": row.get("token_selector", token_selector),
                "token_index": int(float(row.get("token_index", 0) or 0)),
                "token": row.get("token", ""),
                "position_frac": row["position_frac"],
                "position_bin": row["position_bin"],
                "component": component,
                "alpha": float(row.get("alpha", 1.0) or 1.0),
                "probe": row.get("probe", ""),
                "probe_label": row.get("probe_label", ""),
            }
            for metric in PROBE_METRICS:
                a = finite_float(row.get(metric))
                b = finite_float(baseline.get(metric))
                item[f"{metric}_minus_{baseline_component}"] = (
                    a - b if a is not None and b is not None else float("nan")
                )
            out.append(item)
    return out


def aggregate_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    metrics: Sequence[str],
    include_probe: bool = False,
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key_parts: List[Any] = [
            str(row["family"]),
            int(row["layer"]),
            int(row["k"]),
            int(row["position_bin"]),
            str(row["component"]),
        ]
        if include_probe:
            key_parts.append(str(row["probe"]))
        buckets[tuple(key_parts)].append(row)

    out: List[Dict[str, Any]] = []
    for key, group in sorted(buckets.items()):
        family, layer, k, bin_index, component, *rest = key
        item: Dict[str, Any] = {
            "family": family,
            "layer": layer,
            "k": k,
            "position_bin": bin_index,
            "component": component,
            "n_pairs": len(group),
            "position_frac_mean": mean_or_nan([float(row["position_frac"]) for row in group]),
        }
        if include_probe:
            item["probe"] = rest[0]
        for metric in metrics:
            values = [value for value in (finite_float(row.get(metric)) for row in group) if value is not None]
            item[f"{metric}_mean"] = mean_or_nan(values)
            item[f"{metric}_max"] = max(values) if values else float("nan")
            item[f"{metric}_min"] = min(values) if values else float("nan")
        out.append(item)
    return out


def join_position_rows(
    steering_summary: Sequence[Dict[str, Any]],
    probe_summary: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    steering_by_key = {
        (
            str(row["family"]),
            int(row["layer"]),
            int(row["k"]),
            int(row["position_bin"]),
            str(row["component"]),
        ): row
        for row in steering_summary
    }
    out: List[Dict[str, Any]] = []
    for probe in probe_summary:
        key = (
            str(probe["family"]),
            int(probe["layer"]),
            int(probe["k"]),
            int(probe["position_bin"]),
            str(probe["component"]),
        )
        steering = steering_by_key.get(key, {})
        out.append(
            {
                "family": probe["family"],
                "layer": probe["layer"],
                "k": probe["k"],
                "position_bin": probe["position_bin"],
                "position_frac_mean": probe["position_frac_mean"],
                "component": probe["component"],
                "probe": probe["probe"],
                "n_steering_pairs": steering.get("n_pairs", 0),
                "n_probe_pairs": probe.get("n_pairs", 0),
                "next_token_delta_mean": steering.get(
                    "next_token_logprob_delta_minus_random_tangent_mean",
                    float("nan"),
                ),
                "semantic_margin_delta_mean": steering.get(
                    "semantic_margin_delta_minus_random_tangent_mean",
                    float("nan"),
                ),
                "probe_label_margin_delta_mean": probe.get(
                    "label_margin_delta_minus_random_tangent_mean",
                    float("nan"),
                ),
                "probe_positive_prob_delta_mean": probe.get(
                    "positive_prob_delta_minus_random_tangent_mean",
                    float("nan"),
                ),
            }
        )
    return out


def summarize_peak_bins(joined_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        buckets[(str(row["component"]), str(row["probe"]), str(row["family"]), int(row["k"]))].append(row)

    def metric_value(row: Dict[str, Any], metric: str) -> float:
        value = finite_float(row.get(metric))
        return value if value is not None else float("-inf")

    out: List[Dict[str, Any]] = []
    for (component, probe, family, k), rows in sorted(buckets.items()):
        for metric in ["next_token_delta_mean", "probe_label_margin_delta_mean"]:
            valid = [row for row in rows if finite_float(row.get(metric)) is not None]
            if not valid:
                continue
            best = max(valid, key=lambda row: metric_value(row, metric))
            out.append(
                {
                    "family": family,
                    "k": k,
                    "component": component,
                    "probe": probe,
                    "metric": metric,
                    "peak_layer": best["layer"],
                    "peak_position_bin": best["position_bin"],
                    "peak_position_frac_mean": best["position_frac_mean"],
                    "peak_value": best[metric],
                }
            )
    return out


def summarize_cross_family_peaks(joined_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find layer/bin peaks after averaging family summaries."""
    grouped: Dict[Tuple[str, str, str, int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        for metric in ["next_token_delta_mean", "probe_label_margin_delta_mean"]:
            if finite_float(row.get(metric)) is None:
                continue
            key = (
                str(row["component"]),
                str(row["probe"]),
                metric,
                int(row["k"]),
                int(row["layer"]),
                int(row["position_bin"]),
            )
            grouped[key].append(row)

    candidates: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for (component, probe, metric, k, layer, bin_index), rows in sorted(grouped.items()):
        values = [float(row[metric]) for row in rows]
        families = sorted({str(row["family"]) for row in rows})
        candidates[(component, probe, metric, k)].append(
            {
                "component": component,
                "probe": probe,
                "metric": metric,
                "k": k,
                "peak_layer": layer,
                "peak_position_bin": bin_index,
                "peak_position_frac_mean": mean_or_nan([float(row["position_frac_mean"]) for row in rows]),
                "peak_value_mean": mean_or_nan(values),
                "peak_value_min": min(values),
                "peak_value_max": max(values),
                "n_families": len(families),
            }
        )

    out: List[Dict[str, Any]] = []
    for _key, rows in sorted(candidates.items()):
        out.append(max(rows, key=lambda row: float(row["peak_value_mean"])))
    return out


def write_report(
    *,
    output_root: Path,
    joined_rows: Sequence[Dict[str, Any]],
    peak_rows: Sequence[Dict[str, Any]],
    cross_family_peak_rows: Sequence[Dict[str, Any]],
    bins: int,
    steering_pairwise_count: int,
    probe_pairwise_count: int,
) -> None:
    families = sorted({str(row["family"]) for row in joined_rows})
    k_values = sorted({int(row["k"]) for row in joined_rows})
    layers = sorted({int(row["layer"]) for row in joined_rows})
    components = sorted({str(row["component"]) for row in joined_rows})
    populated_bins = sorted({int(row["position_bin"]) for row in joined_rows})
    lines: List[str] = [
        "# HLTD All-Interior Position Gate",
        "",
        "This report bins all-interior one-step steering and learned-probe deltas",
        f"into {bins} normalized token-position bins.",
        "",
        "## Run Coverage",
        "",
        markdown_table(
            ["families", "k", "layers", "components", "populated bins", "steering pairs", "probe pairs"],
            [
                [
                    len(families),
                    ", ".join(str(k) for k in k_values),
                    ", ".join(f"L{layer}" for layer in layers),
                    len(components),
                    len(populated_bins),
                    steering_pairwise_count,
                    probe_pairwise_count,
                ]
            ],
        ),
        "",
        "## Ontology Position Scoreboard",
        "",
    ]
    ontology_rows = [
        row
        for row in joined_rows
        if row["probe"] == "ontology_collapse"
        and row["component"] in {"presence", "coexact", "presence_plus_coexact"}
    ]
    table_rows = [
        [
            row["family"],
            row["k"],
            f"L{row['layer']}",
            f"bin {row['position_bin']}",
            row["component"],
            fmt(row.get("next_token_delta_mean")),
            fmt(row.get("semantic_margin_delta_mean")),
            fmt(row.get("probe_label_margin_delta_mean")),
        ]
        for row in ontology_rows
    ]
    lines.append(
        markdown_table(
            ["family", "k", "layer", "bin", "component", "next", "semantic", "ontology probe"],
            table_rows[:90],
        )
    )
    if len(table_rows) > 90:
        lines.append("")
        lines.append(f"Table truncated at 90 rows; CSV contains {len(table_rows)} ontology rows.")

    lines.extend(["", "## Peak Bins", ""])
    peak_table = [
        [
            row["family"],
            row["k"],
            row["component"],
            row["probe"],
            row["metric"],
            f"L{row['peak_layer']}",
            f"bin {row['peak_position_bin']}",
            fmt(row.get("peak_value")),
        ]
        for row in peak_rows
        if row["probe"] == "ontology_collapse"
        and row["component"] in {"presence", "coexact", "presence_plus_coexact"}
    ]
    lines.append(
        markdown_table(
            ["family", "k", "component", "probe", "metric", "layer", "bin", "value"],
            peak_table[:80],
        )
    )

    lines.extend(["", "## Cross-Family Ontology Peaks", ""])
    cross_table = [
        [
            row["component"],
            row["probe"],
            row["metric"],
            row["k"],
            f"L{row['peak_layer']}",
            f"bin {row['peak_position_bin']}",
            fmt(row.get("peak_value_mean")),
            fmt(row.get("peak_value_min")),
            fmt(row.get("peak_value_max")),
            row["n_families"],
        ]
        for row in cross_family_peak_rows
        if row["probe"] == "ontology_collapse"
        and row["component"] in {"presence", "coexact", "presence_plus_coexact", "coexact_minus_presence"}
    ]
    lines.append(
        markdown_table(
            ["component", "probe", "metric", "k", "layer", "bin", "mean", "min", "max", "families"],
            cross_table[:80],
        )
    )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "Use this gate to check whether structural branch-position bins remain",
            "causally useful after subtracting matched random-tangent interventions.",
            "The family-level peak table is sensitive to prompt-family variation;",
            "the cross-family table is stricter and highlights layer/bin positions",
            "that survive averaging across families.",
            "",
        ]
    )
    (output_root / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steering-root", required=True)
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--output-root", default="spiral_out_hltd_position_gate")
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--token-selector", default="all_interior")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    steering_rows = steering_pairwise_rows(
        load_steering_rows(Path(args.steering_root)),
        bins=int(args.bins),
        token_selector=args.token_selector,
    )
    probe_rows = probe_pairwise_rows(
        load_probe_rows(Path(args.probe_root)),
        bins=int(args.bins),
        token_selector=args.token_selector,
    )
    steering_summary = aggregate_rows(
        steering_rows,
        metrics=[f"{metric}_minus_random_tangent" for metric in STEERING_METRICS],
    )
    probe_summary = aggregate_rows(
        probe_rows,
        metrics=[f"{metric}_minus_random_tangent" for metric in PROBE_METRICS],
        include_probe=True,
    )
    joined_rows = join_position_rows(steering_summary, probe_summary)
    peak_rows = summarize_peak_bins(joined_rows)
    cross_family_peak_rows = summarize_cross_family_peaks(joined_rows)

    write_csv(steering_rows, output_root / "steering_position_pairwise.csv")
    write_csv(probe_rows, output_root / "probe_position_pairwise.csv")
    write_csv(steering_summary, output_root / "steering_position_summary.csv")
    write_csv(probe_summary, output_root / "probe_position_summary.csv")
    write_csv(joined_rows, output_root / "joined_position_summary.csv")
    write_csv(peak_rows, output_root / "position_peak_summary.csv")
    write_csv(cross_family_peak_rows, output_root / "position_cross_family_peak_summary.csv")
    write_report(
        output_root=output_root,
        joined_rows=joined_rows,
        peak_rows=peak_rows,
        cross_family_peak_rows=cross_family_peak_rows,
        bins=int(args.bins),
        steering_pairwise_count=len(steering_rows),
        probe_pairwise_count=len(probe_rows),
    )

    print(f"wrote all-interior position gate summary -> {output_root}")
    print(f"steering pairwise rows: {len(steering_rows)}")
    print(f"probe pairwise rows: {len(probe_rows)}")
    print(f"joined position rows: {len(joined_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
