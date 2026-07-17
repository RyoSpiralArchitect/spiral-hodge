#!/usr/bin/env python3
"""Summarize one-step HLTD steering suite outputs."""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple


RUN_RE = re.compile(r"(?P<family>.+)__(?P<prompt_id>.+)__L(?P<layer>\d+)__k(?P<k>\d+)$")

ROW_FIELDS = [
    "family",
    "prompt_id",
    "layer",
    "k",
    "complex_mode",
    "hodge_solver",
    "betti_1_fraction_target",
    "betti_1_fraction",
    "betti_1_fraction_abs_error",
    "cycle_rank",
    "triangle_rank",
    "triangle_count",
    "triangle_count_full",
    "triangle_fill_actual",
    "filtration_radius_scale_actual",
    "hodge_exact_ratio",
    "hodge_coexact_ratio",
    "hodge_harmonic_ratio",
    "seed",
    "random_tangent_reference",
    "token_selector",
    "selector_component",
    "node_index",
    "token_index",
    "token_count",
    "token",
    "next_token",
    "component",
    "alpha",
    "component_active",
    "delta_norm",
    "natural_step_norm",
    "chart_norm",
    "hidden_direction_norm",
    "base_entropy",
    "steered_entropy",
    "entropy_delta",
    "kl_base_to_steered",
    "kl_steered_to_base",
    "js_divergence",
    "base_top_token",
    "steered_top_token",
    "top_changed",
    "top_shift_token",
    "top_shift_logprob_delta",
    "next_token_logprob_base",
    "next_token_logprob_steered",
    "next_token_logprob_delta",
    "target_token",
    "target_logprob_delta",
    "target_set",
    "target_set_size",
    "control_set_size",
    "target_logprob_mass_base",
    "target_logprob_mass_steered",
    "target_logprob_mass_delta",
    "target_prob_mass_base",
    "target_prob_mass_steered",
    "target_prob_mass_delta",
    "control_logprob_mass_base",
    "control_logprob_mass_steered",
    "control_logprob_mass_delta",
    "control_prob_mass_base",
    "control_prob_mass_steered",
    "control_prob_mass_delta",
    "semantic_margin_delta",
    "semantic_prob_margin_delta",
]

AGG_METRICS = [
    "component_active",
    "hodge_exact_ratio",
    "hodge_coexact_ratio",
    "hodge_harmonic_ratio",
    "kl_base_to_steered",
    "kl_steered_to_base",
    "js_divergence",
    "entropy_delta",
    "top_changed",
    "top_shift_logprob_delta",
    "next_token_logprob_delta",
    "target_logprob_delta",
    "target_logprob_mass_delta",
    "target_prob_mass_delta",
    "control_logprob_mass_delta",
    "control_prob_mass_delta",
    "semantic_margin_delta",
    "semantic_prob_margin_delta",
]

PAIRWISE_METRICS = [
    "kl_base_to_steered",
    "js_divergence",
    "entropy_delta",
    "top_changed",
    "top_shift_logprob_delta",
    "next_token_logprob_delta",
    "target_logprob_delta",
    "target_logprob_mass_delta",
    "target_prob_mass_delta",
    "control_logprob_mass_delta",
    "control_prob_mass_delta",
    "semantic_margin_delta",
    "semantic_prob_margin_delta",
]


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean_or_nan(values: Sequence[float]) -> float:
    return mean(values) if values else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return "nan"
    return f"{number:.{digits}f}"


def parse_run_dir(path: Path) -> Tuple[str, str, int, int]:
    match = RUN_RE.match(path.name)
    if not match:
        raise ValueError(f"Run directory does not match FAMILY__PROMPT__LAYER__kK pattern: {path}")
    return (
        match.group("family"),
        match.group("prompt_id"),
        int(match.group("layer")),
        int(match.group("k")),
    )


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_run_rows(csv_path: Path) -> List[Dict[str, Any]]:
    try:
        family, prompt_id, layer, k = parse_run_dir(csv_path.parent)
    except ValueError:
        family, prompt_id, layer, k = "manual", "", 0, 0
    rows: List[Dict[str, Any]] = []
    for row in read_rows(csv_path):
        item: Dict[str, Any] = {key: row.get(key, "") for key in ROW_FIELDS}
        item.update(
            {
                "family": row.get("family", family) or family,
                "prompt_id": row.get("prompt_id", prompt_id) or prompt_id,
                "layer": int(float(row.get("layer", layer) or layer)),
                "k": int(float(row.get("k", k) or k)),
                "complex_mode": row.get("complex_mode", "legacy") or "legacy",
                "hodge_solver": row.get("hodge_solver", "") or "",
                "seed": row.get("seed", "0") or "0",
                "random_tangent_reference": row.get("random_tangent_reference", "legacy_coexact_node_speed")
                or "legacy_coexact_node_speed",
                "token_selector": row.get("token_selector", "max_component") or "max_component",
                "selector_component": row.get("selector_component", "coexact") or "coexact",
                "node_index": row.get("node_index", "") or "",
            }
        )
        rows.append(item)
    return rows


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ROW_FIELDS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def derived_path(output: Path, label: str, suffix: str = ".csv") -> Path:
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}_{label}{suffix}")


def build_component_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, int, int, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["family"]),
            int(float(row["layer"])),
            int(float(row["k"])),
            str(row.get("complex_mode", "legacy") or "legacy"),
            str(row.get("betti_1_fraction_target", "") or ""),
            str(row.get("token_selector", "max_component")),
            str(row["component"]),
            str(row["alpha"]),
        )
        buckets[key].append(row)

    out: List[Dict[str, Any]] = []
    for (family, layer, k, complex_mode, betti_target, token_selector, component, alpha), group in sorted(
        buckets.items()
    ):
        item: Dict[str, Any] = {
            "family": family,
            "layer": layer,
            "k": k,
            "complex_mode": complex_mode,
            "betti_1_fraction_target": finite_float(betti_target) if betti_target else "",
            "token_selector": token_selector,
            "component": component,
            "alpha": float(alpha),
            "n_rows": len(group),
            "n_seeds": len({str(row.get("seed", "0")) for row in group}),
            "n_tokens": len({str(row.get("token_index", "")) for row in group}),
        }
        for metric in AGG_METRICS:
            values = [value for value in (finite_float(row.get(metric)) for row in group) if value is not None]
            item[f"{metric}_mean"] = mean_or_nan(values)
        out.append(item)
    return out


def build_pairwise_summary(
    rows: Sequence[Dict[str, Any]],
    *,
    baseline_component: str = "random_tangent",
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, int, int, str, str, str, str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["family"]),
            str(row["prompt_id"]),
            int(float(row["layer"])),
            int(float(row["k"])),
            str(row.get("complex_mode", "legacy") or "legacy"),
            str(row.get("betti_1_fraction_target", "") or ""),
            str(row.get("seed", "0")),
            str(row.get("token_selector", "max_component")),
            str(row.get("token_index", "")),
            str(row["alpha"]),
        )
        groups[key][str(row["component"])] = row

    delta_buckets: Dict[Tuple[str, int, int, str, str, str, str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    counts: Dict[Tuple[str, int, int, str, str, str, str, str], int] = defaultdict(int)
    for (
        family,
        _prompt_id,
        layer,
        k,
        complex_mode,
        betti_target,
        _seed,
        token_selector,
        _token_index,
        alpha,
    ), by_component in groups.items():
        baseline = by_component.get(baseline_component)
        if baseline is None:
            continue
        if (finite_float(baseline.get("component_active")) or 0.0) <= 0.0:
            continue
        for component, row in by_component.items():
            if component == baseline_component:
                continue
            if (finite_float(row.get("component_active")) or 0.0) <= 0.0:
                continue
            out_key = (
                family,
                layer,
                k,
                complex_mode,
                betti_target,
                token_selector,
                component,
                alpha,
            )
            added = False
            for metric in PAIRWISE_METRICS:
                a = finite_float(row.get(metric))
                b = finite_float(baseline.get(metric))
                if a is None or b is None:
                    continue
                delta_buckets[out_key][f"{metric}_minus_{baseline_component}"].append(a - b)
                added = True
            if added:
                counts[out_key] += 1

    out: List[Dict[str, Any]] = []
    for (
        family,
        layer,
        k,
        complex_mode,
        betti_target,
        token_selector,
        component,
        alpha,
    ), metric_values in sorted(delta_buckets.items()):
        item: Dict[str, Any] = {
            "family": family,
            "layer": layer,
            "k": k,
            "complex_mode": complex_mode,
            "betti_1_fraction_target": finite_float(betti_target) if betti_target else "",
            "token_selector": token_selector,
            "component": component,
            "baseline_component": baseline_component,
            "alpha": float(alpha),
            "n_pairs": counts[
                (family, layer, k, complex_mode, betti_target, token_selector, component, alpha)
            ],
        }
        for metric, values in sorted(metric_values.items()):
            item[f"{metric}_mean"] = mean_or_nan(values)
        out.append(item)
    return out


def build_layer_pairwise_summary(pairwise_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[int, int, str, str, str, str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    counts: Dict[Tuple[int, int, str, str, str, str, str], int] = defaultdict(int)
    for row in pairwise_rows:
        key = (
            int(float(row["layer"])),
            int(float(row["k"])),
            str(row.get("complex_mode", "legacy") or "legacy"),
            str(row.get("betti_1_fraction_target", "") or ""),
            str(row.get("token_selector", "max_component")),
            str(row["component"]),
            str(row["alpha"]),
        )
        counts[key] += int(row.get("n_pairs", 0) or 0)
        for metric, value in row.items():
            if not metric.endswith("_mean"):
                continue
            number = finite_float(value)
            if number is not None:
                buckets[key][metric].append(number)

    out: List[Dict[str, Any]] = []
    for (layer, k, complex_mode, betti_target, token_selector, component, alpha), metric_values in sorted(
        buckets.items()
    ):
        item: Dict[str, Any] = {
            "layer": layer,
            "k": k,
            "complex_mode": complex_mode,
            "betti_1_fraction_target": finite_float(betti_target) if betti_target else "",
            "token_selector": token_selector,
            "component": component,
            "alpha": float(alpha),
            "n_pairs": counts[(layer, k, complex_mode, betti_target, token_selector, component, alpha)],
            "n_family_rows": max((len(values) for values in metric_values.values()), default=0),
        }
        for metric, values in sorted(metric_values.items()):
            item[metric] = mean_or_nan(values)
        out.append(item)
    return out


def write_report(
    *,
    output: Path,
    rows: Sequence[Dict[str, Any]],
    component_summary: Sequence[Dict[str, Any]],
    pairwise_summary: Sequence[Dict[str, Any]],
    layer_pairwise_summary: Sequence[Dict[str, Any]],
) -> None:
    report_path = derived_path(output, "report", ".md")
    prompt_count = len({(row["family"], row["prompt_id"], row["layer"], row["k"]) for row in rows})
    lines = [
        "# HLTD Steering Suite Summary",
        "",
        "## Run",
        "",
        f"- steering rows: {len(rows)}",
        f"- prompt/layer/k runs: {prompt_count}",
        "",
        "## Component Means",
        "",
        "| family | layer | selector | component | alpha | n | seeds | tokens | active | KL | entropy delta | next-token delta | target mass delta | semantic margin | top changed |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in component_summary:
        lines.append(
            "| {family} | L{layer} | {selector} | {component} | {alpha} | {n} | {seeds} | {tokens} | {active} | {kl} | {entropy} | {next_delta} | {target_mass} | {semantic_margin} | {top_changed} |".format(
                family=row["family"],
                layer=int(row["layer"]),
                selector=row.get("token_selector", "max_component"),
                component=row["component"],
                alpha=fmt(row["alpha"]),
                n=int(row["n_rows"]),
                seeds=int(row.get("n_seeds", 0)),
                tokens=int(row.get("n_tokens", 0)),
                active=fmt(row.get("component_active_mean")),
                kl=fmt(row.get("kl_base_to_steered_mean")),
                entropy=fmt(row.get("entropy_delta_mean")),
                next_delta=fmt(row.get("next_token_logprob_delta_mean")),
                target_mass=fmt(row.get("target_logprob_mass_delta_mean")),
                semantic_margin=fmt(row.get("semantic_margin_delta_mean")),
                top_changed=fmt(row.get("top_changed_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Component Minus Random Tangent",
            "",
            "Only rows where both the component and random tangent are active are included.",
            "",
            "| family | layer | selector | component | alpha | n | KL delta | entropy delta | next-token delta | target mass delta | semantic margin delta | top-shift delta |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pairwise_summary:
        baseline = row["baseline_component"]
        lines.append(
            "| {family} | L{layer} | {selector} | {component} | {alpha} | {n} | {kl} | {entropy} | {next_delta} | {target_mass} | {semantic_margin} | {top_shift} |".format(
                family=row["family"],
                layer=int(row["layer"]),
                selector=row.get("token_selector", "max_component"),
                component=row["component"],
                alpha=fmt(row["alpha"]),
                n=int(row["n_pairs"]),
                kl=fmt(row.get(f"kl_base_to_steered_minus_{baseline}_mean")),
                entropy=fmt(row.get(f"entropy_delta_minus_{baseline}_mean")),
                next_delta=fmt(row.get(f"next_token_logprob_delta_minus_{baseline}_mean")),
                target_mass=fmt(row.get(f"target_logprob_mass_delta_minus_{baseline}_mean")),
                semantic_margin=fmt(row.get(f"semantic_margin_delta_minus_{baseline}_mean")),
                top_shift=fmt(row.get(f"top_shift_logprob_delta_minus_{baseline}_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Layer Pairwise Gate",
            "",
            "Averages family-level component-minus-random rows by layer and selector.",
            "",
            "| layer | selector | component | alpha | n | KL delta | entropy delta | next-token delta | target mass delta | semantic margin delta | top-shift delta |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in layer_pairwise_summary:
        baseline = "random_tangent"
        lines.append(
            "| L{layer} | {selector} | {component} | {alpha} | {n} | {kl} | {entropy} | {next_delta} | {target_mass} | {semantic_margin} | {top_shift} |".format(
                layer=int(row["layer"]),
                selector=row.get("token_selector", "max_component"),
                component=row["component"],
                alpha=fmt(row["alpha"]),
                n=int(row["n_pairs"]),
                kl=fmt(row.get(f"kl_base_to_steered_minus_{baseline}_mean")),
                entropy=fmt(row.get(f"entropy_delta_minus_{baseline}_mean")),
                next_delta=fmt(row.get(f"next_token_logprob_delta_minus_{baseline}_mean")),
                target_mass=fmt(row.get(f"target_logprob_mass_delta_minus_{baseline}_mean")),
                semantic_margin=fmt(row.get(f"semantic_margin_delta_minus_{baseline}_mean")),
                top_shift=fmt(row.get(f"top_shift_logprob_delta_minus_{baseline}_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Conservative Read",
            "",
            "This summary is a one-step causal gate. It measures immediate next-token distribution movement, not multi-step semantic drift or fluency preservation.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="spiral_out_hltd_steering_suite")
    parser.add_argument("--output", default=None)
    parser.add_argument("--baseline-component", default="random_tangent")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    output = Path(args.output) if args.output else run_root / "summary.csv"
    csv_paths = sorted(run_root.glob("*/steering_metrics.csv"))
    direct_csv = run_root / "steering_metrics.csv"
    if direct_csv.exists():
        csv_paths.insert(0, direct_csv)
    if not csv_paths:
        raise FileNotFoundError(f"No steering_metrics.csv files found under {run_root}")

    rows: List[Dict[str, Any]] = []
    for csv_path in csv_paths:
        rows.extend(build_run_rows(csv_path))

    component_summary = build_component_summary(rows)
    pairwise_summary = build_pairwise_summary(rows, baseline_component=args.baseline_component)
    layer_pairwise_summary = build_layer_pairwise_summary(pairwise_summary)

    write_csv(rows, output)
    write_csv(component_summary, derived_path(output, "component"))
    write_csv(pairwise_summary, derived_path(output, "pairwise"))
    write_csv(layer_pairwise_summary, derived_path(output, "layer_pairwise"))
    write_report(
        output=output,
        rows=rows,
        component_summary=component_summary,
        pairwise_summary=pairwise_summary,
        layer_pairwise_summary=layer_pairwise_summary,
    )
    print(f"summarized {len(csv_paths)} steering runs -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
