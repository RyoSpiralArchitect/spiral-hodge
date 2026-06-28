#!/usr/bin/env python3
"""Summarize Spiral Hodge HLTD prompt-suite CSV outputs."""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple


RUN_RE = re.compile(r"(?P<family>.+)__(?P<prompt_id>.+)__k(?P<k>\d+)(?:__(?P<topology>.+))?$")
DEFAULT_TOPOLOGY = "triangles"
VARIANTS = ["real", "shuffle_tokens", "reverse_tokens", "random_hidden"]

RUN_SUMMARY_METRICS = [
    "real_coexact_mean",
    "real_coexact_l5_l8",
    "shuffle_coexact_l5_l8",
    "random_coexact_l5_l8",
    "real_minus_shuffle_coexact_l5_l8",
    "real_minus_random_coexact_l5_l8",
    "real_exact_l5_l8",
    "real_harmonic_max",
    "real_graph_high_freq_l5_l8",
    "shuffle_graph_high_freq_l5_l8",
    "random_graph_high_freq_l5_l8",
    "real_hodge_curl_l5_l8",
    "shuffle_hodge_curl_l5_l8",
    "random_hodge_curl_l5_l8",
    "real_coexact_peak",
    "real_coexact_mid_peak",
    "max_reverse_hltd_coexact_gap",
    "max_reverse_signed_trajectory_gap",
]

BOOTSTRAP_METRICS = [
    "real_coexact_l5_l8",
    "real_minus_shuffle_coexact_l5_l8",
    "real_minus_random_coexact_l5_l8",
    "real_exact_l5_l8",
    "real_harmonic_max",
    "max_reverse_hltd_coexact_gap",
]

LAYER_METRICS = [
    "hltd_coexact_ratio",
    "hltd_exact_ratio",
    "hltd_harmonic_ratio",
    "hltd_semantic_flow_ratio",
    "graph_high_freq_ratio",
    "hodge_curl_ratio",
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


def parse_run_dir(path: Path) -> Tuple[str, str, int, str]:
    match = RUN_RE.match(path.name)
    if not match:
        raise ValueError(f"Run directory does not match FAMILY__PROMPT__kK pattern: {path}")
    topology = match.group("topology") or DEFAULT_TOPOLOGY
    return match.group("family"), match.group("prompt_id"), int(match.group("k")), topology


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_for(rows: Sequence[Dict[str, str]], variant: str) -> List[Dict[str, str]]:
    return sorted(
        [row for row in rows if row.get("variant") == variant],
        key=lambda row: int(float(row.get("layer", "0"))),
    )


def metric_values(rows: Sequence[Dict[str, str]], metric: str, *, layer_min: int = 0, layer_max: int = 999) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for row in rows:
        layer = int(float(row.get("layer", "0")))
        if not (layer_min <= layer <= layer_max):
            continue
        value = finite_float(row.get(metric))
        if value is not None:
            out.append((layer, value))
    return out


def mean_metric(rows: Sequence[Dict[str, str]], metric: str, *, layer_min: int = 0, layer_max: int = 999) -> float:
    vals = [value for _, value in metric_values(rows, metric, layer_min=layer_min, layer_max=layer_max)]
    return mean_or_nan(vals)


def peak_metric(rows: Sequence[Dict[str, str]], metric: str, *, layer_min: int = 0, layer_max: int = 999) -> Tuple[float, int]:
    vals = metric_values(rows, metric, layer_min=layer_min, layer_max=layer_max)
    if not vals:
        return float("nan"), -1
    layer, value = max(vals, key=lambda item: item[1])
    return value, layer


def max_reverse_unsigned_gap(rows: Sequence[Dict[str, str]], metric: str) -> float:
    real = {int(float(row["layer"])): row for row in rows_for(rows, "real")}
    rev = {int(float(row["layer"])): row for row in rows_for(rows, "reverse_tokens")}
    gaps: List[float] = []
    for layer, row in real.items():
        other = rev.get(layer)
        if other is None:
            continue
        a = finite_float(row.get(metric))
        b = finite_float(other.get(metric))
        if a is not None and b is not None:
            gaps.append(abs(a - b))
    return max(gaps) if gaps else float("nan")


def max_reverse_signed_gap(rows: Sequence[Dict[str, str]], metric: str) -> float:
    real = {int(float(row["layer"])): row for row in rows_for(rows, "real")}
    rev = {int(float(row["layer"])): row for row in rows_for(rows, "reverse_tokens")}
    gaps: List[float] = []
    for layer, row in real.items():
        other = rev.get(layer)
        if other is None:
            continue
        a = finite_float(row.get(metric))
        b = finite_float(other.get(metric))
        if a is not None and b is not None:
            gaps.append(abs(a + b))
    return max(gaps) if gaps else float("nan")


def build_run_summary(csv_path: Path) -> Dict[str, Any]:
    family, prompt_id, k, topology = parse_run_dir(csv_path.parent)
    rows = read_rows(csv_path)
    real = rows_for(rows, "real")
    shuffle = rows_for(rows, "shuffle_tokens")
    random_rows = rows_for(rows, "random_hidden")

    real_coexact_l5_l8 = mean_metric(real, "hltd_coexact_ratio", layer_min=5, layer_max=8)
    shuffle_coexact_l5_l8 = mean_metric(shuffle, "hltd_coexact_ratio", layer_min=5, layer_max=8)
    random_coexact_l5_l8 = mean_metric(random_rows, "hltd_coexact_ratio", layer_min=5, layer_max=8)
    peak, peak_layer = peak_metric(real, "hltd_coexact_ratio")
    peak_mid, peak_mid_layer = peak_metric(real, "hltd_coexact_ratio", layer_min=5, layer_max=8)

    return {
        "topology": topology,
        "family": family,
        "prompt_id": prompt_id,
        "k": k,
        "real_coexact_mean": mean_metric(real, "hltd_coexact_ratio"),
        "real_coexact_l5_l8": real_coexact_l5_l8,
        "shuffle_coexact_l5_l8": shuffle_coexact_l5_l8,
        "random_coexact_l5_l8": random_coexact_l5_l8,
        "real_minus_shuffle_coexact_l5_l8": real_coexact_l5_l8 - shuffle_coexact_l5_l8,
        "real_minus_random_coexact_l5_l8": real_coexact_l5_l8 - random_coexact_l5_l8,
        "real_exact_l5_l8": mean_metric(real, "hltd_exact_ratio", layer_min=5, layer_max=8),
        "real_harmonic_max": peak_metric(real, "hltd_harmonic_ratio")[0],
        "real_graph_high_freq_l5_l8": mean_metric(real, "graph_high_freq_ratio", layer_min=5, layer_max=8),
        "shuffle_graph_high_freq_l5_l8": mean_metric(shuffle, "graph_high_freq_ratio", layer_min=5, layer_max=8),
        "random_graph_high_freq_l5_l8": mean_metric(random_rows, "graph_high_freq_ratio", layer_min=5, layer_max=8),
        "real_hodge_curl_l5_l8": mean_metric(real, "hodge_curl_ratio", layer_min=5, layer_max=8),
        "shuffle_hodge_curl_l5_l8": mean_metric(shuffle, "hodge_curl_ratio", layer_min=5, layer_max=8),
        "random_hodge_curl_l5_l8": mean_metric(random_rows, "hodge_curl_ratio", layer_min=5, layer_max=8),
        "real_coexact_peak": peak,
        "real_coexact_peak_layer": peak_layer,
        "real_coexact_mid_peak": peak_mid,
        "real_coexact_mid_peak_layer": peak_mid_layer,
        "max_reverse_hltd_coexact_gap": max_reverse_unsigned_gap(rows, "hltd_coexact_ratio"),
        "max_reverse_signed_trajectory_gap": max_reverse_signed_gap(rows, "trajectory_signed_circulation_alignment"),
    }


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def derived_path(output: Path, label: str, suffix: str = ".csv") -> Path:
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}_{label}{suffix}")


def group_mean(rows: Sequence[Dict[str, Any]], keys: Sequence[str], metric: str) -> Dict[Tuple[str, ...], float]:
    buckets: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    for row in rows:
        value = finite_float(row.get(metric))
        if value is None:
            continue
        group_key = tuple(str(row[key]) for key in keys)
        buckets[group_key].append(value)
    return {name: mean(vals) for name, vals in sorted(buckets.items()) if vals}


def build_family_k_summary(summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        buckets[(str(row["topology"]), str(row["family"]), int(row["k"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (topology, family, k), rows in sorted(buckets.items()):
        item: Dict[str, Any] = {
            "topology": topology,
            "family": family,
            "k": k,
            "n_runs": len(rows),
        }
        for metric in RUN_SUMMARY_METRICS:
            vals = [value for value in (finite_float(row.get(metric)) for row in rows) if value is not None]
            item[f"{metric}_mean"] = mean_or_nan(vals)
        harmonic_vals = [value for value in (finite_float(row.get("real_harmonic_max")) for row in rows) if value is not None]
        item["real_harmonic_max_group_max"] = max(harmonic_vals) if harmonic_vals else float("nan")
        out.append(item)
    return out


def build_layer_summary(csv_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, int, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[Tuple[str, str, int, int], int] = defaultdict(int)

    for csv_path in csv_paths:
        family, _prompt_id, k, topology = parse_run_dir(csv_path.parent)
        seen_layers: set[int] = set()
        for row in read_rows(csv_path):
            layer = int(float(row.get("layer", "0")))
            variant = str(row.get("variant", ""))
            if variant not in VARIANTS:
                continue
            key = (topology, family, k, layer)
            seen_layers.add(layer)
            for metric in LAYER_METRICS:
                value = finite_float(row.get(metric))
                if value is not None:
                    buckets[key][f"{variant}_{metric}"].append(value)
        for layer in seen_layers:
            counts[(topology, family, k, layer)] += 1

    out: List[Dict[str, Any]] = []
    for (topology, family, k, layer), metric_buckets in sorted(buckets.items()):
        item: Dict[str, Any] = {
            "topology": topology,
            "family": family,
            "k": k,
            "layer": layer,
            "n_runs": counts[(topology, family, k, layer)],
        }
        for variant in VARIANTS:
            for metric in LAYER_METRICS:
                field = f"{variant}_{metric}"
                item[field] = mean_or_nan(metric_buckets.get(field, []))
        real = item.get("real_hltd_coexact_ratio")
        shuffle = item.get("shuffle_tokens_hltd_coexact_ratio")
        random_rows = item.get("random_hidden_hltd_coexact_ratio")
        reverse = item.get("reverse_tokens_hltd_coexact_ratio")
        item["real_minus_shuffle_hltd_coexact_ratio"] = real - shuffle if finite_float(real) is not None and finite_float(shuffle) is not None else float("nan")
        item["real_minus_random_hltd_coexact_ratio"] = real - random_rows if finite_float(real) is not None and finite_float(random_rows) is not None else float("nan")
        item["real_minus_reverse_abs_hltd_coexact_ratio"] = abs(real - reverse) if finite_float(real) is not None and finite_float(reverse) is not None else float("nan")
        out.append(item)
    return out


def build_prompt_summary(summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        buckets[(str(row["topology"]), str(row["family"]), str(row["prompt_id"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (topology, family, prompt_id), rows in sorted(buckets.items()):
        item: Dict[str, Any] = {
            "topology": topology,
            "family": family,
            "prompt_id": prompt_id,
            "k_count": len({int(row["k"]) for row in rows}),
            "k_values": " ".join(str(k) for k in sorted({int(row["k"]) for row in rows})),
        }
        for metric in BOOTSTRAP_METRICS:
            vals = [value for value in (finite_float(row.get(metric)) for row in rows) if value is not None]
            item[metric] = mean_or_nan(vals)
        out.append(item)
    return out


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = pct * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bootstrap_mean_ci(values: Sequence[float], *, samples: int, seed: int) -> Tuple[float, float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    boots: List[float] = []
    n = len(vals)
    for _ in range(samples):
        boots.append(mean(vals[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    return mean(vals), percentile(boots, 0.025), percentile(boots, 0.975)


def build_bootstrap_summary(prompt_rows: Sequence[Dict[str, Any]], *, samples: int, seed: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for topology in sorted({str(row["topology"]) for row in prompt_rows}):
        for family in sorted({str(row["family"]) for row in prompt_rows if str(row["topology"]) == topology}):
            rows = [row for row in prompt_rows if str(row["topology"]) == topology and str(row["family"]) == family]
            for offset, metric in enumerate(BOOTSTRAP_METRICS):
                vals = [value for value in (finite_float(row.get(metric)) for row in rows) if value is not None]
                center, lo, hi = bootstrap_mean_ci(vals, samples=samples, seed=seed + offset)
                out.append(
                    {
                        "topology": topology,
                        "family": family,
                        "metric": metric,
                        "n_prompts": len(vals),
                        "mean": center,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    return out


def build_family_gap_summary(prompt_rows: Sequence[Dict[str, Any]], *, samples: int, seed: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    topologies = sorted({str(row["topology"]) for row in prompt_rows})
    for topology in topologies:
        families = sorted({str(row["family"]) for row in prompt_rows if str(row["topology"]) == topology})
        by_family = {
            family: [row for row in prompt_rows if str(row["topology"]) == topology and str(row["family"]) == family]
            for family in families
        }
        for metric_idx, metric in enumerate(["real_coexact_l5_l8", "real_minus_shuffle_coexact_l5_l8"]):
            for pair_idx, (family_a, family_b) in enumerate(itertools.combinations(families, 2)):
                vals_a = [value for value in (finite_float(row.get(metric)) for row in by_family[family_a]) if value is not None]
                vals_b = [value for value in (finite_float(row.get(metric)) for row in by_family[family_b]) if value is not None]
                if not vals_a or not vals_b:
                    continue
                rng = random.Random(seed + 1000 + metric_idx * 100 + pair_idx)
                diffs: List[float] = []
                for _ in range(samples):
                    mean_a = mean(vals_a[rng.randrange(len(vals_a))] for _ in range(len(vals_a)))
                    mean_b = mean(vals_b[rng.randrange(len(vals_b))] for _ in range(len(vals_b)))
                    diffs.append(mean_a - mean_b)
                diffs.sort()
                out.append(
                    {
                        "topology": topology,
                        "metric": metric,
                        "family_a": family_a,
                        "family_b": family_b,
                        "diff_mean": mean(vals_a) - mean(vals_b),
                        "ci_low": percentile(diffs, 0.025),
                        "ci_high": percentile(diffs, 0.975),
                        "n_prompts_a": len(vals_a),
                        "n_prompts_b": len(vals_b),
                    }
                )
    return out


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def family_k_markdown(family_k_rows: Sequence[Dict[str, Any]], metric: str, topology: str) -> str:
    rows = [row for row in family_k_rows if str(row["topology"]) == topology]
    families = sorted({str(row["family"]) for row in rows})
    ks = sorted({int(row["k"]) for row in rows})
    table_rows: List[List[str]] = []
    for family in families:
        values = []
        for k in ks:
            match = [row for row in rows if str(row["family"]) == family and int(row["k"]) == k]
            values.append(fmt(match[0].get(f"{metric}_mean")) if match else "nan")
        table_rows.append([family, *values])
    return markdown_table(["family", *[f"k={k}" for k in ks]], table_rows)


def top_layer_rows(layer_rows: Sequence[Dict[str, Any]], *, topology: str, metric: str, limit: int = 8) -> List[Dict[str, Any]]:
    rows = [row for row in layer_rows if str(row["topology"]) == topology]
    buckets: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        value = finite_float(row.get(metric))
        if value is not None:
            buckets[int(row["layer"])].append(value)
    out = [{"layer": layer, "mean": mean(vals)} for layer, vals in buckets.items() if vals]
    return sorted(out, key=lambda row: row["mean"], reverse=True)[:limit]


def write_markdown_report(
    path: Path,
    *,
    summaries: Sequence[Dict[str, Any]],
    family_k_rows: Sequence[Dict[str, Any]],
    layer_rows: Sequence[Dict[str, Any]],
    bootstrap_rows: Sequence[Dict[str, Any]],
    gap_rows: Sequence[Dict[str, Any]],
    outputs: Dict[str, Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    topologies = sorted({str(row["topology"]) for row in summaries})
    lines: List[str] = [
        "# HLTD Suite Summary",
        "",
        f"Runs summarized: {len(summaries)}",
        "",
        "## Output Files",
        "",
    ]
    for label, output in outputs.items():
        lines.append(f"- `{label}`: `{output}`")
    for topology in topologies:
        lines.extend(["", f"## Topology: `{topology}`", "", "### Family x k: real coexact L5-L8", ""])
        lines.append(family_k_markdown(family_k_rows, "real_coexact_l5_l8", topology))
        lines.extend(["", "### Family x k: real minus shuffle coexact L5-L8", ""])
        lines.append(family_k_markdown(family_k_rows, "real_minus_shuffle_coexact_l5_l8", topology))
        lines.extend(["", "### Top all-family layers", ""])
        layer_table = [
            [f"L{row['layer']}", fmt(row["mean"])]
            for row in top_layer_rows(layer_rows, topology=topology, metric="real_hltd_coexact_ratio", limit=6)
        ]
        lines.append(markdown_table(["layer", "real coexact mean"], layer_table))
        lines.extend(["", "### Top real-minus-shuffle layers", ""])
        delta_table = [
            [f"L{row['layer']}", fmt(row["mean"])]
            for row in top_layer_rows(layer_rows, topology=topology, metric="real_minus_shuffle_hltd_coexact_ratio", limit=6)
        ]
        lines.append(markdown_table(["layer", "real - shuffle coexact"], delta_table))
        lines.extend(["", "### Bootstrap CI by family", ""])
        boot_table = [
            [row["family"], row["metric"], fmt(row["mean"]), fmt(row["ci_low"]), fmt(row["ci_high"])]
            for row in bootstrap_rows
            if str(row["topology"]) == topology and row["metric"] in {"real_coexact_l5_l8", "real_minus_shuffle_coexact_l5_l8"}
        ]
        lines.append(markdown_table(["family", "metric", "mean", "ci_low", "ci_high"], boot_table))
        lines.extend(["", "### Pairwise family gaps", ""])
        gap_table = [
            [row["metric"], f"{row['family_a']} - {row['family_b']}", fmt(row["diff_mean"]), fmt(row["ci_low"]), fmt(row["ci_high"])]
            for row in gap_rows
            if str(row["topology"]) == topology
        ]
        lines.append(markdown_table(["metric", "gap", "diff", "ci_low", "ci_high"], gap_table))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_family_metric_table(rows: Sequence[Dict[str, Any]], metric: str) -> None:
    topologies = sorted({str(row["topology"]) for row in rows})
    for topology in topologies:
        scoped = [row for row in rows if str(row["topology"]) == topology]
        families = sorted({str(row["family"]) for row in scoped})
        ks = sorted({int(row["k"]) for row in scoped})
        print(f"{metric} [{topology}]")
        print("  " + "family".ljust(20) + "".join(f"k={k}".rjust(10) for k in ks))
        for family in families:
            bits = []
            for k in ks:
                match = [row for row in scoped if str(row["family"]) == family and int(row["k"]) == k]
                bits.append(fmt(match[0].get(f"{metric}_mean")) if match else "nan")
            print("  " + family.ljust(20) + "".join(bit.rjust(10) for bit in bits))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="spiral_out_hltd_suite")
    parser.add_argument("--output", default="spiral_out_hltd_suite/summary.csv")
    parser.add_argument("--family-k-output", default=None)
    parser.add_argument("--layer-output", default=None)
    parser.add_argument("--prompt-output", default=None)
    parser.add_argument("--bootstrap-output", default=None)
    parser.add_argument("--family-gap-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--no-markdown", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.run_root)
    csv_paths = sorted(root.glob("*__*__k*/layer_metrics.csv"))
    if not csv_paths:
        raise SystemExit(f"No layer_metrics.csv files found under {root}")

    output = Path(args.output)
    outputs = {
        "summary": output,
        "family_k": Path(args.family_k_output) if args.family_k_output else derived_path(output, "family_k"),
        "layer": Path(args.layer_output) if args.layer_output else derived_path(output, "layer"),
        "prompt": Path(args.prompt_output) if args.prompt_output else derived_path(output, "prompt"),
        "bootstrap": Path(args.bootstrap_output) if args.bootstrap_output else derived_path(output, "bootstrap"),
        "family_gaps": Path(args.family_gap_output) if args.family_gap_output else derived_path(output, "family_gaps"),
    }
    markdown_output = Path(args.markdown_output) if args.markdown_output else derived_path(output, "report", ".md")

    summaries = [build_run_summary(path) for path in csv_paths]
    family_k_rows = build_family_k_summary(summaries)
    layer_rows = build_layer_summary(csv_paths)
    prompt_rows = build_prompt_summary(summaries)
    bootstrap_rows = build_bootstrap_summary(prompt_rows, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    family_gap_rows = build_family_gap_summary(prompt_rows, samples=args.bootstrap_samples, seed=args.bootstrap_seed)

    write_csv(summaries, outputs["summary"])
    write_csv(family_k_rows, outputs["family_k"])
    write_csv(layer_rows, outputs["layer"])
    write_csv(prompt_rows, outputs["prompt"])
    write_csv(bootstrap_rows, outputs["bootstrap"])
    write_csv(family_gap_rows, outputs["family_gaps"])
    if not args.no_markdown:
        write_markdown_report(
            markdown_output,
            summaries=summaries,
            family_k_rows=family_k_rows,
            layer_rows=layer_rows,
            bootstrap_rows=bootstrap_rows,
            gap_rows=family_gap_rows,
            outputs={**outputs, "markdown": markdown_output},
        )

    print(f"wrote {outputs['summary']} ({len(summaries)} runs)")
    for label, path in outputs.items():
        if label != "summary":
            print(f"wrote {path}")
    if not args.no_markdown:
        print(f"wrote {markdown_output}")
    print_family_metric_table(family_k_rows, "real_coexact_l5_l8")
    print_family_metric_table(family_k_rows, "real_minus_shuffle_coexact_l5_l8")
    for topology in sorted({str(row["topology"]) for row in layer_rows}):
        top_layers = top_layer_rows(layer_rows, topology=topology, metric="real_hltd_coexact_ratio", limit=5)
        print(f"top real coexact layers [{topology}]: " + ", ".join(f"L{row['layer']}={row['mean']:.4f}" for row in top_layers))
        top_delta_layers = top_layer_rows(layer_rows, topology=topology, metric="real_minus_shuffle_hltd_coexact_ratio", limit=5)
        print(
            f"top real-minus-shuffle layers [{topology}]: "
            + ", ".join(f"L{row['layer']}={row['mean']:.4f}" for row in top_delta_layers)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
