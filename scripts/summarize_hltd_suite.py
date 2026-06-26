#!/usr/bin/env python3
"""Summarize Spiral Hodge HLTD prompt-suite CSV outputs."""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RUN_RE = re.compile(r"(?P<family>.+)__(?P<prompt_id>.+)__k(?P<k>\d+)$")


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_run_dir(path: Path) -> Tuple[str, str, int]:
    match = RUN_RE.match(path.name)
    if not match:
        raise ValueError(f"Run directory does not match FAMILY__PROMPT__kK pattern: {path}")
    return match.group("family"), match.group("prompt_id"), int(match.group("k"))


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
    return mean(vals) if vals else float("nan")


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
    family, prompt_id, k = parse_run_dir(csv_path.parent)
    rows = read_rows(csv_path)
    real = rows_for(rows, "real")
    shuffle = rows_for(rows, "shuffle_tokens")
    random = rows_for(rows, "random_hidden")

    real_coexact_l5_l8 = mean_metric(real, "hltd_coexact_ratio", layer_min=5, layer_max=8)
    shuffle_coexact_l5_l8 = mean_metric(shuffle, "hltd_coexact_ratio", layer_min=5, layer_max=8)
    random_coexact_l5_l8 = mean_metric(random, "hltd_coexact_ratio", layer_min=5, layer_max=8)
    peak, peak_layer = peak_metric(real, "hltd_coexact_ratio")
    peak_mid, peak_mid_layer = peak_metric(real, "hltd_coexact_ratio", layer_min=5, layer_max=8)

    return {
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
        "random_graph_high_freq_l5_l8": mean_metric(random, "graph_high_freq_ratio", layer_min=5, layer_max=8),
        "real_hodge_curl_l5_l8": mean_metric(real, "hodge_curl_ratio", layer_min=5, layer_max=8),
        "shuffle_hodge_curl_l5_l8": mean_metric(shuffle, "hodge_curl_ratio", layer_min=5, layer_max=8),
        "random_hodge_curl_l5_l8": mean_metric(random, "hodge_curl_ratio", layer_min=5, layer_max=8),
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


def group_mean(rows: Sequence[Dict[str, Any]], key: str, metric: str) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for row in rows:
        value = finite_float(row.get(metric))
        if value is None:
            continue
        buckets.setdefault(str(row[key]), []).append(value)
    return {name: mean(vals) for name, vals in sorted(buckets.items()) if vals}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="spiral_out_hltd_suite")
    parser.add_argument("--output", default="spiral_out_hltd_suite/summary.csv")
    args = parser.parse_args(argv)

    root = Path(args.run_root)
    csv_paths = sorted(root.glob("*__*__k*/layer_metrics.csv"))
    if not csv_paths:
        raise SystemExit(f"No layer_metrics.csv files found under {root}")

    summaries = [build_run_summary(path) for path in csv_paths]
    output = Path(args.output)
    write_csv(summaries, output)

    print(f"wrote {output} ({len(summaries)} runs)")
    for metric in [
        "real_coexact_l5_l8",
        "real_minus_shuffle_coexact_l5_l8",
        "real_minus_random_coexact_l5_l8",
        "real_graph_high_freq_l5_l8",
        "real_harmonic_max",
    ]:
        print(metric)
        for family, value in group_mean(summaries, "family", metric).items():
            print(f"  {family}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
