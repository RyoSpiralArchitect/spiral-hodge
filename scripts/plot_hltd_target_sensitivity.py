#!/usr/bin/env python3
"""Plot closed-loop HLTD target-vocabulary sensitivity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_COMPONENT = "random_tangent"


def parse_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"source label is empty: {value!r}")
    return label, Path(path)


def load_prompt_layer_k_rows(
    *,
    label: str,
    root: Path,
    source_order: int,
    prompt_id: str,
    layer: int,
    k: int,
) -> pd.DataFrame:
    path = root / "closed_loop_prompt_layer_k_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    subset = table[
        (table["prompt_id"].astype(str) == str(prompt_id))
        & (table["layer"].astype(int) == int(layer))
        & (table["k"].astype(int) == int(k))
    ].copy()
    if subset.empty:
        raise ValueError(f"{root}: no rows for prompt_id={prompt_id} layer={layer} k={k}")
    if "target_set" not in subset.columns:
        subset["target_set"] = ""
    subset["target_set"] = subset["target_set"].fillna("").astype(str)
    subset.loc[subset["target_set"] == "", "target_set"] = label
    subset["source_label"] = label
    subset["source_order"] = int(source_order)
    subset["source_root"] = str(root)
    return subset


def collect_rows(
    *,
    sources: Sequence[Tuple[str, Path]],
    prompt_id: str,
    layer: int,
    k: int,
) -> pd.DataFrame:
    tables = [
        load_prompt_layer_k_rows(
            label=label,
            root=root,
            source_order=source_order,
            prompt_id=prompt_id,
            layer=layer,
            k=k,
        )
        for source_order, (label, root) in enumerate(sources)
    ]
    return pd.concat(tables, ignore_index=True, sort=False)


def _matched_random_metric(rows: pd.DataFrame, metric: str, minus_random_metric: str) -> pd.Series:
    if metric not in rows.columns or minus_random_metric not in rows.columns:
        return pd.Series([float("nan")] * len(rows), index=rows.index)
    return rows[metric].astype(float) - rows[minus_random_metric].astype(float)


def plot_target_sensitivity(
    rows: pd.DataFrame,
    *,
    output_path: Path,
    component: str,
    prompt_id: str,
    layer: int,
    k: int,
) -> None:
    branch = rows[rows["component"].astype(str) == str(component)].copy()
    if branch.empty:
        raise ValueError(f"No rows found for component={component!r}")
    branch = branch.sort_values(["source_order", "target_set", "source_label"]).reset_index(drop=True)
    labels = branch["target_set"].astype(str).tolist()
    xs = list(range(len(labels)))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), constrained_layout=True)
    fig.suptitle(f"HLTD target sensitivity: {prompt_id} L{layer} k{k} {component}", fontsize=14)

    width = 0.26
    axes[0].bar(
        [x - width for x in xs],
        branch["branch_gate_rate"].astype(float),
        width=width,
        label="branch gate",
        color="#516b8b",
    )
    axes[0].bar(
        xs,
        branch["branch_specific_gate_rate"].astype(float),
        width=width,
        label="specific gate",
        color="#7aa36f",
    )
    axes[0].bar(
        [x + width for x in xs],
        branch["random_branch_gate_rate"].astype(float),
        width=width,
        label="matched random gate",
        color="#b8b8b8",
    )
    axes[0].set_title("gate rates")
    axes[0].set_ylim(0, 1.08)
    axes[0].legend(frameon=False, fontsize=8)

    branch_margin = branch["mean_target_margin_delta_mean"].astype(float)
    random_margin = _matched_random_metric(
        branch,
        "mean_target_margin_delta_mean",
        "mean_target_margin_delta_minus_random_mean",
    )
    axes[1].bar(xs, branch_margin, color="#9b5f73", label="branch")
    axes[1].scatter(xs, random_margin, color="black", marker="x", label="matched random")
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_title("target margin delta")
    axes[1].legend(frameon=False, fontsize=8)

    branch_drift = branch["token_drift_rate_mean"].astype(float)
    random_drift = _matched_random_metric(
        branch,
        "token_drift_rate_mean",
        "token_drift_rate_minus_random_mean",
    )
    axes[2].bar(xs, branch_drift, color="#786fa6", label="branch")
    axes[2].scatter(xs, random_drift, color="black", marker="x", label="matched random")
    axes[2].set_title("token drift rate")
    drift_max = max(0.6, float(max(branch_drift.max(), random_drift.max(skipna=True))) + 0.1)
    axes[2].set_ylim(0, drift_max)
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xticks(xs, labels)
        ax.tick_params(axis="x", rotation=18)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax.grid(axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_target_sensitivity(
    *,
    sources: Sequence[Tuple[str, Path]],
    output_root: Path,
    prompt_id: str,
    layer: int,
    k: int,
    component: str,
    output_prefix: Optional[str] = None,
) -> Dict[str, str]:
    rows = collect_rows(sources=sources, prompt_id=prompt_id, layer=layer, k=k)
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or f"target_sensitivity_{prompt_id}_l{layer}_k{k}"
    csv_path = output_root / f"{prefix}.csv"
    png_path = output_root / f"{prefix}.png"
    manifest_path = output_root / f"{prefix}_manifest.json"
    rows.to_csv(csv_path, index=False)
    plot_target_sensitivity(
        rows,
        output_path=png_path,
        component=component,
        prompt_id=prompt_id,
        layer=layer,
        k=k,
    )
    manifest = {
        "prompt_id": prompt_id,
        "layer": int(layer),
        "k": int(k),
        "component": component,
        "sources": [{"label": label, "root": str(root)} for label, root in sources],
        "csv": str(csv_path),
        "plot": str(png_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "plot": str(png_path), "manifest": str(manifest_path)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="label=summary_root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--component", default="negative_coexact")
    parser.add_argument("--output-prefix")
    args = parser.parse_args(argv)

    saved = build_target_sensitivity(
        sources=[parse_source(value) for value in args.source],
        output_root=Path(args.output_root),
        prompt_id=args.prompt_id,
        layer=args.layer,
        k=args.k,
        component=args.component,
        output_prefix=args.output_prefix,
    )
    for path in saved.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
