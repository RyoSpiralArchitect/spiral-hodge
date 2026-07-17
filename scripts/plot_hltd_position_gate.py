#!/usr/bin/env python3
"""Plot HLTD all-interior position-gate summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CORE_COMPONENTS = [
    "coexact",
    "coexact_minus_presence",
    "presence",
    "presence_plus_coexact",
]

COMPONENT_LABELS = {
    "coexact": "coexact",
    "coexact_minus_presence": "coexact - presence",
    "presence": "presence",
    "presence_plus_coexact": "presence + coexact",
}

METRIC_LABELS = {
    "next_token_delta_mean": "next-token logprob delta vs random",
    "probe_label_margin_delta_mean": "probe label-margin delta vs random",
}

COMPONENT_COLORS = {
    "coexact": "#2878b5",
    "coexact_minus_presence": "#d65f5f",
    "presence": "#4c9a2a",
    "presence_plus_coexact": "#8064a2",
}


def finite_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def symmetric_limit(values: Sequence[float], *, quantile: float = 0.98) -> float:
    vals = np.abs(finite_values(values))
    if vals.size == 0:
        return 1.0
    limit = float(np.quantile(vals, quantile))
    return max(limit, 1e-6)


def load_tables(summary_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    joined_path = summary_root / "joined_position_summary.csv"
    peak_path = summary_root / "position_cross_family_peak_summary.csv"
    if not joined_path.exists():
        raise FileNotFoundError(joined_path)
    if not peak_path.exists():
        raise FileNotFoundError(peak_path)
    return pd.read_csv(joined_path), pd.read_csv(peak_path)


def metric_grid(
    joined: pd.DataFrame,
    *,
    probe: str,
    component: str,
    k: int,
    metric: str,
) -> Tuple[np.ndarray, List[int], List[int]]:
    subset = joined[
        (joined["probe"] == probe)
        & (joined["component"] == component)
        & (joined["k"].astype(int) == int(k))
    ]
    layers = sorted(int(v) for v in subset["layer"].dropna().unique())
    bins = sorted(int(v) for v in subset["position_bin"].dropna().unique())
    grid = np.full((len(layers), len(bins)), np.nan, dtype=float)
    if not layers or not bins:
        return grid, layers, bins
    grouped = subset.groupby(["layer", "position_bin"], as_index=False)[metric].mean()
    layer_to_idx = {layer: idx for idx, layer in enumerate(layers)}
    bin_to_idx = {bin_index: idx for idx, bin_index in enumerate(bins)}
    for row in grouped.itertuples(index=False):
        layer = int(getattr(row, "layer"))
        bin_index = int(getattr(row, "position_bin"))
        grid[layer_to_idx[layer], bin_to_idx[bin_index]] = float(getattr(row, metric))
    return grid, layers, bins


def plot_metric_heatmaps(
    joined: pd.DataFrame,
    *,
    output_path: Path,
    probe: str,
    metric: str,
    components: Sequence[str],
) -> None:
    ks = sorted(int(v) for v in joined["k"].dropna().unique())
    values = joined[
        (joined["probe"] == probe) & (joined["component"].isin(components))
    ][metric].astype(float)
    limit = symmetric_limit(values)

    fig, axes = plt.subplots(
        len(components),
        len(ks),
        figsize=(3.6 * max(len(ks), 1), 2.4 * max(len(components), 1)),
        squeeze=False,
        constrained_layout=True,
    )
    last_image = None
    for row_idx, component in enumerate(components):
        for col_idx, k in enumerate(ks):
            ax = axes[row_idx][col_idx]
            grid, layers, bins = metric_grid(
                joined,
                probe=probe,
                component=component,
                k=k,
                metric=metric,
            )
            if grid.size:
                last_image = ax.imshow(
                    grid,
                    aspect="auto",
                    origin="lower",
                    cmap="coolwarm",
                    vmin=-limit,
                    vmax=limit,
                )
            ax.set_title(f"k={k}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(COMPONENT_LABELS.get(component, component), fontsize=9)
            if row_idx == len(components) - 1:
                ax.set_xlabel("position bin", fontsize=9)
            if layers:
                ax.set_yticks(range(len(layers)), [f"L{layer}" for layer in layers], fontsize=8)
            if bins:
                tick_positions = sorted(set([0, len(bins) // 2, len(bins) - 1]))
                ax.set_xticks(tick_positions, [str(bins[pos]) for pos in tick_positions], fontsize=8)
            ax.tick_params(length=0)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle(f"{probe}: {METRIC_LABELS.get(metric, metric)}", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_peak_bars(
    peaks: pd.DataFrame,
    *,
    output_path: Path,
    probe: str,
    components: Sequence[str],
) -> None:
    metrics = ["next_token_delta_mean", "probe_label_margin_delta_mean"]
    ks = sorted(int(v) for v in peaks["k"].dropna().unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.2), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    width = 0.8 / max(len(components), 1)
    x = np.arange(len(ks), dtype=float)
    for ax, metric in zip(axes, metrics):
        for idx, component in enumerate(components):
            subset = peaks[
                (peaks["probe"] == probe)
                & (peaks["metric"] == metric)
                & (peaks["component"] == component)
            ]
            values = []
            for k in ks:
                row = subset[subset["k"].astype(int) == k]
                values.append(float(row["peak_value_mean"].iloc[0]) if not row.empty else np.nan)
            offset = (idx - (len(components) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=COMPONENT_LABELS.get(component, component),
                color=COMPONENT_COLORS.get(component),
            )
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.set_xticks(x, [f"k={k}" for k in ks])
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_ylabel("cross-family peak mean")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle(f"{probe}: cross-family peak strength", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_peak_phase(
    peaks: pd.DataFrame,
    *,
    output_path: Path,
    probe: str,
    components: Sequence[str],
) -> None:
    metrics = ["next_token_delta_mean", "probe_label_margin_delta_mean"]
    markers = {12: "o", 16: "s", 24: "^"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.4), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        subset = peaks[
            (peaks["probe"] == probe)
            & (peaks["metric"] == metric)
            & (peaks["component"].isin(components))
        ]
        for row in subset.itertuples(index=False):
            component = str(row.component)
            k = int(row.k)
            x = float(row.peak_position_frac_mean)
            y = float(row.peak_value_mean)
            ax.scatter(
                x,
                y,
                s=72,
                marker=markers.get(k, "o"),
                color=COMPONENT_COLORS.get(component),
                edgecolor="white",
                linewidth=0.7,
            )
            ax.text(
                x + 0.012,
                y,
                f"k{k} L{int(row.peak_layer)} b{int(row.peak_position_bin)}",
                fontsize=7,
                va="center",
            )
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("peak normalized position")
        ax.set_ylabel("cross-family peak mean")
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.grid(alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COMPONENT_COLORS.get(component), label=COMPONENT_LABELS.get(component, component), markersize=8)
        for component in components
    ]
    axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle(f"{probe}: peak position phase", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_plots(
    *,
    summary_root: Path,
    output_dir: Path,
    probe: str,
    components: Sequence[str],
) -> List[Path]:
    joined, peaks = load_tables(summary_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = [
        output_dir / f"{probe}_next_token_heatmap.png",
        output_dir / f"{probe}_probe_margin_heatmap.png",
        output_dir / f"{probe}_peak_bars.png",
        output_dir / f"{probe}_peak_phase.png",
    ]
    plot_metric_heatmaps(
        joined,
        output_path=saved[0],
        probe=probe,
        metric="next_token_delta_mean",
        components=components,
    )
    plot_metric_heatmaps(
        joined,
        output_path=saved[1],
        probe=probe,
        metric="probe_label_margin_delta_mean",
        components=components,
    )
    plot_peak_bars(peaks, output_path=saved[2], probe=probe, components=components)
    plot_peak_phase(peaks, output_path=saved[3], probe=probe, components=components)
    manifest = {
        "summary_root": str(summary_root),
        "probe": probe,
        "components": list(components),
        "plots": [str(path) for path in saved],
    }
    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return saved + [manifest_path]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--probe", default="ontology_collapse")
    parser.add_argument("--components", nargs="+", default=CORE_COMPONENTS)
    args = parser.parse_args(argv)

    summary_root = Path(args.summary_root)
    output_dir = Path(args.output_dir) if args.output_dir else summary_root / "plots"
    saved = build_plots(
        summary_root=summary_root,
        output_dir=output_dir,
        probe=args.probe,
        components=args.components,
    )
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
