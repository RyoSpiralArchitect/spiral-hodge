#!/usr/bin/env python3
"""Plot HLTD branch-band follow-up result summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMPONENT_COLORS = {
    "coexact": "#2878b5",
    "coexact_minus_presence": "#d65f5f",
    "presence": "#4c9a2a",
    "presence_plus_coexact": "#8064a2",
    "negative_coexact": "#6b6b6b",
}

COMPACT_COMPONENT_LABELS = {
    "coexact": "coexact",
    "coexact_minus_presence": "coex-pres",
    "presence": "presence",
    "presence_plus_coexact": "pres+coex",
    "negative_coexact": "-coexact",
}

RESULT_COLORS = {
    "causal_support_confirmed": "#2c7a4b",
    "gate_without_target_advantage": "#d17a22",
    "target_advantage_without_gate": "#7f6bb2",
    "not_confirmed": "#9b5a5a",
    "missing_run": "#b8bdc7",
    "missing_summary": "#d0a85f",
}


def finite_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def symmetric_limit(values: Sequence[float], *, quantile: float = 0.98) -> float:
    vals = np.abs(finite_values(values))
    if vals.size == 0:
        return 1.0
    return max(float(np.quantile(vals, quantile)), 1e-6)


def display_optional_text(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and np.isnan(value):
        return "-"
    text = str(value)
    return "-" if text == "" or text.lower() == "nan" else text


def display_number_text(value: object, *, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def load_tables(result_root: Path) -> Dict[str, pd.DataFrame]:
    result_path = result_root / "branch_band_result_scoreboard.csv"
    if not result_path.exists() or result_path.stat().st_size == 0:
        raise FileNotFoundError(result_path)
    tables: Dict[str, pd.DataFrame] = {"result_scoreboard": pd.read_csv(result_path)}
    layer_path = result_root / "branch_band_layer_result_summary.csv"
    if layer_path.exists() and layer_path.stat().st_size > 0:
        tables["layer_summary"] = pd.read_csv(layer_path)
    return tables


def row_label(row: object) -> str:
    family = str(getattr(row, "family")).replace("_", " ")
    component = COMPACT_COMPONENT_LABELS.get(str(getattr(row, "component")), str(getattr(row, "component")))
    return f"{family}\n{component}"


def compact_result_label(value: object) -> str:
    label = str(value)
    replacements = {
        "causal_support_confirmed": "confirmed",
        "gate_without_target_advantage": "gate only",
        "target_advantage_without_gate": "target only",
        "not_confirmed": "not confirmed",
        "missing_run": "pending",
        "missing_summary": "needs summary",
    }
    return replacements.get(label, label.replace("_", " "))


def plot_result_scoreboard(results: pd.DataFrame, *, output_path: Path) -> None:
    rows = results.copy()
    rows["rank"] = rows["rank"].astype(int)
    rows["priority_score"] = rows["priority_score"].astype(float)
    rows = rows.sort_values("rank").reset_index(drop=True)
    y = np.arange(len(rows), dtype=float)
    labels = [row_label(row) for row in rows.itertuples(index=False)]

    fig_height = max(5.2, 0.48 * len(rows) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, fig_height), constrained_layout=True)
    colors = [RESULT_COLORS.get(str(label), "#999999") for label in rows["result_label"]]
    component_edge = [COMPONENT_COLORS.get(str(component), "#444444") for component in rows["component"]]
    axes[0].barh(y, rows["priority_score"], color=colors, edgecolor=component_edge, linewidth=1.4)
    gate = pd.to_numeric(rows["result_branch_specific_gate_rate"], errors="coerce")
    planned_gate = pd.to_numeric(rows["planned_closed_loop_gate"], errors="coerce")
    axes[0].scatter(planned_gate.fillna(0.0), y, marker="|", s=160, color="#42464f", linewidth=1.5, label="planned gate")
    if gate.notna().any():
        axes[0].scatter(gate.fillna(0.0), y, marker="o", s=34, color="#101820", label="result gate")
    axes[0].set_xlim(0.0, 1.02)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("priority / gate")
    axes[0].set_title("branch-band queue status")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_xlim(0.0, 1.0)
    axes[1].axis("off")
    headers = [
        (0.00, "result"),
        (0.27, "layers"),
        (0.48, "gate"),
        (0.62, "target-rand"),
        (0.82, "n"),
    ]
    for x, header in headers:
        axes[1].text(x, -0.85, header, fontsize=9, weight="bold")
    for idx, row in enumerate(rows.itertuples(index=False)):
        axes[1].text(0.00, idx, compact_result_label(row.result_label), va="center", fontsize=8)
        axes[1].text(0.27, idx, display_optional_text(row.recommended_layers), va="center", fontsize=8)
        axes[1].text(0.48, idx, display_number_text(row.result_branch_specific_gate_rate), va="center", fontsize=8)
        axes[1].text(
            0.62,
            idx,
            display_number_text(row.result_target_margin_delta_minus_random_mean),
            va="center",
            fontsize=8,
        )
        axes[1].text(0.82, idx, display_optional_text(row.matched_random_rows), va="center", fontsize=8)

    fig.suptitle("HLTD branch-band closed-loop result scoreboard", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_layer_result_summary(layer_rows: pd.DataFrame, *, output_path: Path) -> None:
    rows = layer_rows.copy()
    if rows.empty:
        raise ValueError("No branch-band layer result rows available.")
    rows["rank"] = rows["rank"].astype(int)
    rows["layer"] = rows["layer"].astype(int)
    row_index = (
        rows[["rank", "family", "component"]]
        .drop_duplicates()
        .sort_values("rank")
        .reset_index(drop=True)
    )
    layers = sorted(int(layer) for layer in rows["layer"].dropna().unique())
    lookup = {(int(row.rank), int(row.layer)): row for row in rows.itertuples(index=False)}
    gate_values = np.full((len(row_index), len(layers)), np.nan, dtype=float)
    target_values = np.full_like(gate_values, np.nan)
    for row_idx, row in enumerate(row_index.itertuples(index=False)):
        for col_idx, layer in enumerate(layers):
            found = lookup.get((int(row.rank), layer))
            if found is None:
                continue
            gate_values[row_idx, col_idx] = float(found.branch_specific_gate_rate)
            target_values[row_idx, col_idx] = float(found.target_margin_delta_minus_random_mean)

    fig_height = max(5.6, 0.5 * len(row_index) + 2.2)
    fig_width = max(10.8, 0.85 * len(layers) + 7.2)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), constrained_layout=True)
    ylabels = [row_label(row) for row in row_index.itertuples(index=False)]
    xlabels = [f"L{layer}" for layer in layers]

    gate_cmap = plt.colormaps["viridis"].copy()
    gate_cmap.set_bad("#F0F1F4")
    gate_image = axes[0].imshow(np.ma.masked_invalid(gate_values), aspect="auto", cmap=gate_cmap, vmin=0.0, vmax=1.0)
    annotate_matrix(axes[0], gate_values, cmap=gate_cmap, vmin=0.0, vmax=1.0)
    axes[0].set_title("branch-specific gate")
    axes[0].set_xticks(np.arange(len(layers)), xlabels)
    axes[0].set_yticks(np.arange(len(row_index)), ylabels)
    fig.colorbar(gate_image, ax=axes[0], fraction=0.035, pad=0.02)

    limit = symmetric_limit(target_values.ravel())
    target_cmap = plt.colormaps["coolwarm"].copy()
    target_cmap.set_bad("#F0F1F4")
    target_image = axes[1].imshow(
        np.ma.masked_invalid(target_values),
        aspect="auto",
        cmap=target_cmap,
        vmin=-limit,
        vmax=limit,
    )
    annotate_matrix(axes[1], target_values, cmap=target_cmap, vmin=-limit, vmax=limit)
    axes[1].set_title("target margin - random")
    axes[1].set_xticks(np.arange(len(layers)), xlabels)
    axes[1].set_yticks(np.arange(len(row_index)), [])
    fig.colorbar(target_image, ax=axes[1], fraction=0.035, pad=0.02)

    for ax in axes:
        ax.set_xlabel("layer")
        ax.grid(False)

    fig.suptitle("HLTD branch-band layer result summary", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def annotate_matrix(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    cmap: matplotlib.colors.Colormap,
    vmin: float,
    vmax: float,
) -> None:
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            scaled = np.clip((value - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
            red, green, blue, _ = cmap(float(scaled))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            color = "white" if luminance < 0.48 else "#1F2430"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)


def build_plots(*, result_root: Path, output_dir: Path) -> List[Path]:
    tables = load_tables(result_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = [output_dir / "branch_band_result_scoreboard.png"]
    plot_result_scoreboard(tables["result_scoreboard"], output_path=saved[0])
    if "layer_summary" in tables and not tables["layer_summary"].empty:
        layer_path = output_dir / "branch_band_layer_result_summary.png"
        plot_layer_result_summary(tables["layer_summary"], output_path=layer_path)
        saved.append(layer_path)
    manifest = {
        "result_root": str(result_root),
        "plots": [str(path) for path in saved],
    }
    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return saved + [manifest_path]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="spiral_out_hltd_branch_band_results")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir) if args.output_dir else result_root / "plots"
    saved = build_plots(result_root=result_root, output_dir=output_dir)
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
