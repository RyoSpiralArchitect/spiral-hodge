#!/usr/bin/env python3
"""Plot a disjoint counterfactual-probe surface over HLTD branches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DEFAULT_COMPONENTS = [
    "presence",
    "coexact",
    "semantic_flow",
    "harmonic",
    "presence_plus_coexact",
    "coexact_minus_presence",
    "negative_coexact",
]
COMPONENT_LABELS = {
    "presence": "presence",
    "coexact": "coexact",
    "semantic_flow": "semantic flow",
    "harmonic": "harmonic",
    "presence_plus_coexact": "pres + coex",
    "coexact_minus_presence": "coex - pres",
    "negative_coexact": "-coexact",
}
PAIR_METRIC = "label_axis_delta_minus_random_tangent_mean"
PAIR_WIN_RATE = "label_axis_delta_minus_random_tangent_positive_rate"


def _read_csv(path: Path, required: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    return table


def load_probe_surface(
    run_root: Path,
    *,
    probe: str,
    components: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    pairwise = _read_csv(
        run_root / "summary_pairwise.csv",
        ["layer", "k", "component", "alpha", "probe", PAIR_METRIC, PAIR_WIN_RATE],
    )
    component = _read_csv(
        run_root / "summary_component.csv",
        ["layer", "k", "component", "alpha", "probe", "component_active_mean"],
    )
    training = _read_csv(
        run_root / "probe_training_summary.csv",
        ["layer", "probe", "cv_group_accuracy", "grouping", "training_token_selector"],
    )
    manifest_path = run_root / "probe_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    component_set = {str(value) for value in components}
    pairwise = pairwise[
        (pairwise["probe"].astype(str) == str(probe))
        & pairwise["component"].astype(str).isin(component_set)
    ].copy()
    component = component[
        (component["probe"].astype(str) == str(probe))
        & component["component"].astype(str).isin(component_set)
    ].copy()
    training = training[training["probe"].astype(str) == str(probe)].copy()
    if component.empty or training.empty:
        raise ValueError(f"{run_root}: no rows for probe={probe!r}")
    return pairwise, component, training, manifest


def build_surface_table(
    pairwise: pd.DataFrame,
    component: pd.DataFrame,
    *,
    components: Sequence[str],
) -> pd.DataFrame:
    layers = sorted({int(value) for value in component["layer"]})
    ks = sorted({int(value) for value in component["k"]})
    alphas = sorted({float(value) for value in component["alpha"]})
    grid = pd.MultiIndex.from_product(
        [layers, [str(value) for value in components], ks, alphas],
        names=["layer", "component", "k", "alpha"],
    ).to_frame(index=False)

    active = (
        component.groupby(["layer", "component", "k", "alpha"], as_index=False)["component_active_mean"]
        .mean()
    )
    axis = pairwise[["layer", "component", "k", "alpha", PAIR_METRIC, PAIR_WIN_RATE]].copy()
    if axis.duplicated(["layer", "component", "k", "alpha"]).any():
        raise ValueError("Pairwise surface has duplicate layer/component/k/alpha cells.")
    surface = grid.merge(active, on=["layer", "component", "k", "alpha"], how="left")
    surface = surface.merge(axis, on=["layer", "component", "k", "alpha"], how="left")
    surface["component_active_mean"] = surface["component_active_mean"].fillna(0.0)
    surface.loc[surface["component_active_mean"] <= 0.0, PAIR_METRIC] = np.nan
    surface.loc[surface["component_active_mean"] <= 0.0, PAIR_WIN_RATE] = np.nan
    return surface


def summarize_surface(surface: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (layer, component), group in surface.groupby(["layer", "component"], sort=True):
        values = group[PAIR_METRIC].astype(float).to_numpy()
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "layer": int(layer),
                "component": str(component),
                "n_cells": int(len(group)),
                "n_active_cells": int(np.sum(group["component_active_mean"].astype(float) > 0.0)),
                "component_active_mean": float(group["component_active_mean"].astype(float).mean()),
                "label_axis_minus_random_mean": float(np.mean(finite)) if finite.size else float("nan"),
                "label_axis_minus_random_min": float(np.min(finite)) if finite.size else float("nan"),
                "label_axis_minus_random_max": float(np.max(finite)) if finite.size else float("nan"),
                "positive_cell_rate": float(np.mean(finite > 0.0)) if finite.size else float("nan"),
                "null_win_rate_mean": float(group[PAIR_WIN_RATE].astype(float).mean()),
                "null_win_rate_min": float(group[PAIR_WIN_RATE].astype(float).min()),
                "null_win_rate_max": float(group[PAIR_WIN_RATE].astype(float).max()),
            }
        )
    return pd.DataFrame(rows)


def _matrix(
    surface: pd.DataFrame,
    *,
    layer: int,
    components: Sequence[str],
    columns: Sequence[Tuple[int, float]],
    metric: str = PAIR_METRIC,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.full((len(components), len(columns)), np.nan, dtype=float)
    active = np.zeros((len(components), len(columns)), dtype=float)
    subset = surface[surface["layer"].astype(int) == int(layer)]
    for row_no, component in enumerate(components):
        branch = subset[subset["component"].astype(str) == str(component)]
        for item in branch.itertuples(index=False):
            key = (int(item.k), float(item.alpha))
            if key not in columns:
                continue
            col_no = columns.index(key)
            active[row_no, col_no] = float(item.component_active_mean)
            value = getattr(item, metric)
            if pd.notna(value):
                values[row_no, col_no] = float(value)
    return values, active


def plot_surface(
    surface: pd.DataFrame,
    training: pd.DataFrame,
    *,
    output_path: Path,
    prompt_label: str,
    probe: str,
    components: Sequence[str],
) -> None:
    layers = sorted({int(value) for value in surface["layer"]})
    columns = sorted({(int(row.k), float(row.alpha)) for row in surface.itertuples(index=False)})
    matrices = {layer: _matrix(surface, layer=layer, components=components, columns=columns) for layer in layers}
    win_matrices = {
        layer: _matrix(
            surface,
            layer=layer,
            components=components,
            columns=columns,
            metric=PAIR_WIN_RATE,
        )
        for layer in layers
    }
    finite_parts = [values[np.isfinite(values)] for values, _active in matrices.values()]
    nonempty_parts = [part for part in finite_parts if part.size]
    finite = np.concatenate(nonempty_parts) if nonempty_parts else np.asarray([], dtype=float)
    scale = max(float(np.max(np.abs(finite))) if finite.size else 0.0, 0.05)

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#e4e7eb")
    fig, axes = plt.subplots(
        1,
        len(layers),
        figsize=(max(12.5, 5.0 * len(layers)), 6.25),
        constrained_layout=False,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.13, top=0.82, wspace=0.035)
    fig.suptitle(f"Disjoint counterfactual identity probe: {prompt_label}", fontsize=15, y=0.975)
    fig.text(
        0.5,
        0.925,
        "cell = unit identity-transfer probe-axis displacement minus matched random tangent; outline = null percentile <=5% or >=95%",
        ha="center",
        fontsize=9.5,
        color="#596273",
    )
    image = None
    for col_no, layer in enumerate(layers):
        ax = axes[0, col_no]
        values, active = matrices[layer]
        win_values, _win_active = win_matrices[layer]
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=-scale, vmax=scale)
        cv_rows = training[training["layer"].astype(int) == int(layer)]
        cv = float(cv_rows.iloc[0]["cv_group_accuracy"]) if not cv_rows.empty else float("nan")
        ax.set_title(f"Layer {layer}  pair-CV={cv:.3f}")
        ax.set_xticks(range(len(columns)), [f"k{k}\na={alpha:g}" for k, alpha in columns], fontsize=8)
        ax.set_yticks(
            range(len(components)),
            [COMPONENT_LABELS.get(component, component) for component in components],
            fontsize=9,
        )
        if col_no:
            ax.tick_params(axis="y", labelleft=False)
        for row_no in range(len(components)):
            row_is_inactive = bool(np.all(active[row_no] <= 0.0))
            for cell_no in range(len(columns)):
                value = values[row_no, cell_no]
                if not np.isfinite(value):
                    if row_is_inactive and cell_no == len(columns) // 2:
                        ax.text(cell_no, row_no, "inactive", ha="center", va="center", fontsize=7.2, color="#606770")
                    continue
                color = "white" if abs(value) > scale * 0.57 else "#202124"
                ax.text(cell_no, row_no, f"{value:+.2f}", ha="center", va="center", fontsize=7.0, color=color)
                win_rate = win_values[row_no, cell_no]
                if np.isfinite(win_rate) and (win_rate <= 0.05 or win_rate >= 0.95):
                    ax.add_patch(
                        Rectangle(
                            (cell_no - 0.49, row_no - 0.49),
                            0.98,
                            0.98,
                            facecolor="none",
                            edgecolor="#17191c",
                            linewidth=1.4,
                            antialiased=False,
                            zorder=3,
                        )
                    )
    if image is not None:
        fig.colorbar(image, ax=list(axes[0, :]), fraction=0.025, pad=0.015, label="unit probe-axis delta vs random")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_counterfactual_probe_plot(
    *,
    run_root: Path,
    output_root: Path,
    probe: str,
    components: Sequence[str],
) -> Dict[str, str]:
    pairwise, component, training, manifest = load_probe_surface(
        run_root,
        probe=probe,
        components=components,
    )
    surface = build_surface_table(pairwise, component, components=components)
    summary = summarize_surface(surface)
    prompt_ids = [str(value) for value in manifest.get("evaluation_prompt_ids", [])]
    prompt_label = ", ".join(prompt_ids) if prompt_ids else run_root.name

    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"counterfactual_{probe}_branch_surface"
    surface_path = output_root / f"{stem}.csv"
    summary_path = output_root / f"{stem}_summary.csv"
    plot_path = output_root / f"{stem}.png"
    manifest_path = output_root / f"{stem}_manifest.json"
    surface.to_csv(surface_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_surface(
        surface,
        training,
        output_path=plot_path,
        prompt_label=prompt_label,
        probe=probe,
        components=components,
    )
    plot_manifest = {
        "run_root": str(run_root),
        "probe": probe,
        "components": [str(value) for value in components],
        "metric": PAIR_METRIC,
        "evaluation_prompt_ids": prompt_ids,
        "source_split_is_disjoint": bool(manifest.get("split_is_disjoint", False)),
        "surface_csv": str(surface_path),
        "summary_csv": str(summary_path),
        "plot": str(plot_path),
    }
    manifest_path.write_text(json.dumps(plot_manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "surface_csv": str(surface_path),
        "summary_csv": str(summary_path),
        "plot": str(plot_path),
        "manifest": str(manifest_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--probe", default="identity_stress")
    parser.add_argument("--components", nargs="+", default=DEFAULT_COMPONENTS)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    output_root = Path(args.output_root) if args.output_root else run_root / "plots"
    saved = build_counterfactual_probe_plot(
        run_root=run_root,
        output_root=output_root,
        probe=args.probe,
        components=args.components,
    )
    for path in saved.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
