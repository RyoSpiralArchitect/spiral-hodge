#!/usr/bin/env python3
"""Plot prompt-overlap target robustness for matched HLTD closed-loop surfaces."""
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


DEFAULT_COMPONENTS = [
    "presence_plus_coexact",
    "coexact_minus_presence",
    "presence",
    "coexact",
    "negative_coexact",
]
COMPONENT_LABELS = {
    "presence_plus_coexact": "pres+coex",
    "coexact_minus_presence": "coex-pres",
    "presence": "presence",
    "coexact": "coexact",
    "negative_coexact": "-coexact",
}
PAIR_KEYS = ["family", "prompt_id", "layer", "k", "component", "alpha"]
DELTA_METRICS = [
    "branch_specific_gate_rate",
    "mean_target_margin_delta_minus_random_mean",
]


def load_surface(root: Path, *, prompt_id: str, components: Sequence[str]) -> pd.DataFrame:
    path = root / "closed_loop_prompt_layer_k_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    required = set(PAIR_KEYS + DELTA_METRICS + ["target_set"])
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    subset = table[
        (table["prompt_id"].astype(str) == str(prompt_id))
        & table["component"].astype(str).isin([str(component) for component in components])
    ].copy()
    if subset.empty:
        raise ValueError(f"{root}: no surface rows for prompt_id={prompt_id!r}")
    return subset


def pair_surfaces(reference: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
    for label, table in [("reference", reference), ("heldout", heldout)]:
        if table.duplicated(PAIR_KEYS).any():
            raise ValueError(f"{label} surface has duplicate pairing keys")
    paired = reference.merge(
        heldout,
        on=PAIR_KEYS,
        how="inner",
        suffixes=("_reference", "_heldout"),
        validate="one_to_one",
    )
    if len(paired) != len(reference) or len(paired) != len(heldout):
        raise ValueError(
            "Reference and held-out surfaces do not have identical prompt/layer/k/component/alpha cells."
        )
    for metric in DELTA_METRICS:
        paired[f"{metric}_delta"] = paired[f"{metric}_heldout"].astype(float) - paired[
            f"{metric}_reference"
        ].astype(float)
    return paired


def _surface_matrix(
    paired: pd.DataFrame,
    *,
    layer: int,
    metric: str,
    components: Sequence[str],
    columns: Sequence[Tuple[int, float]],
) -> np.ndarray:
    subset = paired[paired["layer"].astype(int) == int(layer)]
    values = np.full((len(components), len(columns)), np.nan, dtype=float)
    for row_no, component in enumerate(components):
        branch = subset[subset["component"].astype(str) == str(component)]
        lookup = {
            (int(row.k), float(row.alpha)): float(getattr(row, metric))
            for row in branch.itertuples(index=False)
        }
        for col_no, key in enumerate(columns):
            if key in lookup:
                values[row_no, col_no] = lookup[key]
    return values


def _annotate(ax: plt.Axes, values: np.ndarray, *, scale: float) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if not np.isfinite(value):
                continue
            color = "white" if abs(float(value)) > scale * 0.58 else "#222222"
            ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=7.2, color=color)


def plot_target_overlap_delta(
    paired: pd.DataFrame,
    *,
    output_path: Path,
    prompt_id: str,
    components: Sequence[str],
    reference_label: str,
    heldout_label: str,
) -> None:
    layers = sorted({int(value) for value in paired["layer"]})
    columns = sorted({(int(row.k), float(row.alpha)) for row in paired.itertuples(index=False)})
    if not layers or not columns:
        raise ValueError("No paired surface cells to plot.")

    matrices: Dict[Tuple[int, str], np.ndarray] = {}
    metric_names = [
        "branch_specific_gate_rate_delta",
        "mean_target_margin_delta_minus_random_mean_delta",
    ]
    for layer in layers:
        for metric in metric_names:
            matrices[(layer, metric)] = _surface_matrix(
                paired,
                layer=layer,
                metric=metric,
                components=components,
                columns=columns,
            )

    scales = []
    for metric in metric_names:
        finite = np.concatenate(
            [matrix[np.isfinite(matrix)] for (layer, name), matrix in matrices.items() if name == metric]
        )
        scales.append(max(float(np.max(np.abs(finite))) if finite.size else 0.0, 0.05))

    fig, axes = plt.subplots(
        2,
        len(layers),
        figsize=(max(12.0, 4.7 * len(layers)), 8.2),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(f"HLTD prompt-target overlap robustness: {prompt_id}", fontsize=15)
    fig.text(
        0.5,
        0.955,
        f"cell delta = {heldout_label} - {reference_label}; trajectories and matched random seeds are identical",
        ha="center",
        fontsize=9.5,
        color="#596273",
    )
    row_titles = ["strict gate delta", "target margin - random delta"]
    cmaps = ["PuOr_r", "coolwarm"]
    images = []
    for row_no, (metric, row_title, cmap, scale) in enumerate(zip(metric_names, row_titles, cmaps, scales)):
        for col_no, layer in enumerate(layers):
            ax = axes[row_no, col_no]
            matrix = matrices[(layer, metric)]
            image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-scale, vmax=scale)
            images.append((row_no, image))
            _annotate(ax, matrix, scale=scale)
            if row_no == 0:
                ax.set_title(f"Layer {layer}")
            ax.set_yticks(
                range(len(components)),
                [COMPONENT_LABELS.get(component, component) for component in components],
            )
            if col_no == 0:
                ax.set_ylabel(row_title)
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xticks(range(len(columns)))
            if row_no == 1:
                ax.set_xticklabels([f"k{k}\na={alpha:g}" for k, alpha in columns], fontsize=8)
            else:
                ax.tick_params(axis="x", labelbottom=False)
    for row_no, scale in enumerate(scales):
        image = next(image for image_row, image in images if image_row == row_no)
        fig.colorbar(image, ax=list(axes[row_no, :]), fraction=0.025, pad=0.015)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_target_overlap_robustness(
    *,
    reference_root: Path,
    heldout_root: Path,
    output_root: Path,
    prompt_id: str,
    components: Sequence[str],
    reference_label: str = "full target",
    heldout_label: str = "prompt-heldout target",
) -> Dict[str, str]:
    reference = load_surface(reference_root, prompt_id=prompt_id, components=components)
    heldout = load_surface(heldout_root, prompt_id=prompt_id, components=components)
    paired = pair_surfaces(reference, heldout)
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = f"target_overlap_robustness_{prompt_id}"
    csv_path = output_root / f"{prefix}.csv"
    plot_path = output_root / f"{prefix}.png"
    manifest_path = output_root / f"{prefix}_manifest.json"
    paired.to_csv(csv_path, index=False)
    plot_target_overlap_delta(
        paired,
        output_path=plot_path,
        prompt_id=prompt_id,
        components=components,
        reference_label=reference_label,
        heldout_label=heldout_label,
    )
    manifest = {
        "prompt_id": prompt_id,
        "reference_root": str(reference_root),
        "heldout_root": str(heldout_root),
        "reference_label": reference_label,
        "heldout_label": heldout_label,
        "components": [str(component) for component in components],
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "plot": str(plot_path), "manifest": str(manifest_path)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--heldout-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--components", nargs="+", default=DEFAULT_COMPONENTS)
    parser.add_argument("--reference-label", default="full target")
    parser.add_argument("--heldout-label", default="prompt-heldout target")
    args = parser.parse_args(argv)

    saved = build_target_overlap_robustness(
        reference_root=Path(args.reference_root),
        heldout_root=Path(args.heldout_root),
        output_root=Path(args.output_root),
        prompt_id=args.prompt_id,
        components=args.components,
        reference_label=args.reference_label,
        heldout_label=args.heldout_label,
    )
    for path in saved.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
