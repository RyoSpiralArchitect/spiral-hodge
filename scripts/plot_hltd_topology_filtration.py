#!/usr/bin/env python3
"""Render branch-persistence plots from an HLTD topology filtration run."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BRANCHES = ("exact", "coexact", "harmonic")
BRANCH_LABELS = {
    "exact": "exact / presence",
    "coexact": "coexact / local circulation",
    "harmonic": "harmonic / open-cycle residual",
}
BRANCH_COLORS = {
    "exact": "#34804b",
    "coexact": "#256fa8",
    "harmonic": "#86549d",
}
K_COLORS = {
    12: "#2b6f92",
    16: "#c17424",
    24: "#7c548f",
}
FAMILY_COLORS = {
    "literal_stable": "#2f766d",
    "metaphor_shift": "#ba6a28",
    "identity_stress": "#7b549d",
    "ontology_collapse": "#b8484e",
}
NULL_COLORS = {
    "vector_shuffle": "#ba6a28",
    "random_tangent": "#3377a8",
}
NULL_LABELS = {
    "vector_shuffle": "vector shuffle",
    "random_tangent": "random tangent",
}
BASE_MATCH_KEYS = ["family", "prompt_id", "layer", "k"]
BETTI_TARGETS = (1.0, 0.75, 0.5, 0.25, 0.0)
BOOTSTRAP_SEED = 1729
BOOTSTRAP_SAMPLES = 5000


def filtration_mode(metrics: pd.DataFrame) -> str:
    """Infer one filtration mode while retaining legacy count-run support."""

    if "filtration_mode" in metrics.columns:
        modes = sorted(str(value) for value in metrics["filtration_mode"].dropna().unique())
        if len(modes) > 1:
            raise ValueError(f"metrics contains mixed filtration modes: {', '.join(modes)}")
        if modes:
            return modes[0]
    if (
        "filtration_radius_scale_requested" in metrics.columns
        and (
            "triangle_fill_requested" not in metrics.columns
            or metrics["triangle_fill_requested"].isna().all()
        )
    ):
        return "radius"
    return "count"


def filtration_axis_column(metrics: pd.DataFrame) -> str:
    return (
        "filtration_radius_scale_requested"
        if filtration_mode(metrics) == "radius"
        else "triangle_fill_requested"
    )


def filtration_axis_label(metrics: pd.DataFrame) -> str:
    if filtration_mode(metrics) == "radius":
        return "triangle radius / median graph edge"
    return "triangle fill fraction"


def filtration_match_keys(metrics: pd.DataFrame) -> list[str]:
    return [*BASE_MATCH_KEYS, filtration_axis_column(metrics)]


def filtration_values(metrics: pd.DataFrame, x_column: str) -> np.ndarray:
    values = np.asarray(metrics[x_column], dtype=np.float64)
    return np.asarray(sorted(set(values[~np.isnan(values)].tolist())), dtype=np.float64)


def display_filtration_x(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    """Map an infinite full-complex endpoint to a finite plotting position."""

    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    finite = np.sort(reference[np.isfinite(reference)])
    if not np.any(np.isinf(reference)):
        return values
    if len(finite) >= 2:
        differences = np.diff(finite)
        positive = differences[differences > 1e-12]
        step = float(np.median(positive)) if len(positive) else 0.25
    elif len(finite) == 1:
        step = max(abs(float(finite[0])) * 0.2, 0.25)
    else:
        step = 1.0
    full_position = (float(finite[-1]) if len(finite) else 0.0) + step
    return np.where(np.isinf(values), full_position, values)


def configure_filtration_axis(
    ax: plt.Axes,
    *,
    reference_values: np.ndarray,
    label: str,
) -> None:
    if np.any(np.isinf(reference_values)):
        ticks = display_filtration_x(reference_values, reference_values)
        labels = ["full" if np.isinf(value) else f"{value:g}" for value in reference_values]
        ax.set_xticks(ticks, labels)
        if len(reference_values) > 7:
            ax.tick_params(axis="x", labelsize=8)
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlabel(label)


def format_filtration_value(value: float) -> str:
    return "full" if np.isinf(float(value)) else f"{float(value):g}"


def quantile_band(
    rows: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped = rows.groupby(x_column, sort=True)[y_column]
    summary = grouped.agg(
        median="median",
        lower=lambda values: values.quantile(0.25),
        upper=lambda values: values.quantile(0.75),
    ).reset_index()
    return (
        summary[x_column].astype(float).to_numpy(),
        summary["median"].astype(float).to_numpy(),
        summary["lower"].astype(float).to_numpy(),
        summary["upper"].astype(float).to_numpy(),
    )


def _real_null_gaps(metrics: pd.DataFrame, *, match_keys: Sequence[str]) -> pd.DataFrame:
    real = metrics[metrics["variant"] == "real"].copy()
    null = metrics[metrics["variant"] != "real"].copy()
    if real.empty or null.empty:
        return pd.DataFrame()
    ratio_columns = [f"{branch}_ratio" for branch in BRANCHES]
    null_summary = null.groupby(list(match_keys), as_index=False)[ratio_columns].mean()
    joined = real.merge(null_summary, on=list(match_keys), suffixes=("_real", "_null"))
    for branch in BRANCHES:
        joined[f"{branch}_gap"] = (
            joined[f"{branch}_ratio_real"] - joined[f"{branch}_ratio_null"]
        )
    return joined


def matched_real_null_gaps(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return prompt/layer/k/filtration matched real-minus-null ratios."""

    return _real_null_gaps(metrics, match_keys=filtration_match_keys(metrics))


def branch_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    x_column = filtration_axis_column(metrics)
    columns = [f"{branch}_ratio" for branch in BRANCHES]
    group_columns = ["variant", "family", "layer", "k", x_column]
    summary = (
        metrics.groupby(group_columns, as_index=False)[
            columns + ["betti_1_fraction", "harmonic_survival_ratio"]
        ]
        .mean()
        .sort_values(group_columns)
    )
    summary.insert(0, "filtration_mode", filtration_mode(metrics))
    return summary


def matched_betti_summary(
    metrics: pd.DataFrame,
    *,
    targets: Sequence[float] = BETTI_TARGETS,
) -> pd.DataFrame:
    """Interpolate each field/null trajectory at shared Betti-1 capacities."""

    value_columns = [f"{branch}_ratio" for branch in BRANCHES]
    for optional in ("semantic_flow_ratio", "harmonic_survival_ratio"):
        if optional in metrics.columns:
            value_columns.append(optional)
    group_columns = ["variant", "seed", *BASE_MATCH_KEYS]
    rows = []
    for key, group in metrics.groupby(group_columns, sort=False, dropna=False):
        aggregated = (
            group.groupby("betti_1_fraction", as_index=False)[value_columns]
            .mean()
            .sort_values("betti_1_fraction")
        )
        x = aggregated["betti_1_fraction"].astype(float).to_numpy()
        if len(x) == 0:
            continue
        base = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        for target in sorted(set(float(value) for value in targets), reverse=True):
            if target < float(x[0]) - 1e-12 or target > float(x[-1]) + 1e-12:
                continue
            row = {
                **base,
                "filtration_mode": filtration_mode(metrics),
                "betti_1_target": target,
                "betti_1_min": float(x[0]),
                "betti_1_max": float(x[-1]),
            }
            for column in value_columns:
                values = aggregated[column].astype(float).to_numpy()
                row[column] = float(np.interp(target, x, values))
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["variant", "family", "prompt_id", "layer", "k", "betti_1_target"],
        ascending=[True, True, True, True, True, False],
    )


def matched_betti_real_null_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return _real_null_gaps(
        summary,
        match_keys=[*BASE_MATCH_KEYS, "betti_1_target"],
    )


def matched_betti_real_null_gaps_by_variant(summary: pd.DataFrame) -> pd.DataFrame:
    """Match real fields to each null construction after averaging null seeds."""

    if summary.empty:
        return pd.DataFrame()
    match_keys = [*BASE_MATCH_KEYS, "betti_1_target"]
    ratio_columns = [f"{branch}_ratio" for branch in BRANCHES]
    real = summary[summary["variant"] == "real"][match_keys + ratio_columns].copy()
    rows = []
    for null_variant in sorted(
        str(value) for value in summary.loc[summary["variant"] != "real", "variant"].unique()
    ):
        null = summary[summary["variant"] == null_variant]
        null_summary = null.groupby(match_keys, as_index=False)[ratio_columns].mean()
        joined = real.merge(null_summary, on=match_keys, suffixes=("_real", "_null"))
        joined.insert(0, "null_variant", null_variant)
        for branch in BRANCHES:
            joined[f"{branch}_gap"] = (
                joined[f"{branch}_ratio_real"] - joined[f"{branch}_ratio_null"]
            )
        rows.append(joined)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["null_variant", "family", "prompt_id", "layer", "k", "betti_1_target"],
        ascending=[True, True, True, True, True, False],
    )


def prompt_bootstrap_gap_summary(
    gaps: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    zero_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Bootstrap prompts after collapsing repeated layer/k measurements."""

    if gaps.empty:
        return pd.DataFrame()
    gap_columns = [f"{branch}_gap" for branch in BRANCHES]
    unit_columns = [*group_columns, "prompt_id"]
    prompt_rows = gaps.groupby(unit_columns, as_index=False)[gap_columns].mean()
    rng = np.random.default_rng(seed)
    rows = []
    for key, group in prompt_rows.groupby(list(group_columns), sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_columns, key_values))
        draw_indices = rng.integers(
            0,
            len(group),
            size=(int(n_bootstrap), len(group)),
        )
        for branch in BRANCHES:
            values = group[f"{branch}_gap"].astype(float).to_numpy()
            draws = values[draw_indices].mean(axis=1)
            lower, upper = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    **base,
                    "branch": branch,
                    "n_prompts": int(len(values)),
                    "mean_gap": float(values.mean()),
                    "median_gap": float(np.median(values)),
                    "bootstrap_ci_lower": float(lower),
                    "bootstrap_ci_upper": float(upper),
                    "positive_prompt_fraction": float(
                        np.mean(values > zero_tolerance)
                    ),
                    "negative_prompt_fraction": float(
                        np.mean(values < -zero_tolerance)
                    ),
                    "near_zero_prompt_fraction": float(
                        np.mean(np.abs(values) <= zero_tolerance)
                    ),
                    "bootstrap_samples": int(n_bootstrap),
                    "bootstrap_seed": int(seed),
                }
            )
    return pd.DataFrame(rows)


def prompt_bootstrap_inference(gaps: pd.DataFrame) -> pd.DataFrame:
    """Summarize all prompts, then condition the same gate on layer and k."""

    scopes = {
        "overall": ["null_variant", "betti_1_target"],
        "layer": ["null_variant", "betti_1_target", "layer"],
        "k": ["null_variant", "betti_1_target", "k"],
    }
    tables = []
    for scope, group_columns in scopes.items():
        table = prompt_bootstrap_gap_summary(gaps, group_columns=group_columns)
        if not table.empty:
            table.insert(0, "scope", scope)
            tables.append(table)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True, sort=False)


def plot_branch_persistence(metrics: pd.DataFrame, *, output_path: Path) -> None:
    real = metrics[metrics["variant"] == "real"].copy()
    gaps = matched_real_null_gaps(metrics)
    match_keys = filtration_match_keys(metrics)
    topology = real.drop_duplicates(match_keys).copy()
    x_column = filtration_axis_column(metrics)
    x_label = filtration_axis_label(metrics)
    x_reference = filtration_values(metrics, x_column)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    ax_energy, ax_gap, ax_topology, ax_phase = axes.ravel()

    for branch in BRANCHES:
        x, center, lower, upper = quantile_band(
            real,
            x_column=x_column,
            y_column=f"{branch}_ratio",
        )
        x_plot = display_filtration_x(x, x_reference)
        color = BRANCH_COLORS[branch]
        ax_energy.plot(x_plot, center, marker="o", color=color, label=BRANCH_LABELS[branch])
        ax_energy.fill_between(x_plot, lower, upper, color=color, alpha=0.13, linewidth=0)
    ax_energy.set_title("A. Hodge energy transfer")
    configure_filtration_axis(ax_energy, reference_values=x_reference, label=x_label)
    ax_energy.set_ylabel("median energy ratio (IQR)")
    ax_energy.set_ylim(-0.02, 1.02)
    ax_energy.grid(alpha=0.22)
    ax_energy.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=8)

    if not gaps.empty:
        for branch in BRANCHES:
            x, center, lower, upper = quantile_band(
                gaps,
                x_column=x_column,
                y_column=f"{branch}_gap",
            )
            x_plot = display_filtration_x(x, x_reference)
            color = BRANCH_COLORS[branch]
            ax_gap.plot(x_plot, center, marker="o", color=color, label=BRANCH_LABELS[branch])
            ax_gap.fill_between(x_plot, lower, upper, color=color, alpha=0.13, linewidth=0)
    ax_gap.axhline(0.0, color="#4c4c4c", linewidth=0.9)
    ax_gap.set_title("B. Real minus matched null")
    configure_filtration_axis(ax_gap, reference_values=x_reference, label=x_label)
    ax_gap.set_ylabel("energy-ratio gap")
    ax_gap.grid(alpha=0.22)

    phase_ks = sorted(int(value) for value in topology["k"].unique())
    annotation_k = min(phase_ks, key=lambda value: abs(value - 16))
    for k in phase_ks:
        subset = topology[topology["k"] == k]
        x, center, lower, upper = quantile_band(
            subset,
            x_column=x_column,
            y_column="betti_1_fraction",
        )
        x_plot = display_filtration_x(x, x_reference)
        color = K_COLORS.get(k, "#555555")
        ax_topology.plot(x_plot, center, marker="o", color=color, label=f"k={k}")
        ax_topology.fill_between(x_plot, lower, upper, color=color, alpha=0.12, linewidth=0)
    ax_topology.set_title("C. First-homology capacity")
    configure_filtration_axis(ax_topology, reference_values=x_reference, label=x_label)
    ax_topology.set_ylabel("Betti-1 / graph cycle rank")
    ax_topology.set_ylim(-0.02, 1.02)
    ax_topology.grid(alpha=0.22)
    ax_topology.legend(frameon=False, fontsize=8)

    annotation_values = {float(x_reference[0]), float(x_reference[-1])}
    for k in phase_ks:
        subset = topology[topology["k"] == k]
        grouped = (
            subset.groupby(x_column, as_index=False)[["betti_1_fraction", "harmonic_ratio"]]
            .median()
            .sort_values(x_column)
        )
        color = K_COLORS.get(k, "#555555")
        ax_phase.plot(
            grouped["betti_1_fraction"],
            grouped["harmonic_ratio"],
            marker="o",
            color=color,
            label=f"k={k}",
        )
        for _, row in grouped.iterrows():
            value = float(row[x_column])
            if k == annotation_k and value in annotation_values:
                is_open = value == float(x_reference[0])
                ax_phase.annotate(
                    format_filtration_value(value),
                    (float(row["betti_1_fraction"]), float(row["harmonic_ratio"])),
                    xytext=((-5, 6) if is_open else (5, 6)),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    ha=("right" if is_open else "left"),
                )
    ax_phase.set_title("D. Harmonic energy follows open cycles")
    ax_phase.set_xlabel("Betti-1 / graph cycle rank")
    ax_phase.set_ylabel("median harmonic ratio")
    ax_phase.set_xlim(-0.02, 1.02)
    ax_phase.set_ylim(-0.02, 1.02)
    ax_phase.grid(alpha=0.22)

    fig.suptitle(
        f"HLTD {filtration_mode(metrics)} filtration: branch persistence on fixed kNN graphs",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_family_persistence(metrics: pd.DataFrame, *, output_path: Path) -> None:
    real = metrics[metrics["variant"] == "real"].copy()
    families = sorted(str(value) for value in real["family"].unique())
    x_column = filtration_axis_column(metrics)
    x_label = filtration_axis_label(metrics)
    x_reference = filtration_values(metrics, x_column)
    ncols = 2
    nrows = int(np.ceil(len(families) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11.5, 4.0 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, family in zip(axes.ravel(), families):
        subset = real[real["family"] == family]
        for branch in BRANCHES:
            x, center, lower, upper = quantile_band(
                subset,
                x_column=x_column,
                y_column=f"{branch}_ratio",
            )
            x_plot = display_filtration_x(x, x_reference)
            color = BRANCH_COLORS[branch]
            ax.plot(x_plot, center, marker="o", color=color, label=BRANCH_LABELS[branch])
            ax.fill_between(x_plot, lower, upper, color=color, alpha=0.13, linewidth=0)
        ax.set_title(family.replace("_", " "))
        configure_filtration_axis(ax, reference_values=x_reference, label=x_label)
        ax.set_ylabel("median energy ratio (IQR)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.22)
    for ax in axes.ravel()[len(families) :]:
        ax.set_visible(False)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle("HLTD branch transfer by prompt family", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_matched_betti(metrics: pd.DataFrame, *, output_path: Path) -> None:
    summary = matched_betti_summary(metrics)
    gaps = matched_betti_real_null_gaps(summary)
    real = summary[summary["variant"] == "real"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    ax_energy, ax_gap, ax_family, ax_k = axes.ravel()
    for branch in BRANCHES:
        x, center, lower, upper = quantile_band(
            real,
            x_column="betti_1_target",
            y_column=f"{branch}_ratio",
        )
        color = BRANCH_COLORS[branch]
        ax_energy.plot(x, center, marker="o", color=color, label=BRANCH_LABELS[branch])
        ax_energy.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)
    ax_energy.set_title("A. Branch energy at matched topology")
    ax_energy.set_ylabel("median energy ratio (IQR)")
    ax_energy.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        fontsize=8,
    )

    if not gaps.empty:
        for branch in BRANCHES:
            x, center, lower, upper = quantile_band(
                gaps,
                x_column="betti_1_target",
                y_column=f"{branch}_gap",
            )
            color = BRANCH_COLORS[branch]
            ax_gap.plot(x, center, marker="o", color=color, label=BRANCH_LABELS[branch])
            ax_gap.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)
        for family in sorted(str(value) for value in gaps["family"].unique()):
            subset = gaps[gaps["family"] == family]
            x, center, _lower, _upper = quantile_band(
                subset,
                x_column="betti_1_target",
                y_column="harmonic_gap",
            )
            ax_family.plot(
                x,
                center,
                marker="o",
                color=FAMILY_COLORS.get(family, "#555555"),
                label=family.replace("_", " "),
            )
        for k in sorted(int(value) for value in gaps["k"].unique()):
            subset = gaps[gaps["k"] == k]
            x, center, _lower, _upper = quantile_band(
                subset,
                x_column="betti_1_target",
                y_column="harmonic_gap",
            )
            ax_k.plot(x, center, marker="o", color=K_COLORS.get(k, "#555555"), label=f"k={k}")

    for ax in axes.ravel():
        ax.axhline(0.0, color="#4c4c4c", linewidth=0.8, zorder=0)
        ax.set_xlim(1.02, -0.02)
        ax.set_xlabel("matched Betti-1 capacity (open to filled)")
        ax.grid(alpha=0.22)
    ax_energy.set_ylim(-0.02, 1.02)
    ax_gap.set_title("B. Real minus null at matched topology")
    ax_gap.set_ylabel("energy-ratio gap")
    ax_family.set_title("C. Harmonic gap by prompt family")
    ax_family.set_ylabel("real-minus-null harmonic ratio")
    ax_family.legend(frameon=False, fontsize=8)
    ax_k.set_title("D. Harmonic gap by neighborhood size")
    ax_k.set_ylabel("real-minus-null harmonic ratio")
    ax_k.legend(frameon=False, fontsize=8)
    fig.suptitle("HLTD branch identity after matching first-homology capacity", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_inference_curve(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    branch: str,
) -> None:
    for null_variant in sorted(str(value) for value in rows["null_variant"].unique()):
        subset = rows[
            (rows["null_variant"] == null_variant) & (rows["branch"] == branch)
        ].sort_values("betti_1_target", ascending=False)
        color = NULL_COLORS.get(null_variant, "#555555")
        x = subset["betti_1_target"].astype(float).to_numpy()
        center = subset["mean_gap"].astype(float).to_numpy()
        lower = subset["bootstrap_ci_lower"].astype(float).to_numpy()
        upper = subset["bootstrap_ci_upper"].astype(float).to_numpy()
        ax.plot(
            x,
            center,
            marker="o",
            color=color,
            label=NULL_LABELS.get(null_variant, null_variant.replace("_", " ")),
        )
        ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
    ax.axhline(0.0, color="#4c4c4c", linewidth=0.8, zorder=0)
    ax.set_xlim(1.02, -0.02)
    ax.set_xlabel("matched Betti-1 capacity (open to filled)")
    ax.set_ylabel("mean real-minus-null ratio (95% prompt CI)")
    ax.grid(alpha=0.22)


def _plot_midpoint_strata(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    stratum: str,
    target: float,
) -> None:
    subset = rows[
        (rows["branch"] == "harmonic")
        & np.isclose(rows["betti_1_target"], target)
    ].copy()
    values = sorted(int(value) for value in subset[stratum].dropna().unique())
    variants = sorted(str(value) for value in subset["null_variant"].unique())
    offsets = np.linspace(-0.08, 0.08, len(variants)) if len(variants) > 1 else [0.0]
    for offset, null_variant in zip(offsets, variants):
        group = subset[subset["null_variant"] == null_variant].set_index(stratum)
        center = np.asarray([group.loc[value, "mean_gap"] for value in values], dtype=float)
        lower = np.asarray(
            [group.loc[value, "bootstrap_ci_lower"] for value in values], dtype=float
        )
        upper = np.asarray(
            [group.loc[value, "bootstrap_ci_upper"] for value in values], dtype=float
        )
        x = np.arange(len(values), dtype=float) + float(offset)
        color = NULL_COLORS.get(null_variant, "#555555")
        ax.errorbar(
            x,
            center,
            yerr=np.vstack([center - lower, upper - center]),
            marker="o",
            capsize=3,
            color=color,
            label=NULL_LABELS.get(null_variant, null_variant.replace("_", " ")),
        )
    ax.axhline(0.0, color="#4c4c4c", linewidth=0.8, zorder=0)
    ax.set_xticks(np.arange(len(values)), [str(value) for value in values])
    ax.set_xlabel("layer" if stratum == "layer" else "neighborhood size k")
    ax.set_ylabel("mean harmonic gap (95% prompt CI)")
    ax.grid(alpha=0.22)


def plot_prompt_inference(
    inference: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    """Plot prompt-paired uncertainty separately for each null construction."""

    overall = inference[inference["scope"] == "overall"].copy()
    layers = inference[inference["scope"] == "layer"].copy()
    ks = inference[inference["scope"] == "k"].copy()
    targets = sorted(float(value) for value in overall["betti_1_target"].unique())
    midpoint = min(targets, key=lambda value: abs(value - 0.5))

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    ax_harmonic, ax_coexact, ax_layer, ax_k = axes.ravel()
    _plot_inference_curve(ax_harmonic, overall, branch="harmonic")
    ax_harmonic.set_title("A. Harmonic gap by null construction")
    ax_harmonic.legend(frameon=False, fontsize=8)
    _plot_inference_curve(ax_coexact, overall, branch="coexact")
    ax_coexact.set_title("B. Coexact gap by null construction")
    _plot_midpoint_strata(ax_layer, layers, stratum="layer", target=midpoint)
    ax_layer.set_title(f"C. Harmonic gap by layer at Betti-1={midpoint:g}")
    _plot_midpoint_strata(ax_k, ks, stratum="k", target=midpoint)
    ax_k.set_title(f"D. Harmonic gap by k at Betti-1={midpoint:g}")
    fig.suptitle("HLTD prompt-paired branch inference", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def first_crossover_value(real: pd.DataFrame) -> pd.Series:
    x_column = filtration_axis_column(real)

    def crossover(group: pd.DataFrame) -> float:
        ordered = group.sort_values(x_column)
        crossed = ordered[ordered["coexact_ratio"] >= ordered["harmonic_ratio"]]
        return float(crossed[x_column].iloc[0]) if not crossed.empty else float("nan")

    values = {
        key: crossover(group)
        for key, group in real.groupby(["prompt_id", "layer", "k"], sort=False)
    }
    return pd.Series(values, dtype=float)


def write_report(
    metrics: pd.DataFrame,
    gaps: pd.DataFrame,
    matched_gaps: pd.DataFrame,
    gaps_by_null: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    output_root: Path,
) -> None:
    real = metrics[metrics["variant"] == "real"].copy()
    mode = filtration_mode(metrics)
    x_column = filtration_axis_column(metrics)
    open_rows = real[np.isclose(real[x_column], 0.0)]
    full_rows = (
        real[np.isinf(real[x_column])]
        if mode == "radius"
        else real[np.isclose(real[x_column], 1.0)]
    )
    crossovers = first_crossover_value(real)
    crossover_label = "radius scale" if mode == "radius" else "fill"
    field_keys = ["variant", "seed", "family", "prompt_id", "layer", "k"]
    exact_range = metrics.groupby(field_keys, dropna=False)["exact_ratio"].agg(
        lambda values: values.max() - values.min()
    )
    flow_values = (
        metrics["semantic_flow_ratio"]
        if "semantic_flow_ratio" in metrics.columns
        else metrics["coexact_ratio"] + metrics["harmonic_ratio"]
    )
    flow_range = metrics.assign(_semantic_flow_ratio=flow_values).groupby(
        field_keys, dropna=False
    )["_semantic_flow_ratio"].agg(
        lambda values: values.max() - values.min()
    )
    harmonic_betti_correlation = real["harmonic_ratio"].corr(real["betti_1_fraction"])
    closure_line = ""
    if "energy_closure_error" in metrics.columns:
        closure_line = (
            f"- Maximum relative branch-energy closure error: "
            f"{metrics['energy_closure_error'].max():.3e}."
        )

    lines = [
        "# HLTD Topology Filtration",
        "",
        f"This gate keeps each kNN graph and edge flow fixed while applying a {mode}",
        "filtration to geometrically ordered clique triangles. It measures whether",
        "harmonic energy persists or transfers into the coexact subspace as graph",
        "cycles become filled boundaries.",
        "",
        "## Headline",
        "",
        f"- Rows: {len(metrics)} ({len(real)} real; {len(metrics) - len(real)} null).",
        f"- Prompt fields: {real[['prompt_id', 'layer', 'k']].drop_duplicates().shape[0]}.",
        f"- Open-complex median harmonic ratio: {open_rows['harmonic_ratio'].median():.4f}.",
        f"- Full-clique median harmonic ratio: {full_rows['harmonic_ratio'].median():.4f}.",
        f"- Median coexact-over-harmonic crossover {crossover_label}: {crossovers.median():.3f}.",
        f"- Full-clique median Betti-1 fraction: {full_rows['betti_1_fraction'].median():.4f}.",
        f"- Real harmonic-ratio/Betti-1 correlation: {harmonic_betti_correlation:.4f}.",
    ]
    if closure_line:
        lines.append(closure_line)
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Topology filtration branch persistence](plots/topology_filtration_branch_persistence.png)",
            "",
            "![Topology filtration by prompt family](plots/topology_filtration_family_persistence.png)",
            "",
            "![Matched Betti-1 branch persistence](plots/topology_filtration_matched_betti.png)",
            "",
            "![Prompt-paired branch inference](plots/topology_filtration_prompt_inference.png)",
            "",
            "## Numerical Contract",
            "",
            f"- Maximum within-field exact-ratio range over the filtration: {exact_range.max():.3e}.",
            f"- Maximum within-field semantic-flow-ratio range: {flow_range.max():.3e}.",
        ]
    )
    if "reconstruction_error" in metrics.columns:
        lines.append(
            f"- Maximum reconstruction error: {metrics['reconstruction_error'].max():.3e}."
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "Harmonic at the open endpoint is the graph-cycle residual, not evidence",
            "of a global concept ring. A concept-ring claim requires harmonic energy",
            "that persists over a nontrivial geometric and topological interval,",
            "separates from matched nulls, and has an independent semantic or causal readout.",
            "",
            "## Filtration-Matched Gaps",
            "",
        ]
    )
    if not gaps.empty:
        for value in sorted(float(item) for item in gaps[x_column].unique()):
            subset = gaps[gaps[x_column] == value]
            lines.append(
                f"- {crossover_label}={format_filtration_value(value)}: median real-null gaps "
                f"exact={subset['exact_gap'].median():+.4f}, "
                f"coexact={subset['coexact_gap'].median():+.4f}, "
                f"harmonic={subset['harmonic_gap'].median():+.4f}."
            )
    if not matched_gaps.empty:
        lines.extend(["", "## Topology-Matched Pooled Gaps", ""])
        for target in sorted(matched_gaps["betti_1_target"].unique(), reverse=True):
            subset = matched_gaps[np.isclose(matched_gaps["betti_1_target"], target)]
            lines.append(
                f"- Betti-1={target:.2f}: median real-null gaps "
                f"exact={subset['exact_gap'].median():+.4f}, "
                f"coexact={subset['coexact_gap'].median():+.4f}, "
                f"harmonic={subset['harmonic_gap'].median():+.4f}."
            )
    overall = inference[inference["scope"] == "overall"] if not inference.empty else inference
    if not overall.empty:
        targets = sorted(float(value) for value in overall["betti_1_target"].unique())
        midpoint = min(targets, key=lambda value: abs(value - 0.5))
        lines.extend(
            [
                "",
                "## Prompt Bootstrap Gate",
                "",
                "Null seeds are averaged within each field, layer and k are averaged within",
                "each prompt, and prompts are resampled with a paired percentile bootstrap.",
                f"At matched Betti-1={midpoint:g}:",
                "",
            ]
        )
        midpoint_rows = overall[np.isclose(overall["betti_1_target"], midpoint)]
        for null_variant in sorted(midpoint_rows["null_variant"].unique()):
            for branch in ("harmonic", "coexact"):
                row = midpoint_rows[
                    (midpoint_rows["null_variant"] == null_variant)
                    & (midpoint_rows["branch"] == branch)
                ].iloc[0]
                lines.append(
                    f"- {NULL_LABELS.get(null_variant, null_variant)} {branch}: "
                    f"mean gap={row['mean_gap']:+.4f}, 95% prompt CI "
                    f"[{row['bootstrap_ci_lower']:+.4f}, {row['bootstrap_ci_upper']:+.4f}], "
                    f"positive prompts={int(round(row['positive_prompt_fraction'] * row['n_prompts']))}/"
                    f"{int(row['n_prompts'])}."
                )
    if not gaps_by_null.empty:
        lines.extend(
            [
                "",
                "The endpoint harmonic/coexact exchange is algebraically forced by the",
                "complex: harmonic owns the non-exact residual when no triangles are filled,",
                "and coexact owns it when the clique cycle space is fully spanned. The",
                "intermediate matched-Betti result is therefore the relevant persistence read.",
            ]
        )
    (output_root / "summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_all(*, metrics_path: Path, output_root: Path) -> None:
    metrics = pd.read_csv(metrics_path)
    x_column = filtration_axis_column(metrics)
    required = {
        "variant",
        "seed",
        "family",
        "prompt_id",
        "layer",
        "k",
        x_column,
        "betti_1_fraction",
        "harmonic_survival_ratio",
        *[f"{branch}_ratio" for branch in BRANCHES],
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"metrics CSV is missing columns: {', '.join(missing)}")
    plots = output_root / "plots"
    plot_branch_persistence(
        metrics,
        output_path=plots / "topology_filtration_branch_persistence.png",
    )
    plot_family_persistence(
        metrics,
        output_path=plots / "topology_filtration_family_persistence.png",
    )
    plot_matched_betti(
        metrics,
        output_path=plots / "topology_filtration_matched_betti.png",
    )
    summary = branch_summary(metrics)
    summary.to_csv(output_root / "summary_branch_persistence.csv", index=False)
    gaps = matched_real_null_gaps(metrics)
    gaps.to_csv(output_root / "summary_real_minus_null.csv", index=False)
    matched = matched_betti_summary(metrics)
    matched.to_csv(output_root / "summary_matched_betti.csv", index=False)
    matched_gaps = matched_betti_real_null_gaps(matched)
    matched_gaps.to_csv(output_root / "summary_matched_betti_gaps.csv", index=False)
    gaps_by_null = matched_betti_real_null_gaps_by_variant(matched)
    gaps_by_null.to_csv(
        output_root / "summary_matched_betti_gaps_by_null.csv",
        index=False,
    )
    inference = prompt_bootstrap_inference(gaps_by_null)
    inference.to_csv(output_root / "summary_prompt_bootstrap.csv", index=False)
    plot_prompt_inference(
        inference,
        output_path=plots / "topology_filtration_prompt_inference.png",
    )
    write_report(
        metrics,
        gaps,
        matched_gaps,
        gaps_by_null,
        inference,
        output_root=output_root,
    )
    print(f"saved plots: {plots}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    render_all(metrics_path=Path(args.metrics), output_root=Path(args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
