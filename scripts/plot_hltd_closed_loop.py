#!/usr/bin/env python3
"""Plot HLTD closed-loop branch summary outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CORE_COMPONENTS = [
    "presence_plus_coexact",
    "coexact_minus_presence",
    "presence",
    "coexact",
    "negative_coexact",
    "random_tangent",
]

COMPONENT_LABELS = {
    "presence_plus_coexact": "presence + coexact",
    "coexact_minus_presence": "coexact - presence",
    "presence": "presence",
    "coexact": "coexact",
    "negative_coexact": "-coexact",
    "random_tangent": "random tangent",
}

COMPACT_COMPONENT_LABELS = {
    "presence_plus_coexact": "pres+coex",
    "coexact_minus_presence": "coex-pres",
    "presence": "presence",
    "coexact": "coexact",
    "negative_coexact": "-coexact",
    "random_tangent": "random",
}

COMPONENT_COLORS = {
    "presence_plus_coexact": "#8064a2",
    "coexact_minus_presence": "#d65f5f",
    "presence": "#4c9a2a",
    "coexact": "#2878b5",
    "negative_coexact": "#6b6b6b",
    "random_tangent": "#6b6b6b",
}

METRIC_LABELS = {
    "token_drift_rate_mean": "token drift rate",
    "mean_selected_logprob_gain_mean": "selected-token logprob gain",
    "mean_kl_base_to_steered_mean": "KL(base||steered)",
    "mean_target_margin_delta_mean": "target margin delta",
}


def finite_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def symmetric_limit(values: Sequence[float], *, quantile: float = 0.98) -> float:
    vals = np.abs(finite_values(values))
    if vals.size == 0:
        return 1.0
    return max(float(np.quantile(vals, quantile)), 1e-6)


def load_tables(summary_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_path = summary_root / "closed_loop_component_summary.csv"
    contrast_path = summary_root / "closed_loop_contrasts.csv"
    if not component_path.exists():
        raise FileNotFoundError(component_path)
    if not contrast_path.exists():
        raise FileNotFoundError(contrast_path)
    return pd.read_csv(component_path), pd.read_csv(contrast_path)


def load_component_summary(summary_root: Path) -> pd.DataFrame:
    component_path = summary_root / "closed_loop_component_summary.csv"
    if not component_path.exists():
        raise FileNotFoundError(component_path)
    return pd.read_csv(component_path)


def load_step_table(summary_root: Path) -> pd.DataFrame:
    step_path = summary_root / "closed_loop_steps.csv"
    if not step_path.exists():
        return pd.DataFrame()
    return pd.read_csv(step_path)


def load_layer_summary(summary_root: Path) -> pd.DataFrame:
    layer_path = summary_root / "closed_loop_layer_summary.csv"
    if not layer_path.exists():
        return pd.DataFrame()
    return pd.read_csv(layer_path)


def load_k_summary(summary_root: Path) -> pd.DataFrame:
    k_path = summary_root / "closed_loop_k_summary.csv"
    if not k_path.exists():
        return pd.DataFrame()
    return pd.read_csv(k_path)


def load_prompt_summary(summary_root: Path) -> pd.DataFrame:
    prompt_path = summary_root / "closed_loop_prompt_summary.csv"
    if not prompt_path.exists():
        return pd.DataFrame()
    return pd.read_csv(prompt_path)


def load_prompt_layer_k_summary(summary_root: Path) -> pd.DataFrame:
    prompt_layer_k_path = summary_root / "closed_loop_prompt_layer_k_summary.csv"
    if not prompt_layer_k_path.exists():
        return pd.DataFrame()
    return pd.read_csv(prompt_layer_k_path)


def combined_component_summaries(summary_roots: Sequence[Path]) -> pd.DataFrame:
    tables = []
    for root in summary_roots:
        table = load_component_summary(root).copy()
        table["alpha"] = table["alpha"].astype(float)
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    combined = pd.concat(tables, ignore_index=True, sort=False)
    numeric_cols = [
        col
        for col in combined.columns
        if col not in {"component", "alpha"} and pd.api.types.is_numeric_dtype(combined[col])
    ]
    grouped = combined.groupby(["component", "alpha"], as_index=False)[numeric_cols].mean()
    return grouped.sort_values(["component", "alpha"]).reset_index(drop=True)


def available_components(table: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    present = {str(value) for value in table["component"].dropna().unique()}
    if requested:
        return [component for component in requested if component in present]
    return sorted(present)


def component_order_map(components: Sequence[str]) -> dict[str, int]:
    return {component: idx for idx, component in enumerate(components)}


def sorted_component_alpha_rows(table: pd.DataFrame, components: Sequence[str]) -> pd.DataFrame:
    components = available_components(table, components)
    order = component_order_map(components)
    rows = table[table["component"].isin(components)].copy()
    rows["_component_order"] = rows["component"].map(lambda value: order.get(str(value), len(order)))
    rows["_alpha_float"] = rows["alpha"].astype(float)
    return rows.sort_values(["_component_order", "_alpha_float"]).drop(columns=["_component_order", "_alpha_float"])


def run_label(component: str, alpha: float) -> str:
    label = COMPONENT_LABELS.get(component, component)
    return f"{label}\na={alpha:g}"


def plot_component_bars(
    component_summary: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    metrics = [
        "token_drift_rate_mean",
        "mean_selected_logprob_gain_mean",
        "mean_kl_base_to_steered_mean",
        "mean_target_margin_delta_mean",
    ]
    rows = sorted_component_alpha_rows(component_summary, components)
    labels = [run_label(str(row.component), float(row.alpha)) for row in rows.itertuples(index=False)]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.0), constrained_layout=True)
    x = np.arange(len(rows), dtype=float)

    for ax, metric in zip(axes.ravel(), metrics):
        values = rows[metric].astype(float).to_numpy() if metric in rows else np.full(len(rows), np.nan)
        colors = [COMPONENT_COLORS.get(str(component), "#999999") for component in rows["component"]]
        ax.bar(x, values, color=colors)
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("HLTD closed-loop branch metrics", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_drift_support_phase(
    contrasts: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    subset = sorted_component_alpha_rows(contrasts, components)
    if subset.empty:
        subset = contrasts.copy()
    fig, ax = plt.subplots(figsize=(8.0, 5.8), constrained_layout=True)
    y_values = subset["mean_selected_base_logprob"].astype(float)
    color_values = subset["mean_target_margin_delta"].astype(float)
    color_limit = symmetric_limit(color_values)
    y_range = float(np.nanmax(y_values) - np.nanmin(y_values)) if np.isfinite(y_values).any() else 1.0
    y_jitter = max(y_range * 0.025, 0.015)
    for idx, row in enumerate(subset.itertuples(index=False)):
        component = str(row.component)
        x = float(row.token_drift_rate)
        y = float(row.mean_selected_base_logprob)
        c = float(row.mean_target_margin_delta)
        ax.scatter(
            x,
            y,
            s=100 + 35 * max(float(row.alpha), 0.0),
            color=plt.cm.coolwarm(0.5 + 0.5 * np.clip(c / color_limit, -1.0, 1.0)),
            edgecolor=COMPONENT_COLORS.get(component, "#222222"),
            linewidth=1.8,
        )
        label = f"{COMPONENT_LABELS.get(component, component)} a={float(row.alpha):g} {str(row.prompt_id)}"
        text_y = y + (idx - (len(subset) - 1) / 2.0) * y_jitter
        ax.text(x + 0.015, text_y, label, fontsize=8, va="center")
    ax.axvline(0.5, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.03, 1.03)
    ymin = float(np.nanmin(y_values)) if np.isfinite(y_values).any() else -5.0
    ymax = float(np.nanmax(y_values)) if np.isfinite(y_values).any() else 0.0
    margin = max((ymax - ymin) * 0.2, 0.2)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlabel("token drift rate vs greedy baseline")
    ax.set_ylabel("mean selected-token base logprob")
    ax.set_title("closed-loop drift/support phase")
    ax.grid(alpha=0.25)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-color_limit, vmax=color_limit))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="target margin delta")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_alpha_response(
    component_summary: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    rows = sorted_component_alpha_rows(component_summary, components)
    metrics = [
        ("token drift", "token_drift_rate_mean"),
        ("logprob gain", "mean_selected_logprob_gain_mean"),
        ("target margin", "mean_target_margin_delta_mean"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.2), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (title, metric) in zip(axes, metrics):
        for component in available_components(rows, components):
            comp_rows = rows[rows["component"] == component].sort_values("alpha")
            if comp_rows.empty:
                continue
            ax.plot(
                comp_rows["alpha"].astype(float),
                comp_rows[metric].astype(float),
                marker="o",
                color=COMPONENT_COLORS.get(component, "#999999"),
                label=COMPONENT_LABELS.get(component, component),
            )
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.set_xlabel("alpha")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("HLTD closed-loop alpha response", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def first_drift_transition_band(rows: pd.DataFrame) -> tuple[float, float] | None:
    ordered = rows.sort_values("alpha")
    drift = ordered["token_drift_rate_mean"].astype(float).to_numpy()
    alphas = ordered["alpha"].astype(float).to_numpy()
    positive = np.flatnonzero(drift > 0.0)
    if positive.size == 0:
        return None
    first = int(positive[0])
    if first == 0:
        return (float(alphas[first]), float(alphas[first]))
    return (float(alphas[first - 1]), float(alphas[first]))


def plot_alpha_transition_band(
    component_summary: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    rows = sorted_component_alpha_rows(component_summary, components)
    if rows.empty:
        raise ValueError("No component rows available for alpha transition plot.")
    metrics = [
        ("token drift rate", "token_drift_rate_mean"),
        ("target margin delta", "mean_target_margin_delta_mean"),
        ("KL(base||steered)", "mean_kl_base_to_steered_mean"),
        ("selected-token logprob gain", "mean_selected_logprob_gain_mean"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.3), constrained_layout=True)
    seen_transition: tuple[float, float, str] | None = None

    for ax, (title, metric) in zip(axes.ravel(), metrics):
        for component in available_components(rows, components):
            comp_rows = rows[rows["component"] == component].sort_values("alpha")
            if comp_rows.empty or metric not in comp_rows:
                continue
            color = COMPONENT_COLORS.get(component, "#999999")
            label = COMPONENT_LABELS.get(component, component)
            ax.plot(
                comp_rows["alpha"].astype(float),
                comp_rows[metric].astype(float),
                marker="o",
                color=color,
                linewidth=1.6,
                label=label,
            )
            band = first_drift_transition_band(comp_rows)
            if band is not None:
                lo, hi = band
                ax.axvspan(lo, hi, color=color, alpha=0.08, linewidth=0)
                ax.axvline(hi, color=color, linestyle="--", linewidth=0.9, alpha=0.75)
                if seen_transition is None and component == "coexact_minus_presence":
                    seen_transition = (lo, hi, label)
        ax.axhline(0.0, color="#464C55", linewidth=0.8)
        ax.set_xlabel("alpha")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    if seen_transition is not None:
        lo, hi, label = seen_transition
        axes[0, 0].annotate(
            f"first observed {label} break: {lo:g}-{hi:g}",
            xy=(hi, 1.0),
            xytext=(hi + 0.035, 0.72),
            arrowprops={"arrowstyle": "->", "color": "#464C55", "linewidth": 0.8},
            fontsize=8,
            color="#1F2430",
        )
    axes[1, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("HLTD closed-loop alpha transition band", fontsize=13)
    fig.text(
        0.01,
        0.955,
        "Combined broad and narrow sweeps; shaded bands mark the first alpha interval where token drift appears.",
        fontsize=9,
        color="#6F768A",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def step_series_label(component: str, alpha: float) -> str:
    if component == "baseline":
        return "baseline"
    return run_label(component, alpha).replace("\n", " ")


def plot_step_traces(
    steps: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    if steps.empty:
        raise ValueError("No step rows available for closed-loop step traces.")
    requested = ["baseline", *components]
    present = {str(value) for value in steps["component"].dropna().unique()}
    chosen = [component for component in requested if component in present]
    if not chosen:
        chosen = sorted(present)
    rows = steps[steps["component"].isin(chosen)].copy()
    rows["alpha"] = rows["alpha"].astype(float)
    rows["step"] = rows["step"].astype(int)
    metrics = [
        ("target margin delta", "target_margin_delta"),
        ("KL(base||steered)", "kl_base_to_steered"),
        ("nearest-node distance", "nearest_distance"),
        ("top-token changed rate", "top_changed"),
    ]
    agg = (
        rows.groupby(["component", "alpha", "step"], as_index=False)[[metric for _, metric in metrics]]
        .mean(numeric_only=True)
        .sort_values(["component", "alpha", "step"])
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.1), constrained_layout=True)
    for ax, (title, metric) in zip(axes.ravel(), metrics):
        for (component, alpha), part in agg.groupby(["component", "alpha"], sort=False):
            component = str(component)
            color = "#464C55" if component == "baseline" else COMPONENT_COLORS.get(component, "#999999")
            linestyle = "--" if component == "baseline" else "-"
            ax.plot(
                part["step"].astype(int),
                part[metric].astype(float),
                marker="o",
                linewidth=1.4,
                linestyle=linestyle,
                color=color,
                label=step_series_label(component, float(alpha)),
            )
        ax.axhline(0.0, color="#464C55", linewidth=0.8)
        ax.set_xlabel("generated step")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("HLTD closed-loop step traces", fontsize=13)
    fig.text(
        0.01,
        0.955,
        "Step-level means from closed-loop branch steering; nearest distance tracks chart lookup stability.",
        fontsize=9,
        color="#6F768A",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_layer_response(
    layer_summary: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    if layer_summary.empty:
        raise ValueError("No layer summary rows available for closed-loop layer response.")
    rows = sorted_component_alpha_rows(layer_summary, components)
    rows["layer"] = rows["layer"].astype(int)
    if rows["layer"].nunique() < 2:
        raise ValueError("Layer response plot requires at least two layers.")
    metrics = [
        ("token drift", "token_drift_rate_mean"),
        ("target margin", "mean_target_margin_delta_mean"),
        ("selected-token gain", "mean_selected_logprob_gain_mean"),
        ("KL(base||steered)", "mean_kl_base_to_steered_mean"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), constrained_layout=True)
    for ax, (title, metric) in zip(axes.ravel(), metrics):
        for component in available_components(rows, components):
            comp_rows = rows[rows["component"] == component].sort_values(["alpha", "layer"])
            for alpha, alpha_rows in comp_rows.groupby("alpha", sort=True):
                label = run_label(component, float(alpha)).replace("\n", " ")
                ax.plot(
                    alpha_rows["layer"].astype(int),
                    alpha_rows[metric].astype(float),
                    marker="o",
                    linewidth=1.4,
                    color=COMPONENT_COLORS.get(component, "#999999"),
                    label=label,
                )
        ax.axhline(0.0, color="#464C55", linewidth=0.8)
        ax.set_xlabel("layer")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = set()
    dedup_handles = []
    dedup_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        dedup_handles.append(handle)
        dedup_labels.append(label)
    axes[1, 1].legend(dedup_handles, dedup_labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("HLTD closed-loop layer response", fontsize=13)
    fig.text(
        0.01,
        0.955,
        "Layer-wise means across closed-loop branch runs; compare traversal drift against semantic-target margin.",
        fontsize=9,
        color="#6F768A",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_k_response(
    k_summary: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    if k_summary.empty:
        raise ValueError("No k summary rows available for closed-loop k response.")
    rows = sorted_component_alpha_rows(k_summary, components)
    rows["k"] = rows["k"].astype(int)
    if rows["k"].nunique() < 2:
        raise ValueError("k response plot requires at least two k values.")
    metrics = [
        ("token drift", "token_drift_rate_mean"),
        ("target margin", "mean_target_margin_delta_mean"),
        ("selected-token gain", "mean_selected_logprob_gain_mean"),
        ("KL(base||steered)", "mean_kl_base_to_steered_mean"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), constrained_layout=True)
    for ax, (title, metric) in zip(axes.ravel(), metrics):
        for component in available_components(rows, components):
            comp_rows = rows[rows["component"] == component].sort_values(["alpha", "k"])
            for alpha, alpha_rows in comp_rows.groupby("alpha", sort=True):
                label = run_label(component, float(alpha)).replace("\n", " ")
                ax.plot(
                    alpha_rows["k"].astype(int),
                    alpha_rows[metric].astype(float),
                    marker="o",
                    linewidth=1.4,
                    color=COMPONENT_COLORS.get(component, "#999999"),
                    label=label,
                )
        ax.axhline(0.0, color="#464C55", linewidth=0.8)
        ax.set_xlabel("k")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = set()
    dedup_handles = []
    dedup_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        dedup_handles.append(handle)
        dedup_labels.append(label)
    axes[1, 1].legend(dedup_handles, dedup_labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("HLTD closed-loop k response", fontsize=13)
    fig.text(
        0.01,
        0.955,
        "k-wise means across closed-loop branch runs; compare graph-neighborhood robustness.",
        fontsize=9,
        color="#6F768A",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def choose_threshold_component(k_summary: pd.DataFrame, components: Sequence[str]) -> str:
    ordered = available_components(k_summary, components)
    if "coexact_minus_presence" in ordered:
        return "coexact_minus_presence"
    if ordered:
        return ordered[0]
    present = sorted(str(value) for value in k_summary["component"].dropna().unique())
    if not present:
        raise ValueError("No components available for alpha-k threshold plot.")
    return present[0]


def pivot_alpha_k_metric(rows: pd.DataFrame, metric: str) -> tuple[np.ndarray, list[float], list[int]]:
    alphas = sorted(float(value) for value in rows["alpha"].dropna().unique())
    ks = sorted(int(value) for value in rows["k"].dropna().unique())
    pivot = rows.pivot_table(index="k", columns="alpha", values=metric, aggfunc="mean")
    pivot = pivot.reindex(index=ks, columns=alphas)
    return pivot.to_numpy(dtype=float), alphas, ks


def pivot_component_alpha_metric(
    rows: pd.DataFrame,
    *,
    metric: str,
    components: Sequence[str],
    alphas: Sequence[float],
) -> np.ndarray:
    pivot = rows.pivot_table(index="component", columns="alpha", values=metric, aggfunc="mean")
    pivot = pivot.reindex(index=list(components), columns=list(alphas))
    return pivot.to_numpy(dtype=float)


def prompt_component_alpha_columns(rows: pd.DataFrame, components: Sequence[str]) -> list[tuple[str, float]]:
    chosen = available_components(rows, components)
    columns: list[tuple[str, float]] = []
    for component in chosen:
        comp_rows = rows[rows["component"] == component]
        for alpha in sorted(float(value) for value in comp_rows["alpha"].dropna().unique()):
            columns.append((component, alpha))
    return columns


def annotate_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    cmap: Optional[matplotlib.colors.Colormap] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    finite = finite_values(values.ravel())
    if finite.size == 0:
        return
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            color = "#1F2430"
            if cmap is not None and vmin is not None and vmax is not None and vmax > vmin:
                scaled = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
                red, green, blue, _ = cmap(float(scaled))
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                color = "white" if luminance < 0.48 else "#1F2430"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)


def plot_alpha_k_threshold(
    k_summary: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    if k_summary.empty:
        raise ValueError("No k summary rows available for closed-loop alpha-k threshold plot.")
    rows = sorted_component_alpha_rows(k_summary, components)
    rows["k"] = rows["k"].astype(int)
    rows["alpha"] = rows["alpha"].astype(float)
    if rows["k"].nunique() < 2 or rows["alpha"].nunique() < 2:
        raise ValueError("Alpha-k threshold plot requires at least two k values and two alphas.")
    component = choose_threshold_component(rows, components)
    comp_rows = rows[rows["component"] == component].copy()
    if comp_rows["k"].nunique() < 2 or comp_rows["alpha"].nunique() < 2:
        raise ValueError(f"Component {component!r} does not have a full alpha-k threshold surface.")

    metrics = [
        ("token drift rate", "token_drift_rate_mean", "viridis", 0.0, 1.0),
        ("target margin delta", "mean_target_margin_delta_mean", "coolwarm", None, None),
        ("KL(base||steered)", "mean_kl_base_to_steered_mean", "magma", 0.0, None),
        ("selected-token logprob gain", "mean_selected_logprob_gain_mean", "coolwarm", None, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), constrained_layout=True)

    for ax, (title, metric, cmap_name, explicit_vmin, explicit_vmax) in zip(axes.ravel(), metrics):
        values, alphas, ks = pivot_alpha_k_metric(comp_rows, metric)
        if cmap_name == "coolwarm":
            limit = symmetric_limit(values.ravel())
            vmin, vmax = -limit, limit
        else:
            finite = finite_values(values.ravel())
            vmin = explicit_vmin if explicit_vmin is not None else float(np.nanmin(finite))
            vmax = explicit_vmax if explicit_vmax is not None else float(np.nanmax(finite))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
        cmap = plt.colormaps[cmap_name].copy()
        cmap.set_bad("#F0F1F4")
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        annotate_heatmap(ax, values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(alphas)), [f"{alpha:g}" for alpha in alphas])
        ax.set_yticks(np.arange(len(ks)), [str(k) for k in ks])
        ax.set_xlabel("alpha")
        ax.set_ylabel("k")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    label = COMPONENT_LABELS.get(component, component)
    fig.suptitle(f"HLTD closed-loop alpha-k threshold surface: {label}", fontsize=13)
    fig.text(
        0.01,
        0.955,
        "Cells are means from closed-loop runs; the threshold pattern should be read against target margin and KL, not drift alone.",
        fontsize=9,
        color="#6F768A",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_alpha_k_branch_map(
    k_summary: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    if k_summary.empty:
        raise ValueError("No k summary rows available for closed-loop alpha-k branch map.")
    rows = sorted_component_alpha_rows(k_summary, components)
    rows["k"] = rows["k"].astype(int)
    rows["alpha"] = rows["alpha"].astype(float)
    chosen_components = available_components(rows, components)
    if len(chosen_components) < 2:
        raise ValueError("Alpha-k branch map requires at least two components.")
    ks = sorted(int(value) for value in rows["k"].dropna().unique())
    alphas = sorted(float(value) for value in rows["alpha"].dropna().unique())
    if len(ks) < 2 or len(alphas) < 2:
        raise ValueError("Alpha-k branch map requires at least two k values and two alphas.")

    metrics = [
        ("token drift rate", "token_drift_rate_mean", "viridis", 0.0, 1.0),
        ("target margin delta", "mean_target_margin_delta_mean", "coolwarm", None, None),
    ]
    fig, axes = plt.subplots(
        len(metrics),
        len(ks),
        figsize=(4.35 * len(ks), 2.15 * len(metrics) + 2.7),
        constrained_layout=True,
        squeeze=False,
    )

    for row_idx, (title, metric, cmap_name, explicit_vmin, explicit_vmax) in enumerate(metrics):
        all_values = rows[metric].astype(float).to_numpy()
        if cmap_name == "coolwarm":
            limit = symmetric_limit(all_values)
            vmin, vmax = -limit, limit
        else:
            finite = finite_values(all_values)
            vmin = explicit_vmin if explicit_vmin is not None else float(np.nanmin(finite))
            vmax = explicit_vmax if explicit_vmax is not None else float(np.nanmax(finite))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
        cmap = plt.colormaps[cmap_name].copy()
        cmap.set_bad("#F0F1F4")
        row_images = []
        for col_idx, k in enumerate(ks):
            ax = axes[row_idx, col_idx]
            k_rows = rows[rows["k"] == k]
            values = pivot_component_alpha_metric(
                k_rows,
                metric=metric,
                components=chosen_components,
                alphas=alphas,
            )
            image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            row_images.append(image)
            annotate_heatmap(ax, values, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(alphas)), [f"{alpha:g}" for alpha in alphas])
            ax.set_yticks(
                np.arange(len(chosen_components)),
                [COMPONENT_LABELS.get(component, component) for component in chosen_components],
            )
            ax.set_xlabel("alpha")
            if col_idx == 0:
                ax.set_ylabel(title)
            ax.set_title(f"k={k}")
            ax.grid(False)
        fig.colorbar(row_images[-1], ax=axes[row_idx, :].tolist(), fraction=0.025, pad=0.02)

    fig.suptitle("HLTD closed-loop branch alpha-k map", fontsize=13)
    fig.text(
        0.01,
        0.955,
        "Each panel compares branches at one graph-neighborhood size; target margin distinguishes semantic pressure from surface drift.",
        fontsize=9,
        color="#6F768A",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def prompt_metric_matrix(
    prompt_summary: pd.DataFrame,
    *,
    prompts: Sequence[str],
    columns: Sequence[tuple[str, float]],
    metric: str,
) -> np.ndarray:
    values = np.full((len(prompts), len(columns)), np.nan, dtype=float)
    for col_idx, (component, alpha) in enumerate(columns):
        sub = prompt_summary[
            (prompt_summary["component"] == component)
            & np.isclose(prompt_summary["alpha"].astype(float), float(alpha))
        ]
        by_prompt = sub.set_index("prompt_id")[metric].astype(float).to_dict()
        for row_idx, prompt in enumerate(prompts):
            if prompt in by_prompt:
                values[row_idx, col_idx] = float(by_prompt[prompt])
    return values


def plot_prompt_branch_gate(
    prompt_summary: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    if prompt_summary.empty:
        raise ValueError("No prompt summary rows available for closed-loop prompt branch gate.")
    rows = sorted_component_alpha_rows(prompt_summary, components)
    rows["alpha"] = rows["alpha"].astype(float)
    prompts = sorted(str(value) for value in rows["prompt_id"].dropna().unique())
    columns = prompt_component_alpha_columns(rows, components)
    if len(prompts) < 2 or len(columns) < 2:
        raise ValueError("Prompt branch gate plot requires at least two prompts and two branch columns.")

    metrics = [
        ("branch gate rate", "branch_gate_rate", "viridis", 0.0, 1.0),
        ("token drift rate", "token_drift_rate_mean", "viridis", 0.0, 1.0),
        ("target margin delta", "mean_target_margin_delta_mean", "coolwarm", None, None),
    ]
    fig_width = max(9.5, 0.62 * len(columns) + 4.2)
    fig_height = max(8.0, 0.42 * len(prompts) * len(metrics) + 4.2)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(fig_width, fig_height), constrained_layout=False)
    fig.subplots_adjust(left=0.26, right=0.9, top=0.84, bottom=0.24, hspace=0.75)
    if len(metrics) == 1:
        axes = [axes]
    xlabels = [f"{COMPACT_COMPONENT_LABELS.get(component, component)}\na={alpha:g}" for component, alpha in columns]
    ylabels = []
    for prompt in prompts:
        family_values = rows.loc[rows["prompt_id"] == prompt, "family"].dropna().unique()
        family = str(family_values[0]) if len(family_values) else ""
        ylabels.append(f"{family}/{prompt}" if family else prompt)

    for ax_idx, (ax, (title, metric, cmap_name, explicit_vmin, explicit_vmax)) in enumerate(zip(axes, metrics)):
        values = prompt_metric_matrix(rows, prompts=prompts, columns=columns, metric=metric)
        if cmap_name == "coolwarm":
            limit = symmetric_limit(values.ravel())
            vmin, vmax = -limit, limit
        else:
            finite = finite_values(values.ravel())
            vmin = explicit_vmin if explicit_vmin is not None else float(np.nanmin(finite))
            vmax = explicit_vmax if explicit_vmax is not None else float(np.nanmax(finite))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
        cmap = plt.colormaps[cmap_name].copy()
        cmap.set_bad("#F0F1F4")
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        annotate_heatmap(ax, values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(columns)))
        if ax_idx == len(metrics) - 1:
            ax.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=9)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_yticks(np.arange(len(prompts)), ylabels)
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)

    fig.suptitle("HLTD closed-loop prompt branch gate", fontsize=13, y=0.965)
    fig.text(
        0.01,
        0.92,
        "Branch gate rate is the fraction of prompt/layer/k/seed cells with drift >= 0.5 and positive target margin.",
        fontsize=9,
        color="#6F768A",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def prompt_summary_has_random_advantage(prompt_summary: pd.DataFrame) -> bool:
    required = {
        "branch_specific_gate_rate",
        "branch_gate_minus_random_rate",
        "token_drift_rate_minus_random_mean",
        "mean_target_margin_delta_minus_random_mean",
    }
    return required.issubset(set(prompt_summary.columns))


def plot_prompt_random_advantage(
    prompt_summary: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    if prompt_summary.empty:
        raise ValueError("No prompt summary rows available for closed-loop prompt random advantage.")
    if not prompt_summary_has_random_advantage(prompt_summary):
        raise ValueError("Prompt random advantage plot requires random-difference summary columns.")
    rows = sorted_component_alpha_rows(prompt_summary, components)
    rows["alpha"] = rows["alpha"].astype(float)
    prompts = sorted(str(value) for value in rows["prompt_id"].dropna().unique())
    columns = prompt_component_alpha_columns(rows, components)
    if len(prompts) < 2 or len(columns) < 2:
        raise ValueError("Prompt random advantage plot requires at least two prompts and two branch columns.")

    metrics = [
        ("branch-specific gate", "branch_specific_gate_rate", "viridis", 0.0, 1.0),
        ("branch gate - random", "branch_gate_minus_random_rate", "coolwarm", -1.0, 1.0),
        ("token drift - random", "token_drift_rate_minus_random_mean", "coolwarm", -1.0, 1.0),
        ("target margin - random", "mean_target_margin_delta_minus_random_mean", "coolwarm", None, None),
    ]
    fig_width = max(9.5, 0.62 * len(columns) + 4.2)
    fig_height = max(8.0, 0.42 * len(prompts) * len(metrics) + 4.2)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(fig_width, fig_height), constrained_layout=False)
    fig.subplots_adjust(left=0.26, right=0.9, top=0.84, bottom=0.24, hspace=0.75)
    if len(metrics) == 1:
        axes = [axes]
    xlabels = [f"{COMPACT_COMPONENT_LABELS.get(component, component)}\na={alpha:g}" for component, alpha in columns]
    ylabels = []
    for prompt in prompts:
        family_values = rows.loc[rows["prompt_id"] == prompt, "family"].dropna().unique()
        family = str(family_values[0]) if len(family_values) else ""
        ylabels.append(f"{family}/{prompt}" if family else prompt)

    for ax_idx, (ax, (title, metric, cmap_name, explicit_vmin, explicit_vmax)) in enumerate(zip(axes, metrics)):
        values = prompt_metric_matrix(rows, prompts=prompts, columns=columns, metric=metric)
        if cmap_name == "coolwarm":
            if explicit_vmin is not None and explicit_vmax is not None:
                vmin, vmax = explicit_vmin, explicit_vmax
            else:
                limit = symmetric_limit(values.ravel())
                vmin, vmax = -limit, limit
        else:
            finite = finite_values(values.ravel())
            vmin = explicit_vmin if explicit_vmin is not None else float(np.nanmin(finite))
            vmax = explicit_vmax if explicit_vmax is not None else float(np.nanmax(finite))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
        cmap = plt.colormaps[cmap_name].copy()
        cmap.set_bad("#F0F1F4")
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        annotate_heatmap(ax, values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(columns)))
        if ax_idx == len(metrics) - 1:
            ax.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=9)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_yticks(np.arange(len(prompts)), ylabels)
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)

    fig.suptitle("HLTD closed-loop component-minus-random advantage", fontsize=13, y=0.965)
    fig.text(
        0.01,
        0.92,
        "Positive cells beat the matched random tangent for the same prompt/layer/k/seed/alpha; zero means no branch-specific advantage.",
        fontsize=9,
        color="#6F768A",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def prompt_layer_k_metric_matrix(
    rows: pd.DataFrame,
    *,
    row_keys: Sequence[tuple[str, str]],
    cells: Sequence[tuple[int, int]],
    metric: str,
) -> np.ndarray:
    values = np.full((len(row_keys), len(cells)), np.nan, dtype=float)
    for row_idx, (prompt_id, component) in enumerate(row_keys):
        sub = rows[
            (rows["prompt_id"].astype(str) == prompt_id)
            & (rows["component"].astype(str) == component)
        ]
        lookup = {
            (int(row.layer), int(row.k)): float(getattr(row, metric))
            for row in sub.itertuples(index=False)
            if hasattr(row, metric)
        }
        for col_idx, cell in enumerate(cells):
            if cell in lookup:
                values[row_idx, col_idx] = lookup[cell]
    return values


def prompt_k_alpha_metric_matrix(
    rows: pd.DataFrame,
    *,
    row_keys: Sequence[tuple[str, str]],
    cells: Sequence[tuple[int, float]],
    metric: str,
) -> np.ndarray:
    values = np.full((len(row_keys), len(cells)), np.nan, dtype=float)
    for row_idx, (prompt_id, component) in enumerate(row_keys):
        sub = rows[
            (rows["prompt_id"].astype(str) == prompt_id)
            & (rows["component"].astype(str) == component)
        ]
        lookup = {
            (int(row.k), float(row.alpha)): float(getattr(row, metric))
            for row in sub.itertuples(index=False)
            if hasattr(row, metric)
        }
        for col_idx, cell in enumerate(cells):
            if cell in lookup:
                values[row_idx, col_idx] = lookup[cell]
    return values


def plot_prompt_layer_alpha_k_surface(
    prompt_layer_k_summary: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    if prompt_layer_k_summary.empty:
        raise ValueError("No prompt layer-k summary rows available for closed-loop branch surface.")
    rows = sorted_component_alpha_rows(prompt_layer_k_summary, components)
    rows["layer"] = rows["layer"].astype(int)
    rows["k"] = rows["k"].astype(int)
    rows["alpha"] = rows["alpha"].astype(float)
    if rows["alpha"].nunique() < 2:
        raise ValueError("Prompt layer alpha-k surface requires at least two alphas.")

    chosen_components = available_components(rows, components)
    prompts = sorted(str(value) for value in rows["prompt_id"].dropna().unique())
    row_keys = [
        (prompt_id, component)
        for prompt_id in prompts
        for component in chosen_components
        if not rows[
            (rows["prompt_id"].astype(str) == prompt_id)
            & (rows["component"].astype(str) == component)
        ].empty
    ]
    layers = sorted(int(value) for value in rows["layer"].dropna().unique())
    cells = sorted(
        {
            (int(row.k), float(row.alpha))
            for row in rows[["k", "alpha"]].drop_duplicates().itertuples(index=False)
        }
    )
    if not row_keys or not layers or len(cells) < 2:
        raise ValueError("Prompt layer alpha-k surface requires rows, layers, and two k-alpha cells.")

    metrics = [
        ("branch-specific gate", "branch_specific_gate_rate", "viridis", 0.0, 1.0),
        ("target margin - random", "mean_target_margin_delta_minus_random_mean", "coolwarm", None, None),
    ]
    fig, axes = plt.subplots(
        len(metrics),
        len(layers),
        figsize=(max(12.4, 4.9 * len(layers)), max(7.6, 1.05 * len(row_keys) + 5.2)),
        constrained_layout=False,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.17, right=0.91, top=0.87, bottom=0.16, hspace=0.38, wspace=0.2)
    xlabels = [f"k{k}\na={alpha:g}" for k, alpha in cells]
    ylabels = [
        f"{prompt_id} / {COMPACT_COMPONENT_LABELS.get(component, component)}"
        for prompt_id, component in row_keys
    ]

    for metric_idx, (title, metric, cmap_name, explicit_vmin, explicit_vmax) in enumerate(metrics):
        all_values = rows[metric].astype(float).to_numpy()
        if cmap_name == "coolwarm":
            limit = symmetric_limit(all_values)
            vmin, vmax = -limit, limit
        else:
            finite = finite_values(all_values)
            vmin = explicit_vmin if explicit_vmin is not None else float(np.nanmin(finite))
            vmax = explicit_vmax if explicit_vmax is not None else float(np.nanmax(finite))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
        cmap = plt.colormaps[cmap_name].copy()
        cmap.set_bad("#F0F1F4")
        images = []
        for layer_idx, layer in enumerate(layers):
            ax = axes[metric_idx, layer_idx]
            layer_rows = rows[rows["layer"] == layer]
            values = prompt_k_alpha_metric_matrix(
                layer_rows,
                row_keys=row_keys,
                cells=cells,
                metric=metric,
            )
            image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            images.append(image)
            annotate_heatmap(ax, values, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(cells)))
            if metric_idx == len(metrics) - 1:
                ax.set_xticklabels(xlabels, fontsize=8)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ax.set_yticks(np.arange(len(row_keys)))
            if layer_idx == 0:
                ax.set_yticklabels(ylabels, fontsize=8)
                ax.set_ylabel(title)
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_title(f"Layer {layer}")
            ax.grid(False)
        fig.colorbar(images[-1], ax=axes[metric_idx, :].tolist(), fraction=0.018, pad=0.015)

    fig.suptitle("HLTD closed-loop branch robustness surface", fontsize=13, y=0.965)
    fig.text(
        0.01,
        0.925,
        "Each deterministic branch is evaluated against matched random-tangent seed strata; repeated branch rows are not independent trajectories.",
        fontsize=9,
        color="#6F768A",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_prompt_layer_k_map(
    prompt_layer_k_summary: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    if prompt_layer_k_summary.empty:
        raise ValueError("No prompt layer-k summary rows available for closed-loop prompt layer-k map.")
    rows = sorted_component_alpha_rows(prompt_layer_k_summary, components)
    rows["layer"] = rows["layer"].astype(int)
    rows["k"] = rows["k"].astype(int)
    rows["alpha"] = rows["alpha"].astype(float)
    if rows["layer"].nunique() < 2 and rows["k"].nunique() < 2:
        raise ValueError("Prompt layer-k map requires at least two layer-k cells.")

    chosen_components = available_components(rows, components)
    prompts = sorted(str(value) for value in rows["prompt_id"].dropna().unique())
    row_keys = [
        (prompt_id, component)
        for prompt_id in prompts
        for component in chosen_components
        if not rows[
            (rows["prompt_id"].astype(str) == prompt_id)
            & (rows["component"].astype(str) == component)
        ].empty
    ]
    cells = sorted(
        {
            (int(row.layer), int(row.k))
            for row in rows[["layer", "k"]].drop_duplicates().itertuples(index=False)
        }
    )
    if len(row_keys) < 1 or len(cells) < 2:
        raise ValueError("Prompt layer-k map requires at least one row and two layer-k cells.")

    metrics = [
        ("branch-specific gate", "branch_specific_gate_rate", "viridis", 0.0, 1.0),
        ("target margin - random", "mean_target_margin_delta_minus_random_mean", "coolwarm", None, None),
    ]
    fig_width = max(11.5, 0.86 * len(cells) + 4.4)
    fig_height = max(5.8, 0.48 * len(row_keys) * len(metrics) + 3.4)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(fig_width, fig_height), constrained_layout=False)
    fig.subplots_adjust(left=0.24, right=0.9, top=0.86, bottom=0.2, hspace=0.74)
    if len(metrics) == 1:
        axes = [axes]

    xlabels = [f"L{layer}\nk{k}" for layer, k in cells]
    ylabels = [
        f"{prompt_id}\n{COMPACT_COMPONENT_LABELS.get(component, component)}"
        for prompt_id, component in row_keys
    ]
    for ax_idx, (ax, (title, metric, cmap_name, explicit_vmin, explicit_vmax)) in enumerate(zip(axes, metrics)):
        values = prompt_layer_k_metric_matrix(rows, row_keys=row_keys, cells=cells, metric=metric)
        if cmap_name == "coolwarm":
            limit = symmetric_limit(values.ravel())
            vmin, vmax = -limit, limit
        else:
            finite = finite_values(values.ravel())
            vmin = explicit_vmin if explicit_vmin is not None else float(np.nanmin(finite))
            vmax = explicit_vmax if explicit_vmax is not None else float(np.nanmax(finite))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
        cmap = plt.colormaps[cmap_name].copy()
        cmap.set_bad("#F0F1F4")
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        annotate_heatmap(ax, values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(cells)))
        if ax_idx == len(metrics) - 1:
            ax.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=9)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_yticks(np.arange(len(row_keys)), ylabels)
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)

    for row_idx in range(1, len(row_keys)):
        if row_keys[row_idx][0] != row_keys[row_idx - 1][0]:
            for ax in axes:
                ax.axhline(row_idx - 0.5, color="white", linewidth=1.6)

    fig.suptitle("HLTD closed-loop prompt layer-k map", fontsize=13, y=0.965)
    fig.text(
        0.01,
        0.92,
        "Rows are prompt/component pairs; cells show where a branch survives each layer/k setting.",
        fontsize=9,
        color="#6F768A",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_plots(
    *,
    summary_root: Path,
    output_dir: Path,
    components: Sequence[str],
    comparison_summary_roots: Sequence[Path] = (),
) -> List[Path]:
    component_summary, contrasts = load_tables(summary_root)
    steps = load_step_table(summary_root)
    layer_summary = load_layer_summary(summary_root)
    k_summary = load_k_summary(summary_root)
    prompt_summary = load_prompt_summary(summary_root)
    prompt_layer_k_summary = load_prompt_layer_k_summary(summary_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = [
        output_dir / "closed_loop_component_bars.png",
        output_dir / "closed_loop_drift_support_phase.png",
        output_dir / "closed_loop_alpha_response.png",
    ]
    plot_component_bars(component_summary, output_path=saved[0], components=components)
    plot_drift_support_phase(contrasts, output_path=saved[1], components=components)
    plot_alpha_response(component_summary, output_path=saved[2], components=components)
    if not steps.empty:
        step_trace_path = output_dir / "closed_loop_step_traces.png"
        plot_step_traces(steps, output_path=step_trace_path, components=components)
        saved.append(step_trace_path)
    if not layer_summary.empty and layer_summary["layer"].nunique() > 1:
        layer_path = output_dir / "closed_loop_layer_response.png"
        plot_layer_response(layer_summary, output_path=layer_path, components=components)
        saved.append(layer_path)
    if not k_summary.empty and k_summary["k"].nunique() > 1:
        k_path = output_dir / "closed_loop_k_response.png"
        plot_k_response(k_summary, output_path=k_path, components=components)
        saved.append(k_path)
        if k_summary["alpha"].nunique() > 1:
            alpha_k_path = output_dir / "closed_loop_alpha_k_threshold.png"
            plot_alpha_k_threshold(k_summary, output_path=alpha_k_path, components=components)
            saved.append(alpha_k_path)
            if len(available_components(k_summary, components)) > 1:
                branch_map_path = output_dir / "closed_loop_alpha_k_branch_map.png"
                plot_alpha_k_branch_map(k_summary, output_path=branch_map_path, components=components)
                saved.append(branch_map_path)
    if not prompt_summary.empty and prompt_summary["prompt_id"].nunique() > 1:
        prompt_gate_path = output_dir / "closed_loop_prompt_branch_gate.png"
        plot_prompt_branch_gate(prompt_summary, output_path=prompt_gate_path, components=components)
        saved.append(prompt_gate_path)
        if prompt_summary_has_random_advantage(prompt_summary):
            prompt_random_path = output_dir / "closed_loop_prompt_random_advantage.png"
            plot_prompt_random_advantage(prompt_summary, output_path=prompt_random_path, components=components)
            saved.append(prompt_random_path)
    if (
        not prompt_layer_k_summary.empty
        and (
            prompt_layer_k_summary["layer"].nunique() > 1
            or prompt_layer_k_summary["k"].nunique() > 1
        )
    ):
        if prompt_layer_k_summary["alpha"].nunique() > 1:
            prompt_surface_path = output_dir / "closed_loop_prompt_layer_alpha_k_surface.png"
            plot_prompt_layer_alpha_k_surface(
                prompt_layer_k_summary,
                output_path=prompt_surface_path,
                components=components,
            )
            saved.append(prompt_surface_path)
        else:
            prompt_layer_k_path = output_dir / "closed_loop_prompt_layer_k_map.png"
            plot_prompt_layer_k_map(prompt_layer_k_summary, output_path=prompt_layer_k_path, components=components)
            saved.append(prompt_layer_k_path)
    if comparison_summary_roots:
        transition_summary = combined_component_summaries([*comparison_summary_roots, summary_root])
        transition_path = output_dir / "closed_loop_alpha_transition.png"
        plot_alpha_transition_band(transition_summary, output_path=transition_path, components=components)
        saved.append(transition_path)
    manifest = {
        "summary_root": str(summary_root),
        "comparison_summary_roots": [str(path) for path in comparison_summary_roots],
        "components": list(components),
        "plots": [str(path) for path in saved],
    }
    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return saved + [manifest_path]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", required=True)
    parser.add_argument("--comparison-summary-roots", nargs="*", default=[])
    parser.add_argument("--output-dir")
    parser.add_argument("--components", nargs="+", default=CORE_COMPONENTS)
    args = parser.parse_args(argv)

    summary_root = Path(args.summary_root)
    comparison_summary_roots = [Path(path) for path in args.comparison_summary_roots]
    output_dir = Path(args.output_dir) if args.output_dir else summary_root / "plots"
    saved = build_plots(
        summary_root=summary_root,
        output_dir=output_dir,
        components=args.components,
        comparison_summary_roots=comparison_summary_roots,
    )
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
