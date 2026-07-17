#!/usr/bin/env python3
"""Plot HLTD branch-Hodge ledger summaries."""
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
    "negative_coexact",
]

COMPONENT_LABELS = {
    "coexact": "coexact",
    "coexact_minus_presence": "coexact - presence",
    "presence": "presence",
    "presence_plus_coexact": "presence + coexact",
    "negative_coexact": "-coexact",
}

COMPACT_COMPONENT_LABELS = {
    "coexact": "coexact",
    "coexact_minus_presence": "coex-pres",
    "presence": "presence",
    "presence_plus_coexact": "pres+coex",
    "negative_coexact": "-coexact",
}

COMPONENT_COLORS = {
    "coexact": "#2878b5",
    "coexact_minus_presence": "#d65f5f",
    "presence": "#4c9a2a",
    "presence_plus_coexact": "#8064a2",
    "negative_coexact": "#6b6b6b",
}

BRANCH_COLORS = {
    "exact": "#4c9a2a",
    "coexact": "#2878b5",
    "harmonic": "#8c6bb1",
    "coexact_shuffle_gap": "#d17a22",
    "coexact_random_gap": "#9b5a2e",
    "hodge_curl": "#5f6caf",
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


def load_tables(summary_root: Path) -> Dict[str, pd.DataFrame]:
    required = {
        "hodge_layer": "hodge_layer.csv",
        "hodge_k_sweep": "hodge_k_sweep.csv",
        "hodge_topology_family_k": "hodge_topology_family_k.csv",
        "causal_k_scoreboard": "causal_k_scoreboard.csv",
    }
    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in required.items():
        path = summary_root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        tables[key] = pd.read_csv(path)
    closed_loop_path = summary_root / "closed_loop_branch_scoreboard.csv"
    if closed_loop_path.exists() and closed_loop_path.stat().st_size > 0:
        tables["closed_loop_branch_scoreboard"] = pd.read_csv(closed_loop_path)
    role_path = summary_root / "branch_role_summary.csv"
    if role_path.exists() and role_path.stat().st_size > 0:
        tables["branch_role_summary"] = pd.read_csv(role_path)
    family_branch_path = summary_root / "family_branch_join.csv"
    if family_branch_path.exists() and family_branch_path.stat().st_size > 0:
        tables["family_branch_join"] = pd.read_csv(family_branch_path)
    diagnostics_path = summary_root / "branch_role_diagnostics.csv"
    if diagnostics_path.exists() and diagnostics_path.stat().st_size > 0:
        tables["branch_role_diagnostics"] = pd.read_csv(diagnostics_path)
    layer_condition_path = summary_root / "branch_layer_condition_summary.csv"
    if layer_condition_path.exists() and layer_condition_path.stat().st_size > 0:
        tables["branch_layer_condition_summary"] = pd.read_csv(layer_condition_path)
    layer_transition_path = summary_root / "branch_layer_transition_summary.csv"
    if layer_transition_path.exists() and layer_transition_path.stat().st_size > 0:
        tables["branch_layer_transition_summary"] = pd.read_csv(layer_transition_path)
    condition_path = summary_root / "branch_condition_summary.csv"
    if condition_path.exists() and condition_path.stat().st_size > 0:
        tables["branch_condition_summary"] = pd.read_csv(condition_path)
    candidate_path = summary_root / "branch_band_candidate_scoreboard.csv"
    if candidate_path.exists() and candidate_path.stat().st_size > 0:
        tables["branch_band_candidate_scoreboard"] = pd.read_csv(candidate_path)
    prompt_path = summary_root / "closed_loop_prompt_join.csv"
    if prompt_path.exists() and prompt_path.stat().st_size > 0:
        tables["closed_loop_prompt_join"] = pd.read_csv(prompt_path)
    reverse_path = summary_root / "reverse_exception_specificity.csv"
    if reverse_path.exists() and reverse_path.stat().st_size > 0:
        tables["reverse_exception_specificity"] = pd.read_csv(reverse_path)
    return tables


def plot_layer_spine(hodge_layer: pd.DataFrame, *, output_path: Path) -> None:
    rows = hodge_layer.sort_values("layer")
    layers = rows["layer"].astype(int).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)

    axes[0].plot(
        layers,
        rows["real_exact_mean"].astype(float),
        marker="o",
        color=BRANCH_COLORS["exact"],
        label="exact / presence",
    )
    axes[0].plot(
        layers,
        rows["real_coexact_mean"].astype(float),
        marker="o",
        color=BRANCH_COLORS["coexact"],
        label="coexact",
    )
    axes[0].plot(
        layers,
        rows["real_harmonic_mean"].astype(float),
        marker="o",
        color=BRANCH_COLORS["harmonic"],
        label="harmonic",
    )
    axes[0].set_title("structural Hodge branch ratios")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("energy ratio")
    axes[0].set_ylim(bottom=-0.02, top=1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        layers,
        rows["real_minus_shuffle_coexact_mean"].astype(float),
        marker="o",
        color=BRANCH_COLORS["coexact_shuffle_gap"],
        label="coexact - shuffle",
    )
    axes[1].plot(
        layers,
        rows["real_minus_random_coexact_mean"].astype(float),
        marker="o",
        color=BRANCH_COLORS["coexact_random_gap"],
        label="coexact - random",
    )
    axes[1].plot(
        layers,
        rows["real_hodge_curl_mean"].astype(float),
        marker="o",
        color=BRANCH_COLORS["hodge_curl"],
        label="Delaunay curl",
    )
    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].set_title("coexact branch separation")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("ratio / delta")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("HLTD branch-Hodge layer spine", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_k_sweep(hodge_k_sweep: pd.DataFrame, *, output_path: Path) -> None:
    rows = hodge_k_sweep.sort_values("k")
    ks = rows["k"].astype(int).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)

    series = [
        ("exact / presence", "real_exact_l5_l8_mean", BRANCH_COLORS["exact"]),
        ("coexact", "real_coexact_l5_l8_mean", BRANCH_COLORS["coexact"]),
        ("harmonic max", "real_harmonic_max_mean", BRANCH_COLORS["harmonic"]),
    ]
    for label, column, color in series:
        axes[0].plot(ks, rows[column].astype(float), marker="o", color=color, label=label)
    axes[0].set_title("structural branch ratios across k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("L5-L8 mean ratio")
    axes[0].set_xticks(ks)
    axes[0].set_ylim(bottom=-0.02, top=1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    gap_series = [
        ("coexact - shuffle", "real_minus_shuffle_coexact_l5_l8_mean", BRANCH_COLORS["coexact_shuffle_gap"]),
        ("coexact - random", "real_minus_random_coexact_l5_l8_mean", BRANCH_COLORS["coexact_random_gap"]),
        ("same-graph reverse gap", "max_same_graph_reverse_coexact_gap_mean", "#555555"),
    ]
    for label, column, color in gap_series:
        axes[1].plot(ks, rows[column].astype(float), marker="o", color=color, label=label)
    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].set_title("robustness gaps across k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("delta")
    axes[1].set_xticks(ks)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("HLTD branch-Hodge k sweep", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_topology_contrast(topology_rows: pd.DataFrame, *, output_path: Path) -> None:
    rows = topology_rows.copy()
    families = sorted(str(v) for v in rows["family"].dropna().unique())
    topologies = sorted(str(v) for v in rows["topology"].dropna().unique())
    x = np.arange(len(families), dtype=float)
    branch_columns = [
        ("exact / presence", "real_exact_l5_l8", BRANCH_COLORS["exact"]),
        ("coexact", "real_coexact_l5_l8", BRANCH_COLORS["coexact"]),
        ("harmonic max", "real_harmonic_max", BRANCH_COLORS["harmonic"]),
    ]
    width = 0.78 / len(branch_columns)

    fig, axes = plt.subplots(
        1,
        max(len(topologies), 1),
        figsize=(5.8 * max(len(topologies), 1), 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, topology in zip(axes.ravel(), topologies):
        topo_rows = rows[rows["topology"] == topology]
        for branch_idx, (label, column, color) in enumerate(branch_columns):
            values = []
            for family in families:
                row = topo_rows[topo_rows["family"] == family]
                values.append(float(row[column].iloc[0]) if not row.empty else np.nan)
            offset = (branch_idx - (len(branch_columns) - 1) / 2.0) * width
            ax.bar(x + offset, values, width=width, color=color, label=label)
        ax.set_xticks(x, families, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(topology.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[0].set_ylabel("ratio / max ratio")
    axes.ravel()[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.suptitle("HLTD topology contrast by prompt family", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def causal_subset(
    causal: pd.DataFrame,
    *,
    probe: str,
    selector: str,
    components: Sequence[str],
) -> pd.DataFrame:
    return causal[
        (causal["probe"] == probe)
        & (causal["selector"] == selector)
        & (causal["component"].isin(components))
    ].copy()


def plot_causal_split(
    causal: pd.DataFrame,
    *,
    output_path: Path,
    probe: str,
    selector: str,
    components: Sequence[str],
) -> None:
    subset = causal_subset(causal, probe=probe, selector=selector, components=components)
    ks = sorted(int(v) for v in subset["k"].dropna().unique())
    metrics = [
        ("next-token logprob delta", "next_token_delta_mean"),
        ("probe label-margin delta", "probe_label_margin_delta_mean"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4), constrained_layout=True)
    width = 0.8 / max(len(components), 1)
    x = np.arange(len(ks), dtype=float)

    for ax, (title, column) in zip(axes, metrics):
        for idx, component in enumerate(components):
            comp_rows = subset[subset["component"] == component]
            values = []
            for k in ks:
                row = comp_rows[comp_rows["k"].astype(int) == k]
                values.append(float(row[column].iloc[0]) if not row.empty else np.nan)
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
        ax.set_title(title)
        ax.set_ylabel("component - random")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle(f"{probe}: causal branch split ({selector})", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_branch_phase(
    causal: pd.DataFrame,
    *,
    output_path: Path,
    probe: str,
    selector: str,
    components: Sequence[str],
) -> None:
    subset = causal_subset(causal, probe=probe, selector=selector, components=components)
    markers = {12: "o", 16: "s", 24: "^"}
    fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    for row in subset.itertuples(index=False):
        component = str(row.component)
        k = int(row.k)
        x = float(row.next_token_delta_mean)
        y = float(row.probe_label_margin_delta_mean)
        ax.scatter(
            x,
            y,
            s=82,
            marker=markers.get(k, "o"),
            color=COMPONENT_COLORS.get(component),
            edgecolor="white",
            linewidth=0.7,
        )
        ax.text(x + 0.012, y, f"k{k} {COMPONENT_LABELS.get(component, component)}", fontsize=7, va="center")
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    xlim = symmetric_limit(subset["next_token_delta_mean"].astype(float))
    ylim = symmetric_limit(subset["probe_label_margin_delta_mean"].astype(float))
    ax.set_xlim(-xlim * 1.15, xlim * 1.35)
    ax.set_ylim(-ylim * 1.15, ylim * 1.15)
    ax.set_xlabel("next-token logprob delta vs random")
    ax.set_ylabel("probe label-margin delta vs random")
    ax.set_title(f"{probe}: traversal/stabilization phase ({selector})")
    ax.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compact_source_label(source: str) -> str:
    if "ontology5_prompt_robust" in source:
        return "five_prompt"
    if "seed_probe_ontology01_05" in source:
        return "ontology01_05"
    label = source
    for prefix in [
        "spiral_out_hltd_closed_loop_",
        "ontology5_prompt_robust_",
        "seed_probe_",
    ]:
        label = label.replace(prefix, "")
    label = label.replace("_l7_k16_a08", "").replace("_l7", "")
    if len(label) > 28:
        label = label[:25] + "..."
    return label


def plot_closed_loop_branch_scoreboard(
    closed_loop: pd.DataFrame,
    *,
    output_path: Path,
    family: str,
    components: Sequence[str],
) -> None:
    rows = closed_loop.copy()
    if "family" in rows:
        family_rows = rows[rows["family"] == family]
        if not family_rows.empty:
            rows = family_rows
    if components:
        rows = rows[rows["component"].isin(components)]
    if rows.empty:
        raise ValueError("No closed-loop branch scoreboard rows available for the requested plot.")

    component_order = {component: idx for idx, component in enumerate(components)}
    rows["_source_order"] = rows["source"].astype(str)
    rows["_component_order"] = rows["component"].map(lambda value: component_order.get(str(value), len(component_order)))
    rows = rows.sort_values(["_source_order", "_component_order", "alpha"]).reset_index(drop=True)
    labels = [
        f"{compact_source_label(str(row.source))}\n{COMPACT_COMPONENT_LABELS.get(str(row.component), str(row.component))}"
        for row in rows.itertuples(index=False)
    ]
    y = np.arange(len(rows), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, max(4.6, 0.48 * len(rows) + 1.8)), constrained_layout=True)
    bar_height = 0.34
    axes[0].barh(
        y - bar_height / 2,
        rows["branch_gate_rate"].astype(float),
        height=bar_height,
        color="#9BA7C0",
        label="raw gate",
    )
    axes[0].barh(
        y + bar_height / 2,
        rows["branch_specific_gate_rate"].astype(float),
        height=bar_height,
        color="#d65f5f",
        label="branch-specific gate",
    )
    axes[0].set_xlim(0.0, 1.02)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("gate rate")
    axes[0].set_title("closed-loop gate rates")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(frameon=False)

    target = rows["mean_target_margin_delta_minus_random_mean"].astype(float)
    limit = symmetric_limit(target)
    colors = [plt.cm.coolwarm(0.5 + 0.5 * np.clip(value / limit, -1.0, 1.0)) for value in target]
    axes[1].barh(y, target, color=colors)
    axes[1].axvline(0.0, color="#444444", linewidth=0.8)
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlim(-limit * 1.12, limit * 1.12)
    axes[1].set_xlabel("target margin - random")
    axes[1].set_title("semantic target advantage")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("HLTD closed-loop branch-specific gates on the Hodge ledger", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_branch_role_summary(
    roles: pd.DataFrame,
    *,
    output_path: Path,
    probe: str,
    selector: str,
    components: Sequence[str],
) -> None:
    rows = roles[
        (roles["probe"] == probe)
        & (roles["selector"] == selector)
        & (roles["component"].isin(components))
    ].copy()
    if rows.empty:
        raise ValueError("No branch role summary rows available for the requested plot.")
    component_order = {component: idx for idx, component in enumerate(components)}
    rows["_component_order"] = rows["component"].map(lambda value: component_order.get(str(value), len(component_order)))
    rows = rows.sort_values("_component_order")

    target = rows["closed_loop_target_margin_delta_minus_random_mean"].astype(float)
    color_limit = symmetric_limit(target)
    gate = rows["closed_loop_branch_specific_gate_rate_mean"].astype(float).fillna(0.0).clip(lower=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
    for row, gate_value, target_value in zip(rows.itertuples(index=False), gate, target):
        component = str(row.component)
        color = plt.cm.coolwarm(0.5 + 0.5 * np.clip(float(target_value) / color_limit, -1.0, 1.0))
        axes[0].scatter(
            float(row.next_token_delta_mean),
            float(row.probe_label_margin_delta_mean),
            s=95 + 360 * float(gate_value),
            color=color,
            edgecolor=COMPONENT_COLORS.get(component, "#222222"),
            linewidth=1.7,
        )
        axes[0].text(
            float(row.next_token_delta_mean) + 0.012,
            float(row.probe_label_margin_delta_mean),
            f"{COMPACT_COMPONENT_LABELS.get(component, component)}\n{str(row.role_label)}",
            fontsize=8,
            va="center",
        )
    axes[0].axhline(0.0, color="#444444", linewidth=0.8)
    axes[0].axvline(0.0, color="#444444", linewidth=0.8)
    xlim = symmetric_limit(rows["next_token_delta_mean"].astype(float))
    ylim = symmetric_limit(rows["probe_label_margin_delta_mean"].astype(float))
    axes[0].set_xlim(-xlim * 1.2, xlim * 1.45)
    axes[0].set_ylim(-ylim * 1.2, ylim * 1.2)
    axes[0].set_xlabel("mean next-token delta vs random")
    axes[0].set_ylabel("mean probe margin vs random")
    axes[0].set_title("mean causal role phase")
    axes[0].grid(alpha=0.25)

    y = np.arange(len(rows), dtype=float)
    axes[1].barh(
        y,
        rows["closed_loop_branch_specific_gate_rate_mean"].astype(float),
        color=[COMPONENT_COLORS.get(str(component), "#999999") for component in rows["component"]],
    )
    axes[1].set_yticks(
        y,
        [COMPACT_COMPONENT_LABELS.get(str(component), str(component)) for component in rows["component"]],
    )
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 1.02)
    axes[1].set_xlabel("closed-loop branch-specific gate")
    axes[1].set_title("autoregressive specificity")
    axes[1].grid(axis="x", alpha=0.25)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-color_limit, vmax=color_limit))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[0], label="closed-loop target margin - random")
    fig.suptitle(f"{probe}: branch role summary ({selector})", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def role_metric_matrix(
    roles: pd.DataFrame,
    *,
    probes: Sequence[str],
    components: Sequence[str],
    metric: str,
) -> np.ndarray:
    values = np.full((len(probes), len(components)), np.nan, dtype=float)
    for row_idx, probe in enumerate(probes):
        for col_idx, component in enumerate(components):
            sub = roles[(roles["probe"] == probe) & (roles["component"] == component)]
            if sub.empty or metric not in sub:
                continue
            values[row_idx, col_idx] = float(sub[metric].iloc[0])
    return values


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


def plot_branch_role_matrix(
    roles: pd.DataFrame,
    *,
    output_path: Path,
    selector: str,
    components: Sequence[str],
) -> None:
    rows = roles[roles["selector"] == selector].copy()
    if components:
        rows = rows[rows["component"].isin(components)]
    if rows.empty:
        raise ValueError("No branch role rows available for role matrix.")
    probes = sorted(str(value) for value in rows["probe"].dropna().unique())
    chosen_components = [component for component in components if component in set(rows["component"].astype(str))]
    if not chosen_components:
        chosen_components = sorted(str(value) for value in rows["component"].dropna().unique())

    metrics = [
        ("next-token delta", "next_token_delta_mean", "coolwarm", None, None),
        ("probe margin", "probe_label_margin_delta_mean", "coolwarm", None, None),
        ("closed-loop specific gate", "closed_loop_branch_specific_gate_rate_mean", "viridis", 0.0, 1.0),
    ]
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(max(8.6, 1.05 * len(chosen_components) + 4.0), 2.25 * len(metrics) + 2.0),
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]
    xlabels = [COMPACT_COMPONENT_LABELS.get(component, component) for component in chosen_components]
    ylabels = [probe.replace("_", "\n") for probe in probes]

    for ax, (title, metric, cmap_name, explicit_vmin, explicit_vmax) in zip(axes, metrics):
        values = role_metric_matrix(rows, probes=probes, components=chosen_components, metric=metric)
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
        annotate_matrix(ax, values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(chosen_components)), xlabels)
        ax.set_yticks(np.arange(len(probes)), ylabels)
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)

    fig.suptitle(f"HLTD branch role matrix ({selector})", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_family_branch_atlas(
    family_branch: pd.DataFrame,
    *,
    output_path: Path,
    selector: str,
    components: Sequence[str],
) -> None:
    rows = family_branch[family_branch["selector"] == selector].copy()
    if components:
        rows = rows[rows["component"].isin(components)]
    if rows.empty:
        raise ValueError("No family branch rows available for branch atlas.")

    grouped = (
        rows.groupby(["family", "probe", "component"], as_index=False)
        .agg(
            next_token_delta=("next_token_delta", "mean"),
            semantic_margin_delta=("semantic_margin_delta", "mean"),
            probe_label_margin_delta=("probe_label_margin_delta", "mean"),
            hodge_coexact=("hodge_coexact", "mean"),
            hodge_real_minus_shuffle_coexact=("hodge_real_minus_shuffle_coexact", "mean"),
        )
        .sort_values(["family", "probe", "component"])
    )
    row_index = (
        grouped[["family", "probe"]]
        .drop_duplicates()
        .sort_values(["family", "probe"])
        .reset_index(drop=True)
    )
    chosen_components = [component for component in components if component in set(grouped["component"].astype(str))]
    if not chosen_components:
        chosen_components = sorted(str(value) for value in grouped["component"].dropna().unique())

    lookup = {
        (str(row.family), str(row.probe), str(row.component)): row
        for row in grouped.itertuples(index=False)
    }
    metrics = [
        ("next-token delta", "next_token_delta"),
        ("probe margin", "probe_label_margin_delta"),
        ("semantic margin", "semantic_margin_delta"),
    ]
    matrices: List[np.ndarray] = []
    for _, metric in metrics:
        values = np.full((len(row_index), len(chosen_components)), np.nan, dtype=float)
        for row_idx, row in enumerate(row_index.itertuples(index=False)):
            for col_idx, component in enumerate(chosen_components):
                found = lookup.get((str(row.family), str(row.probe), component))
                if found is not None:
                    values[row_idx, col_idx] = float(getattr(found, metric))
        matrices.append(values)

    fig_height = max(7.0, 0.42 * len(row_index) + 2.3)
    fig_width = max(13.2, 1.2 * len(chosen_components) + 7.2)
    fig, axes = plt.subplots(1, len(metrics), figsize=(fig_width, fig_height), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    xlabels = [COMPACT_COMPONENT_LABELS.get(component, component) for component in chosen_components]
    ylabels = [
        f"{str(row.family).replace('_', ' ')}\n{str(row.probe).replace('_', ' ')}"
        for row in row_index.itertuples(index=False)
    ]
    cmap = plt.colormaps["coolwarm"].copy()
    cmap.set_bad("#F0F1F4")

    for ax, (title, _), values in zip(axes, metrics, matrices):
        limit = symmetric_limit(values.ravel())
        image = ax.imshow(
            np.ma.masked_invalid(values),
            aspect="auto",
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
        )
        annotate_matrix(ax, values, cmap=cmap, vmin=-limit, vmax=limit)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(chosen_components)), xlabels, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(row_index)), ylabels if ax is axes[0] else [])
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)

    families = row_index["family"].astype(str).to_list()
    for row_idx in range(1, len(families)):
        if families[row_idx] != families[row_idx - 1]:
            for ax in axes:
                ax.axhline(row_idx - 0.5, color="white", linewidth=1.6)

    fig.suptitle(f"HLTD family/probe branch atlas ({selector})", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_branch_role_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    output_path: Path,
    selector: str,
    components: Sequence[str],
) -> None:
    rows = diagnostics[diagnostics["selector"] == selector].copy()
    if components:
        rows = rows[rows["component"].isin(components)]
    if rows.empty:
        raise ValueError("No branch role diagnostics rows available.")

    row_index = (
        rows[["family", "probe"]]
        .drop_duplicates()
        .sort_values(["family", "probe"])
        .reset_index(drop=True)
    )
    chosen_components = [component for component in components if component in set(rows["component"].astype(str))]
    if not chosen_components:
        chosen_components = sorted(str(value) for value in rows["component"].dropna().unique())

    lookup = {
        (str(row.family), str(row.probe), str(row.component)): row
        for row in rows.itertuples(index=False)
    }
    scores = np.full((len(row_index), len(chosen_components)), np.nan, dtype=float)
    for row_idx, row in enumerate(row_index.itertuples(index=False)):
        for col_idx, component in enumerate(chosen_components):
            found = lookup.get((str(row.family), str(row.probe), component))
            if found is not None:
                scores[row_idx, col_idx] = float(found.role_score)

    fig_height = max(6.4, 0.42 * len(row_index) + 2.4)
    fig_width = max(9.6, 1.1 * len(chosen_components) + 5.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#F0F1F4")
    image = ax.imshow(np.ma.masked_invalid(scores), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    for row_idx, row in enumerate(row_index.itertuples(index=False)):
        for col_idx, component in enumerate(chosen_components):
            found = lookup.get((str(row.family), str(row.probe), component))
            value = scores[row_idx, col_idx]
            if found is None or not np.isfinite(value):
                continue
            label = f"{value:.2f}"
            failed = str(getattr(found, "criteria_failed", "") or "")
            if failed and failed != "nan":
                label += "\n!" + failed.replace("_", " ")
            color = "white" if value < 0.55 else "#1F2430"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=7, color=color)

    xlabels = [COMPACT_COMPONENT_LABELS.get(component, component) for component in chosen_components]
    ylabels = [
        f"{str(row.family).replace('_', ' ')}\n{str(row.probe).replace('_', ' ')}"
        for row in row_index.itertuples(index=False)
    ]
    ax.set_xticks(np.arange(len(chosen_components)), xlabels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_index)), ylabels)
    families = row_index["family"].astype(str).to_list()
    for row_idx in range(1, len(families)):
        if families[row_idx] != families[row_idx - 1]:
            ax.axhline(row_idx - 0.5, color="white", linewidth=1.6)
    ax.set_title("expected branch-role score")
    ax.set_xlabel("component")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(f"HLTD family-local branch role diagnostics ({selector})", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compact_condition_label(value: object) -> str:
    label = str(value)
    replacements = {
        "stable_expected": "stable",
        "mostly_expected": "mostly",
        "mixed_condition": "mixed",
        "systematic_partial_break": "partial",
        "systematic_break": "break",
    }
    return replacements.get(label, label.replace("_", " "))


def display_optional_text(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and np.isnan(value):
        return "-"
    text = str(value)
    if text == "" or text.lower() == "nan":
        return "-"
    return text


def display_number_text(value: object, *, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def plot_branch_condition_summary(
    conditions: pd.DataFrame,
    *,
    output_path: Path,
    selector: str,
    components: Sequence[str],
) -> None:
    rows = conditions[conditions["selector"] == selector].copy()
    if components:
        rows = rows[rows["component"].isin(components)]
    if rows.empty:
        raise ValueError("No branch condition summary rows available.")

    families = sorted(str(value) for value in rows["family"].dropna().unique())
    chosen_components = [component for component in components if component in set(rows["component"].astype(str))]
    if not chosen_components:
        chosen_components = sorted(str(value) for value in rows["component"].dropna().unique())
    lookup = {
        (str(row.family), str(row.component)): row
        for row in rows.itertuples(index=False)
    }
    metrics = [
        ("probe-cell pass rate", "role_pass_rate"),
        ("mean role score", "mean_role_score"),
    ]
    matrices: List[np.ndarray] = []
    for _, metric in metrics:
        values = np.full((len(families), len(chosen_components)), np.nan, dtype=float)
        for row_idx, family in enumerate(families):
            for col_idx, component in enumerate(chosen_components):
                found = lookup.get((family, component))
                if found is not None:
                    values[row_idx, col_idx] = float(getattr(found, metric))
        matrices.append(values)

    fig_width = max(11.6, 1.1 * len(chosen_components) + 6.2)
    fig_height = max(4.8, 0.55 * len(families) + 2.4)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), constrained_layout=True)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#F0F1F4")
    xlabels = [COMPACT_COMPONENT_LABELS.get(component, component) for component in chosen_components]
    ylabels = [family.replace("_", " ") for family in families]

    for ax, (title, _), values in zip(axes, metrics, matrices):
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        for row_idx, family in enumerate(families):
            for col_idx, component in enumerate(chosen_components):
                found = lookup.get((family, component))
                value = values[row_idx, col_idx]
                if found is None or not np.isfinite(value):
                    continue
                label = f"{value:.2f}"
                if title == "probe-cell pass rate":
                    label = (
                        f"{int(found.role_pass_count)}/{int(found.n_probe_cells)}\n"
                        f"{compact_condition_label(found.condition_label)}"
                    )
                color = "white" if value < 0.55 else "#1F2430"
                ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8, color=color)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(chosen_components)), xlabels, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(families)), ylabels if ax is axes[0] else [])
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)

    fig.suptitle(f"HLTD family-component branch conditions ({selector})", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_branch_band_candidate_scoreboard(
    candidates: pd.DataFrame,
    *,
    output_path: Path,
    selector: str,
    components: Sequence[str],
    top_n: int = 20,
) -> None:
    rows = candidates[candidates["selector"] == selector].copy()
    if components:
        rows = rows[rows["component"].isin(components)]
    if rows.empty:
        raise ValueError("No branch-band candidate rows available.")

    rows["priority_score"] = rows["priority_score"].astype(float)
    rows = rows.sort_values(["priority_score", "family", "component"], ascending=[False, True, True]).head(top_n)
    rows = rows.reset_index(drop=True)
    y = np.arange(len(rows), dtype=float)
    labels = [
        f"{str(row.family).replace('_', ' ')}\n{COMPACT_COMPONENT_LABELS.get(str(row.component), str(row.component))}"
        for row in rows.itertuples(index=False)
    ]

    fig_height = max(6.2, 0.42 * len(rows) + 2.2)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, fig_height), constrained_layout=True)
    axes[0].barh(
        y,
        rows["priority_score"].astype(float),
        color=[COMPONENT_COLORS.get(str(component), "#999999") for component in rows["component"]],
        alpha=0.88,
        label="priority",
    )
    if "causal_support" in rows:
        axes[0].scatter(
            rows["causal_support"].astype(float),
            y,
            marker="|",
            s=180,
            color="#1F2430",
            linewidth=1.8,
            label="causal support",
        )
    axes[0].set_xlim(0.0, 1.02)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("candidate priority")
    axes[0].set_title("next branch-band queue")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_xlim(0.0, 1.0)
    axes[1].axis("off")
    headers = [
        (0.00, "candidate"),
        (0.33, "layers"),
        (0.55, "gate"),
        (0.70, "target-rand"),
        (0.88, "n"),
    ]
    for x, header in headers:
        axes[1].text(x, -0.85, header, fontsize=9, weight="bold")
    for idx, row in enumerate(rows.itertuples(index=False)):
        axes[1].text(0.00, idx, str(row.candidate_label).replace("_", " "), va="center", fontsize=8)
        axes[1].text(0.33, idx, display_optional_text(row.recommended_layers), va="center", fontsize=8)
        axes[1].text(
            0.55,
            idx,
            display_number_text(row.closed_loop_branch_specific_gate_rate_mean),
            va="center",
            fontsize=8,
        )
        axes[1].text(
            0.70,
            idx,
            display_number_text(row.closed_loop_target_margin_delta_minus_random_mean),
            va="center",
            fontsize=8,
        )
        axes[1].text(0.88, idx, display_optional_text(row.closed_loop_matched_random_rows), va="center", fontsize=8)

    families = rows["family"].astype(str).to_list()
    for row_idx in range(1, len(families)):
        if families[row_idx] != families[row_idx - 1]:
            for ax in axes:
                ax.axhline(row_idx - 0.5, color="#D6D9DF", linewidth=1.0)

    fig.suptitle(f"HLTD branch-band candidate scoreboard ({selector})", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_branch_layer_condition_summary(
    layer_conditions: pd.DataFrame,
    *,
    output_path: Path,
    selector: str,
    components: Sequence[str],
) -> None:
    rows = layer_conditions[layer_conditions["selector"] == selector].copy()
    if components:
        rows = rows[rows["component"].isin(components)]
    if rows.empty:
        raise ValueError("No branch layer condition summary rows available.")

    component_order = {component: idx for idx, component in enumerate(components)}
    row_index = (
        rows[["family", "component"]]
        .drop_duplicates()
        .assign(_component_order=lambda frame: frame["component"].map(lambda value: component_order.get(str(value), len(component_order))))
        .sort_values(["family", "_component_order", "component"])
        .drop(columns=["_component_order"])
        .reset_index(drop=True)
    )
    layers = sorted(int(value) for value in rows["layer"].dropna().unique())
    lookup = {
        (str(row.family), str(row.component), int(row.layer)): row
        for row in rows.itertuples(index=False)
    }
    values = np.full((len(row_index), len(layers)), np.nan, dtype=float)
    for row_idx, row in enumerate(row_index.itertuples(index=False)):
        for col_idx, layer in enumerate(layers):
            found = lookup.get((str(row.family), str(row.component), layer))
            if found is not None:
                values[row_idx, col_idx] = float(found.role_pass_rate)

    fig_width = max(7.8, 0.85 * len(layers) + 5.4)
    fig_height = max(7.8, 0.34 * len(row_index) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#F0F1F4")
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    for row_idx, row in enumerate(row_index.itertuples(index=False)):
        for col_idx, layer in enumerate(layers):
            found = lookup.get((str(row.family), str(row.component), layer))
            value = values[row_idx, col_idx]
            if found is None or not np.isfinite(value):
                continue
            label = f"{int(found.role_pass_count)}/{int(found.n_probe_cells)}"
            if value < 1.0:
                label += f"\n{compact_condition_label(found.condition_label)}"
            color = "white" if value < 0.55 else "#1F2430"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=7, color=color)

    ylabels = [
        f"{str(row.family).replace('_', ' ')}\n{COMPACT_COMPONENT_LABELS.get(str(row.component), str(row.component))}"
        for row in row_index.itertuples(index=False)
    ]
    ax.set_xticks(np.arange(len(layers)), [f"L{layer}" for layer in layers])
    ax.set_yticks(np.arange(len(row_index)), ylabels)
    families = row_index["family"].astype(str).to_list()
    for row_idx in range(1, len(families)):
        if families[row_idx] != families[row_idx - 1]:
            ax.axhline(row_idx - 0.5, color="white", linewidth=1.6)
    ax.set_xlabel("layer")
    ax.set_title("probe-cell pass rate by layer")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(f"HLTD branch condition layer spine ({selector})", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_branch_layer_transition_summary(
    transitions: pd.DataFrame,
    *,
    output_path: Path,
    selector: str,
    components: Sequence[str],
) -> None:
    rows = transitions[transitions["selector"] == selector].copy()
    if components:
        rows = rows[rows["component"].isin(components)]
    if rows.empty:
        raise ValueError("No branch layer transition summary rows available.")

    component_order = {component: idx for idx, component in enumerate(components)}
    rows["_component_order"] = rows["component"].map(lambda value: component_order.get(str(value), len(component_order)))
    rows = rows.sort_values(["family", "_component_order", "component"]).reset_index(drop=True)
    y = np.arange(len(rows), dtype=float)
    labels = [
        f"{str(row.family).replace('_', ' ')}\n{COMPACT_COMPONENT_LABELS.get(str(row.component), str(row.component))}"
        for row in rows.itertuples(index=False)
    ]

    fig_height = max(7.2, 0.42 * len(rows) + 2.4)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, fig_height), constrained_layout=True)
    rates = rows["stable_layer_rate"].astype(float)
    axes[0].barh(
        y,
        rates,
        color=[COMPONENT_COLORS.get(str(component), "#999999") for component in rows["component"]],
    )
    axes[0].set_xlim(0.0, 1.02)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("stable layer rate")
    axes[0].set_title("stable depth support")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_xlim(0.0, 1.0)
    axes[1].axis("off")
    axes[1].text(0.00, -0.85, "transition", fontsize=9, weight="bold")
    axes[1].text(0.32, -0.85, "stable", fontsize=9, weight="bold")
    axes[1].text(0.57, -0.85, "mixed", fontsize=9, weight="bold")
    axes[1].text(0.79, -0.85, "break", fontsize=9, weight="bold")
    for idx, row in enumerate(rows.itertuples(index=False)):
        axes[1].text(0.00, idx, str(row.transition_label).replace("_", " "), va="center", fontsize=8)
        axes[1].text(0.32, idx, display_optional_text(row.stable_layers), va="center", fontsize=8)
        axes[1].text(0.57, idx, display_optional_text(row.mostly_or_mixed_layers), va="center", fontsize=8)
        axes[1].text(0.79, idx, display_optional_text(row.break_layers), va="center", fontsize=8)

    families = rows["family"].astype(str).to_list()
    for row_idx in range(1, len(families)):
        if families[row_idx] != families[row_idx - 1]:
            for ax in axes:
                ax.axhline(row_idx - 0.5, color="#D6D9DF", linewidth=1.0)

    fig.suptitle(f"HLTD branch layer transitions ({selector})", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_closed_loop_prompt_heatmap(
    prompt_rows: pd.DataFrame,
    *,
    output_path: Path,
    components: Sequence[str],
) -> None:
    rows = prompt_rows[prompt_rows["component"].isin(components)].copy()
    if rows.empty:
        raise ValueError("No closed-loop prompt rows available for prompt heatmap.")

    grouped = (
        rows.groupby(["family", "prompt_id", "component"], as_index=False)
        .agg(
            branch_specific_gate_rate=("branch_specific_gate_rate", "max"),
            mean_target_margin_delta_minus_random_mean=(
                "mean_target_margin_delta_minus_random_mean",
                "mean",
            ),
            n_sources=("source", "nunique"),
        )
        .sort_values(["family", "prompt_id", "component"])
    )
    prompt_index = (
        grouped[["family", "prompt_id"]]
        .drop_duplicates()
        .sort_values(["family", "prompt_id"])
        .reset_index(drop=True)
    )
    chosen_components = [component for component in components if component in set(grouped["component"].astype(str))]
    if not chosen_components:
        chosen_components = sorted(str(value) for value in grouped["component"].dropna().unique())

    gate_values = np.full((len(prompt_index), len(chosen_components)), np.nan, dtype=float)
    target_values = np.full_like(gate_values, np.nan)
    lookup = {
        (str(row.family), str(row.prompt_id), str(row.component)): row
        for row in grouped.itertuples(index=False)
    }
    for row_idx, prompt_row in enumerate(prompt_index.itertuples(index=False)):
        family = str(prompt_row.family)
        prompt_id = str(prompt_row.prompt_id)
        for col_idx, component in enumerate(chosen_components):
            row = lookup.get((family, prompt_id, component))
            if row is None:
                continue
            gate_values[row_idx, col_idx] = float(row.branch_specific_gate_rate)
            target_values[row_idx, col_idx] = float(row.mean_target_margin_delta_minus_random_mean)

    fig_height = max(6.0, 0.42 * len(prompt_index) + 2.4)
    fig_width = max(10.2, 1.08 * len(chosen_components) + 5.6)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), constrained_layout=True)
    xlabels = [COMPACT_COMPONENT_LABELS.get(component, component) for component in chosen_components]
    ylabels = [f"{str(row.family).replace('_', ' ')}\n{row.prompt_id}" for row in prompt_index.itertuples(index=False)]

    cmap_gate = plt.colormaps["viridis"].copy()
    cmap_gate.set_bad("#F0F1F4")
    gate_image = axes[0].imshow(np.ma.masked_invalid(gate_values), aspect="auto", cmap=cmap_gate, vmin=0.0, vmax=1.0)
    annotate_matrix(axes[0], gate_values, cmap=cmap_gate, vmin=0.0, vmax=1.0)
    axes[0].set_title("branch-specific gate")
    axes[0].set_xticks(np.arange(len(chosen_components)), xlabels, rotation=20, ha="right")
    axes[0].set_yticks(np.arange(len(prompt_index)), ylabels)
    fig.colorbar(gate_image, ax=axes[0], fraction=0.035, pad=0.02)

    target_limit = symmetric_limit(target_values.ravel())
    cmap_target = plt.colormaps["coolwarm"].copy()
    cmap_target.set_bad("#F0F1F4")
    target_image = axes[1].imshow(
        np.ma.masked_invalid(target_values),
        aspect="auto",
        cmap=cmap_target,
        vmin=-target_limit,
        vmax=target_limit,
    )
    annotate_matrix(axes[1], target_values, cmap=cmap_target, vmin=-target_limit, vmax=target_limit)
    axes[1].set_title("target margin - random")
    axes[1].set_xticks(np.arange(len(chosen_components)), xlabels, rotation=20, ha="right")
    axes[1].set_yticks(np.arange(len(prompt_index)), [])
    fig.colorbar(target_image, ax=axes[1], fraction=0.035, pad=0.02)

    families = prompt_index["family"].astype(str).to_list()
    for row_idx in range(1, len(families)):
        if families[row_idx] != families[row_idx - 1]:
            for ax in axes:
                ax.axhline(row_idx - 0.5, color="white", linewidth=1.6)

    for ax in axes:
        ax.set_xlabel("component")
        ax.grid(False)
    fig.suptitle("HLTD closed-loop prompt branch map", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _reverse_short_label(value: object) -> str:
    text = str(value)
    for prefix in ("identity_", "ontology_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.replace("_", "\n")


def plot_reverse_exception_specificity(
    reverse_rows: pd.DataFrame,
    *,
    output_path: Path,
    component: str,
) -> None:
    rows = reverse_rows[reverse_rows["component"].astype(str) == str(component)].copy()
    if rows.empty:
        raise ValueError(f"No reverse specificity rows available for {component}.")
    if "panel_order" not in rows:
        rows["panel_order"] = 0
    if "source_order" not in rows:
        rows["source_order"] = range(len(rows))
    panels = rows[["panel_order", "panel_label"]].drop_duplicates().sort_values("panel_order")

    fig, axes = plt.subplots(
        len(panels),
        2,
        figsize=(12.4, max(4.0, 3.1 * len(panels))),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(f"HLTD reverse exception specificity ({COMPONENT_LABELS.get(component, component)})", fontsize=13)

    for row_idx, panel in enumerate(panels.itertuples(index=False)):
        panel_rows = rows[rows["panel_label"] == panel.panel_label].copy()
        panel_rows = panel_rows.sort_values(["source_order", "target_set"])
        xs = np.arange(len(panel_rows), dtype=float)
        labels = [_reverse_short_label(value) for value in panel_rows["target_set"]]

        gate_ax = axes[row_idx][0]
        width = 0.28
        gate_ax.bar(
            xs - width / 2,
            panel_rows["branch_specific_gate_rate"].astype(float),
            width=width,
            color="#516b8b",
            label="branch-specific gate",
        )
        gate_ax.bar(
            xs + width / 2,
            panel_rows["random_branch_gate_rate"].astype(float),
            width=width,
            color="#b8b8b8",
            label="matched random gate",
        )
        gate_ax.set_title(f"{panel.panel_label}: gate specificity")
        gate_ax.set_ylim(0.0, 1.08)
        gate_ax.set_xticks(xs, labels)
        gate_ax.grid(axis="y", alpha=0.25)
        gate_ax.legend(frameon=False, fontsize=8)

        direction_ax = axes[row_idx][1]
        target_adv = panel_rows["mean_target_margin_delta_minus_random_mean"].astype(float)
        colors = ["#7aa36f" if float(value) >= 0.0 else "#b75b5b" for value in target_adv]
        direction_ax.bar(xs, target_adv, color=colors, label="target-random margin")
        direction_ax.plot(
            xs,
            panel_rows["token_drift_rate_mean"].astype(float),
            marker="o",
            color="#786fa6",
            linewidth=1.6,
            label="token drift",
        )
        direction_ax.axhline(0.0, color="#333333", linewidth=0.8)
        direction_ax.set_title(f"{panel.panel_label}: semantic direction vs drift")
        direction_ax.set_xticks(xs, labels)
        ymin = min(-0.5, float(target_adv.min()) - 0.1)
        ymax = max(1.1, float(target_adv.max()) + 0.15)
        direction_ax.set_ylim(ymin, ymax)
        direction_ax.grid(axis="y", alpha=0.25)
        direction_ax.legend(frameon=False, fontsize=8)

        for ax in (gate_ax, direction_ax):
            ax.tick_params(axis="x", labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_plots(
    *,
    summary_root: Path,
    output_dir: Path,
    probe: str,
    selector: str,
    components: Sequence[str],
) -> List[Path]:
    tables = load_tables(summary_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = [
        output_dir / "hodge_layer_spine.png",
        output_dir / "hodge_k_sweep.png",
        output_dir / "hodge_topology_contrast.png",
        output_dir / f"{probe}_{selector}_causal_split.png",
        output_dir / f"{probe}_{selector}_branch_phase.png",
    ]
    plot_layer_spine(tables["hodge_layer"], output_path=saved[0])
    plot_k_sweep(tables["hodge_k_sweep"], output_path=saved[1])
    plot_topology_contrast(tables["hodge_topology_family_k"], output_path=saved[2])
    plot_causal_split(
        tables["causal_k_scoreboard"],
        output_path=saved[3],
        probe=probe,
        selector=selector,
        components=components,
    )
    plot_branch_phase(
        tables["causal_k_scoreboard"],
        output_path=saved[4],
        probe=probe,
        selector=selector,
        components=components,
    )
    if "closed_loop_branch_scoreboard" in tables and not tables["closed_loop_branch_scoreboard"].empty:
        closed_loop_path = output_dir / "closed_loop_branch_specific_scoreboard.png"
        plot_closed_loop_branch_scoreboard(
            tables["closed_loop_branch_scoreboard"],
            output_path=closed_loop_path,
            family=probe,
            components=components,
        )
        saved.append(closed_loop_path)
    if "branch_role_summary" in tables and not tables["branch_role_summary"].empty:
        role_path = output_dir / "branch_role_summary.png"
        plot_branch_role_summary(
            tables["branch_role_summary"],
            output_path=role_path,
            probe=probe,
            selector=selector,
            components=components,
        )
        saved.append(role_path)
        matrix_path = output_dir / "branch_role_matrix.png"
        plot_branch_role_matrix(
            tables["branch_role_summary"],
            output_path=matrix_path,
            selector=selector,
            components=components,
        )
        saved.append(matrix_path)
    if "family_branch_join" in tables and not tables["family_branch_join"].empty:
        atlas_path = output_dir / "family_branch_atlas.png"
        plot_family_branch_atlas(
            tables["family_branch_join"],
            output_path=atlas_path,
            selector=selector,
            components=components,
        )
        saved.append(atlas_path)
    if "branch_role_diagnostics" in tables and not tables["branch_role_diagnostics"].empty:
        diagnostics_path = output_dir / "branch_role_diagnostics.png"
        plot_branch_role_diagnostics(
            tables["branch_role_diagnostics"],
            output_path=diagnostics_path,
            selector=selector,
            components=components,
        )
        saved.append(diagnostics_path)
    if "branch_layer_condition_summary" in tables and not tables["branch_layer_condition_summary"].empty:
        layer_condition_path = output_dir / "branch_layer_condition_summary.png"
        plot_branch_layer_condition_summary(
            tables["branch_layer_condition_summary"],
            output_path=layer_condition_path,
            selector=selector,
            components=components,
        )
        saved.append(layer_condition_path)
    if "branch_layer_transition_summary" in tables and not tables["branch_layer_transition_summary"].empty:
        transition_path = output_dir / "branch_layer_transition_summary.png"
        plot_branch_layer_transition_summary(
            tables["branch_layer_transition_summary"],
            output_path=transition_path,
            selector=selector,
            components=components,
        )
        saved.append(transition_path)
    if "branch_condition_summary" in tables and not tables["branch_condition_summary"].empty:
        condition_path = output_dir / "branch_condition_summary.png"
        plot_branch_condition_summary(
            tables["branch_condition_summary"],
            output_path=condition_path,
            selector=selector,
            components=components,
        )
        saved.append(condition_path)
    if "branch_band_candidate_scoreboard" in tables and not tables["branch_band_candidate_scoreboard"].empty:
        candidate_path = output_dir / "branch_band_candidate_scoreboard.png"
        plot_branch_band_candidate_scoreboard(
            tables["branch_band_candidate_scoreboard"],
            output_path=candidate_path,
            selector=selector,
            components=components,
        )
        saved.append(candidate_path)
    if "closed_loop_prompt_join" in tables and not tables["closed_loop_prompt_join"].empty:
        prompt_heatmap_path = output_dir / "closed_loop_prompt_branch_heatmap.png"
        plot_closed_loop_prompt_heatmap(
            tables["closed_loop_prompt_join"],
            output_path=prompt_heatmap_path,
            components=components,
        )
        saved.append(prompt_heatmap_path)
    if "reverse_exception_specificity" in tables and not tables["reverse_exception_specificity"].empty:
        reverse_path = output_dir / "reverse_exception_specificity.png"
        plot_reverse_exception_specificity(
            tables["reverse_exception_specificity"],
            output_path=reverse_path,
            component="negative_coexact",
        )
        saved.append(reverse_path)
    manifest = {
        "summary_root": str(summary_root),
        "probe": probe,
        "selector": selector,
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
    parser.add_argument("--selector", default="middle")
    parser.add_argument("--components", nargs="+", default=CORE_COMPONENTS)
    args = parser.parse_args(argv)

    summary_root = Path(args.summary_root)
    output_dir = Path(args.output_dir) if args.output_dir else summary_root / "plots"
    saved = build_plots(
        summary_root=summary_root,
        output_dir=output_dir,
        probe=args.probe,
        selector=args.selector,
        components=args.components,
    )
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
