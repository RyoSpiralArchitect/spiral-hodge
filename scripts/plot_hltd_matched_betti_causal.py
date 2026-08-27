#!/usr/bin/env python3
"""Plot prompt-level causal contrasts for matched-Betti HLTD steering."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 1729
BOOTSTRAP_SAMPLES = 5000

BRANCHES = ("exact", "coexact", "harmonic")
BRANCH_LABELS = {
    "exact": "exact / presence",
    "coexact": "coexact / local swirl",
    "harmonic": "harmonic / open-cycle residual",
}
BRANCH_COLORS = {
    "exact": "#2f855a",
    "coexact": "#2b6cb0",
    "harmonic": "#805ad5",
}
METRICS = {
    "kl_base_to_steered": "KL movement",
    "next_token_logprob_delta": "Observed next-token support",
    "semantic_margin_delta": "Semantic target-control margin",
}

PAIR_KEYS = [
    "family",
    "prompt_id",
    "layer",
    "k",
    "complex_mode",
    "betti_1_fraction_target",
    "seed",
    "token_selector",
    "selector_component",
    "node_index",
    "token_index",
    "alpha",
]
OPTIONAL_PAIR_KEYS = ["target_set", "random_tangent_reference"]
INFERENCE_KEYS = [
    "layer",
    "k",
    "complex_mode",
    "betti_1_fraction_target",
    "random_tangent_reference",
    "token_selector",
    "selector_component",
    "component",
    "alpha",
    "metric",
]
SIGNED_PROMPT_KEYS = [
    *[key for key in INFERENCE_KEYS if key != "alpha"],
    "family",
    "prompt_id",
    "baseline_component",
]
SIGNED_INFERENCE_KEYS = [
    *[key for key in INFERENCE_KEYS if key != "alpha"],
    "alpha_abs",
    "contrast_type",
]
RESPONSE_PROMPT_KEYS = [
    *[key for key in SIGNED_INFERENCE_KEYS if key != "alpha_abs"],
    "family",
    "prompt_id",
    "baseline_component",
]
RESPONSE_INFERENCE_KEYS = [
    key for key in SIGNED_INFERENCE_KEYS if key != "alpha_abs"
]
SIGNED_CONTRAST_LABELS = {
    "odd": "odd / oriented",
    "even": "even / sign-symmetric",
}
SIGNED_BRANCH_STYLES = {
    "exact": {"color": "#1f4e79", "marker": "o", "linestyle": "-"},
    "coexact": {"color": "#4c78a8", "marker": "s", "linestyle": "--"},
    "harmonic": {"color": "#9ecae1", "marker": "^", "linestyle": ":"},
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def matched_component_gaps(
    rows: pd.DataFrame,
    *,
    baseline_component: str = "random_tangent",
    components: Sequence[str] = BRANCHES,
) -> pd.DataFrame:
    """Pair each Hodge branch with its seed-matched random tangent."""

    required = {
        *PAIR_KEYS,
        "component",
        "component_active",
        *METRICS,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"steering summary is missing columns: {', '.join(missing)}")

    data = rows.copy()
    for column in ["component_active", *METRICS]:
        data[column] = _numeric(data[column])
    keys = [*PAIR_KEYS, *[key for key in OPTIONAL_PAIR_KEYS if key in data.columns]]
    selected = data[data["component"].isin([baseline_component, *components])].copy()
    duplicate = selected.duplicated([*keys, "component"], keep=False)
    if bool(duplicate.any()):
        examples = selected.loc[duplicate, [*keys, "component"]].head(3).to_dict("records")
        raise ValueError(f"duplicate steering rows for the same paired condition: {examples}")

    baseline = selected[selected["component"] == baseline_component].copy()
    baseline = baseline[baseline["component_active"] > 0.0]
    baseline_columns = [*keys, *METRICS]
    baseline = baseline[baseline_columns].rename(
        columns={metric: f"{metric}_baseline" for metric in METRICS}
    )

    output = []
    for component in components:
        branch = selected[selected["component"] == component].copy()
        branch = branch[branch["component_active"] > 0.0]
        branch_columns = [*keys, *METRICS]
        for optional in (
            "hodge_solver",
            "betti_1_fraction",
            "betti_1_fraction_abs_error",
            "cycle_rank",
            "triangle_rank",
            "hodge_exact_ratio",
            "hodge_coexact_ratio",
            "hodge_harmonic_ratio",
        ):
            if optional in branch.columns:
                branch_columns.append(optional)
        paired = branch[branch_columns].merge(
            baseline,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if paired.empty:
            continue
        paired["component"] = component
        paired["baseline_component"] = baseline_component
        for metric in METRICS:
            metric_rows = paired.copy()
            metric_rows["metric"] = metric
            metric_rows["component_value"] = metric_rows[metric]
            metric_rows["baseline_value"] = metric_rows[f"{metric}_baseline"]
            metric_rows["gap"] = metric_rows["component_value"] - metric_rows["baseline_value"]
            metric_rows = metric_rows[np.isfinite(metric_rows["gap"])]
            output.append(metric_rows)
    if not output:
        return pd.DataFrame()
    gaps = pd.concat(output, ignore_index=True, sort=False)
    keep = [
        *keys,
        "component",
        "baseline_component",
        "metric",
        "component_value",
        "baseline_value",
        "gap",
    ]
    keep.extend(
        column
        for column in (
            "hodge_solver",
            "betti_1_fraction",
            "betti_1_fraction_abs_error",
            "cycle_rank",
            "triangle_rank",
            "hodge_exact_ratio",
            "hodge_coexact_ratio",
            "hodge_harmonic_ratio",
        )
        if column in gaps.columns
    )
    return gaps[keep].sort_values(
        ["metric", "component", "alpha", "family", "prompt_id", "seed"]
    )


def collapse_prompt_gaps(gaps: pd.DataFrame) -> pd.DataFrame:
    """Collapse null seeds and selected positions before prompt inference."""

    if gaps.empty:
        return pd.DataFrame()
    data = gaps.copy()
    data["_repeat_key"] = (
        data["seed"].map(lambda value: f"{float(value):.12g}")
        + ":"
        + data["node_index"].map(lambda value: f"{float(value):.12g}")
        + ":"
        + data["token_index"].map(lambda value: f"{float(value):.12g}")
    )
    group_columns = [
        *INFERENCE_KEYS,
        "family",
        "prompt_id",
        "baseline_component",
    ]
    prompt_rows = (
        data.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            gap=("gap", "mean"),
            component_value=("component_value", "mean"),
            baseline_value=("baseline_value", "mean"),
            n_seed_position_pairs=("gap", "size"),
            n_null_seeds=("seed", "nunique"),
            n_positions=("token_index", "nunique"),
            repeat_set=(
                "_repeat_key",
                lambda values: ";".join(sorted(set(values.astype(str)))),
            ),
        )
        .sort_values(["metric", "component", "alpha", "family", "prompt_id"])
    )
    return prompt_rows


def validate_expected_design(
    prompt_rows: pd.DataFrame,
    *,
    expected_prompts: Optional[int] = None,
    expected_null_seeds: Optional[int] = None,
    expected_alpha_magnitudes: Optional[int] = None,
) -> None:
    """Fail before inference when a fixed signed suite is only partially present."""

    expectations = {
        "expected_prompts": expected_prompts,
        "expected_null_seeds": expected_null_seeds,
        "expected_alpha_magnitudes": expected_alpha_magnitudes,
    }
    for name, value in expectations.items():
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive")

    if expected_prompts is not None:
        actual = len(prompt_rows[["family", "prompt_id"]].drop_duplicates())
        if actual != int(expected_prompts):
            raise ValueError(
                f"signed gate expected {int(expected_prompts)} prompts, found {actual}"
            )
    if expected_null_seeds is not None:
        counts = sorted(int(value) for value in prompt_rows["n_null_seeds"].unique())
        if counts != [int(expected_null_seeds)]:
            raise ValueError(
                "signed gate expected "
                f"{int(expected_null_seeds)} null seeds per active prompt condition, "
                f"found counts {counts}"
            )
    if expected_alpha_magnitudes is not None:
        actual = int(prompt_rows["alpha"].astype(float).abs().round(12).nunique())
        if actual != int(expected_alpha_magnitudes):
            raise ValueError(
                "signed gate expected "
                f"{int(expected_alpha_magnitudes)} alpha magnitudes, found {actual}"
            )


def bootstrap_prompt_gaps(
    prompt_rows: pd.DataFrame,
    *,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    zero_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Resample prompt units after all repeated measurements are collapsed."""

    if prompt_rows.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    rows = []
    for key, group in prompt_rows.groupby(INFERENCE_KEYS, sort=True, dropna=False):
        base = dict(zip(INFERENCE_KEYS, key if isinstance(key, tuple) else (key,)))
        values = group["gap"].astype(float).to_numpy()
        draw_indices = rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))
        draws = values[draw_indices].mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                **base,
                "baseline_component": str(group["baseline_component"].iloc[0]),
                "n_prompts": int(len(values)),
                "n_families": int(group["family"].nunique()),
                "mean_gap": float(values.mean()),
                "median_gap": float(np.median(values)),
                "bootstrap_ci_lower": float(lower),
                "bootstrap_ci_upper": float(upper),
                "positive_prompt_fraction": float(np.mean(values > zero_tolerance)),
                "negative_prompt_fraction": float(np.mean(values < -zero_tolerance)),
                "near_zero_prompt_fraction": float(
                    np.mean(np.abs(values) <= zero_tolerance)
                ),
                "bootstrap_samples": int(n_bootstrap),
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "component", "alpha"])


def signed_prompt_contrasts(
    prompt_rows: pd.DataFrame,
    *,
    zero_tolerance: float = 1e-12,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Pair prompt-level +alpha/-alpha gaps and split odd/even effects."""

    if prompt_rows.empty:
        return pd.DataFrame()
    required = {*SIGNED_PROMPT_KEYS, "alpha", "gap"}
    missing = sorted(required.difference(prompt_rows.columns))
    if missing:
        raise ValueError(f"prompt gaps are missing columns: {', '.join(missing)}")

    data = prompt_rows.copy()
    data["alpha"] = _numeric(data["alpha"])
    data["gap"] = _numeric(data["gap"])
    finite = np.isfinite(data["alpha"]) & np.isfinite(data["gap"])
    if not bool(finite.all()):
        raise ValueError("signed gate requires finite alpha and gap values")
    if bool((data["alpha"].abs() <= float(zero_tolerance)).any()):
        raise ValueError("signed gate requires non-zero alpha values")

    data["alpha_abs"] = data["alpha"].abs().round(12)
    data["direction"] = np.where(data["alpha"] > 0.0, "positive", "negative")
    pair_keys = [*SIGNED_PROMPT_KEYS, "alpha_abs"]
    duplicate = data.duplicated([*pair_keys, "direction"], keep=False)
    if bool(duplicate.any()):
        examples = data.loc[duplicate, [*pair_keys, "direction"]].head(3).to_dict("records")
        raise ValueError(f"duplicate prompt rows for the same signed condition: {examples}")

    payload = [
        "alpha",
        "gap",
        "component_value",
        "baseline_value",
        "n_seed_position_pairs",
        "n_null_seeds",
        "n_positions",
        "repeat_set",
    ]
    payload = [column for column in payload if column in data.columns]
    positive = data[data["direction"] == "positive"][[*pair_keys, *payload]].rename(
        columns={column: f"{column}_positive" for column in payload}
    )
    negative = data[data["direction"] == "negative"][[*pair_keys, *payload]].rename(
        columns={column: f"{column}_negative" for column in payload}
    )
    paired = positive.merge(
        negative,
        on=pair_keys,
        how="outer" if require_complete else "inner",
        validate="one_to_one",
        indicator=require_complete,
    )
    if require_complete:
        unmatched = paired[paired["_merge"] != "both"]
        if not unmatched.empty:
            examples = unmatched[[*pair_keys, "_merge"]].head(3).to_dict("records")
            raise ValueError(f"signed gate is missing a +alpha or -alpha prompt pair: {examples}")
        paired = paired.drop(columns="_merge")
    if paired.empty:
        return pd.DataFrame()
    if "repeat_set_positive" in paired.columns:
        repeat_mismatch = paired["repeat_set_positive"] != paired["repeat_set_negative"]
        if bool(repeat_mismatch.any()):
            examples = paired.loc[
                repeat_mismatch,
                [*pair_keys, "repeat_set_positive", "repeat_set_negative"],
            ].head(3).to_dict("records")
            raise ValueError(
                "positive and negative alpha repeat sets differ: "
                f"{examples}"
            )

    output = []
    for contrast_type, values in (
        ("odd", 0.5 * (paired["gap_positive"] - paired["gap_negative"])),
        ("even", 0.5 * (paired["gap_positive"] + paired["gap_negative"])),
    ):
        block = paired.copy()
        block["contrast_type"] = contrast_type
        block["contrast"] = values
        output.append(block)
    signed = pd.concat(output, ignore_index=True, sort=False)
    return signed.sort_values(
        ["metric", "contrast_type", "component", "alpha_abs", "family", "prompt_id"]
    )


def bootstrap_signed_prompt_contrasts(
    signed_rows: pd.DataFrame,
    *,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    zero_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Bootstrap signed contrasts with prompts as the independent units."""

    if signed_rows.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    rows = []
    for key, group in signed_rows.groupby(SIGNED_INFERENCE_KEYS, sort=True, dropna=False):
        base = dict(zip(SIGNED_INFERENCE_KEYS, key if isinstance(key, tuple) else (key,)))
        values = group["contrast"].astype(float).to_numpy()
        draw_indices = rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))
        draws = values[draw_indices].mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                **base,
                "baseline_component": str(group["baseline_component"].iloc[0]),
                "n_prompts": int(len(values)),
                "n_families": int(group["family"].nunique()),
                "mean_contrast": float(values.mean()),
                "median_contrast": float(np.median(values)),
                "bootstrap_ci_lower": float(lower),
                "bootstrap_ci_upper": float(upper),
                "positive_prompt_fraction": float(np.mean(values > zero_tolerance)),
                "negative_prompt_fraction": float(np.mean(values < -zero_tolerance)),
                "near_zero_prompt_fraction": float(
                    np.mean(np.abs(values) <= zero_tolerance)
                ),
                "bootstrap_samples": int(n_bootstrap),
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["metric", "contrast_type", "component", "alpha_abs"]
    )


def fit_prompt_response_coefficients(
    signed_rows: pd.DataFrame,
    *,
    require_complete: bool = False,
) -> pd.DataFrame:
    """Fit odd linear slopes and even quadratic coefficients per prompt."""

    if signed_rows.empty:
        return pd.DataFrame()
    required = {*RESPONSE_PROMPT_KEYS, "alpha_abs", "contrast"}
    missing = sorted(required.difference(signed_rows.columns))
    if missing:
        raise ValueError(f"signed rows are missing columns: {', '.join(missing)}")

    expected_magnitudes = tuple(sorted(signed_rows["alpha_abs"].astype(float).unique()))
    rows = []
    for key, group in signed_rows.groupby(RESPONSE_PROMPT_KEYS, sort=True, dropna=False):
        base = dict(zip(RESPONSE_PROMPT_KEYS, key if isinstance(key, tuple) else (key,)))
        contrast_type = str(base["contrast_type"])
        if contrast_type not in SIGNED_CONTRAST_LABELS:
            raise ValueError(f"unknown signed contrast type: {contrast_type!r}")
        alpha_abs = group["alpha_abs"].astype(float).to_numpy()
        magnitudes = tuple(sorted(np.unique(alpha_abs)))
        if require_complete and magnitudes != expected_magnitudes:
            raise ValueError(
                "prompt response is missing alpha magnitudes: "
                f"{base}; expected {expected_magnitudes}, found {magnitudes}"
            )
        values = group["contrast"].astype(float).to_numpy()
        power = 1 if contrast_type == "odd" else 2
        predictor = alpha_abs**power
        denominator = float(np.dot(predictor, predictor))
        if denominator <= 0.0:
            raise ValueError("response coefficient fit requires positive alpha magnitudes")
        coefficient = float(np.dot(predictor, values) / denominator)
        residual = values - coefficient * predictor
        rmse = float(np.sqrt(np.mean(residual**2)))
        rms = float(np.sqrt(np.mean(values**2)))
        rows.append(
            {
                **base,
                "response_power": int(power),
                "response_coefficient": coefficient,
                "response_rmse": rmse,
                "relative_response_rmse": float(rmse / rms) if rms > 0.0 else 0.0,
                "n_alpha_magnitudes": int(len(magnitudes)),
                "min_alpha_abs": float(alpha_abs.min()),
                "max_alpha_abs": float(alpha_abs.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["metric", "contrast_type", "component", "family", "prompt_id"]
    )


def bootstrap_response_coefficients(
    prompt_coefficients: pd.DataFrame,
    *,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    zero_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Bootstrap prompt-level signed response coefficients."""

    if prompt_coefficients.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    rows = []
    for key, group in prompt_coefficients.groupby(
        RESPONSE_INFERENCE_KEYS,
        sort=True,
        dropna=False,
    ):
        base = dict(zip(RESPONSE_INFERENCE_KEYS, key if isinstance(key, tuple) else (key,)))
        values = group["response_coefficient"].astype(float).to_numpy()
        draw_indices = rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))
        draws = values[draw_indices].mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                **base,
                "baseline_component": str(group["baseline_component"].iloc[0]),
                "response_power": int(group["response_power"].iloc[0]),
                "n_alpha_magnitudes": int(group["n_alpha_magnitudes"].min()),
                "n_prompts": int(len(values)),
                "n_families": int(group["family"].nunique()),
                "mean_response_coefficient": float(values.mean()),
                "median_response_coefficient": float(np.median(values)),
                "bootstrap_ci_lower": float(lower),
                "bootstrap_ci_upper": float(upper),
                "positive_prompt_fraction": float(np.mean(values > zero_tolerance)),
                "negative_prompt_fraction": float(np.mean(values < -zero_tolerance)),
                "mean_relative_response_rmse": float(
                    group["relative_response_rmse"].astype(float).mean()
                ),
                "bootstrap_samples": int(n_bootstrap),
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "contrast_type", "component"])


def plot_causal_branch_gaps(
    prompt_rows: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.7), constrained_layout=True)
    available_alphas = sorted(float(value) for value in inference["alpha"].unique())
    if len(available_alphas) > 1:
        spacing = min(np.diff(available_alphas))
    else:
        spacing = max(abs(available_alphas[0]), 1.0) if available_alphas else 1.0
    offsets = {"exact": -0.08 * spacing, "coexact": 0.0, "harmonic": 0.08 * spacing}

    for ax, (metric, title) in zip(axes, METRICS.items()):
        metric_summary = inference[inference["metric"] == metric]
        metric_prompts = prompt_rows[prompt_rows["metric"] == metric]
        if metric_summary.empty:
            ax.text(0.5, 0.5, "metric unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_axis_off()
            continue
        for component in BRANCHES:
            summary = metric_summary[metric_summary["component"] == component].sort_values("alpha")
            points = metric_prompts[metric_prompts["component"] == component]
            if summary.empty:
                continue
            color = BRANCH_COLORS[component]
            offset = offsets[component]
            for alpha, group in points.groupby("alpha", sort=True):
                x = np.full(len(group), float(alpha) + offset)
                ax.scatter(x, group["gap"], s=10, color=color, alpha=0.18, linewidths=0)
            x = summary["alpha"].astype(float).to_numpy() + offset
            center = summary["mean_gap"].astype(float).to_numpy()
            lower = summary["bootstrap_ci_lower"].astype(float).to_numpy()
            upper = summary["bootstrap_ci_upper"].astype(float).to_numpy()
            ax.errorbar(
                x,
                center,
                yerr=np.vstack([center - lower, upper - center]),
                color=color,
                marker="o",
                markersize=5,
                capsize=3,
                linewidth=1.5,
                label=BRANCH_LABELS[component],
            )
        ax.axhline(0.0, color="#4a5568", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("steering strength alpha")
        ax.set_ylabel("branch minus random tangent")
        ax.set_xticks(available_alphas)
        ax.grid(alpha=0.20)

    if not inference.empty:
        first = inference.iloc[0]
        subtitle = (
            f"L{int(first['layer'])}, k={int(first['k'])}, "
            f"complex={first['complex_mode']}, Betti-1 target={float(first['betti_1_fraction_target']):g}; "
            "dots are prompt means, bars are 95% prompt-bootstrap CIs"
        )
    else:
        subtitle = "branch-minus-random matched contrasts"
    fig.suptitle(f"HLTD matched-topology one-step causal gate\n{subtitle}", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_signed_causal_contrasts(
    signed_rows: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    """Plot oriented and sign-symmetric branch-minus-random contrasts."""

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.2), constrained_layout=True)
    available_alphas = sorted(float(value) for value in inference["alpha_abs"].unique())
    if len(available_alphas) > 1:
        spacing = min(np.diff(available_alphas))
    else:
        spacing = max(available_alphas[0], 1.0) if available_alphas else 1.0
    offsets = {"exact": -0.08 * spacing, "coexact": 0.0, "harmonic": 0.08 * spacing}

    for row_index, contrast_type in enumerate(("odd", "even")):
        for column_index, (metric, title) in enumerate(METRICS.items()):
            ax = axes[row_index, column_index]
            metric_summary = inference[
                (inference["metric"] == metric)
                & (inference["contrast_type"] == contrast_type)
            ]
            metric_prompts = signed_rows[
                (signed_rows["metric"] == metric)
                & (signed_rows["contrast_type"] == contrast_type)
            ]
            if metric_summary.empty:
                ax.text(
                    0.5,
                    0.5,
                    "metric unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(title)
                ax.set_axis_off()
                continue
            for component in BRANCHES:
                summary = metric_summary[
                    metric_summary["component"] == component
                ].sort_values("alpha_abs")
                points = metric_prompts[metric_prompts["component"] == component]
                if summary.empty:
                    continue
                style = SIGNED_BRANCH_STYLES[component]
                offset = offsets[component]
                for alpha_abs, group in points.groupby("alpha_abs", sort=True):
                    x = np.full(len(group), float(alpha_abs) + offset)
                    ax.scatter(
                        x,
                        group["contrast"],
                        s=12,
                        facecolors="none",
                        edgecolors=style["color"],
                        marker=style["marker"],
                        alpha=0.24,
                        linewidths=0.65,
                    )
                x = summary["alpha_abs"].astype(float).to_numpy() + offset
                center = summary["mean_contrast"].astype(float).to_numpy()
                lower = summary["bootstrap_ci_lower"].astype(float).to_numpy()
                upper = summary["bootstrap_ci_upper"].astype(float).to_numpy()
                ax.errorbar(
                    x,
                    center,
                    yerr=np.vstack([center - lower, upper - center]),
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    markersize=5,
                    capsize=3,
                    linewidth=1.5,
                    label=BRANCH_LABELS[component],
                )
            ax.axhline(0.0, color="#4a5568", linewidth=0.9)
            if row_index == 0:
                ax.set_title(title)
            if column_index == 0:
                ax.set_ylabel(f"{SIGNED_CONTRAST_LABELS[contrast_type]} contrast")
            if row_index == 1:
                ax.set_xlabel("absolute steering strength |alpha|")
            ax.set_xticks(available_alphas)
            ax.grid(color="#d8dee6", alpha=0.65, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)

    first = inference.iloc[0]
    subtitle = (
        f"L{int(first['layer'])}, k={int(first['k'])}, "
        f"complex={first['complex_mode']}, Betti-1 target={float(first['betti_1_fraction_target']):g}; "
        "open marks are prompt means, bars are 95% prompt-bootstrap CIs"
    )
    fig.suptitle(f"HLTD signed matched-topology causal contrasts\n{subtitle}", fontsize=13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_signed_response_coefficients(
    inference: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    """Plot prompt-bootstrap odd slopes and even curvature coefficients."""

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2), constrained_layout=True)
    y_positions = {"exact": 2.0, "coexact": 1.0, "harmonic": 0.0}
    y_labels = [BRANCH_LABELS["harmonic"], BRANCH_LABELS["coexact"], BRANCH_LABELS["exact"]]

    for row_index, contrast_type in enumerate(("odd", "even")):
        for column_index, (metric, title) in enumerate(METRICS.items()):
            ax = axes[row_index, column_index]
            panel = inference[
                (inference["metric"] == metric)
                & (inference["contrast_type"] == contrast_type)
            ]
            for component in BRANCHES:
                rows = panel[panel["component"] == component]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                center = float(row["mean_response_coefficient"])
                lower = float(row["bootstrap_ci_lower"])
                upper = float(row["bootstrap_ci_upper"])
                style = SIGNED_BRANCH_STYLES[component]
                ax.errorbar(
                    center,
                    y_positions[component],
                    xerr=np.asarray([[center - lower], [upper - center]]),
                    color=style["color"],
                    marker=style["marker"],
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    markersize=6,
                    capsize=4,
                    linewidth=1.7,
                )
            ax.axvline(0.0, color="#4a5568", linewidth=0.9)
            if row_index == 0:
                ax.set_title(title)
            if contrast_type == "odd":
                ax.set_xlabel("odd slope\ncontrast / |alpha|", fontsize=9)
            else:
                ax.set_xlabel("even curvature\ncontrast / |alpha|^2", fontsize=9)
            ax.set_yticks([0.0, 1.0, 2.0])
            if column_index == 0:
                ax.set_yticklabels(y_labels)
            else:
                ax.set_yticklabels([])
            ax.grid(axis="x", color="#d8dee6", alpha=0.7, linewidth=0.7)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)

    first = inference.iloc[0]
    subtitle = (
        f"L{int(first['layer'])}, k={int(first['k'])}, "
        f"complex={first['complex_mode']}, Betti-1 target={float(first['betti_1_fraction_target']):g}; "
        "odd fit against |alpha|, even fit against |alpha|^2; means and 95% prompt-bootstrap CIs"
    )
    fig.suptitle(f"HLTD signed response coefficients\n{subtitle}", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def write_report(
    inference: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    if inference.empty:
        output_path.write_text("# HLTD Matched-Betti Causal Gate\n\nNo paired rows.\n", encoding="utf-8")
        return
    first = inference.iloc[0]
    lines = [
        "# HLTD Matched-Betti Causal Gate",
        "",
        "## Contract",
        "",
        f"- layer: L{int(first['layer'])}",
        f"- k: {int(first['k'])}",
        f"- complex: {first['complex_mode']}",
        f"- Betti-1 target: {float(first['betti_1_fraction_target']):g}",
        f"- token selector: {first['token_selector']}",
        f"- random tangent reference: {first.get('random_tangent_reference', 'unspecified')}",
        f"- prompt bootstrap: {int(first['bootstrap_samples'])} draws, seed {int(first['bootstrap_seed'])}",
        "- contrast: active Hodge branch minus active norm-matched random tangent",
        "- inference unit: prompt after averaging null seeds and selected positions",
        "- inactive local branches: omitted from that branch contrast, never zero-imputed",
        "",
        "## Prompt-Level Contrasts",
        "",
        "Positive values mean the Hodge branch exceeds its paired random tangent.",
        "",
        "| metric | branch | alpha | prompts | mean gap | 95% prompt CI | positive prompts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in inference.iterrows():
        lines.append(
            f"| {METRICS.get(str(row['metric']), row['metric'])} | "
            f"{BRANCH_LABELS.get(str(row['component']), row['component'])} | "
            f"{float(row['alpha']):g} | {int(row['n_prompts'])} | "
            f"{float(row['mean_gap']):+.5f} | "
            f"[{float(row['bootstrap_ci_lower']):+.5f}, {float(row['bootstrap_ci_upper']):+.5f}] | "
            f"{int(round(float(row['positive_prompt_fraction']) * int(row['n_prompts'])))}/{int(row['n_prompts'])} |"
        )
    prompt_counts = sorted(int(value) for value in inference["n_prompts"].unique())
    if len(prompt_counts) > 1:
        lines.extend(
            [
                "",
                f"Prompt counts vary across branches ({', '.join(str(value) for value in prompt_counts)})",
                "because an exactly zero local component has no steering direction.",
            ]
        )
    lines.extend(
        [
            "",
            "## Conservative Read",
            "",
            "This is an immediate next-token intervention gate. It does not establish",
            "multi-step semantic drift, preserved fluency, or a global concept ring.",
            "At the matched complex, `harmonic` denotes the residual carried by open graph",
            "cycles at the selected first-homology capacity.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_signed_report(
    inference: pd.DataFrame,
    *,
    output_path: Path,
    response_inference: Optional[pd.DataFrame] = None,
) -> None:
    if inference.empty:
        output_path.write_text(
            "# HLTD Signed Matched-Betti Causal Gate\n\nNo complete signed pairs.\n",
            encoding="utf-8",
        )
        return
    first = inference.iloc[0]
    lines = [
        "# HLTD Signed Matched-Betti Causal Gate",
        "",
        "## Contract",
        "",
        f"- layer: L{int(first['layer'])}",
        f"- k: {int(first['k'])}",
        f"- complex: {first['complex_mode']}",
        f"- Betti-1 target: {float(first['betti_1_fraction_target']):g}",
        f"- token selector: {first['token_selector']}",
        f"- random tangent reference: {first.get('random_tangent_reference', 'unspecified')}",
        f"- prompt bootstrap: {int(first['bootstrap_samples'])} draws, seed {int(first['bootstrap_seed'])}",
        "- base contrast: active Hodge branch minus its seed-matched active random tangent",
        "- signed pairing: same prompt after averaging null seeds and selected positions",
        "- odd(a): (gap(+a) - gap(-a)) / 2",
        "- even(a): (gap(+a) + gap(-a)) / 2",
        "- inactive local branches: omitted from that branch contrast, never zero-imputed",
        "",
        "## Prompt-Level Signed Contrasts",
        "",
        "Odd estimates oriented transport; even estimates sign-symmetric magnitude or disruption.",
        "",
        "| parity | metric | branch | |alpha| | prompts | mean | 95% prompt CI | positive prompts |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in inference.iterrows():
        lines.append(
            f"| {SIGNED_CONTRAST_LABELS.get(str(row['contrast_type']), row['contrast_type'])} | "
            f"{METRICS.get(str(row['metric']), row['metric'])} | "
            f"{BRANCH_LABELS.get(str(row['component']), row['component'])} | "
            f"{float(row['alpha_abs']):g} | {int(row['n_prompts'])} | "
            f"{float(row['mean_contrast']):+.5f} | "
            f"[{float(row['bootstrap_ci_lower']):+.5f}, {float(row['bootstrap_ci_upper']):+.5f}] | "
            f"{int(round(float(row['positive_prompt_fraction']) * int(row['n_prompts'])))}"
            f"/{int(row['n_prompts'])} |"
        )

    oriented = inference[
        (inference["contrast_type"] == "odd")
        & (
            (inference["bootstrap_ci_lower"] > 0.0)
            | (inference["bootstrap_ci_upper"] < 0.0)
        )
    ]
    symmetric = inference[
        (inference["contrast_type"] == "even")
        & (
            (inference["bootstrap_ci_lower"] > 0.0)
            | (inference["bootstrap_ci_upper"] < 0.0)
        )
    ]
    lines.extend(
        [
            "",
            "## Interval Screen",
            "",
            f"- odd intervals excluding zero: {len(oriented)}/{int((inference['contrast_type'] == 'odd').sum())}",
            f"- even intervals excluding zero: {len(symmetric)}/{int((inference['contrast_type'] == 'even').sum())}",
            "",
            "Intervals are descriptive gate outcomes under this fixed prompt suite, not a",
            "multiple-comparison-corrected mechanism claim.",
        ]
    )
    if response_inference is not None and not response_inference.empty:
        lines.extend(
            [
                "",
                "## Signed Response Coefficients",
                "",
                "Odd trajectories are fit through zero against |alpha|; even trajectories",
                "are fit through zero against |alpha|^2. Each coefficient is fit per prompt",
                "before prompt bootstrap.",
                "",
                "| parity | metric | branch | prompts | mean coefficient | 95% prompt CI | mean relative RMSE |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in response_inference.iterrows():
            lines.append(
                f"| {SIGNED_CONTRAST_LABELS.get(str(row['contrast_type']), row['contrast_type'])} | "
                f"{METRICS.get(str(row['metric']), row['metric'])} | "
                f"{BRANCH_LABELS.get(str(row['component']), row['component'])} | "
                f"{int(row['n_prompts'])} | "
                f"{float(row['mean_response_coefficient']):+.5f} | "
                f"[{float(row['bootstrap_ci_lower']):+.5f}, {float(row['bootstrap_ci_upper']):+.5f}] | "
                f"{float(row['mean_relative_response_rmse']):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Conservative Read",
            "",
            "An odd effect is the relevant evidence for branch orientation. An even-only",
            "effect is compatible with branch-specific perturbation magnitude, curvature,",
            "or generic nonlinear disruption. This one-step gate still does not establish",
            "multi-step semantic drift, preserved fluency, or a harmonic concept ring.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_all(
    *,
    summary_path: Path,
    output_root: Path,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    require_signed: bool = False,
    expected_prompts: Optional[int] = None,
    expected_null_seeds: Optional[int] = None,
    expected_alpha_magnitudes: Optional[int] = None,
) -> None:
    rows = pd.read_csv(summary_path)
    gaps = matched_component_gaps(rows)
    if gaps.empty:
        raise ValueError("No active exact/coexact/harmonic rows could be paired with random_tangent.")
    prompt_rows = collapse_prompt_gaps(gaps)
    validate_expected_design(
        prompt_rows,
        expected_prompts=expected_prompts,
        expected_null_seeds=expected_null_seeds,
        expected_alpha_magnitudes=expected_alpha_magnitudes,
    )
    inference = bootstrap_prompt_gaps(
        prompt_rows,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    signed_rows = pd.DataFrame()
    signed_inference = pd.DataFrame()
    prompt_coefficients = pd.DataFrame()
    response_inference = pd.DataFrame()
    alpha_values = prompt_rows["alpha"].astype(float)
    has_positive = bool((alpha_values > 0.0).any())
    has_negative = bool((alpha_values < 0.0).any())
    if has_positive and has_negative:
        signed_rows = signed_prompt_contrasts(
            prompt_rows,
            require_complete=require_signed,
        )
        signed_inference = bootstrap_signed_prompt_contrasts(
            signed_rows,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        prompt_coefficients = fit_prompt_response_coefficients(
            signed_rows,
            require_complete=require_signed,
        )
        response_inference = bootstrap_response_coefficients(
            prompt_coefficients,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    elif require_signed:
        directions = []
        if has_positive:
            directions.append("positive")
        if has_negative:
            directions.append("negative")
        found = ", ".join(directions) or "none"
        raise ValueError(f"signed gate requires both positive and negative alpha; found {found}")

    output_root.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(output_root / "summary_branch_minus_random_pairs.csv", index=False)
    prompt_rows.to_csv(output_root / "summary_prompt_branch_gaps.csv", index=False)
    inference.to_csv(output_root / "summary_prompt_bootstrap.csv", index=False)
    plot_causal_branch_gaps(
        prompt_rows,
        inference,
        output_path=output_root / "plots" / "matched_betti_causal_branch_gaps.png",
    )
    write_report(inference, output_path=output_root / "summary_causal_report.md")
    print(f"saved causal plot: {output_root / 'plots' / 'matched_betti_causal_branch_gaps.png'}")

    if not signed_rows.empty:
        signed_rows.to_csv(output_root / "summary_signed_prompt_contrasts.csv", index=False)
        signed_inference.to_csv(output_root / "summary_signed_prompt_bootstrap.csv", index=False)
        prompt_coefficients.to_csv(
            output_root / "summary_signed_prompt_response_coefficients.csv",
            index=False,
        )
        response_inference.to_csv(
            output_root / "summary_signed_response_coefficient_bootstrap.csv",
            index=False,
        )
        plot_signed_causal_contrasts(
            signed_rows,
            signed_inference,
            output_path=output_root / "plots" / "matched_betti_signed_causal_odd_even.png",
        )
        plot_signed_response_coefficients(
            response_inference,
            output_path=output_root / "plots" / "matched_betti_signed_response_coefficients.png",
        )
        write_signed_report(
            signed_inference,
            output_path=output_root / "summary_signed_causal_report.md",
            response_inference=response_inference,
        )
        print(
            "saved signed causal plot: "
            f"{output_root / 'plots' / 'matched_betti_signed_causal_odd_even.png'}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Raw steering suite summary.csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--require-signed",
        action="store_true",
        help="Require complete prompt-level +alpha/-alpha pairs and emit odd/even outputs",
    )
    parser.add_argument("--expected-prompts", type=int, default=None)
    parser.add_argument("--expected-null-seeds", type=int, default=None)
    parser.add_argument("--expected-alpha-magnitudes", type=int, default=None)
    args = parser.parse_args(argv)
    render_all(
        summary_path=Path(args.summary),
        output_root=Path(args.output_root),
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        require_signed=args.require_signed,
        expected_prompts=args.expected_prompts,
        expected_null_seeds=args.expected_null_seeds,
        expected_alpha_magnitudes=args.expected_alpha_magnitudes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
