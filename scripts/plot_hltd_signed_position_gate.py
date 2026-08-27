#!/usr/bin/env python3
"""Localize signed matched-Betti coexact steering across token positions."""
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
DEFAULT_BINS = 12
PRIMARY_COLOR = "#2b6cb0"
METRICS = {
    "next_token_logprob_delta": "Observed next-token support",
    "semantic_margin_delta": "Semantic target-control margin",
}
FAMILY_STYLES = {
    "literal_stable": {"color": "#2b6cb0", "marker": "o", "linestyle": "-"},
    "metaphor_shift": {"color": "#dd6b20", "marker": "s", "linestyle": "--"},
    "identity_stress": {"color": "#805ad5", "marker": "^", "linestyle": "-."},
    "ontology_collapse": {"color": "#4a5568", "marker": "D", "linestyle": ":"},
}

TOPOLOGY_KEYS = [
    "layer",
    "k",
    "complex_mode",
    "betti_1_fraction_target",
    "random_tangent_reference",
    "token_selector",
    "selector_component",
]
PAIR_KEYS = [
    "family",
    "prompt_id",
    *TOPOLOGY_KEYS,
    "seed",
    "node_index",
    "token_index",
    "token_count",
    "alpha",
]
OPTIONAL_PAIR_KEYS = ["target_set"]
TOKEN_KEYS = [
    *TOPOLOGY_KEYS,
    "component",
    "baseline_component",
    "family",
    "prompt_id",
    "node_index",
    "token_index",
    "token_count",
    "position_frac",
    "position_bin",
    "metric",
]
PROMPT_BIN_KEYS = [
    *TOPOLOGY_KEYS,
    "component",
    "baseline_component",
    "family",
    "prompt_id",
    "position_bin",
    "metric",
    "contrast_type",
]
POSITION_INFERENCE_KEYS = [
    *TOPOLOGY_KEYS,
    "component",
    "baseline_component",
    "position_bin",
    "metric",
    "contrast_type",
]
PROMPT_PROFILE_KEYS = [
    *TOPOLOGY_KEYS,
    "component",
    "baseline_component",
    "family",
    "prompt_id",
    "metric",
    "contrast_type",
]
PHASE_INFERENCE_KEYS = [
    *TOPOLOGY_KEYS,
    "component",
    "baseline_component",
    "metric",
    "contrast_type",
    "position_phase",
]
TREND_INFERENCE_KEYS = [
    *TOPOLOGY_KEYS,
    "component",
    "baseline_component",
    "metric",
    "contrast_type",
]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def position_bin(position_frac: float, bins: int) -> int:
    if bins <= 0:
        raise ValueError("bins must be positive")
    clipped = min(max(float(position_frac), 0.0), 1.0)
    return min(int(clipped * bins), bins - 1)


def _attach_positions(rows: pd.DataFrame, *, bins: int) -> pd.DataFrame:
    data = rows.copy()
    data["token_index"] = _numeric(data["token_index"])
    data["token_count"] = _numeric(data["token_count"])
    valid = (
        np.isfinite(data["token_index"])
        & np.isfinite(data["token_count"])
        & (data["token_count"] >= 3)
        & (data["token_index"] >= 1)
        & (data["token_index"] <= data["token_count"] - 2)
    )
    if not bool(valid.all()):
        examples = data.loc[~valid, ["prompt_id", "token_index", "token_count"]].head(3)
        raise ValueError(f"all-interior gate contains invalid token positions: {examples.to_dict('records')}")
    data["token_index"] = data["token_index"].astype(int)
    data["token_count"] = data["token_count"].astype(int)
    data["position_frac"] = data["token_index"] / (data["token_count"] - 1)
    data["position_bin"] = data["position_frac"].map(lambda value: position_bin(value, bins))
    return data


def validate_expected_design(
    rows: pd.DataFrame,
    *,
    component: str,
    token_selector: str,
    expected_prompts: Optional[int] = None,
    expected_null_seeds: Optional[int] = None,
    expected_alpha_magnitudes: Optional[int] = None,
) -> None:
    """Reject a truncated fixed suite before branch activity filtering."""

    expectations = {
        "expected_prompts": expected_prompts,
        "expected_null_seeds": expected_null_seeds,
        "expected_alpha_magnitudes": expected_alpha_magnitudes,
    }
    for name, value in expectations.items():
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive")

    selected = rows[
        (rows["component"] == component) & (rows["token_selector"] == token_selector)
    ].copy()
    if selected.empty:
        raise ValueError(f"no raw {component!r} rows for token selector {token_selector!r}")
    selected["seed"] = _numeric(selected["seed"])
    selected["alpha_abs"] = _numeric(selected["alpha"]).abs().round(12)
    prompt_keys = ["family", "prompt_id"]

    if expected_prompts is not None:
        actual = len(selected[prompt_keys].drop_duplicates())
        if actual != int(expected_prompts):
            raise ValueError(
                f"signed position gate expected {int(expected_prompts)} prompts, found {actual}"
            )
    if expected_null_seeds is not None:
        counts = sorted(
            int(value)
            for value in selected.groupby(prompt_keys, dropna=False)["seed"].nunique().unique()
        )
        if counts != [int(expected_null_seeds)]:
            raise ValueError(
                "signed position gate expected "
                f"{int(expected_null_seeds)} null seeds per prompt, found counts {counts}"
            )
    if expected_alpha_magnitudes is not None:
        counts = sorted(
            int(value)
            for value in selected.groupby(prompt_keys, dropna=False)["alpha_abs"].nunique().unique()
        )
        if counts != [int(expected_alpha_magnitudes)]:
            raise ValueError(
                "signed position gate expected "
                f"{int(expected_alpha_magnitudes)} alpha magnitudes per prompt, "
                f"found counts {counts}"
            )


def seed_matched_component_gaps(
    rows: pd.DataFrame,
    *,
    bins: int = DEFAULT_BINS,
    component: str = "coexact",
    baseline_component: str = "random_tangent",
    token_selector: str = "all_interior",
) -> pd.DataFrame:
    """Pair a branch with its random tangent at each seed, token, and alpha."""

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
    for column in ["component_active", "alpha", "seed", *METRICS]:
        data[column] = _numeric(data[column])
    if not bool(np.isfinite(data["alpha"]).all()):
        raise ValueError("signed position gate requires finite alpha values")
    if bool((data["alpha"].abs() <= 1e-12).any()):
        raise ValueError("signed position gate requires non-zero alpha values")

    data = data[
        (data["token_selector"] == token_selector)
        & data["component"].isin([component, baseline_component])
    ].copy()
    if data.empty:
        raise ValueError(
            f"no {token_selector!r} rows for {component!r} and {baseline_component!r}"
        )
    data = _attach_positions(data, bins=bins)
    keys = [*PAIR_KEYS, *[key for key in OPTIONAL_PAIR_KEYS if key in data.columns]]
    duplicate = data.duplicated([*keys, "component"], keep=False)
    if bool(duplicate.any()):
        examples = data.loc[duplicate, [*keys, "component"]].head(3).to_dict("records")
        raise ValueError(f"duplicate rows for the same seed-matched condition: {examples}")

    branch = data[
        (data["component"] == component) & (data["component_active"] > 0.0)
    ].copy()
    baseline = data[
        (data["component"] == baseline_component) & (data["component_active"] > 0.0)
    ].copy()
    branch_columns = [*keys, "position_frac", "position_bin", *METRICS]
    baseline_columns = [*keys, *METRICS]
    baseline = baseline[baseline_columns].rename(
        columns={metric: f"{metric}_baseline" for metric in METRICS}
    )
    paired = branch[branch_columns].merge(
        baseline,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError(
            f"no active {component!r} rows could be paired with {baseline_component!r}"
        )

    blocks = []
    for metric in METRICS:
        block = paired[[*keys, "position_frac", "position_bin", metric, f"{metric}_baseline"]].copy()
        block["component"] = component
        block["baseline_component"] = baseline_component
        block["metric"] = metric
        block["component_value"] = block[metric]
        block["baseline_value"] = block[f"{metric}_baseline"]
        block["gap"] = block["component_value"] - block["baseline_value"]
        block = block[np.isfinite(block["gap"])]
        blocks.append(block)
    gaps = pd.concat(blocks, ignore_index=True, sort=False)
    keep = [
        *keys,
        "position_frac",
        "position_bin",
        "component",
        "baseline_component",
        "metric",
        "component_value",
        "baseline_value",
        "gap",
    ]
    return gaps[keep].sort_values(
        ["metric", "alpha", "family", "prompt_id", "token_index", "seed"]
    )


def collapse_seed_gaps(gaps: pd.DataFrame) -> pd.DataFrame:
    """Average random-tangent seeds at each exact prompt/token/sign condition."""

    if gaps.empty:
        return pd.DataFrame()
    group_columns = [*TOKEN_KEYS, "alpha"]

    def seed_signature(values: pd.Series) -> str:
        return ",".join(str(int(value)) for value in sorted(set(values.astype(int))))

    return (
        gaps.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            gap=("gap", "mean"),
            component_value=("component_value", "mean"),
            baseline_value=("baseline_value", "mean"),
            n_null_seeds=("seed", "nunique"),
            n_seed_pairs=("seed", "size"),
            seed_set=("seed", seed_signature),
        )
        .sort_values(["metric", "alpha", "family", "prompt_id", "token_index"])
    )


def signed_token_contrasts(seed_rows: pd.DataFrame) -> pd.DataFrame:
    """Construct strict +alpha/-alpha odd and even contrasts per exact token."""

    if seed_rows.empty:
        return pd.DataFrame()
    required = {*TOKEN_KEYS, "alpha", "gap", "seed_set"}
    missing = sorted(required.difference(seed_rows.columns))
    if missing:
        raise ValueError(f"seed-collapsed gaps are missing columns: {', '.join(missing)}")

    data = seed_rows.copy()
    data["alpha"] = _numeric(data["alpha"])
    data["alpha_abs"] = data["alpha"].abs().round(12)
    data["direction"] = np.where(data["alpha"] > 0.0, "positive", "negative")
    pair_keys = [*TOKEN_KEYS, "alpha_abs"]
    duplicate = data.duplicated([*pair_keys, "direction"], keep=False)
    if bool(duplicate.any()):
        examples = data.loc[duplicate, [*pair_keys, "direction"]].head(3).to_dict("records")
        raise ValueError(f"duplicate signed token conditions: {examples}")

    payload = [
        "alpha",
        "gap",
        "component_value",
        "baseline_value",
        "n_null_seeds",
        "n_seed_pairs",
        "seed_set",
    ]
    positive = data[data["direction"] == "positive"][[*pair_keys, *payload]].rename(
        columns={column: f"{column}_positive" for column in payload}
    )
    negative = data[data["direction"] == "negative"][[*pair_keys, *payload]].rename(
        columns={column: f"{column}_negative" for column in payload}
    )
    paired = positive.merge(
        negative,
        on=pair_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = paired[paired["_merge"] != "both"]
    if not unmatched.empty:
        examples = unmatched[[*pair_keys, "_merge"]].head(3).to_dict("records")
        raise ValueError(f"signed position gate is missing a +alpha or -alpha token pair: {examples}")
    paired = paired.drop(columns="_merge")
    seed_mismatch = paired["seed_set_positive"] != paired["seed_set_negative"]
    if bool(seed_mismatch.any()):
        examples = paired.loc[
            seed_mismatch,
            [*pair_keys, "seed_set_positive", "seed_set_negative"],
        ].head(3).to_dict("records")
        raise ValueError(f"positive and negative alpha seed sets differ: {examples}")

    output = []
    for contrast_type, values in (
        ("odd", 0.5 * (paired["gap_positive"] - paired["gap_negative"])),
        ("even", 0.5 * (paired["gap_positive"] + paired["gap_negative"])),
    ):
        block = paired.copy()
        block["contrast_type"] = contrast_type
        block["contrast"] = values
        output.append(block)
    return pd.concat(output, ignore_index=True, sort=False).sort_values(
        ["metric", "contrast_type", "alpha_abs", "family", "prompt_id", "token_index"]
    )


def collapse_prompt_bin_contrasts(signed_rows: pd.DataFrame) -> pd.DataFrame:
    """Average exact-token contrasts inside each prompt and position bin."""

    if signed_rows.empty:
        return pd.DataFrame()
    group_columns = [*PROMPT_BIN_KEYS, "alpha_abs"]

    def token_signature(values: pd.Series) -> str:
        return ",".join(str(int(value)) for value in sorted(set(values.astype(int))))

    return (
        signed_rows.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            contrast=("contrast", "mean"),
            n_tokens=("token_index", "nunique"),
            n_prompt_token_pairs=("token_index", "size"),
            token_set=("token_index", token_signature),
            mean_position_frac=("position_frac", "mean"),
            min_position_frac=("position_frac", "min"),
            max_position_frac=("position_frac", "max"),
            min_null_seeds=("n_null_seeds_positive", "min"),
            max_null_seeds=("n_null_seeds_positive", "max"),
        )
        .sort_values(
            ["metric", "contrast_type", "position_bin", "family", "prompt_id", "alpha_abs"]
        )
    )


def fit_prompt_bin_response_coefficients(
    prompt_bin_rows: pd.DataFrame,
    *,
    bins: int = DEFAULT_BINS,
) -> pd.DataFrame:
    """Fit odd slopes and even curvature after prompt-local token averaging."""

    if prompt_bin_rows.empty:
        return pd.DataFrame()
    required = {*PROMPT_BIN_KEYS, "alpha_abs", "contrast", "token_set"}
    missing = sorted(required.difference(prompt_bin_rows.columns))
    if missing:
        raise ValueError(f"prompt-bin contrasts are missing columns: {', '.join(missing)}")

    expected_magnitudes = tuple(sorted(prompt_bin_rows["alpha_abs"].astype(float).unique()))
    if len(expected_magnitudes) < 2:
        raise ValueError("position response fit requires at least two alpha magnitudes")
    rows = []
    for key, group in prompt_bin_rows.groupby(PROMPT_BIN_KEYS, sort=True, dropna=False):
        base = dict(zip(PROMPT_BIN_KEYS, key if isinstance(key, tuple) else (key,)))
        magnitudes = tuple(sorted(group["alpha_abs"].astype(float).unique()))
        if magnitudes != expected_magnitudes:
            raise ValueError(
                "prompt-bin response is missing alpha magnitudes: "
                f"{base}; expected {expected_magnitudes}, found {magnitudes}"
            )
        if group["token_set"].nunique() != 1:
            raise ValueError(f"active token set changes across alpha magnitudes: {base}")
        contrast_type = str(base["contrast_type"])
        if contrast_type not in {"odd", "even"}:
            raise ValueError(f"unknown signed contrast type: {contrast_type!r}")
        alpha_abs = group["alpha_abs"].astype(float).to_numpy()
        values = group["contrast"].astype(float).to_numpy()
        power = 1 if contrast_type == "odd" else 2
        predictor = alpha_abs**power
        denominator = float(np.dot(predictor, predictor))
        coefficient = float(np.dot(predictor, values) / denominator)
        residual = values - coefficient * predictor
        rmse = float(np.sqrt(np.mean(residual**2)))
        rms = float(np.sqrt(np.mean(values**2)))
        rows.append(
            {
                **base,
                "position_bin_center": (int(base["position_bin"]) + 0.5) / bins,
                "response_power": power,
                "response_coefficient": coefficient,
                "response_rmse": rmse,
                "relative_response_rmse": float(rmse / rms) if rms > 0.0 else 0.0,
                "n_alpha_magnitudes": len(magnitudes),
                "min_alpha_abs": min(magnitudes),
                "max_alpha_abs": max(magnitudes),
                "n_tokens": int(group["n_tokens"].iloc[0]),
                "token_set": str(group["token_set"].iloc[0]),
                "mean_position_frac": float(group["mean_position_frac"].mean()),
                "min_null_seeds": int(group["min_null_seeds"].min()),
                "max_null_seeds": int(group["max_null_seeds"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["metric", "contrast_type", "position_bin", "family", "prompt_id"]
    )


def bootstrap_position_responses(
    prompt_coefficients: pd.DataFrame,
    *,
    bins: int = DEFAULT_BINS,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    zero_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Bootstrap position response coefficients with prompts as units."""

    if prompt_coefficients.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    rows = []
    for key, group in prompt_coefficients.groupby(
        POSITION_INFERENCE_KEYS, sort=True, dropna=False
    ):
        base = dict(zip(POSITION_INFERENCE_KEYS, key if isinstance(key, tuple) else (key,)))
        values = group["response_coefficient"].astype(float).to_numpy()
        draw_indices = rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))
        draws = values[draw_indices].mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        bin_index = int(base["position_bin"])
        rows.append(
            {
                **base,
                "position_bin_lower": bin_index / bins,
                "position_bin_center": (bin_index + 0.5) / bins,
                "position_bin_upper": (bin_index + 1) / bins,
                "response_power": int(group["response_power"].iloc[0]),
                "n_alpha_magnitudes": int(group["n_alpha_magnitudes"].min()),
                "n_prompts": int(len(values)),
                "n_families": int(group["family"].nunique()),
                "n_tokens": int(group["n_tokens"].sum()),
                "mean_response_coefficient": float(values.mean()),
                "median_response_coefficient": float(np.median(values)),
                "bootstrap_ci_lower": float(lower),
                "bootstrap_ci_upper": float(upper),
                "positive_prompt_fraction": float(np.mean(values > zero_tolerance)),
                "negative_prompt_fraction": float(np.mean(values < -zero_tolerance)),
                "mean_relative_response_rmse": float(
                    group["relative_response_rmse"].astype(float).mean()
                ),
                "min_null_seeds": int(group["min_null_seeds"].min()),
                "max_null_seeds": int(group["max_null_seeds"].max()),
                "bootstrap_samples": int(n_bootstrap),
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "contrast_type", "position_bin"])


def position_coverage(
    rows: pd.DataFrame,
    gaps: pd.DataFrame,
    *,
    bins: int,
    component: str,
    token_selector: str = "all_interior",
) -> pd.DataFrame:
    """Count candidate and active prompt/token units without zero imputation."""

    candidates = rows[
        (rows["component"] == component) & (rows["token_selector"] == token_selector)
    ].copy()
    candidates = _attach_positions(candidates, bins=bins)
    candidates["prompt_key"] = candidates["family"].astype(str) + "::" + candidates["prompt_id"].astype(str)
    candidate_units = candidates.drop_duplicates(
        [*TOPOLOGY_KEYS, "family", "prompt_id", "node_index", "token_index"]
    )
    candidate = (
        candidate_units.groupby([*TOPOLOGY_KEYS, "position_bin"], as_index=False, dropna=False)
        .agg(
            candidate_prompts=("prompt_key", "nunique"),
            candidate_tokens=("token_index", "size"),
        )
    )

    active_units = gaps.drop_duplicates(
        [*TOPOLOGY_KEYS, "family", "prompt_id", "node_index", "token_index"]
    ).copy()
    active_units["prompt_key"] = active_units["family"].astype(str) + "::" + active_units["prompt_id"].astype(str)
    active = (
        active_units.groupby([*TOPOLOGY_KEYS, "position_bin"], as_index=False, dropna=False)
        .agg(active_prompts=("prompt_key", "nunique"), active_tokens=("token_index", "size"))
    )
    coverage = candidate.merge(
        active,
        on=[*TOPOLOGY_KEYS, "position_bin"],
        how="left",
        validate="one_to_one",
    )
    coverage[["active_prompts", "active_tokens"]] = coverage[
        ["active_prompts", "active_tokens"]
    ].fillna(0).astype(int)
    coverage["active_prompt_fraction"] = coverage["active_prompts"] / coverage["candidate_prompts"]
    coverage["active_token_fraction"] = coverage["active_tokens"] / coverage["candidate_tokens"]
    coverage["position_bin_lower"] = coverage["position_bin"] / bins
    coverage["position_bin_center"] = (coverage["position_bin"] + 0.5) / bins
    coverage["position_bin_upper"] = (coverage["position_bin"] + 1) / bins
    return coverage.sort_values("position_bin")


def family_position_responses(prompt_coefficients: pd.DataFrame, *, bins: int) -> pd.DataFrame:
    group_columns = [
        *POSITION_INFERENCE_KEYS,
        "family",
    ]
    family = (
        prompt_coefficients.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            mean_response_coefficient=("response_coefficient", "mean"),
            median_response_coefficient=("response_coefficient", "median"),
            n_prompts=("prompt_id", "nunique"),
            n_tokens=("n_tokens", "sum"),
            mean_relative_response_rmse=("relative_response_rmse", "mean"),
        )
    )
    family["position_bin_center"] = (family["position_bin"] + 0.5) / bins
    return family.sort_values(["metric", "contrast_type", "family", "position_bin"])


def prompt_position_phases(
    prompt_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse prompt coefficients into early, middle, and late thirds."""

    data = prompt_coefficients.copy()
    data["position_phase"] = np.select(
        [
            data["position_bin_center"] < (1.0 / 3.0),
            data["position_bin_center"] < (2.0 / 3.0),
        ],
        ["early", "middle"],
        default="late",
    )
    group_columns = [*PROMPT_PROFILE_KEYS, "position_phase"]
    phases = (
        data.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            response_coefficient=("response_coefficient", "mean"),
            n_bins=("position_bin", "nunique"),
            n_tokens=("n_tokens", "sum"),
        )
    )

    pair_keys = PROMPT_PROFILE_KEYS
    early = phases[phases["position_phase"] == "early"][
        [*pair_keys, "response_coefficient", "n_bins", "n_tokens"]
    ].rename(
        columns={
            "response_coefficient": "response_coefficient_early",
            "n_bins": "n_bins_early",
            "n_tokens": "n_tokens_early",
        }
    )
    late = phases[phases["position_phase"] == "late"][
        [*pair_keys, "response_coefficient", "n_bins", "n_tokens"]
    ].rename(
        columns={
            "response_coefficient": "response_coefficient_late",
            "n_bins": "n_bins_late",
            "n_tokens": "n_tokens_late",
        }
    )
    paired = early.merge(late, on=pair_keys, how="inner", validate="one_to_one")
    if not paired.empty:
        paired["position_phase"] = "early_minus_late"
        paired["response_coefficient"] = (
            paired["response_coefficient_early"] - paired["response_coefficient_late"]
        )
        paired["n_bins"] = paired["n_bins_early"] + paired["n_bins_late"]
        paired["n_tokens"] = paired["n_tokens_early"] + paired["n_tokens_late"]
        phases = pd.concat(
            [phases, paired[[*group_columns, "response_coefficient", "n_bins", "n_tokens"]]],
            ignore_index=True,
            sort=False,
        )
    return phases.sort_values(["metric", "contrast_type", "position_phase", "family", "prompt_id"])


def bootstrap_position_phases(
    prompt_phases: pd.DataFrame,
    *,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows = []
    for key, group in prompt_phases.groupby(PHASE_INFERENCE_KEYS, sort=True, dropna=False):
        base = dict(zip(PHASE_INFERENCE_KEYS, key if isinstance(key, tuple) else (key,)))
        values = group["response_coefficient"].astype(float).to_numpy()
        draw_indices = rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))
        draws = values[draw_indices].mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                **base,
                "n_prompts": int(len(values)),
                "n_families": int(group["family"].nunique()),
                "n_tokens": int(group["n_tokens"].sum()),
                "mean_response_coefficient": float(values.mean()),
                "median_response_coefficient": float(np.median(values)),
                "bootstrap_ci_lower": float(lower),
                "bootstrap_ci_upper": float(upper),
                "positive_prompt_fraction": float(np.mean(values > 1e-12)),
                "bootstrap_samples": int(n_bootstrap),
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "contrast_type", "position_phase"])


def fit_prompt_position_trends(prompt_coefficients: pd.DataFrame) -> pd.DataFrame:
    """Fit one exploratory linear position trend per prompt profile."""

    rows = []
    for key, group in prompt_coefficients.groupby(PROMPT_PROFILE_KEYS, sort=True, dropna=False):
        base = dict(zip(PROMPT_PROFILE_KEYS, key if isinstance(key, tuple) else (key,)))
        x = group["position_bin_center"].astype(float).to_numpy()
        y = group["response_coefficient"].astype(float).to_numpy()
        if len(np.unique(x)) < 2:
            raise ValueError(f"position trend requires at least two bins: {base}")
        design = np.column_stack([np.ones(len(x)), x])
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - (intercept + slope * x)
        rows.append(
            {
                **base,
                "position_intercept": float(intercept),
                "position_slope": float(slope),
                "position_trend_rmse": float(np.sqrt(np.mean(residual**2))),
                "n_bins": int(len(np.unique(x))),
                "n_tokens": int(group["n_tokens"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "contrast_type", "family", "prompt_id"])


def bootstrap_position_trends(
    prompt_trends: pd.DataFrame,
    *,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows = []
    for key, group in prompt_trends.groupby(TREND_INFERENCE_KEYS, sort=True, dropna=False):
        base = dict(zip(TREND_INFERENCE_KEYS, key if isinstance(key, tuple) else (key,)))
        values = group["position_slope"].astype(float).to_numpy()
        draw_indices = rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))
        draws = values[draw_indices].mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                **base,
                "n_prompts": int(len(values)),
                "n_families": int(group["family"].nunique()),
                "mean_position_slope": float(values.mean()),
                "median_position_slope": float(np.median(values)),
                "bootstrap_ci_lower": float(lower),
                "bootstrap_ci_upper": float(upper),
                "negative_prompt_fraction": float(np.mean(values < -1e-12)),
                "mean_position_trend_rmse": float(group["position_trend_rmse"].mean()),
                "bootstrap_samples": int(n_bootstrap),
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(rows).sort_values(["metric", "contrast_type"])


def plot_position_profile(inference: pd.DataFrame, *, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), sharex=True, constrained_layout=True)
    for row_index, contrast_type in enumerate(("odd", "even")):
        for column_index, (metric, title) in enumerate(METRICS.items()):
            ax = axes[row_index, column_index]
            panel = inference[
                (inference["metric"] == metric)
                & (inference["contrast_type"] == contrast_type)
            ].sort_values("position_bin")
            x = panel["position_bin_center"].astype(float).to_numpy()
            center = panel["mean_response_coefficient"].astype(float).to_numpy()
            lower = panel["bootstrap_ci_lower"].astype(float).to_numpy()
            upper = panel["bootstrap_ci_upper"].astype(float).to_numpy()
            ax.fill_between(x, lower, upper, color=PRIMARY_COLOR, alpha=0.16, linewidth=0)
            ax.plot(x, center, color=PRIMARY_COLOR, marker="o", markersize=4.5, linewidth=1.8)
            excludes_zero = (lower > 0.0) | (upper < 0.0)
            ax.scatter(
                x[excludes_zero],
                center[excludes_zero],
                color=PRIMARY_COLOR,
                edgecolor="white",
                linewidth=0.8,
                s=42,
                zorder=3,
            )
            ax.axhline(0.0, color="#4a5568", linewidth=0.9)
            ax.axvline(0.5, color="#a0aec0", linewidth=0.8, linestyle="--")
            ax.grid(axis="y", color="#d8dee6", alpha=0.65, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(title)
                ax.set_ylabel("odd slope / |alpha|")
            else:
                ax.set_ylabel("even curvature / |alpha|^2")
                ax.set_xlabel("normalized token position")
            ticks = panel["position_bin_center"].astype(float).to_numpy()[::2]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{value:.2f}" for value in ticks])

    first = inference.iloc[0]
    fig.suptitle(
        "HLTD signed coexact position profile\n"
        f"L{int(first['layer'])}, k={int(first['k'])}, complex={first['complex_mode']}, "
        f"Betti-1 target={float(first['betti_1_fraction_target']):g}; "
        "line is prompt mean, band is 95% prompt-bootstrap CI",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_family_position_profile(family: pd.DataFrame, *, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True, constrained_layout=True)
    for ax, (metric, title) in zip(axes, METRICS.items()):
        panel = family[(family["metric"] == metric) & (family["contrast_type"] == "odd")]
        for family_name, group in panel.groupby("family", sort=True):
            style = FAMILY_STYLES.get(
                str(family_name),
                {"color": "#4a5568", "marker": "o", "linestyle": "-"},
            )
            group = group.sort_values("position_bin")
            ax.plot(
                group["position_bin_center"],
                group["mean_response_coefficient"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=4,
                linewidth=1.5,
                label=str(family_name),
            )
        ax.axhline(0.0, color="#4a5568", linewidth=0.9)
        ax.axvline(0.5, color="#a0aec0", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("normalized token position")
        ax.set_ylabel("descriptive family odd slope")
        ax.grid(axis="y", color="#d8dee6", alpha=0.65, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(
        "HLTD signed coexact position profile by prompt family\n"
        "Family curves are descriptive; prompt-bootstrap inference is pooled across families",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_position_coverage(coverage: pd.DataFrame, *, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.2), constrained_layout=True)
    x = coverage["position_bin_center"].astype(float).to_numpy()
    ax.plot(
        x,
        coverage["active_prompt_fraction"],
        color=PRIMARY_COLOR,
        marker="o",
        linewidth=1.8,
        label="active prompts / candidate prompts",
    )
    ax.plot(
        x,
        coverage["active_token_fraction"],
        color="#4a5568",
        marker="s",
        linestyle="--",
        linewidth=1.5,
        label="active tokens / candidate tokens",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("normalized token position")
    ax.set_ylabel("coexact direction coverage")
    ax.axvline(0.5, color="#a0aec0", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", color="#d8dee6", alpha=0.65, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("Active local coexact coverage (inactive nodes are omitted, not zero-imputed)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def write_report(
    inference: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    output_path: Path,
    phase_inference: Optional[pd.DataFrame] = None,
    trend_inference: Optional[pd.DataFrame] = None,
) -> None:
    first = inference.iloc[0]
    odd = inference[inference["contrast_type"] == "odd"].copy()
    even = inference[inference["contrast_type"] == "even"].copy()
    odd_excludes = odd[(odd["bootstrap_ci_lower"] > 0.0) | (odd["bootstrap_ci_upper"] < 0.0)]
    even_excludes = even[(even["bootstrap_ci_lower"] > 0.0) | (even["bootstrap_ci_upper"] < 0.0)]
    lines = [
        "# HLTD Signed All-Interior Position Gate",
        "",
        "## Contract",
        "",
        f"- layer: L{int(first['layer'])}",
        f"- k: {int(first['k'])}",
        f"- complex: {first['complex_mode']}",
        f"- Betti-1 target: {float(first['betti_1_fraction_target']):g}",
        f"- position bins: {int(coverage['position_bin'].nunique())}",
        f"- prompt bootstrap: {int(first['bootstrap_samples'])} draws, seed {int(first['bootstrap_seed'])}",
        "- contrast: active coexact minus its seed-matched active random tangent",
        "- aggregation: null-seed mean, exact-token signed pair, prompt-local bin mean, prompt response fit, prompt bootstrap",
        "- odd response: fit through zero against |alpha|",
        "- even response: fit through zero against |alpha|^2",
        "- inactive local coexact nodes: omitted from effects, retained in coverage denominators, never zero-imputed",
        "",
        "## Odd Position Profile",
        "",
        "| metric | bin | position range | prompts | active tokens | mean slope | 95% prompt CI | positive prompts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in odd.iterrows():
        lines.append(
            f"| {METRICS.get(str(row['metric']), row['metric'])} | {int(row['position_bin'])} | "
            f"{float(row['position_bin_lower']):.2f}-{float(row['position_bin_upper']):.2f} | "
            f"{int(row['n_prompts'])} | {int(row['n_tokens'])} | "
            f"{float(row['mean_response_coefficient']):+.5f} | "
            f"[{float(row['bootstrap_ci_lower']):+.5f}, {float(row['bootstrap_ci_upper']):+.5f}] | "
            f"{int(round(float(row['positive_prompt_fraction']) * int(row['n_prompts'])))}"
            f"/{int(row['n_prompts'])} |"
        )
    lines.extend(
        [
            "",
            "## Interval Screen",
            "",
            f"- odd position intervals excluding zero: {len(odd_excludes)}/{len(odd)}",
            f"- even position intervals excluding zero: {len(even_excludes)}/{len(even)}",
            "",
            "These per-bin intervals are descriptive gate outcomes and are not corrected",
            "for the position scan. A broad contiguous phase is stronger evidence than an",
            "isolated bin; peak locations are descriptive.",
        ]
    )
    if phase_inference is not None and not phase_inference.empty:
        lines.extend(
            [
                "",
                "## Broad Position Phases",
                "",
                "The sequence is split into normalized early, middle, and late thirds.",
                "The early-minus-late row is paired within prompt before bootstrap.",
                "These equal-third phases were added after inspecting the fixed 12-bin",
                "profile and are exploratory summaries, not pre-registered endpoints.",
                "",
                "| parity | metric | phase | prompts | mean coefficient | 95% prompt CI |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in phase_inference.iterrows():
            lines.append(
                f"| {row['contrast_type']} | {METRICS.get(str(row['metric']), row['metric'])} | "
                f"{row['position_phase']} | {int(row['n_prompts'])} | "
                f"{float(row['mean_response_coefficient']):+.5f} | "
                f"[{float(row['bootstrap_ci_lower']):+.5f}, {float(row['bootstrap_ci_upper']):+.5f}] |"
            )
    if trend_inference is not None and not trend_inference.empty:
        lines.extend(
            [
                "",
                "## Exploratory Position Trend",
                "",
                "A linear coefficient-versus-normalized-position slope is fit within each",
                "prompt, then bootstrapped across prompts. This is a compact descriptive",
                "post-hoc summary of the profile, not a claim that the profile is truly linear.",
                "",
                "| parity | metric | prompts | mean position slope | 95% prompt CI | negative prompts |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in trend_inference.iterrows():
            lines.append(
                f"| {row['contrast_type']} | {METRICS.get(str(row['metric']), row['metric'])} | "
                f"{int(row['n_prompts'])} | {float(row['mean_position_slope']):+.5f} | "
                f"[{float(row['bootstrap_ci_lower']):+.5f}, {float(row['bootstrap_ci_upper']):+.5f}] | "
                f"{int(round(float(row['negative_prompt_fraction']) * int(row['n_prompts'])))}"
                f"/{int(row['n_prompts'])} |"
            )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| bin | candidate prompts | active prompts | prompt coverage | candidate tokens | active tokens | token coverage |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in coverage.iterrows():
        lines.append(
            f"| {int(row['position_bin'])} | {int(row['candidate_prompts'])} | "
            f"{int(row['active_prompts'])} | {float(row['active_prompt_fraction']):.3f} | "
            f"{int(row['candidate_tokens'])} | {int(row['active_tokens'])} | "
            f"{float(row['active_token_fraction']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Conservative Read",
            "",
            "This gate localizes an immediate direction-dependent next-token response.",
            "It does not establish long-horizon semantic traversal, preserved fluency,",
            "or a global harmonic concept ring. Family curves are supplementary because",
            "each family contains only a small number of prompts.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_all(
    *,
    summary_path: Path,
    output_root: Path,
    bins: int = DEFAULT_BINS,
    component: str = "coexact",
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    expected_prompts: Optional[int] = None,
    expected_null_seeds: Optional[int] = None,
    expected_alpha_magnitudes: Optional[int] = None,
) -> None:
    rows = pd.read_csv(summary_path)
    validate_expected_design(
        rows,
        component=component,
        token_selector="all_interior",
        expected_prompts=expected_prompts,
        expected_null_seeds=expected_null_seeds,
        expected_alpha_magnitudes=expected_alpha_magnitudes,
    )
    gaps = seed_matched_component_gaps(rows, bins=bins, component=component)
    seed_rows = collapse_seed_gaps(gaps)
    signed_rows = signed_token_contrasts(seed_rows)
    prompt_bin_rows = collapse_prompt_bin_contrasts(signed_rows)
    prompt_coefficients = fit_prompt_bin_response_coefficients(prompt_bin_rows, bins=bins)
    inference = bootstrap_position_responses(
        prompt_coefficients,
        bins=bins,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    coverage = position_coverage(rows, gaps, bins=bins, component=component)
    family = family_position_responses(prompt_coefficients, bins=bins)
    prompt_phases = prompt_position_phases(prompt_coefficients)
    phase_inference = bootstrap_position_phases(
        prompt_phases,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    prompt_trends = fit_prompt_position_trends(prompt_coefficients)
    trend_inference = bootstrap_position_trends(
        prompt_trends,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    inference = inference.merge(
        coverage,
        on=[*TOPOLOGY_KEYS, "position_bin", "position_bin_lower", "position_bin_center", "position_bin_upper"],
        how="left",
        validate="many_to_one",
    )

    output_root.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(output_root / "summary_seed_matched_position_gaps.csv", index=False)
    seed_rows.to_csv(output_root / "summary_seed_collapsed_position_gaps.csv", index=False)
    signed_rows.to_csv(output_root / "summary_token_signed_contrasts.csv", index=False)
    prompt_bin_rows.to_csv(output_root / "summary_prompt_bin_signed_contrasts.csv", index=False)
    prompt_coefficients.to_csv(
        output_root / "summary_prompt_bin_response_coefficients.csv", index=False
    )
    inference.to_csv(output_root / "summary_position_response_bootstrap.csv", index=False)
    family.to_csv(output_root / "summary_family_position_response.csv", index=False)
    coverage.to_csv(output_root / "summary_position_coverage.csv", index=False)
    prompt_phases.to_csv(output_root / "summary_prompt_position_phases.csv", index=False)
    phase_inference.to_csv(output_root / "summary_position_phase_bootstrap.csv", index=False)
    prompt_trends.to_csv(output_root / "summary_prompt_position_trends.csv", index=False)
    trend_inference.to_csv(output_root / "summary_position_trend_bootstrap.csv", index=False)

    plot_position_profile(
        inference,
        output_path=output_root / "plots" / "signed_coexact_position_profile.png",
    )
    plot_family_position_profile(
        family,
        output_path=output_root / "plots" / "signed_coexact_family_position_profile.png",
    )
    plot_position_coverage(
        coverage,
        output_path=output_root / "plots" / "signed_coexact_position_coverage.png",
    )
    write_report(
        inference,
        coverage,
        output_path=output_root / "summary_signed_position_report.md",
        phase_inference=phase_inference,
        trend_inference=trend_inference,
    )
    print(f"saved signed position plot: {output_root / 'plots' / 'signed_coexact_position_profile.png'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Raw all-interior steering summary.csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--component", default="coexact")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--expected-prompts", type=int, default=None)
    parser.add_argument("--expected-null-seeds", type=int, default=None)
    parser.add_argument("--expected-alpha-magnitudes", type=int, default=None)
    args = parser.parse_args(argv)
    render_all(
        summary_path=Path(args.summary),
        output_root=Path(args.output_root),
        bins=args.bins,
        component=args.component,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        expected_prompts=args.expected_prompts,
        expected_null_seeds=args.expected_null_seeds,
        expected_alpha_magnitudes=args.expected_alpha_magnitudes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
