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
    group_columns = [
        *INFERENCE_KEYS,
        "family",
        "prompt_id",
        "baseline_component",
    ]
    prompt_rows = (
        gaps.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            gap=("gap", "mean"),
            component_value=("component_value", "mean"),
            baseline_value=("baseline_value", "mean"),
            n_seed_position_pairs=("gap", "size"),
            n_null_seeds=("seed", "nunique"),
            n_positions=("token_index", "nunique"),
        )
        .sort_values(["metric", "component", "alpha", "family", "prompt_id"])
    )
    return prompt_rows


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


def render_all(
    *,
    summary_path: Path,
    output_root: Path,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> None:
    rows = pd.read_csv(summary_path)
    gaps = matched_component_gaps(rows)
    if gaps.empty:
        raise ValueError("No active exact/coexact/harmonic rows could be paired with random_tangent.")
    prompt_rows = collapse_prompt_gaps(gaps)
    inference = bootstrap_prompt_gaps(
        prompt_rows,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Raw steering suite summary.csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    render_all(
        summary_path=Path(args.summary),
        output_root=Path(args.output_root),
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
