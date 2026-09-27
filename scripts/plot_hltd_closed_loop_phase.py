#!/usr/bin/env python3
"""Export phase composition, matched-arm gaps, and per-source step maps."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, PercentFormatter

PHASES = ["early", "middle", "late"]
PHASE_COLORS = ["#2878b5", "#e6b84b", "#cc7c9c"]
HATCHES = ["", "//", ".."]
INK = "#282828"
FOCUS_SOURCES = {
    "spiral_out_hltd_closed_loop_layer_pilot_l5_l8_a08": "Layer pilot",
    "spiral_out_hltd_closed_loop_branch_panel_l7_k16_a08_steps8": "8-step panel",
    "spiral_out_hltd_closed_loop_identity02_branch_surface_l4_5_7_k12_16_24_a04_08_12_s0_19": "Identity 02",
    "spiral_out_hltd_branch_band_runs/07__identity_stress__coexact__l4_5_7__k16__a0p8__s0_4": "Identity band",
}
CONDITION = ["source_id", "family", "layer", "k", "target_set", "component", "alpha"]
COMPONENT_LABELS = {"random_tangent": "random", "negative_coexact": "-coexact",
                    "coexact_minus_presence": "coex - pres", "presence_plus_coexact": "pres + coex"}
DISTANCE_MAP = LinearSegmentedColormap.from_list("distance", ["#ffffff", "#2878b5"])
RESPONSE_MAP = LinearSegmentedColormap.from_list("response", ["#2878b5", "#ffffff", "#cc7c9c"])


def occupancy_table(summary: pd.DataFrame) -> pd.DataFrame:
    phases = summary[summary["phase"].isin(PHASES)]
    table = phases.pivot(index=CONDITION, columns="phase", values="occupancy").reset_index()
    counts = summary[summary["phase"] == "all"][[*CONDITION, "n_prompts"]]
    return table.merge(counts, on=CONDITION, validate="one_to_one").sort_values(CONDITION)


def label(row: pd.Series, *, overview: bool = False) -> str:
    component = COMPONENT_LABELS.get(row["component"], row["component"])
    if overview:
        source = FOCUS_SOURCES.get(row["source_id"], row["source_id"].removeprefix("spiral_out_hltd_closed_loop_"))
        return f"{source}  L{int(row['layer'])}  {component}  (n={int(row['n_prompts'])})"
    family = row["family"].replace("_stress", "").replace("_collapse", "")
    return f"{family}  L{int(row['layer'])}/k{int(row['k'])}/a{row['alpha']:g}  {component}"


def stacked(ax: plt.Axes, table: pd.DataFrame, labels: list[str]) -> None:
    left = np.zeros(len(table))
    for phase, color, hatch in zip(PHASES, PHASE_COLORS, HATCHES):
        widths = table[phase].to_numpy()
        ax.barh(np.arange(len(table)), widths, left=left, color=color, hatch=hatch,
                edgecolor=INK, linewidth=0.35, height=0.72, label=phase)
        if len(table) <= 26:
            for i, (start, width) in enumerate(zip(left, widths)):
                if width >= 0.15:
                    ax.text(start + width / 2, i, f"{width:.0%}", ha="center", va="center", fontsize=8,
                            color="white" if phase == "early" else INK)
        left += widths
    ax.set(yticks=np.arange(len(table)), yticklabels=labels, xlim=(0, 1), ylim=(len(table) - .5, -.5))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Share of generated steps (prompt balanced)")
    ax.spines[["top", "right"]].set_visible(False)


def overview_plots(table: pd.DataFrame, gaps: pd.DataFrame, output: Path) -> list[Path]:
    paths = []
    selected = table[table["source_id"].isin(FOCUS_SOURCES) & table["k"].eq(16)
                     & np.isclose(table["alpha"], .8) & table["component"].isin(["coexact", "random_tangent"])]
    overview = not selected.empty
    if selected.empty:
        selected = table[table["component"] != "baseline"].head(24)
    fig, ax = plt.subplots(figsize=(13, max(4.5, .38 * len(selected) + 2)))
    stacked(ax, selected, [label(row, overview=True) for _, row in selected.iterrows()])
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    fig.suptitle("Retrieved original-prompt phase", x=.02, ha="left", fontsize=16)
    subtitle = ("Four previously inspected runs; k=16, alpha=0.8. " if overview else "First 24 condition/arm rows. ")
    fig.text(.02, .942, subtitle + "n = prompts; seeds averaged within prompt.",
             fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(output / "phase_occupancy.png", dpi=160)
    paths.append(output / "phase_occupancy.png")
    plt.close(fig)

    selected = gaps[gaps["source_id"].isin(FOCUS_SOURCES) & gaps["k"].eq(16)
                    & np.isclose(gaps["alpha"], .8) & gaps["component"].eq("coexact")]
    if selected.empty:
        selected = gaps[gaps["random_matched"] > 0].head(24)
    if selected.empty:
        return paths
    metrics = [("gap_late_occupancy", "Late occupancy gap", 100, "Percentage points"),
               ("gap_nearest_distance", "Nearest-distance gap", 1, "Raw PCA chart distance"),
               ("gap_target_margin_delta", "Target-margin gap", 1, "Log-probability margin")]
    fig, axes = plt.subplots(1, 3, figsize=(13, max(4.5, .45 * len(selected) + 2.5)), sharey=True)
    for ax, (metric, title, scale, unit) in zip(axes, metrics):
        values = selected[metric].to_numpy() * scale
        ax.axvline(0, color=INK, linewidth=.8)
        ax.scatter(values, np.arange(len(selected)), color=PHASE_COLORS[0], s=35)
        ax.set(title=title, xlabel=unit, ylim=(len(selected) - .5, -.5))
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y", color="#eeeeee", linewidth=.5)
    axes[0].set(yticks=np.arange(len(selected)), yticklabels=[label(row, overview=True)
                                                          for _, row in selected.iterrows()])
    fig.suptitle("Coexact minus matched random tangent", x=.02, ha="left", fontsize=16)
    fig.text(.02, .925, "Matched generated steps; later prefixes may differ. Descriptive means; no confidence intervals.",
             fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(output / "phase_random_gaps.png", dpi=160)
    paths.append(output / "phase_random_gaps.png")
    plt.close(fig)
    return paths


def render_plots(summary: pd.DataFrame, steps: pd.DataFrame, gaps: pd.DataFrame, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK, "axes.titleweight": "normal", "figure.facecolor": "white"})
    table = occupancy_table(summary)
    paths = overview_plots(table, gaps, output)
    source_dir = output / "by_source"
    source_dir.mkdir(exist_ok=True)
    for source_index, (source, group) in enumerate(table.groupby("source_id", sort=True), start=1):
        distance = steps[steps["source_id"] == source].pivot(index=CONDITION, columns="step", values="nearest_distance")
        response = summary[(summary["source_id"] == source) & summary["phase"].isin(PHASES)].pivot(
            index=CONDITION, columns="phase", values="target_margin_delta")
        finite = response.to_numpy()[np.isfinite(response.to_numpy())]
        bound = max(float(np.max(np.abs(finite))) if len(finite) else 0, 1e-8)
        for page in range(math.ceil(len(group) / 24)):
            block = group.iloc[page * 24:(page + 1) * 24]
            indices = pd.MultiIndex.from_frame(block[CONDITION])
            dist = distance.reindex(indices)
            margins = response.reindex(indices)[PHASES]
            fig, axes = plt.subplots(1, 3, figsize=(17, max(4.5, .37 * len(block) + 2.8)),
                                     gridspec_kw={"width_ratios": [1.35, 1, .75]})
            stacked(axes[0], block, [label(row) for _, row in block.iterrows()])
            axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
            cmap = DISTANCE_MAP.copy()
            cmap.set_bad("#ededed")
            im = axes[1].imshow(np.ma.masked_invalid(dist.to_numpy()), aspect="auto", cmap=cmap,
                                vmin=0, vmax=max(float(np.nanmax(distance.to_numpy())), 1e-8))
            axes[1].set(title="Nearest distance by generated step", yticks=[],
                        xticks=range(len(dist.columns)), xticklabels=[str(int(i)) for i in dist.columns], xlabel="Generation step")
            fig.colorbar(im, ax=axes[1], fraction=.04, pad=.03, label="Raw PCA distance")
            cmap = RESPONSE_MAP.copy()
            cmap.set_bad("#ededed")
            im = axes[2].imshow(np.ma.masked_invalid(margins.to_numpy()), aspect="auto", cmap=cmap,
                                norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
            axes[2].set(title="Active target-margin change", yticks=[], xticks=range(3),
                        xticklabels=PHASES, xlabel="Retrieved prompt phase")
            fig.colorbar(im, ax=axes[2], fraction=.06, pad=.03, label="Log-probability margin")
            short = source.removeprefix("spiral_out_hltd_closed_loop_")
            fig.suptitle(f"Closed-loop phase audit: {short}", x=.02, ha="left", fontsize=11)
            fig.text(.02, .925, f"Page {page + 1}; prompt-balanced means. Gray = no estimate; "
                     "phase-conditioned response is descriptive. Full coverage in CSV.", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, .89))
            path = source_dir / f"source_{source_index:02d}_page_{page + 1:02d}.png"
            fig.savefig(path, dpi=150)
            paths.append(path)
            plt.close(fig)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    frames = [pd.read_csv(args.summary_root / name, keep_default_na=True)
              for name in ["phase_summary.csv", "phase_step_summary.csv", "phase_random_gaps.csv"]]
    for frame in frames:
        frame["target_set"] = frame["target_set"].fillna("")
    render_plots(*frames, args.output_dir or args.summary_root / "plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
