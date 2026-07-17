#!/usr/bin/env python3
"""Plot reverse-branch target specificity across HLTD exception panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_panel(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"panel label is empty: {value!r}")
    return label, Path(path)


def load_panel_rows(*, label: str, path: Path, panel_order: int, component: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    subset = table[table["component"].astype(str) == str(component)].copy()
    if subset.empty:
        raise ValueError(f"{path}: no rows for component={component!r}")
    if "source_order" not in subset.columns:
        subset["source_order"] = range(len(subset))
    subset["panel_label"] = label
    subset["panel_order"] = int(panel_order)
    subset["panel_source"] = str(path)
    return subset


def collect_panels(*, panels: Sequence[Tuple[str, Path]], component: str) -> pd.DataFrame:
    tables = [
        load_panel_rows(label=label, path=path, panel_order=idx, component=component)
        for idx, (label, path) in enumerate(panels)
    ]
    return pd.concat(tables, ignore_index=True, sort=False)


def _short_label(value: object) -> str:
    text = str(value)
    for prefix in ("identity_", "ontology_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.replace("_", "\n")


def _bar_colors(values: pd.Series) -> list[str]:
    return ["#7aa36f" if float(value) >= 0.0 else "#b75b5b" for value in values]


def plot_reverse_specificity(
    rows: pd.DataFrame,
    *,
    output_path: Path,
    component: str,
) -> None:
    panels = rows[["panel_order", "panel_label"]].drop_duplicates().sort_values("panel_order")
    fig, axes = plt.subplots(
        len(panels),
        2,
        figsize=(12.5, max(4.0, 3.15 * len(panels))),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(f"HLTD reverse exception target specificity: {component}", fontsize=14)

    for row_idx, panel in enumerate(panels.itertuples(index=False)):
        panel_rows = rows[rows["panel_label"] == panel.panel_label].copy()
        panel_rows = panel_rows.sort_values(["source_order", "target_set"]).reset_index(drop=True)
        xs = list(range(len(panel_rows)))
        labels = [_short_label(value) for value in panel_rows["target_set"]]

        ax_gate = axes[row_idx][0]
        width = 0.28
        ax_gate.bar(
            [x - width / 2 for x in xs],
            panel_rows["branch_specific_gate_rate"].astype(float),
            width=width,
            color="#516b8b",
            label="branch-specific gate",
        )
        ax_gate.bar(
            [x + width / 2 for x in xs],
            panel_rows["random_branch_gate_rate"].astype(float),
            width=width,
            color="#b8b8b8",
            label="matched random gate",
        )
        ax_gate.set_title(f"{panel.panel_label}: gate specificity")
        ax_gate.set_ylim(0.0, 1.08)
        ax_gate.set_xticks(xs, labels)
        ax_gate.grid(axis="y", alpha=0.25)
        ax_gate.legend(frameon=False, fontsize=8)

        ax_margin = axes[row_idx][1]
        target_adv = panel_rows["mean_target_margin_delta_minus_random_mean"].astype(float)
        ax_margin.bar(xs, target_adv, color=_bar_colors(target_adv), label="target-random margin")
        ax_margin.plot(
            xs,
            panel_rows["token_drift_rate_mean"].astype(float),
            marker="o",
            color="#786fa6",
            linewidth=1.6,
            label="token drift",
        )
        ax_margin.axhline(0.0, color="#333333", linewidth=0.8)
        ax_margin.set_title(f"{panel.panel_label}: semantic direction vs drift")
        ax_margin.set_xticks(xs, labels)
        y_min = min(-0.5, float(target_adv.min()) - 0.1)
        y_max = max(1.1, float(target_adv.max()) + 0.15, float(panel_rows["token_drift_rate_mean"].max()) + 0.1)
        ax_margin.set_ylim(y_min, y_max)
        ax_margin.grid(axis="y", alpha=0.25)
        ax_margin.legend(frameon=False, fontsize=8)

        for ax in (ax_gate, ax_margin):
            ax.tick_params(axis="x", labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_reverse_specificity(
    *,
    panels: Sequence[Tuple[str, Path]],
    output_root: Path,
    component: str,
    output_prefix: str = "reverse_exception_specificity",
) -> Dict[str, str]:
    rows = collect_panels(panels=panels, component=component)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{output_prefix}.csv"
    png_path = output_root / f"{output_prefix}.png"
    manifest_path = output_root / f"{output_prefix}_manifest.json"
    rows.to_csv(csv_path, index=False)
    plot_reverse_specificity(rows, output_path=png_path, component=component)
    manifest = {
        "component": component,
        "panels": [{"label": label, "path": str(path)} for label, path in panels],
        "csv": str(csv_path),
        "plot": str(png_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "plot": str(png_path), "manifest": str(manifest_path)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", action="append", required=True, help="label=target_sensitivity_csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--component", default="negative_coexact")
    parser.add_argument("--output-prefix", default="reverse_exception_specificity")
    args = parser.parse_args(argv)

    saved = build_reverse_specificity(
        panels=[parse_panel(value) for value in args.panel],
        output_root=Path(args.output_root),
        component=args.component,
        output_prefix=args.output_prefix,
    )
    for path in saved.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
