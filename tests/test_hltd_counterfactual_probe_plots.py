from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import plot_hltd_counterfactual_probe_surface as plots


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_fixture(run_root: Path) -> None:
    pairwise_rows = []
    component_rows = []
    for layer in [4, 5]:
        for alpha in [0.4, 0.8]:
            for component, active in [
                ("presence", 1.0),
                ("coexact", 1.0),
                ("harmonic", 0.0),
            ]:
                component_rows.append(
                    {
                        "layer": layer,
                        "k": 12,
                        "component": component,
                        "alpha": alpha,
                        "probe": "identity_stress",
                        "component_active_mean": active,
                    }
                )
                if active:
                    pairwise_rows.append(
                        {
                            "layer": layer,
                            "k": 12,
                            "component": component,
                            "alpha": alpha,
                            "probe": "identity_stress",
                            plots.PAIR_METRIC: (
                                (0.1 if component == "presence" else -0.2) * alpha
                            ),
                            plots.PAIR_WIN_RATE: 1.0 if component == "presence" else 0.0,
                        }
                    )
    write_csv(run_root / "summary_pairwise.csv", pairwise_rows)
    write_csv(run_root / "summary_component.csv", component_rows)
    write_csv(
        run_root / "probe_training_summary.csv",
        [
            {
                "layer": layer,
                "probe": "identity_stress",
                "cv_group_accuracy": 0.7 + layer * 0.01,
                "grouping": "pair_id",
                "training_token_selector": "pair_balanced_interior",
            }
            for layer in [4, 5]
        ],
    )
    (run_root / "probe_manifest.json").write_text(
        json.dumps(
            {
                "evaluation_prompt_ids": ["identity_02"],
                "split_is_disjoint": True,
            }
        ),
        encoding="utf-8",
    )


class TestHLTDCounterfactualProbePlots(unittest.TestCase):
    def test_surface_keeps_inactive_harmonic_cells_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            write_fixture(run_root)
            components = ["presence", "coexact", "harmonic"]
            pairwise, component, _training, _manifest = plots.load_probe_surface(
                run_root,
                probe="identity_stress",
                components=components,
            )

            surface = plots.build_surface_table(pairwise, component, components=components)
            summary = plots.summarize_surface(surface)

            self.assertEqual(len(surface), 12)
            harmonic = summary[summary["component"] == "harmonic"]
            self.assertEqual(harmonic["n_active_cells"].tolist(), [0, 0])
            self.assertTrue(all(math.isnan(value) for value in harmonic["label_axis_minus_random_mean"]))
            presence = summary[summary["component"] == "presence"]
            self.assertEqual(presence["n_active_cells"].tolist(), [2, 2])
            self.assertTrue(all(value > 0.0 for value in presence["label_axis_minus_random_mean"]))
            self.assertEqual(presence["null_win_rate_mean"].tolist(), [1.0, 1.0])

    def test_build_writes_surface_summary_plot_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            output_root = root / "plots"
            write_fixture(run_root)

            saved = plots.build_counterfactual_probe_plot(
                run_root=run_root,
                output_root=output_root,
                probe="identity_stress",
                components=["presence", "coexact", "harmonic"],
            )

            for value in saved.values():
                path = Path(value)
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)
            summary = pd.read_csv(saved["summary_csv"])
            harmonic = summary[summary["component"] == "harmonic"]
            self.assertTrue(harmonic["label_axis_minus_random_mean"].isna().all())


if __name__ == "__main__":
    unittest.main()
