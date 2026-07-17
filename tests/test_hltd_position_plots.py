from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import plot_hltd_position_gate as plots


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestHLTDPositionPlots(unittest.TestCase):
    def test_build_plots_writes_pngs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "summary"
            joined_rows = []
            for component in ["coexact", "presence"]:
                for k in [12, 16]:
                    for layer in [5, 7]:
                        for bin_index in [0, 1]:
                            joined_rows.append(
                                {
                                    "family": "literal_stable",
                                    "layer": layer,
                                    "k": k,
                                    "position_bin": bin_index,
                                    "position_frac_mean": 0.1 * (bin_index + 1),
                                    "component": component,
                                    "probe": "ontology_collapse",
                                    "n_steering_pairs": 2,
                                    "n_probe_pairs": 2,
                                    "next_token_delta_mean": 0.2 if component == "coexact" else -0.1,
                                    "semantic_margin_delta_mean": 0.0,
                                    "probe_label_margin_delta_mean": 0.5 if component == "presence" else 0.1,
                                    "probe_positive_prob_delta_mean": 0.0,
                                }
                            )
            peak_rows = [
                {
                    "component": "coexact",
                    "probe": "ontology_collapse",
                    "metric": "next_token_delta_mean",
                    "k": 12,
                    "peak_layer": 7,
                    "peak_position_bin": 0,
                    "peak_position_frac_mean": 0.1,
                    "peak_value_mean": 0.4,
                    "peak_value_min": 0.2,
                    "peak_value_max": 0.6,
                    "n_families": 1,
                },
                {
                    "component": "presence",
                    "probe": "ontology_collapse",
                    "metric": "next_token_delta_mean",
                    "k": 12,
                    "peak_layer": 5,
                    "peak_position_bin": 1,
                    "peak_position_frac_mean": 0.2,
                    "peak_value_mean": -0.1,
                    "peak_value_min": -0.2,
                    "peak_value_max": 0.0,
                    "n_families": 1,
                },
                {
                    "component": "coexact",
                    "probe": "ontology_collapse",
                    "metric": "probe_label_margin_delta_mean",
                    "k": 12,
                    "peak_layer": 7,
                    "peak_position_bin": 0,
                    "peak_position_frac_mean": 0.1,
                    "peak_value_mean": 0.2,
                    "peak_value_min": 0.1,
                    "peak_value_max": 0.3,
                    "n_families": 1,
                },
                {
                    "component": "presence",
                    "probe": "ontology_collapse",
                    "metric": "probe_label_margin_delta_mean",
                    "k": 12,
                    "peak_layer": 5,
                    "peak_position_bin": 1,
                    "peak_position_frac_mean": 0.2,
                    "peak_value_mean": 0.7,
                    "peak_value_min": 0.6,
                    "peak_value_max": 0.8,
                    "n_families": 1,
                },
            ]
            write_csv(root / "joined_position_summary.csv", joined_rows)
            write_csv(root / "position_cross_family_peak_summary.csv", peak_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                probe="ontology_collapse",
                components=["coexact", "presence"],
            )

            self.assertEqual(len(saved), 5)
            for path in saved:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)


if __name__ == "__main__":
    unittest.main()
