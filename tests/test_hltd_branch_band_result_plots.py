from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import plot_hltd_branch_band_results as plots


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestHLTDBranchBandResultPlots(unittest.TestCase):
    def test_build_plots_writes_pending_and_layer_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            write_csv(
                root / "branch_band_result_scoreboard.csv",
                [
                    {
                        "rank": 1,
                        "family": "identity_stress",
                        "component": "coexact",
                        "candidate_label": "causal_exception_band",
                        "result_label": "causal_support_confirmed",
                        "run_status": "complete",
                        "priority_score": 0.54,
                        "recommended_layers": "L4-L5 L7",
                        "planned_closed_loop_gate": 0.4,
                        "result_branch_specific_gate_rate": 0.66,
                        "result_target_margin_delta_minus_random_mean": 0.1,
                        "matched_random_rows": 3,
                    },
                    {
                        "rank": 2,
                        "family": "ontology_collapse",
                        "component": "negative_coexact",
                        "candidate_label": "structural_band_ready",
                        "result_label": "missing_run",
                        "run_status": "missing_run",
                        "priority_score": 0.77,
                        "recommended_layers": "L4-L8",
                        "planned_closed_loop_gate": 0.08,
                        "result_branch_specific_gate_rate": "",
                        "result_target_margin_delta_minus_random_mean": "",
                        "matched_random_rows": 0,
                    },
                ],
            )
            write_csv(
                root / "branch_band_layer_result_summary.csv",
                [
                    {
                        "rank": 1,
                        "family": "identity_stress",
                        "component": "coexact",
                        "layer": 4,
                        "result_label": "causal_support_confirmed",
                        "branch_specific_gate_rate": 0.5,
                        "target_margin_delta_minus_random_mean": 0.2,
                        "matched_random_rows": 2,
                    },
                    {
                        "rank": 1,
                        "family": "identity_stress",
                        "component": "coexact",
                        "layer": 5,
                        "result_label": "target_advantage_without_gate",
                        "branch_specific_gate_rate": 0.0,
                        "target_margin_delta_minus_random_mean": 0.1,
                        "matched_random_rows": 1,
                    },
                ],
            )

            saved = plots.build_plots(result_root=root, output_dir=root / "plots")

            self.assertEqual(len(saved), 3)
            self.assertTrue((root / "plots" / "branch_band_result_scoreboard.png").exists())
            self.assertTrue((root / "plots" / "branch_band_layer_result_summary.png").exists())
            self.assertTrue((root / "plots" / "plot_manifest.json").exists())
            for path in saved:
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
