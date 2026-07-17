from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_hltd_branch_band_runs as summarize


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestSummarizeHLTDBranchBandRuns(unittest.TestCase):
    def test_build_result_rows_joins_plan_to_executed_prompt_layer_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_root = tmp_path / "run"
            write_csv(
                run_root / "closed_loop_prompt_layer_k_summary.csv",
                [
                    {
                        "family": "identity_stress",
                        "prompt_id": "identity_01",
                        "layer": 4,
                        "k": 16,
                        "component": "coexact",
                        "alpha": 0.8,
                        "matched_random_rows": 2,
                        "branch_gate_rate": 1.0,
                        "branch_specific_gate_rate": 0.5,
                        "random_branch_gate_rate": 0.0,
                        "token_drift_rate_mean": 0.7,
                        "mean_target_margin_delta_minus_random_mean": 0.2,
                    },
                    {
                        "family": "identity_stress",
                        "prompt_id": "identity_02",
                        "layer": 5,
                        "k": 16,
                        "component": "coexact",
                        "alpha": 0.8,
                        "matched_random_rows": 1,
                        "branch_gate_rate": 1.0,
                        "branch_specific_gate_rate": 1.0,
                        "random_branch_gate_rate": 0.0,
                        "token_drift_rate_mean": 0.4,
                        "mean_target_margin_delta_minus_random_mean": -0.1,
                    },
                    {
                        "family": "identity_stress",
                        "prompt_id": "identity_01",
                        "layer": 4,
                        "k": 16,
                        "component": "random_tangent",
                        "alpha": 0.8,
                        "matched_random_rows": 0,
                        "branch_specific_gate_rate": 0.0,
                        "mean_target_margin_delta_minus_random_mean": -0.3,
                    },
                ],
            )
            plan_rows = [
                {
                    "rank": "1",
                    "family": "identity_stress",
                    "component": "coexact",
                    "candidate_label": "causal_exception_band",
                    "priority_score": "0.54",
                    "recommended_layers": "L4-L5",
                    "layers": "4 5",
                    "k_values": "16",
                    "alphas": "0.8",
                    "seeds": "0 1",
                    "closed_loop_gate": "0.4",
                    "closed_loop_target_minus_random": "0.15",
                    "output_root": str(run_root),
                },
                {
                    "rank": "2",
                    "family": "ontology_collapse",
                    "component": "coexact",
                    "candidate_label": "structural_band_ready",
                    "priority_score": "0.77",
                    "recommended_layers": "L4-L8",
                    "layers": "4 5 6 7 8",
                    "output_root": str(tmp_path / "missing"),
                },
            ]

            result_rows = summarize.build_result_rows(plan_rows)
            layer_rows = summarize.build_layer_rows(plan_rows)

            first = result_rows[0]
            self.assertEqual(first["run_status"], "complete")
            self.assertEqual(first["result_label"], "causal_support_confirmed")
            self.assertEqual(first["matched_random_rows"], 3)
            self.assertAlmostEqual(first["result_branch_specific_gate_rate"], 2.0 / 3.0)
            self.assertAlmostEqual(first["result_target_margin_delta_minus_random_mean"], 0.1)

            self.assertEqual(result_rows[1]["run_status"], "missing_run")
            self.assertEqual(result_rows[1]["result_label"], "missing_run")
            self.assertEqual(len(layer_rows), 2)
            self.assertEqual({row["layer"] for row in layer_rows}, {4, 5})

    def test_main_writes_pending_result_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_csv = tmp_path / "plan.csv"
            write_csv(
                plan_csv,
                [
                    {
                        "rank": 1,
                        "family": "metaphor_shift",
                        "component": "coexact",
                        "candidate_label": "structural_band_ready",
                        "priority_score": 0.6,
                        "recommended_layers": "L5-L8",
                        "layers": "5 6 7 8",
                        "output_root": tmp_path / "missing",
                    }
                ],
            )

            rc = summarize.main(
                [
                    "--plan-csv",
                    str(plan_csv),
                    "--output-root",
                    str(tmp_path / "results"),
                ]
            )

            self.assertEqual(rc, 0)
            result_csv = tmp_path / "results" / "branch_band_result_scoreboard.csv"
            self.assertTrue(result_csv.exists())
            rows = list(csv.DictReader(result_csv.open(newline="", encoding="utf-8")))
            self.assertEqual(rows[0]["result_label"], "missing_run")
            self.assertTrue((tmp_path / "results" / "branch_band_result_report.md").exists())


if __name__ == "__main__":
    unittest.main()
