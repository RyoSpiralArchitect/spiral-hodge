from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import plan_hltd_branch_band_runs as planner


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestPlanHLTDBranchBandRuns(unittest.TestCase):
    def test_parse_layer_spans_handles_ranges_and_controls(self) -> None:
        self.assertEqual(planner.parse_layer_spans("L4-L5 L7"), [4, 5, 7])
        self.assertEqual(planner.parse_layer_spans("L8 L6-L7"), [6, 7, 8])
        self.assertEqual(planner.parse_layer_spans("control"), [])
        self.assertEqual(planner.parse_layer_spans(""), [])

    def test_build_plan_rows_writes_closed_loop_followup_commands(self) -> None:
        rows = planner.build_plan_rows(
            candidate_rows=[
                {
                    "family": "identity_stress",
                    "component": "coexact",
                    "candidate_label": "causal_exception_band",
                    "priority_score": "0.54",
                    "recommended_layers": "L4-L5 L7",
                    "closed_loop_branch_specific_gate_rate_mean": "0.4",
                    "closed_loop_target_margin_delta_minus_random_mean": "0.15",
                },
                {
                    "family": "literal_stable",
                    "component": "presence",
                    "candidate_label": "deprioritize_or_control",
                    "priority_score": "0.15",
                    "recommended_layers": "control",
                },
            ],
            output_root=Path("planned_runs"),
            model_path="/models/gpt2",
            suite="data/hltd_prompt_suite.jsonl",
            target_set_file="data/hltd_semantic_targets.json",
            k_values=[12, 16],
            alphas=[0.8],
            seeds=[0, 1],
            pca_components=32,
            generate_steps=4,
            max_rows=4,
            include_labels=["causal_exception_band"],
            exclude_components=[],
            min_priority=0.25,
            device="mps",
            torch_dtype="auto",
        )

        [row] = rows
        self.assertEqual(row["family"], "identity_stress")
        self.assertEqual(row["component"], "coexact")
        self.assertEqual(row["layers"], "4 5 7")
        self.assertIn("--layers 4 5 7", row["run_command"])
        self.assertIn("--k 12 16", row["run_command"])
        self.assertIn("--steering-components coexact random_tangent", row["run_command"])
        self.assertIn("--families identity_stress", row["run_command"])
        self.assertIn("scripts/summarize_hltd_closed_loop.py", row["summarize_command"])
        self.assertIn("scripts/plot_hltd_closed_loop.py", row["plot_command"])

    def test_main_writes_plan_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_csv = tmp_path / "candidates.csv"
            write_csv(
                candidate_csv,
                [
                    {
                        "family": "ontology_collapse",
                        "component": "coexact",
                        "candidate_label": "structural_band_ready",
                        "priority_score": 0.77,
                        "recommended_layers": "L4-L8",
                        "closed_loop_branch_specific_gate_rate_mean": 0.13,
                        "closed_loop_target_margin_delta_minus_random_mean": -0.21,
                    }
                ],
            )

            rc = planner.main(
                [
                    "--candidate-csv",
                    str(candidate_csv),
                    "--output-root",
                    str(tmp_path / "plan"),
                    "--run-output-root",
                    str(tmp_path / "runs"),
                    "--model-path",
                    "/models/gpt2",
                    "--top-n",
                    "1",
                    "--seeds",
                    "0",
                    "--device",
                    "cpu",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue((tmp_path / "plan" / "branch_band_run_plan.csv").exists())
            script_path = tmp_path / "plan" / "run_branch_band_plan.sh"
            self.assertTrue(script_path.exists())
            self.assertTrue((tmp_path / "plan" / "branch_band_run_plan.md").exists())
            script_text = script_path.read_text(encoding="utf-8")
            self.assertIn("[skip run]", script_text)
            self.assertIn("[skip summary]", script_text)
            self.assertIn("[skip plot]", script_text)
            self.assertIn("scripts/summarize_hltd_branch_band_runs.py", script_text)
            self.assertIn("scripts/plot_hltd_branch_band_results.py", script_text)
            rank_script = tmp_path / "plan" / "rank_scripts" / "run_rank_01__ontology_collapse__coexact.sh"
            self.assertTrue(rank_script.exists())
            rank_text = rank_script.read_text(encoding="utf-8")
            self.assertIn("[1/1] ontology_collapse coexact L4-L8", rank_text)
            self.assertIn("scripts/summarize_hltd_branch_band_runs.py", rank_text)
            self.assertIn("scripts/plot_hltd_branch_band_results.py", rank_text)
            plan_rows = list(csv.DictReader((tmp_path / "plan" / "branch_band_run_plan.csv").open(newline="", encoding="utf-8")))
            self.assertEqual(len(plan_rows), 1)
            self.assertEqual(plan_rows[0]["layers"], "4 5 6 7 8")


if __name__ == "__main__":
    unittest.main()
