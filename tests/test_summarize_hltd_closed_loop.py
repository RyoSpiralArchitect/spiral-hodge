from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_hltd_closed_loop as summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestSummarizeHLTDClosedLoop(unittest.TestCase):
    def test_contrast_rows_match_baseline_by_prompt_layer_k_seed(self) -> None:
        rows = [
            {
                "family": "ontology_collapse",
                "prompt_id": "ontology_01",
                "layer": 7,
                "k": 16,
                "seed": 0,
                "component": "baseline",
                "alpha": 0.0,
                "generated_steps": 2,
                "baseline_token_overlap": 1.0,
                "mean_selected_base_logprob": -1.0,
                "mean_selected_logprob_gain": 0.0,
                "mean_kl_base_to_steered": 0.0,
                "mean_entropy_delta": 0.0,
                "mean_target_margin_delta": 0.0,
                "mean_nearest_distance": 0.2,
                "unique_nodes": 1,
                "top_changed_rate": 0.0,
                "generated_text": "AA",
            },
            {
                "family": "ontology_collapse",
                "prompt_id": "ontology_01",
                "layer": 7,
                "k": 16,
                "seed": 0,
                "component": "presence_plus_coexact",
                "alpha": 1.0,
                "generated_steps": 2,
                "baseline_token_overlap": 0.5,
                "mean_selected_base_logprob": -1.5,
                "mean_selected_logprob_gain": 0.2,
                "mean_kl_base_to_steered": 0.1,
                "mean_entropy_delta": -0.03,
                "mean_target_margin_delta": 0.4,
                "mean_nearest_distance": 0.3,
                "unique_nodes": 2,
                "top_changed_rate": 0.5,
                "generated_text": "AB",
            },
        ]

        contrasts = summary.contrast_rows(rows)

        self.assertEqual(len(contrasts), 1)
        row = contrasts[0]
        self.assertEqual(row["baseline_generated_text"], "AA")
        self.assertEqual(row["generated_text"], "AB")
        self.assertAlmostEqual(row["token_drift_rate"], 0.5)
        self.assertAlmostEqual(row["mean_selected_base_logprob_delta"], -0.5)

    def test_main_writes_summary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(
                root / "closed_loop_metrics.csv",
                [
                    {
                        "family": "literal_stable",
                        "prompt_id": "literal_01",
                        "layer": 7,
                        "k": 16,
                        "seed": 0,
                        "component": "baseline",
                        "alpha": 0.0,
                        "generated_steps": 2,
                        "baseline_token_overlap": 1.0,
                        "mean_selected_base_logprob": -0.5,
                        "mean_selected_logprob_gain": 0.0,
                        "mean_kl_base_to_steered": 0.0,
                        "mean_entropy_delta": 0.0,
                        "mean_target_margin_delta": 0.0,
                        "mean_nearest_distance": 0.1,
                        "unique_nodes": 1,
                        "top_changed_rate": 0.0,
                        "generated_text": "\\n\\n",
                    },
                    {
                        "family": "literal_stable",
                        "prompt_id": "literal_01",
                        "layer": 7,
                        "k": 16,
                        "seed": 0,
                        "component": "coexact_minus_presence",
                        "alpha": 1.0,
                        "generated_steps": 2,
                        "baseline_token_overlap": 1.0,
                        "mean_selected_base_logprob": -0.5,
                        "mean_selected_logprob_gain": -0.1,
                        "mean_kl_base_to_steered": 0.2,
                        "mean_entropy_delta": 0.01,
                        "mean_target_margin_delta": -0.4,
                        "mean_nearest_distance": 0.2,
                        "unique_nodes": 1,
                        "top_changed_rate": 0.0,
                        "generated_text": "\\n\\n",
                    },
                ],
            )

            summary.main(["--run-root", str(root)])

            for name in [
                "closed_loop_contrasts.csv",
                "closed_loop_component_summary.csv",
                "closed_loop_family_summary.csv",
                "closed_loop_layer_summary.csv",
                "closed_loop_k_summary.csv",
                "closed_loop_prompt_summary.csv",
                "closed_loop_prompt_layer_k_summary.csv",
                "closed_loop_summary_report.md",
            ]:
                path = root / name
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_layer_summary_groups_by_layer_component_and_alpha(self) -> None:
        contrasts = [
            {
                "layer": 5,
                "component": "coexact",
                "alpha": 0.8,
                "token_drift_rate": 0.25,
                "baseline_token_overlap": 0.75,
                "mean_selected_base_logprob": -1.0,
                "mean_selected_base_logprob_delta": -0.1,
                "mean_selected_logprob_gain": 0.05,
                "mean_kl_base_to_steered": 0.02,
                "mean_entropy_delta": 0.01,
                "mean_target_margin_delta": 0.1,
                "mean_nearest_distance": 0.2,
                "top_changed_rate": 0.25,
            },
            {
                "layer": 5,
                "component": "coexact",
                "alpha": 0.8,
                "token_drift_rate": 0.75,
                "baseline_token_overlap": 0.25,
                "mean_selected_base_logprob": -2.0,
                "mean_selected_base_logprob_delta": -0.2,
                "mean_selected_logprob_gain": 0.15,
                "mean_kl_base_to_steered": 0.04,
                "mean_entropy_delta": 0.03,
                "mean_target_margin_delta": 0.3,
                "mean_nearest_distance": 0.4,
                "top_changed_rate": 0.5,
            },
            {
                "layer": 7,
                "component": "coexact",
                "alpha": 0.8,
                "token_drift_rate": 1.0,
                "baseline_token_overlap": 0.0,
                "mean_selected_base_logprob": -3.0,
                "mean_selected_base_logprob_delta": -0.3,
                "mean_selected_logprob_gain": 0.25,
                "mean_kl_base_to_steered": 0.06,
                "mean_entropy_delta": 0.05,
                "mean_target_margin_delta": 0.5,
                "mean_nearest_distance": 0.6,
                "top_changed_rate": 0.75,
            },
        ]

        rows = summary.layer_summary_rows(contrasts)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["layer"], 5)
        self.assertAlmostEqual(rows[0]["token_drift_rate_mean"], 0.5)
        self.assertAlmostEqual(rows[0]["mean_target_margin_delta_mean"], 0.2)
        self.assertEqual(rows[1]["layer"], 7)
        self.assertAlmostEqual(rows[1]["token_drift_rate_mean"], 1.0)

    def test_k_summary_groups_by_k_component_and_alpha(self) -> None:
        contrasts = [
            {
                "k": 12,
                "component": "coexact_minus_presence",
                "alpha": 0.8,
                "token_drift_rate": 0.25,
                "baseline_token_overlap": 0.75,
                "mean_selected_base_logprob": -1.0,
                "mean_selected_base_logprob_delta": -0.1,
                "mean_selected_logprob_gain": 0.05,
                "mean_kl_base_to_steered": 0.02,
                "mean_entropy_delta": 0.01,
                "mean_target_margin_delta": 0.1,
                "mean_nearest_distance": 0.2,
                "top_changed_rate": 0.25,
            },
            {
                "k": 12,
                "component": "coexact_minus_presence",
                "alpha": 0.8,
                "token_drift_rate": 0.75,
                "baseline_token_overlap": 0.25,
                "mean_selected_base_logprob": -2.0,
                "mean_selected_base_logprob_delta": -0.2,
                "mean_selected_logprob_gain": 0.15,
                "mean_kl_base_to_steered": 0.04,
                "mean_entropy_delta": 0.03,
                "mean_target_margin_delta": 0.3,
                "mean_nearest_distance": 0.4,
                "top_changed_rate": 0.5,
            },
            {
                "k": 24,
                "component": "coexact_minus_presence",
                "alpha": 0.8,
                "token_drift_rate": 1.0,
                "baseline_token_overlap": 0.0,
                "mean_selected_base_logprob": -3.0,
                "mean_selected_base_logprob_delta": -0.3,
                "mean_selected_logprob_gain": 0.25,
                "mean_kl_base_to_steered": 0.06,
                "mean_entropy_delta": 0.05,
                "mean_target_margin_delta": 0.5,
                "mean_nearest_distance": 0.6,
                "top_changed_rate": 0.75,
            },
        ]

        rows = summary.k_summary_rows(contrasts)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["k"], 12)
        self.assertAlmostEqual(rows[0]["token_drift_rate_mean"], 0.5)
        self.assertAlmostEqual(rows[0]["mean_target_margin_delta_mean"], 0.2)
        self.assertEqual(rows[1]["k"], 24)
        self.assertAlmostEqual(rows[1]["token_drift_rate_mean"], 1.0)

    def test_prompt_summary_tracks_branch_gate_rates(self) -> None:
        contrasts = [
            {
                "family": "ontology_collapse",
                "prompt_id": "ontology_05",
                "layer": 7,
                "k": 16,
                "seed": 0,
                "component": "coexact_minus_presence",
                "alpha": 0.8,
                "token_drift_rate": 1.0,
                "baseline_token_overlap": 0.0,
                "mean_selected_logprob_gain": 0.2,
                "mean_kl_base_to_steered": 0.06,
                "mean_target_margin_delta": 0.4,
                "mean_nearest_distance": 0.3,
            },
            {
                "family": "ontology_collapse",
                "prompt_id": "ontology_05",
                "layer": 7,
                "k": 16,
                "seed": 1,
                "component": "coexact_minus_presence",
                "alpha": 0.8,
                "token_drift_rate": 0.0,
                "baseline_token_overlap": 1.0,
                "mean_selected_logprob_gain": 0.1,
                "mean_kl_base_to_steered": 0.03,
                "mean_target_margin_delta": 0.2,
                "mean_nearest_distance": 0.2,
            },
            {
                "family": "ontology_collapse",
                "prompt_id": "ontology_05",
                "layer": 7,
                "k": 16,
                "seed": 0,
                "component": "presence",
                "alpha": 0.8,
                "token_drift_rate": 0.0,
                "baseline_token_overlap": 1.0,
                "mean_selected_logprob_gain": 0.2,
                "mean_kl_base_to_steered": 0.03,
                "mean_target_margin_delta": -0.3,
                "mean_nearest_distance": 0.2,
            },
            {
                "family": "ontology_collapse",
                "prompt_id": "ontology_05",
                "layer": 7,
                "k": 16,
                "seed": 0,
                "component": "random_tangent",
                "alpha": 0.8,
                "token_drift_rate": 0.0,
                "baseline_token_overlap": 1.0,
                "mean_selected_logprob_gain": 0.0,
                "mean_kl_base_to_steered": 0.02,
                "mean_target_margin_delta": -0.1,
                "mean_nearest_distance": 0.2,
            },
            {
                "family": "ontology_collapse",
                "prompt_id": "ontology_05",
                "layer": 7,
                "k": 16,
                "seed": 1,
                "component": "random_tangent",
                "alpha": 0.8,
                "token_drift_rate": 1.0,
                "baseline_token_overlap": 0.0,
                "mean_selected_logprob_gain": 0.0,
                "mean_kl_base_to_steered": 0.04,
                "mean_target_margin_delta": 0.3,
                "mean_nearest_distance": 0.2,
            },
        ]

        rows = summary.prompt_summary_rows(contrasts)

        self.assertEqual(len(rows), 3)
        coexact_minus = [row for row in rows if row["component"] == "coexact_minus_presence"][0]
        presence = [row for row in rows if row["component"] == "presence"][0]
        random = [row for row in rows if row["component"] == "random_tangent"][0]
        self.assertEqual(coexact_minus["matched_random_rows"], 2)
        self.assertAlmostEqual(coexact_minus["branch_gate_rate"], 0.5)
        self.assertAlmostEqual(coexact_minus["branch_specific_gate_rate"], 0.5)
        self.assertAlmostEqual(coexact_minus["target_positive_rate"], 1.0)
        self.assertAlmostEqual(coexact_minus["drift_ge50_rate"], 0.5)
        self.assertAlmostEqual(coexact_minus["random_branch_gate_rate"], 0.5)
        self.assertAlmostEqual(coexact_minus["branch_gate_minus_random_rate"], 0.0)
        self.assertAlmostEqual(coexact_minus["token_drift_rate_minus_random_mean"], 0.0)
        self.assertAlmostEqual(coexact_minus["token_drift_ge_random_rate"], 0.5)
        self.assertAlmostEqual(coexact_minus["mean_target_margin_delta_minus_random_mean"], 0.2)
        self.assertAlmostEqual(coexact_minus["target_margin_gt_random_rate"], 0.5)
        self.assertEqual(presence["matched_random_rows"], 1)
        self.assertAlmostEqual(presence["branch_gate_rate"], 0.0)
        self.assertAlmostEqual(presence["branch_specific_gate_rate"], 0.0)
        self.assertAlmostEqual(presence["target_positive_rate"], 0.0)
        self.assertAlmostEqual(presence["mean_target_margin_delta_minus_random_mean"], -0.2)
        self.assertAlmostEqual(random["branch_specific_gate_rate"], 0.0)

    def test_prompt_layer_k_summary_tracks_exception_cells(self) -> None:
        contrasts = [
            {
                "family": "identity_stress",
                "prompt_id": "identity_04",
                "layer": 5,
                "k": 16,
                "seed": 0,
                "component": "negative_coexact",
                "alpha": 0.8,
                "token_drift_rate": 0.5,
                "baseline_token_overlap": 0.5,
                "mean_selected_logprob_gain": 0.1,
                "mean_kl_base_to_steered": 0.04,
                "mean_target_margin_delta": 0.4,
                "mean_nearest_distance": 0.2,
            },
            {
                "family": "identity_stress",
                "prompt_id": "identity_04",
                "layer": 5,
                "k": 16,
                "seed": 1,
                "component": "negative_coexact",
                "alpha": 0.8,
                "token_drift_rate": 0.0,
                "baseline_token_overlap": 1.0,
                "mean_selected_logprob_gain": 0.0,
                "mean_kl_base_to_steered": 0.02,
                "mean_target_margin_delta": 0.3,
                "mean_nearest_distance": 0.2,
            },
            {
                "family": "identity_stress",
                "prompt_id": "identity_04",
                "layer": 5,
                "k": 16,
                "seed": 0,
                "component": "random_tangent",
                "alpha": 0.8,
                "token_drift_rate": 0.0,
                "baseline_token_overlap": 1.0,
                "mean_selected_logprob_gain": 0.0,
                "mean_kl_base_to_steered": 0.02,
                "mean_target_margin_delta": 0.1,
                "mean_nearest_distance": 0.2,
            },
            {
                "family": "identity_stress",
                "prompt_id": "identity_04",
                "layer": 5,
                "k": 16,
                "seed": 1,
                "component": "random_tangent",
                "alpha": 0.8,
                "token_drift_rate": 0.5,
                "baseline_token_overlap": 0.5,
                "mean_selected_logprob_gain": 0.0,
                "mean_kl_base_to_steered": 0.02,
                "mean_target_margin_delta": 0.2,
                "mean_nearest_distance": 0.2,
            },
        ]

        rows = summary.prompt_layer_k_summary_rows(contrasts)

        negative = [row for row in rows if row["component"] == "negative_coexact"][0]
        random = [row for row in rows if row["component"] == "random_tangent"][0]
        self.assertEqual(negative["layer"], 5)
        self.assertEqual(negative["k"], 16)
        self.assertEqual(negative["matched_random_rows"], 2)
        self.assertAlmostEqual(negative["branch_gate_rate"], 0.5)
        self.assertAlmostEqual(negative["branch_specific_gate_rate"], 0.5)
        self.assertAlmostEqual(negative["random_branch_gate_rate"], 0.5)
        self.assertAlmostEqual(negative["branch_gate_minus_random_rate"], 0.0)
        self.assertAlmostEqual(negative["token_drift_rate_minus_random_mean"], 0.0)
        self.assertAlmostEqual(negative["mean_target_margin_delta_minus_random_mean"], 0.2)
        self.assertAlmostEqual(random["branch_specific_gate_rate"], 0.0)

    def test_prompt_layer_k_summary_keeps_target_sets_separate(self) -> None:
        contrasts = [
            {
                "family": "identity_stress",
                "prompt_id": "identity_04",
                "layer": 7,
                "k": 16,
                "seed": 0,
                "target_set": "identity_stress",
                "component": "negative_coexact",
                "alpha": 0.8,
                "token_drift_rate": 0.5,
                "baseline_token_overlap": 0.5,
                "mean_selected_logprob_gain": 0.1,
                "mean_kl_base_to_steered": 0.04,
                "mean_target_margin_delta": 0.4,
                "mean_nearest_distance": 0.2,
            },
            {
                "family": "identity_stress",
                "prompt_id": "identity_04",
                "layer": 7,
                "k": 16,
                "seed": 0,
                "target_set": "identity_stress",
                "component": "random_tangent",
                "alpha": 0.8,
                "token_drift_rate": 0.0,
                "baseline_token_overlap": 1.0,
                "mean_selected_logprob_gain": 0.0,
                "mean_kl_base_to_steered": 0.02,
                "mean_target_margin_delta": 0.1,
                "mean_nearest_distance": 0.2,
            },
            {
                "family": "identity_stress",
                "prompt_id": "identity_04",
                "layer": 7,
                "k": 16,
                "seed": 0,
                "target_set": "identity_generic_control",
                "component": "negative_coexact",
                "alpha": 0.8,
                "token_drift_rate": 0.5,
                "baseline_token_overlap": 0.5,
                "mean_selected_logprob_gain": 0.1,
                "mean_kl_base_to_steered": 0.04,
                "mean_target_margin_delta": -0.2,
                "mean_nearest_distance": 0.2,
            },
            {
                "family": "identity_stress",
                "prompt_id": "identity_04",
                "layer": 7,
                "k": 16,
                "seed": 0,
                "target_set": "identity_generic_control",
                "component": "random_tangent",
                "alpha": 0.8,
                "token_drift_rate": 0.0,
                "baseline_token_overlap": 1.0,
                "mean_selected_logprob_gain": 0.0,
                "mean_kl_base_to_steered": 0.02,
                "mean_target_margin_delta": 0.0,
                "mean_nearest_distance": 0.2,
            },
        ]

        rows = summary.prompt_layer_k_summary_rows(contrasts)

        negative_rows = [row for row in rows if row["component"] == "negative_coexact"]
        self.assertEqual({row["target_set"] for row in negative_rows}, {"identity_stress", "identity_generic_control"})
        by_target = {row["target_set"]: row for row in negative_rows}
        self.assertAlmostEqual(by_target["identity_stress"]["branch_gate_rate"], 1.0)
        self.assertAlmostEqual(by_target["identity_generic_control"]["branch_gate_rate"], 0.0)
        self.assertAlmostEqual(by_target["identity_stress"]["mean_target_margin_delta_minus_random_mean"], 0.3)
        self.assertAlmostEqual(by_target["identity_generic_control"]["mean_target_margin_delta_minus_random_mean"], -0.2)


if __name__ == "__main__":
    unittest.main()
