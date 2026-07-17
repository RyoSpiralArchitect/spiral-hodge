from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_hltd_branch_hodge as summarize


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestSummarizeHLTDBranchHodge(unittest.TestCase):
    def test_aggregate_layer_hodge_tracks_mean_min_and_max(self) -> None:
        rows = [
            {
                "family": "literal_stable",
                "k": 16,
                "layer": 5,
                "real_coexact": 0.8,
                "real_exact": 0.2,
                "real_harmonic": 0.0,
                "real_minus_shuffle_coexact": 0.1,
            },
            {
                "family": "ontology_collapse",
                "k": 16,
                "layer": 5,
                "real_coexact": 0.9,
                "real_exact": 0.1,
                "real_harmonic": 0.0,
                "real_minus_shuffle_coexact": 0.3,
            },
        ]

        [layer] = summarize.aggregate_layer_hodge(rows)

        self.assertEqual(layer["k"], 16)
        self.assertEqual(layer["layer"], 5)
        self.assertEqual(layer["n_family_rows"], 2)
        self.assertAlmostEqual(layer["real_coexact_mean"], 0.85)
        self.assertAlmostEqual(layer["real_coexact_min"], 0.8)
        self.assertAlmostEqual(layer["real_coexact_max"], 0.9)
        self.assertAlmostEqual(layer["real_minus_shuffle_coexact_mean"], 0.2)

    def test_join_layer_rows_keeps_hodge_steering_and_probe_fields(self) -> None:
        joined = summarize.join_layer_rows(
            hodge_rows=[
                {
                    "k": 16,
                    "layer": 7,
                    "real_coexact_mean": 0.87,
                    "real_exact_mean": 0.13,
                    "real_harmonic_mean": 0.0,
                    "real_minus_shuffle_coexact_mean": 0.07,
                    "real_minus_random_coexact_mean": 0.14,
                    "real_graph_high_freq_mean": 0.2,
                    "real_hodge_curl_mean": 0.01,
                }
            ],
            steering_rows=[
                {
                    "k": 16,
                    "layer": 7,
                    "selector": "middle",
                    "component": "coexact",
                    "next_token_delta": 0.45,
                    "semantic_margin_delta": -0.16,
                    "kl_delta": 0.02,
                }
            ],
            probe_rows=[
                {
                    "k": 16,
                    "layer": 7,
                    "selector": "middle",
                    "component": "coexact",
                    "probe": "ontology_collapse",
                    "probe_label_margin_delta": 0.36,
                    "probe_positive_prob_delta": 0.04,
                }
            ],
        )

        [row] = joined

        self.assertEqual(row["component"], "coexact")
        self.assertEqual(row["probe"], "ontology_collapse")
        self.assertAlmostEqual(row["hodge_coexact"], 0.87)
        self.assertAlmostEqual(row["hodge_exact"], 0.13)
        self.assertAlmostEqual(row["next_token_delta"], 0.45)
        self.assertAlmostEqual(row["semantic_margin_delta"], -0.16)
        self.assertAlmostEqual(row["probe_label_margin_delta"], 0.36)

    def test_aggregate_k_sweep_tracks_structural_branch_ranges(self) -> None:
        rows = [
            {
                "family": "literal_stable",
                "k": 12,
                "real_exact_l5_l8": 0.12,
                "real_coexact_l5_l8": 0.88,
                "real_harmonic_max": 0.0,
                "real_minus_shuffle_coexact_l5_l8": 0.05,
                "max_same_graph_reverse_coexact_gap": 0.001,
            },
            {
                "family": "ontology_collapse",
                "k": 12,
                "real_exact_l5_l8": 0.10,
                "real_coexact_l5_l8": 0.90,
                "real_harmonic_max": 0.01,
                "real_minus_shuffle_coexact_l5_l8": 0.15,
                "max_same_graph_reverse_coexact_gap": 0.002,
            },
            {
                "family": "literal_stable",
                "k": 16,
                "real_exact_l5_l8": 0.14,
                "real_coexact_l5_l8": 0.86,
                "real_harmonic_max": 0.0,
                "real_minus_shuffle_coexact_l5_l8": 0.04,
                "max_same_graph_reverse_coexact_gap": 0.003,
            },
        ]

        sweep = summarize.aggregate_k_sweep(rows)
        by_k = {row["k"]: row for row in sweep}

        self.assertEqual(by_k[12]["n_family_rows"], 2)
        self.assertAlmostEqual(by_k[12]["real_coexact_l5_l8_mean"], 0.89)
        self.assertAlmostEqual(by_k[12]["real_exact_l5_l8_min"], 0.10)
        self.assertAlmostEqual(by_k[12]["real_harmonic_max_max"], 0.01)
        self.assertAlmostEqual(by_k[12]["real_minus_shuffle_coexact_l5_l8_mean"], 0.10)
        self.assertAlmostEqual(by_k[12]["max_same_graph_reverse_coexact_gap_max"], 0.002)
        self.assertAlmostEqual(by_k[16]["real_coexact_l5_l8_mean"], 0.86)

    def test_branch_scoreboard_treats_zero_as_valid_best_value(self) -> None:
        rows = [
            {
                "selector": "middle",
                "component": "coexact",
                "probe": "ontology_collapse",
                "layer": 4,
                "next_token_delta": -1.0,
                "probe_label_margin_delta": -0.5,
            },
            {
                "selector": "middle",
                "component": "coexact",
                "probe": "ontology_collapse",
                "layer": 5,
                "next_token_delta": 0.0,
                "probe_label_margin_delta": 0.0,
            },
            {
                "selector": "middle",
                "component": "coexact",
                "probe": "ontology_collapse",
                "layer": 6,
                "next_token_delta": -0.25,
                "probe_label_margin_delta": -0.1,
            },
        ]

        [score] = summarize.build_branch_score_rows(rows)

        self.assertAlmostEqual(score["next_token_delta_mean"], -0.4166666666666667)
        self.assertEqual(score["next_token_delta_best_layer"], 5)
        self.assertEqual(score["probe_label_margin_delta_best_layer"], 5)
        self.assertAlmostEqual(score["next_token_delta_max"], 0.0)

    def test_causal_k_scoreboard_joins_structural_steering_and_probe_sweeps(self) -> None:
        scores = summarize.build_causal_k_score_rows(
            k_sweep=[
                {
                    "k": 12,
                    "real_exact_l5_l8_mean": 0.11,
                    "real_coexact_l5_l8_mean": 0.89,
                    "real_harmonic_max_max": 0.01,
                    "real_minus_shuffle_coexact_l5_l8_mean": 0.09,
                    "max_same_graph_reverse_coexact_gap_max": 0.0,
                }
            ],
            steering_rows=[
                {
                    "k": 12,
                    "layer": 5,
                    "selector": "middle",
                    "component": "coexact",
                    "next_token_delta": 0.2,
                    "semantic_margin_delta": -0.1,
                    "kl_delta": 0.01,
                    "entropy_delta": -0.02,
                },
                {
                    "k": 12,
                    "layer": 6,
                    "selector": "middle",
                    "component": "coexact",
                    "next_token_delta": 0.4,
                    "semantic_margin_delta": 0.1,
                    "kl_delta": 0.03,
                    "entropy_delta": -0.04,
                },
            ],
            probe_rows=[
                {
                    "k": 12,
                    "layer": 5,
                    "selector": "middle",
                    "component": "coexact",
                    "probe": "ontology_collapse",
                    "probe_label_margin_delta": -0.2,
                    "probe_positive_prob_delta": -0.01,
                    "probe_entropy_delta": 0.02,
                },
                {
                    "k": 12,
                    "layer": 6,
                    "selector": "middle",
                    "component": "coexact",
                    "probe": "ontology_collapse",
                    "probe_label_margin_delta": 0.0,
                    "probe_positive_prob_delta": 0.01,
                    "probe_entropy_delta": 0.04,
                },
            ],
        )

        [score] = scores

        self.assertEqual(score["k"], 12)
        self.assertEqual(score["component"], "coexact")
        self.assertAlmostEqual(score["hodge_coexact_mean"], 0.89)
        self.assertAlmostEqual(score["next_token_delta_mean"], 0.3)
        self.assertEqual(score["next_token_delta_best_layer"], 6)
        self.assertAlmostEqual(score["semantic_margin_delta_mean"], 0.0)
        self.assertAlmostEqual(score["probe_label_margin_delta_mean"], -0.1)
        self.assertEqual(score["probe_label_margin_delta_best_layer"], 6)

    def test_selector_delta_rows_compare_against_baseline_selector(self) -> None:
        deltas = summarize.build_selector_delta_rows(
            [
                {
                    "k": 16,
                    "selector": "middle",
                    "component": "coexact",
                    "probe": "ontology_collapse",
                    "next_token_delta_mean": 0.3,
                    "semantic_margin_delta_mean": -0.1,
                    "probe_label_margin_delta_mean": 0.2,
                },
                {
                    "k": 16,
                    "selector": "max_component",
                    "component": "coexact",
                    "probe": "ontology_collapse",
                    "next_token_delta_mean": 0.1,
                    "semantic_margin_delta_mean": 0.05,
                    "probe_label_margin_delta_mean": -0.4,
                },
            ],
            baseline_selector="middle",
        )

        [row] = deltas

        self.assertEqual(row["compare_selector"], "max_component")
        self.assertAlmostEqual(row["baseline_next_token_delta_mean"], 0.3)
        self.assertAlmostEqual(row["compare_next_token_delta_mean"], 0.1)
        self.assertAlmostEqual(row["next_token_delta_mean_diff"], -0.2)
        self.assertAlmostEqual(row["semantic_margin_delta_mean_diff"], 0.15)
        self.assertAlmostEqual(row["probe_label_margin_delta_mean_diff"], -0.6)

    def test_closed_loop_branch_scoreboard_joins_hodge_family_metrics(self) -> None:
        scores = summarize.build_closed_loop_branch_score_rows(
            closed_loop_rows=[
                {
                    "source": "closed_loop_seed_probe",
                    "family": "ontology_collapse",
                    "prompt_id": "ontology_03",
                    "component": "coexact_minus_presence",
                    "alpha": 0.8,
                    "n_rows": 1,
                    "matched_random_rows": 1,
                    "branch_gate_rate": 1.0,
                    "branch_specific_gate_rate": 1.0,
                    "random_branch_gate_rate": 0.0,
                    "branch_gate_minus_random_rate": 1.0,
                    "token_drift_rate_mean": 1.0,
                    "token_drift_rate_minus_random_mean": 1.0,
                    "mean_target_margin_delta_mean": 0.3,
                    "mean_target_margin_delta_minus_random_mean": 0.2,
                },
                {
                    "source": "closed_loop_seed_probe",
                    "family": "ontology_collapse",
                    "prompt_id": "ontology_02",
                    "component": "coexact_minus_presence",
                    "alpha": 0.8,
                    "n_rows": 1,
                    "matched_random_rows": 1,
                    "branch_gate_rate": 1.0,
                    "branch_specific_gate_rate": 0.0,
                    "random_branch_gate_rate": 0.0,
                    "branch_gate_minus_random_rate": 1.0,
                    "token_drift_rate_mean": 1.0,
                    "token_drift_rate_minus_random_mean": 0.25,
                    "mean_target_margin_delta_mean": 0.1,
                    "mean_target_margin_delta_minus_random_mean": -0.1,
                },
            ],
            family_k=[
                {
                    "family": "ontology_collapse",
                    "real_coexact_l5_l8": 0.89,
                    "real_exact_l5_l8": 0.11,
                    "real_harmonic_max": 0.0,
                    "real_minus_shuffle_coexact_l5_l8": 0.1,
                    "real_minus_random_coexact_l5_l8": 0.16,
                }
            ],
        )

        [score] = scores

        self.assertEqual(score["source"], "closed_loop_seed_probe")
        self.assertEqual(score["family"], "ontology_collapse")
        self.assertEqual(score["component"], "coexact_minus_presence")
        self.assertEqual(score["n_prompt_rows"], 2)
        self.assertEqual(score["n_rows"], 2)
        self.assertAlmostEqual(score["branch_gate_rate"], 1.0)
        self.assertAlmostEqual(score["branch_specific_gate_rate"], 0.5)
        self.assertAlmostEqual(score["branch_specific_prompt_rate"], 0.5)
        self.assertAlmostEqual(score["mean_target_margin_delta_minus_random_mean"], 0.05)
        self.assertAlmostEqual(score["target_advantage_prompt_rate"], 0.5)
        self.assertAlmostEqual(score["hodge_coexact_l5_l8"], 0.89)

    def test_branch_role_summary_combines_causal_and_closed_loop_roles(self) -> None:
        rows = summarize.build_branch_role_summary_rows(
            causal_k_scores=[
                {
                    "k": 12,
                    "selector": "middle",
                    "component": "coexact_minus_presence",
                    "probe": "ontology_collapse",
                    "hodge_coexact_mean": 0.89,
                    "hodge_exact_mean": 0.11,
                    "hodge_harmonic_max": 0.0,
                    "next_token_delta_mean": 0.2,
                    "semantic_margin_delta_mean": -0.1,
                    "probe_label_margin_delta_mean": -0.3,
                    "probe_positive_prob_delta_mean": -0.01,
                },
                {
                    "k": 16,
                    "selector": "middle",
                    "component": "coexact_minus_presence",
                    "probe": "ontology_collapse",
                    "hodge_coexact_mean": 0.87,
                    "hodge_exact_mean": 0.13,
                    "hodge_harmonic_max": 0.0,
                    "next_token_delta_mean": 0.4,
                    "semantic_margin_delta_mean": 0.1,
                    "probe_label_margin_delta_mean": -0.1,
                    "probe_positive_prob_delta_mean": 0.01,
                },
                {
                    "k": 16,
                    "selector": "max_component",
                    "component": "coexact_minus_presence",
                    "probe": "ontology_collapse",
                    "hodge_coexact_mean": 0.87,
                    "hodge_exact_mean": 0.13,
                    "hodge_harmonic_max": 0.0,
                    "next_token_delta_mean": -0.1,
                    "semantic_margin_delta_mean": 0.0,
                    "probe_label_margin_delta_mean": 0.2,
                    "probe_positive_prob_delta_mean": 0.02,
                },
            ],
            closed_loop_branch_scores=[
                {
                    "family": "ontology_collapse",
                    "component": "coexact_minus_presence",
                    "matched_random_rows": 5,
                    "branch_specific_gate_rate": 0.6,
                    "mean_target_margin_delta_minus_random_mean": 0.2,
                },
                {
                    "family": "ontology_collapse",
                    "component": "coexact_minus_presence",
                    "matched_random_rows": 10,
                    "branch_specific_gate_rate": 0.3,
                    "mean_target_margin_delta_minus_random_mean": -0.1,
                },
            ],
            selector="middle",
        )

        [row] = rows

        self.assertEqual(row["selector"], "middle")
        self.assertEqual(row["component"], "coexact_minus_presence")
        self.assertEqual(row["k_values"], "12 16")
        self.assertAlmostEqual(row["hodge_coexact_mean"], 0.88)
        self.assertAlmostEqual(row["next_token_delta_mean"], 0.3)
        self.assertAlmostEqual(row["probe_label_margin_delta_mean"], -0.2)
        self.assertAlmostEqual(row["closed_loop_branch_specific_gate_rate_mean"], 0.4)
        self.assertAlmostEqual(row["closed_loop_target_margin_delta_minus_random_mean"], 0.0)
        self.assertEqual(row["role_label"], "closed_loop_traversal")

    def test_branch_role_diagnostics_score_family_local_expectations(self) -> None:
        rows = summarize.build_branch_role_diagnostic_rows(
            [
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "probe": "ontology_collapse",
                    "component": "coexact",
                    "layer": 5,
                    "hodge_coexact": 0.89,
                    "hodge_exact": 0.11,
                    "hodge_real_minus_shuffle_coexact": 0.08,
                    "next_token_delta": 0.2,
                    "semantic_margin_delta": 0.1,
                    "probe_label_margin_delta": -0.4,
                },
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "probe": "ontology_collapse",
                    "component": "coexact",
                    "layer": 7,
                    "hodge_coexact": 0.87,
                    "hodge_exact": 0.13,
                    "hodge_real_minus_shuffle_coexact": 0.06,
                    "next_token_delta": 0.4,
                    "semantic_margin_delta": -0.1,
                    "probe_label_margin_delta": 0.2,
                },
                {
                    "selector": "middle",
                    "family": "literal_stable",
                    "probe": "ontology_collapse",
                    "component": "presence",
                    "layer": 5,
                    "hodge_coexact": 0.85,
                    "hodge_exact": 0.15,
                    "hodge_real_minus_shuffle_coexact": 0.03,
                    "next_token_delta": 0.3,
                    "semantic_margin_delta": -0.2,
                    "probe_label_margin_delta": 0.8,
                },
                {
                    "selector": "max_component",
                    "family": "literal_stable",
                    "probe": "ontology_collapse",
                    "component": "presence",
                    "layer": 5,
                    "hodge_coexact": 0.85,
                    "hodge_exact": 0.15,
                    "hodge_real_minus_shuffle_coexact": 0.03,
                    "next_token_delta": -0.2,
                    "semantic_margin_delta": -0.2,
                    "probe_label_margin_delta": 0.8,
                },
            ],
            selector="middle",
        )

        by_component = {(row["family"], row["component"]): row for row in rows}
        coexact = by_component[("ontology_collapse", "coexact")]
        self.assertEqual(coexact["expected_role"], "traversal")
        self.assertEqual(coexact["observed_role"], "traversal")
        self.assertEqual(coexact["role_pass"], 1)
        self.assertAlmostEqual(coexact["role_score"], 1.0)
        self.assertAlmostEqual(coexact["next_token_delta_mean"], 0.3)
        self.assertEqual(coexact["layer_values"], "5 7")

        presence = by_component[("literal_stable", "presence")]
        self.assertEqual(presence["expected_role"], "stabilization")
        self.assertEqual(presence["observed_role"], "combined")
        self.assertEqual(presence["role_pass"], 0)
        self.assertAlmostEqual(presence["role_score"], 0.5)
        self.assertEqual(presence["criteria_failed"], "next_not_positive")

    def test_branch_condition_summary_compresses_probe_cells(self) -> None:
        rows = summarize.build_branch_condition_summary_rows(
            [
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "probe": "ontology_collapse",
                    "component": "coexact",
                    "expected_role": "traversal",
                    "observed_role": "traversal",
                    "role_score": 1.0,
                    "role_pass": 1,
                    "criteria_failed": "",
                    "next_token_delta_mean": 0.4,
                    "probe_label_margin_delta_mean": -0.2,
                    "hodge_coexact_mean": 0.88,
                },
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "probe": "identity_stress",
                    "component": "coexact",
                    "expected_role": "traversal",
                    "observed_role": "combined",
                    "role_score": 1.0,
                    "role_pass": 1,
                    "criteria_failed": "",
                    "next_token_delta_mean": 0.2,
                    "probe_label_margin_delta_mean": 0.3,
                    "hodge_coexact_mean": 0.86,
                },
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "probe": "ontology_collapse",
                    "component": "presence_plus_coexact",
                    "expected_role": "combined",
                    "observed_role": "traversal",
                    "role_score": 0.5,
                    "role_pass": 0,
                    "criteria_failed": "probe_positive",
                    "next_token_delta_mean": 0.8,
                    "probe_label_margin_delta_mean": -0.4,
                    "hodge_coexact_mean": 0.88,
                },
                {
                    "selector": "max_component",
                    "family": "ontology_collapse",
                    "probe": "ontology_collapse",
                    "component": "coexact",
                    "expected_role": "traversal",
                    "observed_role": "reverse_control",
                    "role_score": 0.0,
                    "role_pass": 0,
                    "criteria_failed": "next_positive",
                    "next_token_delta_mean": -0.3,
                    "probe_label_margin_delta_mean": -0.2,
                    "hodge_coexact_mean": 0.88,
                },
            ],
            selector="middle",
        )

        by_component = {row["component"]: row for row in rows}
        coexact = by_component["coexact"]
        self.assertEqual(coexact["condition_label"], "stable_expected")
        self.assertEqual(coexact["role_pass_count"], 2)
        self.assertAlmostEqual(coexact["role_pass_rate"], 1.0)
        self.assertEqual(coexact["observed_role_counts"], "combined:1 traversal:1")

        combined = by_component["presence_plus_coexact"]
        self.assertEqual(combined["condition_label"], "systematic_partial_break")
        self.assertEqual(combined["role_pass_count"], 0)
        self.assertAlmostEqual(combined["mean_role_score"], 0.5)
        self.assertEqual(combined["failed_criteria_counts"], "probe_positive:1")

    def test_branch_layer_condition_summary_keeps_layer_spine(self) -> None:
        rows = summarize.build_branch_layer_condition_rows(
            [
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "probe": "ontology_collapse",
                    "component": "coexact_minus_presence",
                    "layer": 5,
                    "hodge_coexact": 0.89,
                    "hodge_exact": 0.11,
                    "next_token_delta": 0.4,
                    "probe_label_margin_delta": -0.2,
                },
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "probe": "identity_stress",
                    "component": "coexact_minus_presence",
                    "layer": 5,
                    "hodge_coexact": 0.89,
                    "hodge_exact": 0.11,
                    "next_token_delta": 0.2,
                    "probe_label_margin_delta": 0.3,
                },
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "probe": "ontology_collapse",
                    "component": "coexact_minus_presence",
                    "layer": 7,
                    "hodge_coexact": 0.87,
                    "hodge_exact": 0.13,
                    "next_token_delta": -0.5,
                    "probe_label_margin_delta": -0.4,
                },
                {
                    "selector": "max_component",
                    "family": "ontology_collapse",
                    "probe": "ontology_collapse",
                    "component": "coexact_minus_presence",
                    "layer": 5,
                    "hodge_coexact": 0.89,
                    "hodge_exact": 0.11,
                    "next_token_delta": -0.3,
                    "probe_label_margin_delta": -0.2,
                },
            ],
            selector="middle",
        )

        by_layer = {row["layer"]: row for row in rows}
        layer5 = by_layer[5]
        self.assertEqual(layer5["condition_label"], "mixed_condition")
        self.assertEqual(layer5["role_pass_count"], 1)
        self.assertAlmostEqual(layer5["role_pass_rate"], 0.5)
        self.assertEqual(layer5["failed_criteria_counts"], "probe_not_positive:1")

        layer7 = by_layer[7]
        self.assertEqual(layer7["condition_label"], "systematic_partial_break")
        self.assertEqual(layer7["role_pass_count"], 0)
        self.assertAlmostEqual(layer7["mean_role_score"], 0.5)
        self.assertEqual(layer7["failed_criteria_counts"], "next_positive:1")

    def test_branch_layer_transition_summary_compresses_stable_spans(self) -> None:
        rows = summarize.build_branch_layer_transition_rows(
            [
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "component": "coexact",
                    "layer": 4,
                    "role_pass_rate": 1.0,
                    "mean_role_score": 1.0,
                    "failed_criteria_counts": "",
                },
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "component": "coexact",
                    "layer": 5,
                    "role_pass_rate": 1.0,
                    "mean_role_score": 1.0,
                    "failed_criteria_counts": "",
                },
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "component": "coexact",
                    "layer": 7,
                    "role_pass_rate": 0.5,
                    "mean_role_score": 0.75,
                    "failed_criteria_counts": "probe_positive:1",
                },
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "component": "coexact",
                    "layer": 8,
                    "role_pass_rate": 0.0,
                    "mean_role_score": 0.0,
                    "failed_criteria_counts": "next_positive:2",
                },
                {
                    "selector": "max_component",
                    "family": "ontology_collapse",
                    "component": "coexact",
                    "layer": 4,
                    "role_pass_rate": 0.0,
                    "mean_role_score": 0.0,
                    "failed_criteria_counts": "next_positive:2",
                },
            ],
            selector="middle",
        )

        [row] = rows
        self.assertEqual(row["transition_label"], "sparse_stability")
        self.assertEqual(row["stable_layers"], "L4-L5")
        self.assertEqual(row["mostly_or_mixed_layers"], "L7")
        self.assertEqual(row["break_layers"], "L8")
        self.assertEqual(row["first_stable_layer"], "L4")
        self.assertEqual(row["last_stable_layer"], "L5")
        self.assertEqual(row["longest_stable_run"], 2)
        self.assertEqual(row["longest_stable_span"], "L4-L5")

    def test_branch_band_candidate_scoreboard_prioritizes_structural_and_closed_loop_support(self) -> None:
        rows = summarize.build_branch_band_candidate_rows(
            branch_layer_transition_summary=[
                {
                    "selector": "middle",
                    "family": "ontology_collapse",
                    "component": "coexact",
                    "expected_role": "traversal",
                    "transition_label": "all_layer_stable",
                    "stable_layer_rate": 1.0,
                    "mean_layer_pass_rate": 1.0,
                    "mean_layer_role_score": 1.0,
                    "stable_layers": "L4-L8",
                    "mostly_or_mixed_layers": "",
                    "break_layers": "",
                    "longest_stable_span": "L4-L8",
                },
                {
                    "selector": "middle",
                    "family": "literal_stable",
                    "component": "presence",
                    "expected_role": "stabilization",
                    "transition_label": "sparse_stability",
                    "stable_layer_rate": 0.5,
                    "mean_layer_pass_rate": 0.5,
                    "mean_layer_role_score": 0.75,
                    "stable_layers": "L5",
                    "mostly_or_mixed_layers": "L7",
                    "break_layers": "",
                    "longest_stable_span": "L5",
                },
                {
                    "selector": "max_component",
                    "family": "ontology_collapse",
                    "component": "coexact",
                    "stable_layer_rate": 0.0,
                    "mean_layer_role_score": 0.0,
                },
            ],
            closed_loop_branch_scores=[
                {
                    "family": "ontology_collapse",
                    "component": "coexact",
                    "matched_random_rows": 2,
                    "branch_specific_gate_rate": 0.5,
                    "branch_gate_rate": 0.75,
                    "token_drift_rate_mean": 0.4,
                    "mean_target_margin_delta_minus_random_mean": 0.2,
                },
                {
                    "family": "ontology_collapse",
                    "component": "coexact",
                    "matched_random_rows": 1,
                    "branch_specific_gate_rate": 1.0,
                    "branch_gate_rate": 1.0,
                    "token_drift_rate_mean": 0.7,
                    "mean_target_margin_delta_minus_random_mean": 0.0,
                },
            ],
            selector="middle",
        )

        self.assertEqual([row["family"] for row in rows], ["ontology_collapse", "literal_stable"])
        coexact = rows[0]
        self.assertEqual(coexact["candidate_label"], "causal_band_ready")
        self.assertEqual(coexact["recommended_layers"], "L4-L8")
        self.assertEqual(coexact["closed_loop_matched_random_rows"], 3)
        self.assertAlmostEqual(coexact["closed_loop_branch_specific_gate_rate_mean"], 2.0 / 3.0)
        self.assertAlmostEqual(coexact["closed_loop_target_margin_delta_minus_random_mean"], 2.0 / 15.0)
        self.assertGreater(coexact["priority_score"], rows[1]["priority_score"])

        presence = rows[1]
        self.assertEqual(presence["candidate_label"], "narrow_layer_probe")
        self.assertEqual(presence["recommended_layers"], "L5")

    def test_reverse_specificity_rows_preserve_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reverse.csv"
            write_csv(
                path,
                [
                    {
                        "panel_label": "identity_04 L7/k16",
                        "target_set": "identity_stress",
                        "component": "negative_coexact",
                        "branch_specific_gate_rate": 1.0,
                        "random_branch_gate_rate": 0.0,
                        "mean_target_margin_delta_minus_random_mean": 0.3,
                        "token_drift_rate_mean": 0.5,
                    }
                ],
            )

            rows = summarize.reverse_specificity_rows([path])

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["panel_label"], "identity_04 L7/k16")
            self.assertEqual(rows[0]["reverse_specificity_source"], str(path))


if __name__ == "__main__":
    unittest.main()
