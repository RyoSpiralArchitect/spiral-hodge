from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import plot_hltd_branch_hodge as plots


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestHLTDBranchHodgePlots(unittest.TestCase):
    def test_build_plots_writes_pngs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "branch"
            hodge_layer_rows = []
            for layer in [5, 7]:
                hodge_layer_rows.append(
                    {
                        "k": 16,
                        "layer": layer,
                        "real_exact_mean": 0.12 + 0.01 * (layer == 7),
                        "real_coexact_mean": 0.88 - 0.01 * (layer == 7),
                        "real_harmonic_mean": 0.001,
                        "real_minus_shuffle_coexact_mean": 0.07,
                        "real_minus_random_coexact_mean": 0.16,
                        "real_hodge_curl_mean": 0.32,
                    }
                )
            hodge_k_rows = []
            for k in [12, 16]:
                hodge_k_rows.append(
                    {
                        "k": k,
                        "real_exact_l5_l8_mean": 0.1,
                        "real_coexact_l5_l8_mean": 0.9,
                        "real_harmonic_max_mean": 0.001,
                        "real_minus_shuffle_coexact_l5_l8_mean": 0.08,
                        "real_minus_random_coexact_l5_l8_mean": 0.2,
                        "max_same_graph_reverse_coexact_gap_mean": 0.0,
                    }
                )
            topology_rows = []
            for topology in ["triangles", "no_triangles"]:
                for family in ["literal_stable", "ontology_collapse"]:
                    topology_rows.append(
                        {
                            "topology": topology,
                            "family": family,
                            "k": 16,
                            "real_exact_l5_l8": 0.12,
                            "real_coexact_l5_l8": 0.86 if topology == "triangles" else 0.0,
                            "real_harmonic_max": 0.0 if topology == "triangles" else 0.88,
                        }
                    )
            causal_rows = []
            for k in [12, 16]:
                for component in ["coexact", "presence"]:
                    causal_rows.append(
                        {
                            "k": k,
                            "selector": "middle",
                            "component": component,
                            "probe": "ontology_collapse",
                            "next_token_delta_mean": 0.25 if component == "coexact" else -0.1,
                            "probe_label_margin_delta_mean": 0.1 if component == "coexact" else 0.8,
                        }
                    )
            closed_loop_rows = []
            for component in ["coexact", "presence"]:
                closed_loop_rows.append(
                    {
                        "source": "closed_loop_seed_probe",
                        "family": "ontology_collapse",
                        "component": component,
                        "alpha": 0.8,
                        "n_prompt_rows": 2,
                        "n_rows": 10,
                        "matched_random_rows": 10,
                        "branch_gate_rate": 0.8 if component == "coexact" else 0.0,
                        "branch_specific_gate_rate": 0.6 if component == "coexact" else 0.0,
                        "branch_specific_prompt_rate": 0.5 if component == "coexact" else 0.0,
                        "random_branch_gate_rate": 0.2,
                        "branch_gate_minus_random_rate": 0.6 if component == "coexact" else -0.2,
                        "token_drift_rate_mean": 0.75 if component == "coexact" else 0.1,
                        "token_drift_rate_minus_random_mean": 0.4 if component == "coexact" else -0.2,
                        "mean_target_margin_delta_mean": 0.3 if component == "coexact" else -0.2,
                        "mean_target_margin_delta_minus_random_mean": 0.25 if component == "coexact" else -0.35,
                        "hodge_coexact_l5_l8": 0.89,
                        "hodge_exact_l5_l8": 0.11,
                        "hodge_harmonic_max": 0.0,
                    }
                )
            role_rows = []
            for component in ["coexact", "presence"]:
                role_rows.append(
                    {
                        "selector": "middle",
                        "component": component,
                        "probe": "ontology_collapse",
                        "n_k_rows": 2,
                        "k_values": "12 16",
                        "hodge_coexact_mean": 0.89,
                        "hodge_exact_mean": 0.11,
                        "hodge_harmonic_max": 0.0,
                        "next_token_delta_mean": 0.3 if component == "coexact" else -0.1,
                        "semantic_margin_delta_mean": 0.0,
                        "probe_label_margin_delta_mean": 0.1 if component == "coexact" else 0.8,
                        "probe_positive_prob_delta_mean": 0.02,
                        "closed_loop_sources": 1,
                        "closed_loop_branch_specific_gate_rate_mean": 0.6 if component == "coexact" else 0.0,
                        "closed_loop_branch_specific_gate_rate_max": 0.6 if component == "coexact" else 0.0,
                        "closed_loop_target_margin_delta_minus_random_mean": 0.25 if component == "coexact" else -0.35,
                        "closed_loop_target_margin_delta_minus_random_max": 0.25 if component == "coexact" else -0.35,
                        "role_label": "combined_closed_loop" if component == "coexact" else "stabilization",
                    }
                )
            family_branch_rows = []
            for family in ["literal_stable", "ontology_collapse"]:
                for probe in ["identity_stress", "ontology_collapse"]:
                    for layer in [5, 7]:
                        for component in ["coexact", "presence"]:
                            family_branch_rows.append(
                                {
                                    "family": family,
                                    "k": 16,
                                    "layer": layer,
                                    "selector": "middle",
                                    "component": component,
                                    "probe": probe,
                                    "hodge_coexact": 0.89,
                                    "hodge_exact": 0.11,
                                    "hodge_harmonic": 0.0,
                                    "hodge_real_minus_shuffle_coexact": 0.08,
                                    "hodge_real_minus_random_coexact": 0.16,
                                    "next_token_delta": 0.25 if component == "coexact" else -0.1,
                                    "semantic_margin_delta": 0.05 if component == "coexact" else -0.05,
                                    "probe_label_margin_delta": 0.1 if component == "coexact" else 0.8,
                                    "probe_positive_prob_delta": 0.02,
                                }
                            )
            diagnostic_rows = []
            layer_condition_rows = []
            layer_transition_rows = []
            condition_rows = []
            candidate_rows = []
            for family in ["literal_stable", "ontology_collapse"]:
                for probe in ["identity_stress", "ontology_collapse"]:
                    for component in ["coexact", "presence"]:
                        diagnostic_rows.append(
                            {
                                "selector": "middle",
                                "family": family,
                                "probe": probe,
                                "component": component,
                                "expected_role": "traversal" if component == "coexact" else "stabilization",
                                "observed_role": "traversal" if component == "coexact" else "combined",
                                "role_score": 1.0 if component == "coexact" else 0.5,
                                "role_pass": 1 if component == "coexact" else 0,
                                "criteria_passed": "next_positive" if component == "coexact" else "probe_positive",
                                "criteria_failed": "" if component == "coexact" else "next_not_positive",
                                "n_layer_rows": 2,
                                "layer_values": "5 7",
                                "next_token_delta_mean": 0.25 if component == "coexact" else 0.1,
                                "probe_label_margin_delta_mean": 0.1 if component == "coexact" else 0.8,
                            }
                        )
                for layer in [5, 7]:
                    for component in ["coexact", "presence"]:
                        layer_condition_rows.append(
                            {
                                "selector": "middle",
                                "family": family,
                                "component": component,
                                "layer": layer,
                                "expected_role": "traversal" if component == "coexact" else "stabilization",
                                "condition_label": "stable_expected" if component == "coexact" else "mixed_condition",
                                "n_probe_cells": 2,
                                "role_pass_count": 2 if component == "coexact" else 1,
                                "role_pass_rate": 1.0 if component == "coexact" else 0.5,
                                "mean_role_score": 1.0 if component == "coexact" else 0.75,
                                "observed_role_counts": "traversal:2" if component == "coexact" else "combined:1 stabilization:1",
                                "failed_criteria_counts": "" if component == "coexact" else "next_not_positive:1",
                                "mean_next_token_delta": 0.25 if component == "coexact" else 0.1,
                                "mean_probe_label_margin_delta": 0.1 if component == "coexact" else 0.8,
                                "mean_hodge_coexact": 0.89,
                                "mean_hodge_exact": 0.11,
                            }
                        )
                for component in ["coexact", "presence"]:
                    layer_transition_rows.append(
                        {
                            "selector": "middle",
                            "family": family,
                            "component": component,
                            "expected_role": "traversal" if component == "coexact" else "stabilization",
                            "transition_label": "all_layer_stable" if component == "coexact" else "sparse_stability",
                            "n_layers": 2,
                            "stable_layer_count": 2 if component == "coexact" else 1,
                            "stable_layer_rate": 1.0 if component == "coexact" else 0.5,
                            "stable_layers": "L5-L7" if component == "coexact" else "L5",
                            "mostly_or_mixed_layers": "" if component == "coexact" else "L7",
                            "break_layers": "",
                            "first_stable_layer": "L5",
                            "last_stable_layer": "L7" if component == "coexact" else "L5",
                            "longest_stable_run": 2 if component == "coexact" else 1,
                            "longest_stable_span": "L5-L7" if component == "coexact" else "L5",
                            "mean_layer_pass_rate": 1.0 if component == "coexact" else 0.5,
                            "mean_layer_role_score": 1.0 if component == "coexact" else 0.75,
                            "failed_criteria_counts": "" if component == "coexact" else "next_not_positive:1",
                        }
                    )
                    condition_rows.append(
                        {
                            "selector": "middle",
                            "family": family,
                            "component": component,
                            "expected_role": "traversal" if component == "coexact" else "stabilization",
                            "condition_label": "stable_expected" if component == "coexact" else "mixed_condition",
                            "n_probe_cells": 2,
                            "role_pass_count": 2 if component == "coexact" else 1,
                            "role_pass_rate": 1.0 if component == "coexact" else 0.5,
                            "mean_role_score": 1.0 if component == "coexact" else 0.75,
                            "observed_role_counts": "traversal:2" if component == "coexact" else "combined:1 stabilization:1",
                            "failed_criteria_counts": "" if component == "coexact" else "next_not_positive:1",
                            "mean_next_token_delta": 0.25 if component == "coexact" else 0.1,
                            "mean_probe_label_margin_delta": 0.1 if component == "coexact" else 0.8,
                            "mean_hodge_coexact": 0.89,
                        }
                    )
                    candidate_rows.append(
                        {
                            "selector": "middle",
                            "family": family,
                            "component": component,
                            "expected_role": "traversal" if component == "coexact" else "stabilization",
                            "candidate_label": "causal_band_ready" if component == "coexact" else "narrow_layer_probe",
                            "priority_score": 0.9 if component == "coexact" else 0.45,
                            "structural_support": 1.0 if component == "coexact" else 0.6,
                            "causal_support": 0.5 if component == "coexact" else 0.0,
                            "recommended_layers": "L5-L7" if component == "coexact" else "L5",
                            "transition_label": "all_layer_stable" if component == "coexact" else "sparse_stability",
                            "stable_layer_rate": 1.0 if component == "coexact" else 0.5,
                            "mean_layer_pass_rate": 1.0 if component == "coexact" else 0.5,
                            "mean_layer_role_score": 1.0 if component == "coexact" else 0.75,
                            "stable_layers": "L5-L7" if component == "coexact" else "L5",
                            "mostly_or_mixed_layers": "" if component == "coexact" else "L7",
                            "break_layers": "",
                            "longest_stable_span": "L5-L7" if component == "coexact" else "L5",
                            "closed_loop_sources": 1 if component == "coexact" else 0,
                            "closed_loop_matched_random_rows": 10 if component == "coexact" else 0,
                            "closed_loop_branch_specific_gate_rate_mean": 0.6 if component == "coexact" else "",
                            "closed_loop_branch_gate_rate_mean": 0.8 if component == "coexact" else "",
                            "closed_loop_token_drift_rate_mean": 0.75 if component == "coexact" else "",
                            "closed_loop_target_margin_delta_minus_random_mean": 0.25 if component == "coexact" else "",
                        }
                    )
            prompt_rows = []
            for prompt_id in ["ontology_01", "ontology_02"]:
                for component in ["coexact", "presence"]:
                    prompt_rows.append(
                        {
                            "source": "closed_loop_seed_probe",
                            "family": "ontology_collapse",
                            "prompt_id": prompt_id,
                            "component": component,
                            "branch_specific_gate_rate": 0.8 if component == "coexact" else 0.0,
                            "mean_target_margin_delta_minus_random_mean": (
                                0.25 if component == "coexact" else -0.35
                            ),
                        }
                    )
            reverse_rows = [
                {
                    "panel_label": "identity_04 L7/k16",
                    "panel_order": 0,
                    "target_set": "identity_stress",
                    "source_order": 0,
                    "component": "negative_coexact",
                    "branch_specific_gate_rate": 1.0,
                    "random_branch_gate_rate": 0.0,
                    "mean_target_margin_delta_minus_random_mean": 0.3,
                    "token_drift_rate_mean": 0.5,
                },
                {
                    "panel_label": "identity_04 L7/k16",
                    "panel_order": 0,
                    "target_set": "identity_generic_control",
                    "source_order": 1,
                    "component": "negative_coexact",
                    "branch_specific_gate_rate": 0.0,
                    "random_branch_gate_rate": 0.2,
                    "mean_target_margin_delta_minus_random_mean": -0.4,
                    "token_drift_rate_mean": 0.5,
                },
            ]
            write_csv(root / "hodge_layer.csv", hodge_layer_rows)
            write_csv(root / "hodge_k_sweep.csv", hodge_k_rows)
            write_csv(root / "hodge_topology_family_k.csv", topology_rows)
            write_csv(root / "causal_k_scoreboard.csv", causal_rows)
            write_csv(root / "closed_loop_branch_scoreboard.csv", closed_loop_rows)
            write_csv(root / "branch_role_summary.csv", role_rows)
            write_csv(root / "family_branch_join.csv", family_branch_rows)
            write_csv(root / "branch_role_diagnostics.csv", diagnostic_rows)
            write_csv(root / "branch_layer_condition_summary.csv", layer_condition_rows)
            write_csv(root / "branch_layer_transition_summary.csv", layer_transition_rows)
            write_csv(root / "branch_condition_summary.csv", condition_rows)
            write_csv(root / "branch_band_candidate_scoreboard.csv", candidate_rows)
            write_csv(root / "closed_loop_prompt_join.csv", prompt_rows)
            write_csv(root / "reverse_exception_specificity.csv", reverse_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                probe="ontology_collapse",
                selector="middle",
                components=["coexact", "presence"],
            )

            self.assertEqual(len(saved), 17)
            self.assertTrue((root / "plots" / "closed_loop_branch_specific_scoreboard.png").exists())
            self.assertTrue((root / "plots" / "branch_role_summary.png").exists())
            self.assertTrue((root / "plots" / "branch_role_matrix.png").exists())
            self.assertTrue((root / "plots" / "family_branch_atlas.png").exists())
            self.assertTrue((root / "plots" / "branch_role_diagnostics.png").exists())
            self.assertTrue((root / "plots" / "branch_layer_condition_summary.png").exists())
            self.assertTrue((root / "plots" / "branch_layer_transition_summary.png").exists())
            self.assertTrue((root / "plots" / "branch_condition_summary.png").exists())
            self.assertTrue((root / "plots" / "branch_band_candidate_scoreboard.png").exists())
            self.assertTrue((root / "plots" / "closed_loop_prompt_branch_heatmap.png").exists())
            self.assertTrue((root / "plots" / "reverse_exception_specificity.png").exists())
            for path in saved:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)


if __name__ == "__main__":
    unittest.main()
