from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import plot_hltd_closed_loop as plots


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestHLTDClosedLoopPlots(unittest.TestCase):
    def test_available_components_honors_requested_filter(self) -> None:
        table = plots.pd.DataFrame({"component": ["coexact", "presence", "random_tangent"]})

        self.assertEqual(plots.available_components(table, ["coexact"]), ["coexact"])

    def test_build_plots_writes_pngs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "closed"
            component_rows = [
                {
                    "component": "presence_plus_coexact",
                    "alpha": 1.0,
                    "n_rows": 1,
                    "token_drift_rate_mean": 0.0,
                    "baseline_token_overlap_mean": 1.0,
                    "mean_selected_base_logprob_mean": -0.7,
                    "mean_selected_base_logprob_delta_mean": 0.0,
                    "mean_selected_logprob_gain_mean": -0.1,
                    "mean_kl_base_to_steered_mean": 0.05,
                    "mean_entropy_delta_mean": 0.2,
                    "mean_target_margin_delta_mean": -0.2,
                    "mean_nearest_distance_mean": 0.2,
                    "top_changed_rate_mean": 0.0,
                },
                {
                    "component": "coexact_minus_presence",
                    "alpha": 1.0,
                    "n_rows": 1,
                    "token_drift_rate_mean": 1.0,
                    "baseline_token_overlap_mean": 0.0,
                    "mean_selected_base_logprob_mean": -2.0,
                    "mean_selected_base_logprob_delta_mean": -1.3,
                    "mean_selected_logprob_gain_mean": 0.2,
                    "mean_kl_base_to_steered_mean": 0.13,
                    "mean_entropy_delta_mean": 0.2,
                    "mean_target_margin_delta_mean": 0.6,
                    "mean_nearest_distance_mean": 0.25,
                    "top_changed_rate_mean": 0.5,
                },
            ]
            contrast_rows = [
                {
                    "family": "ontology_collapse",
                    "prompt_id": "ontology_05",
                    "layer": 7,
                    "k": 16,
                    "seed": 0,
                    "component": row["component"],
                    "alpha": 1.0,
                    "generated_steps": 2,
                    "baseline_token_overlap": row["baseline_token_overlap_mean"],
                    "token_drift_rate": row["token_drift_rate_mean"],
                    "mean_selected_base_logprob": row["mean_selected_base_logprob_mean"],
                    "mean_selected_base_logprob_delta": row["mean_selected_base_logprob_delta_mean"],
                    "mean_selected_logprob_gain": row["mean_selected_logprob_gain_mean"],
                    "mean_kl_base_to_steered": row["mean_kl_base_to_steered_mean"],
                    "mean_entropy_delta": row["mean_entropy_delta_mean"],
                    "mean_target_margin_delta": row["mean_target_margin_delta_mean"],
                    "mean_nearest_distance": row["mean_nearest_distance_mean"],
                    "unique_nodes": 1,
                    "top_changed_rate": row["top_changed_rate_mean"],
                    "baseline_generated_text": "\\n\\n",
                    "generated_text": " The moon" if row["component"] == "coexact_minus_presence" else "\\n\\n",
                }
                for row in component_rows
            ]
            write_csv(root / "closed_loop_component_summary.csv", component_rows)
            write_csv(root / "closed_loop_contrasts.csv", contrast_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                components=["presence_plus_coexact", "coexact_minus_presence"],
            )

            self.assertEqual(len(saved), 4)
            for path in saved:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_build_plots_can_combine_alpha_sweeps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "closed"
            broad = Path(tmp) / "broad"
            narrow_rows = [
                {
                    "component": "coexact_minus_presence",
                    "alpha": 0.7,
                    "n_rows": 1,
                    "token_drift_rate_mean": 0.0,
                    "baseline_token_overlap_mean": 1.0,
                    "mean_selected_base_logprob_mean": -0.8,
                    "mean_selected_base_logprob_delta_mean": 0.0,
                    "mean_selected_logprob_gain_mean": -0.18,
                    "mean_kl_base_to_steered_mean": 0.03,
                    "mean_entropy_delta_mean": 0.18,
                    "mean_target_margin_delta_mean": 0.44,
                    "mean_nearest_distance_mean": 0.23,
                    "top_changed_rate_mean": 0.0,
                },
                {
                    "component": "coexact_minus_presence",
                    "alpha": 0.8,
                    "n_rows": 1,
                    "token_drift_rate_mean": 1.0,
                    "baseline_token_overlap_mean": 0.0,
                    "mean_selected_base_logprob_mean": -2.0,
                    "mean_selected_base_logprob_delta_mean": -1.2,
                    "mean_selected_logprob_gain_mean": 0.2,
                    "mean_kl_base_to_steered_mean": 0.06,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.5,
                    "mean_nearest_distance_mean": 0.25,
                    "top_changed_rate_mean": 0.5,
                },
            ]
            broad_rows = [
                {
                    "component": "coexact_minus_presence",
                    "alpha": 0.5,
                    "n_rows": 1,
                    "token_drift_rate_mean": 0.0,
                    "baseline_token_overlap_mean": 1.0,
                    "mean_selected_base_logprob_mean": -0.8,
                    "mean_selected_base_logprob_delta_mean": 0.0,
                    "mean_selected_logprob_gain_mean": -0.1,
                    "mean_kl_base_to_steered_mean": 0.01,
                    "mean_entropy_delta_mean": 0.1,
                    "mean_target_margin_delta_mean": 0.16,
                    "mean_nearest_distance_mean": 0.23,
                    "top_changed_rate_mean": 0.0,
                },
            ]
            contrast_rows = [
                {
                    "family": "ontology_collapse",
                    "prompt_id": "ontology_05",
                    "layer": 7,
                    "k": 16,
                    "seed": 0,
                    "component": row["component"],
                    "alpha": row["alpha"],
                    "generated_steps": 2,
                    "baseline_token_overlap": row["baseline_token_overlap_mean"],
                    "token_drift_rate": row["token_drift_rate_mean"],
                    "mean_selected_base_logprob": row["mean_selected_base_logprob_mean"],
                    "mean_selected_base_logprob_delta": row["mean_selected_base_logprob_delta_mean"],
                    "mean_selected_logprob_gain": row["mean_selected_logprob_gain_mean"],
                    "mean_kl_base_to_steered": row["mean_kl_base_to_steered_mean"],
                    "mean_entropy_delta": row["mean_entropy_delta_mean"],
                    "mean_target_margin_delta": row["mean_target_margin_delta_mean"],
                    "mean_nearest_distance": row["mean_nearest_distance_mean"],
                    "unique_nodes": 1,
                    "top_changed_rate": row["top_changed_rate_mean"],
                    "baseline_generated_text": "\\n\\n",
                    "generated_text": " The moon" if float(row["alpha"]) >= 0.8 else "\\n\\n",
                }
                for row in narrow_rows
            ]
            write_csv(root / "closed_loop_component_summary.csv", narrow_rows)
            write_csv(root / "closed_loop_contrasts.csv", contrast_rows)
            write_csv(broad / "closed_loop_component_summary.csv", broad_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                components=["coexact_minus_presence"],
                comparison_summary_roots=[broad],
            )

            self.assertEqual(len(saved), 5)
            self.assertTrue((root / "plots" / "closed_loop_alpha_transition.png").exists())
            for path in saved:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_build_plots_writes_step_trace_when_steps_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "closed"
            component_rows = [
                {
                    "component": "coexact_minus_presence",
                    "alpha": 0.8,
                    "n_rows": 1,
                    "token_drift_rate_mean": 1.0,
                    "baseline_token_overlap_mean": 0.0,
                    "mean_selected_base_logprob_mean": -2.0,
                    "mean_selected_base_logprob_delta_mean": -1.2,
                    "mean_selected_logprob_gain_mean": 0.2,
                    "mean_kl_base_to_steered_mean": 0.06,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.5,
                    "mean_nearest_distance_mean": 0.25,
                    "top_changed_rate_mean": 0.5,
                },
            ]
            contrast_rows = [
                {
                    "family": "ontology_collapse",
                    "prompt_id": "ontology_05",
                    "layer": 7,
                    "k": 16,
                    "seed": 0,
                    "component": "coexact_minus_presence",
                    "alpha": 0.8,
                    "generated_steps": 2,
                    "baseline_token_overlap": 0.0,
                    "token_drift_rate": 1.0,
                    "mean_selected_base_logprob": -2.0,
                    "mean_selected_base_logprob_delta": -1.2,
                    "mean_selected_logprob_gain": 0.2,
                    "mean_kl_base_to_steered": 0.06,
                    "mean_entropy_delta": 0.0,
                    "mean_target_margin_delta": 0.5,
                    "mean_nearest_distance": 0.25,
                    "unique_nodes": 2,
                    "top_changed_rate": 0.5,
                    "baseline_generated_text": "\\n\\n",
                    "generated_text": " The moon",
                },
            ]
            step_rows = [
                {
                    "prompt_id": "ontology_05",
                    "family": "ontology_collapse",
                    "step": step,
                    "prefix_len": 29 + step,
                    "prompt_len": 29,
                    "layer": 7,
                    "k": 16,
                    "seed": 0,
                    "component": component,
                    "alpha": alpha,
                    "node_index": 11 - step,
                    "nearest_distance": 0.2 + step * 0.1,
                    "component_active": 1,
                    "delta_norm": 0.0 if component == "baseline" else 33.0,
                    "natural_step_norm": 42.0,
                    "chart_norm": 0.0 if component == "baseline" else 0.5,
                    "hidden_direction_norm": 0.0 if component == "baseline" else 0.5,
                    "base_entropy": 4.0,
                    "steered_entropy": 4.1,
                    "entropy_delta": 0.0 if component == "baseline" else 0.1,
                    "kl_base_to_steered": 0.0 if component == "baseline" else 0.04 + step * 0.02,
                    "base_top_token": "\\n",
                    "base_top_logprob": -1.0,
                    "steered_top_token": " The" if component != "baseline" and step == 0 else "\\n",
                    "steered_top_logprob": -0.9,
                    "top_changed": 1 if component != "baseline" and step == 0 else 0,
                    "next_token_id": 383,
                    "next_token": " The" if step == 0 else " moon",
                    "next_token_base_logprob": -2.0,
                    "next_token_steered_logprob": -1.8,
                    "next_token_logprob_gain": 0.2,
                    "target_set": "ontology_collapse",
                    "target_logprob_mass_base": -12.0,
                    "target_logprob_mass_steered": -11.5,
                    "target_prob_mass_base": 1e-6,
                    "target_prob_mass_steered": 2e-6,
                    "control_logprob_mass_base": -13.0,
                    "control_logprob_mass_steered": -12.8,
                    "control_prob_mass_base": 1e-6,
                    "control_prob_mass_steered": 1e-6,
                    "target_margin_base": 1.0,
                    "target_margin_steered": 1.4,
                    "target_margin_delta": 0.0 if component == "baseline" else 0.4 + step * 0.1,
                }
                for component, alpha in [("baseline", 0.0), ("coexact_minus_presence", 0.8)]
                for step in range(2)
            ]
            write_csv(root / "closed_loop_component_summary.csv", component_rows)
            write_csv(root / "closed_loop_contrasts.csv", contrast_rows)
            write_csv(root / "closed_loop_steps.csv", step_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                components=["coexact_minus_presence"],
            )

            self.assertEqual(len(saved), 5)
            self.assertTrue((root / "plots" / "closed_loop_step_traces.png").exists())
            for path in saved:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_build_plots_writes_layer_response_when_layer_summary_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "closed"
            component_rows = [
                {
                    "component": "coexact_minus_presence",
                    "alpha": 0.8,
                    "n_rows": 2,
                    "token_drift_rate_mean": 0.75,
                    "baseline_token_overlap_mean": 0.25,
                    "mean_selected_base_logprob_mean": -2.0,
                    "mean_selected_base_logprob_delta_mean": -1.0,
                    "mean_selected_logprob_gain_mean": 0.2,
                    "mean_kl_base_to_steered_mean": 0.07,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.4,
                    "mean_nearest_distance_mean": 0.3,
                    "top_changed_rate_mean": 0.5,
                },
            ]
            contrast_rows = [
                {
                    "family": "ontology_collapse",
                    "prompt_id": "ontology_05",
                    "layer": layer,
                    "k": 16,
                    "seed": 0,
                    "component": "coexact_minus_presence",
                    "alpha": 0.8,
                    "generated_steps": 2,
                    "baseline_token_overlap": 0.0 if layer == 7 else 0.5,
                    "token_drift_rate": 1.0 if layer == 7 else 0.5,
                    "mean_selected_base_logprob": -2.0,
                    "mean_selected_base_logprob_delta": -1.0,
                    "mean_selected_logprob_gain": 0.2,
                    "mean_kl_base_to_steered": 0.07,
                    "mean_entropy_delta": 0.0,
                    "mean_target_margin_delta": 0.6 if layer == 7 else 0.2,
                    "mean_nearest_distance": 0.3,
                    "unique_nodes": 2,
                    "top_changed_rate": 0.5,
                    "baseline_generated_text": "\\n\\n",
                    "generated_text": " The moon",
                }
                for layer in [5, 7]
            ]
            layer_rows = [
                {
                    "layer": layer,
                    "component": "coexact_minus_presence",
                    "alpha": 0.8,
                    "n_rows": 1,
                    "token_drift_rate_mean": 1.0 if layer == 7 else 0.5,
                    "baseline_token_overlap_mean": 0.0 if layer == 7 else 0.5,
                    "mean_selected_base_logprob_mean": -2.0,
                    "mean_selected_base_logprob_delta_mean": -1.0,
                    "mean_selected_logprob_gain_mean": 0.2,
                    "mean_kl_base_to_steered_mean": 0.07,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.6 if layer == 7 else 0.2,
                    "mean_nearest_distance_mean": 0.3,
                    "top_changed_rate_mean": 0.5,
                }
                for layer in [5, 7]
            ]
            write_csv(root / "closed_loop_component_summary.csv", component_rows)
            write_csv(root / "closed_loop_contrasts.csv", contrast_rows)
            write_csv(root / "closed_loop_layer_summary.csv", layer_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                components=["coexact_minus_presence"],
            )

            self.assertEqual(len(saved), 5)
            self.assertTrue((root / "plots" / "closed_loop_layer_response.png").exists())
            for path in saved:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_build_plots_writes_k_response_when_k_summary_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "closed"
            component_rows = [
                {
                    "component": "coexact_minus_presence",
                    "alpha": 0.8,
                    "n_rows": 2,
                    "token_drift_rate_mean": 0.75,
                    "baseline_token_overlap_mean": 0.25,
                    "mean_selected_base_logprob_mean": -2.0,
                    "mean_selected_base_logprob_delta_mean": -1.0,
                    "mean_selected_logprob_gain_mean": 0.2,
                    "mean_kl_base_to_steered_mean": 0.07,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.4,
                    "mean_nearest_distance_mean": 0.3,
                    "top_changed_rate_mean": 0.5,
                },
            ]
            contrast_rows = [
                {
                    "family": "ontology_collapse",
                    "prompt_id": "ontology_05",
                    "layer": 7,
                    "k": k,
                    "seed": 0,
                    "component": "coexact_minus_presence",
                    "alpha": 0.8,
                    "generated_steps": 2,
                    "baseline_token_overlap": 0.0 if k == 24 else 0.5,
                    "token_drift_rate": 1.0 if k == 24 else 0.5,
                    "mean_selected_base_logprob": -2.0,
                    "mean_selected_base_logprob_delta": -1.0,
                    "mean_selected_logprob_gain": 0.2,
                    "mean_kl_base_to_steered": 0.07,
                    "mean_entropy_delta": 0.0,
                    "mean_target_margin_delta": 0.6 if k == 24 else 0.2,
                    "mean_nearest_distance": 0.3,
                    "unique_nodes": 2,
                    "top_changed_rate": 0.5,
                    "baseline_generated_text": "\\n\\n",
                    "generated_text": " The moon",
                }
                for k in [12, 24]
            ]
            k_rows = [
                {
                    "k": k,
                    "component": "coexact_minus_presence",
                    "alpha": 0.8,
                    "n_rows": 1,
                    "token_drift_rate_mean": 1.0 if k == 24 else 0.5,
                    "baseline_token_overlap_mean": 0.0 if k == 24 else 0.5,
                    "mean_selected_base_logprob_mean": -2.0,
                    "mean_selected_base_logprob_delta_mean": -1.0,
                    "mean_selected_logprob_gain_mean": 0.2,
                    "mean_kl_base_to_steered_mean": 0.07,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.6 if k == 24 else 0.2,
                    "mean_nearest_distance_mean": 0.3,
                    "top_changed_rate_mean": 0.5,
                }
                for k in [12, 24]
            ]
            write_csv(root / "closed_loop_component_summary.csv", component_rows)
            write_csv(root / "closed_loop_contrasts.csv", contrast_rows)
            write_csv(root / "closed_loop_k_summary.csv", k_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                components=["coexact_minus_presence"],
            )

            self.assertEqual(len(saved), 5)
            self.assertTrue((root / "plots" / "closed_loop_k_response.png").exists())
            for path in saved:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_build_plots_writes_alpha_k_threshold_when_grid_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "closed"
            component_rows = [
                {
                    "component": "coexact_minus_presence",
                    "alpha": alpha,
                    "n_rows": 2,
                    "token_drift_rate_mean": 0.5 if alpha == 0.7 else 1.0,
                    "baseline_token_overlap_mean": 0.5 if alpha == 0.7 else 0.0,
                    "mean_selected_base_logprob_mean": -2.0,
                    "mean_selected_base_logprob_delta_mean": -1.0,
                    "mean_selected_logprob_gain_mean": 0.05 if alpha == 0.7 else 0.2,
                    "mean_kl_base_to_steered_mean": 0.04 if alpha == 0.7 else 0.08,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.3 if alpha == 0.7 else 0.6,
                    "mean_nearest_distance_mean": 0.3,
                    "top_changed_rate_mean": 0.25,
                }
                for alpha in [0.7, 0.8]
            ]
            contrast_rows = [
                {
                    "family": "ontology_collapse",
                    "prompt_id": "ontology_05",
                    "layer": 7,
                    "k": k,
                    "seed": 0,
                    "component": "coexact_minus_presence",
                    "alpha": alpha,
                    "generated_steps": 2,
                    "baseline_token_overlap": 0.0 if alpha == 0.8 else 0.5,
                    "token_drift_rate": 1.0 if alpha == 0.8 else 0.5,
                    "mean_selected_base_logprob": -2.0,
                    "mean_selected_base_logprob_delta": -1.0,
                    "mean_selected_logprob_gain": 0.2 if alpha == 0.8 else 0.05,
                    "mean_kl_base_to_steered": 0.08 if alpha == 0.8 else 0.04,
                    "mean_entropy_delta": 0.0,
                    "mean_target_margin_delta": 0.6 if k == 24 else 0.3,
                    "mean_nearest_distance": 0.3,
                    "unique_nodes": 2,
                    "top_changed_rate": 0.5,
                    "baseline_generated_text": "\\n\\n",
                    "generated_text": " The moon",
                }
                for k in [12, 24]
                for alpha in [0.7, 0.8]
            ]
            k_rows = [
                {
                    "k": k,
                    "component": "coexact_minus_presence",
                    "alpha": alpha,
                    "n_rows": 1,
                    "token_drift_rate_mean": 1.0 if alpha == 0.8 or k == 24 else 0.5,
                    "baseline_token_overlap_mean": 0.0 if alpha == 0.8 or k == 24 else 0.5,
                    "mean_selected_base_logprob_mean": -2.0,
                    "mean_selected_base_logprob_delta_mean": -1.0,
                    "mean_selected_logprob_gain_mean": 0.1 * alpha,
                    "mean_kl_base_to_steered_mean": 0.04 + 0.02 * alpha,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.2 + 0.01 * k + 0.1 * alpha,
                    "mean_nearest_distance_mean": 0.3,
                    "top_changed_rate_mean": 0.5,
                }
                for k in [12, 24]
                for alpha in [0.7, 0.8]
            ]
            write_csv(root / "closed_loop_component_summary.csv", component_rows)
            write_csv(root / "closed_loop_contrasts.csv", contrast_rows)
            write_csv(root / "closed_loop_k_summary.csv", k_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                components=["coexact_minus_presence"],
            )

            self.assertEqual(len(saved), 6)
            self.assertTrue((root / "plots" / "closed_loop_alpha_k_threshold.png").exists())
            for path in saved:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_build_plots_writes_alpha_k_branch_map_when_multi_component_grid_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "closed"
            components = ["coexact_minus_presence", "presence"]
            component_rows = [
                {
                    "component": component,
                    "alpha": alpha,
                    "n_rows": 2,
                    "token_drift_rate_mean": 0.75 if component == "coexact_minus_presence" else 0.0,
                    "baseline_token_overlap_mean": 0.25 if component == "coexact_minus_presence" else 1.0,
                    "mean_selected_base_logprob_mean": -2.0,
                    "mean_selected_base_logprob_delta_mean": -1.0,
                    "mean_selected_logprob_gain_mean": 0.1 if component == "coexact_minus_presence" else 0.2,
                    "mean_kl_base_to_steered_mean": 0.08,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.4 if component == "coexact_minus_presence" else -0.3,
                    "mean_nearest_distance_mean": 0.3,
                    "top_changed_rate_mean": 0.25,
                }
                for component in components
                for alpha in [0.7, 0.8]
            ]
            contrast_rows = [
                {
                    "family": "ontology_collapse",
                    "prompt_id": "ontology_05",
                    "layer": 7,
                    "k": k,
                    "seed": 0,
                    "component": component,
                    "alpha": alpha,
                    "generated_steps": 2,
                    "baseline_token_overlap": 0.0 if component == "coexact_minus_presence" else 1.0,
                    "token_drift_rate": 1.0 if component == "coexact_minus_presence" and alpha == 0.8 else 0.0,
                    "mean_selected_base_logprob": -2.0,
                    "mean_selected_base_logprob_delta": -1.0,
                    "mean_selected_logprob_gain": 0.2,
                    "mean_kl_base_to_steered": 0.08,
                    "mean_entropy_delta": 0.0,
                    "mean_target_margin_delta": 0.5 if component == "coexact_minus_presence" else -0.3,
                    "mean_nearest_distance": 0.3,
                    "unique_nodes": 2,
                    "top_changed_rate": 0.5,
                    "baseline_generated_text": "\\n\\n",
                    "generated_text": " The moon" if component == "coexact_minus_presence" else "\\n\\n",
                }
                for k in [12, 24]
                for alpha in [0.7, 0.8]
                for component in components
            ]
            k_rows = [
                {
                    "k": k,
                    "component": component,
                    "alpha": alpha,
                    "n_rows": 1,
                    "token_drift_rate_mean": (
                        1.0 if component == "coexact_minus_presence" and (k == 24 or alpha == 0.8) else 0.0
                    ),
                    "baseline_token_overlap_mean": (
                        0.0 if component == "coexact_minus_presence" and (k == 24 or alpha == 0.8) else 1.0
                    ),
                    "mean_selected_base_logprob_mean": -2.0,
                    "mean_selected_base_logprob_delta_mean": -1.0,
                    "mean_selected_logprob_gain_mean": 0.1 if component == "coexact_minus_presence" else 0.2,
                    "mean_kl_base_to_steered_mean": 0.04 + 0.02 * alpha,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.5 if component == "coexact_minus_presence" else -0.3,
                    "mean_nearest_distance_mean": 0.3,
                    "top_changed_rate_mean": 0.5,
                }
                for k in [12, 24]
                for alpha in [0.7, 0.8]
                for component in components
            ]
            write_csv(root / "closed_loop_component_summary.csv", component_rows)
            write_csv(root / "closed_loop_contrasts.csv", contrast_rows)
            write_csv(root / "closed_loop_k_summary.csv", k_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                components=components,
            )

            self.assertEqual(len(saved), 7)
            self.assertTrue((root / "plots" / "closed_loop_alpha_k_branch_map.png").exists())
            for path in saved:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_build_plots_writes_prompt_branch_gate_when_prompt_summary_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "closed"
            components = ["coexact_minus_presence", "presence"]
            component_rows = [
                {
                    "component": component,
                    "alpha": 0.8,
                    "n_rows": 2,
                    "token_drift_rate_mean": 0.75 if component == "coexact_minus_presence" else 0.0,
                    "baseline_token_overlap_mean": 0.25 if component == "coexact_minus_presence" else 1.0,
                    "mean_selected_base_logprob_mean": -2.0,
                    "mean_selected_base_logprob_delta_mean": -1.0,
                    "mean_selected_logprob_gain_mean": 0.1 if component == "coexact_minus_presence" else 0.2,
                    "mean_kl_base_to_steered_mean": 0.08,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.4 if component == "coexact_minus_presence" else -0.3,
                    "mean_nearest_distance_mean": 0.3,
                    "top_changed_rate_mean": 0.25,
                }
                for component in components
            ]
            contrast_rows = [
                {
                    "family": "ontology_collapse",
                    "prompt_id": prompt_id,
                    "layer": 7,
                    "k": 16,
                    "seed": 0,
                    "component": component,
                    "alpha": 0.8,
                    "generated_steps": 2,
                    "baseline_token_overlap": 0.0 if component == "coexact_minus_presence" else 1.0,
                    "token_drift_rate": 1.0 if component == "coexact_minus_presence" else 0.0,
                    "mean_selected_base_logprob": -2.0,
                    "mean_selected_base_logprob_delta": -1.0,
                    "mean_selected_logprob_gain": 0.2,
                    "mean_kl_base_to_steered": 0.08,
                    "mean_entropy_delta": 0.0,
                    "mean_target_margin_delta": 0.5 if component == "coexact_minus_presence" else -0.3,
                    "mean_nearest_distance": 0.3,
                    "unique_nodes": 2,
                    "top_changed_rate": 0.5,
                    "baseline_generated_text": "\\n\\n",
                    "generated_text": " The moon" if component == "coexact_minus_presence" else "\\n\\n",
                }
                for prompt_id in ["ontology_03", "ontology_05"]
                for component in components
            ]
            prompt_rows = [
                {
                    "family": "ontology_collapse",
                    "prompt_id": prompt_id,
                    "component": component,
                    "alpha": 0.8,
                    "n_rows": 1,
                    "branch_gate_rate": 1.0 if component == "coexact_minus_presence" else 0.0,
                    "branch_specific_gate_rate": 1.0 if component == "coexact_minus_presence" else 0.0,
                    "target_positive_rate": 1.0 if component == "coexact_minus_presence" else 0.0,
                    "drift_ge50_rate": 1.0 if component == "coexact_minus_presence" else 0.0,
                    "random_branch_gate_rate": 0.0,
                    "branch_gate_minus_random_rate": 1.0 if component == "coexact_minus_presence" else 0.0,
                    "token_drift_rate_mean": 1.0 if component == "coexact_minus_presence" else 0.0,
                    "token_drift_rate_minus_random_mean": 1.0 if component == "coexact_minus_presence" else 0.0,
                    "baseline_token_overlap_mean": 0.0 if component == "coexact_minus_presence" else 1.0,
                    "mean_selected_logprob_gain_mean": 0.2,
                    "mean_kl_base_to_steered_mean": 0.08,
                    "mean_kl_base_to_steered_minus_random_mean": 0.02 if component == "coexact_minus_presence" else 0.0,
                    "mean_target_margin_delta_mean": 0.5 if component == "coexact_minus_presence" else -0.3,
                    "mean_target_margin_delta_minus_random_mean": 0.5 if component == "coexact_minus_presence" else -0.3,
                    "mean_nearest_distance_mean": 0.3,
                }
                for prompt_id in ["ontology_03", "ontology_05"]
                for component in components
            ]
            write_csv(root / "closed_loop_component_summary.csv", component_rows)
            write_csv(root / "closed_loop_contrasts.csv", contrast_rows)
            write_csv(root / "closed_loop_prompt_summary.csv", prompt_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                components=components,
            )

            self.assertEqual(len(saved), 6)
            self.assertTrue((root / "plots" / "closed_loop_prompt_branch_gate.png").exists())
            self.assertTrue((root / "plots" / "closed_loop_prompt_random_advantage.png").exists())
            for path in saved:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_build_plots_writes_prompt_layer_k_map_when_summary_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "closed"
            components = ["negative_coexact", "random_tangent"]
            component_rows = [
                {
                    "component": component,
                    "alpha": 0.8,
                    "n_rows": 4,
                    "token_drift_rate_mean": 0.5 if component == "negative_coexact" else 0.0,
                    "baseline_token_overlap_mean": 0.5 if component == "negative_coexact" else 1.0,
                    "mean_selected_base_logprob_mean": -1.5,
                    "mean_selected_base_logprob_delta_mean": -0.2,
                    "mean_selected_logprob_gain_mean": 0.1,
                    "mean_kl_base_to_steered_mean": 0.04,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.3 if component == "negative_coexact" else 0.0,
                    "mean_nearest_distance_mean": 0.3,
                    "top_changed_rate_mean": 0.25,
                }
                for component in components
            ]
            contrast_rows = [
                {
                    "family": "identity_stress",
                    "prompt_id": "identity_04",
                    "layer": layer,
                    "k": k,
                    "seed": 0,
                    "component": component,
                    "alpha": 0.8,
                    "generated_steps": 2,
                    "baseline_token_overlap": 0.5 if component == "negative_coexact" else 1.0,
                    "token_drift_rate": 0.5 if component == "negative_coexact" else 0.0,
                    "mean_selected_base_logprob": -1.5,
                    "mean_selected_base_logprob_delta": -0.2,
                    "mean_selected_logprob_gain": 0.1,
                    "mean_kl_base_to_steered": 0.04,
                    "mean_entropy_delta": 0.0,
                    "mean_target_margin_delta": 0.3 if component == "negative_coexact" else 0.0,
                    "mean_nearest_distance": 0.3,
                    "unique_nodes": 2,
                    "top_changed_rate": 0.25,
                    "baseline_generated_text": "\\n\\n",
                    "generated_text": " The mask" if component == "negative_coexact" else "\\n\\n",
                }
                for layer in [5, 7]
                for k in [12, 16]
                for component in components
            ]
            prompt_layer_k_rows = [
                {
                    "family": "identity_stress",
                    "prompt_id": "identity_04",
                    "layer": layer,
                    "k": k,
                    "component": component,
                    "alpha": 0.8,
                    "n_rows": 1,
                    "matched_random_rows": 1,
                    "branch_gate_rate": 1.0 if component == "negative_coexact" and k == 16 else 0.0,
                    "branch_specific_gate_rate": 1.0 if component == "negative_coexact" and k == 16 else 0.0,
                    "target_positive_rate": 1.0 if component == "negative_coexact" else 0.0,
                    "drift_ge50_rate": 1.0 if component == "negative_coexact" else 0.0,
                    "random_branch_gate_rate": 0.0,
                    "branch_gate_minus_random_rate": 1.0 if component == "negative_coexact" and k == 16 else 0.0,
                    "token_drift_rate_mean": 0.5 if component == "negative_coexact" else 0.0,
                    "token_drift_rate_minus_random_mean": 0.5 if component == "negative_coexact" else 0.0,
                    "mean_target_margin_delta_mean": 0.3 if component == "negative_coexact" else 0.0,
                    "mean_target_margin_delta_minus_random_mean": (
                        0.3 if component == "negative_coexact" and k == 16 else -0.1
                    ),
                    "mean_nearest_distance_mean": 0.3,
                }
                for layer in [5, 7]
                for k in [12, 16]
                for component in components
            ]
            write_csv(root / "closed_loop_component_summary.csv", component_rows)
            write_csv(root / "closed_loop_contrasts.csv", contrast_rows)
            write_csv(root / "closed_loop_prompt_layer_k_summary.csv", prompt_layer_k_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                components=components,
            )

            self.assertEqual(len(saved), 5)
            self.assertTrue((root / "plots" / "closed_loop_prompt_layer_k_map.png").exists())
            for path in saved:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_build_plots_writes_prompt_layer_map_for_fixed_k_layer_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "closed"
            components = ["coexact", "random_tangent"]
            component_rows = [
                {
                    "component": component,
                    "alpha": 0.8,
                    "n_rows": 2,
                    "token_drift_rate_mean": 0.5 if component == "coexact" else 0.0,
                    "baseline_token_overlap_mean": 0.5 if component == "coexact" else 1.0,
                    "mean_selected_base_logprob_mean": -1.5,
                    "mean_selected_base_logprob_delta_mean": -0.2,
                    "mean_selected_logprob_gain_mean": 0.1,
                    "mean_kl_base_to_steered_mean": 0.04,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.3 if component == "coexact" else 0.0,
                    "mean_nearest_distance_mean": 0.3,
                    "top_changed_rate_mean": 0.25,
                }
                for component in components
            ]
            contrast_rows = [
                {
                    "family": "identity_stress",
                    "prompt_id": "identity_02",
                    "layer": layer,
                    "k": 16,
                    "seed": 0,
                    "component": component,
                    "alpha": 0.8,
                    "generated_steps": 2,
                    "baseline_token_overlap": 0.5 if component == "coexact" else 1.0,
                    "token_drift_rate": 0.5 if component == "coexact" else 0.0,
                    "mean_selected_base_logprob": -1.5,
                    "mean_selected_base_logprob_delta": -0.2,
                    "mean_selected_logprob_gain": 0.1,
                    "mean_kl_base_to_steered": 0.04,
                    "mean_entropy_delta": 0.0,
                    "mean_target_margin_delta": 0.3 if component == "coexact" else 0.0,
                    "mean_nearest_distance": 0.3,
                    "unique_nodes": 2,
                    "top_changed_rate": 0.25,
                    "baseline_generated_text": "\\n\\n",
                    "generated_text": " The mirror" if component == "coexact" else "\\n\\n",
                }
                for layer in [4, 7]
                for component in components
            ]
            prompt_layer_k_rows = [
                {
                    "family": "identity_stress",
                    "prompt_id": "identity_02",
                    "layer": layer,
                    "k": 16,
                    "component": component,
                    "alpha": 0.8,
                    "n_rows": 1,
                    "matched_random_rows": 1,
                    "branch_gate_rate": 1.0 if component == "coexact" else 0.0,
                    "branch_specific_gate_rate": 1.0 if component == "coexact" else 0.0,
                    "target_positive_rate": 1.0 if component == "coexact" else 0.0,
                    "drift_ge50_rate": 1.0 if component == "coexact" else 0.0,
                    "random_branch_gate_rate": 0.0,
                    "branch_gate_minus_random_rate": 1.0 if component == "coexact" else 0.0,
                    "token_drift_rate_mean": 0.5 if component == "coexact" else 0.0,
                    "token_drift_rate_minus_random_mean": 0.5 if component == "coexact" else 0.0,
                    "mean_target_margin_delta_mean": 0.3 if component == "coexact" else 0.0,
                    "mean_target_margin_delta_minus_random_mean": 0.3 if component == "coexact" else 0.0,
                    "mean_nearest_distance_mean": 0.3,
                }
                for layer in [4, 7]
                for component in components
            ]
            write_csv(root / "closed_loop_component_summary.csv", component_rows)
            write_csv(root / "closed_loop_contrasts.csv", contrast_rows)
            write_csv(root / "closed_loop_prompt_layer_k_summary.csv", prompt_layer_k_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                components=components,
            )

            self.assertTrue((root / "plots" / "closed_loop_prompt_layer_k_map.png").exists())
            self.assertIn(root / "plots" / "closed_loop_prompt_layer_k_map.png", saved)
            for path in saved:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_build_plots_writes_prompt_layer_alpha_k_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "closed"
            components = ["coexact", "random_tangent"]
            component_rows = [
                {
                    "component": component,
                    "alpha": alpha,
                    "n_rows": 4,
                    "token_drift_rate_mean": 0.5 if component == "coexact" else 0.25,
                    "baseline_token_overlap_mean": 0.5,
                    "mean_selected_base_logprob_mean": -1.5,
                    "mean_selected_base_logprob_delta_mean": -0.2,
                    "mean_selected_logprob_gain_mean": 0.1,
                    "mean_kl_base_to_steered_mean": 0.04,
                    "mean_entropy_delta_mean": 0.0,
                    "mean_target_margin_delta_mean": 0.3 if component == "coexact" else 0.0,
                    "mean_nearest_distance_mean": 0.3,
                    "top_changed_rate_mean": 0.25,
                }
                for component in components
                for alpha in [0.4, 0.8]
            ]
            contrast_rows = [
                {
                    "family": "identity_stress",
                    "prompt_id": "identity_02",
                    "layer": layer,
                    "k": k,
                    "seed": 0,
                    "component": component,
                    "alpha": alpha,
                    "generated_steps": 2,
                    "baseline_token_overlap": 0.5,
                    "token_drift_rate": 0.5 if component == "coexact" else 0.25,
                    "mean_selected_base_logprob": -1.5,
                    "mean_selected_base_logprob_delta": -0.2,
                    "mean_selected_logprob_gain": 0.1,
                    "mean_kl_base_to_steered": 0.04,
                    "mean_entropy_delta": 0.0,
                    "mean_target_margin_delta": 0.3 if component == "coexact" else 0.0,
                    "mean_nearest_distance": 0.3,
                    "unique_nodes": 2,
                    "top_changed_rate": 0.25,
                    "baseline_generated_text": "base",
                    "generated_text": "steered",
                }
                for layer in [4, 7]
                for k in [12, 16]
                for alpha in [0.4, 0.8]
                for component in components
            ]
            prompt_layer_k_rows = [
                {
                    "family": "identity_stress",
                    "prompt_id": "identity_02",
                    "layer": layer,
                    "k": k,
                    "component": component,
                    "alpha": alpha,
                    "n_rows": 2,
                    "matched_random_rows": 2,
                    "branch_gate_rate": 1.0 if component == "coexact" else 0.0,
                    "branch_specific_gate_rate": 1.0 if component == "coexact" else 0.0,
                    "target_positive_rate": 1.0 if component == "coexact" else 0.0,
                    "drift_ge50_rate": 1.0 if component == "coexact" else 0.0,
                    "random_branch_gate_rate": 0.0,
                    "branch_gate_minus_random_rate": 1.0 if component == "coexact" else 0.0,
                    "token_drift_rate_mean": 0.5 if component == "coexact" else 0.25,
                    "token_drift_rate_minus_random_mean": 0.25 if component == "coexact" else 0.0,
                    "mean_target_margin_delta_mean": 0.3 if component == "coexact" else 0.0,
                    "mean_target_margin_delta_minus_random_mean": 0.3 if component == "coexact" else 0.0,
                    "mean_nearest_distance_mean": 0.3,
                }
                for layer in [4, 7]
                for k in [12, 16]
                for alpha in [0.4, 0.8]
                for component in components
            ]
            write_csv(root / "closed_loop_component_summary.csv", component_rows)
            write_csv(root / "closed_loop_contrasts.csv", contrast_rows)
            write_csv(root / "closed_loop_prompt_layer_k_summary.csv", prompt_layer_k_rows)

            saved = plots.build_plots(
                summary_root=root,
                output_dir=root / "plots",
                components=components,
            )

            surface = root / "plots" / "closed_loop_prompt_layer_alpha_k_surface.png"
            self.assertTrue(surface.exists())
            self.assertIn(surface, saved)
            self.assertFalse((root / "plots" / "closed_loop_prompt_layer_k_map.png").exists())


if __name__ == "__main__":
    unittest.main()
