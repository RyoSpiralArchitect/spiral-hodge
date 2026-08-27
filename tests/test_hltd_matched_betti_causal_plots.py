from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.plot_hltd_matched_betti_causal import (
    bootstrap_response_coefficients,
    bootstrap_signed_prompt_contrasts,
    bootstrap_prompt_gaps,
    collapse_prompt_gaps,
    matched_component_gaps,
    fit_prompt_response_coefficients,
    render_all,
    signed_prompt_contrasts,
)


def synthetic_rows() -> pd.DataFrame:
    rows = []
    branch_offsets = {
        "exact": (0.10, 0.02, -0.01),
        "coexact": (0.20, -0.03, 0.04),
        "harmonic": (-0.05, 0.05, 0.08),
        "random_tangent": (0.0, 0.0, 0.0),
    }
    for family, prompt_id, prompt_shift in [
        ("literal_stable", "literal_01", 0.0),
        ("ontology_collapse", "ontology_01", 0.02),
    ]:
        for seed in [0, 1]:
            for component, offsets in branch_offsets.items():
                rows.append(
                    {
                        "family": family,
                        "prompt_id": prompt_id,
                        "layer": 5,
                        "k": 16,
                        "complex_mode": "matched_betti",
                        "hodge_solver": "orthogonal",
                        "betti_1_fraction_target": 0.5,
                        "betti_1_fraction": 0.5,
                        "betti_1_fraction_abs_error": 0.0,
                        "cycle_rank": 20,
                        "triangle_rank": 10,
                        "hodge_exact_ratio": 0.2,
                        "hodge_coexact_ratio": 0.5,
                        "hodge_harmonic_ratio": 0.3,
                        "seed": seed,
                        "random_tangent_reference": "max_full_branch_node_speed",
                        "token_selector": "middle",
                        "selector_component": "coexact",
                        "node_index": 7,
                        "token_index": 8,
                        "component": component,
                        "alpha": 0.5,
                        "component_active": 1,
                        "kl_base_to_steered": 0.3 + offsets[0] + prompt_shift,
                        "next_token_logprob_delta": -0.1 + offsets[1] + prompt_shift,
                        "semantic_margin_delta": 0.01 + offsets[2] + prompt_shift,
                    }
                )
    return pd.DataFrame(rows)


def synthetic_signed_rows() -> pd.DataFrame:
    rows = []
    effects = {
        "exact": ((0.10, 0.02), (-0.04, 0.01), (0.03, -0.02)),
        "coexact": ((-0.06, 0.03), (0.08, -0.01), (0.12, 0.04)),
        "harmonic": ((0.02, -0.01), (0.01, 0.02), (-0.05, 0.01)),
        "random_tangent": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    }
    metric_names = tuple(
        [
            "kl_base_to_steered",
            "next_token_logprob_delta",
            "semantic_margin_delta",
        ]
    )
    metric_bases = (0.3, -0.1, 0.01)
    for family, prompt_id, prompt_shift in [
        ("literal_stable", "literal_01", 0.0),
        ("ontology_collapse", "ontology_01", 0.02),
    ]:
        for seed in [0, 1]:
            for alpha in [-0.5, -0.25, 0.25, 0.5]:
                for component, component_effects in effects.items():
                    row = {
                        "family": family,
                        "prompt_id": prompt_id,
                        "layer": 5,
                        "k": 16,
                        "complex_mode": "matched_betti",
                        "hodge_solver": "orthogonal",
                        "betti_1_fraction_target": 0.5,
                        "betti_1_fraction": 0.5,
                        "betti_1_fraction_abs_error": 0.0,
                        "cycle_rank": 20,
                        "triangle_rank": 10,
                        "hodge_exact_ratio": 0.2,
                        "hodge_coexact_ratio": 0.5,
                        "hodge_harmonic_ratio": 0.3,
                        "seed": seed,
                        "random_tangent_reference": "max_full_branch_node_speed",
                        "token_selector": "middle",
                        "selector_component": "coexact",
                        "node_index": 7,
                        "token_index": 8,
                        "component": component,
                        "alpha": alpha,
                        "component_active": 1,
                    }
                    for metric, base, (odd_slope, even_curvature) in zip(
                        metric_names,
                        metric_bases,
                        component_effects,
                    ):
                        row[metric] = (
                            base
                            + prompt_shift
                            + alpha * odd_slope
                            + alpha**2 * even_curvature
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


class TestHLTDMatchedBettiCausalPlots(unittest.TestCase):
    def test_prompt_bootstrap_does_not_count_null_seeds_as_prompts(self) -> None:
        gaps = matched_component_gaps(synthetic_rows())
        prompt_rows = collapse_prompt_gaps(gaps)
        inference = bootstrap_prompt_gaps(prompt_rows, n_bootstrap=200, seed=3)

        exact_kl = inference[
            (inference["component"] == "exact")
            & (inference["metric"] == "kl_base_to_steered")
        ].iloc[0]
        self.assertEqual(int(exact_kl["n_prompts"]), 2)
        self.assertAlmostEqual(float(exact_kl["mean_gap"]), 0.10)
        exact_prompt_rows = prompt_rows[
            (prompt_rows["component"] == "exact")
            & (prompt_rows["metric"] == "kl_base_to_steered")
        ]
        self.assertEqual(set(exact_prompt_rows["n_null_seeds"]), {2})

    def test_signed_contrasts_pair_after_seed_collapse(self) -> None:
        gaps = matched_component_gaps(synthetic_signed_rows())
        prompt_rows = collapse_prompt_gaps(gaps)
        signed_rows = signed_prompt_contrasts(prompt_rows)
        inference = bootstrap_signed_prompt_contrasts(
            signed_rows,
            n_bootstrap=200,
            seed=3,
        )

        exact_kl = inference[
            (inference["component"] == "exact")
            & (inference["metric"] == "kl_base_to_steered")
            & (inference["alpha_abs"] == 0.5)
        ]
        odd = exact_kl[exact_kl["contrast_type"] == "odd"].iloc[0]
        even = exact_kl[exact_kl["contrast_type"] == "even"].iloc[0]
        self.assertEqual(int(odd["n_prompts"]), 2)
        self.assertAlmostEqual(float(odd["mean_contrast"]), 0.05)
        self.assertAlmostEqual(float(even["mean_contrast"]), 0.005)
        exact_prompt_rows = signed_rows[
            (signed_rows["component"] == "exact")
            & (signed_rows["metric"] == "kl_base_to_steered")
            & (signed_rows["contrast_type"] == "odd")
        ]
        self.assertEqual(set(exact_prompt_rows["n_null_seeds_positive"]), {2})
        self.assertEqual(set(exact_prompt_rows["n_null_seeds_negative"]), {2})

        prompt_coefficients = fit_prompt_response_coefficients(signed_rows)
        response = bootstrap_response_coefficients(
            prompt_coefficients,
            n_bootstrap=200,
            seed=3,
        )
        exact_kl_response = response[
            (response["component"] == "exact")
            & (response["metric"] == "kl_base_to_steered")
        ]
        odd_response = exact_kl_response[
            exact_kl_response["contrast_type"] == "odd"
        ].iloc[0]
        even_response = exact_kl_response[
            exact_kl_response["contrast_type"] == "even"
        ].iloc[0]
        self.assertAlmostEqual(float(odd_response["mean_response_coefficient"]), 0.10)
        self.assertAlmostEqual(float(even_response["mean_response_coefficient"]), 0.02)
        self.assertAlmostEqual(float(odd_response["mean_relative_response_rmse"]), 0.0)

    def test_signed_contrasts_reject_incomplete_pairs(self) -> None:
        gaps = matched_component_gaps(synthetic_signed_rows())
        prompt_rows = collapse_prompt_gaps(gaps)
        incomplete = prompt_rows[~(
            (prompt_rows["prompt_id"] == "literal_01")
            & (prompt_rows["component"] == "coexact")
            & (prompt_rows["alpha"] < 0.0)
        )]
        with self.assertRaisesRegex(ValueError, "missing a \\+alpha or -alpha"):
            signed_prompt_contrasts(incomplete)

    def test_signed_contrasts_reject_unequal_repeat_sets(self) -> None:
        rows = synthetic_signed_rows()
        incomplete = rows[
            ~(
                (rows["prompt_id"] == "literal_01")
                & (rows["component"] == "coexact")
                & (rows["alpha"] == -0.5)
                & (rows["seed"] == 1)
            )
        ]
        gaps = matched_component_gaps(incomplete)
        prompt_rows = collapse_prompt_gaps(gaps)
        with self.assertRaisesRegex(ValueError, "repeat sets differ"):
            signed_prompt_contrasts(prompt_rows)

    def test_response_fit_rejects_missing_complete_magnitude(self) -> None:
        rows = synthetic_signed_rows()
        incomplete = rows[
            ~(
                (rows["prompt_id"] == "literal_01")
                & (rows["component"] == "coexact")
                & (rows["alpha"].abs() == 0.25)
            )
        ]
        gaps = matched_component_gaps(incomplete)
        prompt_rows = collapse_prompt_gaps(gaps)
        signed_rows = signed_prompt_contrasts(prompt_rows)
        with self.assertRaisesRegex(ValueError, "missing alpha magnitudes"):
            fit_prompt_response_coefficients(
                signed_rows,
                require_complete=True,
            )

    def test_render_all_writes_plot_tables_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.csv"
            synthetic_rows().to_csv(summary, index=False)

            render_all(
                summary_path=summary,
                output_root=root / "causal",
                n_bootstrap=200,
                seed=3,
            )

            output = root / "causal"
            self.assertTrue((output / "plots" / "matched_betti_causal_branch_gaps.png").exists())
            self.assertGreater(
                (output / "plots" / "matched_betti_causal_branch_gaps.png").stat().st_size,
                1000,
            )
            self.assertTrue((output / "summary_branch_minus_random_pairs.csv").exists())
            self.assertTrue((output / "summary_prompt_branch_gaps.csv").exists())
            self.assertTrue((output / "summary_prompt_bootstrap.csv").exists())
            self.assertTrue((output / "summary_causal_report.md").exists())

    def test_require_signed_rejects_one_sided_run_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.csv"
            synthetic_rows().to_csv(summary, index=False)
            output = root / "causal"

            with self.assertRaisesRegex(ValueError, "requires both positive and negative alpha"):
                render_all(
                    summary_path=summary,
                    output_root=output,
                    n_bootstrap=200,
                    seed=3,
                    require_signed=True,
                )
            self.assertFalse(output.exists())

    def test_expected_design_rejects_partial_suite_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.csv"
            synthetic_signed_rows().to_csv(summary, index=False)
            output = root / "causal"

            with self.assertRaisesRegex(ValueError, "expected 3 prompts, found 2"):
                render_all(
                    summary_path=summary,
                    output_root=output,
                    n_bootstrap=200,
                    seed=3,
                    require_signed=True,
                    expected_prompts=3,
                    expected_null_seeds=2,
                    expected_alpha_magnitudes=2,
                )
            self.assertFalse(output.exists())

    def test_render_all_writes_signed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.csv"
            synthetic_signed_rows().to_csv(summary, index=False)

            render_all(
                summary_path=summary,
                output_root=root / "causal",
                n_bootstrap=200,
                seed=3,
                require_signed=True,
            )

            output = root / "causal"
            signed_plot = output / "plots" / "matched_betti_signed_causal_odd_even.png"
            response_plot = (
                output / "plots" / "matched_betti_signed_response_coefficients.png"
            )
            self.assertTrue(signed_plot.exists())
            self.assertGreater(signed_plot.stat().st_size, 1000)
            self.assertTrue(response_plot.exists())
            self.assertGreater(response_plot.stat().st_size, 1000)
            self.assertTrue((output / "summary_signed_prompt_contrasts.csv").exists())
            self.assertTrue((output / "summary_signed_prompt_bootstrap.csv").exists())
            self.assertTrue(
                (output / "summary_signed_prompt_response_coefficients.csv").exists()
            )
            self.assertTrue(
                (output / "summary_signed_response_coefficient_bootstrap.csv").exists()
            )
            self.assertTrue((output / "summary_signed_causal_report.md").exists())


if __name__ == "__main__":
    unittest.main()
