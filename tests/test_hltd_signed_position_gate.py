from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.plot_hltd_signed_position_gate import (
    bootstrap_position_responses,
    bootstrap_position_phases,
    collapse_prompt_bin_contrasts,
    collapse_seed_gaps,
    fit_prompt_position_trends,
    fit_prompt_bin_response_coefficients,
    prompt_position_phases,
    position_bin,
    render_all,
    seed_matched_component_gaps,
    signed_token_contrasts,
)


def synthetic_position_rows() -> pd.DataFrame:
    rows = []
    prompt_specs = [
        ("literal_stable", "literal_01", 0.0),
        ("ontology_collapse", "ontology_01", 1.0),
    ]
    token_specs = [(1, 1.0), (2, 3.0), (7, -1.0)]
    for family, prompt_id, prompt_shift in prompt_specs:
        for token_index, token_slope in token_specs:
            for seed, seed_shift in [(0, -0.2), (1, 0.2)]:
                for alpha in [-1.0, -0.5, 0.5, 1.0]:
                    random_base = 0.1 * seed + 0.01 * token_index
                    odd_slope = token_slope + prompt_shift + seed_shift
                    even_curvature = 0.4 + 0.1 * prompt_shift
                    for component in ["coexact", "random_tangent"]:
                        gap = alpha * odd_slope + alpha**2 * even_curvature
                        rows.append(
                            {
                                "family": family,
                                "prompt_id": prompt_id,
                                "layer": 5,
                                "k": 16,
                                "complex_mode": "matched_betti",
                                "betti_1_fraction_target": 0.5,
                                "seed": seed,
                                "random_tangent_reference": "max_full_branch_node_speed",
                                "token_selector": "all_interior",
                                "selector_component": "coexact",
                                "node_index": token_index - 1,
                                "token_index": token_index,
                                "token_count": 10,
                                "component": component,
                                "component_active": 1,
                                "alpha": alpha,
                                "next_token_logprob_delta": random_base + (gap if component == "coexact" else 0.0),
                                "semantic_margin_delta": random_base + (0.5 * gap if component == "coexact" else 0.0),
                            }
                        )
    return pd.DataFrame(rows)


class TestHLTDSignedPositionGate(unittest.TestCase):
    def test_common_baseline_offsets_cancel_before_both_signed_contrasts(self) -> None:
        rows = synthetic_position_rows()
        shifted = rows.copy()
        # A shared single-vs-batch offset may vary with token, seed and sign.
        offset = 0.17 * shifted["token_index"] + 0.03 * shifted["seed"] - 0.08 * shifted["alpha"]
        for metric in ["next_token_logprob_delta", "semantic_margin_delta"]:
            shifted[metric] += offset
        original = signed_token_contrasts(collapse_seed_gaps(seed_matched_component_gaps(rows, bins=4)))
        corrected = signed_token_contrasts(collapse_seed_gaps(seed_matched_component_gaps(shifted, bins=4)))
        columns = [column for column in original if not column.startswith(("component_value_", "baseline_value_"))]
        pd.testing.assert_frame_equal(original[columns], corrected[columns], check_exact=False, atol=1e-12, rtol=1e-12)

    def test_position_bin_clips_endpoint(self) -> None:
        self.assertEqual(position_bin(0.0, 12), 0)
        self.assertEqual(position_bin(0.5, 12), 6)
        self.assertEqual(position_bin(1.0, 12), 11)

    def test_seed_then_sign_then_prompt_bin_contract(self) -> None:
        gaps = seed_matched_component_gaps(synthetic_position_rows(), bins=4)
        seed_rows = collapse_seed_gaps(gaps)
        signed = signed_token_contrasts(seed_rows)
        prompt_bins = collapse_prompt_bin_contrasts(signed)
        coefficients = fit_prompt_bin_response_coefficients(prompt_bins, bins=4)
        inference = bootstrap_position_responses(
            coefficients,
            bins=4,
            n_bootstrap=200,
            seed=3,
        )

        bin_zero = inference[
            (inference["metric"] == "next_token_logprob_delta")
            & (inference["contrast_type"] == "odd")
            & (inference["position_bin"] == 0)
        ].iloc[0]
        self.assertEqual(int(bin_zero["n_prompts"]), 2)
        self.assertEqual(int(bin_zero["n_tokens"]), 4)
        self.assertAlmostEqual(float(bin_zero["mean_response_coefficient"]), 2.5)
        self.assertEqual(set(coefficients["position_bin_center"]), {0.125, 0.875})

        even = inference[
            (inference["metric"] == "next_token_logprob_delta")
            & (inference["contrast_type"] == "even")
            & (inference["position_bin"] == 0)
        ].iloc[0]
        self.assertAlmostEqual(float(even["mean_response_coefficient"]), 0.45)
        self.assertEqual(set(coefficients["min_null_seeds"]), {2})

        trends = fit_prompt_position_trends(coefficients)
        next_odd = trends[
            (trends["metric"] == "next_token_logprob_delta")
            & (trends["contrast_type"] == "odd")
        ]
        self.assertEqual(len(next_odd), 2)
        self.assertTrue((next_odd["position_slope"].round(12) == -4.0).all())

        phases = prompt_position_phases(coefficients)
        phase_inference = bootstrap_position_phases(phases, n_bootstrap=200, seed=3)
        early_late = phase_inference[
            (phase_inference["metric"] == "next_token_logprob_delta")
            & (phase_inference["contrast_type"] == "odd")
            & (phase_inference["position_phase"] == "early_minus_late")
        ].iloc[0]
        self.assertEqual(int(early_late["n_prompts"]), 2)
        self.assertAlmostEqual(float(early_late["mean_response_coefficient"]), 3.0)

    def test_signed_pair_rejects_missing_direction(self) -> None:
        rows = synthetic_position_rows()
        incomplete = rows[
            ~(
                (rows["prompt_id"] == "literal_01")
                & (rows["token_index"] == 1)
                & (rows["component"] == "coexact")
                & (rows["alpha"] == -1.0)
            )
        ]
        gaps = seed_matched_component_gaps(incomplete, bins=4)
        with self.assertRaisesRegex(ValueError, "missing a \\+alpha or -alpha"):
            signed_token_contrasts(collapse_seed_gaps(gaps))

    def test_signed_pair_rejects_different_seed_sets(self) -> None:
        rows = synthetic_position_rows()
        incomplete = rows[
            ~(
                (rows["prompt_id"] == "literal_01")
                & (rows["token_index"] == 1)
                & (rows["component"] == "coexact")
                & (rows["alpha"] == -1.0)
                & (rows["seed"] == 1)
            )
        ]
        gaps = seed_matched_component_gaps(incomplete, bins=4)
        with self.assertRaisesRegex(ValueError, "seed sets differ"):
            signed_token_contrasts(collapse_seed_gaps(gaps))

    def test_render_all_writes_tables_report_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.csv"
            synthetic_position_rows().to_csv(summary, index=False)
            output = root / "position"
            render_all(
                summary_path=summary,
                output_root=output,
                bins=4,
                n_bootstrap=200,
                seed=3,
                expected_prompts=2,
                expected_null_seeds=2,
                expected_alpha_magnitudes=2,
            )
            for name in [
                "signed_coexact_position_profile.png",
                "signed_coexact_family_position_profile.png",
                "signed_coexact_position_coverage.png",
            ]:
                path = output / "plots" / name
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1000)
            self.assertTrue((output / "summary_position_response_bootstrap.csv").exists())
            self.assertTrue((output / "summary_position_coverage.csv").exists())
            self.assertTrue((output / "summary_position_phase_bootstrap.csv").exists())
            self.assertTrue((output / "summary_position_trend_bootstrap.csv").exists())
            self.assertTrue((output / "summary_signed_position_report.md").exists())

    def test_expected_design_rejects_partial_suite_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.csv"
            synthetic_position_rows().to_csv(summary, index=False)
            output = root / "position"
            with self.assertRaisesRegex(ValueError, "expected 3 prompts, found 2"):
                render_all(
                    summary_path=summary,
                    output_root=output,
                    bins=4,
                    n_bootstrap=200,
                    seed=3,
                    expected_prompts=3,
                    expected_null_seeds=2,
                    expected_alpha_magnitudes=2,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
