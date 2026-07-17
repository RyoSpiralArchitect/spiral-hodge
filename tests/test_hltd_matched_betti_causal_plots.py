from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.plot_hltd_matched_betti_causal import (
    bootstrap_prompt_gaps,
    collapse_prompt_gaps,
    matched_component_gaps,
    render_all,
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


if __name__ == "__main__":
    unittest.main()
