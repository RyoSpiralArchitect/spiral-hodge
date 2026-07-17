from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.plot_hltd_topology_filtration import (
    matched_betti_real_null_gaps_by_variant,
    matched_betti_real_null_gaps,
    matched_betti_summary,
    matched_real_null_gaps,
    prompt_bootstrap_gap_summary,
    render_all,
)
from scripts.run_hltd_topology_filtration import (
    component_is_active,
    filtration_rows_for_field,
    null_node_vectors,
    parse_radius_scale,
)


class TestHLTDTopologyFiltration(unittest.TestCase):
    def test_radius_parser_accepts_full_endpoint(self) -> None:
        self.assertEqual(parse_radius_scale("0.85"), 0.85)
        self.assertTrue(np.isinf(parse_radius_scale("full")))

    def test_alignment_activity_uses_an_energy_ratio_floor(self) -> None:
        self.assertFalse(
            component_is_active(np.asarray([1e-6, 0.0]), reference_norm=1.0)
        )
        self.assertTrue(
            component_is_active(np.asarray([1e-4, 0.0]), reference_norm=1.0)
        )

    def test_null_vectors_preserve_node_speed(self) -> None:
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )

        random_tangent = null_node_vectors(vectors, variant="random_tangent", seed=4)
        shuffled = null_node_vectors(vectors, variant="vector_shuffle", seed=4)

        np.testing.assert_allclose(
            np.linalg.norm(random_tangent, axis=1),
            np.linalg.norm(vectors, axis=1),
        )
        np.testing.assert_allclose(
            np.sort(np.linalg.norm(shuffled, axis=1)),
            np.sort(np.linalg.norm(vectors, axis=1)),
        )

    def test_matched_gap_averages_null_seeds_and_variants(self) -> None:
        base = {
            "family": "literal_stable",
            "prompt_id": "literal_01",
            "layer": 5,
            "k": 16,
            "triangle_fill_requested": 0.5,
        }
        rows = [
            {**base, "variant": "real", "seed": -1, "exact_ratio": 0.2, "coexact_ratio": 0.7, "harmonic_ratio": 0.1},
            {**base, "variant": "vector_shuffle", "seed": 0, "exact_ratio": 0.3, "coexact_ratio": 0.5, "harmonic_ratio": 0.2},
            {**base, "variant": "random_tangent", "seed": 0, "exact_ratio": 0.1, "coexact_ratio": 0.5, "harmonic_ratio": 0.4},
        ]

        gaps = matched_real_null_gaps(pd.DataFrame(rows))

        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(float(gaps.iloc[0]["exact_gap"]), 0.0)
        self.assertAlmostEqual(float(gaps.iloc[0]["coexact_gap"]), 0.2)
        self.assertAlmostEqual(float(gaps.iloc[0]["harmonic_gap"]), -0.2)

    def test_matched_betti_interpolates_each_field_before_null_matching(self) -> None:
        rows = []
        base = {
            "family": "literal_stable",
            "prompt_id": "literal_01",
            "layer": 5,
            "k": 16,
            "filtration_mode": "radius",
        }
        for variant, seed, offset in [("real", -1, 0.1), ("vector_shuffle", 0, 0.0)]:
            for betti, harmonic in [(1.0, 0.8 + offset), (0.0, 0.0)]:
                rows.append(
                    {
                        **base,
                        "variant": variant,
                        "seed": seed,
                        "betti_1_fraction": betti,
                        "exact_ratio": 0.2,
                        "coexact_ratio": 0.8 - harmonic,
                        "harmonic_ratio": harmonic,
                        "harmonic_survival_ratio": harmonic,
                    }
                )

        summary = matched_betti_summary(pd.DataFrame(rows), targets=[0.5])
        gaps = matched_betti_real_null_gaps(summary)

        self.assertEqual(len(summary), 2)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(float(gaps.iloc[0]["harmonic_gap"]), 0.05)
        self.assertAlmostEqual(float(gaps.iloc[0]["coexact_gap"]), -0.05)

    def test_null_specific_betti_gaps_average_seeds_separately(self) -> None:
        base = {
            "family": "literal_stable",
            "prompt_id": "literal_01",
            "layer": 5,
            "k": 16,
            "betti_1_target": 0.5,
        }
        rows = [
            {**base, "variant": "real", "seed": -1, "exact_ratio": 0.2, "coexact_ratio": 0.5, "harmonic_ratio": 0.3},
            {**base, "variant": "vector_shuffle", "seed": 0, "exact_ratio": 0.4, "coexact_ratio": 0.4, "harmonic_ratio": 0.2},
            {**base, "variant": "vector_shuffle", "seed": 1, "exact_ratio": 0.6, "coexact_ratio": 0.3, "harmonic_ratio": 0.1},
            {**base, "variant": "random_tangent", "seed": 0, "exact_ratio": 0.3, "coexact_ratio": 0.6, "harmonic_ratio": 0.1},
        ]

        gaps = matched_betti_real_null_gaps_by_variant(pd.DataFrame(rows))

        self.assertEqual(set(gaps["null_variant"]), {"vector_shuffle", "random_tangent"})
        shuffled = gaps[gaps["null_variant"] == "vector_shuffle"].iloc[0]
        tangent = gaps[gaps["null_variant"] == "random_tangent"].iloc[0]
        self.assertAlmostEqual(float(shuffled["harmonic_gap"]), 0.15)
        self.assertAlmostEqual(float(tangent["harmonic_gap"]), 0.2)

    def test_prompt_bootstrap_collapses_repeated_fields_before_resampling(self) -> None:
        rows = []
        for prompt_id, gap in [("p1", 0.1), ("p2", 0.3)]:
            for layer in [4, 5]:
                rows.append(
                    {
                        "null_variant": "random_tangent",
                        "betti_1_target": 0.5,
                        "prompt_id": prompt_id,
                        "layer": layer,
                        "exact_gap": -gap,
                        "coexact_gap": 0.0,
                        "harmonic_gap": gap,
                    }
                )

        summary = prompt_bootstrap_gap_summary(
            pd.DataFrame(rows),
            group_columns=["null_variant", "betti_1_target"],
            n_bootstrap=200,
            seed=3,
        )

        harmonic = summary[summary["branch"] == "harmonic"].iloc[0]
        self.assertEqual(int(harmonic["n_prompts"]), 2)
        self.assertAlmostEqual(float(harmonic["mean_gap"]), 0.2)
        self.assertEqual(float(harmonic["positive_prompt_fraction"]), 1.0)

    def test_radius_field_uses_orthogonal_energy_decomposition(self) -> None:
        theta = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
        points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        vectors = np.stack([-np.sin(theta), np.cos(theta)], axis=1)

        rows = filtration_rows_for_field(
            points=points,
            vectors=vectors,
            prompt_id="ring_01",
            family="synthetic",
            layer=5,
            k=4,
            filtration_mode="radius",
            fill_fractions=[0.0, 1.0],
            radius_scales=[0.0, 1.25, np.inf],
            null_variants=["random_tangent"],
            null_seeds=[0],
            ridge=1e-5,
            hodge_solver="orthogonal",
        )

        self.assertEqual(len(rows), 6)
        self.assertEqual({row["filtration_mode"] for row in rows}, {"radius"})
        self.assertEqual({row["hodge_solver"] for row in rows}, {"orthogonal"})
        self.assertTrue(all(row["energy_closure_error"] < 1e-10 for row in rows))
        real = [row for row in rows if row["variant"] == "real"]
        self.assertEqual(real[0]["triangle_rank"], 0.0)
        self.assertTrue(np.isinf(real[-1]["filtration_radius_scale_requested"]))

    def test_render_all_writes_both_plots_and_summaries(self) -> None:
        rows = []
        for family, prompt_id in [("literal_stable", "literal_01"), ("identity_stress", "identity_01")]:
            for fill in [0.0, 0.5, 1.0]:
                for variant, seed, offset in [("real", -1, 0.05), ("vector_shuffle", 0, 0.0)]:
                    rows.append(
                        {
                            "family": family,
                            "prompt_id": prompt_id,
                            "layer": 5,
                            "k": 16,
                            "variant": variant,
                            "seed": seed,
                            "triangle_fill_requested": fill,
                            "exact_ratio": 0.2,
                            "coexact_ratio": 0.75 * fill + offset,
                            "harmonic_ratio": 0.8 * (1.0 - fill) - offset,
                            "betti_1_fraction": 1.0 - fill,
                            "harmonic_survival_ratio": 1.0 - fill,
                        }
                    )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_path = root / "metrics.csv"
            pd.DataFrame(rows).to_csv(metrics_path, index=False)

            render_all(metrics_path=metrics_path, output_root=root)

            self.assertTrue((root / "plots" / "topology_filtration_branch_persistence.png").exists())
            self.assertTrue((root / "plots" / "topology_filtration_family_persistence.png").exists())
            self.assertTrue((root / "plots" / "topology_filtration_matched_betti.png").exists())
            self.assertTrue((root / "plots" / "topology_filtration_prompt_inference.png").exists())
            self.assertTrue((root / "summary_branch_persistence.csv").exists())
            self.assertTrue((root / "summary_real_minus_null.csv").exists())
            self.assertTrue((root / "summary_matched_betti.csv").exists())
            self.assertTrue((root / "summary_matched_betti_gaps.csv").exists())
            self.assertTrue((root / "summary_matched_betti_gaps_by_null.csv").exists())
            self.assertTrue((root / "summary_prompt_bootstrap.csv").exists())
            self.assertTrue((root / "summary_report.md").exists())


if __name__ == "__main__":
    unittest.main()
