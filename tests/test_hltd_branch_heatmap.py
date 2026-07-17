from __future__ import annotations

import unittest

from scripts import run_hltd_branch_heatmap as heatmap


class TestHLTDBranchHeatmap(unittest.TestCase):
    def test_position_bin_clips_edges(self) -> None:
        self.assertEqual(heatmap.position_bin(-0.5, 4), 0)
        self.assertEqual(heatmap.position_bin(0.0, 4), 0)
        self.assertEqual(heatmap.position_bin(0.26, 4), 1)
        self.assertEqual(heatmap.position_bin(1.0, 4), 3)
        self.assertEqual(heatmap.position_bin(2.0, 4), 3)

    def test_summarize_position_and_peaks(self) -> None:
        rows = [
            {
                "family": "literal_stable",
                "layer": 5,
                "k": 16,
                "position_bin": 1,
                "position_frac": 0.2,
                "component": "coexact",
                "component_norm": 2.0,
                "component_to_full": 0.5,
                "component_base_share": 0.6,
            },
            {
                "family": "literal_stable",
                "layer": 5,
                "k": 16,
                "position_bin": 1,
                "position_frac": 0.25,
                "component": "coexact",
                "component_norm": 4.0,
                "component_to_full": 0.7,
                "component_base_share": 0.8,
            },
            {
                "family": "literal_stable",
                "layer": 5,
                "k": 16,
                "position_bin": 2,
                "position_frac": 0.5,
                "component": "coexact",
                "component_norm": 3.0,
                "component_to_full": 0.9,
                "component_base_share": 0.4,
            },
        ]

        position_rows = heatmap.summarize_by_position(rows)
        by_bin = {row["position_bin"]: row for row in position_rows}

        self.assertEqual(by_bin[1]["n_nodes"], 2)
        self.assertAlmostEqual(by_bin[1]["component_norm_mean"], 3.0)
        self.assertAlmostEqual(by_bin[1]["component_to_full_mean"], 0.6)
        self.assertAlmostEqual(by_bin[1]["component_base_share_mean"], 0.7)

        [peak] = heatmap.summarize_peaks(position_rows)

        self.assertEqual(peak["peak_position_bin"], 2)
        self.assertAlmostEqual(peak["peak_component_to_full_mean"], 0.9)

    def test_global_peak_summary_groups_by_k_and_component(self) -> None:
        rows = [
            {
                "family": "a",
                "layer": 5,
                "k": 16,
                "component": "presence",
                "peak_position_bin": 2,
                "peak_position_frac_mean": 0.25,
                "peak_component_to_full_mean": 1.0,
            },
            {
                "family": "b",
                "layer": 6,
                "k": 16,
                "component": "presence",
                "peak_position_bin": 4,
                "peak_position_frac_mean": 0.5,
                "peak_component_to_full_mean": 2.0,
            },
        ]

        [summary] = heatmap.summarize_global_peaks(rows)

        self.assertEqual(summary["k"], 16)
        self.assertEqual(summary["component"], "presence")
        self.assertEqual(summary["n_family_layer_rows"], 2)
        self.assertAlmostEqual(summary["peak_position_bin_mean"], 3.0)
        self.assertAlmostEqual(summary["peak_position_frac_mean"], 0.375)
        self.assertAlmostEqual(summary["peak_component_to_full_mean"], 1.5)


if __name__ == "__main__":
    unittest.main()
