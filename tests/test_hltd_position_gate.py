from __future__ import annotations

import unittest

from scripts import summarize_hltd_position_gate as summarize


class TestHLTDPositionGate(unittest.TestCase):
    def test_attach_position_infers_token_count_and_bin(self) -> None:
        row = {
            "family": "literal_stable",
            "prompt_id": "literal_01",
            "layer": 5,
            "k": 16,
            "token_index": 4,
        }
        token_counts = {("literal_stable", "literal_01", 5, 16): 10}

        out = summarize.attach_position(row, token_counts, bins=5)

        self.assertEqual(out["token_count_inferred"], 10)
        self.assertAlmostEqual(out["position_frac"], 4 / 9)
        self.assertEqual(out["position_bin"], 2)

    def test_token_count_lookup_prefers_explicit_token_count(self) -> None:
        rows = [
            {
                "family": "literal_stable",
                "prompt_id": "literal_01",
                "layer": 5,
                "k": 16,
                "token_index": 3,
                "token_count": 64,
            }
        ]

        counts = summarize.token_count_lookup(rows)

        self.assertEqual(counts[("literal_stable", "literal_01", 5, 16)], 64)

    def test_steering_pairwise_rows_subtract_random_tangent_per_token(self) -> None:
        base = {
            "family": "literal_stable",
            "prompt_id": "literal_01",
            "layer": 5,
            "k": 16,
            "seed": "0",
            "token_selector": "all_interior",
            "token_index": 2,
            "token": " map",
            "alpha": "1.0",
            "component_active": 1,
            "kl_base_to_steered": 0.1,
            "entropy_delta": 0.0,
            "next_token_logprob_delta": -0.2,
            "semantic_margin_delta": -0.1,
        }
        rows = [
            {**base, "component": "random_tangent"},
            {
                **base,
                "component": "coexact",
                "kl_base_to_steered": 0.4,
                "next_token_logprob_delta": 0.3,
                "semantic_margin_delta": 0.05,
            },
        ]

        [out] = summarize.steering_pairwise_rows(rows, bins=4)

        self.assertEqual(out["component"], "coexact")
        self.assertEqual(out["position_bin"], 2)
        self.assertAlmostEqual(out["kl_base_to_steered_minus_random_tangent"], 0.3)
        self.assertAlmostEqual(out["next_token_logprob_delta_minus_random_tangent"], 0.5)
        self.assertAlmostEqual(out["semantic_margin_delta_minus_random_tangent"], 0.15)

    def test_summarize_peak_bins_treats_zero_as_valid_peak(self) -> None:
        rows = [
            {
                "family": "literal_stable",
                "layer": 5,
                "k": 16,
                "position_bin": 1,
                "position_frac_mean": 0.25,
                "component": "coexact",
                "probe": "ontology_collapse",
                "next_token_delta_mean": -1.0,
                "probe_label_margin_delta_mean": -0.5,
            },
            {
                "family": "literal_stable",
                "layer": 7,
                "k": 16,
                "position_bin": 2,
                "position_frac_mean": 0.5,
                "component": "coexact",
                "probe": "ontology_collapse",
                "next_token_delta_mean": 0.0,
                "probe_label_margin_delta_mean": 0.0,
            },
        ]

        peaks = summarize.summarize_peak_bins(rows)
        by_metric = {row["metric"]: row for row in peaks}

        self.assertEqual(by_metric["next_token_delta_mean"]["peak_layer"], 7)
        self.assertEqual(by_metric["next_token_delta_mean"]["k"], 16)
        self.assertEqual(by_metric["next_token_delta_mean"]["peak_position_bin"], 2)
        self.assertAlmostEqual(by_metric["next_token_delta_mean"]["peak_value"], 0.0)
        self.assertEqual(by_metric["probe_label_margin_delta_mean"]["peak_layer"], 7)

    def test_summarize_peak_bins_keeps_k_values_separate(self) -> None:
        rows = [
            {
                "family": "literal_stable",
                "layer": 5,
                "k": 16,
                "position_bin": 0,
                "position_frac_mean": 0.05,
                "component": "coexact",
                "probe": "ontology_collapse",
                "next_token_delta_mean": 0.1,
                "probe_label_margin_delta_mean": 0.0,
            },
            {
                "family": "literal_stable",
                "layer": 8,
                "k": 24,
                "position_bin": 9,
                "position_frac_mean": 0.75,
                "component": "coexact",
                "probe": "ontology_collapse",
                "next_token_delta_mean": 9.0,
                "probe_label_margin_delta_mean": 0.0,
            },
        ]

        peaks = summarize.summarize_peak_bins(rows)
        next_peaks = [row for row in peaks if row["metric"] == "next_token_delta_mean"]

        self.assertEqual({row["k"] for row in next_peaks}, {16, 24})
        self.assertEqual({row["peak_position_bin"] for row in next_peaks}, {0, 9})

    def test_summarize_cross_family_peaks_averages_before_peak_by_k(self) -> None:
        rows = [
            {
                "family": "literal_stable",
                "layer": 5,
                "k": 16,
                "position_bin": 0,
                "position_frac_mean": 0.05,
                "component": "coexact",
                "probe": "ontology_collapse",
                "next_token_delta_mean": 2.0,
                "probe_label_margin_delta_mean": 0.0,
            },
            {
                "family": "metaphor_shift",
                "layer": 5,
                "k": 16,
                "position_bin": 0,
                "position_frac_mean": 0.06,
                "component": "coexact",
                "probe": "ontology_collapse",
                "next_token_delta_mean": -1.0,
                "probe_label_margin_delta_mean": 0.0,
            },
            {
                "family": "literal_stable",
                "layer": 7,
                "k": 16,
                "position_bin": 2,
                "position_frac_mean": 0.2,
                "component": "coexact",
                "probe": "ontology_collapse",
                "next_token_delta_mean": 0.4,
                "probe_label_margin_delta_mean": 1.0,
            },
            {
                "family": "metaphor_shift",
                "layer": 7,
                "k": 16,
                "position_bin": 2,
                "position_frac_mean": 0.22,
                "component": "coexact",
                "probe": "ontology_collapse",
                "next_token_delta_mean": 0.6,
                "probe_label_margin_delta_mean": 1.5,
            },
            {
                "family": "literal_stable",
                "layer": 8,
                "k": 24,
                "position_bin": 9,
                "position_frac_mean": 0.8,
                "component": "coexact",
                "probe": "ontology_collapse",
                "next_token_delta_mean": 9.0,
                "probe_label_margin_delta_mean": 9.0,
            },
        ]

        peaks = summarize.summarize_cross_family_peaks(rows)
        by_metric = {(row["metric"], row["k"]): row for row in peaks}

        next_peak = by_metric[("next_token_delta_mean", 16)]
        self.assertEqual(next_peak["peak_layer"], 5)
        self.assertEqual(next_peak["peak_position_bin"], 0)
        self.assertAlmostEqual(next_peak["peak_value_mean"], 0.5)
        self.assertEqual(next_peak["n_families"], 2)

        probe_peak = by_metric[("probe_label_margin_delta_mean", 16)]
        self.assertEqual(probe_peak["peak_layer"], 7)
        self.assertEqual(probe_peak["peak_position_bin"], 2)
        self.assertAlmostEqual(probe_peak["peak_value_mean"], 1.25)

        self.assertEqual(by_metric[("next_token_delta_mean", 24)]["peak_position_bin"], 9)


if __name__ == "__main__":
    unittest.main()
