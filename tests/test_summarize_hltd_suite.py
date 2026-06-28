from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_hltd_suite as summarize


FIELDS = [
    "variant",
    "layer",
    "hltd_coexact_ratio",
    "hltd_exact_ratio",
    "hltd_harmonic_ratio",
    "hltd_semantic_flow_ratio",
    "graph_high_freq_ratio",
    "hodge_curl_ratio",
    "trajectory_signed_circulation_alignment",
    "hltd_same_graph_reverse_coexact_ratio_gap",
    "hltd_same_graph_reverse_semantic_flow_ratio_gap",
]


def write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer, real, shuffle, reverse, random_hidden, signed in [
        (5, 0.8, 0.6, 0.8, 0.5, 0.3),
        (6, 0.9, 0.7, 0.9, 0.5, 0.4),
    ]:
        rows.extend(
            [
                ["real", layer, real, 1.0 - real, 0.0, real, 0.1, 0.2, signed, 0.001, 0.002],
                ["shuffle_tokens", layer, shuffle, 1.0 - shuffle, 0.0, shuffle, 0.2, 0.3, -signed, 0.003, 0.004],
                ["reverse_tokens", layer, reverse, 1.0 - reverse, 0.0, reverse, 0.1, 0.2, -signed, 0.001, 0.002],
                ["random_hidden", layer, random_hidden, 1.0 - random_hidden, 0.0, random_hidden, 0.3, 0.4, signed, 0.005, 0.006],
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)


class TestSummarizeHLTDSuite(unittest.TestCase):
    def test_parse_run_dir_supports_optional_topology_suffix(self) -> None:
        self.assertEqual(
            summarize.parse_run_dir(Path("literal_stable__literal_01__k16")),
            ("literal_stable", "literal_01", 16, "triangles"),
        )
        self.assertEqual(
            summarize.parse_run_dir(Path("literal_stable__literal_01__k16__no_triangles")),
            ("literal_stable", "literal_01", 16, "no_triangles"),
        )

    def test_builds_k_layer_and_bootstrap_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "ontology_collapse__ontology_01__k12__no_triangles" / "layer_metrics.csv"
            write_metrics(csv_path)

            run_summary = summarize.build_run_summary(csv_path)
            self.assertEqual(run_summary["topology"], "no_triangles")
            self.assertAlmostEqual(run_summary["real_coexact_l5_l8"], 0.85)
            self.assertAlmostEqual(run_summary["real_minus_shuffle_coexact_l5_l8"], 0.2)
            self.assertAlmostEqual(run_summary["real_minus_random_coexact_l5_l8"], 0.35)
            self.assertAlmostEqual(run_summary["max_reverse_hltd_coexact_gap"], 0.0)
            self.assertAlmostEqual(run_summary["max_reverse_signed_trajectory_gap"], 0.0)
            self.assertAlmostEqual(run_summary["max_same_graph_reverse_coexact_gap"], 0.001)

            family_k = summarize.build_family_k_summary([run_summary])
            self.assertEqual(family_k[0]["k"], 12)
            self.assertAlmostEqual(family_k[0]["real_coexact_l5_l8_mean"], 0.85)

            layer_rows = summarize.build_layer_summary([csv_path])
            layer5 = [row for row in layer_rows if row["layer"] == 5][0]
            self.assertAlmostEqual(layer5["real_minus_shuffle_hltd_coexact_ratio"], 0.2)
            self.assertAlmostEqual(layer5["real_minus_reverse_abs_hltd_coexact_ratio"], 0.0)

            prompt_rows = summarize.build_prompt_summary([run_summary])
            bootstrap = summarize.build_bootstrap_summary(prompt_rows, samples=10, seed=0)
            self.assertTrue(any(row["metric"] == "real_coexact_l5_l8" for row in bootstrap))
            self.assertTrue(any(row["metric"] == "max_same_graph_reverse_coexact_gap" for row in bootstrap))

    def test_main_writes_extra_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "literal_stable__literal_01__k16" / "layer_metrics.csv"
            write_metrics(csv_path)
            output = root / "summary.csv"

            rc = summarize.main(
                [
                    "--run-root",
                    str(root),
                    "--output",
                    str(output),
                    "--bootstrap-samples",
                    "10",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(output.exists())
            self.assertTrue((root / "summary_family_k.csv").exists())
            self.assertTrue((root / "summary_layer.csv").exists())
            self.assertTrue((root / "summary_bootstrap.csv").exists())
            self.assertTrue((root / "summary_family_gaps.csv").exists())
            self.assertTrue((root / "summary_report.md").exists())


if __name__ == "__main__":
    unittest.main()
