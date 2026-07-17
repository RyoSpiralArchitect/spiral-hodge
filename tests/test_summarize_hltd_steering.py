from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_hltd_steering as summarize


FIELDS = [
    "prompt_id",
    "layer",
    "seed",
    "token_selector",
    "selector_component",
    "node_index",
    "token_index",
    "token",
    "next_token",
    "component",
    "alpha",
    "component_active",
    "kl_base_to_steered",
    "js_divergence",
    "entropy_delta",
    "top_changed",
    "top_shift_logprob_delta",
    "next_token_logprob_delta",
    "target_logprob_delta",
    "target_set",
    "target_set_size",
    "control_set_size",
    "target_logprob_mass_delta",
    "target_prob_mass_delta",
    "control_logprob_mass_delta",
    "control_prob_mass_delta",
    "semantic_margin_delta",
    "semantic_prob_margin_delta",
]


def write_steering_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        [
            "ontology_01",
            5,
            0,
            "max_component",
            "coexact",
            2,
            4,
            " road",
            " and",
            "coexact",
            1.0,
            1,
            0.09,
            0.02,
            -0.2,
            1,
            2.0,
            0.4,
            "",
            "ontology_collapse",
            12,
            8,
            0.2,
            0.01,
            -0.1,
            -0.02,
            0.3,
            0.03,
        ],
        [
            "ontology_01",
            5,
            0,
            "max_component",
            "coexact",
            2,
            4,
            " road",
            " and",
            "random_tangent",
            1.0,
            1,
            0.14,
            0.03,
            -0.1,
            1,
            3.0,
            -0.3,
            "",
            "ontology_collapse",
            12,
            8,
            -0.4,
            -0.03,
            0.1,
            0.01,
            -0.5,
            -0.04,
        ],
        [
            "ontology_01",
            5,
            0,
            "max_component",
            "coexact",
            2,
            4,
            " road",
            " and",
            "presence",
            1.0,
            1,
            0.12,
            0.025,
            -0.3,
            1,
            2.5,
            0.1,
            "",
            "ontology_collapse",
            12,
            8,
            0.0,
            0.0,
            0.2,
            0.02,
            -0.2,
            -0.02,
        ],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)


class TestSummarizeHLTDSteering(unittest.TestCase):
    def test_parse_run_dir(self) -> None:
        self.assertEqual(
            summarize.parse_run_dir(Path("ontology_collapse__ontology_01__L5__k16")),
            ("ontology_collapse", "ontology_01", 5, 16),
        )

    def test_builds_component_and_pairwise_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "ontology_collapse__ontology_01__L5__k16" / "steering_metrics.csv"
            write_steering_metrics(csv_path)

            rows = summarize.build_run_rows(csv_path)
            self.assertEqual(rows[0]["family"], "ontology_collapse")
            self.assertEqual(rows[0]["k"], 16)

            component = summarize.build_component_summary(rows)
            coexact = [row for row in component if row["component"] == "coexact"][0]
            self.assertAlmostEqual(coexact["next_token_logprob_delta_mean"], 0.4)
            self.assertAlmostEqual(coexact["semantic_margin_delta_mean"], 0.3)

            pairwise = summarize.build_pairwise_summary(rows)
            coexact_pair = [row for row in pairwise if row["component"] == "coexact"][0]
            self.assertAlmostEqual(
                coexact_pair["next_token_logprob_delta_minus_random_tangent_mean"],
                0.7,
            )
            self.assertAlmostEqual(
                coexact_pair["kl_base_to_steered_minus_random_tangent_mean"],
                -0.05,
            )
            self.assertAlmostEqual(
                coexact_pair["target_logprob_mass_delta_minus_random_tangent_mean"],
                0.6,
            )
            self.assertAlmostEqual(
                coexact_pair["semantic_margin_delta_minus_random_tangent_mean"],
                0.8,
            )

    def test_main_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "ontology_collapse__ontology_01__L5__k16" / "steering_metrics.csv"
            write_steering_metrics(csv_path)

            rc = summarize.main(["--run-root", str(root), "--output", str(root / "summary.csv")])

            self.assertEqual(rc, 0)
            self.assertTrue((root / "summary.csv").exists())
            self.assertTrue((root / "summary_component.csv").exists())
            self.assertTrue((root / "summary_pairwise.csv").exists())
            self.assertTrue((root / "summary_layer_pairwise.csv").exists())
            self.assertTrue((root / "summary_report.md").exists())


if __name__ == "__main__":
    unittest.main()
