from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import plot_hltd_reverse_specificity as plots


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestHLTDReverseSpecificityPlots(unittest.TestCase):
    def test_build_reverse_specificity_writes_combined_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = root / "identity.csv"
            ontology = root / "ontology.csv"
            output = root / "out"
            base_rows = [
                {
                    "target_set": "identity_stress",
                    "source_order": 0,
                    "component": "negative_coexact",
                    "branch_specific_gate_rate": 1.0,
                    "random_branch_gate_rate": 0.0,
                    "mean_target_margin_delta_minus_random_mean": 0.3,
                    "token_drift_rate_mean": 0.5,
                },
                {
                    "target_set": "identity_generic_control",
                    "source_order": 1,
                    "component": "negative_coexact",
                    "branch_specific_gate_rate": 0.0,
                    "random_branch_gate_rate": 0.2,
                    "mean_target_margin_delta_minus_random_mean": -0.4,
                    "token_drift_rate_mean": 0.5,
                },
                {
                    "target_set": "identity_stress",
                    "source_order": 0,
                    "component": "random_tangent",
                    "branch_specific_gate_rate": 0.0,
                    "random_branch_gate_rate": 0.0,
                    "mean_target_margin_delta_minus_random_mean": 0.0,
                    "token_drift_rate_mean": 0.1,
                },
            ]
            write_csv(identity, base_rows)
            write_csv(
                ontology,
                [
                    dict(
                        row,
                        target_set=str(row["target_set"]).replace("identity", "ontology"),
                    )
                    for row in base_rows
                ],
            )

            saved = plots.build_reverse_specificity(
                panels=[("identity_04 L7/k16", identity), ("ontology_05 L8/k16", ontology)],
                output_root=output,
                component="negative_coexact",
            )

            for path in saved.values():
                p = Path(path)
                self.assertTrue(p.exists(), p)
                self.assertGreater(p.stat().st_size, 0, p)

            combined = list(csv.DictReader(Path(saved["csv"]).open(newline="", encoding="utf-8")))
            self.assertEqual(len(combined), 4)
            self.assertEqual(
                {row["panel_label"] for row in combined},
                {"identity_04 L7/k16", "ontology_05 L8/k16"},
            )


if __name__ == "__main__":
    unittest.main()
