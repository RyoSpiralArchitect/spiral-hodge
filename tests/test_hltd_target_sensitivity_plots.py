from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import plot_hltd_target_sensitivity as plots


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestHLTDTargetSensitivityPlots(unittest.TestCase):
    def test_build_target_sensitivity_writes_csv_png_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = root / "identity"
            source_b = root / "door"
            output = root / "out"
            rows_a = [
                {
                    "family": "identity_stress",
                    "prompt_id": "identity_04",
                    "layer": 7,
                    "k": 16,
                    "target_set": "identity_stress",
                    "component": "negative_coexact",
                    "alpha": 0.8,
                    "n_rows": 5,
                    "matched_random_rows": 5,
                    "branch_gate_rate": 1.0,
                    "branch_specific_gate_rate": 1.0,
                    "random_branch_gate_rate": 0.0,
                    "branch_gate_minus_random_rate": 1.0,
                    "token_drift_rate_mean": 0.5,
                    "token_drift_rate_minus_random_mean": 0.4,
                    "mean_target_margin_delta_mean": 0.25,
                    "mean_target_margin_delta_minus_random_mean": 0.36,
                    "mean_nearest_distance_mean": 0.2,
                },
                {
                    "family": "identity_stress",
                    "prompt_id": "identity_04",
                    "layer": 7,
                    "k": 16,
                    "target_set": "identity_stress",
                    "component": "random_tangent",
                    "alpha": 0.8,
                    "n_rows": 5,
                    "matched_random_rows": 5,
                    "branch_gate_rate": 0.0,
                    "branch_specific_gate_rate": 0.0,
                    "random_branch_gate_rate": 0.0,
                    "branch_gate_minus_random_rate": 0.0,
                    "token_drift_rate_mean": 0.1,
                    "token_drift_rate_minus_random_mean": 0.0,
                    "mean_target_margin_delta_mean": -0.11,
                    "mean_target_margin_delta_minus_random_mean": 0.0,
                    "mean_nearest_distance_mean": 0.2,
                },
            ]
            rows_b = [
                dict(row, target_set="identity_door_object")
                for row in rows_a
            ]
            rows_b[0]["random_branch_gate_rate"] = 0.2
            rows_b[0]["branch_gate_minus_random_rate"] = 0.8
            rows_b[0]["mean_target_margin_delta_mean"] = 0.36
            rows_b[0]["mean_target_margin_delta_minus_random_mean"] = 0.31
            rows_b[1]["branch_gate_rate"] = 0.2
            rows_b[1]["mean_target_margin_delta_mean"] = 0.05
            write_csv(source_a / "closed_loop_prompt_layer_k_summary.csv", rows_a)
            write_csv(source_b / "closed_loop_prompt_layer_k_summary.csv", rows_b)

            saved = plots.build_target_sensitivity(
                sources=[
                    ("identity_stress", source_a),
                    ("identity_door_object", source_b),
                ],
                output_root=output,
                prompt_id="identity_04",
                layer=7,
                k=16,
                component="negative_coexact",
            )

            for path in saved.values():
                p = Path(path)
                self.assertTrue(p.exists(), p)
                self.assertGreater(p.stat().st_size, 0, p)

            combined = list(csv.DictReader(Path(saved["csv"]).open(newline="", encoding="utf-8")))
            self.assertEqual(len(combined), 4)
            self.assertEqual(
                {row["target_set"] for row in combined},
                {"identity_stress", "identity_door_object"},
            )


if __name__ == "__main__":
    unittest.main()
