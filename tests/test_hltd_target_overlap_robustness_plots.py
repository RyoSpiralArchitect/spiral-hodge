from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import plot_hltd_target_overlap_robustness as plots


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def surface_rows(target_set: str, *, gate_shift: float, margin_shift: float) -> list[dict[str, object]]:
    rows = []
    for layer in [5, 7]:
        for k in [12, 16]:
            for alpha in [0.4, 0.8]:
                for component in ["coexact", "coexact_minus_presence"]:
                    rows.append(
                        {
                            "family": "identity_stress",
                            "prompt_id": "identity_02",
                            "layer": layer,
                            "k": k,
                            "target_set": target_set,
                            "component": component,
                            "alpha": alpha,
                            "branch_specific_gate_rate": 0.6 + gate_shift,
                            "mean_target_margin_delta_minus_random_mean": 0.2 + margin_shift,
                        }
                    )
    return rows


class TestHLTDTargetOverlapRobustnessPlots(unittest.TestCase):
    def test_pair_surfaces_computes_heldout_minus_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_root = root / "reference"
            heldout_root = root / "heldout"
            write_csv(
                reference_root / "closed_loop_prompt_layer_k_summary.csv",
                surface_rows("identity_stress", gate_shift=0.0, margin_shift=0.0),
            )
            write_csv(
                heldout_root / "closed_loop_prompt_layer_k_summary.csv",
                surface_rows("identity_stress__prompt_heldout", gate_shift=-0.1, margin_shift=0.05),
            )

            reference = plots.load_surface(
                reference_root,
                prompt_id="identity_02",
                components=["coexact", "coexact_minus_presence"],
            )
            heldout = plots.load_surface(
                heldout_root,
                prompt_id="identity_02",
                components=["coexact", "coexact_minus_presence"],
            )
            paired = plots.pair_surfaces(reference, heldout)

            self.assertEqual(len(paired), 16)
            self.assertAlmostEqual(float(paired.iloc[0]["branch_specific_gate_rate_delta"]), -0.1)
            self.assertAlmostEqual(
                float(paired.iloc[0]["mean_target_margin_delta_minus_random_mean_delta"]),
                0.05,
            )

    def test_build_writes_csv_plot_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_root = root / "reference"
            heldout_root = root / "heldout"
            output_root = root / "out"
            write_csv(
                reference_root / "closed_loop_prompt_layer_k_summary.csv",
                surface_rows("identity_stress", gate_shift=0.0, margin_shift=0.0),
            )
            write_csv(
                heldout_root / "closed_loop_prompt_layer_k_summary.csv",
                surface_rows("identity_stress__prompt_heldout", gate_shift=-0.1, margin_shift=0.05),
            )

            saved = plots.build_target_overlap_robustness(
                reference_root=reference_root,
                heldout_root=heldout_root,
                output_root=output_root,
                prompt_id="identity_02",
                components=["coexact", "coexact_minus_presence"],
            )

            for value in saved.values():
                path = Path(value)
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)


if __name__ == "__main__":
    unittest.main()
