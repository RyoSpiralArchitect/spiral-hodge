from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.audit_hltd_precision_raw import audit_arm


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture() -> tuple[dict, list[dict], list[dict]]:
    protocol = {"prompts": [{"prompt_id": "p", "family": "f", "token_count": 4}],
                "design": {"seeds": [0, 1], "alphas": [-1, -0.5, -0.25, 0.25, 0.5, 1]}, "analysis": {"bins": 12}}
    rows = []
    coefficients = []
    for token, position_bin in [(1, 4), (2, 8)]:
        for seed in protocol["design"]["seeds"]:
            for alpha in protocol["design"]["alphas"]:
                for branch in ["coexact", "random_tangent"]:
                    treated = branch == "coexact"
                    rows.append({"prompt_id": "p", "family": "f", "token_index": token, "token_count": 4,
                                 "seed": seed, "alpha": alpha, "component": branch, "component_active": 1,
                                 "next_token_logprob_steered": -3 + seed / 10 + treated * (0.3 * alpha + 0.2 * alpha ** 2),
                                 "target_logprob_mass_steered": -4 + treated * (0.1 * alpha + 0.05 * alpha ** 2),
                                 "control_logprob_mass_steered": -5,
                                 "next_token_logprob_delta": 999, "semantic_margin_delta": -999})
        for metric, odd, even in [("next_token_logprob_delta", 0.3, 0.2), ("semantic_margin_delta", 0.1, 0.05)]:
            for contrast, value in [("odd", odd), ("even", even)]:
                coefficients.append({"prompt_id": "p", "family": "f", "position_bin": position_bin,
                                     "metric": metric, "contrast_type": contrast, "response_coefficient": value})
    return protocol, rows, coefficients


class TestHLTDPrecisionRawAudit(unittest.TestCase):
    def test_reduction_uses_steered_outputs_not_saved_deltas(self) -> None:
        protocol, rows, coefficients = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            raw, saved = Path(tmp) / "raw.csv", Path(tmp) / "coefficients.csv"
            write_rows(raw, rows)
            write_rows(saved, coefficients)
            audit = audit_arm(raw, saved, protocol)
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["coefficient_count"], 8)
            self.assertLess(audit["max_abs_error"], 1e-14)

    def test_omissions_duplicates_and_changed_estimates_fail(self) -> None:
        protocol, rows, coefficients = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            raw, saved = Path(tmp) / "raw.csv", Path(tmp) / "coefficients.csv"
            write_rows(saved, coefficients)
            for invalid in [rows[1:], rows + [rows[0]]]:
                write_rows(raw, invalid)
                with self.assertRaises(ValueError):
                    audit_arm(raw, saved, protocol)
            write_rows(raw, rows)
            coefficients[0]["response_coefficient"] += 0.1
            write_rows(saved, coefficients)
            with self.assertRaisesRegex(ValueError, "coefficient mismatch"):
                audit_arm(raw, saved, protocol)


if __name__ == "__main__":
    unittest.main()
