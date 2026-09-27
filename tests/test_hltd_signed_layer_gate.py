from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_hltd_signed_layer_gate import decide_primary, early_coverage, file_receipt, validate_protocol_rows, verify_frozen_files


def fixture() -> tuple[pd.DataFrame, dict]:
    constants = {"layer": 7, "k": 16, "complex_mode": "matched_betti", "betti_1_fraction_target": 0.5}
    protocol = {
        "design": {"seeds": [0, 1], "components": ["coexact", "random_tangent"],
                   "alphas": [-1.0, -0.5, 0.5, 1.0], "row_constants": constants},
        "prompts": [{"prompt_id": "p1", "family": "literal", "token_count": 7}],
        "primary": {"bins": [0, 1, 2, 3], "metric": "next_token_logprob_delta", "contrast_type": "odd"},
    }
    rows = []
    for token in range(1, 6):
        for seed in [0, 1]:
            for component in ["coexact", "random_tangent"]:
                for alpha in [-1.0, -0.5, 0.5, 1.0]:
                    rows.append({**constants, "family": "literal", "prompt_id": "p1", "token_index": token,
                                 "node_index": token - 1, "token_count": 7, "seed": seed, "component": component,
                                 "alpha": alpha, "component_active": 1, "next_token_logprob_delta": alpha,
                                 "semantic_margin_delta": alpha / 2, "next_token_logprob_base": -5.0,
                                 "next_token_logprob_steered": -5.0 + alpha})
    return pd.DataFrame(rows), protocol


class TestHLTDSignedLayerGate(unittest.TestCase):
    def test_exact_grid(self) -> None:
        rows, protocol = fixture()
        result = validate_protocol_rows(rows, protocol)
        self.assertEqual(result["raw_rows"], 80)
        self.assertEqual(result["active_coexact_token_units"], 5)

    def test_missing_whole_token_cannot_hide_behind_counts(self) -> None:
        rows, protocol = fixture()
        with self.assertRaisesRegex(ValueError, "grid mismatch"):
            validate_protocol_rows(rows[rows["token_index"] != 2], protocol)

    def test_same_count_seed_replacement_is_rejected(self) -> None:
        rows, protocol = fixture()
        rows.loc[rows["seed"] == 1, "seed"] = 2
        with self.assertRaisesRegex(ValueError, "grid mismatch"):
            validate_protocol_rows(rows, protocol)

    def test_duplicates_and_changed_layer_are_rejected(self) -> None:
        rows, protocol = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_protocol_rows(pd.concat([rows, rows.iloc[:1]]), protocol)
        rows.loc[0, "layer"] = 8
        with self.assertRaisesRegex(ValueError, "condition mismatch"):
            validate_protocol_rows(rows, protocol)

    def test_baseline_mismatch_is_rejected(self) -> None:
        rows, protocol = fixture()
        rows.loc[0, "next_token_logprob_base"] += 0.1
        with self.assertRaisesRegex(ValueError, "same unsteered baseline"):
            validate_protocol_rows(rows, protocol)

    def test_nonfinite_and_noninteger_values_are_rejected(self) -> None:
        rows, protocol = fixture()
        rows.loc[0, "semantic_margin_delta"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_protocol_rows(rows, protocol)
        rows, protocol = fixture()
        rows["token_index"] = rows["token_index"].astype(float)
        rows.loc[0, "token_index"] = 1.1
        with self.assertRaisesRegex(ValueError, "noninteger"):
            validate_protocol_rows(rows, protocol)

    def test_signed_activity_cannot_change(self) -> None:
        rows, protocol = fixture()
        rows.loc[0, "component_active"] = 0
        with self.assertRaisesRegex(ValueError, "depend on alpha"):
            validate_protocol_rows(rows, protocol)

    def test_inactive_coexact_is_retained_only_in_denominators(self) -> None:
        rows, protocol = fixture()
        rows.loc[rows["component"] == "coexact", "component_active"] = 0
        protocol["analysis"] = {"bins": 12}
        result = validate_protocol_rows(rows, protocol)
        self.assertEqual(result["candidate_token_units"], 5)
        self.assertEqual(result["active_coexact_token_units"], 0)
        coverage = early_coverage(rows, protocol)
        self.assertEqual(len(coverage), 4)
        self.assertTrue((coverage["n_active_tokens"] == 0).all())

    def test_inactive_random_seed_cannot_reduce_null_repeats(self) -> None:
        rows, protocol = fixture()
        rows.loc[(rows["component"] == "random_tangent") & (rows["seed"] == 1), "component_active"] = 0
        with self.assertRaisesRegex(ValueError, "all frozen random control seeds"):
            validate_protocol_rows(rows, protocol)

    def test_coverage_and_positive_lower_bound_are_both_required(self) -> None:
        _, protocol = fixture()
        coefficients = pd.DataFrame([
            {"prompt_id": "p1", "position_bin": b, "metric": "next_token_logprob_delta",
             "contrast_type": "odd", "response_coefficient": 0.4} for b in range(4)
        ])
        phases = pd.DataFrame([{"metric": "next_token_logprob_delta", "contrast_type": "odd", "position_phase": "early",
                                "n_prompts": 1, "mean_response_coefficient": 0.4,
                                "bootstrap_ci_lower": 0.1, "bootstrap_ci_upper": 0.8}])
        self.assertEqual(decide_primary(coefficients, phases, protocol)["status"], "SUPPORTED_WITHIN_SAMPLE")
        self.assertEqual(decide_primary(coefficients.iloc[1:], phases, protocol)["status"], "INSUFFICIENT_COVERAGE")
        phases.loc[0, "bootstrap_ci_lower"] = 0
        self.assertEqual(decide_primary(coefficients, phases, protocol)["status"], "NOT_SUPPORTED")

    def test_changed_frozen_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "fixture.json"
            path.write_text("{}\n", encoding="utf-8")
            protocol = {"frozen_files": [file_receipt(path)]}
            verify_frozen_files(protocol, root)
            changed = copy.deepcopy(protocol)
            changed["frozen_files"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "frozen file changed"):
                verify_frozen_files(changed, root)


if __name__ == "__main__":
    unittest.main()
