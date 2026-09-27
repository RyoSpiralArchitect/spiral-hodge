from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import run_hltd_fresh_gate as runner
from scripts.analyze_hltd_fresh_gate import joint_decision, paired_contrast, validate_zero_controls

ROOT = Path(__file__).resolve().parents[1]


def example_protocol() -> dict:
    result = json.loads((ROOT / runner.FRESH_REFERENCE).read_text())
    result["run_root"] = "spiral_out_fresh_unit_test"
    return result


class TestHLTDFreshGate(unittest.TestCase):
    def test_exact_novelty_not_stylistic_independence(self) -> None:
        old = [{"prompt_id": "old", "text": "A very familiar sentence."}]
        new = [{"prompt_id": "new", "text": "A very familiar sentence changed."}]
        audit = runner.novelty_audit(new, old)
        self.assertGreater(audit["max_word_trigram_jaccard"], 0)
        self.assertEqual(audit["exact_duplicate_count"], 0)
        for invalid in [[{"prompt_id": "new", "text": " A VERY familiar sentence. "}],
                        [{"prompt_id": "old", "text": "Different wording."}], new + new]:
            with self.assertRaises(ValueError):
                runner.novelty_audit(invalid, old)

    def test_fixed_pair_design_and_all_four_families(self) -> None:
        p = example_protocol()
        runner.validate_protocol(p)
        self.assertEqual(runner.layer_protocol(p, 8)["design"]["row_constants"]["layer"], 8)
        self.assertEqual(p["design"]["row_constants"]["layer"], 7)
        for key, value in [("layers", [7, 9]), ("prompts", p["prompts"][:-1]), ("run_root", "../escape")]:
            changed = copy.deepcopy(p)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_protocol(changed)
        changed = copy.deepcopy(p)
        changed["prompts"][0]["token_count"] = 100
        with self.assertRaises(ValueError):
            runner.validate_protocol(changed)

    def test_joint_decision_keeps_negative_and_missing_results(self) -> None:
        for first, second, expected in [
            ("SUPPORTED_WITHIN_SAMPLE", "SUPPORTED_WITHIN_SAMPLE", "BOTH_LAYERS_SUPPORTED_ON_FRESH_TEXTS"),
            ("SUPPORTED_WITHIN_SAMPLE", "NOT_SUPPORTED", "NOT_SUPPORTED_AT_BOTH_LAYERS"),
            ("NOT_SUPPORTED", "NOT_SUPPORTED", "NOT_SUPPORTED_AT_BOTH_LAYERS"),
            ("SUPPORTED_WITHIN_SAMPLE", "INSUFFICIENT_COVERAGE", "INSUFFICIENT_COVERAGE"),
        ]:
            self.assertEqual(joint_decision({7: {"status": first}, 8: {"status": second}}), expected)
        with self.assertRaises(ValueError):
            joint_decision({7: {"status": "SUPPORTED_WITHIN_SAMPLE"}})

    def test_paired_prompt_units_and_missing_bin_rejection(self) -> None:
        p = example_protocol()
        base = pd.DataFrame([{"prompt_id": prompt["prompt_id"], "position_bin": b, "metric": "next_token_logprob_delta",
                              "contrast_type": "odd", "response_coefficient": float(i)}
                             for i, prompt in enumerate(p["prompts"]) for b in range(4)])
        changed = base.copy()
        changed["response_coefficient"] += .25
        table, result = paired_contrast({7: base.sample(frac=1, random_state=0), 8: changed}, p)
        self.assertEqual(result["n_prompts"], 20)
        self.assertAlmostEqual(result["mean"], .25)
        self.assertAlmostEqual(result["ci_lower"], .25)
        self.assertAlmostEqual(result["ci_upper"], .25)
        self.assertEqual(table.prompt_id.tolist(), sorted(table.prompt_id.tolist()))
        with self.assertRaisesRegex(ValueError, "early coefficients"):
            paired_contrast({7: base, 8: changed.iloc[:-1]}, p)

    def test_zero_grid_and_shared_chart_are_byte_strict(self) -> None:
        p = example_protocol()
        p["prompts"] = p["prompts"][:1]
        p["prompts"][0].update({"token_count": 24, "input_ids": list(range(24))})
        records = [{"prompt_id": p["prompts"][0]["prompt_id"], "token_index": t, "layer": 7, "dtype": "float32",
                    "zero_hook_max_abs": 0., "row_spread_max_abs": 0., "single_batch_logit_max_abs": .01,
                    "single_batch_next_logprob_delta": .001} for t in range(1, 23)]
        validate_zero_controls({"passed": True, "token_units": records}, p, 7)
        with self.assertRaisesRegex(ValueError, "zero controls"):
            validate_zero_controls({"passed": True, "token_units": records[:-1]}, p, 7)
        bad = copy.deepcopy(records)
        bad[0]["row_spread_max_abs"] = 1e-9
        with self.assertRaisesRegex(ValueError, "tolerance"):
            validate_zero_controls({"passed": True, "token_units": bad}, p, 7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = {key: np.array([0.0], dtype=np.float32) for key in ["hidden", "coords", "pca_components", "pca_mean"]}
            for layer in [7, 8]:
                (root / f"l{layer}").mkdir()
                np.savez(root / f"l{layer}/{p['prompts'][0]['prompt_id']}_fields.npz", **arrays)
            self.assertTrue(runner.compare_charts(root, p)["passed"])
            arrays["hidden"][0] = -0.0
            np.savez(root / f"l8/{p['prompts'][0]['prompt_id']}_fields.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                runner.compare_charts(root, p)

    def test_failure_receipt_and_missing_input_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = example_protocol()
            p["frozen_files"] = [{"path": str(root / "missing.csv"), "bytes": 0, "sha256": "missing"}]
            path = root / "protocol.json"
            path.write_text(json.dumps(p))
            with patch.object(runner, "ROOT", root), patch.object(runner, "validate_protocol"), \
                 patch.object(runner, "_run", side_effect=ValueError("retained failure")) as execute:
                with self.assertRaises(FileNotFoundError):
                    runner.run(path)
                execute.assert_not_called()
                self.assertFalse((root / p["run_root"]).exists())
                p["frozen_files"] = []
                path.write_text(json.dumps(p))
                with self.assertRaisesRegex(ValueError, "retained failure"):
                    runner.run(path)
                receipt = json.loads((root / p["run_root"] / "execution_receipt.json").read_text())
                self.assertEqual(receipt["status"], "FAILED")
                with self.assertRaises(FileExistsError):
                    runner.run(path)


if __name__ == "__main__":
    unittest.main()
