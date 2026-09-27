from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.analyze_hltd_precision_full_gate import compare_full_early, plot_full_comparison
from scripts.run_hltd_precision_full_gate import audit_model_weight_bytes, run, validate_full_protocol
from scripts.run_hltd_precision_gate import ARMS

ROOT = Path(__file__).resolve().parents[1]


def example_protocol() -> dict:
    protocol = json.loads((ROOT / "docs/data/hltd_precision_l7_full/protocol.json").read_text())
    protocol["run_root"] = "spiral_out_precision_full_unit_test"
    return protocol


class TestHLTDPrecisionFullGate(unittest.TestCase):
    def test_protocol_preserves_full_grid_and_numerical_contract(self) -> None:
        protocol = example_protocol()
        validate_full_protocol(protocol)
        for key, value in [("prompts", protocol["prompts"][:-1]), ("pilot_prompt_ids", []),
                           ("run_root", "spiral_out_hltd_precision_l7_pilot4_20260912")]:
            bad = copy.deepcopy(protocol)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_full_protocol(bad)
        bad = copy.deepcopy(protocol)
        bad["design"]["batch_size"] = 1
        with self.assertRaisesRegex(ValueError, "changed full contract"):
            validate_full_protocol(bad)

    def test_byte_audit_rejects_signed_zero_and_half_upcast(self) -> None:
        import torch
        from safetensors.numpy import save_file
        from transformers import GPT2Config, GPT2LMHeadModel

        with torch.device("cpu"):
            model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_embd=8, n_head=2, n_layer=1, n_positions=8,
                                             bos_token_id=0, eos_token_id=1)).float()
        with torch.no_grad():
            model.transformer.wte.weight[0, 0] = -0.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            arrays = {n.removeprefix("transformer."): p.detach().cpu().numpy().copy() for n, p in model.named_parameters()}
            save_file(arrays, str(path))
            self.assertTrue(audit_model_weight_bytes(model, path, "float32")["all_bytes_equal"])
            with torch.no_grad():
                model.transformer.wte.weight[0, 0] = +0.0
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                audit_model_weight_bytes(model, path, "float32")
            with torch.no_grad():
                model.transformer.wte.weight[0, 0] = -0.0
            model = model.half()
            self.assertTrue(audit_model_weight_bytes(model, path, "float16")["all_bytes_equal"])
            model = model.float()
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                audit_model_weight_bytes(model, path, "float32")

    def test_full_precision_failure_and_missing_coverage_are_retained(self) -> None:
        protocol = example_protocol()
        frames = {arm: pd.DataFrame([
            {"prompt_id": p["prompt_id"], "metric": "next_token_logprob_delta", "contrast_type": "odd",
             "position_bin": b, "response_coefficient": 0.4 + i * 0.01}
            for p in protocol["prompts"] for b in range(4)
        ]) for i, arm in enumerate(ARMS)}
        table, result = compare_full_early(frames, protocol)
        self.assertEqual(result["status"], "PRECISION_STABLE_ON_FULL_SUITE")
        self.assertEqual(result["subsets"]["pilot4"]["n_prompts"], 4)
        self.assertEqual(result["subsets"]["remaining16"]["n_prompts"], 16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "comparison.png"
            plot_full_comparison(table, 0.05, path)
            self.assertGreater(path.stat().st_size, 10000)
        changed = protocol["prompts"][-1]["prompt_id"]
        frames["fp32_rebuilt_field"].loc[lambda d: d.prompt_id == changed, "response_coefficient"] = 0.47
        _, result = compare_full_early(frames, protocol)
        self.assertEqual(result["status"], "PRECISION_SENSITIVE_ON_FULL_SUITE")
        self.assertEqual(result["failing_prompts"]["fp32_rebuilt_field"], [changed])
        frames["fp32_rebuilt_field"].loc[lambda d: d.prompt_id == changed, "response_coefficient"] = -0.01
        _, result = compare_full_early(frames, protocol)
        self.assertEqual(result["positive_prompts"]["fp32_rebuilt_field"], 19)
        frames["fp32_rebuilt_field"] = frames["fp32_rebuilt_field"].iloc[:-1]
        _, result = compare_full_early(frames, protocol)
        self.assertEqual(result["status"], "INSUFFICIENT_COVERAGE")
        frames["fp32_fixed_field"].iloc[0, frames["fp32_fixed_field"].columns.get_loc("response_coefficient")] = np.nan
        _, result = compare_full_early(frames, protocol)
        self.assertEqual(len(result["missing_early_prompt_arms"]), 2)

    def test_wrapper_keeps_failure_receipt_and_uses_in_run_byte_audit(self) -> None:
        from scripts import run_hltd_precision_full_gate as full
        from scripts import run_hltd_precision_gate as pilot

        protocol = example_protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "protocol.json"
            path.write_text(json.dumps(protocol))

            def fail_after_load(*args):
                self.assertIs(pilot.audit_model_weights, audit_model_weight_bytes)
                raise ValueError("retained failure")

            with patch.object(full, "ROOT", root), patch.object(full, "verify_frozen_files"), \
                 patch.object(full, "validate_full_protocol"), patch.object(pilot, "_run_pilot", side_effect=fail_after_load):
                with self.assertRaisesRegex(ValueError, "retained failure"):
                    run(path)
            receipt = json.loads((root / protocol["run_root"] / "execution_receipt.json").read_text())
            self.assertEqual(receipt["status"], "FAILED")
            self.assertIn("retained failure", receipt["error"])
            self.assertIsNot(pilot.audit_model_weights, audit_model_weight_bytes)


if __name__ == "__main__":
    unittest.main()
