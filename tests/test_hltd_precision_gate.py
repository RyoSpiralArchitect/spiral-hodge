from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_hltd_precision_gate import compare_early
from scripts.run_hltd_precision_gate import ARMS, audit_model_weights, compare_replay, delta_receipt, require_fixed_field


class TestHLTDPrecisionGate(unittest.TestCase):
    def test_nominal_delta_trace_rejects_changed_direction(self) -> None:
        a = delta_receipt(np.array([[0.1, 0.2]]), 3)
        self.assertEqual(a, delta_receipt(np.array([[0.1, 0.2]], dtype=np.float32), 3))
        require_fixed_field([a], [a])
        with self.assertRaisesRegex(ValueError, "differs"):
            require_fixed_field([a], [delta_receipt(np.array([[0.1, -0.2]]), 3)])
        with self.assertRaises(ValueError):
            require_fixed_field([], [])

    def test_native_weight_audit_rejects_upcast_half_weights(self) -> None:
        import torch
        from safetensors.numpy import save_file
        from transformers import GPT2Config, GPT2LMHeadModel

        with torch.device("cpu"):
            model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_embd=8, n_head=2, n_layer=1, n_positions=8,
                                             bos_token_id=0, eos_token_id=1))
        model = model.float()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            arrays = {name.removeprefix("transformer."): p.detach().cpu().numpy().copy() for name, p in model.named_parameters()}
            save_file(arrays, str(path))
            self.assertTrue(audit_model_weights(model, path, "float32")["all_equal"])
            model = model.half()
            self.assertTrue(audit_model_weights(model, path, "float16")["all_equal"])
            model = model.float()
            with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
                audit_model_weights(model, path, "float32")

    def test_precision_tolerance_and_coverage(self) -> None:
        protocol = {"prompts": [{"family": "literal", "prompt_id": "p"}], "primary": {"bins": [0, 1, 2, 3]},
                    "tolerances": {"early_coefficient_max_abs_change": 0.05}}
        frames = {arm: pd.DataFrame([
            {"prompt_id": "p", "metric": "next_token_logprob_delta", "contrast_type": "odd",
             "position_bin": b, "response_coefficient": 0.5 + i * 0.01} for b in range(4)
        ]) for i, arm in enumerate(ARMS)}
        _, result = compare_early(frames, protocol)
        self.assertEqual(result["status"], "PRECISION_STABLE_ON_PILOT")
        frames["fp32_fixed_field"]["response_coefficient"] = 0.6
        _, result = compare_early(frames, protocol)
        self.assertEqual(result["status"], "PRECISION_SENSITIVE_ON_PILOT")
        frames["fp32_rebuilt_field"] = frames["fp32_rebuilt_field"].iloc[:3]
        _, result = compare_early(frames, protocol)
        self.assertEqual(result["status"], "INSUFFICIENT_COVERAGE")

    def test_replay_compares_steered_outputs_not_changed_baselines(self) -> None:
        row = {"family": "f", "prompt_id": "p", "token_index": 1, "seed": 0, "component": "coexact", "alpha": 1,
               "next_token_logprob_steered": -1.0, "target_logprob_mass_steered": -3.0,
               "control_logprob_mass_steered": -4.0, "component_active": 1,
               "natural_step_norm": 0.5, "chart_norm": 0.1, "hidden_direction_norm": 0.1,
               "next_token_logprob_base": -2.0}
        reference = pd.DataFrame([row])
        replay = reference.copy()
        replay["next_token_logprob_base"] = -2.01
        self.assertTrue(compare_replay(replay, reference, 1e-8)["passed"])
        replay["next_token_logprob_steered"] = -1.01
        self.assertFalse(compare_replay(replay, reference, 1e-8)["passed"])


if __name__ == "__main__":
    unittest.main()
