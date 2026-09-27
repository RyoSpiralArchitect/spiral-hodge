from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import analyze_hltd_l8_gate as analysis
from scripts import run_hltd_l8_gate as runner
from scripts.evaluate_hltd_signed_layer_gate import decide_primary

ROOT = Path(__file__).resolve().parents[1]


def example_protocol() -> dict:
    reference = json.loads((ROOT / runner.REFERENCE).read_text())
    recorded = json.loads((ROOT / "docs/data/hltd_signed_l8_position/protocol.json").read_text())
    raw = next(r for r in runner.load_bridge_manifest(ROOT)["ignored_source_artifacts"]
               if r["path"] == recorded["bridge"]["reference_raw"])
    recorded["frozen_files"].append(raw)
    result = {key: copy.deepcopy(reference[key]) for key in ["suite", "target_set_file", "prompts", "design", "analysis"]}
    result["design"]["row_constants"]["layer"] = 8
    result.update({"reference_protocol": runner.REFERENCE, "run_root": "spiral_out_l8_unit_test",
        "model_path": recorded["model_path"], "frozen_files": recorded["frozen_files"],
        "runtime": {}, "tolerances": {"zero_hook_max_abs": 0.0001},
        "primary": {"metric": "next_token_logprob_delta", "contrast_type": "odd", "bins": [0, 1, 2, 3]},
        "bridge": recorded["bridge"],
        "checkpoint_sha256": next(r["sha256"] for r in reference["frozen_files"] if r["path"].endswith("/model.safetensors"))})
    return result


def coefficients(protocol: dict, value: float = 0.4) -> pd.DataFrame:
    return pd.DataFrame([{"prompt_id": p["prompt_id"], "position_bin": b, "metric": "next_token_logprob_delta",
                          "contrast_type": "odd", "response_coefficient": value}
                         for p in protocol["prompts"] for b in range(4)])


class TestHLTDL8Gate(unittest.TestCase):
    def test_only_layer_changes_and_bridge_keeps_sensitivity_case(self) -> None:
        protocol = example_protocol()
        runner.validate_protocol(protocol)
        bridge = runner.stage_protocol(protocol, "bridge")
        self.assertEqual([p["prompt_id"] for p in bridge["prompts"]], runner.BRIDGE_IDS)
        self.assertEqual(bridge["design"]["row_constants"]["layer"], 7)
        self.assertEqual(protocol["design"]["row_constants"]["layer"], 8)
        self.assertEqual(sum(p["token_count"] - 2 for p in bridge["prompts"]), 154)
        for key, value in [("prompts", protocol["prompts"][:-1]), ("run_root", "../escape"),
                           ("checkpoint_sha256", "changed"), ("tolerances", {"zero_hook_max_abs": 0.1})]:
            changed = copy.deepcopy(protocol)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_protocol(changed)
        changed = copy.deepcopy(protocol)
        changed["design"]["row_constants"]["layer"] = 9
        with self.assertRaisesRegex(ValueError, "changed L8 contract"):
            runner.validate_protocol(changed)

    def test_bridge_fails_closed_on_sensitivity_activity_and_coverage(self) -> None:
        protocol = runner.stage_protocol(example_protocol(), "bridge")
        old = coefficients(protocol)
        new = coefficients(protocol, 0.44)
        rows = pd.DataFrame([{"prompt_id": p["prompt_id"], "token_index": 1,
                              "component": "coexact", "component_active": 1} for p in protocol["prompts"]])
        _, verdict = analysis.compare_bridge(new, old, rows, rows, protocol)
        self.assertTrue(verdict["passed"])
        for value in [0.451, -0.01, np.nan]:
            changed = new.copy()
            changed.loc[changed.prompt_id == "literal_03", "response_coefficient"] = value
            if np.isnan(value):
                with self.assertRaises(ValueError):
                    analysis.compare_bridge(changed, old, rows, rows, protocol)
            else:
                _, verdict = analysis.compare_bridge(changed, old, rows, rows, protocol)
                self.assertFalse(verdict["passed"])
                self.assertEqual(verdict["failing_prompts"], ["literal_03"])
        activity = rows.copy()
        activity.loc[0, "component_active"] = 0
        _, verdict = analysis.compare_bridge(new, old, activity, rows, protocol)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["activity_changes"], 1)
        for changed in [new.iloc[:-1], pd.concat([new, new.iloc[:1]])]:
            with self.assertRaisesRegex(ValueError, "early coefficients"):
                analysis.compare_bridge(changed, old, rows, rows, protocol)
        with self.assertRaisesRegex(ValueError, "token grid"):
            analysis.activity_pairs(rows.iloc[:-1], rows)

    def test_l8_primary_is_not_the_bridge_positivity_test(self) -> None:
        protocol = example_protocol()
        values = coefficients(protocol)
        values.loc[values.prompt_id == "literal_03", "response_coefficient"] = -0.1
        phases = pd.DataFrame([{"metric": "next_token_logprob_delta", "contrast_type": "odd", "position_phase": "early",
                                "n_prompts": 20, "mean_response_coefficient": 0.2,
                                "bootstrap_ci_lower": 0.01, "bootstrap_ci_upper": 0.3}])
        self.assertEqual(decide_primary(values, phases, protocol)["status"], "SUPPORTED_WITHIN_SAMPLE")
        phases.loc[0, "bootstrap_ci_lower"] = 0
        self.assertEqual(decide_primary(values, phases, protocol)["status"], "NOT_SUPPORTED")
        self.assertEqual(decide_primary(values.iloc[:-1], phases, protocol)["status"], "INSUFFICIENT_COVERAGE")

    def test_incomplete_bridge_never_calls_l8(self) -> None:
        protocol = example_protocol()
        model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"),
                                parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type="mps"))])
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runner, "runtime_snapshot", return_value={}), \
             patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}), \
             patch("torch.is_autocast_enabled", return_value=False), \
             patch.object(runner.precision.fast, "_load_model_and_tokenizer", return_value=(model, None)), \
             patch.object(runner, "audit_model_weight_bytes", return_value={}), \
             patch.object(runner, "run_stage", return_value=pd.DataFrame()) as stage, \
             patch.object(analysis, "analyze_stage", return_value={"complete_early_coverage": False}):
            self.assertFalse(runner._run(protocol, Path(tmp)))
            stage.assert_called_once()
            self.assertEqual(stage.call_args.args[2]["design"]["row_constants"]["layer"], 7)
            self.assertFalse((Path(tmp) / "l8").exists())

    def test_run_preserves_failure_stop_and_output_exclusivity(self) -> None:
        for outcome in [False, ValueError("retained failure")]:
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / "protocol.json"
                protocol = example_protocol()
                path.write_text(json.dumps(protocol))
                with patch.object(runner, "ROOT", root), patch.object(runner, "verify_frozen_files"), \
                     patch.object(runner, "validate_protocol"), patch.object(runner, "_run") as execute:
                    if isinstance(outcome, Exception):
                        execute.side_effect = outcome
                        with self.assertRaisesRegex(ValueError, "retained failure"):
                            runner.run(path)
                    else:
                        execute.return_value = outcome
                        runner.run(path)
                    with self.assertRaises(FileExistsError):
                        runner.run(path)
                    self.assertEqual(execute.call_count, 1)
                receipt = json.loads((root / protocol["run_root"] / "execution_receipt.json").read_text())
                self.assertEqual(receipt["status"], "FAILED" if isinstance(outcome, Exception) else "STOPPED_BEFORE_L8")


if __name__ == "__main__":
    unittest.main()
