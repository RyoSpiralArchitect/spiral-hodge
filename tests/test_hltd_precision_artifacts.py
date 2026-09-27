from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/hltd_precision_l7"
ARMS = ["fp16_replay", "fp32_fixed_field", "fp32_rebuilt_field"]


class TestHLTDPrecisionArtifacts(unittest.TestCase):
    def test_hashes_and_fixed_field_trace_receipts(self) -> None:
        manifest = json.loads((ROOT / "docs/figures/hltd_precision_l7_manifest.json").read_text())
        for item in manifest["tracked_data"] + manifest["tracked_figures"]:
            with self.subTest(path=item["path"]):
                payload = (ROOT / item["path"]).read_bytes()
                self.assertEqual(len(payload), item["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
        traces = {arm: next(r for r in manifest["ignored_source_artifacts"]
                            if r["path"].endswith(f"/{arm}/delta_trace.json")) for arm in ARMS[:2]}
        self.assertEqual(traces["fp16_replay"]["sha256"], traces["fp32_fixed_field"]["sha256"])

    def test_load_calibration_replay_and_freeze_receipts(self) -> None:
        protocol_bytes = (DATA / "protocol.json").read_bytes()
        protocol = json.loads(protocol_bytes)
        execution = json.loads((DATA / "execution_receipt.json").read_text())
        self.assertEqual(execution["status"], "COMPLETED")
        self.assertEqual(execution["protocol"]["sha256"], hashlib.sha256(protocol_bytes).hexdigest())
        self.assertLess(datetime.fromisoformat(protocol["frozen_utc"]), datetime.fromisoformat(execution["started_utc"]))
        load = json.loads((DATA / "load_audit.json").read_text())
        fp32 = load["models"]["float32"]
        self.assertEqual(fp32["observed_dtype"], "torch.float32")
        self.assertEqual(fp32["parameter_tensors"], 148)
        self.assertEqual(fp32["parameter_count"], 124439808)
        self.assertTrue(fp32["all_equal"])
        zero = json.loads((DATA / "zero_calibration.json").read_text())
        self.assertTrue(zero["passed"])
        expected = {(p["prompt_id"], t, dtype) for p in protocol["prompts"]
                    for t in range(1, p["token_count"] - 1) for dtype in ["float16", "float32"]}
        actual = {(r["prompt_id"], r["token_index"], r["dtype"]) for r in zero["token_units"]}
        self.assertEqual(expected, actual)
        self.assertEqual(len(zero["token_units"]), len(expected))
        self.assertTrue(all(r["zero_hook_max_abs"] == 0 and r["row_spread_max_abs"] == 0 for r in zero["token_units"]))
        self.assertTrue(json.loads((DATA / "replay_audit.json").read_text())["passed"])
        fixed = json.loads((DATA / "fixed_field_audit.json").read_text())
        self.assertTrue(fixed["passed"])
        self.assertEqual(fixed["matched_batches"], 1024)
        byte_audit = json.loads((DATA / "supplemental_load_bytes_audit.json").read_text())
        self.assertTrue(byte_audit["passed"])
        self.assertLess(datetime.fromisoformat(execution["completed_utc"]), datetime.fromisoformat(byte_audit["created_utc"]))
        for dtype in ["float16", "float32"]:
            audit = byte_audit["models"][dtype]
            self.assertTrue(audit["all_bytes_equal"])
            self.assertEqual(len(audit["tensors"]), 148)
            self.assertEqual(audit["parameter_count"], 124439808)
            self.assertTrue(all(t["source_bytes_equal"] for t in audit["tensors"]))
        raw_audit = json.loads((DATA / "independent_raw_audit.json").read_text())
        self.assertTrue(raw_audit["passed"])
        for arm in ARMS:
            audit = raw_audit["arms"][arm]
            self.assertEqual(audit["raw_rows"], 12288)
            self.assertEqual(audit["coefficient_count"], 188)
            self.assertLessEqual(audit["max_abs_error"], raw_audit["tolerance"])
            coefficient_bytes = (DATA / f"{arm}__summary_prompt_bin_response_coefficients.csv").read_bytes()
            self.assertEqual(hashlib.sha256(coefficient_bytes).hexdigest(), audit["coefficients"]["sha256"])

    def test_early_means_and_tolerance_recomputed(self) -> None:
        values = {}
        for arm in ARMS:
            grouped = defaultdict(dict)
            with (DATA / f"{arm}__summary_prompt_bin_response_coefficients.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["metric"] == "next_token_logprob_delta" and row["contrast_type"] == "odd" and int(row["position_bin"]) < 4:
                        grouped[row["prompt_id"]][int(row["position_bin"])] = float(row["response_coefficient"])
            self.assertEqual(len(grouped), 4)
            for bins in grouped.values():
                self.assertEqual(set(bins), {0, 1, 2, 3})
            values[arm] = {p: sum(b.values()) / 4 for p, b in grouped.items()}
        verdict = json.loads((DATA / "precision_verdict.json").read_text())
        self.assertEqual(verdict["n_prompts"], 4)
        positive = True
        stable = True
        for arm in ARMS:
            self.assertAlmostEqual(verdict["means"][arm], sum(values[arm].values()) / 4, places=12)
            count = sum(v > 0 for v in values[arm].values())
            self.assertEqual(verdict["positive_prompts"][arm], count)
            positive = positive and count == 4
            if arm != "fp16_replay":
                delta = max(abs(values[arm][p] - values["fp16_replay"][p]) for p in values[arm])
                self.assertAlmostEqual(verdict["max_abs_change"][arm], delta, places=12)
                stable = stable and delta <= verdict["absolute_tolerance"]
        expected_status = "PRECISION_STABLE_ON_PILOT" if positive and stable else "PRECISION_SENSITIVE_ON_PILOT"
        self.assertEqual(verdict["status"], expected_status)


if __name__ == "__main__":
    unittest.main()
