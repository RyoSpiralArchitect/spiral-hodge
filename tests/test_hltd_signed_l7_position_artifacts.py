from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/hltd_signed_l7_position"
MANIFEST = ROOT / "docs/figures/hltd_signed_l7_position_manifest.json"


def csv_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestHLTDSignedL7PositionArtifacts(unittest.TestCase):
    def test_compact_artifact_hashes(self) -> None:
        manifest = json.loads(MANIFEST.read_text())
        self.assertTrue(manifest["tracked_data"])
        self.assertTrue(manifest["tracked_figures"])
        for item in [*manifest["tracked_data"], *manifest["tracked_figures"]]:
            with self.subTest(path=item["path"]):
                payload = (ROOT / item["path"]).read_bytes()
                self.assertEqual(len(payload), item["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
                if "rows" in item:
                    self.assertEqual(len(list(csv.reader(payload.decode().splitlines()))) - 1, item["rows"])

    def test_freeze_precedes_execution_and_failure_is_preserved(self) -> None:
        payload = (DATA / "protocol.json").read_bytes()
        protocol = json.loads(payload)
        execution = json.loads((DATA / "execution_receipt.json").read_text())
        self.assertEqual(execution["protocol_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertLess(datetime.fromisoformat(protocol["frozen_utc"]), datetime.fromisoformat(execution["started_utc"]))
        self.assertLess(datetime.fromisoformat(execution["started_utc"]), datetime.fromisoformat(execution["completed_utc"]))
        self.assertEqual(execution["exit_code"], 0)
        self.assertEqual(execution["argv"], protocol["command_argv"])
        failed = json.loads((DATA / "preflight_legacy_fp16.json").read_text())
        self.assertFalse(failed["calibration"]["passed"])
        calibration = json.loads((DATA / "zero_batch_calibration.json").read_text())
        self.assertTrue(calibration["passed"])
        self.assertEqual(calibration["zero_vs_matched_batch_logit_max_abs"], 0)
        self.assertEqual(calibration["batch_row_logit_max_abs"], 0)
        expected = {(p["prompt_id"], t) for p in protocol["prompts"] for t in range(1, p["token_count"] - 1)}
        observed = {(p["prompt_id"], p["token_index"]) for p in calibration["token_units"]}
        self.assertEqual(observed, expected)
        self.assertEqual(len(calibration["token_units"]), len(expected))

    def test_phase_bootstrap_recomputed_without_analyzer_helpers(self) -> None:
        coefficients = csv_rows("summary_prompt_bin_response_coefficients.csv")
        bins = defaultdict(dict)
        for row in coefficients:
            key = (row["metric"], row["contrast_type"], row["family"], row["prompt_id"])
            b = int(row["position_bin"])
            self.assertNotIn(b, bins[key])
            bins[key][b] = float(row["response_coefficient"])
        estimates = defaultdict(dict)
        for (metric, contrast, family, prompt), values in bins.items():
            phases = {}
            for phase, indexes in [("early", range(4)), ("middle", range(4, 8)), ("late", range(8, 12))]:
                observed = [values[b] for b in indexes if b in values]
                if observed:
                    phases[phase] = sum(observed) / len(observed)
            if "early" in phases and "late" in phases:
                phases["early_minus_late"] = phases["early"] - phases["late"]
            for phase, value in phases.items():
                estimates[(metric, contrast, phase)][(family, prompt)] = value
        actual = {(r["metric"], r["contrast_type"], r["position_phase"]): r
                  for r in csv_rows("summary_position_phase_bootstrap.csv")}
        self.assertEqual(set(actual), set(estimates))
        rng = np.random.default_rng(1729)
        for key in sorted(estimates):
            group = estimates[key]
            values = np.asarray([group[k] for k in sorted(group)])
            draws = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
            lower, upper = np.quantile(draws, [0.025, 0.975])
            self.assertAlmostEqual(float(actual[key]["mean_response_coefficient"]), float(values.mean()), places=10)
            self.assertAlmostEqual(float(actual[key]["bootstrap_ci_lower"]), float(lower), places=10)
            self.assertAlmostEqual(float(actual[key]["bootstrap_ci_upper"]), float(upper), places=10)
        verdict = json.loads((DATA / "gate_verdict.json").read_text())
        primary = actual[("next_token_logprob_delta", "odd", "early")]
        self.assertEqual(verdict["n_prompts"], 20)
        self.assertEqual(verdict["expected_prompt_bins"], 80)
        self.assertEqual(verdict["observed_prompt_bins"], 80)
        self.assertEqual(verdict["missing_prompt_bins"], [])
        self.assertAlmostEqual(verdict["ci_lower"], float(primary["bootstrap_ci_lower"]), places=10)
        expected_status = "SUPPORTED_WITHIN_SAMPLE" if verdict["ci_lower"] > 0 else "NOT_SUPPORTED"
        self.assertEqual(verdict["status"], expected_status)


if __name__ == "__main__":
    unittest.main()
