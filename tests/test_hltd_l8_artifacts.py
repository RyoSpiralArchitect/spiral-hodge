from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from scripts.hltd_historical_sources import historical_source_path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/hltd_signed_l8_position"


def rows(name: str) -> list[dict]:
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


class TestHLTDL8Artifacts(unittest.TestCase):
    def test_manifest_binds_both_attempts_and_local_evidence(self) -> None:
        manifest = json.loads((ROOT / "docs/figures/hltd_signed_l8_position_manifest.json").read_text())
        for record in manifest["tracked_data"] + manifest["tracked_figures"]:
            with self.subTest(path=record["path"]):
                payload = (ROOT / record["path"]).read_bytes()
                self.assertEqual(len(payload), record["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])
        for record in manifest["verification_code"]:
            with self.subTest(historical_source=record["path"]):
                self.assertTrue(historical_source_path(record, ROOT).is_file())
        protocol = json.loads((DATA / "protocol_continuation.json").read_text())
        initial = json.loads((DATA / "protocol.json").read_text())
        for key in ["prompts", "design", "primary", "analysis", "runtime", "tolerances"]:
            self.assertEqual(protocol[key], initial[key])
        self.assertEqual(protocol["bridge"]["prompt_ids"], initial["bridge"]["prompt_ids"])
        self.assertEqual(protocol["bridge"]["max_abs_early_change"], 0.05)

    def test_failure_retained_and_freezes_precede_relevant_treatments(self) -> None:
        initial = json.loads((DATA / "protocol.json").read_text())
        protocol = json.loads((DATA / "protocol_continuation.json").read_text())
        failed = json.loads((DATA / "initial_execution_receipt.json").read_text())
        complete = json.loads((DATA / "execution_receipt.json").read_text())
        self.assertEqual(failed["status"], "FAILED")
        self.assertIn("FileNotFoundError", failed["error"])
        self.assertEqual(complete["status"], "COMPLETED")
        self.assertEqual(failed["protocol"]["sha256"], hashlib.sha256((DATA / "protocol.json").read_bytes()).hexdigest())
        self.assertEqual(complete["protocol"]["sha256"], hashlib.sha256((DATA / "protocol_continuation.json").read_bytes()).hexdigest())
        bridge_start = json.loads((DATA / "bridge__treatment_started.json").read_text())["started_utc"]
        l8_start = json.loads((DATA / "l8__treatment_started.json").read_text())["started_utc"]
        times = [initial["frozen_utc"], failed["started_utc"], bridge_start, failed["completed_utc"],
                 protocol["frozen_utc"], complete["started_utc"], l8_start, complete["completed_utc"]]
        self.assertEqual(times, sorted(times, key=datetime.fromisoformat))
        bridge = json.loads((DATA / "bridge_verdict.json").read_text())
        self.assertLess(datetime.fromisoformat(bridge["evaluated_utc"]), datetime.fromisoformat(l8_start))
        self.assertTrue(bridge["passed"])

    def test_sources_zeros_and_independent_raw_audits(self) -> None:
        protocol = json.loads((DATA / "protocol_continuation.json").read_text())
        load = json.loads((DATA / "load_audit.json").read_text())
        self.assertTrue(load["model"]["all_bytes_equal"])
        self.assertEqual(load["model"]["parameter_count"], 124439808)
        self.assertEqual(len(load["model"]["tensors"]), 148)
        self.assertEqual(load["model"]["observed_dtype"], "torch.float32")
        self.assertFalse(load["autocast"])
        self.assertFalse(load["device_fallback"])
        for stage, count in [("bridge", 154), ("l8", 564)]:
            zero = json.loads((DATA / f"{stage}__zero_calibration.json").read_text())
            prompts = protocol["prompts"] if stage == "l8" else [p for p in protocol["prompts"] if p["prompt_id"] in protocol["bridge"]["prompt_ids"]]
            expected = {(p["prompt_id"], t) for p in prompts for t in range(1, p["token_count"] - 1)}
            self.assertEqual({(r["prompt_id"], r["token_index"]) for r in zero["token_units"]}, expected)
            self.assertEqual(len(zero["token_units"]), count)
            self.assertTrue(zero["passed"])
            self.assertTrue(all(r["zero_hook_max_abs"] <= 1e-4 and r["row_spread_max_abs"] == 0 for r in zero["token_units"]))
            audit = json.loads((DATA / f"{stage}__independent_raw_audit.json").read_text())
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["raw_rows"], count * 96)
            self.assertLessEqual(audit["max_abs_error"], 1e-10)
            coeff = DATA / f"{stage}__summary_prompt_bin_response_coefficients.csv"
            self.assertEqual(hashlib.sha256(coeff.read_bytes()).hexdigest(), audit["coefficients"]["sha256"])
        recovered = json.loads((DATA / "reference_raw_audit.json").read_text())
        self.assertTrue(recovered["passed"])
        self.assertEqual(recovered["raw_rows"], 14784)
        table = rows("bridge__early_runtime_comparison.csv")
        self.assertEqual(len(table), 5)
        self.assertTrue(all(float(r["current"]) > 0 and abs(float(r["current"]) - float(r["previous"])) <= 0.05 for r in table))

    def test_phase_bootstraps_and_endpoint_without_production_helpers(self) -> None:
        protocol = json.loads((DATA / "protocol_continuation.json").read_text())
        verdict = json.loads((DATA / "gate_verdict.json").read_text())
        for stage in ["bridge", "l8"]:
            grouped = defaultdict(dict)
            for row in rows(f"{stage}__summary_prompt_bin_response_coefficients.csv"):
                key = (row["metric"], row["contrast_type"], row["family"], row["prompt_id"])
                b = int(row["position_bin"])
                self.assertNotIn(b, grouped[key])
                grouped[key][b] = float(row["response_coefficient"])
            estimates = defaultdict(dict)
            for (metric, contrast, family, prompt), values in grouped.items():
                phases = {}
                for name, indexes in [("early", range(4)), ("middle", range(4, 8)), ("late", range(8, 12))]:
                    present = [values[b] for b in indexes if b in values]
                    if present:
                        phases[name] = sum(present) / len(present)
                if "early" in phases and "late" in phases:
                    phases["early_minus_late"] = phases["early"] - phases["late"]
                for name, value in phases.items():
                    estimates[(metric, contrast, name)][(family, prompt)] = value
            actual = {(r["metric"], r["contrast_type"], r["position_phase"]): r for r in rows(f"{stage}__summary_position_phase_bootstrap.csv")}
            self.assertEqual(set(actual), set(estimates))
            rng = np.random.default_rng(1729)
            for key in sorted(estimates):
                values = np.array([estimates[key][unit] for unit in sorted(estimates[key])])
                draws = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
                lower, upper = np.quantile(draws, [0.025, 0.975])
                self.assertAlmostEqual(float(actual[key]["mean_response_coefficient"]), float(values.mean()), places=10)
                self.assertAlmostEqual(float(actual[key]["bootstrap_ci_lower"]), float(lower), places=10)
                self.assertAlmostEqual(float(actual[key]["bootstrap_ci_upper"]), float(upper), places=10)
            if stage == "l8":
                primary = actual[("next_token_logprob_delta", "odd", "early")]
                for p in protocol["prompts"]:
                    self.assertTrue(set(range(4)).issubset(grouped[("next_token_logprob_delta", "odd", p["family"], p["prompt_id"])]))
                self.assertEqual(verdict["observed_prompt_bins"], 80)
                self.assertAlmostEqual(verdict["mean"], float(primary["mean_response_coefficient"]), places=10)
                self.assertAlmostEqual(verdict["ci_lower"], float(primary["bootstrap_ci_lower"]), places=10)
                self.assertEqual(verdict["status"], "SUPPORTED_WITHIN_SAMPLE" if verdict["ci_lower"] > 0 else "NOT_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
