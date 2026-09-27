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
DATA = ROOT / "docs/data/hltd_precision_l7_full"
ARMS = ["fp16_replay", "fp32_fixed_field", "fp32_rebuilt_field"]


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestHLTDPrecisionFullArtifacts(unittest.TestCase):
    def test_hashes_and_matching_intervention_traces(self) -> None:
        manifest = json.loads((ROOT / "docs/figures/hltd_precision_l7_full_manifest.json").read_text())
        for item in manifest["tracked_data"] + manifest["tracked_figures"] + manifest["verification_code"]:
            with self.subTest(path=item["path"]):
                payload = (ROOT / item["path"]).read_bytes()
                self.assertEqual(len(payload), item["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
        traces = {arm: next(r for r in manifest["ignored_source_artifacts"]
                            if r["path"].endswith(f"/{arm}/delta_trace.json")) for arm in ARMS[:2]}
        self.assertEqual(traces["fp16_replay"]["sha256"], traces["fp32_fixed_field"]["sha256"])

    def test_frozen_execution_loads_and_all_zero_controls(self) -> None:
        payload = (DATA / "protocol.json").read_bytes()
        protocol = json.loads(payload)
        receipt = json.loads((DATA / "execution_receipt.json").read_text())
        self.assertEqual(receipt["status"], "COMPLETED")
        self.assertEqual(receipt["protocol"]["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertLess(datetime.fromisoformat(protocol["frozen_utc"]), datetime.fromisoformat(receipt["started_utc"]))
        self.assertLess(datetime.fromisoformat(receipt["started_utc"]), datetime.fromisoformat(receipt["completed_utc"]))
        self.assertEqual(len(protocol["prompts"]), 20)
        self.assertEqual(len(protocol["pilot_prompt_ids"]), 4)
        load = json.loads((DATA / "load_audit.json").read_text())
        for dtype in ["float16", "float32"]:
            audit = load["models"][dtype]
            self.assertEqual(audit["observed_dtype"], f"torch.{dtype}")
            self.assertTrue(audit["all_bytes_equal"])
            self.assertTrue(audit["lm_head_tied"])
            self.assertEqual(audit["parameter_count"], 124439808)
            self.assertEqual(len(audit["tensors"]), 148)
            self.assertTrue(all(t["source_bytes_equal"] for t in audit["tensors"]))
            self.assertIn("before calibration", audit["timing"])
        zero = json.loads((DATA / "zero_calibration.json").read_text())
        expected = {(p["prompt_id"], t, dtype) for p in protocol["prompts"]
                    for t in range(1, p["token_count"] - 1) for dtype in ["float16", "float32"]}
        actual = {(r["prompt_id"], r["token_index"], r["dtype"]) for r in zero["token_units"]}
        self.assertEqual(expected, actual)
        self.assertEqual(len(actual), 1128)
        self.assertEqual(len(zero["token_units"]), len(actual))
        self.assertTrue(zero["passed"])
        self.assertTrue(all(r["row_spread_max_abs"] == 0 and r["zero_hook_max_abs"] <= protocol["tolerances"]["zero_hook_max_abs"]
                            for r in zero["token_units"]))
        replay = json.loads((DATA / "replay_audit.json").read_text())
        self.assertTrue(replay["passed"])
        self.assertEqual(replay["rows"], 54144)
        fixed = json.loads((DATA / "fixed_field_audit.json").read_text())
        self.assertTrue(fixed["passed"])
        self.assertEqual(fixed["matched_batches"], 4512)
        raw = json.loads((DATA / "independent_raw_audit.json").read_text())
        self.assertTrue(raw["passed"])
        for arm in ARMS:
            audit = raw["arms"][arm]
            self.assertEqual(audit["raw_rows"], 54144)
            self.assertEqual(audit["coefficient_count"], len(read_csv(f"{arm}__summary_prompt_bin_response_coefficients.csv")))
            self.assertLessEqual(audit["max_abs_error"], raw["tolerance"])
            coefficients = (DATA / f"{arm}__summary_prompt_bin_response_coefficients.csv").read_bytes()
            self.assertEqual(hashlib.sha256(coefficients).hexdigest(), audit["coefficients"]["sha256"])

    def test_precision_decision_and_pilot_remainder_means(self) -> None:
        protocol = json.loads((DATA / "protocol.json").read_text())
        verdict = json.loads((DATA / "precision_verdict.json").read_text())
        values = {}
        expected_prompts = {p["prompt_id"] for p in protocol["prompts"]}
        complete = True
        for arm in ARMS:
            groups = defaultdict(dict)
            for row in read_csv(f"{arm}__summary_prompt_bin_response_coefficients.csv"):
                if row["metric"] == "next_token_logprob_delta" and row["contrast_type"] == "odd" and int(row["position_bin"]) < 4:
                    self.assertNotIn(int(row["position_bin"]), groups[row["prompt_id"]])
                    groups[row["prompt_id"]][int(row["position_bin"])] = float(row["response_coefficient"])
            complete = complete and set(groups) == expected_prompts and all(set(bins) == {0, 1, 2, 3} for bins in groups.values())
            values[arm] = {p: sum(bins.values()) / len(bins) for p, bins in groups.items()}
        if not complete:
            self.assertEqual(verdict["status"], "INSUFFICIENT_COVERAGE")
            return
        stable = True
        tolerance = protocol["tolerances"]["early_coefficient_max_abs_change"]
        self.assertEqual(verdict["absolute_tolerance"], tolerance)
        for arm in ARMS:
            self.assertAlmostEqual(verdict["means"][arm], sum(values[arm].values()) / 20, places=12)
            failures = [p["prompt_id"] for p in protocol["prompts"] if values[arm][p["prompt_id"]] <= 0 or
                        (arm != "fp16_replay" and abs(values[arm][p["prompt_id"]] - values["fp16_replay"][p["prompt_id"]]) > tolerance)]
            self.assertEqual(verdict["failing_prompts"][arm], failures)
            stable = stable and not failures
            for subset, prompts in [("pilot4", set(protocol["pilot_prompt_ids"])), ("remaining16", expected_prompts - set(protocol["pilot_prompt_ids"]))]:
                self.assertEqual(verdict["subsets"][subset]["n_prompts"], len(prompts))
                self.assertAlmostEqual(verdict["subsets"][subset]["means"][arm], sum(values[arm][p] for p in prompts) / len(prompts), places=12)
        self.assertEqual(verdict["status"], "PRECISION_STABLE_ON_FULL_SUITE" if stable else "PRECISION_SENSITIVE_ON_FULL_SUITE")
        diagnostic = json.loads((DATA / "largest_change_diagnostic.json").read_text())
        largest = max(expected_prompts, key=lambda p: abs(values["fp32_rebuilt_field"][p] - values["fp16_replay"][p]))
        self.assertEqual(diagnostic["prompt_id"], largest)
        for arm in ARMS:
            branch = diagnostic["arms"][arm]["branch_early_odd_coefficients"]
            self.assertAlmostEqual(branch["coexact"] - branch["random_tangent"], values[arm][largest], places=12)

    def test_all_phase_bootstraps_without_analysis_helpers(self) -> None:
        verdict = json.loads((DATA / "precision_verdict.json").read_text())
        for arm in ARMS:
            bins = defaultdict(dict)
            for row in read_csv(f"{arm}__summary_prompt_bin_response_coefficients.csv"):
                key = (row["metric"], row["contrast_type"], row["family"], row["prompt_id"])
                index = int(row["position_bin"])
                self.assertNotIn(index, bins[key])
                bins[key][index] = float(row["response_coefficient"])
            estimates = defaultdict(dict)
            for (metric, contrast, family, prompt), values in bins.items():
                phases = {}
                for phase, indexes in [("early", range(4)), ("middle", range(4, 8)), ("late", range(8, 12))]:
                    present = [values[b] for b in indexes if b in values]
                    if present:
                        phases[phase] = sum(present) / len(present)
                if "early" in phases and "late" in phases:
                    phases["early_minus_late"] = phases["early"] - phases["late"]
                for phase, value in phases.items():
                    estimates[(metric, contrast, phase)][(family, prompt)] = value
            actual = {(r["metric"], r["contrast_type"], r["position_phase"]): r
                      for r in read_csv(f"{arm}__summary_position_phase_bootstrap.csv")}
            self.assertEqual(set(estimates), set(actual))
            rng = np.random.default_rng(1729)
            for key in sorted(estimates):
                group = estimates[key]
                values = np.asarray([group[k] for k in sorted(group)])
                draws = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
                lower, upper = np.quantile(draws, [0.025, 0.975])
                self.assertAlmostEqual(float(actual[key]["mean_response_coefficient"]), float(values.mean()), places=10)
                self.assertAlmostEqual(float(actual[key]["bootstrap_ci_lower"]), float(lower), places=10)
                self.assertAlmostEqual(float(actual[key]["bootstrap_ci_upper"]), float(upper), places=10)
            support = verdict["response_support"][arm]
            if support["complete_early_coverage"]:
                primary = actual[("next_token_logprob_delta", "odd", "early")]
                self.assertEqual(support["n_prompts"], 20)
                self.assertEqual(support["observed_prompt_bins"], 80)
                self.assertAlmostEqual(support["ci_lower"], float(primary["bootstrap_ci_lower"]), places=10)
                self.assertEqual(support["status"], "SUPPORTED_WITHIN_SAMPLE" if support["ci_lower"] > 0 else "NOT_SUPPORTED")
            else:
                self.assertEqual(support["status"], "INSUFFICIENT_COVERAGE")


if __name__ == "__main__":
    unittest.main()
