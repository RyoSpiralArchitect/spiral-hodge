from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.hltd_historical_sources import historical_source_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/hltd_fresh_l7_l8"


def read_rows(name: str) -> list[dict]:
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


class TestHLTDFreshArtifacts(unittest.TestCase):
    def test_manifest_and_fresh_text_inventory(self) -> None:
        manifest = json.loads((ROOT / "docs/figures/hltd_fresh_l7_l8_manifest.json").read_text())
        for record in manifest["tracked_data"] + manifest["tracked_figures"]:
            with self.subTest(path=record["path"]):
                payload = (ROOT / record["path"]).read_bytes()
                self.assertEqual(len(payload), record["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])
        for record in manifest["verification_code"]:
            with self.subTest(historical_source=record["path"]):
                self.assertTrue(historical_source_path(record, ROOT).is_file())
        qa = json.loads((DATA / "figure_qa.json").read_text())
        self.assertTrue(qa["passed"])
        self.assertEqual({r["path"] for r in qa["figures"]}, {r["path"] for r in manifest["tracked_figures"]})
        for record in qa["figures"]:
            self.assertGreater(record["nonwhite_fraction"], .01)
            self.assertGreater(record["pixel_std"], 1)
        p = json.loads((DATA / "protocol.json").read_text())
        source = [json.loads(line) for line in (ROOT / p["suite"]).read_text().splitlines() if line]
        self.assertEqual(source, [{k: row[k] for k in ["prompt_id", "family", "text"]} for row in p["prompts"]])
        counts = Counter(row["family"] for row in source)
        self.assertEqual(sorted(counts.values()), [5, 5, 5, 5])
        self.assertEqual(p["novelty"]["exact_duplicate_count"], 0)
        historical = []
        for record in p["frozen_files"]:
            if record["path"].endswith(".jsonl") and record["path"] != p["suite"]:
                historical.extend(json.loads(line) for line in (ROOT / record["path"]).read_text().splitlines() if line)
        def normalize(text: str) -> str:
            return " ".join(text.casefold().split())

        self.assertEqual(len(historical), p["novelty"]["historical_rows"])
        self.assertFalse({normalize(row["text"]) for row in source} & {normalize(row["text"]) for row in historical})

    def test_frozen_single_model_and_all_zero_controls(self) -> None:
        p = json.loads((DATA / "protocol.json").read_text())
        receipt = json.loads((DATA / "execution_receipt.json").read_text())
        self.assertEqual(receipt["status"], "COMPLETED")
        self.assertEqual(receipt["protocol"]["sha256"], hashlib.sha256((DATA / "protocol.json").read_bytes()).hexdigest())
        self.assertLess(datetime.fromisoformat(p["frozen_utc"]), datetime.fromisoformat(receipt["started_utc"]))
        load = json.loads((DATA / "load_audit.json").read_text())
        self.assertEqual(load["model_load_count"], 1)
        self.assertEqual(load["runtime"], p["runtime"])
        self.assertTrue(load["model"]["all_bytes_equal"])
        self.assertEqual(load["model"]["parameter_count"], 124439808)
        self.assertEqual(len(load["model"]["tensors"]), 148)
        self.assertEqual(load["model"]["observed_dtype"], "torch.float32")
        self.assertFalse(load["autocast"] or load["device_fallback"])
        expected = {(q["prompt_id"], t) for q in p["prompts"] for t in range(1, q["token_count"] - 1)}
        last_time = datetime.fromisoformat(receipt["started_utc"])
        for layer in [7, 8]:
            zero = json.loads((DATA / f"l{layer}__zero_calibration.json").read_text())
            self.assertTrue(zero["passed"])
            self.assertEqual({(r["prompt_id"], r["token_index"]) for r in zero["token_units"]}, expected)
            self.assertEqual(len(zero["token_units"]), len(expected))
            for r in zero["token_units"]:
                self.assertEqual(r["layer"], layer)
                self.assertEqual(r["dtype"], "float32")
                self.assertEqual(r["row_spread_max_abs"], 0)
                self.assertLessEqual(r["zero_hook_max_abs"], 1e-4)
            calibrated = datetime.fromisoformat(zero["completed_utc"])
            treated = datetime.fromisoformat(json.loads((DATA / f"l{layer}__treatment_started.json").read_text())["started_utc"])
            self.assertLess(last_time, calibrated)
            self.assertLess(calibrated, treated)
            last_time = treated
        self.assertLess(last_time, datetime.fromisoformat(receipt["completed_utc"]))
        charts = json.loads((DATA / "shared_chart_audit.json").read_text())
        self.assertTrue(charts["passed"])
        self.assertEqual({row["prompt_id"] for row in charts["prompts"]}, {q["prompt_id"] for q in p["prompts"]})
        self.assertTrue(all(row["byte_identical"] for row in charts["prompts"]))
        self.assertTrue(json.loads((DATA / "final_weight_audit.json").read_text())["source_bytes_equal_after_both_layers"])

    def test_independent_phase_bootstraps_and_both_layer_decision(self) -> None:
        p = json.loads((DATA / "protocol.json").read_text())
        verdict = json.loads((DATA / "gate_verdict.json").read_text())
        raw_check = json.loads((DATA / "stdlib_raw_audit.json").read_text())
        self.assertTrue(raw_check["site_disabled"])
        self.assertTrue(raw_check["passed"])
        supported, missing = [], False
        for layer in [7, 8]:
            actual_verdict = verdict["layers"][str(layer)]
            coverage = read_rows(f"l{layer}__primary_early_coverage.csv")
            complete = len(coverage) == 80 and all(int(r["n_active_tokens"]) > 0 for r in coverage)
            self.assertEqual(actual_verdict["complete_early_coverage"], complete)
            if not complete:
                self.assertEqual(actual_verdict["status"], "INSUFFICIENT_COVERAGE")
                missing = True
                continue
            grouped = defaultdict(dict)
            for row in read_rows(f"l{layer}__summary_prompt_bin_response_coefficients.csv"):
                key = (row["metric"], row["contrast_type"], row["family"], row["prompt_id"])
                b = int(row["position_bin"])
                self.assertNotIn(b, grouped[key])
                grouped[key][b] = float(row["response_coefficient"])
            estimates = defaultdict(dict)
            for (metric, contrast, family, prompt), bins in grouped.items():
                phases = {}
                for name, indexes in [("early", range(4)), ("middle", range(4, 8)), ("late", range(8, 12))]:
                    available = [bins[b] for b in indexes if b in bins]
                    if available:
                        phases[name] = sum(available) / len(available)
                if "early" in phases and "late" in phases:
                    phases["early_minus_late"] = phases["early"] - phases["late"]
                for name, value in phases.items():
                    estimates[(metric, contrast, name)][(family, prompt)] = value
            saved = {(r["metric"], r["contrast_type"], r["position_phase"]): r
                     for r in read_rows(f"l{layer}__summary_position_phase_bootstrap.csv")}
            self.assertEqual(set(estimates), set(saved))
            rng = np.random.default_rng(1729)
            for key in sorted(estimates):
                values = np.array([estimates[key][unit] for unit in sorted(estimates[key])])
                draws = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
                lower, upper = np.quantile(draws, [0.025, 0.975])
                self.assertAlmostEqual(float(saved[key]["mean_response_coefficient"]), float(values.mean()), places=10)
                self.assertAlmostEqual(float(saved[key]["bootstrap_ci_lower"]), float(lower), places=10)
                self.assertAlmostEqual(float(saved[key]["bootstrap_ci_upper"]), float(upper), places=10)
            primary = saved[("next_token_logprob_delta", "odd", "early")]
            lower = float(primary["bootstrap_ci_lower"])
            self.assertAlmostEqual(actual_verdict["mean"], float(primary["mean_response_coefficient"]), places=10)
            self.assertAlmostEqual(actual_verdict["ci_lower"], lower, places=10)
            self.assertAlmostEqual(actual_verdict["ci_upper"], float(primary["bootstrap_ci_upper"]), places=10)
            self.assertEqual(actual_verdict["status"], "SUPPORTED_WITHIN_SAMPLE" if lower > 0 else "NOT_SUPPORTED")
            supported.append(lower > 0)
            audit = json.loads((DATA / f"l{layer}__independent_raw_audit.json").read_text())
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["raw_rows"], 96 * sum(q["token_count"] - 2 for q in p["prompts"]))
            self.assertLessEqual(audit["max_abs_error"], 1e-10)
            coeff = DATA / f"l{layer}__summary_prompt_bin_response_coefficients.csv"
            self.assertEqual(hashlib.sha256(coeff.read_bytes()).hexdigest(), audit["coefficients"]["sha256"])
            self.assertEqual(raw_check["layers"][str(layer)], audit)
        expected = "INSUFFICIENT_COVERAGE" if missing else (
            "BOTH_LAYERS_SUPPORTED_ON_FRESH_TEXTS" if all(supported) else "NOT_SUPPORTED_AT_BOTH_LAYERS")
        self.assertEqual(verdict["status"], expected)

    def test_paired_secondary_and_layer_relative_scales(self) -> None:
        verdict = json.loads((DATA / "gate_verdict.json").read_text())
        if verdict["status"] == "INSUFFICIENT_COVERAGE":
            self.assertNotIn("paired_secondary", verdict)
            return
        table = sorted(read_rows("early_paired_comparison.csv"), key=lambda row: row["prompt_id"])
        values = np.array([float(r["l8"]) - float(r["l7"]) for r in table])
        self.assertEqual(len(values), 20)
        rng = np.random.default_rng(2718)
        draws = values[rng.integers(0, 20, size=(5000, 20))].mean(axis=1)
        lower, upper = np.quantile(draws, [.025, .975])
        result = verdict["paired_secondary"]
        self.assertAlmostEqual(result["mean"], float(values.mean()), places=10)
        self.assertAlmostEqual(result["ci_lower"], float(lower), places=10)
        self.assertAlmostEqual(result["ci_upper"], float(upper), places=10)
        for row in read_rows("dose_scale_comparison.csv"):
            ratio = float(row["l8_natural_step_norm"]) / float(row["l7_natural_step_norm"])
            self.assertAlmostEqual(float(row["l8_to_l7_scale_ratio"]), ratio, places=10)
        activity = read_rows("activity_comparison.csv")
        self.assertEqual(verdict["activity"]["different_masks"], sum(r["l7"] != r["l8"] for r in activity))


if __name__ == "__main__":
    unittest.main()
