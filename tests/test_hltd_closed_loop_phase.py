from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import summarize_hltd_closed_loop_phase as audit
from scripts.hltd_position import centered_node_position, position_phase


def fixture_rows() -> tuple[list[dict], dict]:
    rows = []
    for prompt in ["p0", "p1"]:
        for seed in [0, 1]:
            for component, alpha in [("baseline", 0.0), ("coexact", .8), ("random_tangent", .8)]:
                for step in range(2):
                    node = 0 if component == "coexact" else 3
                    rows.append(dict(prompt_id=prompt, family="identity", target_set="identity",
                                     seed=seed, component=component, alpha=alpha, step=step,
                                     prefix_len=7 + step, prompt_len=7, node_index=node, layer=5, k=16,
                                     component_active=1, next_token_id=4, top_changed=0, nearest_distance=.2,
                                     delta_norm=0.0 if component == "baseline" else .5,
                                     next_token_base_logprob=-2.0, target_margin_delta=.3,
                                     next_token_logprob_gain=.1, entropy_delta=-.02, kl_base_to_steered=.1))
    manifest = dict(seeds=[0, 1], alphas=[.8], steering_components=["coexact", "random_tangent"],
                    generate_steps=2, layers=[5], k=[16],
                    runs=[dict(prompt_id=p, family="identity", layer=5, k=16, target_set="identity")
                          for p in ["p0", "p1"]])
    return rows, manifest


def write_source(root: Path, rows: list[dict], manifest: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "closed_loop_steps.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_name("closed_loop_manifest.json").write_text(json.dumps(manifest))
    return path


class ClosedLoopPhaseTest(unittest.TestCase):
    def test_centered_mapping_and_exact_third_boundaries(self) -> None:
        self.assertEqual(centered_node_position(0, 7), (1, 1 / 6))
        self.assertEqual(centered_node_position(4, 7), (5, 5 / 6))
        self.assertEqual(position_phase(1 / 3), "middle")
        self.assertEqual(position_phase(2 / 3), "late")
        for node, length in [(-1, 7), (5, 7), (0, 3)]:
            with self.assertRaises(ValueError):
                centered_node_position(node, length)

    def test_legacy_recovery_and_explicit_position_agree(self) -> None:
        rows, manifest = fixture_rows()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, record = audit.annotate_source(write_source(root, rows, manifest), "run", 12)
            self.assertEqual(record["position_origin"], "recovered_centered_step1")
            for row in rows:
                row["node_token_index"], row["position_frac"] = centered_node_position(row["node_index"], row["prompt_len"])
            manifest.update(step_schema_version=2, field_contract={"vector_mode": "centered", "step": 1})
            explicit, _ = audit.annotate_source(write_source(root, rows, manifest), "run", 12)
            np.testing.assert_allclose(data["position_frac"], explicit["position_frac"])
            rows[0]["position_frac"] = .99
            with self.assertRaisesRegex(ValueError, "disagrees"):
                audit.annotate_source(write_source(root, rows, manifest), "run", 12)

    def test_collapse_verifies_all_deterministic_seed_fields(self) -> None:
        rows, manifest = fixture_rows()
        with tempfile.TemporaryDirectory() as tmp:
            data, _ = audit.annotate_source(write_source(Path(tmp), rows, manifest), "run", 12)
            canonical = audit.collapse_deterministic(data)
            self.assertEqual(len(canonical), 16)
            deterministic = canonical[canonical["component"] != "random_tangent"]
            self.assertTrue(deterministic["seed"].eq(-1).all())
            self.assertTrue(deterministic["source_seed_count"].eq(2).all())
            data.loc[(data["component"] == "coexact") & data["seed"].eq(1), "next_token_id"] = 9
            with self.assertRaisesRegex(ValueError, "copies disagree"):
                audit.collapse_deterministic(data)

    def test_manifest_arms_missing_steps_duplicates_and_wrong_contract_fail(self) -> None:
        original, manifest = fixture_rows()
        cases = [original[:-1], original + [original[0]],
                 [r for r in original if not (r["component"] == "coexact" and r["seed"] == 1)],
                 [dict(r, node_index=1.2) if i == 0 else r for i, r in enumerate(original)],
                 [dict(r, prefix_len=999) if i == 0 else r for i, r in enumerate(original)]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rows in cases:
                with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                    audit.annotate_source(write_source(root, rows, manifest), "run", 12)
            manifest["field_contract"] = {"vector_mode": "forward", "step": 1}
            with self.assertRaisesRegex(ValueError, "centered step=1"):
                audit.annotate_source(write_source(root, original, manifest), "run", 12)

    def test_short_trajectory_requires_recorded_eos(self) -> None:
        rows, manifest = fixture_rows()
        short = [r for r in rows if r["step"] == 0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Truncated"):
                audit.annotate_source(write_source(root, short, manifest), "run", 12)
            manifest.update(stop_at_eos=True, eos_token_id=4)
            data, _ = audit.annotate_source(write_source(root, short, manifest), "run", 12)
            self.assertEqual(len(data), 12)

    def test_prompt_weighting_and_inactive_phases(self) -> None:
        rows, manifest = fixture_rows()
        with tempfile.TemporaryDirectory() as tmp:
            data, _ = audit.annotate_source(write_source(Path(tmp), rows, manifest), "run", 12)
        canonical = audit.collapse_deterministic(data)
        # Different trajectory lengths should not make one prompt count twice.
        canonical = canonical[~(canonical["prompt_id"].eq("p0") & canonical["step"].eq(1))].copy()
        canonical.loc[canonical["prompt_id"] == "p0", "target_margin_delta"] = 1.0
        canonical.loc[canonical["prompt_id"] == "p1", "target_margin_delta"] = 3.0
        prompt, summary = audit.prompt_and_condition_summary(audit.summarize_trajectories(canonical))
        branch = summary[summary["component"] == "coexact"].set_index("phase")
        self.assertAlmostEqual(branch.loc["all", "target_margin_delta"], 2.0)
        self.assertEqual(branch.loc["all", "n_prompts"], 2)
        self.assertEqual(branch.loc["late", "occupancy"], 0)
        self.assertTrue(np.isnan(branch.loc["late", "target_margin_delta"]))
        self.assertEqual(branch.loc["late", "n_visited_prompts"], 0)
        canonical.loc[canonical["component"] == "coexact", "active_rate"] = 0
        _, summary = audit.prompt_and_condition_summary(audit.summarize_trajectories(canonical))
        branch = summary[summary["component"] == "coexact"].set_index("phase")
        self.assertEqual(branch.loc["early", "occupancy"], 1)
        self.assertEqual(branch.loc["early", "n_active_prompts"], 0)
        self.assertTrue(np.isnan(branch.loc["early", "target_margin_delta"]))
        self.assertEqual(len(prompt), 24)

    def test_matching_respects_source_seed_target_and_alpha(self) -> None:
        rows, manifest = fixture_rows()
        with tempfile.TemporaryDirectory() as tmp:
            data, _ = audit.annotate_source(write_source(Path(tmp), rows, manifest), "run", 12)
        data.loc[data["component"] == "coexact", "target_margin_delta"] = 2
        data.loc[data["component"] == "random_tangent", "target_margin_delta"] = 1
        _, summary = audit.random_contrasts(data)
        self.assertTrue(summary["gap_target_margin_delta"].eq(1).all())
        self.assertTrue(summary["gap_late_occupancy"].eq(-1).all())
        for key, value in [("source_id", "elsewhere"), ("seed", 8), ("target_set", "other"), ("alpha", 1.1)]:
            changed = data.copy()
            mask = changed["component"] == "random_tangent"
            changed.loc[mask, key] = changed.loc[mask, key] + value if key == "seed" else value
            _, summary = audit.random_contrasts(changed)
            self.assertTrue(summary["random_matched"].eq(0).all())
            self.assertTrue(summary["gap_target_margin_delta"].isna().all())

    def test_failed_preflight_writes_nothing_and_manifest_pins_sources(self) -> None:
        rows, manifest = fixture_rows()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_source(root / "source", rows[:-1], manifest)
            output = root / "audit"
            with self.assertRaises(ValueError):
                audit.run_audit([path], output, scan_root=root, plots=False)
            self.assertFalse(output.exists())
            path = write_source(root / "source", rows, manifest)
            result = audit.run_audit([path], output, scan_root=root, plots=False)
            self.assertEqual(result["sources"][0]["raw_rows"], 24)
            self.assertEqual(result["sources"][0]["unique_rows"], 16)
            self.assertEqual(result["sources"][0]["steps"]["sha256"], audit.file_record(path)["sha256"])
            for artifact in result["artifacts"]:
                self.assertEqual(artifact["sha256"], audit.file_record(output / artifact["path"])["sha256"])
            audit.export_evidence(output, root / "docs", result)
            receipt = json.loads((root / "docs/figures/hltd_closed_loop_phase_manifest.json").read_text())
            self.assertEqual(len(receipt["artifacts"]), 3)
            (output / "phase_summary.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed before evidence export"):
                audit.export_evidence(output, root / "other_docs", result)
            self.assertFalse((root / "other_docs").exists())

    def test_norm_mismatch_fails_before_any_output(self) -> None:
        rows, manifest = fixture_rows()
        for row in rows:
            if row["component"] == "random_tangent":
                row["delta_norm"] = 9
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_source(root / "source", rows, manifest)
            with self.assertRaisesRegex(ValueError, "norms do not match"):
                audit.run_audit([path], root / "audit", scan_root=root, plots=False)
            self.assertFalse((root / "audit").exists())

    def test_missing_layer_in_manifest_run_list_is_not_silently_accepted(self) -> None:
        rows, manifest = fixture_rows()
        manifest["layers"] = [5, 7]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "declared layer/k"):
                audit.annotate_source(write_source(Path(tmp), rows, manifest), "run", 12)

    def test_no_random_control_remains_missing_and_plots_render(self) -> None:
        from matplotlib.image import imread

        rows, manifest = fixture_rows()
        rows = [r for r in rows if r["component"] != "random_tangent"]
        manifest["steering_components"] = ["coexact"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_source(root / "source", rows, manifest)
            result = audit.run_audit([path], root / "audit", scan_root=root, plots=True)
            images = [a for a in result["artifacts"] if a["path"].endswith(".png")]
            self.assertEqual(len(images), 2)
            for artifact in images:
                pixels = imread(root / "audit" / artifact["path"])
                self.assertGreater(float(pixels[:, :, :3].std()), .01)
            self.assertNotIn("phase_random_gaps.png", (root / "audit/phase_audit_report.md").read_text())


if __name__ == "__main__":
    unittest.main()
