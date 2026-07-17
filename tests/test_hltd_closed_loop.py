from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import run_hltd_closed_loop as closed_loop


class FakeReducer:
    def __init__(self) -> None:
        self.components_ = np.eye(2)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)[:, :2]


class FakeTokenizer:
    eos_token_id = None

    def decode(self, ids: list[int]) -> str:
        return "|".join(str(int(x)) for x in ids)


class TestHLTDClosedLoop(unittest.TestCase):
    def test_nearest_node_index_returns_distance(self) -> None:
        idx, dist = closed_loop._nearest_node_index(
            np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]),
            np.asarray([1.8, 0.1]),
        )

        self.assertEqual(idx, 1)
        self.assertAlmostEqual(dist, float(np.sqrt(0.2**2 + 0.1**2)))

    def test_chart_point_from_hidden_matches_reducer_normalization(self) -> None:
        z = closed_loop._chart_point_from_hidden(
            np.asarray([3.0, 4.0]),
            FakeReducer(),
            normalize_hidden=True,
        )

        np.testing.assert_allclose(z, np.asarray([0.6, 0.8]))

    def test_step_rows_for_seed_copies_without_mutating_source(self) -> None:
        source = [{"step": 0, "seed": 3, "value": 1.5}]

        copied = closed_loop._step_rows_for_seed(source, seed=9)

        self.assertEqual(copied, [{"step": 0, "seed": 9, "value": 1.5}])
        self.assertEqual(source, [{"step": 0, "seed": 3, "value": 1.5}])
        self.assertIsNot(copied[0], source[0])

    def test_prompt_heldout_targets_use_casefolded_whole_lexeme_phrases(self) -> None:
        semantic_sets = {
            "identity": {
                "target": ["door", "visitor", "Mirror Room", "name"],
                "control": ["battery"],
            }
        }

        effective, excluded = closed_loop._prompt_heldout_semantic_target_set(
            semantic_sets,
            "identity",
            "A VISITOR crossed the outdoor court and entered the mirror room.",
        )

        self.assertEqual(effective["target"], ["door", "name"])
        self.assertEqual(effective["control"], ["battery"])
        self.assertEqual(excluded, ["visitor", "Mirror Room"])

    def test_prompt_heldout_targets_reject_an_empty_effective_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "removed every target term"):
            closed_loop._prompt_heldout_semantic_target_set(
                {"identity": {"target": ["statue"], "control": []}},
                "identity",
                "The statue moved.",
            )

    def test_summarize_run_writes_overlap_and_generated_text(self) -> None:
        step_rows = [
            {
                "layer": 7,
                "k": 16,
                "seed": 0,
                "component": "coexact",
                "alpha": 1.0,
                "node_index": 2,
                "component_active": 1,
                "nearest_distance": 0.2,
                "delta_norm": 0.5,
                "kl_base_to_steered": 0.1,
                "entropy_delta": -0.05,
                "next_token_base_logprob": -2.0,
                "next_token_steered_logprob": -1.5,
                "next_token_logprob_gain": 0.5,
                "target_margin_delta": 0.25,
                "top_changed": 1,
            },
            {
                "layer": 7,
                "k": 16,
                "seed": 0,
                "component": "coexact",
                "alpha": 1.0,
                "node_index": 3,
                "component_active": 1,
                "nearest_distance": 0.4,
                "delta_norm": 0.4,
                "kl_base_to_steered": 0.2,
                "entropy_delta": 0.01,
                "next_token_base_logprob": -3.0,
                "next_token_steered_logprob": -2.0,
                "next_token_logprob_gain": 1.0,
                "target_margin_delta": 0.75,
                "top_changed": 0,
            },
        ]

        row = closed_loop._summarize_run(
            tokenizer=FakeTokenizer(),
            prompt_id="p0",
            family="ontology_collapse",
            prompt="prompt",
            prompt_input_ids=[10, 11],
            output_ids=[10, 11, 42, 44],
            step_rows=step_rows,
            baseline_new_ids=[42, 43],
        )

        self.assertEqual(row["generated_text"], "42|44")
        self.assertEqual(row["unique_nodes"], 2)
        self.assertAlmostEqual(row["baseline_token_overlap"], 0.5)
        self.assertAlmostEqual(row["mean_selected_logprob_gain"], 0.75)
        self.assertAlmostEqual(row["mean_target_margin_delta"], 0.5)

    def test_write_report_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "prompt_id": "p0",
                    "family": "literal_stable",
                    "prompt": "prompt",
                    "layer": 7,
                    "k": 16,
                    "seed": 0,
                    "component": "baseline",
                    "alpha": 0.0,
                    "generated_steps": 2,
                    "active_steps": 2,
                    "unique_nodes": 1,
                    "mean_nearest_distance": 0.1,
                    "mean_delta_norm": 0.0,
                    "mean_kl_base_to_steered": 0.0,
                    "mean_entropy_delta": 0.0,
                    "mean_selected_base_logprob": -1.0,
                    "mean_selected_steered_logprob": -1.0,
                    "mean_selected_logprob_gain": 0.0,
                    "mean_target_margin_delta": 0.0,
                    "top_changed_rate": 0.0,
                    "baseline_token_overlap": 1.0,
                    "generated_text": "hello",
                    "generated_token_ids": "[1, 2]",
                    "generated_tokens": '["hello"]',
                }
            ]

            closed_loop._write_csv(rows, root / "closed_loop_metrics.csv", closed_loop.RUN_FIELDS)
            closed_loop._write_report(rows, root / "closed_loop_report.md")

            self.assertGreater((root / "closed_loop_metrics.csv").stat().st_size, 0)
            report = (root / "closed_loop_report.md").read_text(encoding="utf-8")
            self.assertIn("HLTD Closed-Loop", report)
            self.assertIn("baseline", report)


if __name__ == "__main__":
    unittest.main()
