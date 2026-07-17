from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import run_hltd_probe_gate as probe_gate
from scripts.run_hltd_steering import _with_derived_components


class TestHLTDProbeGate(unittest.TestCase):
    def test_with_derived_components(self) -> None:
        base = {
            "presence": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "coexact": np.asarray([[0.5, -1.0], [2.0, -2.0]]),
            "semantic_flow": np.asarray([[0.25, 0.5], [0.75, 1.0]]),
        }

        out = _with_derived_components(base)

        np.testing.assert_allclose(out["presence_plus_coexact"], base["presence"] + base["coexact"])
        np.testing.assert_allclose(out["coexact_minus_presence"], base["coexact"] - base["presence"])
        np.testing.assert_allclose(out["presence_minus_coexact"], base["presence"] - base["coexact"])
        np.testing.assert_allclose(out["negative_coexact"], -base["coexact"])
        np.testing.assert_allclose(out["semantic_flow_minus_presence"], base["semantic_flow"] - base["presence"])

    def test_load_probe_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(
                json.dumps(
                    {
                        "identity": {"positive_families": ["identity_stress"]},
                        "ontology": {"positive_families": ["ontology_collapse"]},
                    }
                ),
                encoding="utf-8",
            )

            specs = probe_gate.load_probe_specs(str(path), requested=["identity"])

            self.assertEqual(specs, {"identity": {"identity_stress"}})

    def test_score_probe(self) -> None:
        model = probe_gate.ProbeModel(
            name="identity",
            layer=5,
            positive_families={"identity_stress"},
            mean=np.zeros(2),
            scale=np.ones(2),
            coef=np.asarray([1.0, 0.0]),
            intercept=0.0,
            n_examples=2,
            n_positive=1,
            train_accuracy=1.0,
            train_auc=1.0,
            cv_prompt_accuracy=1.0,
            n_groups=2,
            grouping="prompt_id",
            training_token_selector="all_interior",
        )

        base_prob, base_logit, _ = probe_gate.score_probe(model, np.asarray([0.0, 0.0]))
        high_prob, high_logit, _ = probe_gate.score_probe(model, np.asarray([2.0, 0.0]))

        self.assertAlmostEqual(base_prob, 0.5)
        self.assertAlmostEqual(base_logit, 0.0)
        self.assertGreater(high_prob, base_prob)
        self.assertGreater(high_logit, base_logit)

    def test_training_matrix_groups_counterfactual_pairs(self) -> None:
        hidden = np.zeros((2, 4, 3), dtype=np.float64)
        caches = [
            probe_gate.PromptCache(
                prompt_id="stable",
                group_id="shared_pair",
                family="literal_stable",
                text="stable",
                input_ids=np.arange(4),
                token_texts=["a", "b", "c", "d"],
                hidden=hidden,
                coord=None,
            ),
            probe_gate.PromptCache(
                prompt_id="shift",
                group_id="shared_pair",
                family="identity_stress",
                text="shift",
                input_ids=np.arange(4),
                token_texts=["a", "b", "c", "d"],
                hidden=hidden + 1.0,
                coord=None,
            ),
        ]

        X, y, groups = probe_gate.build_probe_training_matrix(
            caches,
            layer=1,
            positive_families={"identity_stress"},
        )

        self.assertEqual(X.shape, (4, 3))
        self.assertEqual(y.tolist(), [0, 0, 1, 1])
        self.assertEqual(groups.tolist(), ["shared_pair"] * 4)

    def test_pair_balanced_training_matrix_equalizes_class_rows(self) -> None:
        stable_hidden = np.zeros((2, 5, 3), dtype=np.float64)
        shifted_hidden = np.ones((2, 8, 3), dtype=np.float64)
        caches = [
            probe_gate.PromptCache(
                prompt_id="stable",
                group_id="shared_pair",
                family="literal_stable",
                text="stable",
                input_ids=np.arange(5),
                token_texts=["x"] * 5,
                hidden=stable_hidden,
                coord=None,
            ),
            probe_gate.PromptCache(
                prompt_id="shift",
                group_id="shared_pair",
                family="identity_stress",
                text="shift",
                input_ids=np.arange(8),
                token_texts=["x"] * 8,
                hidden=shifted_hidden,
                coord=None,
            ),
        ]

        X, y, groups = probe_gate.build_probe_training_matrix(
            caches,
            layer=1,
            positive_families={"identity_stress"},
            token_selector="pair_balanced_interior",
        )

        self.assertEqual(X.shape, (6, 3))
        self.assertEqual(np.bincount(y).tolist(), [3, 3])
        self.assertEqual(groups.tolist(), ["shared_pair"] * 6)

    def test_training_split_rejects_prompt_reuse(self) -> None:
        training = [{"prompt_id": "shared"}, {"prompt_id": "train_only"}]
        evaluation = [{"prompt_id": "shared"}, {"prompt_id": "eval_only"}]

        with self.assertRaisesRegex(ValueError, "overlap"):
            probe_gate.validate_probe_training_split(training, evaluation)

        overlap = probe_gate.validate_probe_training_split(training, evaluation, allow_overlap=True)
        self.assertEqual(overlap, ["shared"])

    def test_pairwise_summary(self) -> None:
        rows = [
            {
                "family": "identity_stress",
                "prompt_id": "identity_01",
                "layer": 5,
                "k": 16,
                "seed": 0,
                "token_selector": "middle",
                "token_index": 7,
                "alpha": 1.0,
                "probe": "identity",
                "component": "coexact",
                "component_active": 1,
                "positive_prob_delta": 0.3,
                "positive_logit_delta": 1.2,
                "label_margin_delta": 1.2,
                "probe_entropy_delta": -0.1,
            },
            {
                "family": "identity_stress",
                "prompt_id": "identity_01",
                "layer": 5,
                "k": 16,
                "seed": 0,
                "token_selector": "middle",
                "token_index": 7,
                "alpha": 1.0,
                "probe": "identity",
                "component": "random_tangent",
                "component_active": 1,
                "positive_prob_delta": -0.1,
                "positive_logit_delta": -0.4,
                "label_margin_delta": -0.4,
                "probe_entropy_delta": 0.2,
            },
        ]

        pairwise = probe_gate.build_pairwise_summary(rows)
        layer = probe_gate.build_layer_pairwise_summary(pairwise)

        self.assertEqual(pairwise[0]["n_pairs"], 1)
        self.assertAlmostEqual(pairwise[0]["positive_prob_delta_minus_random_tangent_mean"], 0.4)
        self.assertAlmostEqual(pairwise[0]["positive_prob_delta_minus_random_tangent_positive_rate"], 1.0)
        self.assertAlmostEqual(pairwise[0]["label_margin_delta_minus_random_tangent_mean"], 1.6)
        self.assertEqual(layer[0]["n_pairs"], 1)
        self.assertAlmostEqual(layer[0]["label_margin_delta_minus_random_tangent_mean"], 1.6)


if __name__ == "__main__":
    unittest.main()
