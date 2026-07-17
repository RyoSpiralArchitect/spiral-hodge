from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from scripts import run_hltd_steering as steering


class TestHLTDModelHooks(unittest.TestCase):
    def test_random_tangent_uses_full_speed_when_coexact_is_zero(self) -> None:
        components = {
            "full": np.asarray([[2.0, 0.0], [0.0, 3.0]]),
            "exact": np.zeros((2, 2)),
            "coexact": np.zeros((2, 2)),
            "harmonic": np.zeros((2, 2)),
        }

        random = steering._random_tangent_component(components, seed=4)

        np.testing.assert_allclose(np.linalg.norm(random, axis=1), [2.0, 3.0])

    def test_block_for_hidden_layer_supports_gpt2_layout(self) -> None:
        blocks = ["b0", "b1", "b2"]
        model = SimpleNamespace(transformer=SimpleNamespace(h=blocks))

        self.assertEqual(steering._block_for_hidden_layer(model, 2), "b1")

    def test_block_for_hidden_layer_supports_decoder_layers_layout(self) -> None:
        blocks = ["l0", "l1", "l2", "l3"]
        model = SimpleNamespace(model=SimpleNamespace(layers=blocks))

        self.assertEqual(steering._block_for_hidden_layer(model, 3), "l2")

    def test_block_for_hidden_layer_rejects_missing_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not locate decoder blocks"):
            steering._block_for_hidden_layer(SimpleNamespace(), 1)


if __name__ == "__main__":
    unittest.main()
