from __future__ import annotations

import unittest

import numpy as np

from scripts.run_hltd_steering import _node_indices_for_position_bins, _select_node_indices


class FakeField:
    def __init__(self, centers: list[int]) -> None:
        self.token_edges = [(center - 1, center + 1) for center in centers]


class TestHLTDPositionSelector(unittest.TestCase):
    def test_position_bin_selector_collects_all_nodes_in_requested_bins(self) -> None:
        centers = list(range(1, 11))
        selected = _node_indices_for_position_bins(
            centers,
            position_bins=[0, 4],
            position_bin_count=5,
        )

        self.assertEqual(selected, [0, 1, 8, 9])

    def test_position_bin_selector_falls_back_to_nearest_node_for_empty_bin(self) -> None:
        selected = _node_indices_for_position_bins(
            [2],
            position_bins=[4],
            position_bin_count=5,
        )

        self.assertEqual(selected, [0])

    def test_select_node_indices_accepts_position_bin_selector(self) -> None:
        field = FakeField(list(range(1, 11)))
        components = {"coexact": np.ones((10, 2))}

        selected = _select_node_indices(
            components,
            token_selector="position_bin",
            selector_component="coexact",
            token_indices=[],
            position_bins=[0],
            position_bin_count=5,
            field=field,  # type: ignore[arg-type]
        )

        self.assertEqual(selected, [0, 1])


if __name__ == "__main__":
    unittest.main()
