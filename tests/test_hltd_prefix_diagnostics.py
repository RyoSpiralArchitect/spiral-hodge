from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts.diagnose_hltd_prefix_coverage import node_topology, replay_reducer
from scripts.hltd_prefix_transfer import build_frozen_field


def test_graph_connected_node_can_have_no_coexact_face_support():
    decomposition = SimpleNamespace(
        edges=np.array([[0, 1], [1, 2], [0, 2], [0, 3]]),
        triangles=np.array([[0, 1, 2]]),
        C=csr_matrix([[1.], [1.], [-1.], [0.]]),
        coexact=np.array([1., 1., -1., 0.]),
    )
    node = node_topology(decomposition, 3)
    assert node == {"incident_edges": 1, "incident_selected_triangles": 0,
                    "incident_boundary_nonzeros": 0, "incident_coexact_edge_norm": 0.}
    assert node_topology(decomposition, 0)["incident_selected_triangles"] == 1


def test_replayed_pca_uses_only_calibration_and_supports_public_transform():
    rng = np.random.default_rng(20)
    data = {name: rng.normal(size=(9, 8)).astype(np.float32) for name in ("a", "b")}
    field = build_frozen_field(data, n_components=4, k=3)
    reducer = replay_reducer(data, field)
    raw = data["a"].astype(np.float64)
    z = reducer.transform(raw / np.linalg.norm(raw, axis=1, keepdims=True))
    np.testing.assert_array_equal(z[1:-1], field.points[:7])
    with pytest.raises(ValueError, match="PCA changed"):
        replay_reducer(data, replace(field, mean=field.mean + 1))
