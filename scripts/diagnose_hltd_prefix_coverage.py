#!/usr/bin/env python3
"""Post-outcome, offline replay of the frozen prefix gate's inactive nodes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spiral_hodge as hodge
from scripts import hltd_prefix_transfer as transfer
from scripts import run_hltd_prefix_transfer_gate as runner
from scripts.evaluate_hltd_signed_layer_gate import file_receipt
from scripts.run_hltd_precision_gate import save_json, utc_now


def node_topology(decomposition, node: int) -> dict:
    incident = np.any(decomposition.edges == node, axis=1)
    faces = np.any(decomposition.triangles == node, axis=1)
    return {"incident_edges": int(incident.sum()), "incident_selected_triangles": int(faces.sum()),
            "incident_boundary_nonzeros": int(decomposition.C[incident].count_nonzero()),
            "incident_coexact_edge_norm": float(np.linalg.norm(decomposition.coexact[incident]))}


def replay_reducer(calibration, frozen):
    normalized = []
    for name in sorted(calibration):
        raw = np.asarray(calibration[name], dtype=np.float64)
        normalized.append(raw / np.linalg.norm(raw, axis=1, keepdims=True))
    reducer = PCA(n_components=len(frozen.components), svd_solver="full").fit(np.concatenate(normalized))
    if not np.array_equal(reducer.mean_, frozen.mean) or not np.array_equal(reducer.components_, frozen.components):
        raise ValueError("replayed calibration PCA changed")
    return reducer


def diagnose(protocol_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(protocol_path.read_text())
    runner.validate_execution(protocol)
    run = ROOT / protocol["run_root"]
    support = json.loads((run / "support_preflight.json").read_text())
    receipt = json.loads((run / "execution_receipt.json").read_text())
    if receipt["gate_status"] != "INSUFFICIENT_COVERAGE" or receipt["nonzero_rows"] != 0:
        raise ValueError("diagnostic is scoped to this stopped coverage gate")
    with np.load(run / "calibration_hidden.npz", allow_pickle=False) as data:
        calibration = dict(data)
    frozen = runner.read_field(run / "atlas.npz", run / "atlas_metadata.json")
    g = protocol["spec"]["geometry"]
    rebuilt = transfer.build_frozen_field(calibration, n_components=g["pca_components"], k=g["k"],
        target_betti_1_fraction=g["target_betti_1_fraction"], node_ridge=g["node_ridge"],
        min_chart_norm=g["min_chart_norm"], support_quantile=g["support_quantile"])
    if rebuilt.fingerprint() != frozen.fingerprint():
        raise ValueError("full calibration atlas replay was not byte-identical")
    reducer = replay_reducer(calibration, frozen)
    rows = []
    for cell in support["cells"]:
        if cell["status"] != "INACTIVE_COEXACT":
            continue
        raw = calibration[cell["node_prompt"]].astype(np.float64)
        z = reducer.transform(raw / np.linalg.norm(raw, axis=1, keepdims=True))
        field = hodge.token_node_vector_field(z[None], layer=0, mode="centered")
        decomposition, _ = hodge.hodge_latent_traversal_dynamics_matched_betti(
            field.points, field.vectors, k_neighbors=g["k"], target_betti_1_fraction=g["target_betti_1_fraction"])
        vectors = hodge.node_vectors_from_edge_component(field.points, decomposition.edges, decomposition.coexact, ridge=g["node_ridge"])
        local_index = cell["node_token"] - 1
        if not np.array_equal(vectors[local_index], frozen.coexact[cell["node_index"]]):
            raise ValueError("local node reconstruction differs from frozen vector")
        rows.append({key: cell[key] for key in ("prompt_id", "prefix_length", "node_prompt", "node_token",
                                               "node_index", "distance", "coexact_chart_norm")}
                    | node_topology(decomposition, local_index))
    norms = np.linalg.norm(frozen.coexact, axis=1)
    result = {"status": "OFFLINE_POST_OUTCOME_DIAGNOSTIC", "completed_utc": utc_now(),
              "model_loads": 0, "nonzero_treatments": 0, "atlas_replay_byte_identical": True,
              "atlas_fingerprint": rebuilt.fingerprint(), "atlas_nodes": len(norms),
              "exact_zero_nodes": int((norms == 0).sum()), "inactive_nodes": int((norms < g["min_chart_norm"]).sum()),
              "support_limit": frozen.support_limit, "dose_scale": frozen.dose_scale, "inactive_evaluation_cells": rows,
              "scope": "Post-outcome structural explanation only. No threshold, neighbor, corpus or primary decision changed.",
              "inputs": [file_receipt(path) for path in [Path(__file__), protocol_path, run / "calibration_hidden.npz",
                         run / "atlas.npz", run / "atlas_metadata.json", run / "support_preflight.json", run / "execution_receipt.json"]]}
    save_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = diagnose(args.protocol.absolute(), args.output.absolute())
    print(json.dumps({key: value for key, value in result.items() if key != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
