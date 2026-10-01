#!/usr/bin/env python3
"""Frozen-atlas, donor-prompt-excluded readout diagnostics; no model execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hltd_prefix_interpolation as interpolation
from scripts import hltd_prefix_transfer as base
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, verify_frozen_files
from scripts.run_hltd_precision_gate import save_json, utc_now

REFERENCE = "docs/data/hltd_prefix_transfer_v2/protocol.json"
REFERENCE_SHA = "88666f7a5d35c4e2f7dde490a1ce3ae19fa2ef08ef5555908298adca0a16f76c"
METHOD = "docs/data/hltd_prefix_calibration_readout/method.json"
METHOD_SHA = "6f0f833b8f31c912d649093edb5410eac16b5724894f3194d47b3ad16da1e253"
OUTPUT = "docs/data/hltd_prefix_calibration_readout/result"
SOURCES = ["scripts/diagnose_hltd_prefix_calibration.py", "tests/test_hltd_prefix_calibration.py"]
ATLAS_FILES = ["spiral_out_hltd_prefix_transfer_v1/atlas.npz", "spiral_out_hltd_prefix_transfer_v1/atlas_metadata.json"]
DEPENDENCIES = ["scripts/hltd_prefix_transfer.py", "scripts/hltd_prefix_interpolation.py",
                "scripts/evaluate_hltd_signed_layer_gate.py", "scripts/run_hltd_precision_gate.py"]
BIN_NAMES = ("early", "middle", "late")


def pinned_json(path, sha256):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"canonical input changed: {path}")
    return json.loads(payload)


def validate_inventory(atlas, prompts):
    expected = [(p, t) for p in sorted(prompts) for t in range(1, prompts[p]["token_count"] - 1)]
    if list(zip(atlas.node_prompts, atlas.node_tokens, strict=True)) != expected:
        raise ValueError("calibration node inventory or token positions changed")
    if {p for p, _ in atlas.calibration_hashes} != set(prompts):
        raise ValueError("calibration source membership changed")


def load_inputs(root=ROOT):
    protocol = pinned_json(root / REFERENCE, REFERENCE_SHA)
    method = pinned_json(root / METHOD, METHOD_SHA)
    prior = protocol["design"]["prior_protocol"]
    old = pinned_json(root / prior["path"], prior["sha256"])
    records = {(root / r["path"]).resolve(): r for r in protocol["frozen_files"]}
    consumed = [prior["path"], *ATLAS_FILES, *DEPENDENCIES]
    verify_frozen_files({"frozen_files": [records[(root / name).resolve()] for name in consumed]}, root)
    with np.load(root / ATLAS_FILES[0], allow_pickle=False) as data:
        atlas = base.FrozenPrefixField(**dict(data), **json.loads((root / ATLAS_FILES[1]).read_text()))
    readout = protocol["readout"]
    field = interpolation.InterpolatedField(atlas, readout["neighbors"], readout["bandwidth"], readout["kth_support_limit"])
    if field.receipt() != readout or field.fingerprint() != protocol["readout_fingerprint"]:
        raise ValueError("changed atlas/readout fingerprint")
    prompts = {p["prompt_id"]: {"token_count": p["token_count"], "family": p["family"]}
               for p in old["prompts"] if p["split"] == "calibration"}
    validate_inventory(atlas, prompts)
    if len(prompts) != 40 or atlas.points.shape != (1170, 32):
        raise ValueError("unexpected frozen calibration dimensions")
    inputs = [file_receipt(root / p) for p in [REFERENCE, METHOD, *consumed, *SOURCES]]
    return field, prompts, method, inputs


def contract(root=ROOT):
    field, prompts, method, inputs = load_inputs(root)
    return {"diagnostic_id": method["diagnostic_id"], "method": method, "readout": field.receipt(),
            "readout_fingerprint": field.fingerprint(), "prompts": prompts, "frozen_files": inputs,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}, "output": OUTPUT}


def validate(protocol, root=ROOT):
    expected = contract(root)
    if set(protocol) != set(expected) | {"frozen_utc"} or any(protocol[k] != v for k, v in expected.items()):
        raise ValueError("changed calibration diagnostic contract")


def freeze(path, root=ROOT):
    if path.exists() or (root / OUTPUT).exists():
        raise FileExistsError("diagnostic protocol/output already exists")
    protocol = {**contract(root), "frozen_utc": utc_now()}
    validate(protocol, root)
    save_json(path, protocol)
    return protocol


def donor_lookup(field):
    atlas = field.atlas
    owners = np.asarray(atlas.node_prompts)
    indices, distances, weights = [], [], []
    for i, point in enumerate(atlas.points):
        available = owners != owners[i]
        if available.sum() < field.neighbors:
            raise ValueError("not enough cross-prompt donor nodes")
        distance = np.linalg.norm(atlas.points - point, axis=1)
        distance[~available] = np.inf
        selected = np.argsort(distance, kind="stable")[:field.neighbors]
        local = distance[selected]
        logs = -.5 * (local / field.bandwidth) ** 2
        weight = np.exp(logs - logs.max())
        indices.append(selected)
        distances.append(local)
        weights.append(weight / weight.sum())
    return np.asarray(indices), np.asarray(distances), np.asarray(weights)


def aggregate(vectors, indices, weights):
    return np.stack([weight @ vectors[index] for index, weight in zip(indices, weights, strict=True)])


def cosines(target, prediction, threshold):
    tn, pn = np.linalg.norm(target, axis=1), np.linalg.norm(prediction, axis=1)
    valid = (tn >= threshold) & (pn >= threshold)
    value = np.full(len(target), np.nan)
    np.divide(np.sum(target * prediction, axis=1), tn * pn, out=value, where=valid)
    return np.clip(value, -1, 1)


def donor_metrics(vectors, indices, weights):
    neighbors = vectors[indices]
    norms = np.linalg.norm(neighbors, axis=2)
    contribution = weights * norms
    total = contribution.sum(axis=1)
    predicted = aggregate(vectors, indices, weights)
    prediction_norm = np.linalg.norm(predicted, axis=1)
    fraction = np.divide(prediction_norm, total, out=np.full(len(total), np.nan), where=total > 0)
    effective = np.divide(total**2, (contribution**2).sum(axis=1), out=np.full(len(total), np.nan), where=total > 0)
    shares = np.divide(contribution, total[:, None], out=np.zeros_like(contribution), where=total[:, None] > 0)
    units = np.divide(neighbors, norms[:, :, None], out=np.zeros_like(neighbors), where=norms[:, :, None] > 0)
    pair_cosines = np.einsum("nid,njd->nij", units, units)
    pair_weights = contribution[:, :, None] * contribution[:, None, :]
    upper = np.triu(np.ones((indices.shape[1], indices.shape[1]), dtype=bool), 1)
    pair_denominator = pair_weights[:, upper].sum(axis=1)
    pairwise = np.divide((pair_cosines * pair_weights)[:, upper].sum(axis=1), pair_denominator,
                         out=np.full(len(total), np.nan), where=pair_denominator > 0)
    return {"prediction": predicted, "prediction_norm": prediction_norm,
            "resultant_fraction": np.clip(fraction, 0, 1), "effective_contributors": effective,
            "pairwise_agreement": np.clip(pairwise, -1, 1), "contribution_shares": shares,
            "contribution_total": total, "donor_norms": norms}


def finite(value):
    return float(value) if np.isfinite(value) else None


def describe(values):
    data = np.asarray(values, dtype=float)
    valid = data[np.isfinite(data)]
    return {"n_total": len(data), "n_valid": len(valid), "mean": float(valid.mean()) if len(valid) else None,
            **{key: float(np.quantile(valid, q)) if len(valid) else None
               for key, q in (("q10", .1), ("median", .5), ("q90", .9))}}


def prompt_means(values, owners, mask=None):
    values, owners = np.asarray(values, dtype=float), np.asarray(owners)
    eligible = np.isfinite(values)
    if mask is not None:
        eligible &= mask
    return [{"prompt_id": str(owner), "n_valid": int((eligible & (owners == owner)).sum()),
             "mean": float(values[eligible & (owners == owner)].mean()) if (eligible & (owners == owner)).any() else None}
            for owner in sorted(set(owners))]


def balanced_mean(values, owners, mask=None):
    means = prompt_means(values, owners, mask)
    return describe([row["mean"] for row in means])


def paired_comparison(actual, control, owners, mask):
    valid = mask & np.isfinite(actual) & np.isfinite(control)
    a, b = balanced_mean(actual, owners, valid), balanced_mean(control, owners, valid)
    return {"paired_nodes": int(valid.sum()), "paired_prompts": a["n_valid"],
            "actual_cosine": a["mean"], "control_cosine": b["mean"],
            "paired_gap": a["mean"] - b["mean"] if a["n_valid"] else None}


def shuffled_indices(owners, draws, seed):
    owners = np.asarray(owners)
    rng = np.random.default_rng(seed)
    output = np.tile(np.arange(len(owners)), (draws, 1))
    for row in output:
        for owner in sorted(set(owners)):
            indices = np.flatnonzero(owners == owner)
            row[indices] = rng.permutation(indices)
    return output


def random_vectors(vectors, seed):
    values = []
    for i, norm in enumerate(np.linalg.norm(vectors, axis=1)):
        value = np.random.default_rng(np.random.SeedSequence([int(seed), i])).normal(size=vectors.shape[1])
        length = np.linalg.norm(value)
        if not np.isfinite(length) or length <= 0:
            raise ValueError("invalid random donor; no redraw")
        values.append(norm * value / length)
    return np.asarray(values)


def node_rows(field, prompts, indices, distances, weights, metrics):
    atlas = field.atlas
    owners = np.asarray(atlas.node_prompts)
    tokens = np.asarray(atlas.node_tokens)
    counts = np.array([prompts[p]["token_count"] for p in owners])
    relative = tokens / (counts - 1)
    bins = np.minimum((3 * relative).astype(int), 2)
    target_norm = np.linalg.norm(atlas.coexact, axis=1)
    cosine = cosines(atlas.coexact, metrics["prediction"], atlas.min_chart_norm)
    nearest_cosine = cosines(atlas.coexact, atlas.coexact[indices[:, 0]], atlas.min_chart_norm)
    supported = (distances[:, 0] <= atlas.support_limit) & (distances[:, -1] <= field.kth_support_limit)
    rows = []
    for i, selected in enumerate(indices):
        weight, shares = weights[i], metrics["contribution_shares"][i]
        owner_mass = [float(weight[owners[selected] == p].sum()) for p in set(owners[selected])]
        status = "DEFINED"
        if target_norm[i] < atlas.min_chart_norm:
            status = "INACTIVE_TARGET"
        elif metrics["prediction_norm"][i] < atlas.min_chart_norm:
            status = "INACTIVE_PREDICTION"
        row = {"node_index": i, "prompt_id": str(owners[i]), "family": prompts[owners[i]]["family"],
               "token_index": int(tokens[i]), "prefix_length": int(tokens[i] + 1), "token_count": int(counts[i]),
               "relative_position": float(relative[i]), "position_bin": BIN_NAMES[bins[i]],
               "recovery_status": status, "distance_supported": bool(supported[i]),
               "target_norm": float(target_norm[i]), "prediction_norm": float(metrics["prediction_norm"][i]),
               "recovery_cosine": finite(cosine[i]), "nearest_cosine": finite(nearest_cosine[i]),
               "nearest_distance": float(distances[i, 0]), "eighth_distance": float(distances[i, -1]),
               "resultant_fraction": finite(metrics["resultant_fraction"][i]),
               "cancellation_fraction": finite(1 - metrics["resultant_fraction"][i]),
               "pairwise_agreement": finite(metrics["pairwise_agreement"][i]),
               "effective_contributors": finite(metrics["effective_contributors"][i]),
               "max_contribution_share": float(shares.max()) if metrics["contribution_total"][i] > 0 else None,
               "inactive_donors": int((metrics["donor_norms"][i] < atlas.min_chart_norm).sum()),
               "unique_donor_prompts": len(owner_mass), "largest_donor_prompt_mass": max(owner_mass),
               "weighted_token_gap": float(weight @ np.abs(tokens[selected] - tokens[i])),
               "weighted_relative_gap": float(weight @ np.abs(relative[selected] - relative[i])),
               "same_position_bin_mass": float(weight[bins[selected] == bins[i]].sum())}
        for j, name in enumerate(BIN_NAMES):
            row[f"geometric_mass_{name}"] = float(weight[bins[selected] == j].sum())
            row[f"contribution_mass_{name}"] = float(shares[bins[selected] == j].sum()) if metrics["contribution_total"][i] > 0 else None
        rows.append(row)
    return rows, cosine, nearest_cosine, supported


def analyze(field, prompts, method):
    atlas, owners = field.atlas, np.asarray(field.atlas.node_prompts)
    indices, distances, weights = donor_lookup(field)
    metrics = donor_metrics(atlas.coexact, indices, weights)
    rows, actual, nearest, supported = node_rows(field, prompts, indices, distances, weights, metrics)
    scopes = {"all": np.ones(len(rows), dtype=bool), "supported": supported}
    comparison_rows = []

    def compare(name, draw, vectors):
        prediction = aggregate(vectors, indices, weights)
        control = cosines(atlas.coexact, prediction, atlas.min_chart_norm)
        denominator = (weights * np.linalg.norm(vectors[indices], axis=2)).sum(axis=1)
        resultant = np.divide(np.linalg.norm(prediction, axis=1), denominator,
                              out=np.full(len(rows), np.nan), where=denominator > 0)
        for scope, mask in scopes.items():
            record = paired_comparison(actual, control, owners, mask)
            paired = mask & np.isfinite(actual) & np.isfinite(control)
            comparison_rows.append({"comparator": name, "draw": draw, "scope": scope, **record,
                "actual_resultant_fraction": balanced_mean(metrics["resultant_fraction"], owners, paired)["mean"],
                "control_resultant_fraction": balanced_mean(resultant, owners, paired)["mean"]})

    for seed in method["random_seeds"]:
        compare("v2_random", seed, random_vectors(atlas.coexact, seed))
    permutations = shuffled_indices(owners, method["shuffle_draws"], method["shuffle_seed"])
    for draw, permutation in enumerate(permutations):
        compare("within_prompt_shuffle", draw, atlas.coexact[permutation])
    scalar_keys = ["recovery_cosine", "resultant_fraction", "cancellation_fraction", "pairwise_agreement",
                   "effective_contributors", "max_contribution_share", "unique_donor_prompts",
                   "largest_donor_prompt_mass", "weighted_token_gap", "weighted_relative_gap", "same_position_bin_mass"]
    stats = {}
    for key in scalar_keys:
        value = np.array([r[key] for r in rows], dtype=float)
        stats[key] = {"nodes": describe(value), "equal_prompt": balanced_mean(value, owners)}
    prompt_rows = []
    for prompt in sorted(prompts):
        mask = owners == prompt
        row = {"prompt_id": prompt, "family": prompts[prompt]["family"], "nodes": int(mask.sum()),
               "defined_recovery_nodes": int((mask & np.isfinite(actual)).sum()),
               "supported_recovery_nodes": int((mask & supported & np.isfinite(actual)).sum())}
        for key in scalar_keys:
            row[key] = describe([r[key] for r, include in zip(rows, mask, strict=True) if include])["mean"]
        prompt_rows.append(row)
    position_rows = []
    for source in BIN_NAMES:
        mask = np.array([row["position_bin"] == source for row in rows])
        for target in BIN_NAMES:
            position_rows.append({"query_bin": source, "donor_bin": target, "query_nodes": int(mask.sum()),
                "query_prompts": len(set(owners[mask])),
                **{name: balanced_mean([r[f"{name}_{target}"] for r in rows], owners, mask)["mean"]
                   for name in ("geometric_mass", "contribution_mass")}})
    associations = {}
    for key in ("resultant_fraction", "weighted_relative_gap", "nearest_distance"):
        value = np.array([r[key] for r in rows], dtype=float)
        valid = np.isfinite(actual) & np.isfinite(value)
        rho = spearmanr(actual[valid], value[valid]).statistic if valid.sum() > 2 and np.ptp(value[valid]) > 0 and np.ptp(actual[valid]) > 0 else np.nan
        associations[key] = {"paired_nodes": int(valid.sum()), "pooled_spearman_rho": finite(rho)}
    controls = {}
    for name in ("v2_random", "within_prompt_shuffle"):
        controls[name] = {}
        for scope in scopes:
            selected = [r for r in comparison_rows if r["comparator"] == name and r["scope"] == scope]
            controls[name][scope] = {key: describe([r[key] for r in selected]) for key in
                ("paired_nodes", "paired_prompts", "actual_cosine", "control_cosine", "paired_gap", "actual_resultant_fraction", "control_resultant_fraction")}
    summary = {"status": "DESCRIPTIVE_CALIBRATION_DIAGNOSTIC", "model_loads": 0, "model_forward_calls": 0,
               "evaluation_prefix_queries": 0, "nodes": len(rows), "prompts": len(prompts),
               "recovery_status_counts": dict(Counter(r["recovery_status"] for r in rows)),
               "distance_supported_nodes": int(supported.sum()), "defined_recovery_nodes": int(np.isfinite(actual).sum()),
               "supported_recovery_nodes": int((supported & np.isfinite(actual)).sum()),
               "supported_recovery_equal_prompt": balanced_mean(actual, owners, supported),
               "statistics": stats, "nearest_paired": {name: paired_comparison(actual, nearest, owners, mask) for name, mask in scopes.items()},
               "controls": controls, "associations": associations, "scope": method["scope"]}
    arrays = {"neighbor_indices": indices, "distances": distances, "weights": weights, "prediction": metrics["prediction"],
              "shuffle_indices": permutations}
    return summary, rows, prompt_rows, position_rows, comparison_rows, arrays


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(path, root=ROOT):
    protocol = json.loads(path.read_text())
    validate(protocol, root)
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(output)
    field, prompts, method, _ = load_inputs(root)
    started = utc_now()
    summary, rows, prompt_rows, positions, controls, arrays = analyze(field, prompts, method)
    validate(protocol, root)
    summary.update({"started_utc": started, "completed_utc": utc_now(), "protocol": file_receipt(path),
                    "readout_fingerprint": field.fingerprint()})
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("nodes.csv", rows), ("prompts.csv", prompt_rows), ("position_mix.csv", positions), ("controls.csv", controls)):
        write_csv(output / name, values)
    with (output / "replay_arrays.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    save_json(output / "summary.json", summary)
    save_json(output / "manifest.json", {"protocol": file_receipt(path),
              "outputs": [file_receipt(p) for p in sorted(output.iterdir())]})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", type=Path)
    group.add_argument("--run", type=Path)
    group.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.freeze:
        result = freeze(args.freeze.absolute())
        print(json.dumps({"frozen_utc": result["frozen_utc"], "readout": result["readout"]}, indent=2))
    elif args.run:
        result = run(args.run.absolute())
        print(json.dumps(result, indent=2))
    else:
        validate(json.loads(args.validate.read_text()))
        print("calibration diagnostic inputs validated; no model or evaluation arrays loaded")


if __name__ == "__main__":
    main()
