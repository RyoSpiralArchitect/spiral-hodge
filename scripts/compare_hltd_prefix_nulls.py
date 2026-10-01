#!/usr/bin/env python3
"""Compare fixed calibration nulls before inspecting saved v2 responses."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import diagnose_hltd_prefix_calibration as cal
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, verify_frozen_files
from scripts.run_hltd_precision_gate import save_json, utc_now

BASE = "docs/data/hltd_prefix_null_comparison"
PLAN = BASE + "/plan.json"
PROTOCOL = BASE + "/protocol.json"
OUTPUT = BASE + "/calibration"
OLD = "docs/data/hltd_prefix_calibration_readout"
PINS = {
    OLD + "/protocol.json": "a58a24fad8b05015cfebf9a88a5bf6f49c2f405e3add042bbe80bae1ff1ce681",
    OLD + "/result/manifest.json": "0fe5745e8d501ea2437d540eacb68ab4d2b4b41d201fa2fb8d4907eb3f83c916",
    cal.REFERENCE: cal.REFERENCE_SHA,
    "docs/data/hltd_prefix_transfer_v2/result/manifest.json": "98f68c894a4f7a7cea37c1bcc4cbc1bc14b98cbb290da7d76b60acfad48a8819",
}
SOURCES = ["scripts/compare_hltd_prefix_nulls.py", "scripts/analyze_hltd_prefix_nulls_v2.py",
           "tests/test_hltd_prefix_nulls.py", "scripts/audit_hltd_prefix_transfer.py",
           "scripts/analyze_hltd_prefix_v2.py", "scripts/analyze_hltd_prefix_transfer_gate.py",
           "scripts/run_hltd_prefix_v2.py", "scripts/run_hltd_prefix_transfer_gate.py"]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verified_calibration(root=ROOT):
    prior = cal.pinned_json(root / (OLD + "/protocol.json"), PINS[OLD + "/protocol.json"])
    cal.validate(prior, root)
    manifest = cal.pinned_json(root / (OLD + "/result/manifest.json"), PINS[OLD + "/result/manifest.json"])
    if manifest["protocol"] != file_receipt(root / (OLD + "/protocol.json")):
        raise ValueError("calibration manifest protocol mismatch")
    verify_frozen_files({"frozen_files": manifest["outputs"]}, root)
    return cal.load_inputs(root), manifest


def contract(root=ROOT):
    (_, _, method, inputs), manifest = verified_calibration(root)
    plan = json.loads((root / PLAN).read_text())
    for key in ("random_seeds", "shuffle_draws", "shuffle_seed"):
        if plan["calibration"][key] != method[key]:
            raise ValueError("existing null settings changed")
    # Pin the v2 manifest without opening any of its response/observation files.
    for name, sha in PINS.items():
        cal.pinned_json(root / name, sha)
    records = inputs + manifest["outputs"] + [file_receipt(root / p) for p in [PLAN, *PINS, *SOURCES]]
    unique = {r["path"]: r for r in records}
    return {"diagnostic_id": plan["diagnostic_id"], "plan": plan,
            "frozen_files": sorted(unique.values(), key=lambda r: r["path"]),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "outputs": [OUTPUT, BASE + "/v2"]}


def validate(protocol, root=ROOT):
    verify_frozen_files(protocol, root)
    expected = contract(root)
    if set(protocol) != set(expected) | {"frozen_utc"} or any(protocol[k] != v for k, v in expected.items()):
        raise ValueError("follow-up contract changed")


def freeze(path, root=ROOT):
    if path.exists() or any((root / p).exists() for p in (OUTPUT, BASE + "/v2")):
        raise FileExistsError("follow-up already exists")
    protocol = {**contract(root), "frozen_utc": utc_now()}
    validate(protocol, root)
    save_json(path, protocol)
    return protocol


def position_bins(atlas, prompts):
    counts = np.array([prompts[p]["token_count"] for p in atlas.node_prompts])
    relative = np.array(atlas.node_tokens) / (counts - 1)
    return np.minimum((3 * relative).astype(int), 2)


def stratified_indices(owners, bins, draws, seed):
    owners, bins = np.asarray(owners), np.asarray(bins)
    if owners.ndim != 1 or bins.shape != owners.shape or draws < 1 or not np.isin(bins, [0, 1, 2]).all():
        raise ValueError("invalid shuffle strata")
    groups = [np.flatnonzero((owners == owner) & (bins == b)) for owner, b in sorted(set(zip(owners, bins, strict=True)))]
    result = np.tile(np.arange(len(owners)), (draws, 1))
    rng = np.random.default_rng(seed)
    for row in result:
        for indices in groups:
            row[indices] = rng.permutation(indices)
    return result


def common_mask(actual, controls):
    result = np.isfinite(actual)
    for values in controls.values():
        if values.ndim != 2 or values.shape[1] != len(actual) or not len(values):
            raise ValueError("invalid null cosine grid")
        result &= np.isfinite(values).all(axis=0)
    return result


def compare_scores(actual, controls, owners, supported):
    common = common_mask(actual, controls)
    draws, comparisons, prompt_rows = [], [], []
    for scope, eligible in (("all", np.ones(len(actual), dtype=bool)), ("supported", supported)):
        shared = common & eligible
        for name, scores in {"actual": actual[None, :], **controls}.items():
            for draw, score in enumerate(scores):
                for matching, mask in (("common", shared), ("per_draw", eligible)):
                    draws.append({"scope": scope, "comparator": name, "draw": draw, "matching": matching,
                                  **cal.paired_comparison(actual, score, owners, mask)})
            selected = [r for r in draws if r["scope"] == scope and r["comparator"] == name and r["matching"] == "common"]
            stats = cal.describe([r["control_cosine"] for r in selected])
            comparisons.append({"scope": scope, "comparator": name, "draws": len(scores),
                                "paired_nodes": int(shared.sum()), "paired_prompts": selected[0]["paired_prompts"],
                                "mean_cosine": stats["mean"], "draw_q10": stats["q10"], "draw_q90": stats["q90"],
                                "actual_minus_control": cal.describe([r["paired_gap"] for r in selected])["mean"]})
            # Every draw has identical eligible nodes, so these averages commute.
            averaged = np.full(len(actual), np.nan)
            averaged[shared] = scores[:, shared].mean(axis=0)
            for row in cal.prompt_means(averaged, owners, shared):
                prompt_rows.append({"scope": scope, "comparator": name, **row})
    return common, draws, comparisons, prompt_rows


def analyze(field, prompts, plan):
    atlas = field.atlas
    owners = np.asarray(atlas.node_prompts)
    indices, distances, weights = cal.donor_lookup(field)
    prediction = cal.aggregate(atlas.coexact, indices, weights)
    actual = cal.cosines(atlas.coexact, prediction, atlas.min_chart_norm)
    supported = (distances[:, 0] <= atlas.support_limit) & (distances[:, -1] <= field.kth_support_limit)
    bins = position_bins(atlas, prompts)
    plain = cal.shuffled_indices(owners, plan["shuffle_draws"], plan["shuffle_seed"])
    stratified = stratified_indices(owners, bins, plan["shuffle_draws"], plan["shuffle_seed"])

    def score(vectors):
        return cal.cosines(atlas.coexact, cal.aggregate(vectors, indices, weights), atlas.min_chart_norm)

    controls = {"v2_random": np.stack([score(cal.random_vectors(atlas.coexact, seed)) for seed in plan["random_seeds"]]),
                "within_prompt_shuffle": np.stack([score(atlas.coexact[p]) for p in plain]),
                "within_prompt_position_shuffle": np.stack([score(atlas.coexact[p]) for p in stratified])}
    common, draws, comparisons, prompt_rows = compare_scores(actual, controls, owners, supported)
    strata = []
    for owner, b in sorted(set(zip(owners, bins, strict=True))):
        nodes = np.flatnonzero((owners == owner) & (bins == b))
        strata.append({"prompt_id": str(owner), "position_bin": cal.BIN_NAMES[b], "nodes": len(nodes),
                       "singleton": len(nodes) == 1,
                       "fixed_point_fraction": float((stratified[:, nodes] == nodes[None, :]).mean())})
    masks = [{"node_index": i, "prompt_id": str(owners[i]), "actual_defined": bool(np.isfinite(actual[i])),
              "distance_supported": bool(supported[i]), "common_defined": bool(common[i]),
              **{name + "_invalid_draws": int((~np.isfinite(values[:, i])).sum()) for name, values in controls.items()}}
             for i in range(len(actual))]
    summary = {"status": "DESCRIPTIVE_CALIBRATION_NULL_COMPARISON", "model_loads": 0, "model_forward_calls": 0,
               "evaluation_prefix_queries": 0, "nodes": len(actual), "prompts": len(prompts),
               "actual_defined_nodes": int(np.isfinite(actual).sum()), "common_nodes": int(common.sum()),
               "excluded_by_null_validity": int((np.isfinite(actual) & ~common).sum()),
               "common_supported_nodes": int((common & supported).sum()),
               "strata": len(strata), "singleton_strata": sum(s["singleton"] for s in strata),
               "stratified_fixed_point_fraction": float((stratified == np.arange(len(actual))).mean()),
               "unrestricted_actual_mean": cal.balanced_mean(actual, owners)["mean"], "comparison": comparisons,
               "scope": plan["scope"], "reference_ranges": "Across fixed null draws; not confidence intervals."}
    arrays = {"neighbor_indices": indices, "distances": distances, "weights": weights, "prediction": prediction,
              "actual": actual, "common_mask": common, "supported_mask": supported,
              "shuffle_indices": plain, "stratified_indices": stratified, **controls}
    return summary, {"draws.csv": draws, "comparison.csv": comparisons, "prompts.csv": prompt_rows,
                     "masks.csv": masks, "strata.csv": strata}, arrays


def verify_old_replay(tables, arrays, root=ROOT):
    with np.load(root / (OLD + "/result/replay_arrays.npz"), allow_pickle=False) as old:
        for key in old.files:
            before, after = old[key], arrays[key]
            if before.shape != after.shape or before.dtype != after.dtype or before.tobytes() != after.tobytes():
                raise ValueError("old calibration array differs: " + key)
    indexed = {(r["comparator"], r["scope"], int(r["draw"])): r for r in tables["draws.csv"] if r["matching"] == "per_draw"}
    errors = []
    for row in read_csv(root / (OLD + "/result/controls.csv")):
        new = indexed[row["comparator"], row["scope"], int(row["draw"])]
        for key in ("paired_nodes", "paired_prompts", "actual_cosine", "control_cosine", "paired_gap"):
            errors.append(abs(float(row[key]) - new[key]))
    if max(errors) > 1e-12:
        raise ValueError("old paired control statistics changed")
    return {"passed": True, "prior_array_replay": "byte-identical values", "max_statistic_error": max(errors)}


def run(path, root=ROOT):
    protocol = json.loads(path.read_text())
    validate(protocol, root)
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(output)
    (field, prompts, _, _), _ = verified_calibration(root)
    started = utc_now()
    summary, tables, arrays = analyze(field, prompts, protocol["plan"]["calibration"])
    summary["prior_replay"] = verify_old_replay(tables, arrays, root)
    validate(protocol, root)
    summary.update({"started_utc": started, "completed_utc": utc_now(), "protocol": file_receipt(path)})
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        cal.write_csv(output / name, rows)
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
        print(json.dumps({"frozen_utc": result["frozen_utc"], "diagnostic_id": result["diagnostic_id"]}))
    elif args.run:
        print(json.dumps(run(args.run.absolute()), indent=2, allow_nan=False))
    else:
        validate(json.loads(args.validate.read_text()))
        print("follow-up contract valid; no v2 response files or model assets loaded")


if __name__ == "__main__":
    main()
