#!/usr/bin/env python3
"""One prospectively frozen v2 interpolation run; v1 evidence stays immutable."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hltd_prefix_interpolation as interpolation
from scripts import hltd_prefix_transfer as base
from scripts import run_hltd_prefix_transfer_gate as v1
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, verify_frozen_files
from scripts.run_hltd_fresh_gate import novelty_audit
from scripts.run_hltd_precision_gate import save_json, utc_now
from scripts.run_hltd_steering_suite import read_suite

DESIGN = "docs/data/hltd_prefix_transfer_v2/design.json"
DESIGN_SHA = "c6c583d7f058ef15da2416528bf74a2db66769aea7090976cac0cae3f90df101"
SUITE = "data/hltd_prefix_transfer_v2/prompts.jsonl"
SUITE_SHA = "b46faf9493ee44511663c08e2602f389937fb55172744bf633ec019abb9902b7"
RUN_ROOT = "spiral_out_hltd_prefix_transfer_v2"
SOURCES = ["scripts/hltd_prefix_interpolation.py", "scripts/run_hltd_prefix_v2.py",
           "scripts/analyze_hltd_prefix_v2.py", "tests/test_hltd_prefix_v2.py"]


def pinned(path, sha256, root=ROOT):
    payload = (root / path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"canonical input changed: {path}")
    return payload


def design(root=ROOT):
    return json.loads(pinned(DESIGN, DESIGN_SHA, root))


def merge_receipts(records, root):
    merged = {}
    for row in records:
        key = (root / row["path"]).resolve()
        if key in merged and any(merged[key][k] != row[k] for k in ("sha256", "bytes")):
            raise ValueError("conflicting frozen receipts")
        merged.setdefault(key, row)
    return sorted(merged.values(), key=lambda r: r["path"])


def load_prior(root=ROOT):
    d = design(root)
    previous = json.loads(pinned(**d["prior_protocol"], root=root))
    v1.validate_execution(previous, root)
    manifest = json.loads(pinned(**d["prior_manifest"], root=root))
    verdict = json.loads(pinned(**d["prior_verdict"], root=root))
    if verdict["status"] != "INSUFFICIENT_COVERAGE" or verdict["nonzero_rows"] != 0:
        raise ValueError("unexpected v1 disposition")
    records = [manifest["protocol"], *manifest["raw_files"], *manifest["derived_files"]]
    verify_frozen_files({"frozen_files": records}, root)
    old_run = root / previous["run_root"]
    atlas = v1.read_field(old_run / "atlas.npz", old_run / "atlas_metadata.json")
    atlas_receipt = json.loads((old_run / "atlas_freeze.json").read_text())
    if atlas.fingerprint() != atlas_receipt["fingerprint"] or previous["runtime"] != v1.runtime_snapshot():
        raise ValueError("changed v1 atlas/runtime")
    return previous, atlas, merge_receipts([*previous["frozen_files"], *records], root)


def inventory(root=ROOT):
    d = design(root)
    pinned(SUITE, SUITE_SHA, root)
    rows = read_suite(root / SUITE)
    if Counter(r["split"] for r in rows) != Counter(d["data"]["split_counts"]):
        raise ValueError("changed v2 split counts")
    for split, count in d["data"]["split_counts"].items():
        expected = Counter({family: count // 4 for family in d["data"]["families"]})
        if Counter(r["family"] for r in rows if r["split"] == split) != expected:
            raise ValueError("changed v2 family balance")
    history_paths = sorted(str(p.relative_to(root)) for p in (root / "data").rglob("*.jsonl")
                           if p != root / SUITE)
    history = [r for name in history_paths for r in read_suite(root / name)]
    novelty = novelty_audit(rows, history)
    scenarios = [r.get("scenario_id") for r in rows]
    old_scenarios = {r["scenario_id"] for r in history if "scenario_id" in r}
    if (any(not isinstance(s, str) or not s.strip() for s in scenarios)
            or len(set(scenarios)) != len(scenarios) or set(scenarios) & old_scenarios):
        raise ValueError("missing or overlapping authored scenarios")
    return rows, novelty, history_paths


def contract(root=ROOT):
    d = design(root)
    previous, atlas, old_receipts = load_prior(root)
    rows, novelty, history = inventory(root)
    spec = {name: copy.deepcopy(previous["spec"][name]) for name in
            ("geometry", "evaluation", "analysis", "runtime_gate", "future_invariance_gate")}
    spec["readout"] = d["lookup"]
    spec["protocol_id"] = d["protocol_id"]
    tokenizer = v1.preparation.load_tokenizer(Path(previous["model_path"]))
    prompts = v1.preparation.tokenize_inventory(rows, tokenizer, spec)
    field = interpolation.calibrate_readout(atlas, neighbors=d["lookup"]["neighbors"], quantile=d["lookup"]["support_quantile"])
    names = [DESIGN, SUITE, *SOURCES, *history, d["prior_manifest"]["path"], d["prior_verdict"]["path"]]
    return {"schema_version": 2, "protocol_id": d["protocol_id"], "design": d, "spec": spec,
            "prompts": prompts, "novelty_audit": novelty, "run_root": RUN_ROOT,
            "model_path": previous["model_path"], "runtime": previous["runtime"],
            "readout": field.receipt(), "readout_fingerprint": field.fingerprint(),
            "execution_details": v1.DETAILS,
            "frozen_files": merge_receipts([*old_receipts, *v1.preparation.local_receipts(names, root)], root)}


def validate(protocol, root=ROOT):
    expected = contract(root)
    if set(protocol) != set(expected) | {"frozen_utc"}:
        raise ValueError("changed v2 protocol keys")
    for key in expected:
        if protocol[key] != expected[key]:
            raise ValueError(f"changed v2 execution contract: {key}")
    verify_frozen_files(protocol, root)


def freeze(path, root=ROOT):
    if path.exists() or (root / RUN_ROOT).exists():
        raise FileExistsError("v2 protocol/run exists; no automatic retry")
    protocol = contract(root)
    protocol["frozen_utc"] = utc_now()
    validate(protocol, root)
    save_json(path, protocol)
    return protocol


def load_readout(protocol, root=ROOT):
    old_run = root / v1.RUN_ROOT
    atlas = v1.read_field(old_run / "atlas.npz", old_run / "atlas_metadata.json")
    record = protocol["readout"]
    field = interpolation.InterpolatedField(atlas, record["neighbors"], record["bandwidth"], record["kth_support_limit"])
    if field.receipt() != record or field.fingerprint() != protocol["readout_fingerprint"]:
        raise ValueError("readout bytes/parameters changed")
    return field


def delta_grid(field, hidden, spec, seed):
    jobs, values = [], []
    for component in spec["evaluation"]["components"]:
        for alpha in spec["evaluation"]["alphas"]:
            direction = interpolation.query_hidden(field, hidden, seed=seed, alpha=alpha)
            values.append(direction.coexact_delta if component == "coexact" else direction.random_delta)
            jobs.append({"seed": seed, "component": component, "alpha": alpha, **direction.receipt()})
    deltas = np.asarray(values, dtype=np.float32)
    if not np.isfinite(deltas).all():
        raise ValueError("nonfinite v2 nominal delta")
    return jobs, deltas


def direction_receipt(field, hidden, spec):
    initial = interpolation.query_hidden(field, hidden, seed=0, alpha=1).receipt()
    grids = [delta_grid(field, hidden, spec, seed) for seed in spec["evaluation"]["seeds"]]
    jobs = [j for group, _ in grids for j in group]
    statuses = {j["status"] for j in jobs}
    status = next((s for s in ("OUT_OF_SUPPORT", "INACTIVE_COEXACT", "INACTIVE_RANDOM") if s in statuses), "SUPPORTED")
    return {**initial, "seed0_status": initial["status"], "status": status,
            "all_direction_receipts": jobs,
            "nominal_float32_delta_sha256": [base.array_digest(values) for _, values in grids]}


def observe(runtime, field, token_ids, length, spec):
    cache = {}

    def provider(ids):
        cache["observed_ids"] = ids
        cache.update(runtime.capture(ids))
        return cache["hidden"][-1]

    interpolation.direction_at_prefix(field, token_ids, length, provider, seed=0, alpha=1)
    zeros = np.zeros((12, len(cache["hidden"][-1])), dtype=np.float32)
    zero_logits = runtime.steer(cache["observed_ids"], zeros)
    if zero_logits.shape != cache["logits"].shape or not np.isfinite(zero_logits).all():
        raise ValueError("invalid zero logits")
    errors = {"zero_hook_max_abs": float(np.max(np.abs(zero_logits - cache["logits"]))),
              "row_spread_max_abs": float(np.max(np.abs(cache["logits"] - cache["logits"][0])))}
    if (errors["zero_hook_max_abs"] > spec["runtime_gate"]["zero_hook_logit_max_abs"]
            or errors["row_spread_max_abs"] != 0 or not all(math.isfinite(v) for v in errors.values())):
        raise ValueError("v2 zero/batch audit failed")
    cache["receipt"] = {**direction_receipt(field, cache["hidden"][-1], spec), **errors,
        "observed_ids": list(cache["observed_ids"]), "prefix_hidden_sha256": base.array_digest(cache["hidden"]),
        "baseline_logits_sha256": base.array_digest(cache["logits"]), "zero_logits_sha256": base.array_digest(zero_logits),
        "support_limit": field.atlas.support_limit, "kth_support_limit": field.kth_support_limit}
    cache["zero_logits"] = zero_logits
    return cache


def pilot(runtime, field, protocol, output):
    records, arrays = [], {}
    for p in [p for p in protocol["prompts"] if p["split"] == "pilot"]:
        for length in protocol["spec"]["evaluation"]["prefix_lengths"]:
            ids = p["input_ids"]
            variants = [ids, ids[:length] + v1.DETAILS["pilot_suffix_replacement_ids"],
                        ids + v1.DETAILS["pilot_suffix_append_ids"], ids[:length]]
            receipts = []
            for i, variant in enumerate(variants):
                observation = observe(runtime, field, variant, length, protocol["spec"])
                for name in ("hidden", "logits", "zero_logits"):
                    arrays[f"{len(records)}_{i}_{name}"] = observation[name]
                receipts.append(observation["receipt"])
            passed = all(r == receipts[0] for r in receipts)
            records.append({"prompt_id": p["prompt_id"], "prefix_length": length, "byte_identical": passed, "variants": receipts})
            print(f"v2 pilot {p['prompt_id']} prefix={length}: invariant={passed}", flush=True)
    result = {"passed": len(records) == 12 and all(r["byte_identical"] for r in records),
              "cells": len(records), "nonzero_treatments": 0, "completed_utc": utc_now(), "records": records}
    v1.save_arrays(output / "pilot_observations.npz", **arrays)
    save_json(output / "pilot_audit.json", result)
    return result


def preflight(runtime, field, protocol, output):
    cells, caches, arrays = [], [], {}
    for p in [p for p in protocol["prompts"] if p["split"] == "evaluation"]:
        for length in protocol["spec"]["evaluation"]["prefix_lengths"]:
            observation = observe(runtime, field, p["input_ids"], length, protocol["spec"])
            for name in ("hidden", "logits", "zero_logits"):
                arrays[f"{len(cells)}_{name}"] = observation[name]
            cell = {"prompt_id": p["prompt_id"], "family": p["family"], "prefix_length": length, **observation["receipt"]}
            cells.append(cell)
            caches.append((p, length, observation))
            print(f"v2 support {p['prompt_id']} prefix={length}: {cell['status']}", flush=True)
    result = {"passed": len(cells) == 60 and all(c["status"] == "SUPPORTED" for c in cells),
              "planned_cells": 60, "observed_cells": len(cells), "completed_utc": utc_now(), "cells": cells,
              "status_counts": dict(Counter(c["status"] for c in cells))}
    v1.save_arrays(output / "evaluation_observations.npz", **arrays)
    save_json(output / "support_preflight.json", result)
    return result, caches


def treat(runtime, field, protocol, caches, output, ledger):
    spec = protocol["spec"]
    with (output / "raw_treatments.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = None
        for p, length, observation in caches:
            hidden = observation["hidden"][-1]
            # Target identity is accessed only by the scorer, after prefix-only query construction.
            target = p["input_ids"][length]
            baseline = float(v1._log_softmax(observation["logits"][0])[target])
            for position, seed in enumerate(spec["evaluation"]["seeds"]):
                jobs, deltas = delta_grid(field, hidden, spec, seed)
                if (any(j["status"] != "SUPPORTED" for j in jobs)
                        or base.array_digest(deltas) != observation["receipt"]["nominal_float32_delta_sha256"][position]):
                    raise ValueError("v2 direction changed after preflight")
                ledger["nonzero_forward_rows_attempted"] += len(deltas)
                steered = runtime.steer(observation["observed_ids"], deltas)
                if steered.shape != observation["logits"].shape:
                    raise ValueError("treatment batch shape changed")
                for job, delta, logits in zip(jobs, deltas, steered, strict=True):
                    lp = float(v1._log_softmax(logits)[target])
                    probability = math.exp(lp)
                    if not math.isfinite(lp) or not 0 < probability <= 1 or not 0 < math.exp(baseline) <= 1:
                        raise ValueError("unrepresentable target probability")
                    row = {"prompt_id": p["prompt_id"], "family": p["family"], "prefix_length": length,
                           "layer": 7, "target_id": target,
                           **{key: job[key] for key in ("seed", "component", "alpha", "node_index", "distance", "dose_scale")},
                           "nominal_delta_sha256": base.array_digest(delta), "nominal_delta_norm": float(np.linalg.norm(delta.astype(float))),
                           "next_token_logprob_base": baseline, "next_token_prob_base": math.exp(baseline),
                           "next_token_logprob_steered": lp, "next_token_prob_steered": probability,
                           "next_token_logprob_delta": lp - baseline}
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    ledger["nonzero_rows"] += 1
                handle.flush()
            print(f"v2 treated {p['prompt_id']} prefix={length}: {ledger['nonzero_rows']}/5760 rows", flush=True)
    if ledger["nonzero_rows"] != spec["evaluation"]["planned_treatment_rows"]:
        raise ValueError("incomplete v2 treatment grid")


def stages(runtime, field, protocol, output, ledger):
    before = field.fingerprint()
    save_json(output / "readout_freeze.json", {"frozen_utc": utc_now(), "protocol": ledger["protocol"],
              "readout": field.receipt(), "fingerprint": before, "nonzero_rows": 0})
    audit = pilot(runtime, field, protocol, output)
    if not audit["passed"]:
        return "INVALID_FUTURE_DEPENDENCE"
    support, caches = preflight(runtime, field, protocol, output)
    if field.fingerprint() != before:
        raise ValueError("v2 field mutated during preflight")
    if not support["passed"]:
        return "INSUFFICIENT_COVERAGE"
    save_json(output / "pre_treatment_gate.json", {"passed": True, "completed_utc": utc_now(),
        "nonzero_rows_at_gate": ledger["nonzero_rows"], "nonzero_forward_rows_attempted": ledger["nonzero_forward_rows_attempted"],
        "readout_fingerprint": before,
        "files": [file_receipt(output / n) for n in ("readout_freeze.json", "pilot_audit.json", "support_preflight.json")]})
    treat(runtime, field, protocol, caches, output, ledger)
    if field.fingerprint() != before:
        raise ValueError("v2 field mutated during treatment")
    return "TREATMENTS_COMPLETE_PENDING_ANALYSIS"


def run(path, root=ROOT):
    protocol = json.loads(path.read_text())
    validate(protocol, root)
    field = load_readout(protocol, root)
    output = root / RUN_ROOT
    output.mkdir(exist_ok=False)
    ledger = {"protocol": file_receipt(path), "started_utc": utc_now(), "status": "RUNNING",
              "nonzero_rows": 0, "nonzero_forward_rows_attempted": 0}
    save_json(output / "execution_started.json", ledger)
    model, before, error = None, None, None
    try:
        import torch

        if (not torch.backends.mps.is_available() or torch.is_autocast_enabled("mps")
                or torch.is_autocast_enabled("cpu") or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"):
            raise ValueError("MPS/native-FP32 gate failed")
        model, tokenizer = v1._load_model_and_tokenizer(protocol["model_path"], device="mps", local_files_only=True,
                                                       trust_remote_code=False, torch_dtype="float32")
        model.config.use_cache = False
        if {p.device.type for p in model.parameters()} != {"mps"} or model.config._attn_implementation != "sdpa":
            raise ValueError("unexpected device or attention")
        before = v1.audit_model_weight_bytes(model, Path(protocol["model_path"]) / "model.safetensors", "float32")
        if before["parameter_tensors"] != 148 or before["parameter_count"] != 124439808:
            raise ValueError("unexpected GPT-2 shape")
        old_load = json.loads((root / v1.RUN_ROOT / "load_audit.json").read_text())
        if before != old_load["model"]:
            raise ValueError("cached atlas and current model weight audits differ")
        for p in protocol["prompts"]:
            if tokenizer.encode(p["text"], add_special_tokens=False, truncation=False) != p["input_ids"]:
                raise ValueError("v2 token IDs changed")
        save_json(output / "load_audit.json", {"model": before, "runtime": v1.runtime_snapshot(), "model_load_count": 1,
                  "device": "mps", "attention": "sdpa", "autocast": False, "mps_fallback": False})
        ledger["gate_status"] = stages(v1.PrefixRuntime(model, protocol["spec"]), field, protocol, output, ledger)
    except Exception as exc:
        error = exc
        ledger.update({"gate_status": "INVALID_INPUT_OR_NUMERICS", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            if before is not None:
                after = v1.audit_model_weight_bytes(model, Path(protocol["model_path"]) / "model.safetensors", "float32")
                if after != before:
                    raise ValueError("model weights changed")
                save_json(output / "final_weight_audit.json", {"passed": True, "model": after, "completed_utc": utc_now()})
            validate(protocol, root)
        except Exception as exc:
            error = exc
            ledger.update({"gate_status": "INVALID_INPUT_OR_NUMERICS", "final_audit_error": f"{type(exc).__name__}: {exc}"})
    ledger.update({"status": "FAILED" if error else "COMPLETED", "completed_utc": utc_now()})
    save_json(output / "execution_receipt.json", ledger)
    if error:
        raise error
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", type=Path)
    group.add_argument("--run", type=Path)
    group.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.freeze:
        p = freeze(args.freeze.absolute())
        print(json.dumps({"frozen_utc": p["frozen_utc"], "readout": p["readout"], "novelty_rows": p["novelty_audit"]["fresh_rows"]}, indent=2))
    elif args.run:
        print(json.dumps(run(args.run.absolute()), indent=2))
    else:
        validate(json.loads(args.validate.read_text()))
        print("v2 protocol validated; no model parameters loaded")


if __name__ == "__main__":
    main()
