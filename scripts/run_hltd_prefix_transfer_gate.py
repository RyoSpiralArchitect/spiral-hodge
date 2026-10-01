#!/usr/bin/env python3
"""Freeze and execute the prepared prefix-transfer gate, without retries."""
from __future__ import annotations

import argparse
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

from scripts import hltd_prefix_transfer as transfer
from scripts import prepare_hltd_prefix_gate as preparation
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, verify_frozen_files
from scripts.run_hltd_l8_gate import runtime_snapshot
from scripts.run_hltd_precision_full_gate import audit_model_weight_bytes
from scripts.run_hltd_precision_gate import save_json, utc_now
from scripts.run_hltd_steering import _load_model_and_tokenizer, _logits_with_deltas, _log_softmax

PREPARATION = "docs/data/hltd_prefix_transfer/preparation.json"
PREPARATION_SHA256 = "5149d5dc71f5844ff5cd7e8389c5bbdc03d69580f0cb1f7b33516c0940d0dd6d"
RUN_ROOT = "spiral_out_hltd_prefix_transfer_v1"
EXECUTION_SOURCES = [
    "scripts/run_hltd_prefix_transfer_gate.py",
    "scripts/analyze_hltd_prefix_transfer_gate.py",
    "scripts/audit_hltd_prefix_transfer.py",
    "tests/test_hltd_prefix_execution.py",
]
DETAILS = {
    "calibration_batch_size": 1, "prefix_batch_size": 12,
    "hidden_layer_index": 7, "hook_block_index": 6, "use_cache": False,
    "pilot_suffix_replacement_ids": [0, 1, 2, 3],
    "pilot_suffix_append_ids": [4, 5, 6, 7],
    "pilot_nonzero_deltas": "compare all nominal float32 delta bytes without applying them",
    "evaluation_order": "preparation prompt order, prefix 8/16/24, seed 0..7, component, alpha",
    "bootstrap_rng": "numpy.random.default_rng(1729), 5000 by 20 integer indices over sorted prompt IDs",
    "preparation_receipts": "validate the pinned preparation and its original file inventory; new stage sources are additional",
    "nonfinite_direction_status": "INSUFFICIENT_COVERAGE; other runtime/input errors are INVALID_INPUT_OR_NUMERICS",
}


def load_preparation(root: Path = ROOT) -> dict:
    payload = (root / PREPARATION).read_bytes()
    if hashlib.sha256(payload).hexdigest() != PREPARATION_SHA256:
        raise ValueError("canonical preparation changed")
    prepared = json.loads(payload)
    verify_frozen_files(prepared, root)
    if prepared["spec"] != preparation.load_spec(root):
        raise ValueError("changed prepared specification")
    if prepared["preparation_environment"] != preparation.preparation_environment():
        raise ValueError("changed preparation environment")
    original_models = [r for r in prepared["frozen_files"] if Path(r["path"]).is_absolute()]
    if original_models != preparation.model_receipts(Path(prepared["model_path"]), root):
        raise ValueError("changed prepared model inventory")
    return prepared


def execution_contract(root: Path = ROOT) -> dict:
    prepared = load_preparation(root)
    return {
        "schema_version": 1, "protocol_id": "hltd-prefix-transfer-execution-v1",
        "preparation_sha256": PREPARATION_SHA256, "run_root": RUN_ROOT,
        "spec": prepared["spec"], "prompts": prepared["prompts"],
        "model_path": prepared["model_path"], "runtime": runtime_snapshot(), "execution_details": DETAILS,
        "frozen_files": prepared["frozen_files"] + preparation.local_receipts([PREPARATION, *EXECUTION_SOURCES], root),
    }


def validate_execution(protocol: dict, root: Path = ROOT) -> None:
    contract = execution_contract(root)
    if set(protocol) != set(contract) | {"frozen_utc"}:
        raise ValueError("changed execution protocol keys")
    for key, value in contract.items():
        if protocol[key] != value:
            raise ValueError(f"changed execution contract: {key}")
    verify_frozen_files(protocol, root)


def freeze(path: Path, root: Path = ROOT) -> dict:
    if path.exists() or (root / RUN_ROOT).exists():
        raise FileExistsError("execution protocol or run already exists; no retry")
    protocol = execution_contract(root)
    tokenizer = preparation.load_tokenizer(Path(protocol["model_path"]))
    rows, _, _ = preparation.corpus_inventory(root)
    if preparation.tokenize_inventory(rows, tokenizer, protocol["spec"]) != protocol["prompts"]:
        raise ValueError("prepared token IDs changed")
    protocol["frozen_utc"] = utc_now()
    validate_execution(protocol, root)
    save_json(path, protocol)
    return protocol


def save_arrays(path: Path, **arrays) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def persist_field(field: transfer.FrozenPrefixField, output: Path) -> dict:
    arrays = output / "atlas.npz"
    meta = output / "atlas_metadata.json"
    save_arrays(arrays, **{key: getattr(field, key) for key in ("mean", "components", "points", "coexact")})
    save_json(meta, {key: getattr(field, key) for key in (
        "node_prompts", "node_tokens", "calibration_hashes", "dose_scale", "support_limit", "min_chart_norm")})
    restored = read_field(arrays, meta)
    if restored.fingerprint() != field.fingerprint():
        raise ValueError("atlas serialization changed bytes")
    return {"fingerprint": field.fingerprint(), "files": [file_receipt(arrays), file_receipt(meta)]}


def read_field(arrays: Path, metadata: Path) -> transfer.FrozenPrefixField:
    with np.load(arrays, allow_pickle=False) as data:
        return transfer.FrozenPrefixField(**dict(data), **json.loads(metadata.read_text()))


def delta_grid(field, hidden, spec, seed: int) -> tuple[list[dict], np.ndarray]:
    jobs, values = [], []
    for component in spec["evaluation"]["components"]:
        for alpha in spec["evaluation"]["alphas"]:
            direction = transfer.query_hidden(field, hidden, seed=seed, alpha=alpha)
            delta = direction.coexact_delta if component == "coexact" else direction.random_delta
            values.append(delta)
            jobs.append({"seed": seed, "component": component, "alpha": alpha, **direction.receipt()})
    nominal = np.asarray(values, dtype=np.float32)
    if not np.isfinite(nominal).all():
        raise ValueError("nonfinite nominal delta")
    return jobs, nominal


class PrefixRuntime:
    """The only model inputs are the explicitly supplied observed token IDs."""

    def __init__(self, model, spec):
        self.model = model
        self.spec = spec

    def inputs(self, ids):
        import torch

        values = torch.tensor([list(ids)], dtype=torch.long, device="mps")
        return {"input_ids": values, "attention_mask": torch.ones_like(values), "use_cache": False}

    def capture(self, ids, batch: int = 12) -> dict:
        import torch

        inputs = self.inputs(ids)
        repeated = {key: value.repeat(batch, 1) if torch.is_tensor(value) else value for key, value in inputs.items()}
        with torch.no_grad():
            output = self.model(**repeated, output_hidden_states=True, return_dict=True)
        if any(t.dtype != torch.float32 or t.device.type != "mps" for t in (output.logits, *output.hidden_states)):
            raise ValueError("forward dtype/device changed")
        hidden = output.hidden_states[7].detach().cpu().numpy()
        logits = output.logits[:, -1].detach().cpu().numpy()
        if not np.isfinite(hidden).all() or not np.isfinite(logits).all():
            raise ValueError("nonfinite prefix forward")
        if not np.array_equal(hidden, np.broadcast_to(hidden[:1], hidden.shape)):
            raise ValueError("hidden-state batch rows differ")
        return {"hidden": hidden[0].copy(), "logits": logits.copy()}

    def steer(self, ids, deltas):
        result = _logits_with_deltas(self.model, self.inputs(ids), layer=7,
                                     token_index=len(ids) - 1, deltas=deltas)
        if result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError("nonfinite or non-FP32 intervention logits")
        return result


def observe_prefix(runtime, field, token_ids, length, spec) -> dict:
    cache = {}

    def provider(observed):
        cache["observed_ids"] = observed
        cache.update(runtime.capture(observed))
        return cache["hidden"][-1]

    direction = transfer.direction_at_prefix(field, token_ids, length, provider, seed=0, alpha=1)
    hidden = cache["hidden"][-1]
    zero = runtime.steer(cache["observed_ids"], np.zeros((12, len(hidden)), dtype=np.float32))
    if zero.shape != cache["logits"].shape:
        raise ValueError("zero-hook logits shape changed")
    errors = {
        "zero_hook_max_abs": float(np.abs(zero - cache["logits"]).max()),
        "row_spread_max_abs": float(np.abs(cache["logits"] - cache["logits"][0]).max()),
    }
    if (not all(math.isfinite(v) for v in errors.values())
            or errors["zero_hook_max_abs"] > spec["runtime_gate"]["zero_hook_logit_max_abs"]
            or errors["row_spread_max_abs"] != 0):
        raise ValueError(f"zero/batch calibration failed: {errors}")
    grids = [delta_grid(field, hidden, spec, seed) for seed in spec["evaluation"]["seeds"]]
    receipt = {
        **direction.receipt(), **errors, "observed_ids": list(cache["observed_ids"]),
        "prefix_hidden_sha256": transfer.array_digest(cache["hidden"]),
        "baseline_logits_sha256": transfer.array_digest(cache["logits"]),
        "zero_logits_sha256": transfer.array_digest(zero),
        "all_direction_receipts": [job for jobs, _ in grids for job in jobs],
        "nominal_float32_delta_sha256": [transfer.array_digest(deltas) for _, deltas in grids],
    }
    return {**cache, "receipt": receipt, "direction": direction}


def run_pilot(runtime, field, protocol, output: Path) -> dict:
    records = []
    spec = protocol["spec"]
    for prompt in [p for p in protocol["prompts"] if p["split"] == "pilot"]:
        for length in spec["evaluation"]["prefix_lengths"]:
            ids = prompt["input_ids"]
            prefix = ids[:length]
            variants = [ids, prefix + DETAILS["pilot_suffix_replacement_ids"],
                        ids + DETAILS["pilot_suffix_append_ids"], prefix]
            observations = [observe_prefix(runtime, field, variant, length, spec) for variant in variants]
            receipts = [o["receipt"] for o in observations]
            passed = all(r == receipts[0] for r in receipts)
            records.append({"prompt_id": prompt["prompt_id"], "prefix_length": length,
                            "byte_identical": passed, "variants": receipts})
            print(f"pilot {prompt['prompt_id']} prefix={length}: invariant={passed}", flush=True)
    result = {"passed": all(r["byte_identical"] for r in records), "cells": len(records),
              "nonzero_treatments": 0, "completed_utc": utc_now(), "records": records}
    save_json(output / "pilot_audit.json", result)
    return result


def preflight(runtime, field, protocol, output: Path) -> tuple[dict, list]:
    rows, caches = [], []
    spec = protocol["spec"]
    for prompt in [p for p in protocol["prompts"] if p["split"] == "evaluation"]:
        for length in spec["evaluation"]["prefix_lengths"]:
            observation = observe_prefix(runtime, field, prompt["input_ids"], length, spec)
            receipt = observation["receipt"]
            rows.append({"prompt_id": prompt["prompt_id"], "family": prompt["family"], "prefix_length": length,
                         "coexact_chart_norm": float(np.linalg.norm(field.coexact[receipt["node_index"]])),
                         "support_limit": field.support_limit, **receipt})
            caches.append((prompt, length, observation))
            print(f"support {prompt['prompt_id']} prefix={length}: {receipt['status']}", flush=True)
    result = {"passed": len(rows) == 60 and all(r["status"] == "SUPPORTED" for r in rows),
              "planned_cells": 60, "observed_cells": len(rows), "completed_utc": utc_now(),
              "status_counts": dict(Counter(r["status"] for r in rows)), "cells": rows}
    save_arrays(output / "evaluation_prefixes.npz",
                hidden=np.array([c["hidden"][-1] for _, _, c in caches]),
                baseline_logits=np.array([c["logits"][0] for _, _, c in caches]))
    save_json(output / "support_preflight.json", result)
    return result, caches


def write_treatments(runtime, field, protocol, caches, output: Path, ledger: dict) -> None:
    spec = protocol["spec"]
    with (output / "raw_treatments.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = None
        for prompt, length, observation in caches:
            base = _log_softmax(observation["logits"][0])
            # Target access is confined to scoring, after the prefix-only direction boundary.
            target = prompt["input_ids"][length]
            base_lp = float(base[target])
            for seed in spec["evaluation"]["seeds"]:
                jobs, deltas = delta_grid(field, observation["hidden"][-1], spec, seed)
                if any(job["status"] != "SUPPORTED" for job in jobs):
                    raise ValueError("support changed after preflight")
                if transfer.array_digest(deltas) != observation["receipt"]["nominal_float32_delta_sha256"][seed]:
                    raise ValueError("actual intervention deltas differ from preflight")
                logits = runtime.steer(observation["observed_ids"], deltas)
                if logits.shape != observation["logits"].shape:
                    raise ValueError("treatment batch shape changed")
                for job, delta, vector in zip(jobs, deltas, logits, strict=True):
                    lp = float(_log_softmax(vector)[target])
                    prob = math.exp(lp)
                    if not math.isfinite(lp) or not 0 < prob <= 1 or not 0 < math.exp(base_lp) <= 1:
                        raise ValueError("unrepresentable target probability")
                    row = {"prompt_id": prompt["prompt_id"], "family": prompt["family"], "layer": 7,
                           "prefix_length": length, "target_id": target,
                           **{key: job[key] for key in ("seed", "component", "alpha", "node_index", "distance", "dose_scale")},
                           "nominal_delta_sha256": transfer.array_digest(delta),
                           "nominal_delta_norm": float(np.linalg.norm(delta.astype(float))),
                           "next_token_logprob_base": base_lp, "next_token_prob_base": math.exp(base_lp),
                           "next_token_logprob_steered": lp, "next_token_prob_steered": prob,
                           "next_token_logprob_delta": lp - base_lp}
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    ledger["nonzero_rows"] += 1
                handle.flush()
            print(f"treated {prompt['prompt_id']} prefix={length}: {ledger['nonzero_rows']}/5760 rows", flush=True)
    if ledger["nonzero_rows"] != spec["evaluation"]["planned_treatment_rows"]:
        raise ValueError("incomplete treatment grid")


def execute_stages(runtime, protocol, output: Path, ledger: dict) -> str:
    spec = protocol["spec"]
    calibration = {}
    for prompt in [p for p in protocol["prompts"] if p["split"] == "calibration"]:
        calibration[prompt["prompt_id"]] = runtime.capture(tuple(prompt["input_ids"]), batch=1)["hidden"]
        print(f"calibration {len(calibration)}/40: {prompt['prompt_id']}", flush=True)
    save_arrays(output / "calibration_hidden.npz", **calibration)
    g = spec["geometry"]
    field = transfer.build_frozen_field(calibration, n_components=g["pca_components"], k=g["k"],
        target_betti_1_fraction=g["target_betti_1_fraction"], node_ridge=g["node_ridge"],
        min_chart_norm=g["min_chart_norm"], support_quantile=g["support_quantile"])
    atlas = persist_field(field, output)
    atlas.update({"frozen_utc": utc_now(), "calibration_count": len(calibration),
                  "preparation_sha256": PREPARATION_SHA256, "spec_sha256": preparation.SPEC_SHA256,
                  "geometry": g, "hidden_receipt": file_receipt(output / "calibration_hidden.npz"),
                  "execution_protocol": ledger["protocol"]})
    save_json(output / "atlas_freeze.json", atlas)
    pilot = run_pilot(runtime, field, protocol, output)
    if not pilot["passed"]:
        return "INVALID_FUTURE_DEPENDENCE"
    support, caches = preflight(runtime, field, protocol, output)
    if field.fingerprint() != atlas["fingerprint"]:
        raise ValueError("field changed during validation")
    if not support["passed"]:
        return "INSUFFICIENT_COVERAGE"
    save_json(output / "pre_treatment_gate.json", {"passed": True, "completed_utc": utc_now(),
        "nonzero_rows_at_gate": ledger["nonzero_rows"], "atlas_fingerprint": atlas["fingerprint"],
        "files": [file_receipt(output / name) for name in ("atlas_freeze.json", "pilot_audit.json", "support_preflight.json")]})
    write_treatments(runtime, field, protocol, caches, output, ledger)
    if field.fingerprint() != atlas["fingerprint"]:
        raise ValueError("field changed during treatment")
    return "TREATMENTS_COMPLETE_PENDING_ANALYSIS"


def run(path: Path, root: Path = ROOT) -> dict:
    protocol = json.loads(path.read_text())
    validate_execution(protocol, root)
    output = root / protocol["run_root"]
    output.mkdir(exist_ok=False)
    ledger = {"protocol": file_receipt(path), "started_utc": utc_now(), "status": "RUNNING", "nonzero_rows": 0}
    save_json(output / "execution_started.json", ledger)
    model = None
    before = None
    error = None
    try:
        import torch

        if (not torch.backends.mps.is_available() or torch.is_autocast_enabled("mps")
                or torch.is_autocast_enabled("cpu") or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"):
            raise ValueError("MPS/native-FP32 runtime gate failed")
        model, tokenizer = _load_model_and_tokenizer(protocol["model_path"], device="mps",
            local_files_only=True, trust_remote_code=False, torch_dtype="float32")
        model.config.use_cache = False
        if {p.device.type for p in model.parameters()} != {"mps"} or model.config._attn_implementation != "sdpa":
            raise ValueError("unexpected model device or attention")
        before = audit_model_weight_bytes(model, Path(protocol["model_path"]) / "model.safetensors", "float32")
        if before["parameter_tensors"] != 148 or before["parameter_count"] != 124439808:
            raise ValueError("unexpected GPT-2 shape")
        for prompt in protocol["prompts"]:
            if tokenizer.encode(prompt["text"], add_special_tokens=False, truncation=False) != prompt["input_ids"]:
                raise ValueError("runtime tokenizer changed")
        save_json(output / "load_audit.json", {"model": before, "runtime": runtime_snapshot(), "model_load_count": 1,
                  "device": "mps", "attention": "sdpa", "autocast": False, "mps_fallback": False})
        ledger["gate_status"] = execute_stages(PrefixRuntime(model, protocol["spec"]), protocol, output, ledger)
    except Exception as exc:
        error = exc
        ledger.update({"gate_status": "INVALID_INPUT_OR_NUMERICS", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            if before is not None:
                after = audit_model_weight_bytes(model, Path(protocol["model_path"]) / "model.safetensors", "float32")
                if after != before:
                    raise ValueError("model weights changed")
                save_json(output / "final_weight_audit.json", {"passed": True, "model": after,
                          "completed_utc": utc_now(), "source_bytes_equal_after_execution": True})
            validate_execution(protocol, root)
        except Exception as exc:
            error = exc
            ledger.update({"gate_status": "INVALID_INPUT_OR_NUMERICS", "final_audit_error": f"{type(exc).__name__}: {exc}"})
    ledger.update({"status": "FAILED" if error else "COMPLETED", "completed_utc": utc_now()})
    save_json(output / "execution_receipt.json", ledger)
    if error:
        raise error
    return ledger


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", type=Path)
    actions.add_argument("--run", type=Path)
    actions.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.freeze:
        protocol = freeze(args.freeze.absolute())
        print(json.dumps({"frozen_utc": protocol["frozen_utc"], "run_root": RUN_ROOT}, indent=2))
    elif args.run:
        print(json.dumps(run(args.run.absolute()), indent=2))
    else:
        validate_execution(json.loads(args.validate.read_text()))
        print("execution protocol validated; no model loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
