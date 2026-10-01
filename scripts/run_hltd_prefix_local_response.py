#!/usr/bin/env python3
"""Diagnose local token-fit alignment without promoting it to semantic evidence."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_hltd_prefix_v2 as replay
from scripts import diagnose_hltd_prefix_calibration as cal
from scripts import run_hltd_prefix_transfer_gate as v1
from scripts import run_hltd_prefix_v2 as v2
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, verify_frozen_files
from scripts.run_hltd_precision_gate import save_json, utc_now
from scripts.run_hltd_steering import _block_for_hidden_layer, _log_softmax

BASE = "docs/data/hltd_prefix_local_response"
METHOD = BASE + "/method.json"
RUN = "spiral_out_hltd_prefix_local_response"
RESULT = BASE + "/result"
V2_RESULT = "docs/data/hltd_prefix_transfer_v2/result"
PINS = {
    cal.REFERENCE: cal.REFERENCE_SHA,
    V2_RESULT + "/manifest.json": "98f68c894a4f7a7cea37c1bcc4cbc1bc14b98cbb290da7d76b60acfad48a8819",
    "docs/data/hltd_prefix_null_comparison/v2/manifest.json": "1dda9e0002aa85cd430bc8a7ab2ee0aaab5c90a78d21746c1ac0d887401c32e9",
}
SOURCES = ["scripts/run_hltd_prefix_local_response.py", "tests/test_hltd_prefix_local_response.py",
           "scripts/diagnose_hltd_prefix_calibration.py"]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def contract(root=ROOT):
    pinned = {name: cal.pinned_json(root / name, sha) for name, sha in PINS.items()}
    prior = pinned[cal.REFERENCE]
    manifest = pinned[V2_RESULT + "/manifest.json"]
    records = [*prior["frozen_files"], manifest["protocol"], *manifest["raw_files"], *manifest["derived_files"]]
    verify_frozen_files({"frozen_files": records}, root)
    if v1.runtime_snapshot() != prior["runtime"]:
        raise ValueError("v2 runtime changed")
    original = {Path(r["path"]).name for r in records if Path(r["path"]).parent == Path(prior["model_path"])}
    if {p.name for p in Path(prior["model_path"]).iterdir() if p.is_file()} != original:
        raise ValueError("model/tokenizer inventory changed")
    verdict = json.loads((root / V2_RESULT / "verdict.json").read_text())
    if verdict["status"] != "NOT_SUPPORTED" or verdict["nonzero_rows"] != 5760:
        raise ValueError("unexpected prior disposition")
    method = json.loads((root / METHOD).read_text())
    if method["semantic_direction_status"] != "NOT_TESTED":
        raise ValueError("token-fit diagnostics cannot validate semantics")
    added = [file_receipt(root / p) for p in [*PINS, METHOD, *SOURCES]]
    frozen = v2.merge_receipts(records + added, root)
    return {"diagnostic_id": method["diagnostic_id"], "method": method, "runtime": prior["runtime"],
            "model_path": prior["model_path"], "readout_fingerprint": prior["readout_fingerprint"],
            "run_root": RUN, "result_root": RESULT, "frozen_files": frozen}


def validate(protocol, root=ROOT):
    verify_frozen_files(protocol, root)
    expected = contract(root)
    if set(protocol) != set(expected) | {"frozen_utc"} or any(protocol[k] != v for k, v in expected.items()):
        raise ValueError("local-response contract changed")


def freeze(path, root=ROOT):
    if path.exists() or (root / RUN).exists() or (root / RESULT).exists():
        raise FileExistsError("local-response protocol/run already exists")
    protocol = {**contract(root), "frozen_utc": utc_now()}
    validate(protocol, root)
    save_json(path, protocol)
    return protocol


def small_protocol(prior, method):
    result = copy.deepcopy(prior)
    result["spec"]["evaluation"]["alphas"] = [-a for a in reversed(method["magnitudes"])] + method["magnitudes"]
    return result


def activation_gradient(model, inputs, target, *, layer=7, batch=12, ledger=None):
    import torch

    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("gradient diagnostic requires frozen, gradient-free eval parameters")
    counters = ledger if ledger is not None else {}
    captured = {}
    repeated = {key: value.repeat(batch, 1) if torch.is_tensor(value) else value for key, value in inputs.items()}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if "leaf" in captured:
            raise ValueError("hook executed more than once")
        leaf = torch.zeros(hidden.shape[-1], dtype=hidden.dtype, device=hidden.device, requires_grad=True)
        captured.update({"leaf": leaf, "hidden": hidden.detach()})
        changed = hidden.clone()
        changed[:, -1, :] = hidden[:, -1, :] + leaf
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = _block_for_hidden_layer(model, layer).register_forward_hook(hook)
    try:
        with torch.enable_grad():
            counters["gradient_forwards_attempted"] = counters.get("gradient_forwards_attempted", 0) + 1
            output = model(**repeated, return_dict=True)
            logits = output.logits[:, -1]
            objective = torch.log_softmax(logits, dim=-1)[:, target].mean()
            counters["backwards_attempted"] = counters.get("backwards_attempted", 0) + 1
            gradient, = torch.autograd.grad(objective, captured["leaf"])
        counters["gradients_completed"] = counters.get("gradients_completed", 0) + 1
    finally:
        handle.remove()
    if any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise ValueError("parameter gradient leakage")
    result = {"gradient": gradient.detach().cpu().numpy(), "hidden": captured["hidden"].cpu().numpy(),
              "logits": logits.detach().cpu().numpy(), "objective": float(objective.detach().cpu())}
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (result["gradient"], result["hidden"], result["logits"])):
        raise ValueError("invalid activation gradient or dtype")
    return result


def validate_bridge(saved_hidden, saved_logits, captured, grad, target, method):
    hidden = grad["hidden"]
    if (captured["hidden"].dtype != saved_hidden.dtype or captured["hidden"].shape != saved_hidden.shape
            or captured["hidden"].tobytes() != saved_hidden.tobytes()
            or hidden.shape != (12, *saved_hidden.shape)
            or hidden.dtype != saved_hidden.dtype
            or hidden.tobytes() != np.broadcast_to(saved_hidden, hidden.shape).tobytes()):
        raise ValueError("saved prefix hidden bytes changed")
    for logits in (captured["logits"], grad["logits"]):
        if logits.shape != saved_logits.shape or not np.isfinite(logits).all() or np.max(np.abs(logits - logits[0])) != 0:
            raise ValueError("baseline batch shape/spread changed")
    errors = {"no_hook_vs_saved": float(np.max(np.abs(captured["logits"] - saved_logits))),
              "autograd_vs_saved": float(np.max(np.abs(grad["logits"] - saved_logits))),
              "autograd_vs_no_hook": float(np.max(np.abs(grad["logits"] - captured["logits"])))}
    objective_error = abs(grad["objective"] - float(_log_softmax(grad["logits"][0])[target]))
    norm = float(np.linalg.norm(grad["gradient"].astype(float)))
    if (max(errors.values()) > method["baseline_logit_tolerance"]
            or not np.isfinite(objective_error)
            or objective_error > method["objective_logprob_tolerance"]
            or grad["gradient"].shape != (saved_hidden.shape[1],) or not np.isfinite(norm) or norm <= 0):
        raise ValueError("baseline or gradient validation failed")
    return {**errors, "objective_error": objective_error, "gradient_norm": norm}


def execute(model, protocol, output, ledger, root=ROOT):
    prior = cal.pinned_json(root / cal.REFERENCE, cal.REFERENCE_SHA)
    method = protocol["method"]
    small = small_protocol(prior, method)
    source = root / prior["run_root"]
    support = json.loads((source / "support_preflight.json").read_text())
    replay.analysis.require_support(prior, support)
    field = v2.load_readout(prior, root)
    prompts = {p["prompt_id"]: p for p in prior["prompts"]}
    runtime = v1.PrefixRuntime(model, prior["spec"])
    records, caches, arrays = [], [], {}
    with np.load(source / "evaluation_observations.npz", allow_pickle=False) as saved:
        for i, cell in enumerate(support["cells"]):
            prompt, n = prompts[cell["prompt_id"]], cell["prefix_length"]
            ids = tuple(prompt["input_ids"][:n])
            old_hidden, old_logits = saved[f"{i}_hidden"], saved[f"{i}_logits"]
            # Freeze the direction before the scorer obtains its next-token label.
            direction = v2.direction_receipt(field, old_hidden[-1], prior["spec"])
            if any(cell[k] != value for k, value in direction.items()):
                raise ValueError("frozen direction changed")
            ledger["baseline_forwards_attempted"] += 1
            captured = runtime.capture(ids)
            target = prompt["input_ids"][n]
            grad = activation_gradient(model, runtime.inputs(ids), target, ledger=ledger)
            errors = validate_bridge(old_hidden, old_logits, captured, grad, target, method)
            base_lp = float(_log_softmax(captured["logits"][0])[target])
            record = {"prompt_id": prompt["prompt_id"], "family": prompt["family"], "prefix_length": n,
                      "target_id": target, "observed_ids": list(ids), "baseline_logprob": base_lp, **errors,
                      "gradient_sha256": v1.transfer.array_digest(grad["gradient"]),
                      "hidden_sha256": v1.transfer.array_digest(captured["hidden"]),
                      "logits_sha256": v1.transfer.array_digest(captured["logits"][0])}
            records.append(record)
            arrays[f"{i}_gradient"] = grad["gradient"]
            arrays[f"{i}_logits"] = captured["logits"][0]
            caches.append((cell, ids, old_hidden[-1].copy(), base_lp, target))
    if len(records) != method["planned_cells"] or field.fingerprint() != protocol["readout_fingerprint"]:
        raise ValueError("preflight inventory changed")
    v1.save_arrays(output / "gradients.npz", **arrays)
    save_json(output / "gradient_preflight.json", {"passed": True, "cells": records, "completed_utc": utc_now(),
              "nonzero_forward_rows_attempted": ledger["nonzero_forward_rows_attempted"]})
    save_json(output / "pre_treatment_gate.json", {"passed": True, "completed_utc": utc_now(),
              "files": [file_receipt(output / n) for n in ("gradient_preflight.json", "gradients.npz")],
              "nonzero_forward_rows_attempted": ledger["nonzero_forward_rows_attempted"]})
    with (output / "raw_treatments.csv").open("x", newline="") as handle:
        writer = None
        for cell, ids, hidden, base_lp, target in caches:
            for seed in prior["spec"]["evaluation"]["seeds"]:
                jobs, deltas = v2.delta_grid(field, hidden, small["spec"], seed)
                if any(j["status"] != "SUPPORTED" for j in jobs):
                    raise ValueError("direction support changed")
                ledger["nonzero_forward_rows_attempted"] += len(deltas)
                ledger["treatment_forwards_attempted"] += 1
                logits = runtime.steer(ids, deltas)
                if logits.shape != (12, 50257):
                    raise ValueError("unexpected treatment batch")
                for job, delta, vector in zip(jobs, deltas, logits, strict=True):
                    lp = float(_log_softmax(vector)[target])
                    row = {"prompt_id": cell["prompt_id"], "family": cell["family"], "prefix_length": cell["prefix_length"],
                           "layer": 7, "target_id": target, **{k: job[k] for k in ("seed", "component", "alpha", "node_index", "distance", "dose_scale")},
                           "nominal_delta_sha256": v1.transfer.array_digest(delta), "nominal_delta_norm": float(np.linalg.norm(delta.astype(float))),
                           "next_token_logprob_base": base_lp, "next_token_logprob_steered": lp,
                           "next_token_prob_base": float(np.exp(base_lp)), "next_token_prob_steered": float(np.exp(lp)),
                           "next_token_logprob_delta": lp - base_lp}
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    ledger["nonzero_rows"] += 1
                handle.flush()
            print(f"local response {cell['prompt_id']} prefix={len(ids)}: {ledger['nonzero_rows']}/5760", flush=True)
    if ledger["nonzero_rows"] != method["planned_nonzero_rows"]:
        raise ValueError("incomplete small-dose grid")


def run(path, root=ROOT):
    protocol = json.loads(path.read_text())
    validate(protocol, root)
    output = root / RUN
    output.mkdir(exist_ok=False)
    ledger = {"protocol": file_receipt(path), "status": "RUNNING", "started_utc": utc_now(),
              "baseline_forwards_attempted": 0, "gradient_forwards_attempted": 0, "backwards_attempted": 0,
              "gradients_completed": 0, "treatment_forwards_attempted": 0,
              "nonzero_forward_rows_attempted": 0, "nonzero_rows": 0, "model_loads_attempted": 0,
              "semantic_direction_status": "NOT_TESTED"}
    save_json(output / "execution_started.json", ledger)
    model, before, error = None, None, None
    try:
        import torch
        if (not torch.backends.mps.is_available() or torch.is_autocast_enabled("mps") or torch.is_autocast_enabled("cpu")
                or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"):
            raise ValueError("native MPS runtime gate failed")
        ledger["model_loads_attempted"] += 1
        model, tokenizer = v1._load_model_and_tokenizer(protocol["model_path"], device="mps", local_files_only=True,
                                                        trust_remote_code=False, torch_dtype="float32")
        model.config.use_cache = False
        model.requires_grad_(False)
        if {p.device.type for p in model.parameters()} != {"mps"} or model.config._attn_implementation != "sdpa":
            raise ValueError("model device or attention changed")
        before = v1.audit_model_weight_bytes(model, Path(protocol["model_path"]) / "model.safetensors", "float32")
        if before["parameter_tensors"] != 148 or before["parameter_count"] != 124439808:
            raise ValueError("model shape changed")
        prior = cal.pinned_json(root / cal.REFERENCE, cal.REFERENCE_SHA)
        for prompt in prior["prompts"]:
            if tokenizer.encode(prompt["text"], add_special_tokens=False, truncation=False) != prompt["input_ids"]:
                raise ValueError("tokenization changed")
        save_json(output / "load_audit.json", {"model": before, "runtime": v1.runtime_snapshot(), "model_load_count": 1,
                  "all_parameters_frozen": True, "optimizer_created": False, "completed_utc": utc_now()})
        execute(model, protocol, output, ledger, root)
    except Exception as exc:
        error = exc
        ledger["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if before is not None:
                after = v1.audit_model_weight_bytes(model, Path(protocol["model_path"]) / "model.safetensors", "float32")
                if after != before or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                    raise ValueError("model weights or parameter gradients changed")
                save_json(output / "final_weight_audit.json", {"passed": True, "model": after,
                          "parameters_frozen_and_grad_none": True, "completed_utc": utc_now()})
            validate(protocol, root)
        except Exception as exc:
            error = exc
            ledger["final_audit_error"] = f"{type(exc).__name__}: {exc}"
    ledger.update({"status": "FAILED" if error else "COMPLETED", "completed_utc": utc_now()})
    ledger["raw_files"] = [file_receipt(p) for p in sorted(output.iterdir()) if p.is_file()]
    save_json(output / "execution_receipt.json", ledger)
    if error:
        raise error
    return ledger


def directional_statistics(gradient, delta, plus, minus, baseline, magnitude, method):
    gradient, delta = np.asarray(gradient, dtype=float), np.asarray(delta, dtype=float)
    gn, dn = np.linalg.norm(gradient), np.linalg.norm(delta)
    if gn <= 0 or dn <= 0 or magnitude <= 0 or not np.isfinite([gn, dn, plus, minus, baseline, magnitude]).all():
        raise ValueError("undefined directional response")
    slope = float(gradient @ delta)
    fd = (plus - minus) / (2 * magnitude)
    error = abs(fd - slope)
    tolerance = method["derivative_absolute_tolerance"] + method["derivative_relative_tolerance"] * abs(slope)
    return {"gradient_norm": float(gn), "direction_norm": float(dn), "cosine": float(np.clip(slope / (gn * dn), -1, 1)),
            "local_slope": slope, "finite_slope": fd, "slope_error": error,
            "odd_delta": (plus - minus) / 2, "even_delta": (plus + minus) / 2 - baseline,
            "numerical_tolerance": tolerance, "within_tolerance": error <= tolerance}


def disposition(primary_rows, expected_rows):
    if len(primary_rows) != expected_rows:
        raise ValueError("incomplete numerical validation grid")
    return {"numerical_status": "LOCAL_DERIVATIVE_VALIDATED" if all(r["within_tolerance"] for r in primary_rows)
            else "LOCAL_DERIVATIVE_NOT_VALIDATED", "semantic_direction_status": "NOT_TESTED"}


def analyze(path, root=ROOT):
    protocol = json.loads(path.read_text())
    validate(protocol, root)
    output = root / RESULT
    if output.exists():
        raise FileExistsError(output)
    run_root = root / RUN
    receipt = json.loads((run_root / "execution_receipt.json").read_text())
    expected_files = {"execution_started.json", "load_audit.json", "final_weight_audit.json",
                      "gradients.npz", "gradient_preflight.json", "pre_treatment_gate.json", "raw_treatments.csv"}
    if ({Path(r["path"]).resolve() for r in receipt.get("raw_files", [])} != {run_root.resolve() / p for p in expected_files}
            or len(receipt.get("raw_files", [])) != len(expected_files)):
        raise ValueError("raw run inventory incomplete or changed")
    verify_frozen_files({"frozen_files": receipt["raw_files"]}, root)
    if (receipt["protocol"] != file_receipt(path) or receipt["status"] != "COMPLETED"
            or receipt["nonzero_rows"] != 5760 or receipt["nonzero_forward_rows_attempted"] != 5760
            or any(receipt[k] != 60 for k in ("gradients_completed", "baseline_forwards_attempted", "gradient_forwards_attempted", "backwards_attempted"))
            or receipt["model_loads_attempted"] != 1 or receipt["treatment_forwards_attempted"] != 480
            or receipt["semantic_direction_status"] != "NOT_TESTED"):
        raise ValueError("run incomplete or changed")
    gate = json.loads((run_root / "pre_treatment_gate.json").read_text())
    if (not gate["passed"] or gate["nonzero_forward_rows_attempted"] != 0
            or gate["files"] != [file_receipt(run_root / n) for n in ("gradient_preflight.json", "gradients.npz")]):
        raise ValueError("pre-treatment gate failed")
    verify_frozen_files({"frozen_files": gate["files"]}, root)
    final = json.loads((run_root / "final_weight_audit.json").read_text())
    load = json.loads((run_root / "load_audit.json").read_text())
    if (not final["passed"] or final["model"] != load["model"] or not final["parameters_frozen_and_grad_none"]
            or load["runtime"] != protocol["runtime"] or load["model_load_count"] != 1
            or not load["all_parameters_frozen"] or load["optimizer_created"]
            or not load["model"]["all_bytes_equal"] or load["model"]["parameter_count"] != 124439808):
        raise ValueError("final model audit failed")
    prior = cal.pinned_json(root / cal.REFERENCE, cal.REFERENCE_SHA)
    source = root / prior["run_root"]
    support = json.loads((source / "support_preflight.json").read_text())
    preflight = json.loads((run_root / "gradient_preflight.json").read_text())
    times = [protocol["frozen_utc"], receipt["started_utc"], load["completed_utc"], preflight["completed_utc"],
             gate["completed_utc"], final["completed_utc"], receipt["completed_utc"]]
    parsed = [datetime.fromisoformat(t) for t in times]
    if (any(t.tzinfo is None for t in parsed) or parsed != sorted(parsed)
            or not preflight["passed"] or preflight["nonzero_forward_rows_attempted"] != 0):
        raise ValueError("local-response chronology/preflight invalid")
    small = small_protocol(prior, protocol["method"])
    raw = read_csv(run_root / "raw_treatments.csv")
    old = read_csv(source / "raw_treatments.csv")
    replay.analysis.validate_rows(raw, small, support)
    replay.analysis.validate_rows(old, prior, support)
    field = v2.load_readout(prior, root)
    rows = []
    with np.load(run_root / "gradients.npz", allow_pickle=False) as grads, np.load(source / "evaluation_observations.npz", allow_pickle=False) as saved:
        if set(grads.files) != {f"{i}_{name}" for i in range(60) for name in ("gradient", "logits")} or len(preflight["cells"]) != 60:
            raise ValueError("gradient inventory changed")
        for label, values, magnitudes in (("new_small_dose", raw, protocol["method"]["magnitudes"]),
                                           ("saved_v2", old, prior["spec"]["analysis"]["positive_magnitudes"])):
            indexed = {(r["prompt_id"], int(r["prefix_length"]), int(r["seed"]), r["component"], float(r["alpha"])): r for r in values}
            for i, cell in enumerate(support["cells"]):
                gradient, record = grads[f"{i}_gradient"], preflight["cells"][i]
                if (record["prompt_id"], record["prefix_length"]) != (cell["prompt_id"], cell["prefix_length"]):
                    raise ValueError("gradient cell mapping changed")
                if v1.transfer.array_digest(gradient) != record["gradient_sha256"] or v1.transfer.array_digest(grads[f"{i}_logits"]) != record["logits_sha256"]:
                    raise ValueError("gradient observation bytes changed")
                fresh_logits = grads[f"{i}_logits"]
                prompt = next(p for p in prior["prompts"] if p["prompt_id"] == cell["prompt_id"])
                n = cell["prefix_length"]
                if (gradient.shape != (768,) or gradient.dtype != np.float32 or not np.isfinite(gradient).all()
                        or fresh_logits.shape != (50257,) or fresh_logits.dtype != np.float32 or not np.isfinite(fresh_logits).all()
                        or record["observed_ids"] != prompt["input_ids"][:n] or record["target_id"] != prompt["input_ids"][n]
                        or v1.transfer.array_digest(saved[f"{i}_hidden"]) != record["hidden_sha256"]
                        or np.max(np.abs(fresh_logits - saved[f"{i}_logits"][0])) > protocol["method"]["baseline_logit_tolerance"]):
                    raise ValueError("gradient replay metadata invalid")
                fresh_lp = float(_log_softmax(fresh_logits)[record["target_id"]])
                if fresh_lp != record["baseline_logprob"]:
                    raise ValueError("fresh baseline changed")
                for seed in prior["spec"]["evaluation"]["seeds"]:
                    if label == "new_small_dose":
                        jobs, deltas = v2.delta_grid(field, saved[f"{i}_hidden"][-1], small["spec"], seed)
                        for job, delta in zip(jobs, deltas, strict=True):
                            recorded = indexed[cell["prompt_id"], n, seed, job["component"], job["alpha"]]
                            if (recorded["nominal_delta_sha256"] != v1.transfer.array_digest(delta)
                                    or float(recorded["next_token_logprob_base"]) != fresh_lp):
                                raise ValueError("small-dose delta or matched baseline changed")
                    direction = v2.interpolation.query_hidden(field, saved[f"{i}_hidden"][-1], seed=seed, alpha=1)
                    for component, delta in (("coexact", direction.coexact_delta), ("random_tangent", direction.random_delta)):
                        delta = np.asarray(delta, dtype=np.float32)
                        for a in magnitudes:
                            pos, neg = [indexed[cell["prompt_id"], cell["prefix_length"], seed, component, sign * a] for sign in (1, -1)]
                            base = float(pos["next_token_logprob_base"])
                            if base != float(neg["next_token_logprob_base"]):
                                raise ValueError("signed baselines differ")
                            stats = directional_statistics(gradient, delta, float(pos["next_token_logprob_steered"]),
                                float(neg["next_token_logprob_steered"]), base, a, protocol["method"])
                            rows.append({"source": label, "prompt_id": cell["prompt_id"], "family": cell["family"],
                                         "prefix_length": cell["prefix_length"], "seed": seed, "component": component, "magnitude": a, **stats})
    primary = [r for r in rows if r["source"] == "new_small_dose" and r["magnitude"] == protocol["method"]["primary_numerical_magnitude"]]
    status = disposition(primary, 960)
    cell_rows = []
    for cell in support["cells"]:
        selected = [r for r in primary if (r["prompt_id"], r["prefix_length"]) == (cell["prompt_id"], cell["prefix_length"])]
        row = {"prompt_id": cell["prompt_id"], "family": cell["family"], "prefix_length": cell["prefix_length"]}
        for component in ("coexact", "random_tangent"):
            group = [r for r in selected if r["component"] == component]
            for key in ("local_slope", "finite_slope", "cosine"):
                row[component + "_" + key] = float(np.mean([r[key] for r in group]))
        row["local_gap"] = row["coexact_local_slope"] - row["random_tangent_local_slope"]
        cell_rows.append(row)
    prompt_rows = []
    numeric = [k for k in cell_rows[0] if k not in {"prompt_id", "family", "prefix_length"}]
    for prompt in sorted({r["prompt_id"] for r in cell_rows}):
        group = [r for r in cell_rows if r["prompt_id"] == prompt]
        prompt_rows.append({"prompt_id": prompt, **{k: float(np.mean([r[k] for r in group])) for k in numeric}})
    curves = []
    for label in ("new_small_dose", "saved_v2"):
        for a in sorted({r["magnitude"] for r in rows if r["source"] == label}):
            for component in ("coexact", "random_tangent"):
                group = [r for r in rows if r["source"] == label and r["magnitude"] == a and r["component"] == component]
                curves.append({"source": label, "magnitude": a, "component": component,
                    **{k: cal.balanced_mean([r[k] for r in group], [r["prompt_id"] for r in group])["mean"]
                       for k in ("local_slope", "finite_slope", "odd_delta", "even_delta", "slope_error")}})
    summary = {**status, "diagnostic_status": "COMPLETED", "protocol": file_receipt(path), "completed_utc": utc_now(),
               "cells": 60, "prompts": 20, "new_nonzero_rows": len(raw), "gradient_calls": receipt["gradients_completed"],
               "primary_numerical_rows": len(primary), "numerical_failures": sum(not r["within_tolerance"] for r in primary),
               "max_primary_slope_error": max(r["slope_error"] for r in primary),
               "prompt_statistics": {key: {**cal.describe([r[key] for r in prompt_rows]),
                   "positive_prompts": sum(r[key] > 0 for r in prompt_rows)} for key in numeric},
               "scope": protocol["method"]["semantic_boundary"], "prior_v2_status_unchanged": "NOT_SUPPORTED"}
    validate(protocol, root)
    verify_frozen_files({"frozen_files": receipt["raw_files"]}, root)
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("directional_responses.csv", rows), ("cells.csv", cell_rows), ("prompts.csv", prompt_rows), ("curves.csv", curves)):
        cal.write_csv(output / name, values)
    save_json(output / "summary.json", summary)
    save_json(output / "manifest.json", {"protocol": file_receipt(path),
              "raw_files": [file_receipt(p) for p in sorted(run_root.iterdir()) if p.is_file()],
              "derived_files": [file_receipt(p) for p in sorted(output.iterdir()) if p.is_file()]})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("freeze", "run", "analyze", "validate"):
        group.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    for name in ("freeze", "run", "analyze"):
        path = getattr(args, name)
        if path:
            result = globals()[name](path.absolute())
            print(json.dumps({k: v for k, v in result.items() if k != "frozen_files"}, indent=2, allow_nan=False))
            return
    validate(json.loads(args.validate.read_text()))
    print("local-response inputs valid; no model loaded")


if __name__ == "__main__":
    main()
