#!/usr/bin/env python3
"""One zero-only instrument audit; never retry or promote the failed gate."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_hltd_prefix_local_response as local
from scripts.run_hltd_precision_gate import save_json, utc_now

BASE = local.BASE + "/bridge_audit"
FAILED_PROTOCOL = local.BASE + "/protocol.json"
FAILED_PROTOCOL_SHA = "18b4abc806d58b1b1c5f81468cda66946a619eda8531bb0c25e0ed79586f2e8c"
FAILED_RECEIPT = local.RUN + "/execution_receipt.json"
FAILED_RECEIPT_SHA = "b92bd487f80b8ec6bf261c03c98ff54db739cb8d5196ca1bd598d54246ba8173"
SOURCES = ["scripts/audit_hltd_local_response_failure.py", "tests/test_hltd_local_response_failure.py"]


def bridge_diagnostics(saved_hidden, saved_logits, captured, grad, target, method):
    from scripts.run_hltd_steering import _log_softmax

    comparisons = {}
    for name, left, right in (("no_hook_vs_saved", captured["logits"], saved_logits),
                               ("autograd_vs_saved", grad["logits"], saved_logits),
                               ("autograd_vs_no_hook", grad["logits"], captured["logits"])):
        difference = left.astype(float) - right.astype(float)
        left_lp, right_lp = _log_softmax(left[0]), _log_softmax(right[0])
        comparisons[name] = {"max_abs_logit_error": float(np.abs(difference).max()),
                             "rms_logit_error": float(np.sqrt(np.mean(difference**2))),
                             "mean_logit_shift": float(difference.mean()),
                             "max_abs_logprob_error": float(np.abs(left_lp - right_lp).max()),
                             "target_logprob_difference": float(left_lp[target] - right_lp[target]),
                             "kl_right_to_left": float(np.exp(right_lp) @ (right_lp - left_lp)),
                             "within_original_logit_tolerance": bool(np.abs(difference).max() <= method["baseline_logit_tolerance"])}
    result = {"comparisons": comparisons, "gradient_norm": float(np.linalg.norm(grad["gradient"].astype(float))),
              "gradient_finite": bool(np.isfinite(grad["gradient"]).all()),
              "captured_hidden_bytes_equal": captured["hidden"].tobytes() == saved_hidden.tobytes(),
              "gradient_hidden_bytes_equal": grad["hidden"].tobytes() == np.broadcast_to(saved_hidden, grad["hidden"].shape).tobytes(),
              "no_hook_row_spread": float(np.max(np.abs(captured["logits"] - captured["logits"][0]))),
              "autograd_row_spread": float(np.max(np.abs(grad["logits"] - grad["logits"][0]))),
              "objective_error": abs(grad["objective"] - float(_log_softmax(grad["logits"][0])[target])),
              "original_logit_tolerance": method["baseline_logit_tolerance"],
              "semantic_direction_status": "NOT_TESTED", "nonzero_treatments": 0}
    try:
        local.validate_bridge(saved_hidden, saved_logits, captured, grad, target, method)
        result["original_bridge_passes"] = True
    except ValueError as exc:
        result.update({"original_bridge_passes": False, "original_error": str(exc)})
    return result


def contract(root=ROOT):
    original = local.cal.pinned_json(root / FAILED_PROTOCOL, FAILED_PROTOCOL_SHA)
    local.validate(original, root)
    failed = local.cal.pinned_json(root / FAILED_RECEIPT, FAILED_RECEIPT_SHA)
    local.verify_frozen_files({"frozen_files": failed["raw_files"]}, root)
    if (failed["status"] != "FAILED" or failed["protocol"] != local.file_receipt(root / FAILED_PROTOCOL)
            or any(failed[k] != 0 for k in ("nonzero_rows", "nonzero_forward_rows_attempted", "treatment_forwards_attempted"))
            or any(failed[k] != 10 for k in ("baseline_forwards_attempted", "gradient_forwards_attempted", "gradients_completed"))):
        raise ValueError("unexpected failed attempt")
    records = [local.file_receipt(root / p) for p in [FAILED_PROTOCOL, FAILED_RECEIPT, *SOURCES]]
    return {"audit_id": "hltd-local-response-zero-bridge-audit-v1", "selected_cell_index": 9,
            "selection_reason": "First failed cell from the sealed execution counter and frozen deterministic cell order, not an outcome-selected scientific cohort.",
            "scope": "One no-hook forward and one zero-leaf gradient forward/backward at the original failing prefix. No nonzero deltas, gate retry, threshold relaxation, backend change or semantic effect estimate.",
            "original_thresholds_unchanged": True, "model_path": original["model_path"], "method": original["method"],
            "frozen_files": records, "output": BASE + "/result", "runtime": original["runtime"]}


def validate(protocol, root=ROOT):
    local.verify_frozen_files(protocol, root)
    expected = contract(root)
    if set(protocol) != set(expected) | {"frozen_utc"} or any(protocol[k] != v for k, v in expected.items()):
        raise ValueError("bridge audit contract changed")


def freeze(path, root=ROOT):
    if path.exists() or (root / BASE / "result").exists():
        raise FileExistsError("bridge audit already exists")
    result = {**contract(root), "frozen_utc": utc_now()}
    validate(result, root)
    save_json(path, result)
    return result


def run(path, root=ROOT):
    protocol = json.loads(path.read_text())
    validate(protocol, root)
    output = root / protocol["output"]
    output.mkdir(parents=True, exist_ok=False)
    ledger = {"protocol": local.file_receipt(path), "started_utc": utc_now(), "status": "RUNNING",
              "baseline_forwards_attempted": 0, "model_loads_attempted": 0, "nonzero_treatments": 0}
    save_json(output / "started.json", ledger)
    model, before, error = None, None, None
    try:
        import torch
        if (not torch.backends.mps.is_available() or torch.is_autocast_enabled("mps") or torch.is_autocast_enabled("cpu")
                or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"):
            raise ValueError("native MPS gate failed")
        ledger["model_loads_attempted"] += 1
        model, _ = local.v1._load_model_and_tokenizer(protocol["model_path"], device="mps", local_files_only=True,
                                                    trust_remote_code=False, torch_dtype="float32")
        model.config.use_cache = False
        model.requires_grad_(False)
        if model.config._attn_implementation != "sdpa" or {p.device.type for p in model.parameters()} != {"mps"}:
            raise ValueError("model device/attention changed")
        before = local.v1.audit_model_weight_bytes(model, Path(protocol["model_path"]) / "model.safetensors", "float32")
        save_json(output / "load_audit.json", {"model": before, "runtime": local.v1.runtime_snapshot(), "model_load_count": 1})
        prior = local.cal.pinned_json(root / local.cal.REFERENCE, local.cal.REFERENCE_SHA)
        source = root / prior["run_root"]
        i = protocol["selected_cell_index"]
        cell = json.loads((source / "support_preflight.json").read_text())["cells"][i]
        prompt = next(p for p in prior["prompts"] if p["prompt_id"] == cell["prompt_id"])
        n = cell["prefix_length"]
        ids, target = tuple(prompt["input_ids"][:n]), prompt["input_ids"][n]
        runtime = local.v1.PrefixRuntime(model, prior["spec"])
        ledger["baseline_forwards_attempted"] += 1
        captured = runtime.capture(ids)
        grad = local.activation_gradient(model, runtime.inputs(ids), target, ledger=ledger)
        with np.load(source / "evaluation_observations.npz", allow_pickle=False) as arrays:
            result = bridge_diagnostics(arrays[f"{i}_hidden"], arrays[f"{i}_logits"], captured, grad, target, protocol["method"])
        result.update({"prompt_id": cell["prompt_id"], "prefix_length": n, "observed_ids": list(ids), "target_id": target,
                       "status": "FAILURE_REPRODUCED" if not result["original_bridge_passes"] else "FAILURE_NOT_REPRODUCED",
                       "scope": protocol["scope"], "completed_utc": utc_now()})
        local.v1.save_arrays(output / "observations.npz", hidden=captured["hidden"], no_hook_logits=captured["logits"],
                             gradient_hidden=grad["hidden"], gradient_logits=grad["logits"], gradient=grad["gradient"])
        save_json(output / "diagnostic.json", result)
    except Exception as exc:
        error = exc
        ledger["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if before is not None:
                after = local.v1.audit_model_weight_bytes(model, Path(protocol["model_path"]) / "model.safetensors", "float32")
                if before != after or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                    raise ValueError("model weights or gradients changed")
                save_json(output / "final_weight_audit.json", {"passed": True, "model": after, "completed_utc": utc_now()})
            validate(protocol, root)
        except Exception as exc:
            error = exc
            ledger["final_audit_error"] = f"{type(exc).__name__}: {exc}"
    ledger.update({"status": "FAILED" if error else "COMPLETED", "completed_utc": utc_now(),
                   "files": [local.file_receipt(p) for p in sorted(output.iterdir()) if p.is_file()]})
    save_json(output / "receipt.json", ledger)
    if error:
        raise error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", type=Path)
    group.add_argument("--run", type=Path)
    args = parser.parse_args()
    result = freeze(args.freeze.absolute()) if args.freeze else run(args.run.absolute())
    print(json.dumps({k: v for k, v in result.items() if k != "frozen_files"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
