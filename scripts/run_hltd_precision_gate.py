#!/usr/bin/env python3
"""Run a frozen GPT-2 precision pilot without changing the historical runners."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spiral_hodge as hodge
from scripts import run_hltd_steering_fast_suite as fast
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, validate_protocol_rows, verify_frozen_files
from scripts.run_hltd_steering import _log_softmax
from scripts.run_hltd_steering_suite import read_suite

ARMS = {
    "fp16_replay": ("float16", "float16"),
    "fp32_fixed_field": ("float32", "float16"),
    "fp32_rebuilt_field": ("float32", "float32"),
}
PILOT_REFERENCE = "docs/data/hltd_precision_l7/protocol.json"
PILOT_REFERENCE_SHA256 = "df86bf783131119e142293db7e471fc6dde7f91c6a2c6593b51f9f68cd6441e4"


def validate_pilot_protocol(protocol: dict, root: Path = ROOT) -> None:
    payload = (root / PILOT_REFERENCE).read_bytes()
    if hashlib.sha256(payload).hexdigest() != PILOT_REFERENCE_SHA256:
        raise ValueError("canonical pilot protocol changed")
    reference = json.loads(payload)
    metadata = {"run_root", "frozen_files", "frozen_utc", "command_argv", "analysis_argv"}
    if set(protocol) - metadata != set(reference) - metadata:
        raise ValueError("changed pilot contract keys")
    for key in set(reference) - metadata:
        if protocol[key] != reference[key]:
            raise ValueError(f"changed pilot contract: {key}")
    frozen = protocol["frozen_files"]
    expected = {record["path"]: record for record in reference["frozen_files"]}
    if len(frozen) != len(expected) or {record["path"] for record in frozen} != set(expected):
        raise ValueError("changed pilot frozen input inventory")
    for record in frozen:
        source = Path(record["path"])
        local_code = (not source.is_absolute() and ".." not in source.parts and source.suffix == ".py"
                      and source.parts[0] in {"scripts", "tests", "spiral_hodge.py"})
        if not local_code and any(record[key] != expected[record["path"]][key] for key in ("sha256", "bytes")):
            raise ValueError(f"changed pilot input: {source}")
    out = Path(protocol["run_root"])
    if out.is_absolute() or len(out.parts) != 1 or not out.name.startswith("spiral_out_"):
        raise ValueError("run root must be a new top-level spiral_out_ directory")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def audit_model_weights(model, checkpoint: Path, dtype: str) -> dict:
    """Check stored F32 values, not an upcast of already rounded loaded weights."""
    from safetensors import safe_open

    if dtype not in {"float16", "float32"}:
        raise ValueError("only explicit float16/float32 are supported")
    tensor_count = 0
    parameter_count = 0
    with safe_open(str(checkpoint), framework="numpy") as source:
        for name, parameter in model.named_parameters():
            if str(parameter.dtype) != f"torch.{dtype}":
                raise ValueError(f"wrong loaded dtype: {name}: {parameter.dtype}")
            key = name.removeprefix("transformer.")
            original = source.get_tensor(key)
            if original.dtype != np.float32:
                raise ValueError(f"checkpoint tensor is not F32: {key}")
            expected = original.astype(dtype, copy=False)
            actual = parameter.detach().cpu().numpy()
            if not np.isfinite(actual).all() or not np.array_equal(actual, expected):
                raise ValueError(f"loaded weight differs from checkpoint: {name}")
            tensor_count += 1
            parameter_count += parameter.numel()
    if not tensor_count or model.lm_head.weight is not model.transformer.wte.weight:
        raise ValueError("expected a nonempty GPT-2 model with tied output embeddings")
    return {"requested_dtype": dtype, "observed_dtype": f"torch.{dtype}",
            "parameter_tensors": tensor_count, "parameter_count": parameter_count,
            "checkpoint_comparison": "exact F32" if dtype == "float32" else "exact F32-to-F16 cast",
            "all_equal": True, "lm_head_tied": True}


def delta_receipt(deltas: np.ndarray, token_index: int) -> dict:
    values = np.asarray(deltas, dtype="<f4", order="C")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("nominal interventions must be a finite batch of vectors")
    return {"token_index": int(token_index), "shape": list(values.shape),
            "nominal_fp32_sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def require_fixed_field(reference: list[dict], candidate: list[dict]) -> None:
    if not reference or reference != candidate:
        raise ValueError("fixed-field intervention trace differs from FP16 replay")


def calibrate_prompt(model, inputs, *, layer: int, batch_size: int, dtype: str) -> tuple:
    import torch

    output, hidden = fast._extract_prompt_outputs(model, inputs)
    observed = {str(t.dtype) for t in [output.logits, *output.hidden_states]}
    if observed != {f"torch.{dtype}"}:
        raise ValueError(f"forward dtype mismatch: {observed}")
    single = output.logits[0].detach().float().cpu().numpy()
    ids = inputs["input_ids"][0].detach().cpu().numpy()
    repeated = {key: value.repeat(batch_size, 1) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**repeated, return_dict=True).logits
    if str(logits.dtype) != f"torch.{dtype}":
        raise ValueError("batched forward changed dtype")
    batch = logits.detach().float().cpu().numpy()
    controls = SimpleNamespace(logits=logits[:1].detach().cpu().clone())
    records = []
    for token in range(1, len(ids) - 1):
        zero = fast._logits_with_deltas(
            model, inputs, layer=layer, token_index=token,
            deltas=np.zeros((batch_size, hidden.shape[-1]), dtype=np.float32),
        )
        base = batch[:, token]
        errors = {
            "zero_hook_max_abs": float(np.max(np.abs(zero - base))),
            "row_spread_max_abs": float(np.max(np.abs(base - base[0]))),
            "single_batch_logit_max_abs": float(np.max(np.abs(base - single[token]))),
            "single_batch_next_logprob_delta": float(
                _log_softmax(base[0])[ids[token + 1]] - _log_softmax(single[token])[ids[token + 1]]
            ),
        }
        if not all(np.isfinite(x) for x in errors.values()):
            raise ValueError(f"nonfinite zero calibration at token {token}")
        records.append({"token_index": token, **errors})
    return hidden, controls, records


def compare_replay(replay: pd.DataFrame, historical: pd.DataFrame, tolerance: float) -> dict:
    keys = ["family", "prompt_id", "token_index", "seed", "component", "alpha"]
    historical = historical[historical["prompt_id"].isin(replay["prompt_id"].unique())]
    columns = ["next_token_logprob_steered", "target_logprob_mass_steered", "control_logprob_mass_steered",
               "component_active", "natural_step_norm", "chart_norm", "hidden_direction_norm"]
    paired = replay[keys + columns].merge(
        historical[keys + columns], on=keys, how="outer", validate="one_to_one",
        suffixes=("_new", "_old"), indicator=True,
    )
    if not bool((paired["_merge"] == "both").all()):
        raise ValueError("legacy replay unit grid differs")
    errors = {}
    for column in columns:
        difference = paired[f"{column}_new"] - paired[f"{column}_old"]
        if not np.isfinite(difference).all():
            raise ValueError(f"nonfinite legacy comparison: {column}")
        errors[column] = float(difference.abs().max())
    return {"rows": len(paired), "tolerance": tolerance, "max_abs_errors": errors,
            "passed": bool(max(errors.values()) <= tolerance),
            "scope": "Steered outputs and field scalars; matched-batch baseline deltas need not equal historical single-batch deltas."}


def run(protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_pilot_protocol(protocol, ROOT)
    verify_frozen_files(protocol)
    output_root = ROOT / protocol["run_root"]
    output_root.mkdir(exist_ok=False)
    receipt = {"protocol": file_receipt(protocol_path), "started_utc": utc_now(), "status": "RUNNING"}
    save_json(output_root / "execution_started.json", receipt)
    try:
        _run_pilot(protocol, output_root)
    except Exception as exc:
        receipt.update({"status": "FAILED", "completed_utc": utc_now(), "error": f"{type(exc).__name__}: {exc}"})
        save_json(output_root / "execution_receipt.json", receipt)
        raise
    receipt.update({"status": "COMPLETED", "completed_utc": utc_now()})
    save_json(output_root / "execution_receipt.json", receipt)


def _run_pilot(protocol: dict, output_root: Path) -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("MPS device fallback must be disabled")
    import platform
    import torch
    import transformers

    design = protocol["design"]
    if torch.is_autocast_enabled("mps") or torch.is_autocast_enabled("cpu"):
        raise ValueError("autocast must be disabled")
    models = {}
    load_audits = {}
    tokenizer = None
    for dtype in ("float16", "float32"):
        model, tok = fast._load_model_and_tokenizer(
            protocol["model_path"], device=design["device"], local_files_only=True,
            trust_remote_code=False, torch_dtype=dtype,
        )
        if {parameter.device.type for parameter in model.parameters()} != {design["device"]}:
            raise ValueError("model is not entirely on the requested device")
        load_audits[dtype] = audit_model_weights(model, Path(protocol["model_path"]) / "model.safetensors", dtype)
        models[dtype] = model
        tokenizer = tok
        print(f"verified checkpoint tensors: {dtype}", flush=True)
    save_json(output_root / "load_audit.json", {
        "models": load_audits, "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
        "device": design["device"], "autocast": False, "device_fallback": False,
    })
    suite = {item["prompt_id"]: item for item in read_suite(ROOT / protocol["suite"])}
    caches = {}
    zero_audits = []
    batch_size = len(design["components"]) * len(design["alphas"])
    layer = design["row_constants"]["layer"]
    for prompt in protocol["prompts"]:
        item = suite[prompt["prompt_id"]]
        if item["family"] != prompt["family"]:
            raise ValueError("prompt family differs from frozen input")
        inputs = fast._prompt_inputs(tokenizer, item["text"], device=design["device"], max_length=design["max_length"])
        ids = inputs["input_ids"][0].detach().cpu().numpy()
        if ids.tolist() != prompt["input_ids"]:
            raise ValueError("tokenization differs from frozen input")
        cache = {"inputs": inputs, "input_ids": ids, "item": item}
        arrays = {}
        for dtype, model in models.items():
            hidden, controls, records = calibrate_prompt(model, inputs, layer=layer, batch_size=batch_size, dtype=dtype)
            for row in records:
                zero_audits.append({"prompt_id": prompt["prompt_id"], "dtype": dtype, **row})
            coord = hodge.make_semantic_coordinates(
                hidden, method="pca", n_components=design["pca_components"],
                normalize_hidden=design["normalize_hidden"], random_state=design["pca_seed"], verbose=False,
            )
            cache[dtype] = {"hidden": hidden, "controls": controls, "coord": coord}
            arrays.update({f"hidden_{dtype}": hidden, f"coords_{dtype}": coord.coords,
                           f"pca_components_{dtype}": coord.reducer.components_, f"pca_mean_{dtype}": coord.reducer.mean_})
        np.savez_compressed(output_root / f"{prompt['prompt_id']}_fields.npz", **arrays)
        caches[prompt["prompt_id"]] = cache
        print(f"zero-calibrated both precisions: {prompt['prompt_id']}", flush=True)
    zero_pass = all(row["zero_hook_max_abs"] <= protocol["tolerances"]["zero_hook_max_abs"]
                    and row["row_spread_max_abs"] == 0 for row in zero_audits)
    save_json(output_root / "zero_calibration.json", {"passed": zero_pass, "completed_utc": utc_now(),
                                                       "token_units": zero_audits})
    if not zero_pass:
        raise ValueError("zero-control calibration failed before nonzero interventions")

    semantic_sets = fast._load_semantic_target_sets(ROOT / protocol["target_set_file"])
    historical = pd.read_csv(ROOT / protocol["historical_raw"])
    traces = {}
    for arm, (model_dtype, field_dtype) in ARMS.items():
        arm_rows = []
        arm_traces = []
        for prompt in protocol["prompts"]:
            cache = caches[prompt["prompt_id"]]
            item = cache["item"]
            field = cache[field_dtype]
            target_set = fast._semantic_set_key(semantic_sets, requested_key=None, prompt_id=item["prompt_id"], family=item["family"])
            targets, controls = fast._semantic_token_ids(tokenizer, semantic_sets, target_set)
            original_forward = fast._logits_with_deltas

            # Audit the unmodified historical kernel's inputs without changing its numeric path.
            def traced_forward(*args, **kwargs):
                arm_traces.append({"prompt_id": item["prompt_id"], **delta_receipt(kwargs["deltas"], kwargs["token_index"])})
                return original_forward(*args, **kwargs)

            with patch.object(fast, "_logits_with_deltas", traced_forward):
                rows, _energy, _topology = fast._layer_rows(
                    model=models[model_dtype], tokenizer=tokenizer, inputs=cache["inputs"], input_ids=cache["input_ids"],
                    outputs=cache[model_dtype]["controls"], hidden=field["hidden"], coord=field["coord"],
                    prompt_id=item["prompt_id"], layer=layer, k=design["row_constants"]["k"],
                    alphas=design["alphas"], steering_components=design["components"], selector_component="coexact",
                    token_selectors=["all_interior"], token_indices=[], position_bins=[], position_bin_count=12,
                    random_seeds=design["seeds"], ridge=design["ridge"], complex_mode="matched_betti",
                    target_betti_1_fraction=design["row_constants"]["betti_1_fraction_target"],
                    node_ridge=design["node_ridge"], min_chart_norm=design["min_chart_norm"],
                    target_id=None, target_set=target_set, target_set_ids=targets, control_set_ids=controls,
                )
            for row in rows:
                row.update({"family": item["family"], "precision_arm": arm, "model_dtype": model_dtype,
                            "field_dtype": field_dtype, "baseline_mode": "matched_batch", "baseline_batch_size": batch_size})
            fast._write_csv(rows, output_root / arm / item["prompt_id"] / "steering_metrics.csv")
            arm_rows.extend(rows)
            print(f"completed {arm}: {item['prompt_id']}", flush=True)
        frame = pd.DataFrame(arm_rows)
        grid = validate_protocol_rows(frame, protocol)
        frame.to_csv(output_root / arm / "summary.csv", index=False)
        save_json(output_root / arm / "design_validation.json", grid)
        save_json(output_root / arm / "delta_trace.json", arm_traces)
        traces[arm] = arm_traces
        if arm == "fp16_replay":
            audit = compare_replay(frame, historical, protocol["tolerances"]["legacy_replay_max_abs"])
            save_json(output_root / "replay_audit.json", audit)
            if not audit["passed"]:
                raise ValueError("FP16 replay differs from frozen historical data; do not proceed")
        if arm == "fp32_fixed_field":
            require_fixed_field(traces["fp16_replay"], arm_traces)
            save_json(output_root / "fixed_field_audit.json", {"passed": True, "matched_batches": len(arm_traces),
                                                               "scope": "Identical nominal F32 delta bytes before model-dtype addition; no field/scale change."})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
