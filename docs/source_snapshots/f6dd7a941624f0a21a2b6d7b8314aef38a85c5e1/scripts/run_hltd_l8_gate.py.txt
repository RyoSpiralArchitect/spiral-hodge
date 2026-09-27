#!/usr/bin/env python3
"""Freeze/run native-FP32 L8, conditional on a five-text L7 runtime bridge."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_hltd_precision_gate as precision
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, validate_protocol_rows, verify_frozen_files
from scripts.run_hltd_precision_full_gate import audit_model_weight_bytes

REFERENCE = "docs/data/hltd_precision_l7_full/protocol.json"
BRIDGE_IDS = ["literal_01", "literal_03", "metaphor_01", "identity_01", "ontology_01"]
PACKAGES = ["torch", "transformers", "numpy", "pandas", "scipy", "scikit-learn", "safetensors", "tokenizers"]


def runtime_snapshot() -> dict:
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES}}


def stage_protocol(protocol: dict, stage: str) -> dict:
    result = copy.deepcopy(protocol)
    if stage == "bridge":
        result["prompts"] = [p for p in result["prompts"] if p["prompt_id"] in BRIDGE_IDS]
        result["design"]["row_constants"]["layer"] = 7
    elif stage != "l8":
        raise ValueError(f"unknown stage: {stage}")
    return result


def validate_protocol(protocol: dict, root: Path = ROOT) -> None:
    reference = json.loads((root / REFERENCE).read_text())
    design = copy.deepcopy(reference["design"])
    design["row_constants"]["layer"] = 8
    for key, expected in [("reference_protocol", REFERENCE), ("prompts", reference["prompts"]),
                          ("design", design), ("analysis", reference["analysis"]),
                          ("suite", reference["suite"]), ("target_set_file", reference["target_set_file"])]:
        if protocol[key] != expected:
            raise ValueError(f"changed L8 contract: {key}")
    if protocol["bridge"]["prompt_ids"] != BRIDGE_IDS or protocol["bridge"]["max_abs_early_change"] != 0.05:
        raise ValueError("changed bridge selection or tolerance")
    if (protocol["primary"]["metric"], protocol["primary"]["contrast_type"], protocol["primary"]["bins"]) != (
        "next_token_logprob_delta", "odd", [0, 1, 2, 3],
    ):
        raise ValueError("changed L8 endpoint")
    if protocol["tolerances"] != {"zero_hook_max_abs": 0.0001}:
        raise ValueError("changed zero-control tolerance")
    expected_sha = next(r["sha256"] for r in reference["frozen_files"] if r["path"].endswith("/model.safetensors"))
    if protocol["checkpoint_sha256"] != expected_sha:
        raise ValueError("changed source checkpoint")
    if Path(protocol["run_root"]).is_absolute() or not protocol["run_root"].startswith("spiral_out_") or len(Path(protocol["run_root"]).parts) != 1:
        raise ValueError("run root must be a new top-level spiral_out_ directory")


def freeze(path: Path, model_path: Path, run_root: str) -> dict:
    reference = json.loads((ROOT / REFERENCE).read_text())
    model_path = model_path.expanduser().absolute()
    checkpoint = file_receipt(model_path / "model.safetensors")
    old_checkpoint = next(r for r in reference["frozen_files"] if r["path"].endswith("/model.safetensors"))
    if checkpoint["sha256"] != old_checkpoint["sha256"]:
        raise ValueError("local checkpoint is not the frozen native-F32 source")
    protocol = {key: copy.deepcopy(reference[key]) for key in ["suite", "target_set_file", "prompts", "design", "analysis"]}
    protocol["design"]["row_constants"]["layer"] = 8
    protocol.update({
        "schema_version": 1, "protocol_id": "hltd-l8-native-fp32-runtime-bridge-v1",
        "reference_protocol": REFERENCE, "model_path": str(model_path), "checkpoint_sha256": checkpoint["sha256"],
        "run_root": run_root, "runtime": runtime_snapshot(), "tolerances": {"zero_hook_max_abs": 0.0001},
        "prior_input_audit": {"original_frozen_file_count": len(reference["frozen_files"]),
            "unchanged_noncheckpoint_files": len(reference["frozen_files"]) - 1,
            "original_checkpoint_path": old_checkpoint["path"],
            "original_checkpoint_present": (ROOT / old_checkpoint["path"]).exists(),
            "cached_checkpoint_matches_original_sha256": True},
        "knowledge_at_freeze": "All historical L7 results and earlier exploratory layer/position results are known. "
            "L8 was selected after those observations, not as a blinded layer. No new native-FP32 L8 treatment has been inspected. "
            "Transformers changed from 5.13.0 to 4.57.6; the original checkpoint path is missing but the cached bytes match. "
            "Freeze before both new L7 bridge and L8 nonzero doses. No independent prompt replication.",
        "primary": {"metric": "next_token_logprob_delta", "contrast_type": "odd", "bins": [0, 1, 2, 3],
            "pass_rule": "Complete 80 early prompt/bins and strictly positive lower 95% prompt-bootstrap bound. "
                "Otherwise NOT_SUPPORTED or INSUFFICIENT_COVERAGE. No requirement that every individual prompt be positive.",
            "missing_rule": "No imputation, prompt exclusion, bin reselection or threshold change."},
        "bridge": {"prompt_ids": BRIDGE_IDS, "max_abs_early_change": 0.05,
            "selection": "Original four FP32 pilot texts plus the previously identified literal_03 sensitivity case; original suite order.",
            "reference_raw": reference["run_root"] + "/fp32_rebuilt_field/summary.csv",
            "reference_coefficients": "docs/data/hltd_precision_l7_full/fp32_rebuilt_field__summary_prompt_bin_response_coefficients.csv",
            "pass_rule": "Complete early coverage, all five current early coefficients positive, <=0.05 absolute change per prompt, "
                "identical full-interior activity masks, valid exact raw grid and independent coefficient audit. "
                "Failure stops before L8, without retries or relaxed gates.",
            "boundary": "Prespecified engineering continuity on five known texts only, not exact runtime or full20 equivalence."},
        "secondary": ["Full L8 position/phase and lexical-margin profiles, descriptive without multiplicity correction.",
            "Per-prompt L8 minus previous-runtime L7 coefficients, descriptive only: runtimes and layer-relative dose norms differ.",
            "Keep literal_03 and all inactive positions in the saved evidence."],
        "validity": ["Native FP32 source-byte check on actual in-run weights before any nonzero intervention.",
            "Explicit MPS, no autocast/device fallback, all forward activations/logits float32, attention implementation SDPA.",
            "Matched batch-12 baseline, zero-hook tolerance 1e-4, exact row equality; all stage zero controls before stage treatments.",
            "PCA is fitted to all layers/positions of each full teacher-forced text, including future tokens. This is an offline field, not causal online prediction."],
        "command_argv": [sys.executable, "-u", "scripts/run_hltd_l8_gate.py", "--protocol", str(path.relative_to(ROOT))],
        "analysis_argv": [sys.executable, "scripts/analyze_hltd_l8_gate.py", "--protocol", str(path.relative_to(ROOT))],
    })
    paths = [ROOT / r["path"] for r in reference["frozen_files"] if r != old_checkpoint]
    for expected in reference["frozen_files"]:
        if expected == old_checkpoint:
            continue
        actual = file_receipt(ROOT / expected["path"])
        if any(actual[k] != expected[k] for k in ["sha256", "bytes"]):
            raise ValueError(f"historical source/input changed: {expected['path']}")
    paths.extend(model_path.iterdir())
    paths.extend(ROOT / name for name in [REFERENCE, protocol["bridge"]["reference_raw"],
        protocol["bridge"]["reference_coefficients"], "scripts/run_hltd_l8_gate.py", "scripts/analyze_hltd_l8_gate.py",
        "scripts/audit_hltd_precision_raw.py", "scripts/run_hltd_precision_full_gate.py", "tests/test_hltd_l8_gate.py"])
    import transformers
    transformer_root = Path(transformers.__file__).parent
    paths.extend(transformer_root / name for name in ["modeling_utils.py", "modeling_attn_mask_utils.py",
        "masking_utils.py", "integrations/sdpa_attention.py", "models/gpt2/modeling_gpt2.py"])
    protocol["frozen_files"] = []
    for source in sorted(set(paths)):
        if source.is_file():
            record = file_receipt(source)
            record["path"] = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
            protocol["frozen_files"].append(record)
    validate_protocol(protocol)
    if (ROOT / run_root).exists():
        raise FileExistsError(ROOT / run_root)
    protocol["frozen_utc"] = precision.utc_now()
    precision.save_json(path, protocol)
    return protocol


def run_stage(model, tokenizer, protocol: dict, output: Path) -> pd.DataFrame:
    fast = precision.fast
    design = protocol["design"]
    layer = design["row_constants"]["layer"]
    output.mkdir()
    suite = {p["prompt_id"]: p for p in precision.read_suite(ROOT / protocol["suite"])}
    caches, zero_rows = {}, []
    for prompt in protocol["prompts"]:
        item = suite[prompt["prompt_id"]]
        if item["family"] != prompt["family"]:
            raise ValueError("prompt family differs")
        inputs = fast._prompt_inputs(tokenizer, item["text"], device=design["device"], max_length=design["max_length"])
        ids = inputs["input_ids"][0].detach().cpu().numpy()
        if ids.tolist() != prompt["input_ids"]:
            raise ValueError("tokenization differs from frozen text")
        hidden, controls, records = precision.calibrate_prompt(model, inputs, layer=layer, batch_size=12, dtype="float32")
        zero_rows.extend({"prompt_id": item["prompt_id"], "layer": layer, "dtype": "float32", **r} for r in records)
        coord = precision.hodge.make_semantic_coordinates(hidden, method="pca", n_components=design["pca_components"],
            normalize_hidden=design["normalize_hidden"], random_state=design["pca_seed"], verbose=False)
        caches[item["prompt_id"]] = (item, inputs, ids, hidden, controls, coord)
        np.savez_compressed(output / f"{item['prompt_id']}_fields.npz", hidden=hidden, coords=coord.coords,
                            pca_components=coord.reducer.components_, pca_mean=coord.reducer.mean_)
        print(f"L{layer} zero-calibrated: {item['prompt_id']}", flush=True)
    passed = all(r["zero_hook_max_abs"] <= protocol["tolerances"]["zero_hook_max_abs"] and r["row_spread_max_abs"] == 0 for r in zero_rows)
    precision.save_json(output / "zero_calibration.json", {"passed": passed, "completed_utc": precision.utc_now(), "token_units": zero_rows})
    if not passed:
        raise ValueError(f"L{layer} zero calibration failed before treatment")
    semantic_sets = fast._load_semantic_target_sets(ROOT / protocol["target_set_file"])
    all_rows, traces = [], []
    precision.save_json(output / "treatment_started.json", {"started_utc": precision.utc_now(), "layer": layer})
    for item, inputs, ids, hidden, controls, coord in caches.values():
        target_set = fast._semantic_set_key(semantic_sets, requested_key=None, prompt_id=item["prompt_id"], family=item["family"])
        targets, control_ids = fast._semantic_token_ids(tokenizer, semantic_sets, target_set)
        original = fast._logits_with_deltas

        def traced(*args, **kwargs):
            traces.append({"prompt_id": item["prompt_id"], **precision.delta_receipt(kwargs["deltas"], kwargs["token_index"])})
            return original(*args, **kwargs)

        with patch.object(fast, "_logits_with_deltas", traced):
            rows, energy, topology = fast._layer_rows(
                model=model, tokenizer=tokenizer, inputs=inputs, input_ids=ids, outputs=controls, hidden=hidden, coord=coord,
                prompt_id=item["prompt_id"], layer=layer, k=design["row_constants"]["k"], alphas=design["alphas"],
                steering_components=design["components"], selector_component="coexact", token_selectors=["all_interior"],
                token_indices=[], position_bins=[], position_bin_count=12, random_seeds=design["seeds"], ridge=design["ridge"],
                complex_mode="matched_betti", target_betti_1_fraction=design["row_constants"]["betti_1_fraction_target"],
                node_ridge=design["node_ridge"], min_chart_norm=design["min_chart_norm"], target_id=None,
                target_set=target_set, target_set_ids=targets, control_set_ids=control_ids,
            )
        for row in rows:
            row.update({"family": item["family"], "model_dtype": "float32", "field_dtype": "float32",
                        "baseline_mode": "matched_batch", "baseline_batch_size": 12})
        fast._write_csv(rows, output / item["prompt_id"] / "steering_metrics.csv")
        precision.save_json(output / item["prompt_id"] / "field_summary.json", {"energy": energy, "topology": topology})
        all_rows.extend(rows)
        print(f"L{layer} completed: {item['prompt_id']}", flush=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / "summary.csv", index=False)
    precision.save_json(output / "delta_trace.json", traces)
    precision.save_json(output / "design_validation.json", validate_protocol_rows(frame, protocol))
    return frame


def _run(protocol: dict, output: Path) -> bool:
    import torch
    from scripts.analyze_hltd_l8_gate import analyze_stage, compare_bridge

    if runtime_snapshot() != protocol["runtime"]:
        raise ValueError("runtime changed after freeze")
    if torch.is_autocast_enabled("mps") or torch.is_autocast_enabled("cpu") or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("autocast and MPS fallback are forbidden")
    model, tokenizer = precision.fast._load_model_and_tokenizer(protocol["model_path"], device="mps",
        local_files_only=True, trust_remote_code=False, torch_dtype="float32")
    if {p.device.type for p in model.parameters()} != {"mps"} or model.config._attn_implementation != "sdpa":
        raise ValueError("unexpected device or attention implementation")
    audit = audit_model_weight_bytes(model, Path(protocol["model_path"]) / "model.safetensors", "float32")
    precision.save_json(output / "load_audit.json", {"model": audit, "runtime": runtime_snapshot(), "device": "mps",
        "attention_implementation": model.config._attn_implementation, "autocast": False, "device_fallback": False})
    print("Native-FP32 source bytes verified; starting frozen L7 bridge", flush=True)
    bridge_protocol = stage_protocol(protocol, "bridge")
    current_rows = run_stage(model, tokenizer, bridge_protocol, output / "bridge")
    response = analyze_stage(output / "bridge", bridge_protocol)
    if not response["complete_early_coverage"]:
        verdict = {"passed": False, "status": "BRIDGE_INSUFFICIENT_COVERAGE", "response": response}
    else:
        previous_rows = pd.read_csv(ROOT / protocol["bridge"]["reference_raw"])
        previous_rows = previous_rows[previous_rows.prompt_id.isin(BRIDGE_IDS)]
        validate_protocol_rows(previous_rows, bridge_protocol)
        previous = pd.read_csv(ROOT / protocol["bridge"]["reference_coefficients"])
        table, verdict = compare_bridge(
            pd.read_csv(output / "bridge/analysis/summary_prompt_bin_response_coefficients.csv"),
            previous[previous.prompt_id.isin(BRIDGE_IDS)], current_rows, previous_rows, bridge_protocol,
        )
        table.to_csv(output / "bridge/analysis/early_runtime_comparison.csv", index=False)
        diagnostic = precision.compare_replay(current_rows, previous_rows, tolerance=1e-8)
        diagnostic["scope"] = "Diagnostic only: strict old-runtime replay is not the five-text engineering bridge criterion."
        precision.save_json(output / "bridge/analysis/strict_replay_diagnostic.json", diagnostic)
    verdict["evaluated_utc"] = precision.utc_now()
    precision.save_json(output / "bridge_verdict.json", verdict)
    print(json.dumps(verdict), flush=True)
    if not verdict["passed"]:
        return False
    run_stage(model, tokenizer, protocol, output / "l8")
    return True


def run(protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text())
    verify_frozen_files(protocol)
    validate_protocol(protocol)
    output = ROOT / protocol["run_root"]
    output.mkdir(exist_ok=False)
    receipt = {"protocol": file_receipt(protocol_path), "started_utc": precision.utc_now(), "status": "RUNNING"}
    precision.save_json(output / "execution_started.json", receipt)
    try:
        completed = _run(protocol, output)
        verify_frozen_files(protocol)
    except Exception as exc:
        receipt.update({"status": "FAILED", "completed_utc": precision.utc_now(), "error": f"{type(exc).__name__}: {exc}"})
        precision.save_json(output / "execution_receipt.json", receipt)
        raise
    receipt.update({"status": "COMPLETED" if completed else "STOPPED_BEFORE_L8", "completed_utc": precision.utc_now()})
    precision.save_json(output / "execution_receipt.json", receipt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--run-root")
    args = parser.parse_args(argv)
    if args.freeze:
        if args.model_path is None or args.run_root is None:
            parser.error("freeze requires --model-path and --run-root")
        protocol = freeze(args.protocol.absolute(), args.model_path, args.run_root)
        print(json.dumps({"protocol": str(args.protocol), "frozen_utc": protocol["frozen_utc"],
                          "files": len(protocol["frozen_files"]), "runtime": protocol["runtime"]}, indent=2))
    else:
        if args.model_path is not None or args.run_root is not None:
            parser.error("runtime overrides are forbidden; use the frozen protocol")
        run(args.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
