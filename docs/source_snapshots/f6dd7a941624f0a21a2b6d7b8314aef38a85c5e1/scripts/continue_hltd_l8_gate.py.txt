#!/usr/bin/env python3
"""Continue after the missing historical aggregate, without repeating treatment."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_hltd_l8_gate as original
from scripts.analyze_hltd_l8_gate import compare_bridge
from scripts.audit_hltd_precision_raw import audit_arm
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, validate_protocol_rows, verify_frozen_files


def require_receipts(records: list[dict], root: Path = ROOT) -> None:
    for expected in records:
        actual = file_receipt(root / expected["path"])
        if any(actual[k] != expected[k] for k in ["sha256", "bytes"]):
            raise ValueError(f"changed required input: {expected['path']}")


def validate_prior_attempt(protocol: dict, receipt: dict, root: Path = ROOT) -> None:
    expected_error = "FileNotFoundError: [Errno 2] No such file or directory: " + repr(str(root / protocol["bridge"]["reference_raw"]))
    if receipt.get("status") != "FAILED" or receipt.get("error") != expected_error:
        raise ValueError("only the missing historical aggregate failure can be continued")
    if (root / protocol["run_root"] / "l8").exists():
        raise ValueError("L8 already started; do not continue this attempt")


def freeze(previous_path: Path, path: Path, run_root: str) -> dict:
    previous = json.loads(previous_path.read_text())
    verify_frozen_files(previous)
    attempt = ROOT / previous["run_root"]
    receipt = json.loads((attempt / "execution_receipt.json").read_text())
    if receipt["protocol"]["sha256"] != file_receipt(previous_path)["sha256"]:
        raise ValueError("prior receipt is bound to another protocol")
    validate_prior_attempt(previous, receipt)
    manifest_path = ROOT / "docs/figures/hltd_precision_l7_full_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    reference = json.loads((ROOT / original.REFERENCE).read_text())
    names = [f"{reference['run_root']}/fp32_rebuilt_field/{p}/steering_metrics.csv" for p in original.BRIDGE_IDS]
    indexed = {r["path"]: r for r in manifest["ignored_source_artifacts"]}
    shards = [indexed[name] for name in names]
    require_receipts(shards)
    protocol = copy.deepcopy(previous)
    protocol["schema_version"] = 2
    protocol["protocol_id"] = "hltd-l8-native-fp32-runtime-bridge-v1-input-recovery"
    protocol["run_root"] = run_root
    protocol["knowledge_at_freeze"] += (
        " Amendment: the five new L7 bridge texts have been measured and analyzed; no L8 treatment exists. "
        "The first attempt stopped on a missing historical aggregate CSV after its bridge analysis. "
        "Original per-prompt CSVs still match the old manifest. Reuse the first attempt's raw measurements, "
        "recompute the same bridge against these shards, and retain every original threshold/endpoint. "
        "No historical aggregate byte-restoration or independent repeat is claimed."
    )
    protocol["bridge"].pop("reference_raw")
    protocol["bridge"]["reference_prompt_shards"] = shards
    protocol["continuation"] = {"previous_protocol": str(previous_path.relative_to(ROOT)),
                                "previous_run_root": previous["run_root"],
                                "mode": "reuse completed bridge, repair reference input, first L8 treatment only"}
    paths = [previous_path, attempt / "execution_receipt.json", attempt / "load_audit.json", manifest_path,
             Path(__file__), ROOT / "tests/test_hltd_l8_continuation.py"]
    paths.extend(f for f in (attempt / "bridge").rglob("*") if f.is_file())
    records = list(protocol["frozen_files"]) + shards
    for source in paths:
        record = file_receipt(source)
        record["path"] = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
        records.append(record)
    protocol["frozen_files"] = list({r["path"]: r for r in records}.values())
    require_receipts(protocol["frozen_files"])
    original.validate_protocol(protocol)
    if original.runtime_snapshot() != protocol["runtime"]:
        raise ValueError("runtime changed since the completed bridge")
    if (ROOT / run_root).exists():
        raise FileExistsError(ROOT / run_root)
    protocol["command_argv"] = [sys.executable, "-u", "scripts/continue_hltd_l8_gate.py", "--protocol", str(path.relative_to(ROOT))]
    protocol["analysis_argv"] = [sys.executable, "scripts/analyze_hltd_l8_gate.py", "--protocol", str(path.relative_to(ROOT))]
    protocol["frozen_utc"] = original.precision.utc_now()
    original.precision.save_json(path, protocol)
    return protocol


def continue_run(protocol: dict, output: Path) -> bool:
    import os
    import torch

    if original.runtime_snapshot() != protocol["runtime"]:
        raise ValueError("runtime changed after freeze")
    require_receipts(protocol["bridge"]["reference_prompt_shards"])
    previous_path = ROOT / protocol["continuation"]["previous_protocol"]
    previous = json.loads(previous_path.read_text())
    attempt = ROOT / previous["run_root"]
    validate_prior_attempt(previous, json.loads((attempt / "execution_receipt.json").read_text()))
    bridge_protocol = original.stage_protocol(protocol, "bridge")
    previous_rows = []
    families = {p["prompt_id"]: p["family"] for p in bridge_protocol["prompts"]}
    for record in protocol["bridge"]["reference_prompt_shards"]:
        frame = pd.read_csv(ROOT / record["path"])
        frame["family"] = frame.prompt_id.map(families)
        previous_rows.append(frame)
    previous_rows = pd.concat(previous_rows, ignore_index=True)
    validate_protocol_rows(previous_rows, bridge_protocol)
    current_rows = pd.read_csv(attempt / "bridge/summary.csv")
    validate_protocol_rows(current_rows, bridge_protocol)
    current_coeff = attempt / "bridge/analysis/summary_prompt_bin_response_coefficients.csv"
    raw_audit = audit_arm(attempt / "bridge/summary.csv", current_coeff, bridge_protocol)
    previous_coeff = pd.read_csv(ROOT / protocol["bridge"]["reference_coefficients"])
    previous_coeff = previous_coeff[previous_coeff.prompt_id.isin(original.BRIDGE_IDS)]
    previous_rows.to_csv(output / "reference_bridge_from_shards.csv", index=False)
    previous_coeff.to_csv(output / "reference_bridge_coefficients.csv", index=False)
    reference_audit = audit_arm(output / "reference_bridge_from_shards.csv",
                               output / "reference_bridge_coefficients.csv", bridge_protocol)
    original.precision.save_json(output / "reference_raw_audit.json", reference_audit)
    table, verdict = compare_bridge(pd.read_csv(current_coeff), previous_coeff,
                                    current_rows, previous_rows, bridge_protocol)
    verdict["evaluated_utc"] = original.precision.utc_now()
    verdict["reused_bridge_run"] = previous["run_root"]
    shutil.copytree(attempt / "bridge", output / "bridge")
    original.precision.save_json(output / "reference_recovery.json", {
        "source_shards": protocol["bridge"]["reference_prompt_shards"], "source_hashes_verified": True,
        "raw_rows": len(previous_rows), "family_source": protocol["suite"],
        "scope": "Original numeric row data, family restored from frozen prompt metadata; not original aggregate bytes."})
    original.precision.save_json(output / "bridge/analysis/reused_raw_audit.json", raw_audit)
    table.to_csv(output / "bridge/analysis/early_runtime_comparison.csv", index=False)
    diagnostic = original.precision.compare_replay(current_rows, previous_rows, tolerance=1e-8)
    diagnostic["scope"] = "Diagnostic only; bridge criterion remains the original coefficient/activity rule."
    original.precision.save_json(output / "bridge/analysis/strict_replay_diagnostic.json", diagnostic)
    original.precision.save_json(output / "bridge_verdict.json", verdict)
    print(json.dumps(verdict), flush=True)
    if not verdict["passed"]:
        return False
    if torch.is_autocast_enabled("mps") or torch.is_autocast_enabled("cpu") or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("autocast and MPS fallback are forbidden")
    model, tokenizer = original.precision.fast._load_model_and_tokenizer(protocol["model_path"], device="mps",
        local_files_only=True, trust_remote_code=False, torch_dtype="float32")
    if {p.device.type for p in model.parameters()} != {"mps"} or model.config._attn_implementation != "sdpa":
        raise ValueError("unexpected device or attention implementation")
    audit = original.audit_model_weight_bytes(model, Path(protocol["model_path"]) / "model.safetensors", "float32")
    original.precision.save_json(output / "load_audit.json", {"model": audit, "runtime": original.runtime_snapshot(),
        "device": "mps", "attention_implementation": "sdpa", "autocast": False, "device_fallback": False})
    original.run_stage(model, tokenizer, protocol, output / "l8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-from", type=Path)
    parser.add_argument("--run-root")
    args = parser.parse_args(argv)
    if args.freeze_from:
        if not args.run_root:
            parser.error("--freeze-from requires --run-root")
        protocol = freeze(args.freeze_from.absolute(), args.protocol.absolute(), args.run_root)
        print(json.dumps({"frozen_utc": protocol["frozen_utc"], "files": len(protocol["frozen_files"])}, indent=2))
    else:
        if args.run_root:
            parser.error("runtime overrides are forbidden")
        with patch.object(original, "_run", continue_run):
            original.run(args.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
