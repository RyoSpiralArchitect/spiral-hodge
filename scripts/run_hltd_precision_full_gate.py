#!/usr/bin/env python3
"""Extend the frozen precision pilot to its original 20-prompt signed suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_hltd_precision_gate as pilot
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, verify_frozen_files

FULL_REFERENCE = "docs/data/hltd_precision_l7_full/protocol.json"
FULL_REFERENCE_SHA256 = "cb00ff05f7a6345578ba446abb062bd94718ccf7dd41f8288d9a0199d8beed82"


def audit_model_weight_bytes(model, checkpoint: Path, dtype: str) -> dict:
    """Check the actual in-run tensors, including signed-zero byte identity."""
    from safetensors import safe_open

    if dtype not in {"float16", "float32"}:
        raise ValueError("only explicit float16/float32 are supported")
    tensors = []
    parameter_count = 0
    with safe_open(str(checkpoint), framework="numpy") as source:
        for name, parameter in model.named_parameters():
            if str(parameter.dtype) != f"torch.{dtype}":
                raise ValueError(f"wrong loaded dtype: {name}: {parameter.dtype}")
            original = source.get_tensor(name.removeprefix("transformer."))
            if original.dtype != np.float32:
                raise ValueError(f"checkpoint tensor is not F32: {name}")
            expected = original.astype(dtype, copy=False)
            actual = parameter.detach().cpu().numpy()
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError(f"invalid loaded tensor: {name}")
            actual_bytes = actual.tobytes(order="C")
            if actual_bytes != expected.tobytes(order="C"):
                raise ValueError(f"loaded bytes differ from checkpoint: {name}")
            tensors.append({"name": name, "shape": list(actual.shape), "bytes": len(actual_bytes),
                            "sha256": hashlib.sha256(actual_bytes).hexdigest(), "source_bytes_equal": True})
            parameter_count += parameter.numel()
    if not tensors or model.lm_head.weight is not model.transformer.wte.weight:
        raise ValueError("expected a nonempty GPT-2 model with tied output embeddings")
    return {"requested_dtype": dtype, "observed_dtype": f"torch.{dtype}",
            "parameter_tensors": len(tensors), "parameter_count": parameter_count,
            "checkpoint_comparison": "bytewise native F32" if dtype == "float32" else "bytewise F32-to-F16 cast",
            "all_equal": True, "all_bytes_equal": True, "lm_head_tied": True, "tensors": tensors,
            "timing": "actual model objects, before calibration and nonzero interventions"}


def validate_full_protocol(protocol: dict, root: Path = ROOT) -> None:
    pilot.validate_recorded_protocol(protocol, reference_path=FULL_REFERENCE,
                                    reference_sha256=FULL_REFERENCE_SHA256, label="full", root=root)
    previous = json.loads((root / protocol["reference_protocol"]).read_text())
    pilot_protocol = json.loads((root / protocol["pilot_protocol"]).read_text())
    if (root / protocol["run_root"]).resolve() in {
        (root / previous["run_root"]).resolve(), (root / pilot_protocol["run_root"]).resolve(),
    }:
        raise ValueError("full gate must not overwrite previous run locations")


def run(protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text())
    validate_full_protocol(protocol, ROOT)
    verify_frozen_files(protocol)
    output = ROOT / protocol["run_root"]
    output.mkdir(exist_ok=False)
    receipt = {"protocol": file_receipt(protocol_path), "started_utc": pilot.utc_now(), "status": "RUNNING"}
    pilot.save_json(output / "execution_started.json", receipt)
    try:
        # Only the load audit is replaced; the frozen numerical runner is unchanged.
        with patch.object(pilot, "audit_model_weights", audit_model_weight_bytes):
            pilot._run_pilot(protocol, output)
    except Exception as exc:
        receipt.update({"status": "FAILED", "completed_utc": pilot.utc_now(), "error": f"{type(exc).__name__}: {exc}"})
        pilot.save_json(output / "execution_receipt.json", receipt)
        raise
    receipt.update({"status": "COMPLETED", "completed_utc": pilot.utc_now()})
    pilot.save_json(output / "execution_receipt.json", receipt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
