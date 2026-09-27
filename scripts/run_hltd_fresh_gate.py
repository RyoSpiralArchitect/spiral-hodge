#!/usr/bin/env python3
"""Freeze and run the new-text L7/L8 gate in a single native-FP32 runtime."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_hltd_l8_gate as previous
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, verify_frozen_files
from scripts.hltd_historical_sources import verify_historical_inputs
from scripts.run_hltd_precision_gate import save_json, utc_now
from scripts.run_hltd_steering_suite import read_suite

REFERENCE = "docs/data/hltd_signed_l8_position/protocol_continuation.json"
REFERENCE_SHA256 = "224030e9a3c55db4fe73b2167dc93accb3db5dd065f22940de6b6bec2c93081e"
FRESH_REFERENCE = "docs/data/hltd_fresh_l7_l8/protocol.json"
FRESH_REFERENCE_SHA256 = "74cfcb5f91ba56ffabc1eb8d8ef0f3006237921bc57175868b412edcec624ca1"
SUITE = "data/hltd_fresh20_prompt_suite.jsonl"
FAMILIES = ["literal_stable", "metaphor_shift", "identity_stress", "ontology_collapse"]


def normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def novelty_audit(fresh: list[dict], historical: list[dict]) -> dict:
    ids = [p["prompt_id"] for p in fresh]
    texts = [normalized_text(p["text"]) for p in fresh]
    if len(ids) != len(set(ids)) or len(texts) != len(set(texts)):
        raise ValueError("duplicate fresh prompt ID or text")
    old_ids = {p["prompt_id"] for p in historical}
    old_texts = {normalized_text(p["text"]) for p in historical}
    if old_ids.intersection(ids) or old_texts.intersection(texts):
        raise ValueError("fresh prompts overlap historical IDs or normalized texts")

    def grams(text: str) -> set[tuple]:
        words = re.findall(r"[a-z0-9]+", text.casefold())
        return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}

    neighbors = []
    for prompt in fresh:
        a = grams(prompt["text"])
        scores = [(len(a & grams(p["text"])) / max(1, len(a | grams(p["text"]))), p["prompt_id"])
                  for p in historical]
        score, closest = max(scores, default=(0.0, ""))
        neighbors.append({"prompt_id": prompt["prompt_id"], "closest_historical_prompt": closest,
                          "word_trigram_jaccard": score,
                          "normalized_text_sha256": hashlib.sha256(normalized_text(prompt["text"]).encode()).hexdigest()})
    return {"exact_duplicate_count": 0, "historical_rows": len(historical), "fresh_rows": len(fresh),
            "nearest_historical": neighbors, "max_word_trigram_jaccard": max(r["word_trigram_jaccard"] for r in neighbors),
            "scope": "Repository data JSONL suites only; overlap score is descriptive, not an admission threshold. "
                "Authored new strings, not a random or external sample; no claim about training exposure or novel syntax/style."}


def layer_protocol(protocol: dict, layer: int) -> dict:
    if layer not in protocol["layers"]:
        raise ValueError("layer outside the frozen pair")
    result = copy.deepcopy(protocol)
    result["design"]["row_constants"]["layer"] = layer
    return result


def load_recorded_protocol(path: str, sha256: str, root: Path) -> dict:
    payload = (root / path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"recorded fresh-gate reference changed: {path}")
    return json.loads(payload)


def validate_protocol(protocol: dict, root: Path = ROOT) -> None:
    reference = load_recorded_protocol(REFERENCE, REFERENCE_SHA256, root)
    recorded = load_recorded_protocol(FRESH_REFERENCE, FRESH_REFERENCE_SHA256, root)
    expected_design = copy.deepcopy(reference["design"])
    expected_design["row_constants"]["layer"] = 7
    for key, value in [("reference_protocol", REFERENCE), ("suite", SUITE), ("layers", [7, 8]),
                       ("design", expected_design), ("analysis", reference["analysis"]),
                       ("model_path", reference["model_path"]), ("runtime", reference["runtime"]),
                       ("target_set_file", reference["target_set_file"]),
                       ("tolerances", reference["tolerances"]), ("checkpoint_sha256", reference["checkpoint_sha256"]),
                       ("paired_bootstrap", {"samples": 5000, "seed": 2718, "unit": "paired prompt", "quantiles": [0.025, 0.975]})]:
        if protocol[key] != value:
            raise ValueError(f"changed fresh-text contract: {key}")
    previous.validate_model_binding(protocol, reference, root)
    primary = protocol["primary"]
    if (primary["metric"], primary["contrast_type"], primary["bins"]) != ("next_token_logprob_delta", "odd", [0, 1, 2, 3]):
        raise ValueError("changed primary endpoint")
    prompts = protocol["prompts"]
    if prompts != recorded["prompts"]:
        raise ValueError("changed recorded fresh prompt inventory")
    previous.validate_data_binding(protocol, recorded, "recorded fresh")
    if Counter(p["family"] for p in prompts) != Counter({family: 5 for family in FAMILIES}):
        raise ValueError("expected exactly five fresh prompts in each of four families")
    if len({p["prompt_id"] for p in prompts}) != 20:
        raise ValueError("nonunique prompt IDs")
    for prompt in prompts:
        if not 24 <= prompt["token_count"] <= 48 or prompt["token_count"] != len(prompt["input_ids"]):
            raise ValueError("invalid prompt token count; truncation forbidden")
    out = Path(protocol["run_root"])
    if out.is_absolute() or len(out.parts) != 1 or not out.name.startswith("spiral_out_"):
        raise ValueError("run root must be a new top-level spiral_out_ directory")


def freeze(path: Path, run_root: str) -> dict:
    from transformers import AutoTokenizer

    reference = load_recorded_protocol(REFERENCE, REFERENCE_SHA256, ROOT)
    reference_code_snapshots = verify_historical_inputs(reference, ROOT)
    if previous.runtime_snapshot() != reference["runtime"]:
        raise ValueError("runtime changed since L8; stop before choosing a new comparison")
    fresh = read_suite(ROOT / SUITE)
    historical_paths = sorted(p for p in (ROOT / "data").glob("*.jsonl") if p != ROOT / SUITE)
    historical = [item for p in historical_paths for item in read_suite(p)]
    novelty = novelty_audit(fresh, historical)
    tokenizer = AutoTokenizer.from_pretrained(reference["model_path"], local_files_only=True, trust_remote_code=False)
    protocol = {k: copy.deepcopy(reference[k]) for k in ["model_path", "checkpoint_sha256", "runtime", "target_set_file", "design", "analysis", "tolerances"]}
    protocol["design"]["row_constants"]["layer"] = 7
    protocol.update({"schema_version": 1, "protocol_id": "hltd-fresh20-l7-l8-native-fp32-v1",
        "reference_protocol": REFERENCE, "suite": SUITE, "layers": [7, 8], "run_root": run_root,
        "reference_code_snapshots": reference_code_snapshots,
        "knowledge_at_freeze": "This prospective freeze repeats the exact recorded fresh20 text/token inventory; "
            "its prior outcomes are already known. It is a rerun, not a new held-out-text replication. "
            "No response-based replacement, family reassignment, bin selection, or optional stopping. "
            "A different prompt pool requires a separately defined protocol, not changes to this v1 suite.",
        "primary": {"metric": "next_token_logprob_delta", "contrast_type": "odd", "bins": [0, 1, 2, 3],
            "pass_rule": "Each layer must have all 80 early prompt/bins and strictly positive lower 95% prompt-bootstrap bound. "
                "The joint status is BOTH_LAYERS_SUPPORTED_ON_FRESH_TEXTS only if both pass. "
                "Otherwise report NOT_SUPPORTED_AT_BOTH_LAYERS or INSUFFICIENT_COVERAGE, retaining both layer results.",
            "missing_rule": "No imputation, excluded prompts, reduced-prompt primary estimate, or sign requirement for each prompt."},
        "paired_bootstrap": {"samples": 5000, "seed": 2718, "unit": "paired prompt", "quantiles": [0.025, 0.975]},
        "secondary": ["Prespecified paired early L8-minus-L7 coefficient interval, 5000 draws seed2718 over sorted prompt IDs; "
            "descriptive and unadjusted, not a layer-only fixed-magnitude effect.",
            "Full position/phase profiles and unchanged legacy lexical target/control margins. "
            "These vocabularies were built around old texts, so fresh-text margins are transfer diagnostics, not semantic-control success.",
            "Activity overlap and per-prompt natural hidden-step norm scales across layers."],
        "execution_policy": "Load exactly one native-FP32 model instance and run L7 then L8, without reading response summaries between layers. "
            "Audit source bytes before and after both runs. Require matching per-prompt unsteered hidden/PCA array bytes across layers. "
            "Run both layers regardless of response sign; stop only for failed numerical/input validity. No automatic retry.",
        "boundaries": ["Teacher-forced full-text centered fields and PCA include future positions: offline, not online-available directions.",
            "Alpha uses each layer's natural hidden-step norm. No common-norm arm is included.",
            "No population, cross-model, fluency, identity/affordance-control or semantic-success claim.",
            "Exact text novelty does not establish independent style, syntax, concept families or model pretraining exposure."],
        "novelty": novelty,
        "prompts": [{"prompt_id": item["prompt_id"], "family": item["family"], "text": item["text"],
                     "input_ids": tokenizer.encode(item["text"]), "token_count": len(tokenizer.encode(item["text"]))} for item in fresh],
        "command_argv": [sys.executable, "-u", "scripts/run_hltd_fresh_gate.py", "--protocol", str(path.relative_to(ROOT))],
        "analysis_argv": [sys.executable, "scripts/analyze_hltd_fresh_gate.py", "--protocol", str(path.relative_to(ROOT))],
    })
    if (ROOT / run_root).exists():
        raise FileExistsError(ROOT / run_root)
    # Every required input is mandatory; never silently skip a missing source.
    sources = {ROOT / r["path"] for r in reference["frozen_files"] if Path(r["path"]).suffix == ".py"}
    sources.update(historical_paths)
    sources.update((Path(protocol["model_path"])).iterdir())
    sources.update(ROOT / p for p in [REFERENCE, FRESH_REFERENCE, SUITE, protocol["target_set_file"],
        "scripts/run_hltd_fresh_gate.py", "scripts/analyze_hltd_fresh_gate.py", "tests/test_hltd_fresh_gate.py",
        "scripts/hltd_historical_sources.py"])
    protocol["frozen_files"] = []
    for source in sorted(sources):
        record = file_receipt(source)
        record["path"] = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
        protocol["frozen_files"].append(record)
    validate_protocol(protocol)
    protocol["frozen_utc"] = utc_now()
    save_json(path, protocol)
    return protocol


def compare_charts(root: Path, protocol: dict) -> dict:
    result = []
    for prompt in protocol["prompts"]:
        arrays = {}
        for layer in protocol["layers"]:
            path = root / f"l{layer}/{prompt['prompt_id']}_fields.npz"
            with np.load(path, allow_pickle=False) as data:
                arrays[layer] = {key: (str(data[key].dtype), data[key].shape, hashlib.sha256(data[key].tobytes()).hexdigest())
                                 for key in ["hidden", "coords", "pca_components", "pca_mean"]}
        if arrays[7] != arrays[8]:
            raise ValueError(f"unsteered hidden/chart bytes changed: {prompt['prompt_id']}")
        result.append({"prompt_id": prompt["prompt_id"], "arrays": arrays[7], "byte_identical": True})
    return {"passed": True, "prompts": result, "scope": "Same full-text hidden activations/PCA charts across layer runs; component fields and layer-relative scales still differ."}


def _run(protocol: dict, output: Path) -> None:
    import torch

    if previous.runtime_snapshot() != protocol["runtime"]:
        raise ValueError("runtime changed after freeze")
    if torch.is_autocast_enabled("mps") or torch.is_autocast_enabled("cpu") or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("autocast and MPS fallback forbidden")
    model, tokenizer = previous.precision.fast._load_model_and_tokenizer(protocol["model_path"], device="mps",
        local_files_only=True, trust_remote_code=False, torch_dtype="float32")
    if {p.device.type for p in model.parameters()} != {"mps"} or model.config._attn_implementation != "sdpa":
        raise ValueError("unexpected device or attention implementation")
    checkpoint = Path(protocol["model_path"]) / "model.safetensors"
    audit = previous.audit_model_weight_bytes(model, checkpoint, "float32")
    if audit["parameter_count"] != 124439808 or audit["parameter_tensors"] != 148:
        raise ValueError("unexpected model shape")
    save_json(output / "load_audit.json", {"model": audit, "runtime": previous.runtime_snapshot(), "device": "mps",
        "attention_implementation": "sdpa", "autocast": False, "device_fallback": False, "model_load_count": 1})
    for layer in protocol["layers"]:
        previous.run_stage(model, tokenizer, layer_protocol(protocol, layer), output / f"l{layer}")
    save_json(output / "shared_chart_audit.json", compare_charts(output, protocol))
    final_audit = previous.audit_model_weight_bytes(model, checkpoint, "float32")
    if final_audit != audit:
        raise ValueError("model weights changed during the paired run")
    save_json(output / "final_weight_audit.json", {"passed": True, "parameter_tensors": 148,
        "parameter_count": 124439808, "source_bytes_equal_after_both_layers": True})


def run(path: Path) -> None:
    protocol = json.loads(path.read_text())
    verify_frozen_files(protocol)
    validate_protocol(protocol)
    output = ROOT / protocol["run_root"]
    output.mkdir(exist_ok=False)
    receipt = {"protocol": file_receipt(path), "started_utc": utc_now(), "status": "RUNNING"}
    save_json(output / "execution_started.json", receipt)
    try:
        _run(protocol, output)
        verify_frozen_files(protocol)
    except Exception as exc:
        receipt.update({"status": "FAILED", "completed_utc": utc_now(), "error": f"{type(exc).__name__}: {exc}"})
        save_json(output / "execution_receipt.json", receipt)
        raise
    receipt.update({"status": "COMPLETED", "completed_utc": utc_now()})
    save_json(output / "execution_receipt.json", receipt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run-root")
    args = parser.parse_args(argv)
    if args.freeze:
        if not args.run_root:
            parser.error("--freeze requires --run-root")
        p = freeze(args.protocol.absolute(), args.run_root)
        print(json.dumps({"frozen_utc": p["frozen_utc"], "files": len(p["frozen_files"]), "novelty": p["novelty"],
                          "tokens": [q["token_count"] for q in p["prompts"]],
                          "rows_per_layer": 96 * sum(q["token_count"] - 2 for q in p["prompts"])}, indent=2))
    else:
        if args.run_root:
            parser.error("runtime overrides are forbidden")
        run(args.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
