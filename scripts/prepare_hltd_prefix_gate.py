#!/usr/bin/env python3
"""Prepare a pinned prefix-transfer design without loading model parameters."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
from numbers import Integral
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_hltd_signed_layer_gate import file_receipt, verify_frozen_files
from scripts.run_hltd_fresh_gate import normalized_text, novelty_audit
from scripts.run_hltd_precision_gate import save_json, utc_now
from scripts.run_hltd_steering_suite import read_suite

SPEC_PATH = "docs/data/hltd_prefix_transfer/spec.json"
SPEC_SHA256 = "b59e5761922086414810797f785d43b3c30eff45558e009f4303a131ff0c1767"


def load_spec(root: Path = ROOT) -> dict:
    payload = (root / SPEC_PATH).read_bytes()
    if hashlib.sha256(payload).hexdigest() != SPEC_SHA256:
        raise ValueError("canonical prefix-transfer specification changed")
    return json.loads(payload)


def validate_splits(rows: list[dict], spec: dict) -> None:
    if Counter(row["split"] for row in rows) != Counter(spec["split_counts"]):
        raise ValueError("changed split counts")
    ids = [row["prompt_id"] for row in rows]
    texts = [normalized_text(row["text"]) for row in rows]
    if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
        raise ValueError("overlapping prompt IDs or normalized text")
    for split, count in spec["split_counts"].items():
        expected = Counter({family: count // len(spec["families"]) for family in spec["families"]})
        if Counter(row["family"] for row in rows if row["split"] == split) != expected:
            raise ValueError(f"changed family balance in {split}")
    new_rows = [row for row in rows if row["split"] != "calibration"]
    scenarios = [row.get("scenario_id") for row in new_rows]
    if any(not isinstance(value, str) or not value.strip() for value in scenarios) or len(set(scenarios)) != len(scenarios):
        raise ValueError("overlapping or missing authored scenario IDs")


def corpus_inventory(root: Path = ROOT) -> tuple[list[dict], dict, list[str]]:
    spec = load_spec(root)
    rows = []
    for source in spec["corpora"]:
        verify_frozen_files({"frozen_files": [source]}, root)
        corpus = read_suite(root / source["path"])
        if len(corpus) != source["rows"]:
            raise ValueError("changed corpus row count")
        for item in corpus:
            split = item.get("split") if source["split"] == "row" else source["split"]
            rows.append({"prompt_id": item["prompt_id"], "family": item["family"], "text": item["text"],
                         "split": split, "source": source["path"],
                         "scenario_id": item.get("scenario_id", "historical:" + item["prompt_id"])})
    validate_splits(rows, spec)
    selected = {source["path"] for source in spec["corpora"]}
    other_paths = sorted(str(path.relative_to(root)) for path in (root / "data").glob("*.jsonl")
                         if str(path.relative_to(root)) not in selected)
    historical = [row for row in rows if row["split"] == "calibration"]
    for name in other_paths:
        historical.extend(read_suite(root / name))
    audit = novelty_audit([row for row in rows if row["split"] != "calibration"], historical)
    return rows, audit, other_paths


def model_reference(root: Path = ROOT) -> dict:
    reference = load_spec(root)["model_reference"]
    payload = (root / reference["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError("canonical model reference changed")
    return json.loads(payload)


def model_receipts(model_path: Path, root: Path = ROOT) -> list[dict]:
    reference = model_reference(root)
    expected = {Path(r["path"]).name: r for r in reference["frozen_files"]
                if Path(r["path"]).parent == Path(reference["model_path"])}
    if {p.name for p in model_path.iterdir() if p.is_file()} != set(expected):
        raise ValueError("model asset inventory differs from canonical reference")
    records = []
    for name, original in sorted(expected.items()):
        actual = file_receipt(model_path / name)
        if any(actual[key] != original[key] for key in ("bytes", "sha256")):
            raise ValueError(f"changed model/tokenizer bytes: {name}")
        records.append(actual)
    return records


def tokenize_inventory(rows: list[dict], tokenizer, spec: dict) -> list[dict]:
    inventory = []
    for row in rows:
        ids = tokenizer.encode(row["text"], add_special_tokens=False, truncation=False)
        if any(not isinstance(t, Integral) or isinstance(t, bool) or t < 0 for t in ids):
            raise ValueError("invalid tokenizer output")
        minimum = spec["geometry"]["k"] + 3 if row["split"] == "calibration" else spec["evaluation"]["minimum_text_tokens"]
        if not minimum <= len(ids) <= spec["evaluation"]["maximum_text_tokens"]:
            raise ValueError(f"invalid token count, no truncation or replacement: {row['prompt_id']}")
        inventory.append({**row, "input_ids": [int(t) for t in ids], "token_count": len(ids)})
    return inventory


def source_paths(root: Path, spec: dict, historical_paths: list[str]) -> list[str]:
    paths = {SPEC_PATH, spec["model_reference"]["path"], "spiral_hodge.py", "pyproject.toml", "requirements.txt"}
    paths.update(["scripts/prepare_hltd_prefix_gate.py", "scripts/hltd_prefix_transfer.py"])
    paths.update(source["path"] for source in spec["corpora"])
    paths.update(historical_paths)
    paths.update(str(path.relative_to(root)) for path in (root / "scripts").glob("*.py"))
    paths.update(["tests/test_hltd_prefix_transfer.py", "tests/test_hltd_prefix_protocol.py"])
    return sorted(paths)


def local_receipts(names: list[str], root: Path) -> list[dict]:
    return [{**file_receipt(root / name), "path": name} for name in names]


def preparation_environment() -> dict:
    return {"python": sys.version,
            "packages": {name: importlib.metadata.version(name)
                         for name in ["transformers", "tokenizers", "numpy", "scipy", "scikit-learn"]}}


def validate_preparation(protocol: dict, tokenizer, root: Path = ROOT) -> None:
    expected_keys = {"schema_version", "status", "execution_allowed", "model_parameters_loaded", "atlas", "results",
                     "prepared_utc", "spec", "spec_sha256", "model_path", "prompts", "novelty_audit",
                     "frozen_files", "preparation_environment", "planned_primary_cells", "planned_treatment_rows"}
    if set(protocol) != expected_keys:
        raise ValueError("changed preparation keys")
    spec = load_spec(root)
    for key, value in {"schema_version": 1, "spec": spec, "spec_sha256": SPEC_SHA256,
                       "status": "PREPARED_NOT_EXECUTED", "execution_allowed": False,
                       "model_parameters_loaded": False, "atlas": None, "results": None,
                       "planned_primary_cells": 60, "planned_treatment_rows": 5760}.items():
        if protocol[key] != value:
            raise ValueError(f"changed preparation contract: {key}")
    if protocol["preparation_environment"] != preparation_environment():
        raise ValueError("changed preparation environment")
    rows, novelty, historical_paths = corpus_inventory(root)
    if protocol["prompts"] != tokenize_inventory(rows, tokenizer, spec):
        raise ValueError("changed frozen text/token/split inventory")
    if protocol["novelty_audit"] != novelty:
        raise ValueError("changed novelty audit")
    model_path = Path(protocol["model_path"])
    if not model_path.is_absolute():
        raise ValueError("model path must be absolute")
    expected = local_receipts(source_paths(root, spec, historical_paths), root) + model_receipts(model_path, root)
    if protocol["frozen_files"] != expected:
        raise ValueError("changed preparation source/model receipt inventory")
    verify_frozen_files(protocol, root)


def load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)


def prepare(output: Path, *, model_path: Path | None = None, root: Path = ROOT) -> dict:
    if output.exists():
        raise FileExistsError(output)
    spec = load_spec(root)
    rows, novelty, historical_paths = corpus_inventory(root)
    model_path = (model_path or Path(model_reference(root)["model_path"])).expanduser().absolute()
    models = model_receipts(model_path, root)
    tokenizer = load_tokenizer(model_path)
    inventory = tokenize_inventory(rows, tokenizer, spec)
    protocol = {
        "schema_version": 1, "status": "PREPARED_NOT_EXECUTED", "execution_allowed": False,
        "model_parameters_loaded": False, "atlas": None, "results": None, "prepared_utc": utc_now(),
        "spec": spec, "spec_sha256": SPEC_SHA256, "model_path": str(model_path),
        "prompts": inventory, "novelty_audit": novelty,
        "frozen_files": local_receipts(source_paths(root, spec, historical_paths), root) + models,
        "preparation_environment": preparation_environment(),
        "planned_primary_cells": len([r for r in rows if r["split"] == "evaluation"]) * len(spec["evaluation"]["prefix_lengths"]),
        "planned_treatment_rows": spec["evaluation"]["planned_treatment_rows"],
    }
    validate_preparation(protocol, tokenizer, root)
    save_json(output, protocol)
    return protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path, metavar="OUTPUT_JSON")
    action.add_argument("--validate", type=Path, metavar="PREPARATION_JSON")
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args(argv)
    if args.prepare:
        protocol = prepare(args.prepare.absolute(), model_path=args.model_path)
    else:
        if args.model_path is not None:
            parser.error("validation does not accept model overrides")
        protocol = json.loads(args.validate.read_text())
        model_receipts(Path(protocol["model_path"]), ROOT)
        validate_preparation(protocol, load_tokenizer(Path(protocol["model_path"])))
    print(json.dumps({"status": protocol["status"], "execution_allowed": protocol["execution_allowed"],
                      "split_counts": dict(Counter(row["split"] for row in protocol["prompts"])),
                      "planned_primary_cells": protocol["planned_primary_cells"],
                      "planned_treatment_rows": protocol["planned_treatment_rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
