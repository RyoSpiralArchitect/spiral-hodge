from __future__ import annotations

import copy
import json
import shutil
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import prepare_hltd_prefix_gate as prepare
from scripts.evaluate_hltd_signed_layer_gate import file_receipt

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens, truncation):
        assert not add_special_tokens and not truncation
        return [sum(word.encode()) for word in text.split()]


@pytest.fixture
def prepared(tmp_path):
    spec = prepare.load_spec()
    _, _, historical = prepare.corpus_inventory()
    for name in prepare.source_paths(ROOT, spec, historical):
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    model = tmp_path / "model"
    model.mkdir()
    reference = prepare.model_reference()
    originals = [r for r in reference["frozen_files"] if Path(r["path"]).parent == Path(reference["model_path"])]
    synthetic_reference = {"model_path": str(model), "frozen_files": []}
    for record in originals:
        path = model / Path(record["path"]).name
        path.write_text("synthetic bytes, never loaded: " + path.name)
        synthetic_reference["frozen_files"].append(file_receipt(path))
    output = tmp_path / "planned/preparation.json"
    # Exercise real inventory/byte checks using small model-asset fixtures.
    with patch.object(prepare, "model_reference", return_value=synthetic_reference), \
            patch.object(prepare, "load_tokenizer", return_value=FakeTokenizer()):
        protocol = prepare.prepare(output, model_path=model, root=tmp_path)
        yield tmp_path, protocol, output, model


def test_specification_and_actual_splits_are_fixed():
    spec = prepare.load_spec()
    rows, audit, _ = prepare.corpus_inventory()
    assert Counter(row["split"] for row in rows) == {"calibration": 40, "pilot": 4, "evaluation": 20}
    assert spec["evaluation"]["prefix_lengths"] == [8, 16, 24]
    assert spec["evaluation"]["planned_primary_cells"] == 20 * 3
    assert spec["evaluation"]["planned_treatment_rows"] == 20 * 3 * 8 * 2 * 6
    assert not spec["evaluation"]["pilot_nonzero_treatments"]
    assert not spec["execution_policy"]["execution_runner_available_at_preparation"]
    assert audit["exact_duplicate_count"] == 0
    assert audit["fresh_rows"] == 24
    assert audit["historical_rows"] == 69


@pytest.mark.parametrize("change", ["count", "family", "id", "text", "scenario"])
def test_split_mutations_are_rejected(change):
    rows, _, _ = prepare.corpus_inventory()
    new = [row for row in rows if row["split"] != "calibration"]
    if change == "count":
        new[0]["split"] = "evaluation"
    elif change == "family":
        new[0]["family"] = "metaphor_shift"
    elif change == "id":
        new[0]["prompt_id"] = rows[0]["prompt_id"]
    elif change == "text":
        new[0]["text"] = "  " + rows[0]["text"].upper() + "  "
    else:
        new[-1]["scenario_id"] = new[0]["scenario_id"]
    with pytest.raises(ValueError):
        prepare.validate_splits(rows, prepare.load_spec())


def test_preparation_binds_sources_and_tokens_without_execution(prepared):
    root, protocol, output, _ = prepared
    assert json.loads(output.read_text()) == protocol
    prepare.validate_preparation(protocol, FakeTokenizer(), root)
    assert protocol["status"] == "PREPARED_NOT_EXECUTED"
    assert protocol["execution_allowed"] is False
    assert protocol["model_parameters_loaded"] is False
    assert protocol["atlas"] is None and protocol["results"] is None
    assert len(protocol["prompts"]) == 64
    assert not list(root.glob("spiral_out_*"))
    names = {r["path"] for r in protocol["frozen_files"]}
    assert {prepare.SPEC_PATH, "scripts/prepare_hltd_prefix_gate.py", "scripts/hltd_prefix_transfer.py"} <= names


@pytest.mark.parametrize("change", ["spec", "execution", "status", "result", "atlas", "tokens", "split", "text",
                                    "inventory", "receipt", "environment", "extra_key", "planned_rows"])
def test_preparation_cannot_be_relabelled_or_change_inputs(prepared, change):
    root, original, _, _ = prepared
    protocol = copy.deepcopy(original)
    if change == "spec":
        protocol["spec"]["geometry"]["layer"] = 8
    elif change == "execution":
        protocol["execution_allowed"] = True
    elif change == "status":
        protocol["status"] = "COMPLETED"
    elif change in {"result", "atlas"}:
        protocol["results" if change == "result" else "atlas"] = {"passed": True}
    elif change == "tokens":
        protocol["prompts"][-1]["input_ids"][0] += 1
    elif change in {"split", "text"}:
        protocol["prompts"][-1][change] = "replacement"
    elif change == "inventory":
        protocol["frozen_files"] = []
    elif change == "receipt":
        protocol["frozen_files"][0]["sha256"] = "0" * 64
    elif change == "environment":
        protocol["preparation_environment"]["packages"]["numpy"] = "unrecorded"
    elif change == "extra_key":
        protocol["alternative_spec"] = "edited.json"
    else:
        protocol["planned_treatment_rows"] = 100000
    with pytest.raises(ValueError):
        prepare.validate_preparation(protocol, FakeTokenizer(), root)


@pytest.mark.parametrize("source", ["spec", "corpus", "code", "model"])
def test_changed_live_bytes_invalidate_preparation(prepared, source):
    root, protocol, _, model = prepared
    path = {"spec": root / prepare.SPEC_PATH,
            "corpus": root / "data/hltd_prefix_transfer_prompt_suite.jsonl",
            "code": root / "scripts/hltd_prefix_transfer.py", "model": model / "tokenizer.json"}[source]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        prepare.validate_preparation(protocol, FakeTokenizer(), root)


@pytest.mark.parametrize("change", ["missing", "extra", "changed"])
def test_model_inventory_failure_precedes_tokenization_and_output(prepared, change):
    root, _, _, model = prepared
    if change == "missing":
        (model / "vocab.json").unlink()
    elif change == "extra":
        (model / "added_tokens.json").write_text("{}")
    else:
        (model / "config.json").write_text("{}")
    output = root / "rejected/protocol.json"
    with patch.object(prepare, "load_tokenizer") as tokenizer:
        with pytest.raises(ValueError):
            prepare.prepare(output, model_path=model, root=root)
        tokenizer.assert_not_called()
    assert not output.parent.exists()


def test_output_is_not_overwritten_or_retried(prepared):
    root, _, output, model = prepared
    before = output.read_bytes()
    with patch.object(prepare, "load_tokenizer") as tokenizer:
        with pytest.raises(FileExistsError):
            prepare.prepare(output, model_path=model, root=root)
        tokenizer.assert_not_called()
    assert output.read_bytes() == before


@pytest.mark.parametrize("count", [24, 97])
def test_evaluation_token_counts_do_not_trigger_truncation_or_replacement(count):
    rows, _, _ = prepare.corpus_inventory()
    row = next(r for r in rows if r["split"] == "evaluation")
    class InvalidLengthTokenizer:
        def encode(self, text, *, add_special_tokens, truncation):
            assert not truncation
            return list(range(count))
    with pytest.raises(ValueError, match="no truncation or replacement"):
        prepare.tokenize_inventory([row], InvalidLengthTokenizer(), prepare.load_spec())
