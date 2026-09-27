from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from scripts import analyze_hltd_precision_full_gate as full_analysis
from scripts import analyze_hltd_precision_gate as pilot_analysis
from scripts import run_hltd_l8_gate as l8
from scripts import run_hltd_fresh_gate as fresh
from scripts import run_hltd_precision_full_gate as full_runner
from scripts import run_hltd_precision_gate as pilot
from scripts.evaluate_hltd_signed_layer_gate import file_receipt, verify_frozen_files
from scripts.hltd_historical_sources import SNAPSHOT, historical_source_path, verify_historical_inputs

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def recorded_pilot() -> dict:
    return json.loads((ROOT / pilot.PILOT_REFERENCE).read_text())


def copy_precision_references(root: Path) -> None:
    for name in [pilot.PILOT_REFERENCE, full_runner.FULL_REFERENCE, "docs/data/hltd_signed_l7_position/protocol.json"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())


@pytest.mark.parametrize("change", [
    "prompts", "seeds", "alphas", "arms", "tolerances", "historical_raw",
    "bins", "metric", "suite", "model_path", "inventory", "input_hash",
    "extra_key", "parent_root", "absolute_root", "nested_root",
])
def test_pilot_contract_changes_stop_before_output_or_model(tmp_path: Path, change: str) -> None:
    protocol = recorded_pilot()
    protocol["run_root"] = "spiral_out_test_review"
    if change == "prompts":
        protocol["prompts"] = protocol["prompts"][:-1]
    elif change in {"seeds", "alphas"}:
        protocol["design"][change] = protocol["design"][change][:-1]
    elif change == "arms":
        protocol["arms"].pop("fp16_replay")
    elif change == "tolerances":
        protocol["tolerances"]["early_coefficient_max_abs_change"] = 1
    elif change in {"historical_raw", "suite", "model_path"}:
        protocol[change] = "changed-input"
    elif change in {"bins", "metric"}:
        protocol["primary"][change] = [4, 5, 6, 7] if change == "bins" else "semantic_margin_delta"
    elif change == "inventory":
        protocol["frozen_files"] = []
    elif change == "input_hash":
        record = next(r for r in protocol["frozen_files"] if r["path"] == protocol["suite"])
        record["sha256"] = "0" * 64
    elif change == "extra_key":
        protocol["alternate_reference"] = "untrusted.json"
    else:
        protocol["run_root"] = {"parent_root": "../spiral_out_escape", "absolute_root": str(tmp_path / "escape"),
                                "nested_root": "spiral_out_parent/child"}[change]
    reference_path = tmp_path / pilot.PILOT_REFERENCE
    reference_path.parent.mkdir(parents=True)
    reference_path.write_bytes((ROOT / pilot.PILOT_REFERENCE).read_bytes())
    path = tmp_path / "candidate.json"
    write_json(path, protocol)
    with patch.object(pilot, "ROOT", tmp_path), patch.object(pilot, "verify_frozen_files") as verify, \
            patch.object(pilot, "_run_pilot") as execute:
        with pytest.raises(ValueError):
            pilot.run(path)
        verify.assert_not_called()
        execute.assert_not_called()
    assert not list(tmp_path.glob("spiral_out_*"))


def test_pilot_accepts_same_contract_and_rejects_rewritten_reference(tmp_path: Path) -> None:
    protocol = recorded_pilot()
    pilot.validate_pilot_protocol(protocol)
    protocol["run_root"] = "spiral_out_prospective_review"
    pilot.validate_pilot_protocol(protocol)
    changed = copy.deepcopy(protocol)
    changed["design"]["seeds"] = [0]
    write_json(tmp_path / pilot.PILOT_REFERENCE, changed)
    with pytest.raises(ValueError, match="canonical pilot protocol changed"):
        pilot.validate_pilot_protocol(changed, tmp_path)


@pytest.mark.parametrize("runner", [pilot, full_runner], ids=["pilot", "full"])
def test_precision_runners_reject_device_fallback_before_model_work(tmp_path: Path, runner) -> None:
    copy_precision_references(tmp_path)
    reference = (ROOT / pilot.PILOT_REFERENCE).read_bytes()
    protocol = (json.loads(reference) if runner is pilot else
                json.loads((ROOT / "docs/data/hltd_precision_l7_full/protocol.json").read_text()))
    protocol["run_root"] = "spiral_out_fallback_test"
    path = tmp_path / "candidate.json"
    write_json(path, protocol)
    with patch.object(runner, "ROOT", tmp_path), patch.object(runner, "verify_frozen_files"), \
            patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}), \
            patch.object(pilot.fast, "_load_model_and_tokenizer") as load:
        with pytest.raises(ValueError, match="MPS device fallback must be disabled"):
            runner.run(path)
        load.assert_not_called()
    output = tmp_path / protocol["run_root"]
    receipt = json.loads((output / "execution_receipt.json").read_text())
    assert receipt["status"] == "FAILED"
    assert "fallback" in receipt["error"]
    assert not (output / "load_audit.json").exists()


@pytest.mark.parametrize("change", ["pilot_protocol", "reference_protocol", "coupled_companions",
    "canonical_full", "model_path", "tolerances", "seeds", "alphas", "arms", "historical_raw",
    "prompts", "primary", "inventory", "input_hash", "absolute_root", "parent_root"])
def test_full_contract_cannot_select_edited_companions(tmp_path: Path, change: str) -> None:
    copy_precision_references(tmp_path)
    protocol = json.loads((ROOT / full_runner.FULL_REFERENCE).read_text())
    protocol["run_root"] = "spiral_out_full_review"
    if change in {"pilot_protocol", "reference_protocol", "coupled_companions"}:
        companion = recorded_pilot()
        previous = json.loads((ROOT / protocol["reference_protocol"]).read_text())
        if change == "coupled_companions":
            companion["design"]["seeds"] = [0]
            previous["prompts"][0]["prompt_id"] = "substituted_prompt"
            protocol["design"] = copy.deepcopy(companion["design"])
            protocol["prompts"] = copy.deepcopy(previous["prompts"])
        if change != "reference_protocol":
            write_json(tmp_path / "edited/pilot.json", companion)
            protocol["pilot_protocol"] = "edited/pilot.json"
        if change != "pilot_protocol":
            write_json(tmp_path / "edited/previous.json", previous)
            protocol["reference_protocol"] = "edited/previous.json"
    elif change in {"canonical_full", "model_path", "historical_raw"}:
        protocol["model_path" if change == "canonical_full" else change] = "changed-source"
        if change == "canonical_full":
            write_json(tmp_path / full_runner.FULL_REFERENCE, protocol)
    elif change == "tolerances":
        protocol["tolerances"]["early_coefficient_max_abs_change"] = 1
    elif change in {"seeds", "alphas"}:
        protocol["design"][change] = protocol["design"][change][:-1]
    elif change == "arms":
        protocol["arms"].pop("fp16_replay")
    elif change == "prompts":
        protocol["prompts"] = protocol["prompts"][:-1]
    elif change == "primary":
        protocol["primary"]["metric"] = "semantic_margin_delta"
    elif change == "inventory":
        protocol["frozen_files"] = []
    elif change == "input_hash":
        record = next(r for r in protocol["frozen_files"] if r["path"] == protocol["pilot_protocol"])
        record["sha256"] = "0" * 64
    else:
        protocol["run_root"] = str(tmp_path / "escape") if change == "absolute_root" else "../escape"
    path = tmp_path / "candidate.json"
    write_json(path, protocol)
    with patch.object(full_runner, "ROOT", tmp_path), patch.object(full_runner, "verify_frozen_files") as verify, \
            patch.object(pilot, "_run_pilot") as execute:
        with pytest.raises(ValueError):
            full_runner.run(path)
        verify.assert_not_called()
        execute.assert_not_called()
    assert not list(tmp_path.glob("spiral_out_*"))


@pytest.mark.parametrize("runner,protocol_file", [
    (l8, "docs/data/hltd_signed_l8_position/protocol.json"),
    (fresh, "docs/data/hltd_fresh_l7_l8/protocol.json"),
], ids=["l8", "fresh"])
@pytest.mark.parametrize("change", ["model_path", "missing_weight", "weight_hash", "tokenizer_hash", "no_receipts"])
def test_selected_model_must_be_bound_to_reference_receipts(tmp_path: Path, runner, protocol_file: str, change: str) -> None:
    protocol = json.loads((ROOT / protocol_file).read_text())
    protocol["run_root"] = "spiral_out_model_binding_test"
    if change == "model_path":
        protocol["model_path"] = str(tmp_path / "other_model")
    elif change == "no_receipts":
        protocol["frozen_files"] = []
    else:
        name = "tokenizer_config.json" if change == "tokenizer_hash" else "model.safetensors"
        selected = str(Path(protocol["model_path"]) / name)
        if change == "missing_weight":
            protocol["frozen_files"] = [r for r in protocol["frozen_files"] if r["path"] != selected]
        else:
            next(r for r in protocol["frozen_files"] if r["path"] == selected)["sha256"] = "0" * 64
    path = tmp_path / "candidate.json"
    write_json(path, protocol)
    with patch.object(runner, "ROOT", tmp_path), patch.object(runner, "verify_frozen_files"), \
            patch.object(runner, "_run") as execute:
        with pytest.raises(ValueError):
            runner.run(path)
        execute.assert_not_called()
    assert not (tmp_path / protocol["run_root"]).exists()


def test_fresh_gate_keeps_the_reference_runtime() -> None:
    protocol = json.loads((ROOT / "docs/data/hltd_fresh_l7_l8/protocol.json").read_text())
    protocol["runtime"]["packages"]["torch"] = "different"
    with pytest.raises(ValueError, match="runtime"):
        fresh.validate_protocol(protocol)


def precision_fixture(root: Path, activity: str) -> tuple[Path, dict]:
    protocol = {
        "run_root": "spiral_out_coverage_test",
        "design": {"seeds": [0, 1], "components": ["coexact", "random_tangent"],
                   "alphas": [-1, -.5, -.25, .25, .5, 1], "row_constants": {"layer": 7}},
        "prompts": [{"prompt_id": "p", "family": "literal", "token_count": 25}],
        "primary": {"metric": "next_token_logprob_delta", "contrast_type": "odd", "bins": [0, 1, 2, 3]},
        "analysis": {"bins": 12, "bootstrap_samples": 10, "bootstrap_seed": 0},
        "pilot_prompt_ids": ["p"], "tolerances": {"early_coefficient_max_abs_change": .05},
    }
    path = root / "protocol.json"
    write_json(path, protocol)
    output = root / protocol["run_root"]
    write_json(output / "execution_receipt.json", {"status": "COMPLETED", "protocol": file_receipt(path)})
    for name in ["zero_calibration", "replay_audit", "fixed_field_audit"]:
        write_json(output / f"{name}.json", {"passed": True})
    write_json(output / "load_audit.json", {"models": {dtype: {
        "all_bytes_equal": True, "parameter_tensors": 148, "parameter_count": 124439808,
    } for dtype in ["float16", "float32"]}})
    for arm, (model_dtype, field_dtype) in pilot.ARMS.items():
        records = []
        for token in range(1, 24):
            active = 1
            if arm == "fp32_rebuilt_field":
                active = int(activity == "complete" or (activity == "sparse" and token == 1))
            for seed in protocol["design"]["seeds"]:
                for component in protocol["design"]["components"]:
                    for alpha in protocol["design"]["alphas"]:
                        applied = active if component == "coexact" else 1
                        delta = alpha * applied
                        records.append({"family": "literal", "prompt_id": "p", "token_index": token,
                            "node_index": token - 1, "token_count": 25, "seed": seed, "component": component,
                            "alpha": alpha, "layer": 7, "component_active": applied,
                            "next_token_logprob_base": -5., "next_token_logprob_steered": -5. + delta,
                            "next_token_logprob_delta": delta, "semantic_margin_delta": delta / 2,
                            "precision_arm": arm, "model_dtype": model_dtype, "field_dtype": field_dtype,
                            "baseline_mode": "matched_batch", "baseline_batch_size": 12})
        write_json(output / arm / "delta_trace.json", [{"same": "fixed-field trace"}])
        pd.DataFrame(records).to_csv(output / arm / "summary.csv", index=False)
    return path, protocol


@pytest.mark.parametrize("analyzer", [pilot_analysis, full_analysis], ids=["pilot", "full"])
@pytest.mark.parametrize("activity", ["inactive", "sparse"])
def test_analyzer_saves_insufficient_before_any_fit(tmp_path: Path, analyzer, activity: str) -> None:
    path, protocol = precision_fixture(tmp_path, activity)
    with patch.object(analyzer, "ROOT", tmp_path), patch.object(analyzer, "verify_frozen_files"), \
            patch.object(full_analysis, "validate_full_protocol"), patch.object(analyzer, "render_all") as render:
        if analyzer is pilot_analysis:
            assert analyzer.main(["--protocol", str(path)]) == 0
        else:
            assert analyzer.analyze(path)["status"] == "INSUFFICIENT_COVERAGE"
        render.assert_not_called()
    output = tmp_path / protocol["run_root"] / "analysis"
    verdict = json.loads((output / "precision_verdict.json").read_text())
    assert verdict["status"] == "INSUFFICIENT_COVERAGE"
    assert not verdict["complete_early_coverage"]
    assert verdict["active_tokens"]["fp32_rebuilt_field"] == (0 if activity == "inactive" else 1)
    assert verdict["active_tokens"]["fp16_replay"] == 23
    assert {row["arm"] for row in verdict["missing_early_prompt_arms"]} == {"fp32_rebuilt_field"}
    coverage = pd.read_csv(output / "primary_early_coverage.csv")
    assert len(coverage) == 12
    assert set(coverage.precision_arm) == set(pilot.ARMS)
    assert len(pd.read_csv(output / "activity_comparison.csv")) == 23
    assert not list(output.glob("*.png"))
    assert not list(output.glob("*/summary_prompt_bin_response_coefficients.csv"))


def test_complete_coverage_keeps_existing_analysis_path(tmp_path: Path) -> None:
    path, protocol = precision_fixture(tmp_path, "complete")
    root = tmp_path / protocol["run_root"]
    frames = {arm: pd.read_csv(root / arm / "summary.csv") for arm in pilot.ARMS}
    assert pilot_analysis.save_insufficient_coverage(frames, protocol, path, root / "analysis") is None
    assert not (root / "analysis").exists()


def l8_freeze_fixture(root: Path) -> tuple[Path, list[Path]]:
    reference = json.loads((ROOT / l8.REFERENCE).read_text())
    model = root / "model"
    model.mkdir()
    reference_model = root / "reference_model"
    reference_model.mkdir()
    model_receipts = []
    for name in sorted(l8.MODEL_FILES):
        payload = f"synthetic {name}, never loaded".encode()
        (model / name).write_bytes(payload)
        (reference_model / name).write_bytes(payload)
        model_receipts.append(file_receipt(reference_model / name))
    frozen = root / "scripts/reference.py"
    frozen.parent.mkdir()
    frozen.write_text("# fixed historical source\n")
    reference["model_path"] = str(reference_model)
    reference["frozen_files"] = [*model_receipts, {**file_receipt(frozen), "path": "scripts/reference.py"}]
    reference["run_root"] = "spiral_out_reference"
    reference_path = root / l8.REFERENCE
    write_json(reference_path, reference)
    required = [root / reference["run_root"] / "fp32_rebuilt_field/summary.csv",
                root / "docs/data/hltd_precision_l7_full/fp32_rebuilt_field__summary_prompt_bin_response_coefficients.csv"]
    required.extend(root / p for p in ["scripts/run_hltd_l8_gate.py", "scripts/analyze_hltd_l8_gate.py",
        "scripts/audit_hltd_precision_raw.py", "scripts/run_hltd_precision_full_gate.py", "tests/test_hltd_l8_gate.py",
        "scripts/hltd_historical_sources.py"])
    required.extend(root / "transformers" / p for p in ["modeling_utils.py", "modeling_attn_mask_utils.py",
        "masking_utils.py", "integrations/sdpa_attention.py", "models/gpt2/modeling_gpt2.py"])
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    return model, required


@pytest.mark.parametrize("missing", [0, 1, 2, -1], ids=["raw", "coefficients", "source", "third-party"])
def test_l8_freeze_rejects_missing_required_file_without_protocol(tmp_path: Path, missing: int) -> None:
    model, required = l8_freeze_fixture(tmp_path)
    required[missing].unlink()
    path = tmp_path / "new/protocol.json"
    with patch.object(l8, "ROOT", tmp_path), patch.object(l8, "runtime_snapshot", return_value={}), \
            patch.dict("sys.modules", {"transformers": SimpleNamespace(__file__=str(tmp_path / "transformers/__init__.py"))}):
        with pytest.raises(FileNotFoundError) as error:
            l8.freeze(path, model, "spiral_out_new")
    assert str(required[missing]) in str(error.value)
    assert not path.parent.exists()
    assert not (tmp_path / "spiral_out_new").exists()


@pytest.mark.parametrize("remove_original", ["never", "before_freeze", "after_freeze"])
def test_l8_freeze_records_all_required_files_and_current_source(tmp_path: Path, remove_original: str) -> None:
    model, required = l8_freeze_fixture(tmp_path)
    original_model = tmp_path / "reference_model"
    if remove_original == "before_freeze":
        for source in original_model.iterdir():
            source.unlink()
        original_model.rmdir()
    historical = tmp_path / "scripts/reference.py"
    archived = tmp_path / SNAPSHOT / "scripts/reference.py.txt"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(historical.read_bytes())
    historical.write_text("# current patched source\n")
    original_validator = l8.validate_protocol
    with patch.object(l8, "ROOT", tmp_path), patch.object(l8, "runtime_snapshot", return_value={}), \
            patch.object(l8, "validate_protocol", side_effect=lambda p: original_validator(p, tmp_path)), \
            patch.dict("sys.modules", {"transformers": SimpleNamespace(__file__=str(tmp_path / "transformers/__init__.py"))}):
        protocol = l8.freeze(tmp_path / "new/protocol.json", model, "spiral_out_new")
    recorded_paths = {tmp_path / record["path"] for record in protocol["frozen_files"]}
    assert set(required) <= recorded_paths
    assert {model / name for name in l8.MODEL_FILES} <= recorded_paths
    assert not any(path.is_relative_to(original_model) for path in recorded_paths)
    assert protocol["prior_input_audit"]["verified_relocated_model_files"] == len(l8.MODEL_FILES)
    assert protocol["prior_input_audit"]["source_snapshots"][0]["snapshot"] == str(archived.relative_to(tmp_path))
    record = next(r for r in protocol["frozen_files"] if r["path"] == "scripts/reference.py")
    assert record["sha256"] == file_receipt(historical)["sha256"]
    if remove_original == "after_freeze":
        for source in original_model.iterdir():
            source.unlink()
        original_model.rmdir()
    verify_frozen_files(protocol, tmp_path)


def test_relocation_does_not_skip_unrelated_historical_inputs(tmp_path: Path) -> None:
    model, _ = l8_freeze_fixture(tmp_path)
    missing = tmp_path / "historical.csv"
    missing.write_text("original data\n")
    reference_path = tmp_path / l8.REFERENCE
    reference = json.loads(reference_path.read_text())
    reference["frozen_files"].append(file_receipt(missing))
    write_json(reference_path, reference)
    missing.unlink()
    with patch.object(l8, "ROOT", tmp_path), patch.object(l8, "runtime_snapshot", return_value={}):
        with pytest.raises(ValueError, match="not repository-local source"):
            l8.freeze(tmp_path / "new/protocol.json", model, "spiral_out_new")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("missing", sorted(l8.MODEL_FILES))
def test_l8_freeze_rejects_partial_model_cache(tmp_path: Path, missing: str) -> None:
    model, _ = l8_freeze_fixture(tmp_path)
    (model / missing).unlink()
    path = tmp_path / "new/protocol.json"
    with patch.object(l8, "ROOT", tmp_path), patch.object(l8, "runtime_snapshot") as runtime, \
            patch.object(l8.precision.fast, "_load_model_and_tokenizer") as load:
        with pytest.raises(FileNotFoundError) as error:
            l8.freeze(path, model, "spiral_out_new")
        runtime.assert_not_called()
        load.assert_not_called()
    assert str(model / missing) in str(error.value)
    assert not path.parent.exists()
    assert not (tmp_path / "spiral_out_new").exists()


def test_l8_freeze_rejects_changed_relocated_model_config(tmp_path: Path) -> None:
    model, _ = l8_freeze_fixture(tmp_path)
    (model / "config.json").write_text("different model configuration")
    with patch.object(l8, "ROOT", tmp_path):
        with pytest.raises(ValueError, match="relocated model input differs"):
            l8.freeze(tmp_path / "new/protocol.json", model, "spiral_out_new")
    assert not (tmp_path / "new").exists()


def test_historical_source_snapshot_does_not_weaken_runtime_receipts(tmp_path: Path) -> None:
    source = tmp_path / "scripts/source.py"
    source.parent.mkdir()
    source.write_text("original source\n")
    record = {**file_receipt(source), "path": "scripts/source.py"}
    snapshot = tmp_path / SNAPSHOT / (record["path"] + ".txt")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes())
    source.write_text("patched source\n")
    protocol = {"frozen_files": [record]}
    assert historical_source_path(record, tmp_path) == snapshot
    assert verify_historical_inputs(protocol, tmp_path)[0]["snapshot"] == str(snapshot.relative_to(tmp_path))
    with pytest.raises(ValueError, match="frozen file changed"):
        verify_frozen_files(protocol, tmp_path)
    snapshot.write_text("corrupted archive\n")
    with pytest.raises(ValueError, match="historical source"):
        historical_source_path(record, tmp_path)


@pytest.mark.parametrize("name", ["data/summary.csv", "docs/protocol.json", "../escape.py", "/external/source.py"])
def test_historical_fallback_never_substitutes_data_or_external_sources(tmp_path: Path, name: str) -> None:
    record = {"path": name, "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    with pytest.raises(ValueError, match="not repository-local source"):
        historical_source_path(record, tmp_path)


def test_preserved_source_manifest_is_byte_exact() -> None:
    manifest = json.loads((ROOT / SNAPSHOT / "manifest.json").read_text())
    assert manifest["source_commit"] == SNAPSHOT.name
    assert len(manifest["files"]) == 12
    for record in manifest["files"]:
        actual = file_receipt(ROOT / SNAPSHOT / (record["path"] + ".txt"))
        assert actual["sha256"] == record["sha256"]
        assert actual["bytes"] == record["bytes"]
