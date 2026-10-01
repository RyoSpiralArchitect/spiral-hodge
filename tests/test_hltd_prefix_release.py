import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import verify_hltd_prefix_release as release


def make_files(directory):
    directory.mkdir()
    expected = {name: directory / name for name in ("verdict.json", "bootstrap_indices.csv")}
    for path in expected.values():
        path.write_text("recorded")
    return expected, [release.receipt(path) for path in expected.values()]


def test_bound_paths_are_exactly_the_files_verified(tmp_path):
    expected, records = make_files(tmp_path / "result")
    assert release.bind_files(records, expected) == expected


@pytest.mark.parametrize("change", ["redirect", "missing", "duplicate", "extra", "bytes", "content"])
def test_changed_or_unbound_consumers_are_rejected(tmp_path, change):
    expected, records = make_files(tmp_path / "result")
    if change in {"redirect", "extra"}:
        alternate, alt_records = make_files(tmp_path / "alternate")
        if change == "redirect":
            # A valid receipt elsewhere must not authorize consuming a local copy.
            records[0] = alt_records[0]
            expected["verdict.json"].write_text("unverified local replacement")
        else:
            records.append(release.receipt(alternate["verdict.json"]))
    elif change == "missing":
        records.pop()
    elif change == "duplicate":
        records.append(records[0])
    elif change == "bytes":
        records[0]["bytes"] += 1
    else:
        expected["verdict.json"].write_text("changed")
    with pytest.raises(ValueError):
        release.bind_files(records, expected)


def test_unrecorded_chart_input_is_rejected_even_with_valid_other_receipts(tmp_path):
    expected, records = make_files(tmp_path / "stage")
    missing = tmp_path / "stage/draws.csv"
    missing.write_text("unverified")
    with pytest.raises(ValueError, match="incomplete"):
        release.bind_files(records, {**expected, "draws.csv": missing})


def test_candidate_manifest_cannot_redefine_its_own_trust(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"outputs": []}))
    sha = release.receipt(path)["sha256"]
    assert release.pinned_json(path, sha) == {"outputs": []}
    path.write_text(json.dumps({"outputs": [{"path": "different"}]}))
    with pytest.raises(ValueError, match="canonical"):
        release.pinned_json(path, sha)


def test_binding_failure_precedes_historical_arithmetic(monkeypatch):
    def reject(root):
        raise ValueError("unbound inputs")
    monkeypatch.setattr(release, "bind_release", reject)
    with patch("scripts.audit_hltd_prefix_transfer.audit") as audit:
        with pytest.raises(ValueError, match="unbound"):
            release.verify()
        audit.assert_not_called()


def test_all_consumed_stage_outputs_are_in_exact_inventories():
    assert {"verdict.json", "bootstrap_indices.csv"} <= set(release.OUTPUTS["v2"][1])
    assert {"summary.json", "nodes.csv", "controls.csv", "prompts.csv", "position_mix.csv"} <= set(release.OUTPUTS["calibration"][1])
    assert {"draws.csv", "summary.json"} <= set(release.OUTPUTS["null_calibration"][1])
    assert "dose_summary.csv" in release.OUTPUTS["null_v2"][1]
    assert "raw_treatments.csv" in release.RAW["v2"][1]
    assert all(Path(path).name == "manifest.json" for path, _ in release.MANIFESTS.values())
