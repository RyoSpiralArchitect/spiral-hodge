#!/usr/bin/env python3
"""Receipt-bound successor for the frozen prefix auditors and chart exporter.

The historical tools remain byte-identical for old protocol replay. Use this
entry point for current verification; it never rewrites an experimental run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFESTS = {
    "calibration": ("docs/data/hltd_prefix_calibration_readout/result/manifest.json", "0fe5745e8d501ea2437d540eacb68ab4d2b4b41d201fa2fb8d4907eb3f83c916"),
    "v1": ("docs/data/hltd_prefix_transfer/result/manifest.json", "97dc8a81d43055b005c0e27a2bf17d93a9329fa3945ecebed27227efddad17be"),
    "v2": ("docs/data/hltd_prefix_transfer_v2/result/manifest.json", "98f68c894a4f7a7cea37c1bcc4cbc1bc14b98cbb290da7d76b60acfad48a8819"),
    "null_calibration": ("docs/data/hltd_prefix_null_comparison/calibration/manifest.json", "839e2c4535651abb385f8a6c166b192a76cf0aac2c2e35260c5a88d4f5441cbf"),
    "null_v2": ("docs/data/hltd_prefix_null_comparison/v2/manifest.json", "1dda9e0002aa85cd430bc8a7ab2ee0aaab5c90a78d21746c1ac0d887401c32e9"),
    "charts": ("docs/data/hltd_prefix_null_comparison/charts/manifest.json", "b1639f0b53e498abc7396b8e54cb993aa419a2a87be84edd858ff046f2584651"),
}
OUTPUTS = {
    "calibration": ("outputs", ("controls.csv", "nodes.csv", "position_mix.csv", "prompts.csv", "replay_arrays.npz", "summary.json")),
    "v1": ("derived_files", ("support.csv", "verdict.json")),
    "v2": ("derived_files", ("bootstrap_indices.csv", "coefficients.csv", "support.csv", "verdict.json")),
    "null_calibration": ("outputs", ("comparison.csv", "draws.csv", "masks.csv", "prompts.csv", "replay_arrays.npz", "strata.csv", "summary.json")),
    "null_v2": ("outputs", ("associations.csv", "cells.csv", "dose_cells.csv", "dose_summary.csv", "summary.json")),
    "charts": ("outputs", ("widget_inputs.json",)),
}
RAW = {
    "v1": ("spiral_out_hltd_prefix_transfer_v1", ("atlas.npz", "atlas_freeze.json", "atlas_metadata.json", "calibration_hidden.npz", "evaluation_prefixes.npz", "execution_receipt.json", "execution_started.json", "final_weight_audit.json", "load_audit.json", "pilot_audit.json", "support_preflight.json")),
    "v2": ("spiral_out_hltd_prefix_transfer_v2", ("evaluation_observations.npz", "execution_receipt.json", "execution_started.json", "final_weight_audit.json", "load_audit.json", "pilot_audit.json", "pilot_observations.npz", "pre_treatment_gate.json", "raw_treatments.csv", "readout_freeze.json", "support_preflight.json")),
}
SOURCES = {
    "scripts/audit_hltd_prefix_calibration.py": "63a8057cc452c819945d98709b6db28f5e75878332321691bcdfc2ea03daba8d",
    "scripts/audit_hltd_prefix_transfer.py": "1996f6d68096194fd3c5d64fdb059b6f0e36c747fbad50fbe5bb7cd21bd2f5d5",
    "scripts/export_hltd_prefix_null_charts.py": "b48c3e7ced58a0c430afe4f47671ff9226148db66a0f4a2dc46ed50d5084ecbe",
}


def receipt(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def pinned_json(path, sha):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("canonical manifest changed: " + str(path))
    return json.loads(raw)


def bind_files(records, expected, root=ROOT):
    """Require an exact inventory and bind every consumer to its verified path."""
    by_path = {path.resolve(): key for key, path in expected.items()}
    if len(by_path) != len(expected):
        raise ValueError("aliased expected paths")
    bound = {}
    for record in records:
        path = (root / record["path"]).resolve()
        key = by_path.get(path)
        if key is None or key in bound:
            raise ValueError("unexpected, redirected or duplicate artifact: " + str(path))
        actual = receipt(path)
        if any(actual[k] != record[k] for k in ("sha256", "bytes")):
            raise ValueError("changed artifact: " + str(path))
        bound[key] = path
    if set(bound) != set(expected):
        raise ValueError("incomplete artifact inventory")
    return bound


def bind_release(root=ROOT):
    manifests, outputs, raw = {}, {}, {}
    for name, (relative, sha) in MANIFESTS.items():
        path = root / relative
        manifest = pinned_json(path, sha)
        key, names = OUTPUTS[name]
        outputs[name] = bind_files(manifest[key], {n: path.parent / n for n in names}, root)
        manifests[name] = manifest
        if name in RAW:
            directory, names = RAW[name]
            raw[name] = bind_files(manifest["raw_files"], {n: root / directory / n for n in names}, root)
        if name != "charts":
            filename = "execution_protocol.json" if name == "v1" else "protocol.json"
            bind_files([manifest["protocol"]], {"protocol": path.parent.parent / filename}, root)
    base = root / "docs/data/hltd_prefix_null_comparison"
    bind_files([manifests["null_v2"]["calibration_manifest"], manifests["null_v2"]["v2_manifest"]], {
        "stage_a": base / "calibration/manifest.json",
        "original_v2": root / MANIFESTS["v2"][0],
    }, root)
    chart_inputs = bind_files(manifests["charts"]["sources"], {
        "protocol": base / "protocol.json", "stage_a": base / "calibration/manifest.json",
        "stage_b": base / "v2/manifest.json", "exporter": root / "scripts/export_hltd_prefix_null_charts.py",
        "draws": outputs["null_calibration"]["draws.csv"], "calibration_sql": base / "calibration_chart.sql",
        "treatments": raw["v2"]["raw_treatments.csv"], "v2_sql": base / "v2_chart.sql",
    }, root)
    for name, sha in SOURCES.items():
        if receipt(root / name)["sha256"] != sha:
            raise ValueError("historical source changed: " + name)
    return manifests, outputs, chart_inputs


def verify(root=ROOT):
    manifests, outputs, chart_inputs = bind_release(root)
    from scripts import audit_hltd_prefix_calibration as calibration
    from scripts import audit_hltd_prefix_transfer as transfer
    from scripts import export_hltd_prefix_null_charts as charts

    results = {"calibration": calibration.audit(outputs["calibration"]["summary.json"].parent)}
    for name in ("v1", "v2"):
        results[name] = transfer.audit(outputs[name]["verdict.json"].parent)
    saved = json.loads(outputs["charts"]["widget_inputs.json"].read_text())
    for name, data_key, query_key, table in (
        ("calibration", "draws", "calibration_sql", "calibration_draws"),
        ("v2", "treatments", "v2_sql", "v2_treatments"),
    ):
        rows = charts.query_csv(chart_inputs[data_key], table, chart_inputs[query_key].read_text())
        if rows != saved[name]["table"]["rows"]:
            raise ValueError("saved chart differs from verified CSV/query inputs")
    # Recheck after the legacy arithmetic helpers, before issuing any receipt.
    if bind_release(root) != (manifests, outputs, chart_inputs):
        raise ValueError("release inputs changed during verification")
    return {"status": "RECORDED_EVIDENCE_VERIFIED", "semantic_direction_status": "NOT_TESTED",
            "prior_verdicts_unchanged": {k: results[k]["status"] for k in ("v1", "v2")},
            "audits": results, "chart_rows": {k: len(saved[k]["table"]["rows"]) for k in ("calibration", "v2")},
            "manifests": [receipt(root / path) for path, _ in MANIFESTS.values()],
            "verifier": receipt(Path(__file__).resolve()),
            "scope": "Canonical manifest, exact inventory/path binding, arithmetic and saved-chart replay only. No model execution or semantic validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    result = verify()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(payload)
    print(payload)


if __name__ == "__main__":
    main()
