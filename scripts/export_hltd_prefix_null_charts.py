#!/usr/bin/env python3
"""Export reviewed widget tables with executed SQLite provenance, no model code."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/data/hltd_prefix_null_comparison"


def receipt(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(record):
    if receipt(Path(record["path"])) != record:
        raise ValueError("changed chart source: " + record["path"])


def query_csv(path, table_name, sql):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows, columns = list(reader), reader.fieldnames
    with sqlite3.connect(":memory:") as db:
        db.row_factory = sqlite3.Row
        quoted = [f'"{column.replace(chr(34), chr(34) * 2)}"' for column in columns]
        db.execute(f'CREATE TABLE "{table_name}" ({",".join(quoted)})')
        db.executemany(f'INSERT INTO "{table_name}" VALUES ({",".join("?" for _ in columns)})',
                       [[r[key] for key in columns] for r in rows])
        return [dict(row) for row in db.execute(sql)]


def export():
    output = BASE / "charts"
    if output.exists():
        raise FileExistsError(output)
    protocol = receipt(BASE / "protocol.json")
    if protocol["sha256"] != "d1fc6ed31b5a8a40c82f0bcc4e45dd461e8d58ce6a94517ed8f6d10f6246fc91":
        raise ValueError("unexpected follow-up protocol")
    for stage in ("calibration", "v2"):
        manifest = json.loads((BASE / stage / "manifest.json").read_text())
        if manifest["protocol"] != protocol:
            raise ValueError("chart stage protocol mismatch")
        for record in manifest["outputs"]:
            verify(record)
    if manifest["calibration_manifest"] != receipt(BASE / "calibration/manifest.json"):
        raise ValueError("stage comparison changed")
    verify(manifest["v2_manifest"])
    original = json.loads(Path(manifest["v2_manifest"]["path"]).read_text())
    raw_record = next(r for r in original["raw_files"] if Path(r["path"]).name == "raw_treatments.csv")
    verify(raw_record)
    definitions = [
        ("calibration", BASE / "calibration/draws.csv", "calibration_draws", "calibration_chart.sql",
         "Average fixed-draw, equal-prompt target cosines on the shared 1117-node mask. Each draw first averages nodes within each of 40 prompts; undefined directions are excluded identically across every arm. Not semantic ground truth."),
        ("v2", Path(raw_record["path"]), "v2_treatments", "v2_chart.sql",
         "Recorded target next-token log-probability change in natural-log units. Average 8 seeds within cell, 3 prefixes within prompt, and 20 prompts equally; all 5760 saved treatments, no model execution. Not fluency.")]
    payload, records = {}, [protocol, receipt(BASE / "calibration/manifest.json"), receipt(BASE / "v2/manifest.json"), receipt(Path(__file__))]
    for name, path, table, query_path, description in definitions:
        query = (BASE / query_path).read_text()
        rows = query_csv(path, table, query)
        payload[name] = {"source": {"path": str(path.relative_to(ROOT)), "label": path.name,
            "query": {"engine": "SQLite in-memory import from the pinned CSV", "language": "sql", "sql": query,
                      "description": description, "tables_used": [table], "executed_at": datetime.now(timezone.utc).isoformat()}},
            "table": {"rows": rows, "row_count": len(rows), "truncated": False}}
        records.extend([receipt(path), receipt(BASE / query_path)])
    expected = {(r["comparator"]): r["mean_cosine"] for r in json.loads((BASE / "calibration/summary.json").read_text())["comparison"] if r["scope"] == "all"}
    errors = [abs(r["mean_cosine"] - expected[r["comparator"]]) for r in payload["calibration"]["table"]["rows"]]
    with (BASE / "v2/dose_summary.csv").open(newline="") as handle:
        doses = {(r["component"], float(r["magnitude"])): r for r in csv.DictReader(handle) if r["scope"] == "all"}
    for row in payload["v2"]["table"]["rows"]:
        key = "delta_plus" if row["alpha"] > 0 else "delta_minus"
        errors.append(abs(row["mean_delta_logprob"] - float(doses[row["component"], abs(row["alpha"])][key])))
        if (row["prompts"], row["prefixes"], row["seeds"]) != (20, 3, 8):
            raise ValueError("chart denominator mismatch")
    if len(payload["calibration"]["table"]["rows"]) != 4 or len(payload["v2"]["table"]["rows"]) != 12 or max(errors) > 1e-12:
        raise ValueError("independent SQL aggregation disagrees with frozen summaries")
    output.mkdir(exist_ok=False)
    with (output / "widget_inputs.json").open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    result = {"passed": True, "scope": "Independent SQLite aggregation from recorded CSV values, not an independent model run.",
              "max_aggregation_error": max(errors), "sources": records, "outputs": [receipt(output / "widget_inputs.json")]}
    with (output / "manifest.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


if __name__ == "__main__":
    print(json.dumps(export(), indent=2))
