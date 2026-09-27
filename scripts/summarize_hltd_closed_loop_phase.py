#!/usr/bin/env python3
"""Audit original-prompt phase occupancy in existing closed-loop logs."""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hltd_position import centered_node_position, position_phase

CONDITION = ["source_id", "family", "layer", "k", "target_set", "component", "alpha"]
TRAJECTORY = [*CONDITION, "prompt_id", "seed"]
PAIR_KEY = ["source_id", "family", "prompt_id", "layer", "k", "target_set", "alpha", "seed", "step"]
PHASES = ["early", "middle", "late"]
INTEGER_FIELDS = [
    "step", "prefix_len", "prompt_len", "layer", "k", "seed", "node_index",
    "component_active", "next_token_id", "top_changed",
]
RESPONSE_FIELDS = [
    "target_margin_delta", "next_token_logprob_gain", "entropy_delta", "kl_base_to_steered",
]
FLOAT_FIELDS = ["alpha", "nearest_distance", "delta_norm", "next_token_base_logprob", *RESPONSE_FIELDS]
MEASURES = [
    "position_frac", "nearest_distance", "active_rate", "baseline_token_mismatch",
    "next_token_base_logprob", *RESPONSE_FIELDS,
]
PAIR_MEASURES = [
    "position_frac", "nearest_distance", "next_token_base_logprob", "baseline_token_mismatch",
    *RESPONSE_FIELDS,
]


def file_record(path: Path, payload: bytes | None = None) -> dict[str, Any]:
    if payload is None:
        payload = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def validate_schedule(data: pd.DataFrame, manifest: dict[str, Any]) -> None:
    """Require the declared arms and complete prefixes, including shared seed copies."""
    try:
        seeds = [int(x) for x in manifest["seeds"]]
        alphas = [float(x) for x in manifest["alphas"]]
        components = list(manifest["steering_components"])
        runs = manifest["runs"]
        limit = int(manifest["generate_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid source manifest schedule.") from exc
    if not seeds or not runs or not alphas or not components or limit < 1:
        raise ValueError("Empty source manifest schedule.")
    if len(set(seeds)) != len(seeds) or len(set(alphas)) != len(alphas) or len(set(components)) != len(components):
        raise ValueError("Duplicate manifest seeds, alphas, or components.")
    if "layers" not in manifest or "k" not in manifest:
        raise ValueError("Missing manifest layer/k schedule.")
    run_keys = [(r["family"], r["prompt_id"], r.get("target_set", ""), int(r["layer"]), int(r["k"])) for r in runs]
    prompt_keys = {r[:3] for r in run_keys}
    expected_runs = {(*p, int(layer), int(k)) for p, layer, k in
                     itertools.product(prompt_keys, manifest["layers"], manifest["k"])}
    if len(set(run_keys)) != len(run_keys) or set(run_keys) != expected_runs:
        raise ValueError("Manifest run list does not cover the declared layer/k schedule.")
    if manifest.get("prompt_ids") and {r[1] for r in prompt_keys} != set(manifest["prompt_ids"]):
        raise ValueError("Manifest run list does not cover the declared prompt IDs.")
    keys = ["family", "prompt_id", "layer", "k", "target_set", "component", "alpha", "seed"]
    expected = set()
    arms = [("baseline", 0.0), *itertools.product([c for c in components if c != "baseline"], alphas)]
    for run in runs:
        for (component, alpha), seed in itertools.product(arms, seeds):
            expected.add((run["family"], run["prompt_id"], int(run["layer"]), int(run["k"]),
                          run.get("target_set", ""), component, alpha, seed))
    actual = set(data[keys].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"Manifest arm coverage mismatch: {len(expected - actual)} missing, "
                         f"{len(actual - expected)} unexpected.")
    for _, group in data.groupby(keys, dropna=False, sort=False):
        group = group.sort_values("step")
        n = len(group)
        if list(group["step"]) != list(range(n)) or n > limit:
            raise ValueError("Duplicate, missing, or out-of-range generation steps.")
        eos_id = manifest.get("eos_token_id")
        if n < limit and not (manifest.get("stop_at_eos") and eos_id is not None
                              and int(group.iloc[-1]["next_token_id"]) == int(eos_id)):
            raise ValueError("Truncated trajectory without a recorded EOS termination.")
        if group["prompt_len"].nunique() != 1:
            raise ValueError("Prompt length changed within a trajectory.")


def annotate_source(path: Path, source_id: str, bins: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = path.with_name("closed_loop_manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    source_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    contract = manifest.get("field_contract")
    if manifest.get("step_schema_version", 1) >= 2 and not contract:
        raise ValueError(f"{source_id}: schema v2 requires the field contract.")
    if contract and (contract.get("vector_mode") != "centered" or contract.get("step") != 1):
        raise ValueError(f"{source_id}: phase mapping requires centered step=1 fields.")
    data = pd.read_csv(io.BytesIO(source_bytes), dtype=str, keep_default_na=False)
    header = set(data.columns)
    required = {*INTEGER_FIELDS, *FLOAT_FIELDS, "family", "prompt_id", "component", "target_set"}
    if data.empty or not required.issubset(data.columns):
        raise ValueError(f"{source_id}: empty source or missing columns: {sorted(required - set(data.columns))}")
    if data[["family", "prompt_id", "component"]].eq("").any().any():
        raise ValueError(f"{source_id}: blank prompt or component identity.")
    for name in [*INTEGER_FIELDS, *FLOAT_FIELDS]:
        values = pd.to_numeric(data[name], errors="coerce")
        optional = name == "target_margin_delta"
        missing = data[name].str.lower().isin(["", "nan"])
        if not (np.isfinite(values) | (optional & missing)).all():
            raise ValueError(f"{source_id}: invalid numeric field {name}.")
        if name in INTEGER_FIELDS:
            if not values.eq(np.floor(values)).all():
                raise ValueError(f"{source_id}: non-integral field {name}.")
            values = values.astype(int)
        data[name] = values
    if not data["prefix_len"].eq(data["prompt_len"] + data["step"]).all():
        raise ValueError(f"{source_id}: prefix length does not match prompt plus generation step.")
    if (data[["step", "seed", "nearest_distance", "delta_norm"]] < 0).any().any():
        raise ValueError(f"{source_id}: negative index, distance, or norm.")
    if not data["component_active"].isin([0, 1]).all() or not data["top_changed"].isin([0, 1]).all():
        raise ValueError(f"{source_id}: invalid activity/change flags.")
    validate_schedule(data, manifest)
    positions = [centered_node_position(int(i), int(n))
                 for i, n in zip(data["node_index"], data["prompt_len"])]
    for column, expected in zip(["node_token_index", "position_frac"], zip(*positions)):
        if column in data:
            recorded = pd.to_numeric(data[column], errors="coerce")
            if not np.allclose(recorded, expected, rtol=0, atol=1e-12):
                raise ValueError(f"{source_id}: recorded {column} disagrees with centered mapping.")
        data[column] = expected
    if manifest.get("step_schema_version", 1) >= 2:
        if not {"node_token_index", "position_frac"}.issubset(header):
            raise ValueError(f"{source_id}: schema v2 requires explicit position fields.")
    data["position_phase"] = data["position_frac"].map(position_phase)
    data["position_bin"] = np.minimum((data["position_frac"] * bins).astype(int), bins - 1)
    data["source_id"] = source_id
    data["active_rate"] = ((data["component"] != "baseline") & data["component_active"].eq(1)
                           & data["delta_norm"].gt(0)).astype(float)
    baseline_key = ["source_id", "family", "prompt_id", "layer", "k", "target_set", "seed", "step"]
    baseline = data.loc[data["component"] == "baseline", [*baseline_key, "next_token_id"]]
    baseline = baseline.rename(columns={"next_token_id": "baseline_next_token_id"})
    data = data.merge(baseline, on=baseline_key, how="left", validate="many_to_one")
    data["baseline_token_mismatch"] = np.where(
        data["baseline_next_token_id"].notna(),
        data["next_token_id"].ne(data["baseline_next_token_id"]).astype(float), np.nan,
    )
    record = {
        "source_id": source_id, "steps": file_record(path, source_bytes),
        "manifest": file_record(manifest_path, manifest_bytes),
        "model_ref": manifest.get("model_ref", ""), "raw_rows": len(data),
        "n_prompts": data[["family", "prompt_id"]].drop_duplicates().shape[0],
        "random_seeds": sorted(data.loc[data["component"] == "random_tangent", "seed"].unique().tolist()),
        "field_contract": contract or "legacy_unrecorded",
        "position_origin": "recorded_and_verified" if "node_token_index" in header
                           else "recovered_centered_step1",
        "declared_max_steps": manifest["generate_steps"],
    }
    return data, record


def collapse_deterministic(data: pd.DataFrame) -> pd.DataFrame:
    deterministic = data[data["component"] != "random_tangent"].copy()
    identity = [*CONDITION, "prompt_id", "step"]
    compared = [c for c in deterministic if c not in {*identity, "seed"}]
    group = deterministic.groupby(identity, sort=False, dropna=False)
    if not group[compared].nunique(dropna=False).le(1).all().all():
        raise ValueError("Deterministic seed copies disagree; cannot collapse them.")
    collapsed = group.first().reset_index()
    collapsed["source_seed_count"] = group.size().to_numpy()
    collapsed["seed"] = -1
    random = data[data["component"] == "random_tangent"].copy()
    random["source_seed_count"] = 1
    return pd.concat([collapsed, random], ignore_index=True).sort_values([*TRAJECTORY, "step"])


def mean_or_nan(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.mean()) if not finite.empty else float("nan")


def summarize_trajectories(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for values, group in data.groupby(TRAJECTORY, sort=True, dropna=False):
        base = dict(zip(TRAJECTORY, values))
        for phase in ["all", *PHASES]:
            visits = group if phase == "all" else group[group["position_phase"] == phase]
            active = visits[visits["active_rate"] == 1]
            row = {**base, "phase": phase, "n_steps": len(group), "n_visits": len(visits),
                   "n_active_visits": len(active), "occupancy": len(visits) / len(group)}
            for metric in MEASURES:
                row[metric] = mean_or_nan((active if metric in RESPONSE_FIELDS else visits)[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def prompt_and_condition_summary(trajectories: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [*CONDITION, "prompt_id", "phase"]
    measurements = ["occupancy", *MEASURES]
    grouped = trajectories.groupby(keys, sort=True, dropna=False)
    prompt = grouped[measurements].mean().reset_index()
    counts = grouped.agg(n_trajectories=("seed", "size"), n_visits=("n_visits", "sum"),
                         n_active_visits=("n_active_visits", "sum")).reset_index()
    prompt = prompt.merge(counts, on=keys, validate="one_to_one")
    grouped = prompt.groupby([*CONDITION, "phase"], sort=True, dropna=False)
    summary = grouped[measurements].mean().reset_index()
    summary["n_prompts"] = grouped.size().to_numpy()
    summary["n_visited_prompts"] = grouped["n_visits"].apply(lambda x: int(x.gt(0).sum())).to_numpy()
    summary["n_active_prompts"] = grouped["n_active_visits"].apply(lambda x: int(x.gt(0).sum())).to_numpy()
    for metric in RESPONSE_FIELDS:
        summary[f"n_prompts_{metric}"] = grouped[metric].count().to_numpy()
    return prompt, summary


def step_summary(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for phase in PHASES:
        data[f"{phase}_occupancy"] = data["position_phase"].eq(phase).astype(float)
    data.loc[data["active_rate"] == 0, RESPONSE_FIELDS] = np.nan
    metrics = [*MEASURES, *(f"{p}_occupancy" for p in PHASES)]
    prompt = data.groupby([*CONDITION, "prompt_id", "step"], dropna=False)[metrics].mean().reset_index()
    grouped = prompt.groupby([*CONDITION, "step"], sort=True, dropna=False)
    summary = grouped[metrics].mean().reset_index()
    summary["n_prompts"] = grouped.size().to_numpy()
    for metric in RESPONSE_FIELDS:
        summary[f"n_prompts_{metric}"] = grouped[metric].count().to_numpy()
    return summary


def random_contrasts(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare arms at aligned generation steps; later prefixes may differ."""
    for phase in PHASES:
        data = data.assign(**{f"{phase}_occupancy": data["position_phase"].eq(phase).astype(float)})
    fields = [*PAIR_MEASURES, *(f"{p}_occupancy" for p in PHASES)]
    random = data[data["component"] == "random_tangent"]
    branch = data[~data["component"].isin(["baseline", "random_tangent"])]
    cols = [*PAIR_KEY, "active_rate", "delta_norm", *fields]
    paired = branch.merge(random[cols], on=PAIR_KEY, how="left", suffixes=("", "_random"),
                          validate="many_to_one", indicator=True)
    paired["random_matched"] = paired["_merge"].eq("both").astype(float)
    paired["both_active"] = paired["active_rate"].eq(1) & paired["active_rate_random"].eq(1)
    if not np.isclose(paired.loc[paired["both_active"], "delta_norm"],
                      paired.loc[paired["both_active"], "delta_norm_random"], rtol=1e-6, atol=1e-9).all():
        raise ValueError("Active branch/random intervention norms do not match.")
    for metric in fields:
        paired[f"gap_{metric}"] = paired[metric] - paired[f"{metric}_random"]
        if metric in RESPONSE_FIELDS:
            paired.loc[~paired["both_active"], f"gap_{metric}"] = np.nan
    gaps = [f"gap_{metric}" for metric in fields]
    paired["both_active"] = paired["both_active"].astype(float)
    metrics = ["random_matched", "both_active", *gaps]
    # Equal steps within seed, equal seeds within prompt, then equal prompts.
    seed = paired.groupby(TRAJECTORY, dropna=False)[metrics].mean().reset_index()
    prompt = seed.groupby([*CONDITION, "prompt_id"], dropna=False)[metrics].mean().reset_index()
    summary = prompt.groupby(CONDITION, sort=True, dropna=False)[metrics].mean().reset_index()
    counts = prompt.groupby(CONDITION, sort=True, dropna=False)
    summary["n_prompts"] = counts.size().to_numpy()
    for metric in gaps:
        summary[f"n_prompts_{metric}"] = counts[metric].count().to_numpy()
    return prompt, summary


def write_report(root: Path, sources: list[dict[str, Any]], summary: pd.DataFrame,
                 plot_paths: Sequence[Path]) -> None:
    raw_rows = sum(s["raw_rows"] for s in sources)
    unique_rows = sum(s["unique_rows"] for s in sources)
    lines = ["# Closed-Loop Phase Audit", "", f"Sources: {len(sources)}; raw rows: {raw_rows:,}; "
             f"rows after verified deterministic seed collapse: {unique_rows:,}.", "",
             "## Reading the Audit", "",
             "Each source remains a separate experiment. Scorer-only reruns and overlapping "
             "prompts are not pooled into independent evidence. Means balance seeds within "
             "each prompt, then prompts within each source/family/layer/k/alpha/target set.", "",
             "For centered step=1 fields, node_token_index = node_index + 1 and "
             "position_frac = node_token_index / (prompt_len - 1). Early is [0,1/3), "
             "middle [1/3,2/3), late [2/3,1]. These are retrieved original-prompt positions, "
             "not positions of the generated tokens.", "",
             "Phase occupancy includes inactive visits. Response means omit inactive visits; "
             "an unvisited or inactive phase has no response estimate. Coverage accompanies "
             "each table. Baseline activity is zero even when a legacy log used a placeholder 1.", "",
             "Random gaps match source, prompt, layer, k, alpha, target set, seed, and generated "
             "step. Later prefixes can differ between arms. Missing random controls remain "
             "missing. No confidence intervals or causal phase-adjustment estimates are fit.", "",
             "Phase is observed after earlier interventions, so phase-conditioned associations "
             "do not identify a causal phase effect. Raw PCA nearest distances lack a local "
             "scale and an off-chart residual; they cannot certify manifold adherence. "
             "Selected-token logprob gain is not an independent fluency test. The legacy "
             "closed-loop ridge/clique runs also differ from the signed matched-Betti gate.", "",
             "## Figures", ""]
    for path in plot_paths:
        if path.parent.name == "plots":
            lines += [f"![{path.stem}]({path.relative_to(root).as_posix()})", ""]
    lines += ["Full condition/step plots are in `plots/by_source/` when figures are enabled. "
              "Source numbers follow the inventory order below.", "",
             "## Source Inventory", "", "| source | prompts | raw rows | unique rows | mapping |",
             "| --- | ---: | ---: | ---: | --- |"]
    for source in sources:
        lines.append(f"| {source['source_id']} | {source['n_prompts']} | {source['raw_rows']} | "
                     f"{source['unique_rows']} | {source['position_origin']} |")
    lines += ["", "## Coexact Position Overview", "",
              "Each row preserves one source/family/layer/k/alpha/target-set condition.", "",
              "| source | family | L | k | alpha | prompts | early | middle | late |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    focus = summary[(summary["component"] == "coexact") & (summary["phase"] != "all")]
    for values, group in focus.groupby(CONDITION, sort=True, dropna=False):
        row = dict(zip(CONDITION, values))
        phases = dict(zip(group["phase"], group["occupancy"]))
        lines.append(f"| {row['source_id']} | {row['family']} | {row['layer']} | {row['k']} | "
                     f"{row['alpha']:g} | {int(group.iloc[0]['n_prompts'])} | "
                     + " | ".join(f"{phases[p]:.1%}" for p in PHASES) + " |")
    (root / "phase_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(paths: Sequence[Path], output_root: Path, *, scan_root: Path, bins: int = 12,
              plots: bool = True) -> dict[str, Any]:
    if bins < 3 or bins % 3:
        raise ValueError("Position bins must be a positive multiple of three.")
    resolved = [p.resolve() for p in paths]
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("No sources, or duplicate resolved source paths.")
    source_frames, canonical_frames, sources = [], [], []
    for path in sorted(resolved):
        try:
            source_id = path.parent.relative_to(scan_root.resolve()).as_posix()
        except ValueError:
            source_id = path.parent.as_posix()
        data, record = annotate_source(path, source_id, bins)
        canonical = collapse_deterministic(data)
        record["unique_rows"] = len(canonical)
        record["deterministic_rows_removed"] = len(data) - len(canonical)
        source_frames.append(data)
        canonical_frames.append(canonical)
        sources.append(record)
    raw = pd.concat(source_frames, ignore_index=True)
    canonical = pd.concat(canonical_frames, ignore_index=True)
    trajectories = summarize_trajectories(canonical)
    prompts, summary = prompt_and_condition_summary(trajectories)
    steps = step_summary(canonical)
    prompt_gaps, gaps = random_contrasts(raw)
    # Validate every input and construct every table before creating any output.
    output_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "phase_steps.csv": canonical, "phase_trajectories.csv": trajectories,
        "phase_prompt_summary.csv": prompts, "phase_summary.csv": summary,
        "phase_step_summary.csv": steps, "phase_prompt_random_gaps.csv": prompt_gaps,
        "phase_random_gaps.csv": gaps,
    }
    for name, frame in tables.items():
        frame.to_csv(output_root / name, index=False)
    plot_paths = []
    if plots:
        from scripts.plot_hltd_closed_loop_phase import render_plots
        plot_paths = render_plots(summary, steps, gaps, output_root / "plots")
    write_report(output_root, sources, summary, plot_paths)
    artifact_paths = [*(output_root / name for name in tables),
                      output_root / "phase_audit_report.md", *plot_paths]
    artifacts = [{**file_record(p), "path": p.relative_to(output_root).as_posix(),
                  **({"rows": len(tables[p.name])} if p.name in tables else {})} for p in artifact_paths]
    code = [Path(__file__).resolve(), ROOT / "scripts/hltd_position.py",
            ROOT / "scripts/plot_hltd_closed_loop_phase.py"]
    manifest = {"schema": "hltd_closed_loop_phase_audit.v1", "bins": bins,
                "aggregation": "steps within seed, seeds within prompt, equal prompts; sources separate",
                "phase_interpretation": "descriptive_post_treatment", "sources": sources,
                "code": [file_record(p) for p in code if p.exists()], "artifacts": artifacts}
    (output_root / "phase_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def export_evidence(output_root: Path, docs_root: Path, manifest: dict[str, Any]) -> None:
    """Copy compact reviewed evidence and pin both source and copied artifacts."""
    copies = {
        name: f"data/hltd_closed_loop_phase/{name}" for name in [
            "phase_summary.csv", "phase_random_gaps.csv", "phase_prompt_random_gaps.csv",
        ]
    }
    copies.update({"plots/phase_occupancy.png": "figures/hltd_closed_loop_phase_occupancy.png",
                   "plots/phase_random_gaps.png": "figures/hltd_closed_loop_phase_random_gaps.png"})
    artifacts = {a["path"]: a for a in manifest["artifacts"]}
    required = [name for name in copies if name in artifacts]
    # Reject stale analysis files before updating any tracked evidence.
    for name in required:
        if file_record(output_root / name)["sha256"] != artifacts[name]["sha256"]:
            raise ValueError(f"Analysis artifact changed before evidence export: {name}")
    tracked = []
    for name in required:
        dest = docs_root / copies[name]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_root / name, dest)
        tracked.append({**artifacts[name], "path": copies[name]})
    evidence = {"schema": "hltd_closed_loop_phase_evidence.v1",
                "analysis_manifest": file_record(output_root / "phase_audit_manifest.json"),
                "sources": manifest["sources"], "code": manifest["code"], "artifacts": tracked}
    target = docs_root / "figures/hltd_closed_loop_phase_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, default=Path("."))
    parser.add_argument("--run-roots", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("spiral_out_hltd_closed_loop_phase_audit"))
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--evidence-root", type=Path, help="Copy compact evidence into a documentation directory.")
    args = parser.parse_args(argv)
    paths = [p / "closed_loop_steps.csv" for p in args.run_roots] if args.run_roots else sorted(
        args.scan_root.glob("spiral_out*/**/closed_loop_steps.csv"))
    manifest = run_audit(paths, args.output_root, scan_root=args.scan_root, bins=args.bins, plots=not args.no_plots)
    if args.evidence_root:
        export_evidence(args.output_root, args.evidence_root, manifest)
    print(f"Audited {len(manifest['sources'])} sources; report: {args.output_root / 'phase_audit_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
