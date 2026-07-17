#!/usr/bin/env python3
"""Plan closed-loop runs from the HLTD branch-band candidate scoreboard."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_MODEL_PATH = "/Users/ryospiralarchitect/SpiralReality/model/gpt2"
DEFAULT_COMPONENTS = ["coexact", "coexact_minus_presence", "presence", "presence_plus_coexact", "negative_coexact"]


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def shell_join(parts: Iterable[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return text or "x"


def parse_layer_spans(value: Any) -> List[int]:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "control", "-"}:
        return []
    layers: List[int] = []
    for token in text.split():
        match = re.fullmatch(r"L?(\d+)(?:-L?(\d+))?", token)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            start, end = end, start
        layers.extend(range(start, end + 1))
    return sorted(set(layers))


def compact_number_slug(prefix: str, values: Sequence[float]) -> str:
    parts = []
    for value in values:
        if float(value).is_integer():
            parts.append(str(int(value)))
        else:
            parts.append(str(value).replace(".", "p"))
    return prefix + "_".join(parts)


def layer_slug(layers: Sequence[int]) -> str:
    return "l" + "_".join(str(layer) for layer in layers) if layers else "control"


def run_root_for_candidate(
    *,
    root: Path,
    rank: int,
    family: str,
    component: str,
    layers: Sequence[int],
    k_values: Sequence[int],
    alphas: Sequence[float],
    seeds: Sequence[int],
) -> Path:
    alpha_slug = compact_number_slug("a", alphas).replace("0p", "0p")
    seed_slug = "s" + ("_".join(str(seed) for seed in seeds) if len(seeds) <= 3 else f"{min(seeds)}_{max(seeds)}")
    k_slug = "k" + "_".join(str(k) for k in k_values)
    name = "__".join(
        [
            f"{rank:02d}",
            slugify(family),
            slugify(component),
            layer_slug(layers),
            k_slug,
            alpha_slug,
            seed_slug,
        ]
    )
    return root / name


def build_plan_rows(
    *,
    candidate_rows: Sequence[Dict[str, str]],
    output_root: Path,
    model_path: str,
    suite: str,
    target_set_file: str,
    k_values: Sequence[int],
    alphas: Sequence[float],
    seeds: Sequence[int],
    pca_components: int,
    generate_steps: int,
    max_rows: int,
    include_labels: Sequence[str],
    exclude_components: Sequence[str],
    min_priority: float,
    device: str,
    torch_dtype: str,
) -> List[Dict[str, Any]]:
    include_label_set = {str(label) for label in include_labels}
    exclude_component_set = {str(component) for component in exclude_components}
    sorted_rows = sorted(candidate_rows, key=lambda row: finite_float(row.get("priority_score")), reverse=True)
    out: List[Dict[str, Any]] = []
    for row in sorted_rows:
        if len(out) >= max_rows:
            break
        label = str(row.get("candidate_label", ""))
        component = str(row.get("component", ""))
        priority = finite_float(row.get("priority_score"))
        if label not in include_label_set:
            continue
        if component in exclude_component_set:
            continue
        if not math.isfinite(priority) or priority < float(min_priority):
            continue
        layers = parse_layer_spans(row.get("recommended_layers"))
        if not layers:
            continue
        family = str(row.get("family", ""))
        rank = len(out) + 1
        run_root = run_root_for_candidate(
            root=output_root,
            rank=rank,
            family=family,
            component=component,
            layers=layers,
            k_values=k_values,
            alphas=alphas,
            seeds=seeds,
        )
        steering_components = [component, "random_tangent"]
        run_parts: List[Any] = [
            "python3",
            "scripts/run_hltd_closed_loop.py",
            "--model-path",
            model_path,
            "--suite",
            suite,
            "--output-root",
            run_root,
            "--layers",
            *layers,
            "--k",
            *k_values,
            "--components",
            pca_components,
            "--generate-steps",
            generate_steps,
            "--alphas",
            *alphas,
            "--seeds",
            *seeds,
            "--families",
            family,
            "--steering-components",
            *steering_components,
            "--target-set-file",
            target_set_file,
        ]
        if device:
            run_parts.extend(["--device", device])
        if torch_dtype:
            run_parts.extend(["--torch-dtype", torch_dtype])

        summarize_parts = [
            "python3",
            "scripts/summarize_hltd_closed_loop.py",
            "--run-root",
            run_root,
        ]
        plot_parts = [
            "python3",
            "scripts/plot_hltd_closed_loop.py",
            "--summary-root",
            run_root,
            "--output-dir",
            run_root / "plots",
            "--components",
            *steering_components,
        ]
        out.append(
            {
                "rank": rank,
                "family": family,
                "component": component,
                "candidate_label": label,
                "priority_score": priority,
                "recommended_layers": row.get("recommended_layers", ""),
                "layers": " ".join(str(layer) for layer in layers),
                "k_values": " ".join(str(k) for k in k_values),
                "alphas": " ".join(str(alpha) for alpha in alphas),
                "seeds": " ".join(str(seed) for seed in seeds),
                "closed_loop_gate": row.get("closed_loop_branch_specific_gate_rate_mean", ""),
                "closed_loop_target_minus_random": row.get("closed_loop_target_margin_delta_minus_random_mean", ""),
                "output_root": str(run_root),
                "run_command": shell_join(run_parts),
                "summarize_command": shell_join(summarize_parts),
                "plot_command": shell_join(plot_parts),
            }
        )
    return out


def write_shell_script(
    rows: Sequence[Dict[str, Any]],
    path: Path,
    *,
    plan_csv: Path,
    result_output_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for row in rows:
        lines.extend(shell_lines_for_row(row, total_rows=len(rows)))
    lines.extend(shell_lines_for_result_summary(plan_csv=plan_csv, result_output_root=result_output_root))
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def shell_lines_for_row(row: Dict[str, Any], *, total_rows: int) -> List[str]:
    run_root = Path(str(row["output_root"]))
    metrics_path = run_root / "closed_loop_metrics.csv"
    summary_path = run_root / "closed_loop_prompt_layer_k_summary.csv"
    manifest_path = run_root / "plots" / "plot_manifest.json"
    return [
        f"echo '[{row['rank']}/{total_rows}] {row['family']} {row['component']} {row['recommended_layers']}'",
        f"if [[ -s {shlex.quote(str(metrics_path))} ]]; then",
        f"  echo '[skip run] {run_root}'",
        "else",
        "  " + str(row["run_command"]),
        "fi",
        f"if [[ -s {shlex.quote(str(summary_path))} ]]; then",
        f"  echo '[skip summary] {run_root}'",
        "else",
        "  " + str(row["summarize_command"]),
        "fi",
        f"if [[ -s {shlex.quote(str(manifest_path))} ]]; then",
        f"  echo '[skip plot] {run_root}'",
        "else",
        "  " + str(row["plot_command"]),
        "fi",
        "",
    ]


def shell_lines_for_result_summary(*, plan_csv: Path, result_output_root: Path) -> List[str]:
    return [
        "echo '[summary] branch-band result scoreboard'",
        shell_join(
            [
                "python3",
                "scripts/summarize_hltd_branch_band_runs.py",
                "--plan-csv",
                plan_csv,
                "--output-root",
                result_output_root,
            ]
        ),
        shell_join(
            [
                "python3",
                "scripts/plot_hltd_branch_band_results.py",
                "--result-root",
                result_output_root,
                "--output-dir",
                result_output_root / "plots",
            ]
        ),
        "",
    ]


def write_rank_scripts(
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    *,
    plan_csv: Path,
    result_output_root: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for row in rows:
        rank = int(row["rank"])
        family = slugify(str(row["family"]))
        component = slugify(str(row["component"]))
        path = output_dir / f"run_rank_{rank:02d}__{family}__{component}.sh"
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
        ]
        lines.extend(shell_lines_for_row(row, total_rows=len(rows)))
        lines.extend(shell_lines_for_result_summary(plan_csv=plan_csv, result_output_root=result_output_root))
        path.write_text("\n".join(lines), encoding="utf-8")
        path.chmod(0o755)
        paths.append(path)
    return paths


def write_markdown(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HLTD Branch-Band Run Plan",
        "",
        "This file is generated from `branch_band_candidate_scoreboard.csv`.",
        "Each row is a closed-loop follow-up for one family/component layer band.",
        "",
        "| rank | family | component | candidate | layers | priority | gate | target-random | output |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {rank} | {family} | {component} | {candidate_label} | {layers} | {priority:.4f} | {gate} | {target} | `{output}` |".format(
                rank=row["rank"],
                family=row["family"],
                component=row["component"],
                candidate_label=row["candidate_label"],
                layers=row["layers"],
                priority=float(row["priority_score"]),
                gate=row["closed_loop_gate"] or "nan",
                target=row["closed_loop_target_minus_random"] or "nan",
                output=row["output_root"],
            )
        )
    lines.extend(
        [
            "",
            "## Rank Scripts",
            "",
            "Each rank script is resume-safe and refreshes the combined result scoreboard after it finishes.",
            "",
            "| rank | script |",
            "| ---: | --- |",
        ]
    )
    for row in rows:
        rank = int(row["rank"])
        script_name = f"rank_scripts/run_rank_{rank:02d}__{slugify(str(row['family']))}__{slugify(str(row['component']))}.sh"
        lines.append(f"| {rank} | `{script_name}` |")
    lines.extend(["", "## Commands", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['rank']}. {row['family']} / {row['component']}",
                "",
                "```bash",
                str(row["run_command"]),
                str(row["summarize_command"]),
                str(row["plot_command"]),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", default="spiral_out_hltd_branch_hodge/branch_band_candidate_scoreboard.csv")
    parser.add_argument("--output-root", default="spiral_out_hltd_branch_band_plan")
    parser.add_argument("--run-output-root", default="spiral_out_hltd_branch_band_runs")
    parser.add_argument("--result-output-root", default="spiral_out_hltd_branch_band_results")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--suite", default="data/hltd_prompt_suite.jsonl")
    parser.add_argument("--target-set-file", default="data/hltd_semantic_targets.json")
    parser.add_argument("--k", type=int, nargs="+", default=[16])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--generate-steps", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument(
        "--include-labels",
        nargs="+",
        default=["causal_band_ready", "structural_band_ready", "causal_exception_band", "narrow_layer_probe"],
    )
    parser.add_argument("--exclude-components", nargs="*", default=[])
    parser.add_argument("--min-priority", type=float, default=0.25)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--torch-dtype", default="auto")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    rows = build_plan_rows(
        candidate_rows=read_csv(Path(args.candidate_csv)),
        output_root=Path(args.run_output_root),
        model_path=args.model_path,
        suite=args.suite,
        target_set_file=args.target_set_file,
        k_values=args.k,
        alphas=args.alphas,
        seeds=args.seeds,
        pca_components=int(args.components),
        generate_steps=int(args.generate_steps),
        max_rows=int(args.top_n),
        include_labels=args.include_labels,
        exclude_components=args.exclude_components,
        min_priority=float(args.min_priority),
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    plan_csv = output_root / "branch_band_run_plan.csv"
    write_csv(rows, plan_csv)
    write_shell_script(
        rows,
        output_root / "run_branch_band_plan.sh",
        plan_csv=plan_csv,
        result_output_root=Path(args.result_output_root),
    )
    rank_script_paths = write_rank_scripts(
        rows,
        output_root / "rank_scripts",
        plan_csv=plan_csv,
        result_output_root=Path(args.result_output_root),
    )
    write_markdown(rows, output_root / "branch_band_run_plan.md")
    manifest = {
        "candidate_csv": str(args.candidate_csv),
        "run_output_root": str(args.run_output_root),
        "result_output_root": str(args.result_output_root),
        "model_path": str(args.model_path),
        "suite": str(args.suite),
        "target_set_file": str(args.target_set_file),
        "k": [int(k) for k in args.k],
        "alphas": [float(alpha) for alpha in args.alphas],
        "seeds": [int(seed) for seed in args.seeds],
        "top_n": int(args.top_n),
        "include_labels": list(args.include_labels),
        "exclude_components": list(args.exclude_components),
        "min_priority": float(args.min_priority),
        "rank_scripts": [str(path) for path in rank_script_paths],
        "rows": len(rows),
    }
    (output_root / "branch_band_run_plan_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote branch-band run plan -> {output_root}")
    print(f"planned rows: {len(rows)}")
    for row in rows:
        print(f"{row['rank']:02d} {row['family']}/{row['component']} layers={row['layers']} priority={float(row['priority_score']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
