#!/usr/bin/env python3
"""Run Spiral Hodge HLTD analysis over a JSONL prompt suite."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


def read_suite(path: Path) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            for key in ["prompt_id", "family", "text"]:
                if not str(item.get(key, "")).strip():
                    raise ValueError(f"{path}:{line_no} missing required key {key!r}")
            prompts.append(item)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def run_command(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="data/hltd_prompt_suite.jsonl", help="JSONL prompt suite")
    parser.add_argument("--model-path", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--output-root", default="spiral_out_hltd_suite", help="Directory for per-prompt runs")
    parser.add_argument("--k", type=int, nargs="+", default=[16], help="One or more HLTD k values")
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--fourier-backend", choices=["direct", "finufft", "jax"], default="direct")
    parser.add_argument("--null-models", default="all")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--no-hltd-triangles", action="store_true", help="Disable 3-clique coexact component")
    parser.add_argument(
        "--topology-label",
        default=None,
        help="Optional suffix for run directories, e.g. no_triangles for topology ablations",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    suite_path = Path(args.suite)
    prompts = read_suite(suite_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for item in prompts:
        prompt_id = str(item["prompt_id"])
        family = str(item["family"])
        text = str(item["text"])
        for k in args.k:
            topology_label = args.topology_label
            if topology_label is None and args.no_hltd_triangles:
                topology_label = "no_triangles"
            run_name = f"{family}__{prompt_id}__k{k}"
            if topology_label:
                run_name = f"{run_name}__{topology_label}"
            run_dir = output_root / run_name
            cmd = [
                sys.executable,
                "spiral_hodge.py",
                "--model-path",
                str(Path(args.model_path).expanduser()),
                "--local-files-only",
                "--text",
                text,
                "--max-length",
                str(args.max_length),
                "--all-layers",
                "--components",
                str(args.components),
                "--hltd",
                "--hltd-vector-mode",
                "centered",
                "--hltd-k",
                str(k),
                "--null-models",
                args.null_models,
                "--fourier-backend",
                args.fourier_backend,
                "--output-dir",
                str(run_dir),
                "--csv-output",
                "layer_metrics.csv",
            ]
            if args.no_hltd_triangles:
                cmd.append("--no-hltd-triangles")
            if args.save_plots:
                cmd.append("--save-plots")
            run_command(cmd, dry_run=args.dry_run)

            report_cmd = [
                sys.executable,
                "spiral_hodge_report.py",
                "--run-dir",
                str(run_dir),
                "--output",
                "report.html",
                "--title",
                f"HLTD {family} {prompt_id} k={k}" + (f" {topology_label}" if topology_label else ""),
            ]
            run_command(report_cmd, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
