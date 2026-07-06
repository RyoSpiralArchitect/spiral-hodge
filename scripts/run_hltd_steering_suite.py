#!/usr/bin/env python3
"""Run one-step HLTD steering over a JSONL prompt suite."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


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


def select_prompts(
    prompts: Sequence[Dict[str, Any]],
    *,
    families: Optional[Sequence[str]] = None,
    prompt_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    max_prompts_per_family: Optional[int] = None,
) -> List[Dict[str, Any]]:
    family_set = {str(x) for x in families or []}
    prompt_set = {str(x) for x in prompt_ids or []}
    counts: Dict[str, int] = defaultdict(int)
    out: List[Dict[str, Any]] = []
    for item in prompts:
        family = str(item["family"])
        prompt_id = str(item["prompt_id"])
        if family_set and family not in family_set:
            continue
        if prompt_set and prompt_id not in prompt_set:
            continue
        if max_prompts_per_family is not None and counts[family] >= max_prompts_per_family:
            continue
        out.append(item)
        counts[family] += 1
        if limit is not None and len(out) >= limit:
            break
    return out


def run_command(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    print(shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="data/hltd_prompt_suite.jsonl", help="JSONL prompt suite")
    parser.add_argument("--model-path", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--output-root", default="spiral_out_hltd_steering_suite")
    parser.add_argument("--layers", type=int, nargs="+", default=[5])
    parser.add_argument("--k", type=int, nargs="+", default=[16])
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument(
        "--steering-components",
        nargs="+",
        default=["presence", "coexact", "semantic_flow", "harmonic", "random_tangent"],
    )
    parser.add_argument("--selector-component", default="coexact")
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--prompt-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-prompts-per-family", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-chart-norm", type=float, default=1e-6)
    parser.add_argument("--target-text", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args(argv)

    suite_path = Path(args.suite)
    prompts = select_prompts(
        read_suite(suite_path),
        families=args.families,
        prompt_ids=args.prompt_ids,
        limit=args.limit,
        max_prompts_per_family=args.max_prompts_per_family,
    )
    if not prompts:
        raise ValueError("No prompts matched the requested filters.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for item in prompts:
        prompt_id = str(item["prompt_id"])
        family = str(item["family"])
        text = str(item["text"])
        for layer in args.layers:
            for k in args.k:
                run_dir = output_root / f"{family}__{prompt_id}__L{layer}__k{k}"
                cmd = [
                    sys.executable,
                    "scripts/run_hltd_steering.py",
                    "--model-path",
                    str(Path(args.model_path).expanduser()),
                    "--text",
                    text,
                    "--prompt-id",
                    prompt_id,
                    "--layer",
                    str(layer),
                    "--k",
                    str(k),
                    "--components",
                    str(args.components),
                    "--max-length",
                    str(args.max_length),
                    "--selector-component",
                    str(args.selector_component),
                    "--device",
                    str(args.device),
                    "--seed",
                    str(args.seed),
                    "--min-chart-norm",
                    str(args.min_chart_norm),
                    "--output-dir",
                    str(run_dir),
                    "--alphas",
                    *[str(x) for x in args.alphas],
                    "--steering-components",
                    *[str(x) for x in args.steering_components],
                ]
                if args.target_text is not None:
                    cmd.extend(["--target-text", str(args.target_text)])
                run_command(cmd, dry_run=args.dry_run)

    if not args.no_summary:
        summary_cmd = [
            sys.executable,
            "scripts/summarize_hltd_steering.py",
            "--run-root",
            str(output_root),
            "--output",
            str(output_root / "summary.csv"),
        ]
        run_command(summary_cmd, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
