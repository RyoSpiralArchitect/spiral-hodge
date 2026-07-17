#!/usr/bin/env python3
"""Map HLTD branch strength across all interior token positions."""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spiral_hodge as hodge
from scripts.run_hltd_steering import _load_model_and_tokenizer, _with_derived_components
from scripts.run_hltd_steering_fast_suite import _extract_prompt_outputs, _prompt_inputs
from scripts.run_hltd_steering_suite import read_suite, select_prompts


DEFAULT_COMPONENTS = [
    "presence",
    "coexact",
    "harmonic",
    "semantic_flow",
    "presence_plus_coexact",
    "coexact_minus_presence",
]


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean_or_nan(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return mean(vals) if vals else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return "nan"
    return f"{number:.{digits}f}"


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def position_bin(position_frac: float, bins: int) -> int:
    if bins <= 0:
        raise ValueError("bins must be positive")
    clipped = min(max(float(position_frac), 0.0), 1.0)
    return min(int(clipped * bins), bins - 1)


def decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)]).replace("\n", "\\n")


def node_branch_rows(
    *,
    tokenizer: Any,
    input_ids: np.ndarray,
    prompt_id: str,
    family: str,
    layer: int,
    k: int,
    field: Any,
    decomp: Any,
    component_vectors: Dict[str, np.ndarray],
    components: Sequence[str],
    bins: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    full_vectors = np.asarray(component_vectors["full"], dtype=np.float64)
    full_norms = np.linalg.norm(full_vectors, axis=1)
    token_count = int(len(input_ids))
    denom = max(token_count - 1, 1)
    for node_index, (edge_a, edge_b) in enumerate(field.token_edges):
        token_index = int((edge_a + edge_b) // 2)
        token = decode_token(tokenizer, int(input_ids[token_index]))
        position_frac = float(token_index / denom)
        bin_index = position_bin(position_frac, bins)
        full_norm = float(full_norms[node_index])
        base_total = 0.0
        for base_component in ["presence", "coexact", "harmonic"]:
            if base_component in component_vectors:
                base_total += float(np.linalg.norm(component_vectors[base_component][node_index]))
        for component in components:
            if component not in component_vectors:
                continue
            vec = np.asarray(component_vectors[component][node_index], dtype=np.float64)
            norm = float(np.linalg.norm(vec))
            rows.append(
                {
                    "family": family,
                    "prompt_id": prompt_id,
                    "layer": int(layer),
                    "k": int(k),
                    "token_count": token_count,
                    "node_index": int(node_index),
                    "token_index": token_index,
                    "token": token,
                    "position_frac": position_frac,
                    "position_bin": int(bin_index),
                    "component": component,
                    "component_norm": norm,
                    "full_norm": full_norm,
                    "component_to_full": norm / max(full_norm, 1e-12),
                    "component_base_share": norm / max(base_total, 1e-12),
                    "hltd_exact_ratio": decomp.energy.get("exact_ratio", float("nan")),
                    "hltd_coexact_ratio": decomp.energy.get("coexact_ratio", float("nan")),
                    "hltd_harmonic_ratio": decomp.energy.get("harmonic_ratio", float("nan")),
                    "hltd_semantic_flow_ratio": decomp.energy.get("semantic_flow_ratio", float("nan")),
                }
            )
    return rows


def summarize_by_position(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, int, int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                str(row["family"]),
                int(row["layer"]),
                int(row["k"]),
                int(row["position_bin"]),
                str(row["component"]),
            )
        ].append(row)

    out: List[Dict[str, Any]] = []
    for (family, layer, k, bin_index, component), group in sorted(buckets.items()):
        norms = [float(row["component_norm"]) for row in group]
        ratios = [float(row["component_to_full"]) for row in group]
        shares = [float(row["component_base_share"]) for row in group]
        positions = [float(row["position_frac"]) for row in group]
        out.append(
            {
                "family": family,
                "layer": layer,
                "k": k,
                "position_bin": bin_index,
                "component": component,
                "n_nodes": len(group),
                "position_frac_mean": mean_or_nan(positions),
                "component_norm_mean": mean_or_nan(norms),
                "component_to_full_mean": mean_or_nan(ratios),
                "component_base_share_mean": mean_or_nan(shares),
                "component_norm_max": max(norms) if norms else float("nan"),
                "component_to_full_max": max(ratios) if ratios else float("nan"),
            }
        )
    return out


def summarize_peaks(position_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in position_rows:
        buckets[(str(row["family"]), int(row["layer"]), int(row["k"]), str(row["component"]))].append(row)

    def peak_key(row: Dict[str, Any]) -> float:
        value = finite_float(row.get("component_to_full_mean"))
        return value if value is not None else float("-inf")

    out: List[Dict[str, Any]] = []
    for (family, layer, k, component), group in sorted(buckets.items()):
        best = max(group, key=peak_key)
        out.append(
            {
                "family": family,
                "layer": layer,
                "k": k,
                "component": component,
                "peak_position_bin": best["position_bin"],
                "peak_position_frac_mean": best["position_frac_mean"],
                "peak_component_to_full_mean": best["component_to_full_mean"],
                "peak_component_norm_mean": best["component_norm_mean"],
            }
        )
    return out


def summarize_global_peaks(peak_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in peak_rows:
        buckets[(int(row["k"]), str(row["component"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (k, component), group in sorted(buckets.items()):
        peak_bins = [float(row["peak_position_bin"]) for row in group]
        peak_fracs = [float(row["peak_position_frac_mean"]) for row in group]
        peak_ratios = [float(row["peak_component_to_full_mean"]) for row in group]
        out.append(
            {
                "k": k,
                "component": component,
                "n_family_layer_rows": len(group),
                "peak_position_bin_mean": mean_or_nan(peak_bins),
                "peak_position_bin_min": min(peak_bins) if peak_bins else float("nan"),
                "peak_position_bin_max": max(peak_bins) if peak_bins else float("nan"),
                "peak_position_frac_mean": mean_or_nan(peak_fracs),
                "peak_component_to_full_mean": mean_or_nan(peak_ratios),
            }
        )
    return out


def write_report(
    *,
    output_root: Path,
    position_rows: Sequence[Dict[str, Any]],
    peak_rows: Sequence[Dict[str, Any]],
    global_peaks: Sequence[Dict[str, Any]],
    bins: int,
    k_focus: int,
) -> None:
    lines: List[str] = [
        "# HLTD Branch Position Heatmap",
        "",
        "This report maps reconstructed HLTD branch vector norms across all",
        "interior token positions. Position bins are normalized over the prompt",
        f"token span into {bins} bins.",
        "",
        "## Global Peaks",
        "",
    ]
    global_rows = [
        [
            f"k={row['k']}",
            row["component"],
            fmt(row.get("peak_position_bin_mean")),
            fmt(row.get("peak_position_bin_min"), 1),
            fmt(row.get("peak_position_bin_max"), 1),
            fmt(row.get("peak_position_frac_mean")),
            fmt(row.get("peak_component_to_full_mean")),
        ]
        for row in global_peaks
    ]
    lines.append(
        markdown_table(
            [
                "k",
                "component",
                "mean peak bin",
                "min bin",
                "max bin",
                "mean peak position",
                "mean peak/full",
            ],
            global_rows,
        )
    )

    focus_rows = [
        row
        for row in peak_rows
        if int(row["k"]) == int(k_focus)
        and int(row["layer"]) in {4, 5, 6, 7, 8}
        and row["component"] in {"presence", "coexact", "presence_plus_coexact"}
    ]
    lines.extend(["", f"## Family Peaks at k={k_focus}", ""])
    family_rows = [
        [
            row["family"],
            f"L{row['layer']}",
            row["component"],
            row["peak_position_bin"],
            fmt(row.get("peak_position_frac_mean")),
            fmt(row.get("peak_component_to_full_mean")),
        ]
        for row in focus_rows
    ]
    lines.append(
        markdown_table(
            ["family", "layer", "component", "peak bin", "peak position", "peak/full"],
            family_rows[:80],
        )
    )
    if len(family_rows) > 80:
        lines.append("")
        lines.append(f"Table truncated in report at 80 rows; CSV contains {len(family_rows)} rows.")

    lines.extend(
        [
            "",
            "## Read",
            "",
            "Use this as a structural localization map, not as a causal steering",
            "claim. Peaks show where reconstructed branch vector norms are largest",
            "relative to the local full token-step norm.",
            "",
        ]
    )
    (output_root / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="data/hltd_prompt_suite.jsonl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", default="spiral_out_hltd_branch_heatmap")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--k", type=int, nargs="+", default=[16])
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--branch-components", nargs="+", default=DEFAULT_COMPONENTS)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--report-k", type=int, default=None, help="k value to highlight in the Markdown report")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--prompt-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-prompts-per-family", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--node-ridge", type=float, default=1e-4)
    parser.add_argument("--no-normalize-hidden", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    prompt_items = select_prompts(
        read_suite(Path(args.suite)),
        families=args.families,
        prompt_ids=args.prompt_ids,
        limit=args.limit,
        max_prompts_per_family=args.max_prompts_per_family,
    )
    if not prompt_items:
        raise ValueError("No prompts matched the requested filters.")
    if args.bins <= 0:
        raise ValueError("--bins must be positive")

    model_ref, model_is_local = hodge.resolve_hf_model_ref("gpt2", args.model_path)
    device = hodge.choose_device(args.device)
    local_files_only = bool(model_is_local or hodge.hf_offline_enabled())
    print(f"loading model once: {model_ref}", flush=True)
    model, tokenizer = _load_model_and_tokenizer(
        model_ref,
        device=device,
        local_files_only=local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    node_rows: List[Dict[str, Any]] = []
    run_count = 0
    for prompt_no, item in enumerate(prompt_items, start=1):
        prompt_id = str(item["prompt_id"])
        family = str(item["family"])
        text = str(item["text"])
        print(f"[prompt {prompt_no}/{len(prompt_items)}] {family}/{prompt_id}", flush=True)
        inputs = _prompt_inputs(tokenizer, text, device=device, max_length=args.max_length)
        input_ids = inputs["input_ids"][0].detach().cpu().numpy()
        if input_ids.size < 4:
            raise ValueError(f"{prompt_id}: need at least 4 tokens for centered HLTD heatmap.")
        _outputs, hidden = _extract_prompt_outputs(model, inputs)
        coord = hodge.make_semantic_coordinates(
            hidden,
            method="pca",
            n_components=args.components,
            normalize_hidden=not args.no_normalize_hidden,
            random_state=args.seed,
            verbose=False,
        )
        total_layers = hidden.shape[0]
        for layer in args.layers:
            if layer <= 0 or layer >= total_layers:
                raise ValueError(f"--layers contains {layer}, but valid hidden layers are 1..{total_layers - 1}")
            field = hodge.token_node_vector_field(coord.coords, layer=layer, mode="centered")
            for k in args.k:
                run_count += 1
                print(f"  [heatmap {run_count}] L{layer} k{k}", flush=True)
                decomp = hodge.hodge_latent_traversal_dynamics(
                    field.points,
                    field.vectors,
                    k_neighbors=int(k),
                    ridge=args.ridge,
                    use_triangles=True,
                )
                component_vectors = _with_derived_components(
                    hodge.hltd_component_node_vectors(decomp, ridge=args.node_ridge)
                )
                node_rows.extend(
                    node_branch_rows(
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        prompt_id=prompt_id,
                        family=family,
                        layer=int(layer),
                        k=int(k),
                        field=field,
                        decomp=decomp,
                        component_vectors=component_vectors,
                        components=args.branch_components,
                        bins=int(args.bins),
                    )
                )

    position_rows = summarize_by_position(node_rows)
    peak_rows = summarize_peaks(position_rows)
    global_peak_rows = summarize_global_peaks(peak_rows)

    write_csv(node_rows, output_root / "node_branch_metrics.csv")
    write_csv(position_rows, output_root / "summary_position.csv")
    write_csv(peak_rows, output_root / "summary_peaks.csv")
    write_csv(global_peak_rows, output_root / "summary_global_peaks.csv")
    report_k = int(args.report_k) if args.report_k is not None else int(args.k[len(args.k) // 2])
    write_report(
        output_root=output_root,
        position_rows=position_rows,
        peak_rows=peak_rows,
        global_peaks=global_peak_rows,
        bins=int(args.bins),
        k_focus=report_k,
    )
    elapsed = time.perf_counter() - started
    print(f"branch heatmap complete: {len(node_rows)} node/component rows in {elapsed:.1f}s -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
