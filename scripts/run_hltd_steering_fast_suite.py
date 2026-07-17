#!/usr/bin/env python3
"""Run one-step HLTD steering over a prompt suite in one Python process."""
from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spiral_hodge as hodge
from scripts import summarize_hltd_steering
from scripts.run_hltd_steering import (
    _component_delta_row,
    _decode_token,
    _load_model_and_tokenizer,
    _load_semantic_target_sets,
    _logits_with_deltas,
    _natural_centered_step_norm,
    _random_tangent_component,
    _select_node_indices,
    _semantic_set_key,
    _semantic_token_ids,
    _steering_decomposition,
    _steering_provenance,
    _target_token_id,
    _with_derived_components,
    _write_csv,
    _write_report,
    RANDOM_TANGENT_REFERENCE,
)
from scripts.run_hltd_steering_suite import read_suite, select_prompts


def _prompt_inputs(tokenizer: Any, text: str, *, device: str, max_length: int) -> Dict[str, Any]:
    tok_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
    if max_length is not None:
        tok_kwargs.update({"truncation": True, "max_length": max_length})
    return tokenizer(text, **tok_kwargs).to(device)


def _extract_prompt_outputs(model: Any, inputs: Dict[str, Any]) -> tuple[Any, np.ndarray]:
    import torch

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = torch.stack(outputs.hidden_states, dim=0)[:, 0].detach().float().cpu().numpy()
    return outputs, hidden


def _token_indices_from_args(token_index: int | None, token_indices: Sequence[int] | None) -> List[int]:
    out: List[int] = []
    if token_indices is not None:
        out.extend(int(x) for x in token_indices)
    if token_index is not None:
        out.append(int(token_index))
    return sorted(set(out))


def _layer_rows(
    *,
    model: Any,
    tokenizer: Any,
    inputs: Dict[str, Any],
    input_ids: np.ndarray,
    outputs: Any,
    hidden: np.ndarray,
    coord: Any,
    prompt_id: str,
    layer: int,
    k: int,
    alphas: Sequence[float],
    steering_components: Sequence[str],
    selector_component: str,
    token_selectors: Sequence[str],
    token_indices: Sequence[int],
    position_bins: Sequence[int],
    position_bin_count: int,
    random_seeds: Sequence[int],
    ridge: float,
    complex_mode: str,
    target_betti_1_fraction: float,
    node_ridge: float,
    min_chart_norm: float,
    target_id: int | None,
    target_set: str,
    target_set_ids: Sequence[int],
    control_set_ids: Sequence[int],
) -> tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Any]]:
    layers, tokens, dim = hidden.shape
    if layer <= 0 or layer >= layers:
        raise ValueError(f"--layer must be between 1 and {layers - 1}, got {layer}")

    field = hodge.token_node_vector_field(coord.coords, layer=layer, mode="centered")
    decomp, topology = _steering_decomposition(
        field,
        k_neighbors=k,
        ridge=ridge,
        complex_mode=complex_mode,
        target_betti_1_fraction=target_betti_1_fraction,
    )
    component_vectors = _with_derived_components(hodge.hltd_component_node_vectors(decomp, ridge=node_ridge))
    provenance = _steering_provenance(decomp, topology)

    natural_step_norm = _natural_centered_step_norm(hidden[layer])
    if not np.isfinite(natural_step_norm) or natural_step_norm <= 0.0:
        raise ValueError(f"Could not compute a positive natural hidden step norm for layer {layer}.")

    rows: List[Dict[str, Any]] = []
    for seed in random_seeds:
        seeded_components = dict(component_vectors)
        seeded_components["random_tangent"] = _random_tangent_component(component_vectors, seed=int(seed))

        for token_selector in token_selectors:
            node_indices = _select_node_indices(
                seeded_components,
                token_selector=token_selector,
                selector_component=selector_component,
                token_indices=token_indices,
                position_bins=position_bins,
                position_bin_count=position_bin_count,
                field=field,
            )
            for node_index in node_indices:
                edge_a, edge_b = field.token_edges[node_index]
                token_index = int((edge_a + edge_b) // 2)
                if token_index >= tokens - 1:
                    raise ValueError("Selected token must have a next token for next-token logits.")

                token = _decode_token(tokenizer, int(input_ids[token_index]))
                next_token_id = int(input_ids[token_index + 1])
                base_logits = outputs.logits[0, token_index].detach().float().cpu().numpy()

                jobs: List[Dict[str, Any]] = []
                deltas: List[np.ndarray] = []
                for component in steering_components:
                    if component not in seeded_components:
                        valid = ", ".join(sorted(seeded_components))
                        raise ValueError(f"Unknown steering component {component!r}. Valid: {valid}")
                    chart_vec = np.asarray(seeded_components[component][node_index], dtype=np.float64)
                    hidden_vec = hodge.pca_chart_vectors_to_hidden(chart_vec, coord.reducer)
                    hidden_norm = float(np.linalg.norm(hidden_vec))
                    chart_norm = float(np.linalg.norm(chart_vec))
                    component_active = bool(chart_norm >= float(min_chart_norm) and hidden_norm > 0.0)
                    if component_active:
                        direction = hidden_vec / hidden_norm
                    else:
                        direction = np.zeros(dim, dtype=np.float64)

                    for alpha in alphas:
                        delta = float(alpha) * natural_step_norm * direction
                        deltas.append(delta)
                        jobs.append(
                            {
                                "component": component,
                                "alpha": float(alpha),
                                "delta_norm": float(np.linalg.norm(delta)),
                                "chart_norm": chart_norm,
                                "hidden_norm": hidden_norm,
                                "component_active": component_active,
                            }
                        )

                if not jobs:
                    continue
                steered_batch = _logits_with_deltas(
                    model,
                    inputs,
                    layer=layer,
                    token_index=token_index,
                    deltas=np.vstack(deltas),
                )
                for job, steered_logits in zip(jobs, steered_batch):
                    row = _component_delta_row(
                        prompt_id=prompt_id,
                        layer=layer,
                        k=k,
                        seed=int(seed),
                        token_selector=token_selector,
                        selector_component=selector_component,
                        node_index=node_index,
                        token_index=token_index,
                        token_count=tokens,
                        token=token,
                        next_token_id=next_token_id,
                        component=str(job["component"]),
                        alpha=float(job["alpha"]),
                        delta_norm=float(job["delta_norm"]),
                        natural_step_norm=natural_step_norm,
                        chart_norm=float(job["chart_norm"]),
                        hidden_direction_norm=float(job["hidden_norm"]),
                        component_active=bool(job["component_active"]),
                        base_logits=base_logits,
                        steered_logits=steered_logits,
                        tokenizer=tokenizer,
                        target_id=target_id,
                        target_set=target_set,
                        target_set_ids=target_set_ids,
                        control_set_ids=control_set_ids,
                    )
                    row["random_tangent_reference"] = RANDOM_TANGENT_REFERENCE
                    row.update(provenance)
                    rows.append(row)
    return rows, decomp.energy, topology


def _planned_rows(
    *,
    prompts: Sequence[Dict[str, Any]],
    layers: Sequence[int],
    ks: Sequence[int],
    seeds: Sequence[int],
    token_selectors: Sequence[str],
    steering_components: Sequence[str],
    alphas: Sequence[float],
    position_bins: Sequence[int] = (),
) -> int:
    # This is exact for max_component/middle/fixed with one selected token. For
    # all_interior, the token count depends on the tokenized prompt, so this is
    # intentionally a lower-bound preview.
    selected_token_factor = max(1, len(position_bins)) if "position_bin" in token_selectors else 1
    return (
        len(prompts)
        * len(layers)
        * len(ks)
        * len(seeds)
        * len(token_selectors)
        * selected_token_factor
        * len(steering_components)
        * len(alphas)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="data/hltd_prompt_suite.jsonl", help="JSONL prompt suite")
    parser.add_argument("--model-path", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--output-root", default="spiral_out_hltd_steering_fast_suite")
    parser.add_argument("--layers", type=int, nargs="+", default=[5])
    parser.add_argument("--k", type=int, nargs="+", default=[16])
    parser.add_argument(
        "--complex-mode",
        choices=["full_clique", "matched_betti"],
        default="full_clique",
        help="Triangle-complex contract used for the Hodge branches",
    )
    parser.add_argument(
        "--target-betti-1-fraction",
        type=float,
        default=0.5,
        help="Target first-Betti fraction when --complex-mode=matched_betti",
    )
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument(
        "--steering-components",
        nargs="+",
        default=["presence", "coexact", "semantic_flow", "harmonic", "random_tangent"],
    )
    parser.add_argument("--selector-component", default="coexact")
    parser.add_argument(
        "--token-selectors",
        nargs="+",
        default=None,
        choices=["max_component", "middle", "fixed", "all_interior", "position_bin"],
    )
    parser.add_argument("--token-index", type=int, default=None)
    parser.add_argument("--token-indices", type=int, nargs="+", default=None)
    parser.add_argument("--position-bins", type=int, nargs="+", default=None)
    parser.add_argument("--position-bin-count", type=int, default=12)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--prompt-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-prompts-per-family", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random tangent seeds")
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--node-ridge", type=float, default=1e-4)
    parser.add_argument("--min-chart-norm", type=float, default=1e-6)
    parser.add_argument("--target-text", default=None)
    parser.add_argument("--target-set-file", default=None, help="Optional JSON file with semantic target/control sets")
    parser.add_argument("--target-set-key", default=None, help="Semantic target set key; defaults to prompt_id/family/default")
    parser.add_argument("--no-normalize-hidden", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args(argv)
    if not 0.0 <= float(args.target_betti_1_fraction) <= 1.0:
        parser.error("--target-betti-1-fraction must be between 0 and 1")

    prompts = select_prompts(
        read_suite(Path(args.suite)),
        families=args.families,
        prompt_ids=args.prompt_ids,
        limit=args.limit,
        max_prompts_per_family=args.max_prompts_per_family,
    )
    if not prompts:
        raise ValueError("No prompts matched the requested filters.")

    token_indices = _token_indices_from_args(args.token_index, args.token_indices)
    position_bins = sorted(set(int(x) for x in (args.position_bins or [])))
    default_selectors = ["fixed"] if token_indices else (["position_bin"] if position_bins else ["max_component"])
    token_selectors = list(args.token_selectors or default_selectors)
    random_seeds = list(args.seeds or [args.seed])
    output_root = Path(args.output_root)

    planned = _planned_rows(
        prompts=prompts,
        layers=args.layers,
        ks=args.k,
        seeds=random_seeds,
        token_selectors=token_selectors,
        steering_components=args.steering_components,
        alphas=args.alphas,
        position_bins=position_bins,
    )
    print(
        "planned fast steering: "
        f"{len(prompts)} prompts x {len(args.layers)} layers x {len(args.k)} k values, "
        f">= {planned} rows; complex={args.complex_mode}, "
        f"target_betti_1={args.target_betti_1_fraction:g}",
        flush=True,
    )
    if args.dry_run:
        print(
            shlex.join(
                [
                    sys.executable,
                    "scripts/run_hltd_steering_fast_suite.py",
                    "--suite",
                    str(args.suite),
                    "--model-path",
                    str(Path(args.model_path).expanduser()),
                    "--output-root",
                    str(output_root),
                    "--complex-mode",
                    str(args.complex_mode),
                    "--target-betti-1-fraction",
                    str(args.target_betti_1_fraction),
                ]
            )
        )
        return 0

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
    target_id = _target_token_id(tokenizer, args.target_text)
    semantic_sets = _load_semantic_target_sets(args.target_set_file)
    output_root.mkdir(parents=True, exist_ok=True)

    suite_start = time.perf_counter()
    run_count = 0
    for prompt_no, item in enumerate(prompts, start=1):
        prompt_start = time.perf_counter()
        prompt_id = str(item["prompt_id"])
        family = str(item["family"])
        text = str(item["text"])
        print(f"[prompt {prompt_no}/{len(prompts)}] {family}/{prompt_id}", flush=True)
        target_set = _semantic_set_key(
            semantic_sets,
            requested_key=args.target_set_key,
            prompt_id=prompt_id,
            family=family,
        )
        target_set_ids, control_set_ids = _semantic_token_ids(tokenizer, semantic_sets, target_set)

        inputs = _prompt_inputs(tokenizer, text, device=device, max_length=args.max_length)
        input_ids = inputs["input_ids"][0].detach().cpu().numpy()
        if input_ids.size < 4:
            raise ValueError(f"{prompt_id}: need at least 4 tokens for centered HLTD steering.")
        outputs, hidden = _extract_prompt_outputs(model, inputs)
        layers, _tokens, _dim = hidden.shape
        for layer in args.layers:
            if layer <= 0 or layer >= layers:
                raise ValueError(f"--layers contains {layer}, but valid hidden layers are 1..{layers - 1}")

        coord = hodge.make_semantic_coordinates(
            hidden,
            method="pca",
            n_components=args.components,
            normalize_hidden=not args.no_normalize_hidden,
            random_state=args.seed,
            verbose=False,
        )

        for layer in args.layers:
            for k in args.k:
                run_count += 1
                run_dir = output_root / f"{family}__{prompt_id}__L{layer}__k{k}"
                print(f"  [run {run_count}] L{layer} k{k}", flush=True)
                rows, energy, topology = _layer_rows(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    input_ids=input_ids,
                    outputs=outputs,
                    hidden=hidden,
                    coord=coord,
                    prompt_id=prompt_id,
                    layer=layer,
                    k=k,
                    alphas=args.alphas,
                    steering_components=args.steering_components,
                    selector_component=args.selector_component,
                    token_selectors=token_selectors,
                    token_indices=token_indices,
                    position_bins=position_bins,
                    position_bin_count=args.position_bin_count,
                    random_seeds=random_seeds,
                    ridge=args.ridge,
                    complex_mode=args.complex_mode,
                    target_betti_1_fraction=args.target_betti_1_fraction,
                    node_ridge=args.node_ridge,
                    min_chart_norm=args.min_chart_norm,
                    target_id=target_id,
                    target_set=target_set,
                    target_set_ids=target_set_ids,
                    control_set_ids=control_set_ids,
                )
                _write_csv(rows, run_dir / "steering_metrics.csv")
                _write_report(
                    rows,
                    run_dir / "steering_report.md",
                    prompt=text,
                    model_ref=model_ref,
                    layer=layer,
                    k_neighbors=k,
                    hltd_energy=energy,
                    hltd_topology=topology,
                )
        elapsed = time.perf_counter() - prompt_start
        print(f"[prompt done] {family}/{prompt_id}: {elapsed:.1f}s", flush=True)

    if not args.no_summary:
        summarize_hltd_steering.main(
            [
                "--run-root",
                str(output_root),
                "--output",
                str(output_root / "summary.csv"),
            ]
        )
    elapsed = time.perf_counter() - suite_start
    print(f"fast suite complete: {run_count} prompt/layer/k runs in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
