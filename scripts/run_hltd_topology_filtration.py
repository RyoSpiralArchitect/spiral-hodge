#!/usr/bin/env python3
"""Run HLTD over a nested triangle filtration on fixed kNN graphs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spiral_hodge as hodge
from scripts.run_hltd_steering import _load_model_and_tokenizer
from scripts.run_hltd_steering_fast_suite import _extract_prompt_outputs, _prompt_inputs
from scripts.run_hltd_steering_suite import read_suite, select_prompts


BRANCHES = ("exact", "coexact", "harmonic")
DIAGNOSTIC_MIN_ENERGY_RATIO = 1e-10
DEFAULT_RADIUS_SCALES = (0.0, 0.75, 0.85, 0.95, 1.0, 1.05, 1.1, 1.15, 1.3, np.inf)


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_radius_scale(value: str) -> float:
    """Parse a normalized triangle radius, accepting ``full`` as infinity."""

    normalized = str(value).strip().lower()
    if normalized in {"full", "inf", "infinity"}:
        return float("inf")
    try:
        parsed = float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid radius scale: {value}") from exc
    if np.isnan(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("radius scale must be non-negative or 'full'")
    return parsed


def serialized_radius_scale(value: float) -> Any:
    return "full" if np.isinf(float(value)) else float(value)


def null_node_vectors(vectors: np.ndarray, *, variant: str, seed: int) -> np.ndarray:
    """Construct speed-preserving node-vector nulls."""

    vectors = np.asarray(vectors, dtype=np.float64)
    if variant == "real":
        return vectors.copy()
    rng = np.random.default_rng(int(seed))
    if variant == "vector_shuffle":
        return vectors[rng.permutation(len(vectors))].copy()
    if variant == "random_tangent":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        random_vectors = rng.normal(size=vectors.shape)
        random_norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
        return random_vectors / np.maximum(random_norms, 1e-12) * norms
    raise ValueError(f"unknown variant: {variant}")


def relative_operator_norm(
    operator: Any,
    vector: np.ndarray,
    *,
    reference_norm: float,
    min_fraction: float = np.sqrt(DIAGNOSTIC_MIN_ENERGY_RATIO),
) -> float:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(vector))
    if denom <= max(1e-12, float(min_fraction) * float(reference_norm)):
        return float("nan")
    return float(np.linalg.norm(operator @ vector) / denom)


def component_alignment(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-30:
        return float("nan")
    return float(np.dot(a, b) / denom)


def component_is_active(
    vector: np.ndarray,
    *,
    reference_norm: float,
    min_energy_ratio: float = DIAGNOSTIC_MIN_ENERGY_RATIO,
) -> bool:
    threshold = np.sqrt(float(min_energy_ratio)) * float(reference_norm)
    return float(np.linalg.norm(vector)) > max(1e-12, threshold)


def filtration_rows_for_field(
    *,
    points: np.ndarray,
    vectors: np.ndarray,
    prompt_id: str,
    family: str,
    layer: int,
    k: int,
    filtration_mode: str,
    fill_fractions: Sequence[float],
    radius_scales: Sequence[float],
    null_variants: Sequence[str],
    null_seeds: Sequence[int],
    ridge: float,
    hodge_solver: str,
) -> List[Dict[str, Any]]:
    """Decompose one field over a fixed graph and nested triangle complexes."""

    points = np.asarray(points, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    edges = hodge.build_knn_edges(points, k=int(k))
    B = hodge.vertex_edge_incidence(len(points), edges)
    ordered_triangles, _triangle_scores, _edge_scale = (
        hodge.clique_triangle_filtration_geometry(points, edges)
    )
    C_full = hodge.triangle_boundary_matrix(edges, ordered_triangles)
    exact_basis, _exact_rank_after, _exact_accepted = (
        hodge.incremental_orthonormal_column_basis(B.T)
    )
    cycle_rank = max(len(edges) - exact_basis.shape[1], 0)
    coexact_basis_full, triangle_rank_after, _triangle_accepted = (
        hodge.incremental_orthonormal_column_basis(
            C_full,
            maximum_rank=cycle_rank,
        )
    )
    variant_jobs: List[Tuple[str, int, np.ndarray]] = [("real", -1, vectors)]
    for variant in null_variants:
        for seed in null_seeds:
            variant_jobs.append(
                (
                    str(variant),
                    int(seed),
                    null_node_vectors(vectors, variant=str(variant), seed=int(seed)),
                )
            )
    flow_batch = np.vstack(
        [hodge.edge_flow_from_node_vectors(points, job_vectors, edges) for _, _, job_vectors in variant_jobs]
    )

    if filtration_mode == "count":
        filtration_steps = sorted(set(float(value) for value in fill_fractions))
    elif filtration_mode == "radius":
        filtration_steps = sorted(set(float(value) for value in radius_scales))
    else:
        raise ValueError(f"unknown filtration mode: {filtration_mode}")
    open_harmonic: Dict[Tuple[str, int], np.ndarray] = {}
    previous_harmonic: Dict[Tuple[str, int], np.ndarray] = {}
    open_harmonic_energy: Dict[Tuple[str, int], float] = {}
    rows: List[Dict[str, Any]] = []
    for filtration_index, filtration_step in enumerate(filtration_steps):
        if filtration_mode == "count":
            C, triangles, _scores, filtration = hodge.triangle_clique_filtration(
                points,
                edges,
                fill_fraction=filtration_step,
            )
        else:
            C, triangles, _scores, filtration = hodge.triangle_clique_radius_filtration(
                points,
                edges,
                radius_scale=filtration_step,
            )
        triangle_count = len(triangles)
        triangle_rank = (
            int(triangle_rank_after[triangle_count - 1]) if triangle_count else 0
        )
        topology = hodge.hodge_complex_topology_diagnostics(
            len(points),
            edges,
            C,
            triangle_rank=triangle_rank,
        )
        if hodge_solver == "orthogonal":
            coexact_basis = coexact_basis_full[:, :triangle_rank]
            exact_batch, coexact_batch, harmonic_batch, energies = (
                hodge.hodge_decompose_graph_edge_flows_from_bases(
                    flow_batch,
                    exact_basis,
                    coexact_basis,
                )
            )
        elif hodge_solver == "ridge":
            exact_batch, coexact_batch, harmonic_batch, _phi, _psi, energies = (
                hodge.hodge_decompose_graph_edge_flows(
                    flow_batch,
                    B,
                    C,
                    ridge=float(ridge),
                )
            )
        else:
            raise ValueError(f"unknown Hodge solver: {hodge_solver}")
        for job_index, (variant, seed, _variant_vectors) in enumerate(variant_jobs):
            flow = flow_batch[job_index]
            exact = exact_batch[job_index]
            coexact = coexact_batch[job_index]
            harmonic = harmonic_batch[job_index]
            energy = energies[job_index]
            flow_norm = float(np.linalg.norm(flow))
            harmonic_active = component_is_active(
                harmonic,
                reference_norm=flow_norm,
            )
            key = (variant, seed)
            if key not in open_harmonic:
                open_harmonic[key] = harmonic.copy()
                open_harmonic_energy[key] = float(energy["harmonic"])
            prior = previous_harmonic.get(key)
            harmonic_survival = float(
                energy["harmonic"] / max(open_harmonic_energy[key], 1e-30)
            )
            prior_active = (
                prior is not None
                and component_is_active(prior, reference_norm=flow_norm)
            )
            row: Dict[str, Any] = {
                "family": family,
                "prompt_id": prompt_id,
                "layer": int(layer),
                "k": int(k),
                "variant": variant,
                "seed": int(seed),
                "filtration_index": int(filtration_index),
                "hodge_solver": hodge_solver,
                **filtration,
                **topology,
                "total_energy": float(energy["total"]),
                "exact_energy": float(energy["exact"]),
                "coexact_energy": float(energy["coexact"]),
                "harmonic_energy": float(energy["harmonic"]),
                "exact_ratio": float(energy["exact_ratio"]),
                "coexact_ratio": float(energy["coexact_ratio"]),
                "harmonic_ratio": float(energy["harmonic_ratio"]),
                "semantic_flow_ratio": float(energy["semantic_flow_ratio"]),
                "energy_closure_error": float(
                    abs(
                        energy["total"]
                        - energy["exact"]
                        - energy["coexact"]
                        - energy["harmonic"]
                    )
                    / max(energy["total"], 1e-30)
                ),
                "exact_coexact_alignment": float(energy["exact_coexact_alignment"]),
                "exact_harmonic_alignment": float(energy["exact_harmonic_alignment"]),
                "coexact_harmonic_alignment": float(energy["coexact_harmonic_alignment"]),
                "harmonic_survival_ratio": harmonic_survival,
                "harmonic_alignment_open": (
                    component_alignment(harmonic, open_harmonic[key])
                    if harmonic_active
                    else float("nan")
                ),
                "harmonic_alignment_previous": (
                    component_alignment(harmonic, prior)
                    if harmonic_active and prior_active
                    else (1.0 if harmonic_active and prior is None else float("nan"))
                ),
                "harmonic_divergence_relative": relative_operator_norm(
                    B,
                    harmonic,
                    reference_norm=flow_norm,
                ),
                "harmonic_curl_adjoint_relative": relative_operator_norm(
                    C.T,
                    harmonic,
                    reference_norm=flow_norm,
                ),
                "coexact_divergence_relative": relative_operator_norm(
                    B,
                    coexact,
                    reference_norm=flow_norm,
                ),
                "exact_curl_adjoint_relative": relative_operator_norm(
                    C.T,
                    exact,
                    reference_norm=flow_norm,
                ),
                "reconstruction_error": float(energy["reconstruction_error"]),
            }
            rows.append(row)
            previous_harmonic[key] = harmonic.copy()
    return rows


def validate_args(args: argparse.Namespace) -> None:
    if args.filtration_mode == "count":
        fills = [float(value) for value in args.triangle_fill]
        if not fills or any(value < 0.0 or value > 1.0 for value in fills):
            raise ValueError("--triangle-fill values must lie in [0, 1]")
        if 0.0 not in fills or 1.0 not in fills:
            raise ValueError("--triangle-fill must include both 0 and 1 endpoints")
    else:
        radii = [float(value) for value in args.radius_scale]
        if not radii or any(np.isnan(value) or value < 0.0 for value in radii):
            raise ValueError("--radius-scale values must be non-negative or 'full'")
        if 0.0 not in radii or not any(np.isinf(value) for value in radii):
            raise ValueError("--radius-scale must include both 0 and full endpoints")
    if any(int(seed) < 0 for seed in args.null_seeds):
        raise ValueError("--null-seeds must be non-negative")
    if any(variant not in {"vector_shuffle", "random_tangent"} for variant in args.null_variants):
        raise ValueError("--null-variants supports vector_shuffle and random_tangent")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="data/hltd_prompt_suite.jsonl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", default="spiral_out_hltd_topology_filtration")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 5, 7])
    parser.add_argument("--k", type=int, nargs="+", default=[12, 16, 24])
    parser.add_argument(
        "--filtration-mode",
        choices=["count", "radius"],
        default="count",
    )
    parser.add_argument(
        "--triangle-fill",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--radius-scale",
        type=parse_radius_scale,
        nargs="+",
        default=list(DEFAULT_RADIUS_SCALES),
        help="triangle longest-edge / median graph-edge thresholds; use 'full' for the endpoint",
    )
    parser.add_argument(
        "--null-variants",
        nargs="+",
        default=["vector_shuffle", "random_tangent"],
    )
    parser.add_argument("--null-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--prompt-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-prompts-per-family", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument(
        "--hodge-solver",
        choices=["orthogonal", "ridge"],
        default="orthogonal",
    )
    parser.add_argument("--no-normalize-hidden", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    validate_args(args)

    prompt_items = select_prompts(
        read_suite(Path(args.suite)),
        families=args.families,
        prompt_ids=args.prompt_ids,
        limit=args.limit,
        max_prompts_per_family=args.max_prompts_per_family,
    )
    if not prompt_items:
        raise ValueError("No prompts matched the requested filters.")

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
    rows: List[Dict[str, Any]] = []
    run_count = 0
    for prompt_no, item in enumerate(prompt_items, start=1):
        prompt_id = str(item["prompt_id"])
        family = str(item["family"])
        text = str(item["text"])
        print(f"[prompt {prompt_no}/{len(prompt_items)}] {family}/{prompt_id}", flush=True)
        inputs = _prompt_inputs(tokenizer, text, device=device, max_length=int(args.max_length))
        input_ids = inputs["input_ids"][0].detach().cpu().numpy()
        if input_ids.size < 4:
            raise ValueError(f"{prompt_id}: need at least 4 tokens for centered HLTD")
        _outputs, hidden = _extract_prompt_outputs(model, inputs)
        coord = hodge.make_semantic_coordinates(
            hidden,
            method="pca",
            n_components=int(args.components),
            normalize_hidden=not args.no_normalize_hidden,
            random_state=int(args.seed),
            verbose=False,
        )
        total_layers = hidden.shape[0]
        for layer in args.layers:
            if layer <= 0 or layer >= total_layers:
                raise ValueError(f"invalid layer {layer}; expected 1..{total_layers - 1}")
            field = hodge.token_node_vector_field(coord.coords, layer=int(layer), mode="centered")
            for k in args.k:
                run_count += 1
                print(f"  [field {run_count}] L{layer} k{k}", flush=True)
                rows.extend(
                    filtration_rows_for_field(
                        points=field.points,
                        vectors=field.vectors,
                        prompt_id=prompt_id,
                        family=family,
                        layer=int(layer),
                        k=int(k),
                        filtration_mode=args.filtration_mode,
                        fill_fractions=args.triangle_fill,
                        radius_scales=args.radius_scale,
                        null_variants=args.null_variants,
                        null_seeds=args.null_seeds,
                        ridge=float(args.ridge),
                        hodge_solver=args.hodge_solver,
                    )
                )

    metrics_path = output_root / "topology_filtration_metrics.csv"
    write_csv(rows, metrics_path)
    metadata = {
        "model_ref": str(model_ref),
        "prompt_ids": [str(item["prompt_id"]) for item in prompt_items],
        "families": sorted({str(item["family"]) for item in prompt_items}),
        "layers": [int(value) for value in args.layers],
        "k": [int(value) for value in args.k],
        "filtration_mode": args.filtration_mode,
        "triangle_fill": sorted(set(float(value) for value in args.triangle_fill)),
        "radius_scale": [
            serialized_radius_scale(value)
            for value in sorted(set(float(value) for value in args.radius_scale))
        ],
        "hodge_solver": args.hodge_solver,
        "null_variants": list(args.null_variants),
        "null_seeds": [int(value) for value in args.null_seeds],
        "components": int(args.components),
        "max_length": int(args.max_length),
        "ridge": float(args.ridge),
        "row_count": len(rows),
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    print(f"filtration complete: {len(rows)} rows in {elapsed:.1f}s -> {metrics_path}", flush=True)

    if not args.no_plot:
        from scripts.plot_hltd_topology_filtration import render_all

        render_all(metrics_path=metrics_path, output_root=output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
