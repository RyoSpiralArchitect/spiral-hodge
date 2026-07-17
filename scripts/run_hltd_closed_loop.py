#!/usr/bin/env python3
"""Run closed-loop HLTD branch steering during greedy generation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spiral_hodge as hodge
from scripts.run_hltd_steering import (
    _decode_token,
    _entropy_from_logp,
    _kl_from_logp,
    _load_model_and_tokenizer,
    _load_semantic_target_sets,
    _log_softmax,
    _logprob_mass,
    _logits_with_delta,
    _natural_centered_step_norm,
    _prob_from_logmass,
    _semantic_set_key,
    _semantic_token_ids,
    _top_token,
    _with_derived_components,
)
from scripts.run_hltd_steering_suite import read_suite, select_prompts


BASELINE_COMPONENT = "baseline"
_LEXEME_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _normalized_lexemes(text: str) -> Tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _LEXEME_RE.finditer(str(text)))


def _contains_lexeme_phrase(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    width = len(needle)
    if width == 0 or width > len(haystack):
        return False
    return any(tuple(haystack[start : start + width]) == tuple(needle) for start in range(len(haystack) - width + 1))


def _prompt_heldout_semantic_target_set(
    semantic_sets: Dict[str, Dict[str, List[str]]],
    key: str,
    prompt: str,
) -> Tuple[Dict[str, List[str]], List[str]]:
    if not key:
        return {"target": [], "control": []}, []
    spec = semantic_sets[key]
    prompt_lexemes = _normalized_lexemes(prompt)
    kept: List[str] = []
    excluded: List[str] = []
    for term in spec.get("target", []):
        term_lexemes = _normalized_lexemes(term)
        if term_lexemes and _contains_lexeme_phrase(prompt_lexemes, term_lexemes):
            excluded.append(str(term))
        else:
            kept.append(str(term))
    if not kept:
        raise ValueError(f"Prompt-overlap exclusion removed every target term from semantic set {key!r}.")
    return {"target": kept, "control": [str(term) for term in spec.get("control", [])]}, excluded


def _inputs_from_ids(input_ids: Sequence[int], *, device: str) -> Dict[str, Any]:
    import torch

    ids = torch.as_tensor([list(int(x) for x in input_ids)], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _forward_hidden_and_logits(
    model: Any,
    input_ids: Sequence[int],
    *,
    device: str,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    import torch

    inputs = _inputs_from_ids(input_ids, device=device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = torch.stack(outputs.hidden_states, dim=0)[:, 0].detach().float().cpu().numpy()
    logits = outputs.logits[0, -1].detach().float().cpu().numpy()
    return inputs, hidden, logits


def _chart_point_from_hidden(hidden_vec: np.ndarray, reducer: Any, *, normalize_hidden: bool) -> np.ndarray:
    x = np.asarray(hidden_vec, dtype=np.float64)[None, :]
    if normalize_hidden:
        x = hodge._safe_l2_normalize(x)
    return np.asarray(reducer.transform(x)[0], dtype=np.float64)


def _nearest_node_index(points: np.ndarray, z: np.ndarray) -> Tuple[int, float]:
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] == 0:
        raise ValueError("nearest-node lookup requires non-empty [nodes, dim] points.")
    diff = P - np.asarray(z, dtype=np.float64)[None, :]
    dist = np.linalg.norm(diff, axis=1)
    idx = int(np.nanargmin(dist))
    return idx, float(dist[idx])


def _random_tangent_component(component_vectors: Dict[str, np.ndarray], *, seed: int) -> np.ndarray:
    coexact = np.asarray(component_vectors["coexact"], dtype=np.float64)
    coexact_norms = np.linalg.norm(coexact, axis=1, keepdims=True)
    rng = np.random.default_rng(int(seed))
    random_chart = rng.normal(size=coexact.shape)
    random_norms = np.linalg.norm(random_chart, axis=1, keepdims=True)
    return random_chart / np.maximum(random_norms, 1e-12) * coexact_norms


def _step_rows_for_seed(rows: Sequence[Dict[str, Any]], *, seed: int) -> List[Dict[str, Any]]:
    """Copy a deterministic trajectory into a matched-random seed stratum."""
    return [{**row, "seed": int(seed)} for row in rows]


def _target_margin(logp: np.ndarray, target_ids: Sequence[int], control_ids: Sequence[int]) -> float:
    target = _logprob_mass(logp, target_ids)
    control = _logprob_mass(logp, control_ids)
    if not math.isfinite(target) or not math.isfinite(control):
        return float("nan")
    return float(target - control)


def _mean_finite(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _run_closed_loop_component(
    *,
    model: Any,
    tokenizer: Any,
    prompt_input_ids: Sequence[int],
    device: str,
    coord: Any,
    field: Any,
    component_vectors: Dict[str, np.ndarray],
    layer: int,
    k: int,
    component: str,
    alpha: float,
    seed: int,
    generate_steps: int,
    natural_step_norm: float,
    normalize_hidden: bool,
    min_chart_norm: float,
    target_set: str,
    target_set_ids: Sequence[int],
    control_set_ids: Sequence[int],
    stop_at_eos: bool,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    ids = [int(x) for x in prompt_input_ids]
    prompt_len = len(ids)
    step_rows: List[Dict[str, Any]] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)

    for step in range(int(generate_steps)):
        inputs, hidden_now, base_logits = _forward_hidden_and_logits(model, ids, device=device)
        token_index = len(ids) - 1
        base_logp = _log_softmax(base_logits)
        base_entropy = _entropy_from_logp(base_logp)
        base_top_id, base_top_token, base_top_logprob = _top_token(base_logp, tokenizer)

        z = _chart_point_from_hidden(hidden_now[layer, -1], coord.reducer, normalize_hidden=normalize_hidden)
        node_index, nearest_distance = _nearest_node_index(field.points, z)

        chart_norm = 0.0
        hidden_direction_norm = 0.0
        component_active = False
        delta = np.zeros(hidden_now.shape[-1], dtype=np.float64)
        if component != BASELINE_COMPONENT:
            if component not in component_vectors:
                valid = ", ".join(sorted(component_vectors))
                raise ValueError(f"Unknown steering component {component!r}. Valid: {valid}")
            chart_vec = np.asarray(component_vectors[component][node_index], dtype=np.float64)
            hidden_vec = hodge.pca_chart_vectors_to_hidden(chart_vec, coord.reducer)
            chart_norm = float(np.linalg.norm(chart_vec))
            hidden_direction_norm = float(np.linalg.norm(hidden_vec))
            component_active = bool(chart_norm >= float(min_chart_norm) and hidden_direction_norm > 0.0)
            if component_active:
                direction = hidden_vec / hidden_direction_norm
                delta = float(alpha) * float(natural_step_norm) * direction

        if component == BASELINE_COMPONENT or not component_active:
            steered_logits = base_logits
        else:
            steered_logits = _logits_with_delta(
                model,
                inputs,
                layer=layer,
                token_index=token_index,
                delta=delta,
            )
        steered_logp = _log_softmax(steered_logits)
        steered_entropy = _entropy_from_logp(steered_logp)
        steered_top_id, steered_top_token, steered_top_logprob = _top_token(steered_logp, tokenizer)
        next_token_id = int(steered_top_id)
        next_token = _decode_token(tokenizer, next_token_id)

        base_target_margin = _target_margin(base_logp, target_set_ids, control_set_ids)
        steered_target_margin = _target_margin(steered_logp, target_set_ids, control_set_ids)
        target_margin_delta = (
            steered_target_margin - base_target_margin
            if math.isfinite(base_target_margin) and math.isfinite(steered_target_margin)
            else float("nan")
        )
        target_mass_base = _logprob_mass(base_logp, target_set_ids)
        target_mass_steered = _logprob_mass(steered_logp, target_set_ids)
        control_mass_base = _logprob_mass(base_logp, control_set_ids)
        control_mass_steered = _logprob_mass(steered_logp, control_set_ids)

        step_rows.append(
            {
                "step": int(step),
                "prefix_len": int(len(ids)),
                "prompt_len": int(prompt_len),
                "layer": int(layer),
                "k": int(k),
                "seed": int(seed),
                "component": component,
                "alpha": float(alpha),
                "node_index": int(node_index),
                "nearest_distance": float(nearest_distance),
                "component_active": int(bool(component_active or component == BASELINE_COMPONENT)),
                "delta_norm": float(np.linalg.norm(delta)),
                "natural_step_norm": float(natural_step_norm),
                "chart_norm": float(chart_norm),
                "hidden_direction_norm": float(hidden_direction_norm),
                "base_entropy": float(base_entropy),
                "steered_entropy": float(steered_entropy),
                "entropy_delta": float(steered_entropy - base_entropy),
                "kl_base_to_steered": _kl_from_logp(base_logp, steered_logp),
                "base_top_token": base_top_token,
                "base_top_logprob": float(base_top_logprob),
                "steered_top_token": steered_top_token,
                "steered_top_logprob": float(steered_top_logprob),
                "top_changed": int(base_top_id != steered_top_id),
                "next_token_id": int(next_token_id),
                "next_token": next_token,
                "next_token_base_logprob": float(base_logp[next_token_id]),
                "next_token_steered_logprob": float(steered_logp[next_token_id]),
                "next_token_logprob_gain": float(steered_logp[next_token_id] - base_logp[next_token_id]),
                "target_set": target_set,
                "target_logprob_mass_base": target_mass_base,
                "target_logprob_mass_steered": target_mass_steered,
                "target_prob_mass_base": _prob_from_logmass(target_mass_base),
                "target_prob_mass_steered": _prob_from_logmass(target_mass_steered),
                "control_logprob_mass_base": control_mass_base,
                "control_logprob_mass_steered": control_mass_steered,
                "control_prob_mass_base": _prob_from_logmass(control_mass_base),
                "control_prob_mass_steered": _prob_from_logmass(control_mass_steered),
                "target_margin_base": base_target_margin,
                "target_margin_steered": steered_target_margin,
                "target_margin_delta": target_margin_delta,
            }
        )
        ids.append(next_token_id)
        if stop_at_eos and eos_id is not None and next_token_id == int(eos_id):
            break
    return ids, step_rows


def _generated_tokens(tokenizer: Any, ids: Sequence[int], prompt_len: int) -> List[str]:
    return [_decode_token(tokenizer, int(token_id)) for token_id in ids[prompt_len:]]


def _summarize_run(
    *,
    tokenizer: Any,
    prompt_id: str,
    family: str,
    prompt: str,
    prompt_input_ids: Sequence[int],
    output_ids: Sequence[int],
    step_rows: Sequence[Dict[str, Any]],
    baseline_new_ids: Sequence[int],
) -> Dict[str, Any]:
    prompt_len = len(prompt_input_ids)
    new_ids = [int(x) for x in output_ids[prompt_len:]]
    overlap = float("nan")
    if baseline_new_ids:
        n = min(len(new_ids), len(baseline_new_ids))
        overlap = float(sum(int(a == b) for a, b in zip(new_ids[:n], baseline_new_ids[:n])) / max(n, 1))
    node_indices = [int(row["node_index"]) for row in step_rows]
    active_steps = sum(int(row.get("component_active", 0)) for row in step_rows)
    generated_text = tokenizer.decode(new_ids)
    return {
        "prompt_id": prompt_id,
        "family": family,
        "prompt": prompt,
        "layer": int(step_rows[0]["layer"]) if step_rows else "",
        "k": int(step_rows[0]["k"]) if step_rows else "",
        "seed": int(step_rows[0]["seed"]) if step_rows else "",
        "component": str(step_rows[0]["component"]) if step_rows else "",
        "alpha": float(step_rows[0]["alpha"]) if step_rows else "",
        "target_set": str(step_rows[0].get("target_set", "")) if step_rows else "",
        "generated_steps": len(new_ids),
        "active_steps": int(active_steps),
        "unique_nodes": len(set(node_indices)),
        "mean_nearest_distance": _mean_finite([float(row["nearest_distance"]) for row in step_rows]),
        "mean_delta_norm": _mean_finite([float(row["delta_norm"]) for row in step_rows]),
        "mean_kl_base_to_steered": _mean_finite([float(row["kl_base_to_steered"]) for row in step_rows]),
        "mean_entropy_delta": _mean_finite([float(row["entropy_delta"]) for row in step_rows]),
        "mean_selected_base_logprob": _mean_finite([float(row["next_token_base_logprob"]) for row in step_rows]),
        "mean_selected_steered_logprob": _mean_finite(
            [float(row["next_token_steered_logprob"]) for row in step_rows]
        ),
        "mean_selected_logprob_gain": _mean_finite([float(row["next_token_logprob_gain"]) for row in step_rows]),
        "mean_target_margin_delta": _mean_finite([float(row["target_margin_delta"]) for row in step_rows]),
        "top_changed_rate": _mean_finite([float(row["top_changed"]) for row in step_rows]),
        "baseline_token_overlap": overlap,
        "generated_text": generated_text.replace("\n", "\\n"),
        "generated_token_ids": json.dumps(new_ids),
        "generated_tokens": json.dumps(_generated_tokens(tokenizer, output_ids, prompt_len)),
    }


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path, fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


STEP_FIELDS = [
    "prompt_id",
    "family",
    "step",
    "prefix_len",
    "prompt_len",
    "layer",
    "k",
    "seed",
    "component",
    "alpha",
    "node_index",
    "nearest_distance",
    "component_active",
    "delta_norm",
    "natural_step_norm",
    "chart_norm",
    "hidden_direction_norm",
    "base_entropy",
    "steered_entropy",
    "entropy_delta",
    "kl_base_to_steered",
    "base_top_token",
    "base_top_logprob",
    "steered_top_token",
    "steered_top_logprob",
    "top_changed",
    "next_token_id",
    "next_token",
    "next_token_base_logprob",
    "next_token_steered_logprob",
    "next_token_logprob_gain",
    "target_set",
    "target_logprob_mass_base",
    "target_logprob_mass_steered",
    "target_prob_mass_base",
    "target_prob_mass_steered",
    "control_logprob_mass_base",
    "control_logprob_mass_steered",
    "control_prob_mass_base",
    "control_prob_mass_steered",
    "target_margin_base",
    "target_margin_steered",
    "target_margin_delta",
]

RUN_FIELDS = [
    "prompt_id",
    "family",
    "prompt",
    "layer",
    "k",
    "seed",
    "component",
    "alpha",
    "target_set",
    "generated_steps",
    "active_steps",
    "unique_nodes",
    "mean_nearest_distance",
    "mean_delta_norm",
    "mean_kl_base_to_steered",
    "mean_entropy_delta",
    "mean_selected_base_logprob",
    "mean_selected_steered_logprob",
    "mean_selected_logprob_gain",
    "mean_target_margin_delta",
    "top_changed_rate",
    "baseline_token_overlap",
    "generated_text",
    "generated_token_ids",
    "generated_tokens",
]


def _write_report(run_rows: Sequence[Dict[str, Any]], path: Path) -> None:
    def fmt(value: Any) -> str:
        try:
            x = float(value)
        except Exception:
            return str(value)
        if math.isnan(x):
            return "nan"
        return f"{x:.4g}"

    lines = [
        "# HLTD Closed-Loop Branch Steering Report",
        "",
        "This gate applies an HLTD branch direction at every greedy decoding step.",
        "At each step, the current last-token hidden state is projected into the",
        "initial PCA chart and matched to the nearest original branch-field node.",
        "",
        "## Runs",
        "",
        "| family | prompt | layer | k | component | alpha | mean base logp | mean gain | mean KL | target margin | overlap | generated |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in run_rows:
        lines.append(
            "| {family} | {prompt_id} | L{layer} | {k} | {component} | {alpha} | {base_logp} | {gain} | {kl} | {target} | {overlap} | `{text}` |".format(
                family=row.get("family", ""),
                prompt_id=row.get("prompt_id", ""),
                layer=row.get("layer", ""),
                k=row.get("k", ""),
                component=row.get("component", ""),
                alpha=fmt(row.get("alpha")),
                base_logp=fmt(row.get("mean_selected_base_logprob")),
                gain=fmt(row.get("mean_selected_logprob_gain")),
                kl=fmt(row.get("mean_kl_base_to_steered")),
                target=fmt(row.get("mean_target_margin_delta")),
                overlap=fmt(row.get("baseline_token_overlap")),
                text=str(row.get("generated_text", ""))[:90],
            )
        )
    lines.extend(
        [
            "",
            "## Conservative Read",
            "",
            "This is a closed-loop activation-steering gate, not proof of stable",
            "semantic control. The useful first checks are whether a branch preserves",
            "reasonable base-model logprob, changes tokens relative to baseline,",
            "and keeps nearest-node distance from exploding.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_prompt_layer_k(
    *,
    model: Any,
    tokenizer: Any,
    item: Dict[str, Any],
    device: str,
    layer: int,
    k: int,
    pca_components: int,
    max_length: int,
    generate_steps: int,
    alphas: Sequence[float],
    steering_components: Sequence[str],
    seeds: Sequence[int],
    ridge: float,
    node_ridge: float,
    min_chart_norm: float,
    normalize_hidden: bool,
    target_set: str,
    target_set_ids: Sequence[int],
    control_set_ids: Sequence[int],
    stop_at_eos: bool,
    reuse_deterministic_seeds: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    text = str(item["text"])
    prompt_id = str(item["prompt_id"])
    family = str(item["family"])
    tok_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
    if max_length is not None:
        tok_kwargs.update({"truncation": True, "max_length": max_length})
    inputs = tokenizer(text, **tok_kwargs).to(device)
    prompt_input_ids = inputs["input_ids"][0].detach().cpu().numpy().astype(int).tolist()
    if len(prompt_input_ids) < 4:
        raise ValueError(f"{prompt_id}: need at least 4 tokens for centered HLTD closed-loop steering.")

    _, hidden, _ = _forward_hidden_and_logits(model, prompt_input_ids, device=device)
    layer_count, _tokens, _dim = hidden.shape
    if layer <= 0 or layer >= layer_count:
        raise ValueError(f"--layers contains {layer}, but valid hidden layers are 1..{layer_count - 1}")

    coord = hodge.make_semantic_coordinates(
        hidden,
        method="pca",
        n_components=pca_components,
        normalize_hidden=normalize_hidden,
        random_state=0,
        verbose=False,
    )
    field = hodge.token_node_vector_field(coord.coords, layer=layer, mode="centered")
    decomp = hodge.hodge_latent_traversal_dynamics(
        field.points,
        field.vectors,
        k_neighbors=k,
        ridge=ridge,
        use_triangles=True,
    )
    base_components = _with_derived_components(hodge.hltd_component_node_vectors(decomp, ridge=node_ridge))
    natural_step_norm = _natural_centered_step_norm(hidden[layer])
    if not np.isfinite(natural_step_norm) or natural_step_norm <= 0.0:
        raise ValueError(f"{prompt_id}/L{layer}: natural hidden step norm is not positive.")

    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("closed-loop steering requires at least one seed.")

    run_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    cached_runs: Dict[Tuple[str, float], Tuple[List[int], List[Dict[str, Any]]]] = {}
    reference_seed = seed_values[0]

    if reuse_deterministic_seeds:
        deterministic_components = [
            str(component)
            for component in steering_components
            if component not in {BASELINE_COMPONENT, "random_tangent"}
        ]
        print(f"    seed {reference_seed} baseline (shared)", flush=True)
        cached_runs[(BASELINE_COMPONENT, 0.0)] = _run_closed_loop_component(
            model=model,
            tokenizer=tokenizer,
            prompt_input_ids=prompt_input_ids,
            device=device,
            coord=coord,
            field=field,
            component_vectors=base_components,
            layer=layer,
            k=k,
            component=BASELINE_COMPONENT,
            alpha=0.0,
            seed=reference_seed,
            generate_steps=generate_steps,
            natural_step_norm=natural_step_norm,
            normalize_hidden=normalize_hidden,
            min_chart_norm=min_chart_norm,
            target_set=target_set,
            target_set_ids=target_set_ids,
            control_set_ids=control_set_ids,
            stop_at_eos=stop_at_eos,
        )
        for component in deterministic_components:
            for alpha in alphas:
                print(
                    f"    seed {reference_seed} {component} alpha={float(alpha):.4g} (shared)",
                    flush=True,
                )
                cached_runs[(component, float(alpha))] = _run_closed_loop_component(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_input_ids=prompt_input_ids,
                    device=device,
                    coord=coord,
                    field=field,
                    component_vectors=base_components,
                    layer=layer,
                    k=k,
                    component=component,
                    alpha=float(alpha),
                    seed=reference_seed,
                    generate_steps=generate_steps,
                    natural_step_norm=natural_step_norm,
                    normalize_hidden=normalize_hidden,
                    min_chart_norm=min_chart_norm,
                    target_set=target_set,
                    target_set_ids=target_set_ids,
                    control_set_ids=control_set_ids,
                    stop_at_eos=stop_at_eos,
                )

    for seed in seed_values:
        component_vectors = dict(base_components)
        component_vectors["random_tangent"] = _random_tangent_component(base_components, seed=int(seed))
        if reuse_deterministic_seeds:
            baseline_ids, shared_baseline_steps = cached_runs[(BASELINE_COMPONENT, 0.0)]
            baseline_steps = _step_rows_for_seed(shared_baseline_steps, seed=seed)
        else:
            print(f"    seed {seed} baseline", flush=True)
            baseline_ids, baseline_steps = _run_closed_loop_component(
                model=model,
                tokenizer=tokenizer,
                prompt_input_ids=prompt_input_ids,
                device=device,
                coord=coord,
                field=field,
                component_vectors=component_vectors,
                layer=layer,
                k=k,
                component=BASELINE_COMPONENT,
                alpha=0.0,
                seed=seed,
                generate_steps=generate_steps,
                natural_step_norm=natural_step_norm,
                normalize_hidden=normalize_hidden,
                min_chart_norm=min_chart_norm,
                target_set=target_set,
                target_set_ids=target_set_ids,
                control_set_ids=control_set_ids,
                stop_at_eos=stop_at_eos,
            )
        baseline_new_ids = baseline_ids[len(prompt_input_ids) :]
        for row in baseline_steps:
            row.update({"prompt_id": prompt_id, "family": family})
        step_rows.extend(baseline_steps)
        run_rows.append(
            _summarize_run(
                tokenizer=tokenizer,
                prompt_id=prompt_id,
                family=family,
                prompt=text,
                prompt_input_ids=prompt_input_ids,
                output_ids=baseline_ids,
                step_rows=baseline_steps,
                baseline_new_ids=baseline_new_ids,
            )
        )

        for component in steering_components:
            if component == BASELINE_COMPONENT:
                continue
            for alpha in alphas:
                component_name = str(component)
                cache_key = (component_name, float(alpha))
                if reuse_deterministic_seeds and component_name != "random_tangent":
                    output_ids, shared_component_steps = cached_runs[cache_key]
                    component_steps = _step_rows_for_seed(shared_component_steps, seed=seed)
                else:
                    print(f"    seed {seed} {component_name} alpha={float(alpha):.4g}", flush=True)
                    output_ids, component_steps = _run_closed_loop_component(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_input_ids=prompt_input_ids,
                        device=device,
                        coord=coord,
                        field=field,
                        component_vectors=component_vectors,
                        layer=layer,
                        k=k,
                        component=component_name,
                        alpha=float(alpha),
                        seed=seed,
                        generate_steps=generate_steps,
                        natural_step_norm=natural_step_norm,
                        normalize_hidden=normalize_hidden,
                        min_chart_norm=min_chart_norm,
                        target_set=target_set,
                        target_set_ids=target_set_ids,
                        control_set_ids=control_set_ids,
                        stop_at_eos=stop_at_eos,
                    )
                for row in component_steps:
                    row.update({"prompt_id": prompt_id, "family": family})
                step_rows.extend(component_steps)
                run_rows.append(
                    _summarize_run(
                        tokenizer=tokenizer,
                        prompt_id=prompt_id,
                        family=family,
                        prompt=text,
                        prompt_input_ids=prompt_input_ids,
                        output_ids=output_ids,
                        step_rows=component_steps,
                        baseline_new_ids=baseline_new_ids,
                    )
                )
    return run_rows, step_rows, decomp.energy


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="data/hltd_prompt_suite.jsonl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", default="spiral_out_hltd_closed_loop")
    parser.add_argument("--layers", type=int, nargs="+", default=[7])
    parser.add_argument("--k", type=int, nargs="+", default=[16])
    parser.add_argument("--components", type=int, default=32, help="PCA chart dimensions")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--generate-steps", type=int, default=16)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--steering-components",
        nargs="+",
        default=["presence_plus_coexact", "coexact_minus_presence", "presence", "coexact", "random_tangent"],
    )
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--prompt-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-prompts-per-family", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--node-ridge", type=float, default=1e-4)
    parser.add_argument("--min-chart-norm", type=float, default=1e-6)
    parser.add_argument("--target-set-file", default=None)
    parser.add_argument("--target-set-key", default=None)
    parser.add_argument(
        "--exclude-prompt-target-overlap",
        action="store_true",
        help="Exclude semantic target terms that occur as whole lexeme phrases in each prompt.",
    )
    parser.add_argument("--no-normalize-hidden", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default=None,
        help="Optional dtype for loading larger local models.",
    )
    parser.add_argument("--no-stop-at-eos", action="store_true")
    parser.add_argument(
        "--no-reuse-deterministic-seeds",
        action="store_true",
        help="Recompute greedy and non-random branches for every matched-random seed.",
    )
    args = parser.parse_args(argv)

    prompts = select_prompts(
        read_suite(Path(args.suite)),
        families=args.families,
        prompt_ids=args.prompt_ids,
        limit=args.limit,
        max_prompts_per_family=args.max_prompts_per_family,
    )
    if not prompts:
        raise ValueError("No prompts matched the requested filters.")

    model_ref, model_is_local = hodge.resolve_hf_model_ref("gpt2", args.model_path)
    device = hodge.choose_device(args.device)
    local_files_only = bool(model_is_local or hodge.hf_offline_enabled())
    print(f"loading model: {model_ref}", flush=True)
    model, tokenizer = _load_model_and_tokenizer(
        model_ref,
        device=device,
        local_files_only=local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
    )
    semantic_sets = _load_semantic_target_sets(args.target_set_file)
    seeds = list(args.seeds or [args.seed])
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_run_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    manifest_runs: List[Dict[str, Any]] = []
    start = time.perf_counter()
    run_count = 0
    for prompt_no, item in enumerate(prompts, start=1):
        prompt_id = str(item["prompt_id"])
        family = str(item["family"])
        target_set_source = _semantic_set_key(
            semantic_sets,
            requested_key=args.target_set_key,
            prompt_id=prompt_id,
            family=family,
        )
        effective_spec = semantic_sets.get(target_set_source, {"target": [], "control": []})
        excluded_prompt_target_terms: List[str] = []
        target_set = target_set_source
        effective_semantic_sets = semantic_sets
        if args.exclude_prompt_target_overlap and target_set_source:
            effective_spec, excluded_prompt_target_terms = _prompt_heldout_semantic_target_set(
                semantic_sets,
                target_set_source,
                str(item["text"]),
            )
            target_set = f"{target_set_source}__prompt_heldout"
            effective_semantic_sets = {target_set: effective_spec}
        target_set_ids, control_set_ids = _semantic_token_ids(tokenizer, effective_semantic_sets, target_set)
        print(f"[prompt {prompt_no}/{len(prompts)}] {family}/{prompt_id}", flush=True)
        if excluded_prompt_target_terms:
            print(f"  excluded prompt-overlap targets: {', '.join(excluded_prompt_target_terms)}", flush=True)
        for layer in args.layers:
            for k in args.k:
                run_count += 1
                print(f"  [closed-loop {run_count}] L{layer} k{k}", flush=True)
                run_rows, step_rows, energy = run_prompt_layer_k(
                    model=model,
                    tokenizer=tokenizer,
                    item=item,
                    device=device,
                    layer=int(layer),
                    k=int(k),
                    pca_components=int(args.components),
                    max_length=int(args.max_length),
                    generate_steps=int(args.generate_steps),
                    alphas=args.alphas,
                    steering_components=args.steering_components,
                    seeds=seeds,
                    ridge=float(args.ridge),
                    node_ridge=float(args.node_ridge),
                    min_chart_norm=float(args.min_chart_norm),
                    normalize_hidden=not args.no_normalize_hidden,
                    target_set=target_set,
                    target_set_ids=target_set_ids,
                    control_set_ids=control_set_ids,
                    stop_at_eos=not args.no_stop_at_eos,
                    reuse_deterministic_seeds=not args.no_reuse_deterministic_seeds,
                )
                all_run_rows.extend(run_rows)
                all_step_rows.extend(step_rows)
                manifest_runs.append(
                    {
                        "prompt_id": prompt_id,
                        "family": family,
                        "layer": int(layer),
                        "k": int(k),
                        "target_set": target_set,
                        "target_set_source": target_set_source,
                        "effective_target_terms": list(effective_spec.get("target", [])),
                        "control_terms": list(effective_spec.get("control", [])),
                        "excluded_prompt_target_terms": excluded_prompt_target_terms,
                        "target_token_count": len(set(target_set_ids)),
                        "control_token_count": len(set(control_set_ids)),
                        "hltd_energy": energy,
                    }
                )

    metrics_path = output_root / "closed_loop_metrics.csv"
    steps_path = output_root / "closed_loop_steps.csv"
    report_path = output_root / "closed_loop_report.md"
    manifest_path = output_root / "closed_loop_manifest.json"
    _write_csv(all_run_rows, metrics_path, RUN_FIELDS)
    _write_csv(all_step_rows, steps_path, STEP_FIELDS)
    _write_report(all_run_rows, report_path)
    manifest = {
        "model_ref": model_ref,
        "suite": str(args.suite),
        "layers": [int(x) for x in args.layers],
        "k": [int(x) for x in args.k],
        "components": int(args.components),
        "generate_steps": int(args.generate_steps),
        "alphas": [float(x) for x in args.alphas],
        "steering_components": [str(x) for x in args.steering_components],
        "seeds": [int(x) for x in seeds],
        "seed_role": "matched_random_tangent_stratum",
        "reuse_deterministic_seeds": not args.no_reuse_deterministic_seeds,
        "target_set_file": str(args.target_set_file) if args.target_set_file else "",
        "target_set_key": str(args.target_set_key) if args.target_set_key else "",
        "exclude_prompt_target_overlap": bool(args.exclude_prompt_target_overlap),
        "target_sets_used": sorted({str(run["target_set"]) for run in manifest_runs}),
        "prompt_ids": [str(x) for x in args.prompt_ids] if args.prompt_ids else [],
        "families": [str(x) for x in args.families] if args.families else [],
        "runs": manifest_runs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - start
    print(f"saved metrics: {metrics_path}")
    print(f"saved steps: {steps_path}")
    print(f"saved report: {report_path}")
    print(f"closed-loop complete: {run_count} prompt/layer/k runs in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
