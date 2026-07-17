#!/usr/bin/env python3
"""Run one-step HLTD causal steering on a local Hugging Face causal LM."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spiral_hodge as hodge


STEERING_PROVENANCE_FIELDS = [
    "complex_mode",
    "hodge_solver",
    "betti_1_fraction_target",
    "betti_1_fraction",
    "betti_1_fraction_abs_error",
    "cycle_rank",
    "triangle_rank",
    "triangle_count",
    "triangle_count_full",
    "triangle_fill_actual",
    "filtration_radius_scale_actual",
    "hodge_exact_ratio",
    "hodge_coexact_ratio",
    "hodge_harmonic_ratio",
]
RANDOM_TANGENT_REFERENCE = "max_full_branch_node_speed"


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) - _logsumexp(np.asarray(x, dtype=np.float64))


def _entropy_from_logp(logp: np.ndarray) -> float:
    p = np.exp(logp)
    return float(-np.sum(p * logp))


def _kl_from_logp(p_logp: np.ndarray, q_logp: np.ndarray) -> float:
    p = np.exp(p_logp)
    return float(np.sum(p * (p_logp - q_logp)))


def _js_from_logp(a_logp: np.ndarray, b_logp: np.ndarray) -> float:
    a = np.exp(a_logp)
    b = np.exp(b_logp)
    m = 0.5 * (a + b)
    m_logp = np.log(np.maximum(m, 1e-300))
    return 0.5 * _kl_from_logp(a_logp, m_logp) + 0.5 * _kl_from_logp(b_logp, m_logp)


def _decode_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)])
    return text.replace("\n", "\\n")


def _top_token(logp: np.ndarray, tokenizer: Any) -> tuple[int, str, float]:
    idx = int(np.argmax(logp))
    return idx, _decode_token(tokenizer, idx), float(logp[idx])


def _target_token_id(tokenizer: Any, target_text: Optional[str]) -> Optional[int]:
    if target_text is None or not str(target_text).strip():
        return None
    ids = tokenizer(str(target_text), add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    return int(ids[0])


def _load_semantic_target_sets(path: Optional[str]) -> Dict[str, Dict[str, List[str]]]:
    if path is None or not str(path).strip():
        return {}
    with Path(path).expanduser().open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Semantic target set file must contain an object: {path}")
    out: Dict[str, Dict[str, List[str]]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Semantic target set {key!r} must be an object.")
        target = [str(x) for x in value.get("target", [])]
        control = [str(x) for x in value.get("control", [])]
        if not target:
            raise ValueError(f"Semantic target set {key!r} must include non-empty 'target'.")
        out[str(key)] = {"target": target, "control": control}
    return out


def _semantic_set_key(
    target_sets: Dict[str, Dict[str, List[str]]],
    *,
    requested_key: Optional[str],
    prompt_id: str,
    family: Optional[str] = None,
) -> str:
    if not target_sets:
        return ""
    if requested_key:
        if requested_key not in target_sets:
            raise ValueError(f"Unknown semantic target set {requested_key!r}. Valid: {', '.join(sorted(target_sets))}")
        return requested_key
    for key in [prompt_id, family, "default"]:
        if key and key in target_sets:
            return str(key)
    return ""


def _first_token_ids(tokenizer: Any, terms: Sequence[str]) -> List[int]:
    ids: set[int] = set()
    for term in terms:
        text = str(term)
        variants = [text]
        if text and not text.startswith(" "):
            variants.append(" " + text)
        for variant in variants:
            encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if encoded:
                ids.add(int(encoded[0]))
    return sorted(ids)


def _semantic_token_ids(
    tokenizer: Any,
    target_sets: Dict[str, Dict[str, List[str]]],
    key: str,
) -> tuple[List[int], List[int]]:
    if not key:
        return [], []
    spec = target_sets[key]
    return _first_token_ids(tokenizer, spec.get("target", [])), _first_token_ids(tokenizer, spec.get("control", []))


def _with_derived_components(component_vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {str(key): np.asarray(value, dtype=np.float64) for key, value in component_vectors.items()}
    if "presence" in out and "coexact" in out:
        out.setdefault("presence_plus_coexact", out["presence"] + out["coexact"])
        out.setdefault("coexact_minus_presence", out["coexact"] - out["presence"])
        out.setdefault("presence_minus_coexact", out["presence"] - out["coexact"])
    if "presence" in out:
        out.setdefault("negative_presence", -out["presence"])
    if "coexact" in out:
        out.setdefault("negative_coexact", -out["coexact"])
    if "semantic_flow" in out and "presence" in out:
        out.setdefault("semantic_flow_minus_presence", out["semantic_flow"] - out["presence"])
        out.setdefault("presence_plus_semantic_flow", out["presence"] + out["semantic_flow"])
    return out


def _random_tangent_component(
    component_vectors: Dict[str, np.ndarray],
    *,
    seed: int,
) -> np.ndarray:
    """Draw chart directions while retaining a nonzero local field-speed scale."""

    if "full" not in component_vectors:
        raise ValueError("Random tangent construction requires the full node field.")
    full = np.asarray(component_vectors["full"], dtype=np.float64)
    reference_norms = np.linalg.norm(full, axis=1)
    branch_norms = [
        np.linalg.norm(np.asarray(component_vectors[key], dtype=np.float64), axis=1)
        for key in ("exact", "coexact", "harmonic")
        if key in component_vectors
    ]
    if branch_norms:
        reference_norms = np.maximum(reference_norms, np.max(np.vstack(branch_norms), axis=0))
    positive = reference_norms[np.isfinite(reference_norms) & (reference_norms > 1e-12)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    reference_norms = np.where(
        np.isfinite(reference_norms) & (reference_norms > 1e-12),
        reference_norms,
        fallback,
    )

    rng = np.random.default_rng(int(seed))
    random_chart = rng.normal(size=full.shape)
    random_norms = np.linalg.norm(random_chart, axis=1, keepdims=True)
    return random_chart / np.maximum(random_norms, 1e-12) * reference_norms[:, None]


def _steering_decomposition(
    field: Any,
    *,
    k_neighbors: int,
    ridge: float,
    complex_mode: str,
    target_betti_1_fraction: float,
) -> tuple[hodge.HLTDDecomposition, Dict[str, Any]]:
    if complex_mode == "matched_betti":
        return hodge.hodge_latent_traversal_dynamics_matched_betti(
            field.points,
            field.vectors,
            k_neighbors=k_neighbors,
            target_betti_1_fraction=target_betti_1_fraction,
        )
    if complex_mode != "full_clique":
        raise ValueError(f"Unknown complex mode: {complex_mode}")

    decomp = hodge.hodge_latent_traversal_dynamics(
        field.points,
        field.vectors,
        k_neighbors=k_neighbors,
        ridge=ridge,
        use_triangles=True,
    )
    topology: Dict[str, Any] = {
        "complex_mode": "full_clique",
        "hodge_solver": "ridge",
        "betti_1_fraction_target": float("nan"),
        "betti_1_fraction": float("nan"),
        "betti_1_fraction_abs_error": float("nan"),
        "cycle_rank": float("nan"),
        "triangle_rank": float("nan"),
        "triangle_count": float(len(decomp.triangles)),
        "triangle_count_full": float(len(decomp.triangles)),
        "triangle_fill_actual": 1.0,
        "filtration_radius_scale_actual": float("nan"),
    }
    return decomp, topology


def _steering_provenance(
    decomp: hodge.HLTDDecomposition,
    topology: Dict[str, Any],
) -> Dict[str, Any]:
    out = {key: topology.get(key, "") for key in STEERING_PROVENANCE_FIELDS}
    out.update(
        {
            "hodge_exact_ratio": float(decomp.energy.get("exact_ratio", float("nan"))),
            "hodge_coexact_ratio": float(decomp.energy.get("coexact_ratio", float("nan"))),
            "hodge_harmonic_ratio": float(decomp.energy.get("harmonic_ratio", float("nan"))),
        }
    )
    return out


def _logprob_mass(logp: np.ndarray, token_ids: Sequence[int]) -> float:
    if not token_ids:
        return float("nan")
    valid = [int(i) for i in token_ids if 0 <= int(i) < logp.shape[0]]
    if not valid:
        return float("nan")
    return _logsumexp(logp[np.asarray(sorted(set(valid)), dtype=int)])


def _prob_from_logmass(value: float) -> float:
    return float(np.exp(value)) if math.isfinite(float(value)) else float("nan")


def _component_delta_row(
    *,
    prompt_id: str,
    layer: int,
    k: int,
    seed: int,
    token_selector: str,
    selector_component: str,
    node_index: int,
    token_index: int,
    token_count: int,
    token: str,
    next_token_id: int,
    component: str,
    alpha: float,
    delta_norm: float,
    natural_step_norm: float,
    chart_norm: float,
    hidden_direction_norm: float,
    component_active: bool,
    base_logits: np.ndarray,
    steered_logits: np.ndarray,
    tokenizer: Any,
    target_id: Optional[int],
    target_set: str = "",
    target_set_ids: Optional[Sequence[int]] = None,
    control_set_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    base_logp = _log_softmax(base_logits)
    steered_logp = _log_softmax(steered_logits)
    base_entropy = _entropy_from_logp(base_logp)
    steered_entropy = _entropy_from_logp(steered_logp)
    base_top_id, base_top_token, base_top_logp = _top_token(base_logp, tokenizer)
    steered_top_id, steered_top_token, steered_top_logp = _top_token(steered_logp, tokenizer)
    shift = steered_logp - base_logp
    if float(delta_norm) <= 1e-12:
        shift_token = "no-op"
        shift_delta = 0.0
    else:
        shift_id = int(np.argmax(shift))
        shift_token = _decode_token(tokenizer, shift_id)
        shift_delta = float(shift[shift_id])
    target_mass_base = _logprob_mass(base_logp, target_set_ids or [])
    target_mass_steered = _logprob_mass(steered_logp, target_set_ids or [])
    control_mass_base = _logprob_mass(base_logp, control_set_ids or [])
    control_mass_steered = _logprob_mass(steered_logp, control_set_ids or [])
    target_mass_delta = target_mass_steered - target_mass_base
    control_mass_delta = control_mass_steered - control_mass_base
    semantic_margin_delta = target_mass_delta - control_mass_delta

    row: Dict[str, Any] = {
        "prompt_id": prompt_id,
        "layer": int(layer),
        "k": int(k),
        "seed": int(seed),
        "token_selector": token_selector,
        "selector_component": selector_component,
        "node_index": int(node_index),
        "token_index": int(token_index),
        "token_count": int(token_count),
        "token": token,
        "next_token": _decode_token(tokenizer, next_token_id),
        "component": component,
        "alpha": float(alpha),
        "delta_norm": float(delta_norm),
        "natural_step_norm": float(natural_step_norm),
        "chart_norm": float(chart_norm),
        "hidden_direction_norm": float(hidden_direction_norm),
        "component_active": int(bool(component_active)),
        "base_entropy": base_entropy,
        "steered_entropy": steered_entropy,
        "entropy_delta": steered_entropy - base_entropy,
        "kl_base_to_steered": _kl_from_logp(base_logp, steered_logp),
        "kl_steered_to_base": _kl_from_logp(steered_logp, base_logp),
        "js_divergence": _js_from_logp(base_logp, steered_logp),
        "base_top_token": base_top_token,
        "base_top_logprob": base_top_logp,
        "steered_top_token": steered_top_token,
        "steered_top_logprob": steered_top_logp,
        "top_changed": int(base_top_id != steered_top_id),
        "top_shift_token": shift_token,
        "top_shift_logprob_delta": shift_delta,
        "next_token_logprob_base": float(base_logp[next_token_id]),
        "next_token_logprob_steered": float(steered_logp[next_token_id]),
        "next_token_logprob_delta": float(shift[next_token_id]),
        "target_token": "",
        "target_logprob_base": float("nan"),
        "target_logprob_steered": float("nan"),
        "target_logprob_delta": float("nan"),
        "target_set": target_set,
        "target_set_size": len(set(target_set_ids or [])),
        "control_set_size": len(set(control_set_ids or [])),
        "target_logprob_mass_base": target_mass_base,
        "target_logprob_mass_steered": target_mass_steered,
        "target_logprob_mass_delta": target_mass_delta,
        "target_prob_mass_base": _prob_from_logmass(target_mass_base),
        "target_prob_mass_steered": _prob_from_logmass(target_mass_steered),
        "target_prob_mass_delta": _prob_from_logmass(target_mass_steered) - _prob_from_logmass(target_mass_base),
        "control_logprob_mass_base": control_mass_base,
        "control_logprob_mass_steered": control_mass_steered,
        "control_logprob_mass_delta": control_mass_delta,
        "control_prob_mass_base": _prob_from_logmass(control_mass_base),
        "control_prob_mass_steered": _prob_from_logmass(control_mass_steered),
        "control_prob_mass_delta": _prob_from_logmass(control_mass_steered) - _prob_from_logmass(control_mass_base),
        "semantic_margin_delta": semantic_margin_delta,
        "semantic_prob_margin_delta": (
            _prob_from_logmass(target_mass_steered)
            - _prob_from_logmass(target_mass_base)
            - _prob_from_logmass(control_mass_steered)
            + _prob_from_logmass(control_mass_base)
        ),
    }
    if target_id is not None:
        row["target_token"] = _decode_token(tokenizer, target_id)
        row["target_logprob_base"] = float(base_logp[target_id])
        row["target_logprob_steered"] = float(steered_logp[target_id])
        row["target_logprob_delta"] = float(shift[target_id])
    return row


def _load_model_and_tokenizer(
    model_ref: str,
    *,
    device: str,
    local_files_only: bool,
    trust_remote_code: bool,
    torch_dtype: str | None = None,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model_kwargs: Dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if torch_dtype:
        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(f"Unsupported torch dtype {torch_dtype!r}.")
        model_kwargs["torch_dtype"] = dtype_map[torch_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        **model_kwargs,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _block_for_hidden_layer(model: Any, layer: int) -> Any:
    if layer <= 0:
        raise ValueError("Steering layer must be >= 1; layer 0 is the embedding output.")
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None) if transformer is not None else None
    if blocks is None:
        # Llama, Mistral, Gemma-like decoder-only models expose decoder blocks
        # through model.model.layers. Hidden-state index 1 corresponds to block 0.
        decoder = getattr(model, "model", None)
        blocks = getattr(decoder, "layers", None) if decoder is not None else None
    if blocks is None:
        raise ValueError(
            "Could not locate decoder blocks. Supported layouts include "
            "GPT-2 style model.transformer.h and Llama/Mistral style model.model.layers."
        )
    block_idx = int(layer) - 1
    if block_idx < 0 or block_idx >= len(blocks):
        raise ValueError(f"Layer {layer} maps to block {block_idx}, but model has {len(blocks)} blocks.")
    return blocks[block_idx]


def _logits_with_delta(
    model: Any,
    inputs: Dict[str, Any],
    *,
    layer: int,
    token_index: int,
    delta: np.ndarray,
) -> np.ndarray:
    return _logits_with_deltas(model, inputs, layer=layer, token_index=token_index, deltas=np.asarray([delta]))[0]


def _logits_with_deltas(
    model: Any,
    inputs: Dict[str, Any],
    *,
    layer: int,
    token_index: int,
    deltas: np.ndarray,
) -> np.ndarray:
    import torch

    block = _block_for_hidden_layer(model, layer)
    delta_array = np.asarray(deltas, dtype=np.float32)
    if delta_array.ndim != 2:
        raise ValueError(f"deltas must be [batch, dim], got shape {delta_array.shape}")
    delta_tensor = torch.as_tensor(delta_array, dtype=torch.float32, device="cpu")
    batch = int(delta_array.shape[0])
    batch_inputs: Dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            reps = [batch] + [1] * (value.ndim - 1)
            batch_inputs[key] = value.repeat(*reps)
        else:
            batch_inputs[key] = value

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0].clone()
            add = delta_tensor.to(device=hidden.device, dtype=hidden.dtype)
            hidden[:, token_index, :] = hidden[:, token_index, :] + add
            return (hidden,) + output[1:]
        hidden = output.clone()
        add = delta_tensor.to(device=hidden.device, dtype=hidden.dtype)
        hidden[:, token_index, :] = hidden[:, token_index, :] + add
        return hidden

    handle = block.register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model(**batch_inputs, return_dict=True)
    finally:
        handle.remove()
    return outputs.logits[:, token_index].detach().float().cpu().numpy()


def _natural_centered_step_norm(hidden_layer: np.ndarray) -> float:
    H = np.asarray(hidden_layer, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] < 3:
        return float("nan")
    steps = 0.5 * (H[2:] - H[:-2])
    norms = np.linalg.norm(steps, axis=1)
    finite = norms[np.isfinite(norms)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _token_centers(field: hodge.NodeVectorFieldBundle) -> List[int]:
    return [int((a + b) // 2) for a, b in field.token_edges]


def _position_bin_for_token(token_index: int, centers: Sequence[int], position_bin_count: int) -> int:
    if position_bin_count <= 0:
        raise ValueError("--position-bin-count must be positive.")
    if not centers:
        return 0
    denom = max(max(int(x) for x in centers) + 1, 1)
    frac = min(max(float(token_index) / float(denom), 0.0), 1.0)
    return min(int(frac * int(position_bin_count)), int(position_bin_count) - 1)


def _node_indices_for_position_bins(
    centers: Sequence[int],
    *,
    position_bins: Sequence[int],
    position_bin_count: int,
) -> List[int]:
    if not position_bins:
        raise ValueError("token_selector='position_bin' requires --position-bins.")
    if position_bin_count <= 0:
        raise ValueError("--position-bin-count must be positive.")
    requested = sorted({int(value) for value in position_bins})
    invalid = [value for value in requested if value < 0 or value >= int(position_bin_count)]
    if invalid:
        raise ValueError(f"position bins must be in [0, {int(position_bin_count) - 1}], got {invalid}.")
    if not centers:
        return [0]

    denom = max(max(int(x) for x in centers) + 1, 1)
    selected: set[int] = set()
    for bin_index in requested:
        matches = [
            idx
            for idx, center in enumerate(centers)
            if _position_bin_for_token(int(center), centers, position_bin_count) == bin_index
        ]
        if matches:
            selected.update(matches)
            continue
        target_frac = (float(bin_index) + 0.5) / float(position_bin_count)
        selected.add(
            min(
                range(len(centers)),
                key=lambda idx: abs((float(centers[idx]) / float(denom)) - target_frac),
            )
        )
    return sorted(selected)


def _select_node_indices(
    component_vectors: Dict[str, np.ndarray],
    *,
    token_selector: str,
    selector_component: str,
    token_indices: Sequence[int],
    position_bins: Sequence[int],
    position_bin_count: int,
    field: hodge.NodeVectorFieldBundle,
) -> List[int]:
    centers = _token_centers(field)
    if token_selector == "fixed":
        if not token_indices:
            raise ValueError("token_selector='fixed' requires --token-index or --token-indices.")
        selected: List[int] = []
        for token_index in token_indices:
            if int(token_index) not in centers:
                raise ValueError(f"token_index={token_index} is not an interior node for centered HLTD.")
            selected.append(centers.index(int(token_index)))
        return sorted(set(selected))

    if token_selector == "all_interior":
        return list(range(len(field.token_edges)))

    if token_selector == "position_bin":
        return _node_indices_for_position_bins(
            centers,
            position_bins=position_bins,
            position_bin_count=position_bin_count,
        )

    if token_selector == "middle":
        if not centers:
            return [0]
        middle_token = 0.5 * (min(centers) + max(centers))
        return [min(range(len(centers)), key=lambda idx: abs(centers[idx] - middle_token))]

    if token_selector != "max_component":
        raise ValueError(f"Unknown token selector {token_selector!r}.")

    if selector_component not in component_vectors:
        valid = ", ".join(sorted(component_vectors))
        raise ValueError(f"Unknown selector component {selector_component!r}. Valid: {valid}")
    norms = np.linalg.norm(component_vectors[selector_component], axis=1)
    if not np.any(np.isfinite(norms)):
        return [0]
    return [int(np.nanargmax(norms))]


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fields = [
        "prompt_id",
        "layer",
        "k",
        *STEERING_PROVENANCE_FIELDS,
        "seed",
        "random_tangent_reference",
        "token_selector",
        "selector_component",
        "node_index",
        "token_index",
        "token_count",
        "token",
        "next_token",
        "component",
        "alpha",
        "delta_norm",
        "natural_step_norm",
        "chart_norm",
        "hidden_direction_norm",
        "component_active",
        "base_entropy",
        "steered_entropy",
        "entropy_delta",
        "kl_base_to_steered",
        "kl_steered_to_base",
        "js_divergence",
        "base_top_token",
        "base_top_logprob",
        "steered_top_token",
        "steered_top_logprob",
        "top_changed",
        "top_shift_token",
        "top_shift_logprob_delta",
        "next_token_logprob_base",
        "next_token_logprob_steered",
        "next_token_logprob_delta",
        "target_token",
        "target_logprob_base",
        "target_logprob_steered",
        "target_logprob_delta",
        "target_set",
        "target_set_size",
        "control_set_size",
        "target_logprob_mass_base",
        "target_logprob_mass_steered",
        "target_logprob_mass_delta",
        "target_prob_mass_base",
        "target_prob_mass_steered",
        "target_prob_mass_delta",
        "control_logprob_mass_base",
        "control_logprob_mass_steered",
        "control_logprob_mass_delta",
        "control_prob_mass_base",
        "control_prob_mass_steered",
        "control_prob_mass_delta",
        "semantic_margin_delta",
        "semantic_prob_margin_delta",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_report(
    rows: Sequence[Dict[str, Any]],
    path: Path,
    *,
    prompt: str,
    model_ref: str,
    layer: int,
    k_neighbors: int,
    hltd_energy: Dict[str, float],
    hltd_topology: Optional[Dict[str, Any]] = None,
) -> None:
    def fmt(value: Any) -> str:
        try:
            x = float(value)
        except Exception:
            return str(value)
        if math.isnan(x):
            return "nan"
        return f"{x:.6g}"

    strongest = max(rows, key=lambda r: float(r.get("kl_base_to_steered", 0.0))) if rows else None
    seeds = sorted({int(r["seed"]) for r in rows}) if rows else []
    selectors = sorted({str(r["token_selector"]) for r in rows}) if rows else []
    token_indices = sorted({int(r["token_index"]) for r in rows}) if rows else []
    token_preview = ", ".join(str(x) for x in token_indices[:16])
    if len(token_indices) > 16:
        token_preview += " ..."
    topology = hltd_topology or {}
    lines = [
        "# HLTD One-Step Steering Report",
        "",
        "## Run",
        "",
        f"- model: `{model_ref}`",
        f"- layer: `{layer}`",
        f"- k_neighbors: `{k_neighbors}`",
        f"- complex mode: `{topology.get('complex_mode', 'full_clique')}`",
        f"- Hodge solver: `{topology.get('hodge_solver', 'ridge')}`",
        f"- target Betti-1 fraction: `{fmt(topology.get('betti_1_fraction_target'))}`",
        f"- actual Betti-1 fraction: `{fmt(topology.get('betti_1_fraction'))}`",
        f"- triangle rank / cycle rank: `{fmt(topology.get('triangle_rank'))} / {fmt(topology.get('cycle_rank'))}`",
        f"- random tangent seeds: `{', '.join(str(x) for x in seeds)}`",
        f"- random tangent reference: `{RANDOM_TANGENT_REFERENCE}`",
        f"- token selectors: `{', '.join(selectors)}`",
        f"- selected token indices: `{token_preview}`",
        f"- prompt: `{prompt}`",
        "",
        "## HLTD Energy",
        "",
        "| component | ratio | energy |",
        "| --- | ---: | ---: |",
        f"| exact / presence | {fmt(hltd_energy.get('exact_ratio'))} | {fmt(hltd_energy.get('exact'))} |",
        f"| coexact / local swirl | {fmt(hltd_energy.get('coexact_ratio'))} | {fmt(hltd_energy.get('coexact'))} |",
        f"| harmonic / global loop | {fmt(hltd_energy.get('harmonic_ratio'))} | {fmt(hltd_energy.get('harmonic'))} |",
        f"| semantic flow | {fmt(hltd_energy.get('semantic_flow_ratio'))} | {fmt(hltd_energy.get('semantic_flow'))} |",
        "",
        "## Steering Metrics",
        "",
        "| seed | selector | token | component | active | alpha | KL(base||steered) | entropy delta | next-token delta | target mass delta | semantic margin | steered top |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {selector} | {token_index} | {component} | {active} | {alpha} | {kl} | {entropy} | {next_delta} | {target_mass} | {semantic_margin} | `{top}` |".format(
                seed=row["seed"],
                selector=row["token_selector"],
                token_index=row["token_index"],
                component=row["component"],
                active=row["component_active"],
                alpha=fmt(row["alpha"]),
                kl=fmt(row["kl_base_to_steered"]),
                entropy=fmt(row["entropy_delta"]),
                next_delta=fmt(row["next_token_logprob_delta"]),
                target_mass=fmt(row.get("target_logprob_mass_delta")),
                semantic_margin=fmt(row.get("semantic_margin_delta")),
                top=row["steered_top_token"],
            )
        )
    lines.extend(["", "## Conservative Read", ""])
    if strongest:
        lines.append(
            "The strongest one-step logit movement in this run was "
            f"`{strongest['component']}` at alpha `{fmt(strongest['alpha'])}` "
            f"with KL `{fmt(strongest['kl_base_to_steered'])}`."
        )
    lines.append("Rows with `active = 0` are no-op controls because the selected component norm was below the gate.")
    lines.append(
        "This gate only tests immediate next-token distribution movement. It does "
        "not yet establish multi-step semantic drift or fluency preservation."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--text", required=True, help="Prompt/text to analyze")
    parser.add_argument("--prompt-id", default="manual")
    parser.add_argument("--output-dir", default="spiral_out_hltd_steering")
    parser.add_argument("--layer", type=int, default=5, help="Hidden-state layer to steer; GPT-2 layer 0 is embeddings")
    parser.add_argument("--k", type=int, default=16, help="HLTD kNN graph degree")
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
    parser.add_argument("--components", type=int, default=32, help="PCA chart dimensions")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument(
        "--steering-components",
        nargs="+",
        default=["presence", "coexact", "semantic_flow", "harmonic", "random_tangent"],
        help="Components to test",
    )
    parser.add_argument("--selector-component", default="coexact", help="Component whose max node norm selects token")
    parser.add_argument("--token-index", type=int, default=None, help="Optional interior token index to steer")
    parser.add_argument(
        "--token-indices",
        type=int,
        nargs="+",
        default=None,
        help="Interior token indices for token_selector='fixed'",
    )
    parser.add_argument(
        "--token-selectors",
        nargs="+",
        default=None,
        choices=["max_component", "middle", "fixed", "all_interior", "position_bin"],
        help="Token selection rules to evaluate",
    )
    parser.add_argument(
        "--position-bins",
        type=int,
        nargs="+",
        default=None,
        help="Normalized token-position bins for token_selector='position_bin'",
    )
    parser.add_argument("--position-bin-count", type=int, default=12)
    parser.add_argument("--target-text", default=None, help="Optional next-token target text; first token is used")
    parser.add_argument("--target-set-file", default=None, help="Optional JSON file with semantic target/control sets")
    parser.add_argument("--target-set-key", default=None, help="Semantic target set key; defaults to prompt_id/family/default")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default=None,
        help="Optional dtype for loading larger local models.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random tangent seeds")
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--node-ridge", type=float, default=1e-4)
    parser.add_argument(
        "--min-chart-norm",
        type=float,
        default=1e-6,
        help="Treat component node vectors below this PCA-chart norm as inactive no-op directions",
    )
    parser.add_argument("--no-normalize-hidden", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    if not 0.0 <= float(args.target_betti_1_fraction) <= 1.0:
        parser.error("--target-betti-1-fraction must be between 0 and 1")

    import torch

    model_ref, model_is_local = hodge.resolve_hf_model_ref("gpt2", args.model_path)
    device = hodge.choose_device(args.device)
    local_files_only = bool(model_is_local or hodge.hf_offline_enabled())
    model, tokenizer = _load_model_and_tokenizer(
        model_ref,
        device=device,
        local_files_only=local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
    )

    tok_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
    if args.max_length is not None:
        tok_kwargs.update({"truncation": True, "max_length": args.max_length})
    inputs = tokenizer(args.text, **tok_kwargs).to(device)
    input_ids = inputs["input_ids"][0].detach().cpu().numpy()
    if input_ids.size < 4:
        raise ValueError("Need at least 4 tokens for centered HLTD steering.")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = torch.stack(outputs.hidden_states, dim=0)[:, 0].detach().float().cpu().numpy()
    layers, tokens, dim = hidden.shape
    if args.layer <= 0 or args.layer >= layers:
        raise ValueError(f"--layer must be between 1 and {layers - 1}, got {args.layer}")

    coord = hodge.make_semantic_coordinates(
        hidden,
        method="pca",
        n_components=args.components,
        normalize_hidden=not args.no_normalize_hidden,
        random_state=args.seed,
        verbose=True,
    )
    field = hodge.token_node_vector_field(coord.coords, layer=args.layer, mode="centered")
    decomp, topology = _steering_decomposition(
        field,
        k_neighbors=args.k,
        ridge=args.ridge,
        complex_mode=args.complex_mode,
        target_betti_1_fraction=args.target_betti_1_fraction,
    )
    component_vectors = _with_derived_components(hodge.hltd_component_node_vectors(decomp, ridge=args.node_ridge))
    provenance = _steering_provenance(decomp, topology)

    natural_step_norm = _natural_centered_step_norm(hidden[args.layer])
    if not np.isfinite(natural_step_norm) or natural_step_norm <= 0.0:
        raise ValueError("Could not compute a positive natural hidden step norm.")

    token_indices: List[int] = []
    if args.token_indices is not None:
        token_indices.extend(int(x) for x in args.token_indices)
    if args.token_index is not None:
        token_indices.append(int(args.token_index))
    token_indices = sorted(set(token_indices))
    position_bins = sorted(set(int(x) for x in (args.position_bins or [])))

    default_selectors = ["fixed"] if token_indices else (["position_bin"] if position_bins else ["max_component"])
    token_selectors = list(args.token_selectors or default_selectors)
    random_seeds = list(args.seeds or [args.seed])
    target_id = _target_token_id(tokenizer, args.target_text)
    semantic_sets = _load_semantic_target_sets(args.target_set_file)
    target_set = _semantic_set_key(
        semantic_sets,
        requested_key=args.target_set_key,
        prompt_id=args.prompt_id,
    )
    target_set_ids, control_set_ids = _semantic_token_ids(tokenizer, semantic_sets, target_set)
    rows: List[Dict[str, Any]] = []
    for seed in random_seeds:
        seeded_components = dict(component_vectors)
        seeded_components["random_tangent"] = _random_tangent_component(
            component_vectors,
            seed=int(seed),
        )

        for token_selector in token_selectors:
            node_indices = _select_node_indices(
                seeded_components,
                token_selector=token_selector,
                selector_component=args.selector_component,
                token_indices=token_indices,
                position_bins=position_bins,
                position_bin_count=args.position_bin_count,
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

                for component in args.steering_components:
                    if component not in seeded_components:
                        valid = ", ".join(sorted(seeded_components))
                        raise ValueError(f"Unknown steering component {component!r}. Valid: {valid}")
                    chart_vec = np.asarray(seeded_components[component][node_index], dtype=np.float64)
                    hidden_vec = hodge.pca_chart_vectors_to_hidden(chart_vec, coord.reducer)
                    hidden_norm = float(np.linalg.norm(hidden_vec))
                    chart_norm = float(np.linalg.norm(chart_vec))
                    component_active = bool(chart_norm >= float(args.min_chart_norm) and hidden_norm > 0.0)
                    if component_active:
                        direction = hidden_vec / hidden_norm
                    else:
                        direction = np.zeros(dim, dtype=np.float64)

                    for alpha in args.alphas:
                        delta = float(alpha) * natural_step_norm * direction
                        steered_logits = _logits_with_delta(
                            model,
                            inputs,
                            layer=args.layer,
                            token_index=token_index,
                            delta=delta,
                        )
                    row = _component_delta_row(
                        prompt_id=args.prompt_id,
                        layer=args.layer,
                        k=args.k,
                        seed=int(seed),
                        token_selector=token_selector,
                        selector_component=args.selector_component,
                        node_index=node_index,
                        token_index=token_index,
                        token_count=tokens,
                        token=token,
                        next_token_id=next_token_id,
                        component=component,
                        alpha=float(alpha),
                        delta_norm=float(np.linalg.norm(delta)),
                        natural_step_norm=natural_step_norm,
                        chart_norm=chart_norm,
                        hidden_direction_norm=hidden_norm,
                        component_active=component_active,
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

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "steering_metrics.csv"
    report_path = outdir / "steering_report.md"
    _write_csv(rows, csv_path)
    _write_report(
        rows,
        report_path,
        prompt=args.text,
        model_ref=model_ref,
        layer=args.layer,
        k_neighbors=args.k,
        hltd_energy=decomp.energy,
        hltd_topology=topology,
    )
    print(f"saved csv: {csv_path}")
    print(f"saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
