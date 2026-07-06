#!/usr/bin/env python3
"""Run one-step HLTD causal steering on a local Hugging Face causal LM."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spiral_hodge as hodge


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


def _component_delta_row(
    *,
    prompt_id: str,
    layer: int,
    token_index: int,
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

    row: Dict[str, Any] = {
        "prompt_id": prompt_id,
        "layer": int(layer),
        "token_index": int(token_index),
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
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
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
        raise ValueError("This steering script currently expects a GPT-2 style model.transformer.h block stack.")
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
    import torch

    block = _block_for_hidden_layer(model, layer)
    delta_tensor = torch.as_tensor(delta, dtype=torch.float32, device="cpu")

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
            outputs = model(**inputs, return_dict=True)
    finally:
        handle.remove()
    return outputs.logits[0, token_index].detach().float().cpu().numpy()


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


def _select_node_index(
    component_vectors: Dict[str, np.ndarray],
    *,
    selector: str,
    token_index: Optional[int],
    field: hodge.NodeVectorFieldBundle,
) -> int:
    if token_index is not None:
        centers = [int((a + b) // 2) for a, b in field.token_edges]
        if int(token_index) not in centers:
            raise ValueError(f"token_index={token_index} is not an interior node for centered HLTD.")
        return centers.index(int(token_index))

    if selector not in component_vectors:
        valid = ", ".join(sorted(component_vectors))
        raise ValueError(f"Unknown selector component {selector!r}. Valid: {valid}")
    norms = np.linalg.norm(component_vectors[selector], axis=1)
    if not np.any(np.isfinite(norms)):
        return 0
    return int(np.nanargmax(norms))


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fields = [
        "prompt_id",
        "layer",
        "token_index",
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
    node_index: int,
    token_index: int,
    token: str,
    next_token: str,
    hltd_energy: Dict[str, float],
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
    lines = [
        "# HLTD One-Step Steering Report",
        "",
        "## Run",
        "",
        f"- model: `{model_ref}`",
        f"- layer: `{layer}`",
        f"- k_neighbors: `{k_neighbors}`",
        f"- selected node: `{node_index}`",
        f"- token: `{token_index}` `{token}` -> `{next_token}`",
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
        "| component | active | alpha | KL(base||steered) | entropy delta | next-token delta | top shift token | top shift delta | steered top |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {component} | {active} | {alpha} | {kl} | {entropy} | {next_delta} | `{shift}` | {shift_delta} | `{top}` |".format(
                component=row["component"],
                active=row["component_active"],
                alpha=fmt(row["alpha"]),
                kl=fmt(row["kl_base_to_steered"]),
                entropy=fmt(row["entropy_delta"]),
                next_delta=fmt(row["next_token_logprob_delta"]),
                shift=row["top_shift_token"],
                shift_delta=fmt(row["top_shift_logprob_delta"]),
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
    parser.add_argument("--target-text", default=None, help="Optional next-token target text; first token is used")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
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

    import torch

    model_ref, model_is_local = hodge.resolve_hf_model_ref("gpt2", args.model_path)
    device = hodge.choose_device(args.device)
    local_files_only = bool(model_is_local or hodge.hf_offline_enabled())
    model, tokenizer = _load_model_and_tokenizer(
        model_ref,
        device=device,
        local_files_only=local_files_only,
        trust_remote_code=args.trust_remote_code,
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
    decomp = hodge.hodge_latent_traversal_dynamics(
        field.points,
        field.vectors,
        k_neighbors=args.k,
        ridge=args.ridge,
        use_triangles=True,
    )
    component_vectors = hodge.hltd_component_node_vectors(decomp, ridge=args.node_ridge)

    rng = np.random.default_rng(args.seed)
    coexact_norms = np.linalg.norm(component_vectors["coexact"], axis=1)
    random_chart = rng.normal(size=component_vectors["coexact"].shape)
    random_norms = np.linalg.norm(random_chart, axis=1, keepdims=True)
    random_chart = random_chart / np.maximum(random_norms, 1e-12) * coexact_norms[:, None]
    component_vectors["random_tangent"] = random_chart

    node_index = _select_node_index(
        component_vectors,
        selector=args.selector_component,
        token_index=args.token_index,
        field=field,
    )
    edge_a, edge_b = field.token_edges[node_index]
    token_index = int((edge_a + edge_b) // 2)
    if token_index >= tokens - 1:
        raise ValueError("Selected token must have a next token for next-token logits.")

    token = _decode_token(tokenizer, int(input_ids[token_index]))
    next_token_id = int(input_ids[token_index + 1])
    next_token = _decode_token(tokenizer, next_token_id)
    base_logits = outputs.logits[0, token_index].detach().float().cpu().numpy()
    natural_step_norm = _natural_centered_step_norm(hidden[args.layer])
    if not np.isfinite(natural_step_norm) or natural_step_norm <= 0.0:
        raise ValueError("Could not compute a positive natural hidden step norm.")

    target_id = _target_token_id(tokenizer, args.target_text)
    rows: List[Dict[str, Any]] = []
    for component in args.steering_components:
        if component not in component_vectors:
            valid = ", ".join(sorted(component_vectors))
            raise ValueError(f"Unknown steering component {component!r}. Valid: {valid}")
        chart_vec = np.asarray(component_vectors[component][node_index], dtype=np.float64)
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
            rows.append(
                _component_delta_row(
                    prompt_id=args.prompt_id,
                    layer=args.layer,
                    token_index=token_index,
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
                )
            )

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
        node_index=node_index,
        token_index=token_index,
        token=token,
        next_token=next_token,
        hltd_energy=decomp.energy,
    )
    print(f"saved csv: {csv_path}")
    print(f"saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
