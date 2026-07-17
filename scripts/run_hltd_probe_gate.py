#!/usr/bin/env python3
"""Run learned hidden-state probe gates over HLTD steering directions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spiral_hodge as hodge
from scripts.run_hltd_steering import (
    _load_model_and_tokenizer,
    _natural_centered_step_norm,
    _select_node_indices,
    _with_derived_components,
)
from scripts.run_hltd_steering_fast_suite import _random_tangent_component
from scripts.run_hltd_steering_suite import read_suite, select_prompts


COMPONENTS = ["presence", "coexact", "semantic_flow", "harmonic", "random_tangent"]
DISSOCIATION_COMPONENTS = [
    "presence",
    "coexact",
    "presence_plus_coexact",
    "coexact_minus_presence",
    "negative_coexact",
    "random_tangent",
]
PROBE_METRICS = [
    "component_active",
    "positive_prob_delta",
    "positive_logit_delta",
    "positive_axis_delta",
    "label_margin_delta",
    "label_axis_delta",
    "probe_entropy_delta",
]


@dataclass
class PromptCache:
    prompt_id: str
    group_id: str
    family: str
    text: str
    input_ids: np.ndarray
    token_texts: List[str]
    hidden: np.ndarray
    coord: Any


@dataclass
class ProbeModel:
    name: str
    layer: int
    positive_families: set[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    n_examples: int
    n_positive: int
    train_accuracy: float
    train_auc: float
    cv_prompt_accuracy: float
    n_groups: int
    grouping: str
    training_token_selector: str


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean_or_nan(values: Sequence[float]) -> float:
    return mean(values) if values else float("nan")


def sample_std_or_nan(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return "nan"
    return f"{number:.{digits}f}"


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def binary_entropy(prob: float) -> float:
    p = float(np.clip(prob, 1e-12, 1.0 - 1e-12))
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def load_probe_specs(path: str, requested: Optional[Sequence[str]] = None) -> Dict[str, set[str]]:
    with Path(path).expanduser().open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Probe label file must contain an object: {path}")
    requested_set = {str(x) for x in requested or []}
    specs: Dict[str, set[str]] = {}
    for name, value in raw.items():
        if requested_set and str(name) not in requested_set:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Probe spec {name!r} must be an object.")
        families = {str(x) for x in value.get("positive_families", [])}
        if not families:
            raise ValueError(f"Probe spec {name!r} must include positive_families.")
        specs[str(name)] = families
    if requested_set:
        missing = requested_set - set(specs)
        if missing:
            raise ValueError(f"Unknown requested probes: {', '.join(sorted(missing))}")
    return specs


def prompt_inputs(tokenizer: Any, text: str, *, device: str, max_length: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"return_tensors": "pt"}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max_length})
    return tokenizer(text, **kwargs).to(device)


def extract_hidden(model: Any, inputs: Dict[str, Any]) -> np.ndarray:
    import torch

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    return torch.stack(outputs.hidden_states, dim=0)[:, 0].detach().float().cpu().numpy()


def load_prompt_caches(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Dict[str, Any]],
    device: str,
    max_length: int,
    components: int,
    normalize_hidden: bool,
    seed: int,
) -> List[PromptCache]:
    caches: List[PromptCache] = []
    for idx, item in enumerate(prompts, start=1):
        prompt_id = str(item["prompt_id"])
        group_id = str(item.get("pair_id") or prompt_id)
        family = str(item["family"])
        text = str(item["text"])
        print(f"[extract {idx}/{len(prompts)}] {family}/{prompt_id}", flush=True)
        inputs = prompt_inputs(tokenizer, text, device=device, max_length=max_length)
        input_ids = inputs["input_ids"][0].detach().cpu().numpy()
        token_texts = [tokenizer.decode([int(token_id)]).replace("\n", "\\n") for token_id in input_ids]
        if input_ids.size < 4:
            raise ValueError(f"{prompt_id}: need at least 4 tokens for centered HLTD probe gate.")
        hidden = extract_hidden(model, inputs)
        coord = hodge.make_semantic_coordinates(
            hidden,
            method="pca",
            n_components=components,
            normalize_hidden=normalize_hidden,
            random_state=seed,
            verbose=False,
        )
        caches.append(
            PromptCache(
                prompt_id=prompt_id,
                group_id=group_id,
                family=family,
                text=text,
                input_ids=input_ids,
                token_texts=token_texts,
                hidden=hidden,
                coord=coord,
            )
        )
    return caches


def build_probe_training_matrix(
    caches: Sequence[PromptCache],
    *,
    layer: int,
    positive_families: set[str],
    token_selector: str = "all_interior",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_selectors = {"all_interior", "middle", "last_interior", "pair_balanced_interior"}
    if token_selector not in valid_selectors:
        raise ValueError(f"Unknown probe training token selector {token_selector!r}.")

    balanced_counts: Dict[str, int] = {}
    if token_selector == "pair_balanced_interior":
        by_group: Dict[str, List[PromptCache]] = defaultdict(list)
        for cache in caches:
            by_group[cache.group_id].append(cache)
        for group_id, group in by_group.items():
            labels = {cache.family in positive_families for cache in group}
            if labels != {False, True}:
                raise ValueError(
                    f"Counterfactual group {group_id!r} must contain both probe classes for pair balancing."
                )
            balanced_counts[group_id] = min(int(cache.hidden.shape[1]) - 2 for cache in group)

    xs: List[np.ndarray] = []
    ys: List[int] = []
    groups: List[str] = []
    for cache in caches:
        H = np.asarray(cache.hidden[layer], dtype=np.float64)
        interior = np.arange(1, H.shape[0] - 1, dtype=int)
        if token_selector == "middle":
            token_indices = [int(interior[len(interior) // 2])]
        elif token_selector == "last_interior":
            token_indices = [int(interior[-1])]
        elif token_selector == "pair_balanced_interior":
            count = balanced_counts[cache.group_id]
            positions = np.rint(np.linspace(0, len(interior) - 1, num=count)).astype(int)
            token_indices = [int(interior[position]) for position in positions]
        else:
            token_indices = [int(index) for index in interior]
        for token_index in token_indices:
            xs.append(H[token_index])
            ys.append(1 if cache.family in positive_families else 0)
            groups.append(cache.group_id)
    return np.vstack(xs), np.asarray(ys, dtype=int), np.asarray(groups, dtype=object)


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_vec = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (X - mean_vec) / scale, mean_vec, scale


def safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    if len(set(int(x) for x in y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, prob))


def prompt_cv_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int) -> float:
    """Return leave-one-group-out accuracy.

    Historical outputs call this prompt-CV. When a training suite supplies
    ``pair_id``, both members of the counterfactual pair share one group and
    are held out together.
    """
    from sklearn.linear_model import LogisticRegression

    accuracies: List[float] = []
    for prompt_id in sorted(set(str(x) for x in groups)):
        heldout = groups == prompt_id
        train = ~heldout
        if len(set(int(x) for x in y[train])) < 2:
            continue
        X_train, mean_vec, scale = standardize_fit(X[train])
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X_train, y[train])
        X_test = (X[heldout] - mean_vec) / scale
        pred = (clf.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        accuracies.append(float(np.mean(pred == y[heldout])))
    return mean_or_nan(accuracies)


def train_probe(
    caches: Sequence[PromptCache],
    *,
    layer: int,
    name: str,
    positive_families: set[str],
    seed: int,
    training_token_selector: str = "all_interior",
) -> ProbeModel:
    from sklearn.linear_model import LogisticRegression

    X, y, groups = build_probe_training_matrix(
        caches,
        layer=layer,
        positive_families=positive_families,
        token_selector=training_token_selector,
    )
    if len(set(int(x) for x in y)) < 2:
        raise ValueError(f"Probe {name!r} at layer {layer} has only one class.")
    X_scaled, mean_vec, scale = standardize_fit(X)
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_scaled, y)
    prob = clf.predict_proba(X_scaled)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return ProbeModel(
        name=name,
        layer=layer,
        positive_families=set(positive_families),
        mean=mean_vec,
        scale=scale,
        coef=np.asarray(clf.coef_[0], dtype=np.float64),
        intercept=float(clf.intercept_[0]),
        n_examples=int(y.size),
        n_positive=int(y.sum()),
        train_accuracy=float(np.mean(pred == y)),
        train_auc=safe_auc(y, prob),
        cv_prompt_accuracy=prompt_cv_accuracy(X, y, groups, seed=seed),
        n_groups=len(set(str(x) for x in groups)),
        grouping="pair_id" if any(cache.group_id != cache.prompt_id for cache in caches) else "prompt_id",
        training_token_selector=training_token_selector,
    )


def score_probe(probe: ProbeModel, hidden_vec: np.ndarray) -> Tuple[float, float, float]:
    z = (np.asarray(hidden_vec, dtype=np.float64) - probe.mean) / probe.scale
    logit = float(np.dot(probe.coef, z) + probe.intercept)
    prob = sigmoid(logit)
    return prob, logit, binary_entropy(prob)


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def probe_training_rows(probes_by_layer: Dict[int, Dict[str, ProbeModel]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for layer, by_name in sorted(probes_by_layer.items()):
        for probe in by_name.values():
            rows.append(
                {
                    "probe": probe.name,
                    "layer": layer,
                    "positive_families": ",".join(sorted(probe.positive_families)),
                    "n_examples": probe.n_examples,
                    "n_positive": probe.n_positive,
                    "train_accuracy": probe.train_accuracy,
                    "train_auc": probe.train_auc,
                    "cv_prompt_accuracy": probe.cv_prompt_accuracy,
                    "cv_group_accuracy": probe.cv_prompt_accuracy,
                    "n_groups": probe.n_groups,
                    "grouping": probe.grouping,
                    "training_token_selector": probe.training_token_selector,
                    "coef_norm": float(np.linalg.norm(probe.coef)),
                }
            )
    return rows


def run_probe_gate(
    *,
    caches: Sequence[PromptCache],
    probes_by_layer: Dict[int, Dict[str, ProbeModel]],
    layers: Sequence[int],
    ks: Sequence[int],
    alphas: Sequence[float],
    steering_components: Sequence[str],
    selector_component: str,
    token_selectors: Sequence[str],
    token_indices: Sequence[int],
    position_bins: Sequence[int],
    position_bin_count: int,
    seeds: Sequence[int],
    ridge: float,
    node_ridge: float,
    min_chart_norm: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for prompt_no, cache in enumerate(caches, start=1):
        print(f"[probe {prompt_no}/{len(caches)}] {cache.family}/{cache.prompt_id}", flush=True)
        layers_count, tokens, dim = cache.hidden.shape
        for layer in layers:
            if layer <= 0 or layer >= layers_count:
                raise ValueError(f"--layers contains {layer}, but valid hidden layers are 1..{layers_count - 1}")
            field = hodge.token_node_vector_field(cache.coord.coords, layer=layer, mode="centered")
            component_vectors_by_k: Dict[int, Dict[str, np.ndarray]] = {}
            energy_by_k: Dict[int, Dict[str, float]] = {}
            for k in ks:
                decomp_k = hodge.hodge_latent_traversal_dynamics(
                    field.points,
                    field.vectors,
                    k_neighbors=int(k),
                    ridge=ridge,
                    use_triangles=True,
                )
                component_vectors_by_k[int(k)] = _with_derived_components(
                    hodge.hltd_component_node_vectors(decomp_k, ridge=node_ridge)
                )
                energy_by_k[int(k)] = decomp_k.energy

            natural_step_norm = _natural_centered_step_norm(cache.hidden[layer])
            if not np.isfinite(natural_step_norm) or natural_step_norm <= 0.0:
                raise ValueError(f"{cache.prompt_id}/L{layer}: natural hidden step norm is not positive.")

            for k in ks:
                component_vectors = component_vectors_by_k[int(k)]
                energy = energy_by_k[int(k)]
                for seed in seeds:
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
                            if token_index >= tokens:
                                continue
                            base_hidden = np.asarray(cache.hidden[layer, token_index], dtype=np.float64)
                            token = cache.token_texts[token_index] if token_index < len(cache.token_texts) else ""
                            for component in steering_components:
                                if component not in seeded_components:
                                    valid = ", ".join(sorted(seeded_components))
                                    raise ValueError(f"Unknown steering component {component!r}. Valid: {valid}")
                                chart_vec = np.asarray(seeded_components[component][node_index], dtype=np.float64)
                                hidden_vec = hodge.pca_chart_vectors_to_hidden(chart_vec, cache.coord.reducer)
                                hidden_norm = float(np.linalg.norm(hidden_vec))
                                chart_norm = float(np.linalg.norm(chart_vec))
                                component_active = bool(chart_norm >= float(min_chart_norm) and hidden_norm > 0.0)
                                direction = hidden_vec / hidden_norm if component_active else np.zeros(dim, dtype=np.float64)
                                for alpha in alphas:
                                    delta = float(alpha) * natural_step_norm * direction
                                    steered_hidden = base_hidden + delta
                                    for probe_name, probe in probes_by_layer[layer].items():
                                        base_prob, base_logit, base_entropy = score_probe(probe, base_hidden)
                                        steered_prob, steered_logit, steered_entropy = score_probe(probe, steered_hidden)
                                        label = 1 if cache.family in probe.positive_families else 0
                                        prob_delta = steered_prob - base_prob
                                        logit_delta = steered_logit - base_logit
                                        coef_norm = float(np.linalg.norm(probe.coef))
                                        axis_delta = logit_delta / max(coef_norm, 1e-12)
                                        label_sign = 1.0 if label else -1.0
                                        rows.append(
                                            {
                                                "family": cache.family,
                                                "prompt_id": cache.prompt_id,
                                                "layer": int(layer),
                                                "k": int(k),
                                                "seed": int(seed),
                                                "token_selector": token_selector,
                                                "selector_component": selector_component,
                                                "node_index": int(node_index),
                                                "token_index": int(token_index),
                                                "token_count": int(tokens),
                                                "token": token,
                                                "component": component,
                                                "alpha": float(alpha),
                                                "probe": probe_name,
                                                "probe_label": int(label),
                                                "positive_families": ",".join(sorted(probe.positive_families)),
                                                "component_active": int(bool(component_active)),
                                                "delta_norm": float(np.linalg.norm(delta)),
                                                "natural_step_norm": natural_step_norm,
                                                "chart_norm": chart_norm,
                                                "hidden_direction_norm": hidden_norm,
                                                "hltd_exact_ratio": energy.get("exact_ratio", float("nan")),
                                                "hltd_coexact_ratio": energy.get("coexact_ratio", float("nan")),
                                                "hltd_harmonic_ratio": energy.get("harmonic_ratio", float("nan")),
                                                "hltd_semantic_flow_ratio": energy.get(
                                                    "semantic_flow_ratio",
                                                    float("nan"),
                                                ),
                                                "positive_prob_base": base_prob,
                                                "positive_prob_steered": steered_prob,
                                                "positive_prob_delta": prob_delta,
                                                "positive_logit_base": base_logit,
                                                "positive_logit_steered": steered_logit,
                                                "positive_logit_delta": logit_delta,
                                                "positive_axis_delta": axis_delta,
                                                "label_margin_delta": label_sign * logit_delta,
                                                "label_axis_delta": label_sign * axis_delta,
                                                "probe_entropy_base": base_entropy,
                                                "probe_entropy_steered": steered_entropy,
                                                "probe_entropy_delta": steered_entropy - base_entropy,
                                            }
                                        )
    return rows


def build_component_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, int, int, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["family"]),
            int(row["layer"]),
            int(row["k"]),
            str(row["token_selector"]),
            str(row["component"]),
            str(row["alpha"]),
            str(row["probe"]),
        )
        buckets[key].append(row)
    out: List[Dict[str, Any]] = []
    for (family, layer, k, selector, component, alpha, probe), group in sorted(buckets.items()):
        item: Dict[str, Any] = {
            "family": family,
            "layer": layer,
            "k": k,
            "token_selector": selector,
            "component": component,
            "alpha": float(alpha),
            "probe": probe,
            "n_rows": len(group),
            "n_seeds": len({str(row["seed"]) for row in group}),
            "n_tokens": len({str(row["token_index"]) for row in group}),
            "probe_label_mean": mean_or_nan([float(row["probe_label"]) for row in group]),
        }
        for metric in PROBE_METRICS:
            values = [value for value in (finite_float(row.get(metric)) for row in group) if value is not None]
            item[f"{metric}_mean"] = mean_or_nan(values)
        out.append(item)
    return out


def build_pairwise_summary(
    rows: Sequence[Dict[str, Any]],
    *,
    baseline_component: str = "random_tangent",
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, int, int, str, str, str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["family"]),
            str(row["prompt_id"]),
            int(row["layer"]),
            int(row["k"]),
            str(row["seed"]),
            str(row["token_selector"]),
            str(row["token_index"]),
            str(row["alpha"]),
            str(row["probe"]),
        )
        groups[key][str(row["component"])] = row

    buckets: Dict[Tuple[str, int, int, str, str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[Tuple[str, int, int, str, str, str, str], int] = defaultdict(int)
    for (family, _prompt_id, layer, k, _seed, selector, _token_index, alpha, probe), by_component in groups.items():
        baseline = by_component.get(baseline_component)
        if baseline is None:
            continue
        if (finite_float(baseline.get("component_active")) or 0.0) <= 0.0:
            continue
        for component, row in by_component.items():
            if component == baseline_component:
                continue
            if (finite_float(row.get("component_active")) or 0.0) <= 0.0:
                continue
            out_key = (family, layer, k, selector, component, alpha, probe)
            added = False
            for metric in PROBE_METRICS:
                a = finite_float(row.get(metric))
                b = finite_float(baseline.get(metric))
                if a is None or b is None:
                    continue
                buckets[out_key][f"{metric}_minus_{baseline_component}"].append(a - b)
                added = True
            if added:
                counts[out_key] += 1

    out: List[Dict[str, Any]] = []
    for (family, layer, k, selector, component, alpha, probe), values_by_metric in sorted(buckets.items()):
        item: Dict[str, Any] = {
            "family": family,
            "layer": layer,
            "k": k,
            "token_selector": selector,
            "component": component,
            "baseline_component": baseline_component,
            "alpha": float(alpha),
            "probe": probe,
            "n_pairs": counts[(family, layer, k, selector, component, alpha, probe)],
        }
        for metric, values in sorted(values_by_metric.items()):
            item[f"{metric}_mean"] = mean_or_nan(values)
            item[f"{metric}_std"] = sample_std_or_nan(values)
            item[f"{metric}_sem"] = (
                sample_std_or_nan(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")
            )
            item[f"{metric}_positive_rate"] = mean_or_nan(
                [float(value > 0.0) for value in values]
            )
        out.append(item)
    return out


def build_layer_pairwise_summary(pairwise_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[int, int, str, str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[Tuple[int, int, str, str, str, str], int] = defaultdict(int)
    for row in pairwise_rows:
        key = (
            int(row["layer"]),
            int(row["k"]),
            str(row["token_selector"]),
            str(row["component"]),
            str(row["alpha"]),
            str(row["probe"]),
        )
        counts[key] += int(row.get("n_pairs", 0) or 0)
        for metric, value in row.items():
            if not metric.endswith("_mean"):
                continue
            number = finite_float(value)
            if number is not None:
                buckets[key][metric].append(number)

    out: List[Dict[str, Any]] = []
    for (layer, k, selector, component, alpha, probe), values_by_metric in sorted(buckets.items()):
        item: Dict[str, Any] = {
            "layer": layer,
            "k": k,
            "token_selector": selector,
            "component": component,
            "alpha": float(alpha),
            "probe": probe,
            "n_pairs": counts[(layer, k, selector, component, alpha, probe)],
            "n_family_rows": max((len(values) for values in values_by_metric.values()), default=0),
        }
        for metric, values in sorted(values_by_metric.items()):
            item[metric] = mean_or_nan(values)
        out.append(item)
    return out


def write_report(
    *,
    output_root: Path,
    probe_rows: Sequence[Dict[str, Any]],
    training_rows: Sequence[Dict[str, Any]],
    component_summary: Sequence[Dict[str, Any]],
    pairwise_summary: Sequence[Dict[str, Any]],
    layer_pairwise_summary: Sequence[Dict[str, Any]],
    training_suite: str,
    evaluation_suite: str,
    split_is_disjoint: bool,
) -> None:
    lines = [
        "# HLTD Learned Probe Gate Summary",
        "",
        "## Run",
        "",
        f"- probe rows: {len(probe_rows)}",
        f"- prompt/layer/k runs: {len({(r['prompt_id'], r['layer'], r['k']) for r in probe_rows})}",
        f"- training suite: `{training_suite}`",
        f"- evaluation suite: `{evaluation_suite}`",
        f"- disjoint prompt IDs: `{str(split_is_disjoint).lower()}`",
        "",
        "## Probe Training",
        "",
        "| probe | layer | positives | n | groups | grouping | training tokens | train acc | train AUC | group-CV acc | coef norm |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in training_rows:
        lines.append(
            "| {probe} | L{layer} | {pos} | {n} | {groups} | {grouping} | {tokens} | {acc} | {auc} | {cv} | {norm} |".format(
                probe=row["probe"],
                layer=int(row["layer"]),
                pos=int(row["n_positive"]),
                n=int(row["n_examples"]),
                groups=int(row["n_groups"]),
                grouping=row["grouping"],
                tokens=row["training_token_selector"],
                acc=fmt(row["train_accuracy"]),
                auc=fmt(row["train_auc"]),
                cv=fmt(row["cv_prompt_accuracy"]),
                norm=fmt(row["coef_norm"]),
            )
        )
    lines.extend(
        [
            "",
            "## Component Means",
            "",
            "| family | layer | selector | probe | component | n | active | prob delta | logit delta | label-axis delta | label-margin delta | entropy delta |",
            "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in component_summary:
        lines.append(
            "| {family} | L{layer} | {selector} | {probe} | {component} | {n} | {active} | {prob} | {logit} | {axis} | {margin} | {entropy} |".format(
                family=row["family"],
                layer=int(row["layer"]),
                selector=row["token_selector"],
                probe=row["probe"],
                component=row["component"],
                n=int(row["n_rows"]),
                active=fmt(row.get("component_active_mean")),
                prob=fmt(row.get("positive_prob_delta_mean")),
                logit=fmt(row.get("positive_logit_delta_mean")),
                axis=fmt(row.get("label_axis_delta_mean")),
                margin=fmt(row.get("label_margin_delta_mean")),
                entropy=fmt(row.get("probe_entropy_delta_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Component Minus Random Tangent",
            "",
            "| family | layer | selector | probe | component | n | prob delta | logit delta | label-axis delta | null win rate | label-margin delta | entropy delta |",
            "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pairwise_summary:
        baseline = row["baseline_component"]
        lines.append(
            "| {family} | L{layer} | {selector} | {probe} | {component} | {n} | {prob} | {logit} | {axis} | {win} | {margin} | {entropy} |".format(
                family=row["family"],
                layer=int(row["layer"]),
                selector=row["token_selector"],
                probe=row["probe"],
                component=row["component"],
                n=int(row["n_pairs"]),
                prob=fmt(row.get(f"positive_prob_delta_minus_{baseline}_mean")),
                logit=fmt(row.get(f"positive_logit_delta_minus_{baseline}_mean")),
                axis=fmt(row.get(f"label_axis_delta_minus_{baseline}_mean")),
                win=fmt(row.get(f"label_axis_delta_minus_{baseline}_positive_rate")),
                margin=fmt(row.get(f"label_margin_delta_minus_{baseline}_mean")),
                entropy=fmt(row.get(f"probe_entropy_delta_minus_{baseline}_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Layer Pairwise Gate",
            "",
            "| layer | selector | probe | component | n | prob delta | logit delta | label-axis delta | label-margin delta | entropy delta |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in layer_pairwise_summary:
        baseline = "random_tangent"
        lines.append(
            "| L{layer} | {selector} | {probe} | {component} | {n} | {prob} | {logit} | {axis} | {margin} | {entropy} |".format(
                layer=int(row["layer"]),
                selector=row["token_selector"],
                probe=row["probe"],
                component=row["component"],
                n=int(row["n_pairs"]),
                prob=fmt(row.get(f"positive_prob_delta_minus_{baseline}_mean")),
                logit=fmt(row.get(f"positive_logit_delta_minus_{baseline}_mean")),
                axis=fmt(row.get(f"label_axis_delta_minus_{baseline}_mean")),
                margin=fmt(row.get(f"label_margin_delta_minus_{baseline}_mean")),
                entropy=fmt(row.get(f"probe_entropy_delta_minus_{baseline}_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Conservative Read",
            "",
            "This gate scores one-step hidden-state moves with lightweight linear probes. A disjoint counterfactual training suite removes direct prompt reuse, but the probe remains a small in-domain linear classifier rather than an external ontology classifier or closed-loop fluency test.",
        ]
    )
    (output_root / "summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_token_indices(token_index: Optional[int], token_indices: Optional[Sequence[int]]) -> List[int]:
    out: List[int] = []
    if token_indices is not None:
        out.extend(int(x) for x in token_indices)
    if token_index is not None:
        out.append(int(token_index))
    return sorted(set(out))


def validate_probe_training_split(
    training_items: Sequence[Dict[str, Any]],
    evaluation_items: Sequence[Dict[str, Any]],
    *,
    allow_overlap: bool = False,
) -> List[str]:
    training_ids = {str(item["prompt_id"]) for item in training_items}
    evaluation_ids = {str(item["prompt_id"]) for item in evaluation_items}
    overlap = sorted(training_ids & evaluation_ids)
    if overlap and not allow_overlap:
        preview = ", ".join(overlap[:8])
        suffix = "..." if len(overlap) > 8 else ""
        raise ValueError(
            "Probe training and evaluation prompt IDs overlap: "
            f"{preview}{suffix}. Pass --allow-probe-training-overlap only for legacy same-suite runs."
        )
    return overlap


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="data/hltd_prompt_suite.jsonl")
    parser.add_argument(
        "--probe-training-suite",
        default=None,
        help="Optional independent JSONL suite used only to train the probes.",
    )
    parser.add_argument(
        "--allow-probe-training-overlap",
        action="store_true",
        help="Allow training/evaluation prompt ID overlap for legacy same-suite runs.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", default="spiral_out_hltd_probe_gate")
    parser.add_argument("--probe-label-file", default="data/hltd_probe_labels.json")
    parser.add_argument("--probes", nargs="+", default=None)
    parser.add_argument(
        "--probe-training-token-selector",
        choices=["all_interior", "middle", "last_interior", "pair_balanced_interior"],
        default="all_interior",
        help="Hidden positions used to fit each probe.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[5])
    parser.add_argument("--k", type=int, nargs="+", default=[16])
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--steering-components", nargs="+", default=COMPONENTS)
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
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--node-ridge", type=float, default=1e-4)
    parser.add_argument("--min-chart-norm", type=float, default=1e-6)
    parser.add_argument("--no-normalize-hidden", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    evaluation_suite_path = Path(args.suite)
    prompt_items = select_prompts(
        read_suite(evaluation_suite_path),
        families=args.families,
        prompt_ids=args.prompt_ids,
        limit=args.limit,
        max_prompts_per_family=args.max_prompts_per_family,
    )
    if not prompt_items:
        raise ValueError("No prompts matched the requested filters.")

    training_suite_path = Path(args.probe_training_suite) if args.probe_training_suite else evaluation_suite_path
    if args.probe_training_suite:
        training_items = read_suite(training_suite_path)
    else:
        training_items = list(prompt_items)
    overlap = validate_probe_training_split(
        training_items,
        prompt_items,
        allow_overlap=bool(args.allow_probe_training_overlap or not args.probe_training_suite),
    )

    probe_specs = load_probe_specs(args.probe_label_file, requested=args.probes)
    random_seeds = list(args.seeds or [args.seed])
    token_indices = parse_token_indices(args.token_index, args.token_indices)
    position_bins = sorted(set(int(x) for x in (args.position_bins or [])))
    token_selectors = list(args.token_selectors or (["position_bin"] if position_bins else ["middle"]))
    planned_rows = (
        len(prompt_items)
        * len(args.layers)
        * len(args.k)
        * len(random_seeds)
        * len(token_selectors)
        * len(args.steering_components)
        * len(args.alphas)
        * len(probe_specs)
    )
    print(
        "planned probe gate: "
        f"{len(training_items)} training prompts, "
        f"{len(prompt_items)} evaluation prompts x {len(args.layers)} layers x {len(args.k)} k values, "
        f">= {planned_rows} rows",
        flush=True,
    )
    if args.dry_run:
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

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    training_caches = load_prompt_caches(
        model=model,
        tokenizer=tokenizer,
        prompts=training_items,
        device=device,
        max_length=args.max_length,
        components=args.components,
        normalize_hidden=not args.no_normalize_hidden,
        seed=args.seed,
    )
    if training_suite_path.resolve() == evaluation_suite_path.resolve() and training_items == prompt_items:
        caches = training_caches
    else:
        caches = load_prompt_caches(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt_items,
            device=device,
            max_length=args.max_length,
            components=args.components,
            normalize_hidden=not args.no_normalize_hidden,
            seed=args.seed,
        )

    probes_by_layer: Dict[int, Dict[str, ProbeModel]] = {}
    for layer in args.layers:
        probes_by_layer[int(layer)] = {}
        for probe_name, families in probe_specs.items():
            print(f"[train probe] L{layer} {probe_name}", flush=True)
            probes_by_layer[int(layer)][probe_name] = train_probe(
                training_caches,
                layer=int(layer),
                name=probe_name,
                positive_families=families,
                seed=args.seed,
                training_token_selector=args.probe_training_token_selector,
            )

    training_rows = probe_training_rows(probes_by_layer)
    write_csv(training_rows, output_root / "probe_training_summary.csv")
    probe_rows = run_probe_gate(
        caches=caches,
        probes_by_layer=probes_by_layer,
        layers=args.layers,
        ks=args.k,
        alphas=args.alphas,
        steering_components=args.steering_components,
        selector_component=args.selector_component,
        token_selectors=token_selectors,
        token_indices=token_indices,
        position_bins=position_bins,
        position_bin_count=args.position_bin_count,
        seeds=random_seeds,
        ridge=args.ridge,
        node_ridge=args.node_ridge,
        min_chart_norm=args.min_chart_norm,
    )
    component_summary = build_component_summary(probe_rows)
    pairwise_summary = build_pairwise_summary(probe_rows)
    layer_pairwise_summary = build_layer_pairwise_summary(pairwise_summary)
    write_csv(probe_rows, output_root / "probe_metrics.csv")
    write_csv(component_summary, output_root / "summary_component.csv")
    write_csv(pairwise_summary, output_root / "summary_pairwise.csv")
    write_csv(layer_pairwise_summary, output_root / "summary_layer_pairwise.csv")
    write_report(
        output_root=output_root,
        probe_rows=probe_rows,
        training_rows=training_rows,
        component_summary=component_summary,
        pairwise_summary=pairwise_summary,
        layer_pairwise_summary=layer_pairwise_summary,
        training_suite=str(training_suite_path),
        evaluation_suite=str(evaluation_suite_path),
        split_is_disjoint=not bool(overlap),
    )
    manifest = {
        "model_ref": model_ref,
        "training_suite": str(training_suite_path),
        "training_suite_sha256": file_sha256(training_suite_path),
        "evaluation_suite": str(evaluation_suite_path),
        "evaluation_suite_sha256": file_sha256(evaluation_suite_path),
        "training_prompt_ids": [str(item["prompt_id"]) for item in training_items],
        "evaluation_prompt_ids": [str(item["prompt_id"]) for item in prompt_items],
        "prompt_id_overlap": overlap,
        "split_is_disjoint": not bool(overlap),
        "probe_label_file": str(args.probe_label_file),
        "probe_label_file_sha256": file_sha256(Path(args.probe_label_file)),
        "probes": sorted(probe_specs),
        "probe_training_token_selector": args.probe_training_token_selector,
        "layers": [int(x) for x in args.layers],
        "k": [int(x) for x in args.k],
        "components": int(args.components),
        "alphas": [float(x) for x in args.alphas],
        "steering_components": [str(x) for x in args.steering_components],
        "token_selectors": token_selectors,
        "seeds": [int(x) for x in random_seeds],
        "training_grouping": sorted({probe.grouping for by_name in probes_by_layer.values() for probe in by_name.values()}),
    }
    (output_root / "probe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"probe gate complete: {len(probe_rows)} rows in {elapsed:.1f}s -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
