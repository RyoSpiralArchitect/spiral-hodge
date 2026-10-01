"""Calibration-only Hodge fields and a prefix-only direction boundary.

No model loading, text generation, or evaluation-set field fitting occurs here.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

import spiral_hodge as hodge


def _readonly(value: np.ndarray) -> np.ndarray:
    # Back with immutable bytes, not merely a writeable owning array's flag.
    data = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(data.tobytes(), dtype=np.float64).reshape(data.shape)


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([array.dtype.str, array.shape]).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenPrefixField:
    mean: np.ndarray
    components: np.ndarray
    points: np.ndarray
    coexact: np.ndarray
    node_prompts: tuple[str, ...]
    node_tokens: tuple[int, ...]
    calibration_hashes: tuple[tuple[str, str], ...]
    dose_scale: float
    support_limit: float
    min_chart_norm: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_prompts", tuple(self.node_prompts))
        object.__setattr__(self, "node_tokens", tuple(self.node_tokens))
        object.__setattr__(self, "calibration_hashes", tuple(tuple(item) for item in self.calibration_hashes))
        for name in ("mean", "components", "points", "coexact"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite field array: {name}")
            object.__setattr__(self, name, _readonly(value))
        if self.components.ndim != 2 or self.mean.shape != (self.components.shape[1],):
            raise ValueError("invalid PCA shapes")
        if self.points.ndim != 2 or self.points.shape[1] != self.components.shape[0] or not len(self.points):
            raise ValueError("invalid field point shape")
        if self.coexact.shape != self.points.shape:
            raise ValueError("coexact/point shape mismatch")
        if len(self.node_prompts) != len(self.points) or len(self.node_tokens) != len(self.points):
            raise ValueError("node metadata length mismatch")
        if not np.allclose(self.components @ self.components.T, np.eye(len(self.components)), atol=1e-10, rtol=0):
            raise ValueError("PCA components must be orthonormal")
        if not np.isfinite([self.dose_scale, self.support_limit, self.min_chart_norm]).all():
            raise ValueError("nonfinite field scale or threshold")
        if self.dose_scale <= 0 or self.support_limit < 0 or self.min_chart_norm <= 0:
            raise ValueError("invalid field scale or threshold")

    @property
    def components_(self) -> np.ndarray:
        return self.components

    def fingerprint(self) -> str:
        metadata = {
            "arrays": {name: array_digest(getattr(self, name)) for name in ("mean", "components", "points", "coexact")},
            "node_prompts": self.node_prompts, "node_tokens": self.node_tokens,
            "calibration_hashes": self.calibration_hashes,
            "dose_scale": self.dose_scale, "support_limit": self.support_limit,
            "min_chart_norm": self.min_chart_norm,
        }
        return hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_frozen_field(
    calibration: Mapping[str, np.ndarray], *, n_components: int = 32, k: int = 16,
    target_betti_1_fraction: float = 0.5, node_ridge: float = 1e-4,
    min_chart_norm: float = 1e-6, support_quantile: float = 0.99,
) -> FrozenPrefixField:
    """Fit one shared L7 chart, decompose each calibration trajectory separately.

    Values are raw [tokens, hidden_dim] states, not multi-layer tensors.
    The eventual runner must bind these keys to its frozen calibration split.
    """
    if len(calibration) < 2 or any(not isinstance(key, str) or not key for key in calibration):
        raise ValueError("at least two named calibration trajectories required")
    if not isinstance(k, Integral) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(n_components, Integral) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    if not np.isfinite([node_ridge, min_chart_norm, support_quantile]).all():
        raise ValueError("nonfinite field setting")
    if node_ridge <= 0 or min_chart_norm <= 0 or not 0 <= support_quantile <= 1:
        raise ValueError("invalid field setting")
    raw, normalized, hashes, steps = {}, [], [], []
    for prompt_id in sorted(calibration):
        values = np.asarray(calibration[prompt_id], dtype=np.float64)
        if values.ndim != 2 or values.shape[0] - 2 <= k or not np.isfinite(values).all():
            raise ValueError(f"invalid or too short calibration trajectory: {prompt_id}")
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError(f"zero calibration hidden state: {prompt_id}")
        if raw and values.shape[1] != next(iter(raw.values())).shape[1]:
            raise ValueError("calibration hidden dimensions differ")
        raw[prompt_id] = values
        normalized.append(values / norms[:, None])
        hashes.append((prompt_id, array_digest(np.asarray(calibration[prompt_id]))))
        steps.append(float(np.median(np.linalg.norm(0.5 * (values[2:] - values[:-2]), axis=1))))
    x = np.concatenate(normalized)
    if n_components > min(x.shape):
        raise ValueError("not enough calibration dimensions or samples for PCA")
    reducer = PCA(n_components=n_components, svd_solver="full")
    reducer.fit(x)
    points, vectors, owners, tokens = [], [], [], []
    for (prompt_id, values), normalized_values in zip(raw.items(), normalized, strict=True):
        z = reducer.transform(normalized_values)
        node_field = hodge.token_node_vector_field(z[None, :, :], layer=0, mode="centered")
        decomposition, _ = hodge.hodge_latent_traversal_dynamics_matched_betti(
            node_field.points, node_field.vectors, k_neighbors=k,
            target_betti_1_fraction=target_betti_1_fraction,
        )
        coexact = hodge.node_vectors_from_edge_component(
            node_field.points, decomposition.edges, decomposition.coexact, ridge=node_ridge,
        )
        points.append(node_field.points)
        vectors.append(coexact)
        owners.extend([prompt_id] * len(node_field.points))
        tokens.extend(range(1, len(values) - 1))
    pooled = np.concatenate(points)
    owners_array = np.asarray(owners)
    # Support is calibrated against other texts, never against the node itself.
    cross_text_distances = []
    for prompt_id in raw:
        mask = owners_array == prompt_id
        cross_text_distances.extend(cdist(pooled[mask], pooled[~mask]).min(axis=1))
    return FrozenPrefixField(
        mean=reducer.mean_, components=reducer.components_, points=pooled,
        coexact=np.concatenate(vectors), node_prompts=tuple(owners), node_tokens=tuple(tokens),
        calibration_hashes=tuple(hashes), dose_scale=float(np.median(steps)),
        support_limit=float(np.quantile(cross_text_distances, support_quantile, method="higher")),
        min_chart_norm=float(min_chart_norm),
    )


@dataclass(frozen=True)
class PrefixDirection:
    node_index: int
    node_prompt: str
    node_token: int
    distance: float
    status: str
    chart: np.ndarray
    coexact_delta: np.ndarray
    random_delta: np.ndarray
    dose_scale: float

    def __post_init__(self) -> None:
        for name in ("chart", "coexact_delta", "random_delta"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    def receipt(self) -> dict:
        return {
            "node_index": self.node_index, "node_prompt": self.node_prompt, "node_token": self.node_token,
            "distance": self.distance, "status": self.status, "dose_scale": self.dose_scale,
            "chart_sha256": array_digest(self.chart),
            "coexact_delta_sha256": array_digest(self.coexact_delta),
            "random_delta_sha256": array_digest(self.random_delta),
        }


def query_hidden(field: FrozenPrefixField, hidden: np.ndarray, *, seed: int, alpha: float) -> PrefixDirection:
    """No chart fitting, velocity estimation, target ID, or suffix is accepted."""
    h = np.asarray(hidden, dtype=np.float64)
    if h.shape != field.mean.shape or not np.isfinite(h).all() or np.linalg.norm(h) <= 1e-12:
        raise ValueError("expected one finite nonzero prefix hidden vector")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0 or not np.isfinite(alpha):
        raise ValueError("invalid seed or alpha")
    chart = (h / np.linalg.norm(h) - field.mean) @ field.components.T
    distances = np.linalg.norm(field.points - chart, axis=1)
    index = int(np.argmin(distances))
    vector = field.coexact[index]
    hidden_vector = hodge.pca_chart_vectors_to_hidden(vector, field)
    hidden_norm = float(np.linalg.norm(hidden_vector))
    status = "SUPPORTED"
    if distances[index] > field.support_limit:
        status = "OUT_OF_SUPPORT"
    elif np.linalg.norm(vector) < field.min_chart_norm or hidden_norm <= 0:
        status = "INACTIVE_COEXACT"
    coexact_delta = np.zeros_like(h)
    random_delta = np.zeros_like(h)
    if status == "SUPPORTED":
        random_chart = np.random.default_rng(np.random.SeedSequence([int(seed), index])).normal(size=len(vector))
        random_hidden = hodge.pca_chart_vectors_to_hidden(random_chart, field)
        random_norm = np.linalg.norm(random_hidden)
        if not np.isfinite(random_norm) or random_norm <= 0:
            raise ValueError("degenerate random control; no fallback or redraw")
        magnitude = float(alpha) * field.dose_scale
        coexact_delta = magnitude * hidden_vector / hidden_norm
        random_delta = magnitude * random_hidden / random_norm
    return PrefixDirection(index, field.node_prompts[index], field.node_tokens[index], float(distances[index]),
                           status, chart, coexact_delta, random_delta, field.dose_scale)


def direction_for_prefix(
    field: FrozenPrefixField, prefix_ids: Sequence[int],
    hidden_provider: Callable[[tuple[int, ...]], np.ndarray], *, seed: int, alpha: float,
) -> PrefixDirection:
    """The provider sees only immutable observed IDs; the scorer stays separate."""
    if not len(prefix_ids) or any(not isinstance(t, Integral) or isinstance(t, bool) or t < 0 for t in prefix_ids):
        raise ValueError("prefix must contain nonnegative integer token IDs")
    observed = tuple(int(t) for t in prefix_ids)
    hidden = hidden_provider(observed)
    return query_hidden(field, hidden, seed=seed, alpha=alpha)


def direction_at_prefix(
    field: FrozenPrefixField, token_ids: Sequence[int], prefix_length: int,
    hidden_provider: Callable[[tuple[int, ...]], np.ndarray], *, seed: int, alpha: float,
) -> PrefixDirection:
    """Dataset adapter: truncate before hidden extraction; never read the target."""
    if (not isinstance(prefix_length, Integral) or isinstance(prefix_length, bool)
            or prefix_length < 1 or prefix_length > len(token_ids)):
        raise ValueError("unavailable prefix length")
    return direction_for_prefix(field, token_ids[:prefix_length], hidden_provider, seed=seed, alpha=alpha)
