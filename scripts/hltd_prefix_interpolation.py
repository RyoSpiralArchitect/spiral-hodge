"""V2 calibration-only readout. The frozen v1 field is never modified."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.spatial.distance import cdist

from scripts import hltd_prefix_transfer as base


@dataclass(frozen=True)
class InterpolatedField:
    atlas: base.FrozenPrefixField
    neighbors: int
    bandwidth: float
    kth_support_limit: float

    def __post_init__(self):
        if (not isinstance(self.neighbors, Integral) or isinstance(self.neighbors, bool)
                or not 1 <= self.neighbors <= len(self.atlas.points)):
            raise ValueError("invalid interpolation neighbor count")
        if not np.isfinite([self.bandwidth, self.kth_support_limit]).all() or min(self.bandwidth, self.kth_support_limit) <= 0:
            raise ValueError("invalid calibration-only interpolation scale")

    def receipt(self):
        return {"atlas_fingerprint": self.atlas.fingerprint(), "neighbors": self.neighbors,
                "bandwidth": self.bandwidth, "kth_support_limit": self.kth_support_limit}

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.receipt(), sort_keys=True, allow_nan=False).encode()).hexdigest()


def calibrate_readout(atlas: base.FrozenPrefixField, *, neighbors: int = 8, quantile: float = .99) -> InterpolatedField:
    if not isinstance(neighbors, Integral) or isinstance(neighbors, bool) or neighbors < 1 or not 0 <= quantile <= 1:
        raise ValueError("invalid readout configuration")
    owners = np.asarray(atlas.node_prompts)
    kth = []
    for prompt in sorted(set(atlas.node_prompts)):
        mask = owners == prompt
        if int((~mask).sum()) < neighbors:
            raise ValueError("not enough other-prompt calibration nodes")
        distances = cdist(atlas.points[mask], atlas.points[~mask])
        kth.extend(np.partition(distances, neighbors - 1, axis=1)[:, neighbors - 1])
    return InterpolatedField(atlas, int(neighbors), float(np.median(kth)),
                             float(np.quantile(kth, quantile, method="higher")))


@dataclass(frozen=True)
class InterpolatedDirection(base.PrefixDirection):
    neighbor_indices: tuple[int, ...]
    neighbor_distances: tuple[float, ...]
    weights: tuple[float, ...]
    coexact_chart_norm: float
    random_chart_norm: float
    inactive_neighbors: int

    def receipt(self):
        return {**super().receipt(), "neighbor_indices": list(self.neighbor_indices),
                "neighbor_distances": list(self.neighbor_distances), "weights": list(self.weights),
                "coexact_chart_norm": self.coexact_chart_norm, "random_chart_norm": self.random_chart_norm,
                "inactive_neighbors": self.inactive_neighbors}


def query_hidden(field: InterpolatedField, hidden, *, seed: int, alpha: float) -> InterpolatedDirection:
    atlas = field.atlas
    h = np.asarray(hidden, dtype=np.float64)
    norm = float(np.linalg.norm(h))
    if h.shape != atlas.mean.shape or not np.isfinite(h).all() or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("expected one finite nonzero prefix hidden vector")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0 or not np.isfinite(alpha):
        raise ValueError("invalid seed or alpha")
    chart = (h / norm - atlas.mean) @ atlas.components.T
    distances = np.linalg.norm(atlas.points - chart, axis=1)
    indices = np.argsort(distances, kind="stable")[:field.neighbors]
    local_distances = distances[indices]
    log_weights = -.5 * (local_distances / field.bandwidth) ** 2
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    vectors = atlas.coexact[indices]
    norms = np.linalg.norm(vectors, axis=1)
    vector = weights @ vectors
    random_nodes = []
    for index, node_norm in zip(indices, norms, strict=True):
        random = np.random.default_rng(np.random.SeedSequence([int(seed), int(index)])).normal(size=len(chart))
        random_norm = np.linalg.norm(random)
        if not np.isfinite(random_norm) or random_norm <= 0:
            raise ValueError("degenerate random node direction; no redraw")
        random_nodes.append(node_norm * random / random_norm)
    random_vector = weights @ np.asarray(random_nodes)
    coexact_norm, random_norm = float(np.linalg.norm(vector)), float(np.linalg.norm(random_vector))
    if not np.isfinite([coexact_norm, random_norm]).all():
        raise ValueError("nonfinite interpolated direction")
    status = "SUPPORTED"
    if local_distances[0] > atlas.support_limit or local_distances[-1] > field.kth_support_limit:
        status = "OUT_OF_SUPPORT"
    elif coexact_norm < atlas.min_chart_norm:
        status = "INACTIVE_COEXACT"
    elif random_norm < atlas.min_chart_norm:
        status = "INACTIVE_RANDOM"
    deltas = [np.zeros_like(h), np.zeros_like(h)]
    if status == "SUPPORTED":
        for i, value in enumerate((vector, random_vector)):
            hidden_vector = base.hodge.pca_chart_vectors_to_hidden(value, atlas)
            hidden_norm = np.linalg.norm(hidden_vector)
            if not np.isfinite(hidden_norm) or hidden_norm <= 0:
                raise ValueError("invalid backprojected direction")
            deltas[i] = float(alpha) * atlas.dose_scale * hidden_vector / hidden_norm
    nearest = int(indices[0])
    return InterpolatedDirection(nearest, atlas.node_prompts[nearest], atlas.node_tokens[nearest],
        float(local_distances[0]), status, chart, *deltas, atlas.dose_scale,
        tuple(int(i) for i in indices), tuple(float(x) for x in local_distances), tuple(float(x) for x in weights),
        coexact_norm, random_norm, int((norms < atlas.min_chart_norm).sum()))


def direction_at_prefix(field, token_ids, prefix_length, hidden_provider, *, seed, alpha):
    if (not isinstance(prefix_length, Integral) or isinstance(prefix_length, bool)
            or prefix_length < 1 or prefix_length > len(token_ids)):
        raise ValueError("unavailable prefix length")
    observed = token_ids[:prefix_length]
    if any(not isinstance(t, Integral) or isinstance(t, bool) or t < 0 for t in observed):
        raise ValueError("prefix must contain nonnegative integer token IDs")
    return query_hidden(field, hidden_provider(tuple(int(t) for t in observed)), seed=seed, alpha=alpha)
