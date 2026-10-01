from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from scripts import hltd_prefix_transfer as transfer


@pytest.fixture(scope="module")
def calibration():
    rng = np.random.default_rng(413)
    return {name: rng.normal(size=(23, 12)).astype(np.float32) for name in ("cal_a", "cal_b", "cal_c")}


@pytest.fixture(scope="module")
def field(calibration):
    return transfer.build_frozen_field(calibration, n_components=6, k=4, min_chart_norm=1e-12)


def test_field_uses_only_calibration_and_never_connects_trajectory_endpoints(calibration, field):
    assert len(field.points) == sum(len(h) - 2 for h in calibration.values())
    assert set(field.node_prompts) == set(calibration)
    for name in calibration:
        indices = [t for owner, t in zip(field.node_prompts, field.node_tokens, strict=True) if owner == name]
        assert indices == list(range(1, 22))
    expected_scale = np.median([np.median(np.linalg.norm(.5 * (h[2:].astype(float) - h[:-2]), axis=1))
                                for h in calibration.values()])
    assert field.dose_scale == pytest.approx(expected_scale, abs=1e-12)
    owners = np.asarray(field.node_prompts)
    distances = np.concatenate([cdist(field.points[owners == name], field.points[owners != name]).min(axis=1)
                                for name in sorted(calibration)])
    assert field.support_limit == np.quantile(distances, .99, method="higher")


def test_input_order_is_not_a_field_hyperparameter(calibration, field):
    reversed_input = dict(reversed(list(calibration.items())))
    other = transfer.build_frozen_field(reversed_input, n_components=6, k=4, min_chart_norm=1e-12)
    assert other.fingerprint() == field.fingerprint()


@pytest.mark.parametrize("prefix_length", [8, 16, 24])
@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("alpha", [-1., .25, 1.])
def test_suffix_replacement_extension_and_removal_are_byte_invariant(calibration, field, prefix_length, seed, alpha):
    prefix = tuple(range(prefix_length))
    calls = []

    def hidden_provider(observed):
        assert observed == prefix
        assert isinstance(observed, tuple)
        calls.append(observed)
        return calibration["cal_a"][7]

    original = prefix + (40, 41, 42)
    variants = [original, prefix + (99, 98), original + (100,) * 200, prefix]
    before = field.fingerprint()
    with patch.object(transfer.PCA, "fit", side_effect=AssertionError("no evaluation refit")), \
            patch.object(transfer.hodge, "hodge_latent_traversal_dynamics_matched_betti",
                         side_effect=AssertionError("no evaluation decomposition")):
        directions = [transfer.direction_at_prefix(field, ids, prefix_length, hidden_provider, seed=seed, alpha=alpha)
                      for ids in variants]
    assert len(calls) == 4
    assert directions[0].status == "SUPPORTED"
    assert all(result.receipt() == directions[0].receipt() for result in directions)
    assert field.fingerprint() == before


def test_hidden_norm_does_not_set_evaluation_dose(calibration, field):
    original = transfer.query_hidden(field, calibration["cal_a"][7], seed=0, alpha=.5)
    scaled = transfer.query_hidden(field, calibration["cal_a"][7].astype(float) * 2, seed=0, alpha=.5)
    assert scaled.receipt() == original.receipt()
    assert np.linalg.norm(original.coexact_delta) == pytest.approx(.5 * field.dose_scale)
    assert np.linalg.norm(original.random_delta) == pytest.approx(.5 * field.dose_scale)


def test_sign_pairing_and_random_seed_do_not_change_coexact(calibration, field):
    h = calibration["cal_a"][7]
    plus = transfer.query_hidden(field, h, seed=0, alpha=1)
    minus = transfer.query_hidden(field, h, seed=0, alpha=-1)
    different_seed = transfer.query_hidden(field, h, seed=7, alpha=1)
    assert plus.status == "SUPPORTED"
    np.testing.assert_array_equal(plus.coexact_delta, -minus.coexact_delta)
    np.testing.assert_array_equal(plus.random_delta, -minus.random_delta)
    np.testing.assert_array_equal(plus.coexact_delta, different_seed.coexact_delta)
    assert not np.array_equal(plus.random_delta, different_seed.random_delta)


def test_out_of_support_and_inactive_fields_never_fall_back_to_another_node(calibration, field):
    h = calibration["cal_a"][7]
    inactive = replace(field, coexact=np.zeros_like(field.coexact))
    response = transfer.query_hidden(inactive, h, seed=0, alpha=1)
    assert response.status == "INACTIVE_COEXACT"
    assert not response.coexact_delta.any() and not response.random_delta.any()
    strict = replace(field, support_limit=0.)
    response = transfer.query_hidden(strict, h.astype(float) + .0001, seed=0, alpha=1)
    assert response.status == "OUT_OF_SUPPORT"
    assert not response.coexact_delta.any() and not response.random_delta.any()


def test_field_and_direction_arrays_are_immutable(calibration, field):
    before = field.fingerprint()
    for name in ("mean", "components", "points", "coexact"):
        with pytest.raises(ValueError):
            getattr(field, name).setflags(write=True)
    delta = transfer.query_hidden(field, calibration["cal_a"][7], seed=0, alpha=1)
    with pytest.raises(ValueError):
        delta.coexact_delta[0] = 0
    assert before == field.fingerprint()


@pytest.mark.parametrize("change", ["one_prompt", "too_short", "nonfinite", "zero", "dimensions", "pca", "k", "ridge"])
def test_invalid_calibration_is_rejected(calibration, change):
    data = {key: value.copy() for key, value in calibration.items()}
    kwargs = {"n_components": 6, "k": 4}
    if change == "one_prompt":
        data = {"cal_a": data["cal_a"]}
    elif change == "too_short":
        data["cal_a"] = data["cal_a"][:6]
    elif change in {"nonfinite", "zero"}:
        data["cal_a"][0] = np.nan if change == "nonfinite" else 0
    elif change == "dimensions":
        data["cal_a"] = data["cal_a"][:, :-1]
    elif change == "pca":
        kwargs["n_components"] = 100
    elif change == "k":
        kwargs["k"] = 0
    else:
        kwargs["node_ridge"] = -1
    with pytest.raises(ValueError):
        transfer.build_frozen_field(data, **kwargs)


@pytest.mark.parametrize("ids,length", [([], 0), ([1, 2], 3), ([1, 2], 1.5), ([1, 2], True), ([-1, 2], 1), ([True, 2], 1)])
def test_invalid_observed_prefix_is_rejected_before_provider(field, ids, length):
    with patch("builtins.print") as provider:
        with pytest.raises(ValueError):
            transfer.direction_at_prefix(field, ids, length, provider, seed=0, alpha=1)
        provider.assert_not_called()


@pytest.mark.parametrize("hidden", [np.zeros(12), np.full(12, np.nan), np.zeros((2, 12)), np.ones(11)])
def test_invalid_query_hidden_state_is_rejected(field, hidden):
    with pytest.raises(ValueError):
        transfer.query_hidden(field, hidden, seed=0, alpha=1)
