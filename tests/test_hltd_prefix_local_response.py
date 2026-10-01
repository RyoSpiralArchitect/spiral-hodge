from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scripts import run_hltd_prefix_local_response as local


@pytest.fixture
def method():
    return json.loads((local.ROOT / local.METHOD).read_text())


@pytest.fixture
def toy():
    import torch
    from torch import nn

    class Block(nn.Module):
        def forward(self, x):
            return (x,)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(5, 3, device="cpu")
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([Block() for _ in range(7)])
            self.projection = nn.Linear(3, 5, bias=False, device="cpu")
            self.fail = False

        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            for block in self.transformer.h:
                x = block(x)[0]
            if self.fail:
                raise RuntimeError("after hook")
            return SimpleNamespace(logits=self.projection(x))

    torch.manual_seed(21)
    model = Model().eval().requires_grad_(False)
    return model, {"input_ids": torch.tensor([[1, 2, 3]], device="cpu"), "use_cache": False}


def test_leaf_gradient_matches_analytic_softmax_and_does_not_touch_weights(toy):
    import torch
    model, inputs = toy
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    ledger = {}
    result = local.activation_gradient(model, inputs, 2, ledger=ledger)
    w = model.projection.weight.detach().cpu().numpy()
    logits = result["logits"][0].astype(float)
    p = np.exp(logits - np.max(logits))
    p /= p.sum()
    expected = w[2] - p @ w
    np.testing.assert_allclose(result["gradient"], expected, atol=2e-7, rtol=2e-6)
    assert result["hidden"].shape == (12, 3, 3)
    np.testing.assert_array_equal(result["logits"], np.broadcast_to(result["logits"][:1], result["logits"].shape))
    assert ledger == {"gradient_forwards_attempted": 1, "backwards_attempted": 1, "gradients_completed": 1}
    for key, value in model.named_parameters():
        assert torch.equal(value, before[key]) and value.grad is None and not value.requires_grad
    assert len(model.transformer.h[6]._forward_hooks) == 0


def test_mean_objective_is_not_multiplied_by_batch_count(toy):
    model, inputs = toy
    single = local.activation_gradient(model, inputs, 1, batch=1)
    repeated = local.activation_gradient(model, inputs, 1, batch=12)
    np.testing.assert_allclose(single["gradient"], repeated["gradient"], atol=2e-7, rtol=2e-6)


def test_failed_gradient_removes_hook_and_records_attempt(toy):
    model, inputs = toy
    model.fail = True
    ledger = {}
    with pytest.raises(RuntimeError, match="after hook"):
        local.activation_gradient(model, inputs, 1, ledger=ledger)
    assert len(model.transformer.h[6]._forward_hooks) == 0
    assert ledger == {"gradient_forwards_attempted": 1}


def test_training_or_parameter_gradients_rejected_before_forward(toy):
    import torch
    model, inputs = toy
    for change in (lambda: model.train(), lambda: model.requires_grad_(True),
                   lambda: setattr(model.projection.weight, "grad", torch.ones_like(model.projection.weight))):
        model.eval().requires_grad_(False)
        model.projection.weight.grad = None
        change()
        ledger = {}
        with pytest.raises(ValueError, match="frozen"):
            local.activation_gradient(model, inputs, 1, ledger=ledger)
        assert ledger == {}


def test_bridge_matches_last_prefix_without_exposing_future_tokens(toy, method):
    model, inputs = toy
    result = local.activation_gradient(model, inputs, 1)
    hidden = result["hidden"][0]
    logits = result["logits"]
    capture = {"hidden": hidden.copy(), "logits": logits.copy()}
    check = local.validate_bridge(hidden, logits, capture, result, 1, method)
    assert check["autograd_vs_saved"] == 0 and check["gradient_norm"] > 0


@pytest.mark.parametrize("kind", ["hidden", "logit", "gradient_zero", "gradient_nan", "objective_nan", "spread"])
def test_bridge_rejects_invalid_observations(toy, method, kind):
    model, inputs = toy
    result = local.activation_gradient(model, inputs, 1)
    hidden, logits = result["hidden"][0].copy(), result["logits"].copy()
    captured = {"hidden": hidden.copy(), "logits": logits.copy()}
    if kind == "hidden":
        captured["hidden"][0, 0] += .1
    elif kind == "logit":
        captured["logits"] += 1
    elif kind == "gradient_zero":
        result["gradient"][:] = 0
    elif kind == "gradient_nan":
        result["gradient"][:] = np.nan
    elif kind == "objective_nan":
        result["objective"] = float("nan")
    else:
        result["logits"][0] += 1
    with pytest.raises(ValueError):
        local.validate_bridge(hidden, logits, captured, result, 1, method)


def test_small_protocol_only_changes_signed_doses_and_preserves_original(method):
    prior = {"spec": {"evaluation": {"alphas": [-1, 1], "seeds": list(range(8))}}, "prompts": ["frozen"]}
    before = copy.deepcopy(prior)
    small = local.small_protocol(prior, method)
    assert prior == before
    assert small["spec"]["evaluation"]["alphas"] == [-.04, -.02, -.01, .01, .02, .04]
    assert small["prompts"] == prior["prompts"]


def test_directional_derivative_distinguishes_even_cost_and_cubic_nonlinearity(method):
    def f(x):
        return -5 + 2 * x - 3 * x**2 + 7 * x**3
    for a in method["magnitudes"]:
        row = local.directional_statistics([2., 0], [1., 0], f(a), f(-a), f(0), a, method)
        assert row["local_slope"] == 2 and row["cosine"] == 1
        assert row["finite_slope"] == pytest.approx(2 + 7 * a**2)
        assert row["even_delta"] == pytest.approx(-3 * a**2)
        assert row["odd_delta"] == pytest.approx(2 * a + 7 * a**3)


def test_gradient_validation_does_not_promote_semantic_claims(method):
    positive = local.directional_statistics([1., 0], [1., 0], -4.99, -5.01, -5., .01, method)
    result = local.disposition([positive], 1)
    assert result == {"numerical_status": "LOCAL_DERIVATIVE_VALIDATED", "semantic_direction_status": "NOT_TESTED"}
    changed = {**positive, "within_tolerance": False}
    result = local.disposition([changed], 1)
    assert result["numerical_status"] == "LOCAL_DERIVATIVE_NOT_VALIDATED"
    assert result["semantic_direction_status"] == "NOT_TESTED"
    with pytest.raises(ValueError, match="incomplete"):
        local.disposition([positive], 2)


@pytest.mark.parametrize("gradient,delta,a", [([0., 0], [1., 0], .01), ([1., 0], [0., 0], .01),
                                            ([np.nan, 0], [1., 0], .01), ([1., 0], [1., 0], 0)])
def test_undefined_derivatives_are_not_filled_with_zero(method, gradient, delta, a):
    with pytest.raises(ValueError, match="undefined"):
        local.directional_statistics(gradient, delta, -5, -5, -5, a, method)


def test_source_tampering_invalidates_frozen_contract(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("original")
    def contract(root):
        return {"frozen_files": [local.file_receipt(source)]}
    monkeypatch.setattr(local, "contract", contract)
    path = tmp_path / "protocol.json"
    protocol = local.freeze(path, tmp_path)
    with pytest.raises(FileExistsError):
        local.freeze(path, tmp_path)
    source.write_text("changed")
    with pytest.raises(ValueError):
        local.validate(protocol, tmp_path)


def test_invalid_protocol_stops_before_model_or_output(tmp_path, monkeypatch):
    path = tmp_path / "protocol.json"
    path.write_text("{}")
    def reject(*args):
        raise ValueError("test invalid input")
    monkeypatch.setattr(local, "validate", reject)
    with patch.object(local.v1, "_load_model_and_tokenizer") as load:
        with pytest.raises(ValueError, match="invalid input"):
            local.run(path, tmp_path)
        load.assert_not_called()
    assert not (tmp_path / local.RUN).exists()
