import numpy as np
import pytest

from scripts import audit_hltd_local_response_failure as audit


@pytest.fixture
def example():
    from scripts.run_hltd_steering import _log_softmax
    hidden = np.ones((8, 3), dtype=np.float32)
    logits = np.tile(np.array([1., 2., 3.], dtype=np.float32), (12, 1))
    captured = {"hidden": hidden.copy(), "logits": logits.copy()}
    grad = {"hidden": np.broadcast_to(hidden, (12, *hidden.shape)).copy(), "logits": logits.copy(),
            "gradient": np.array([1., 0, 0], dtype=np.float32), "objective": float(_log_softmax(logits[0])[1])}
    method = {"baseline_logit_tolerance": 1e-4, "objective_logprob_tolerance": 1e-4}
    return hidden, logits, captured, grad, 1, method


def test_zero_audit_reports_exact_agreement_but_no_semantic_claim(example):
    result = audit.bridge_diagnostics(*example)
    assert result["original_bridge_passes"]
    assert result["semantic_direction_status"] == "NOT_TESTED" and result["nonzero_treatments"] == 0


def test_common_logit_shift_can_fail_gate_without_changing_probabilities(example):
    example[3]["logits"] += np.float32(1 / 1024)
    result = audit.bridge_diagnostics(*example)
    error = result["comparisons"]["autograd_vs_no_hook"]
    assert not result["original_bridge_passes"] and not error["within_original_logit_tolerance"]
    assert error["max_abs_logit_error"] == 1 / 1024
    assert error["max_abs_logprob_error"] == pytest.approx(0, abs=1e-14)
    assert result["original_logit_tolerance"] == 1e-4


def test_diagnostic_preserves_hidden_mismatch_failure(example):
    example[2]["hidden"][0, 0] += .5
    result = audit.bridge_diagnostics(*example)
    assert not result["captured_hidden_bytes_equal"] and not result["original_bridge_passes"]


def test_failure_receipt_is_pinned_before_any_loading(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError("changed canonical failed receipt")
    monkeypatch.setattr(audit.local.cal, "pinned_json", reject)
    with pytest.raises(ValueError, match="canonical"):
        audit.contract(tmp_path)
