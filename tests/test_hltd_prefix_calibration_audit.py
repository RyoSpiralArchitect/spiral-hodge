import math

import pytest

from scripts import audit_hltd_prefix_calibration as audit


def test_independent_distribution_counts_missing_and_preserves_zero():
    result = audit.distribution([None, "", 0., -1., 1.])
    assert result == {"n_total": 5, "n_valid": 3, "mean": 0., "q10": -.8, "median": 0., "q90": .8}
    assert audit.distribution([None, ""])["mean"] is None


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_values_fail(value):
    with pytest.raises(ValueError, match="nonfinite"):
        audit.distribution([0., value])


def test_independent_grouping_uses_each_prompt_and_keeps_missing():
    rows = [{"prompt_id": "a", "value": "1"}, {"prompt_id": "a", "value": "0"}, {"prompt_id": "b", "value": "-1"}]
    assert audit.grouped_means(rows, "value", ["a", "b", "c"]) == {"a": .5, "b": -1., "c": None}


def test_changed_receipts_are_rejected(tmp_path):
    path = tmp_path / "data"
    path.write_text("original")
    record = audit.receipt(path)
    audit.check_receipt(record)
    path.write_text("different")
    with pytest.raises(ValueError, match="changed artifact"):
        audit.check_receipt(record)
