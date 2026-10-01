from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest as modes


@pytest.mark.parametrize("module,name,fixtures,expected", [
    ("test_hltd_prefix_execution.py", "test_execution_plan_reuses_immutable_preparation", [], True),
    ("test_hltd_prefix_v2.py", "test_any_seed_with_inactive_control_invalidates_cell", ["protocol"], True),
    ("test_hltd_prefix_v2.py", "test_suffix_replacement_is_byte_invariant", ["field"], False),
    ("test_hltd_prefix_portable_execution.py", "test_any_seed_with_inactive_control_invalidates_cell", ["protocol"], False),
    ("test_hltd_prefix_calibration.py", "test_real_contract_only_opens_pinned_calibration_inputs", [], True),
    ("test_hltd_prefix_nulls.py", "test_stage_a_contract_never_opens_v2_responses_or_model_assets", [], True),
    ("test_hltd_prefix_nulls.py", "test_new_unrelated_check", [], False),
])
def test_only_explicit_host_bound_cases_are_classified(module, name, fixtures, expected):
    item = SimpleNamespace(path=Path(module), name=name, fixturenames=fixtures)
    assert modes.needs_local_evidence(item) is expected


@pytest.mark.parametrize("enabled", [False, True])
def test_opt_in_runs_original_checks_without_fallback(enabled):
    markers = []
    item = SimpleNamespace(path=Path("test_hltd_prefix_v2.py"), name="test_case",
                           fixturenames=["protocol"], add_marker=markers.append)
    config = SimpleNamespace(getoption=lambda _name: enabled)
    modes.pytest_collection_modifyitems(config, [item])
    assert [mark.name for mark in markers] == (["local_evidence"] if enabled else ["local_evidence", "skip"])
