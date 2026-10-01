"""Keep frozen, host-bound integration checks explicit without editing them."""
from pathlib import Path

import pytest


LOCAL_FUNCTIONS = {
    ("test_hltd_prefix_execution.py", "test_execution_plan_reuses_immutable_preparation"),
    ("test_hltd_prefix_calibration.py", "test_real_contract_only_opens_pinned_calibration_inputs"),
    ("test_hltd_prefix_nulls.py", "test_stage_a_contract_never_opens_v2_responses_or_model_assets"),
}


def needs_local_evidence(item):
    module = Path(item.path).name
    name = getattr(item, "originalname", None) or item.name
    return ((module, name) in LOCAL_FUNCTIONS
            or (module == "test_hltd_prefix_v2.py" and "protocol" in item.fixturenames))


def pytest_addoption(parser):
    parser.addoption("--hltd-local-evidence", action="store_true", default=False,
                     help="Run host-bound frozen-evidence checks; missing or changed inputs fail.")


def pytest_configure(config):
    config.addinivalue_line("markers", "local_evidence: requires the original pinned local HLTD artifacts/runtime")


def pytest_collection_modifyitems(config, items):
    enabled = config.getoption("--hltd-local-evidence")
    for item in items:
        if needs_local_evidence(item):
            item.add_marker(pytest.mark.local_evidence)
            if not enabled:
                item.add_marker(pytest.mark.skip(reason="requires --hltd-local-evidence and the original pinned inputs"))
