"""Model-free counterparts reuse assertions, not the host-bound contract fixture."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_hltd_prefix_v2 import (
    atlas as atlas,
    field as field,
    test_any_seed_with_inactive_control_invalidates_cell as test_any_seed_with_inactive_control_invalidates_cell,
    test_complete_pilot_uses_only_prefixes_and_zero_hooks as test_complete_pilot_uses_only_prefixes_and_zero_hooks,
    test_existing_output_is_not_overwritten as test_existing_output_is_not_overwritten,
    test_gate_failure_prevents_nonzero_forwards as test_gate_failure_prevents_nonzero_forwards,
    test_protocol_changes_fail_before_model_load as test_protocol_changes_fail_before_model_load,
    test_query_replay_detects_changed_neighbor_weights as test_query_replay_detects_changed_neighbor_weights,
    test_real_adapter_and_treatment_only_pass_observed_ids as test_real_adapter_and_treatment_only_pass_observed_ids,
)


@pytest.fixture(scope="module")
def protocol():
    path = Path(__file__).resolve().parents[1] / "docs/data/hltd_prefix_transfer_v2/protocol.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "88666f7a5d35c4e2f7dde490a1ce3ae19fa2ef08ef5555908298adca0a16f76c"
    # Settings and token IDs for fake runtimes only, not a real-input validation.
    return json.loads(raw)
