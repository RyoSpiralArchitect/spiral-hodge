from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from scripts.continue_hltd_l8_gate import require_receipts, validate_prior_attempt
from scripts.evaluate_hltd_signed_layer_gate import file_receipt


class TestHLTDL8Continuation(unittest.TestCase):
    def test_missing_or_changed_required_sources_are_not_silently_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw.csv"
            source.write_text("a,b\n1,2\n")
            record = file_receipt(source)
            require_receipts([record], root)
            source.write_text("a,b\n1,3\n")
            with self.assertRaisesRegex(ValueError, "changed required input"):
                require_receipts([record], root)
            with self.assertRaises(FileNotFoundError):
                require_receipts([{**record, "path": str(root / 'missing.csv')}], root)

    def test_continuation_only_repairs_input_failure_before_l8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = {"run_root": "attempt", "bridge": {"reference_raw": "missing.csv"}}
            receipt = {"status": "FAILED", "error": "FileNotFoundError: [Errno 2] No such file or directory: " + repr(str(root / "missing.csv"))}
            validate_prior_attempt(protocol, receipt, root)
            for key, value in [("status", "STOPPED_BEFORE_L8"), ("error", "ValueError: bridge sensitive")]:
                changed = copy.deepcopy(receipt)
                changed[key] = value
                with self.assertRaisesRegex(ValueError, "only the missing"):
                    validate_prior_attempt(protocol, changed, root)
            (root / "attempt/l8").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "already started"):
                validate_prior_attempt(protocol, receipt, root)


if __name__ == "__main__":
    unittest.main()
