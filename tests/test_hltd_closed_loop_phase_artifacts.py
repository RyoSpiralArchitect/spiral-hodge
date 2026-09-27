from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


class ClosedLoopPhaseArtifactsTest(unittest.TestCase):
    def test_compact_evidence_matches_receipt_without_local_raw_runs(self) -> None:
        root = Path(__file__).resolve().parents[1] / "docs"
        receipt = json.loads((root / "figures/hltd_closed_loop_phase_manifest.json").read_text())
        self.assertEqual(receipt["schema"], "hltd_closed_loop_phase_evidence.v1")
        self.assertEqual(len(receipt["artifacts"]), 5)
        for item in receipt["artifacts"]:
            with self.subTest(path=item["path"]):
                path = root / item["path"]
                payload = path.read_bytes()
                self.assertEqual(len(payload), item["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
                if "rows" in item:
                    with path.open(newline="") as handle:
                        self.assertEqual(sum(1 for _ in csv.reader(handle)) - 1, item["rows"])


if __name__ == "__main__":
    unittest.main()
