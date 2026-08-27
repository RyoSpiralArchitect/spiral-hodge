from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "figures" / "hltd_signed_position_manifest.json"


class TestHLTDSignedPositionArtifacts(unittest.TestCase):
    def test_tracked_artifacts_match_manifest(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        artifacts = [*manifest["tracked_data"], *manifest["tracked_figures"]]
        self.assertGreater(len(artifacts), 0)

        for artifact in artifacts:
            with self.subTest(path=artifact["path"]):
                path = ROOT / artifact["path"]
                self.assertTrue(path.is_file())
                payload = path.read_bytes()
                self.assertEqual(len(payload), int(artifact["bytes"]))
                self.assertEqual(hashlib.sha256(payload).hexdigest(), artifact["sha256"])
                if "rows" in artifact:
                    with path.open(newline="", encoding="utf-8") as handle:
                        rows = sum(1 for _ in csv.reader(handle)) - 1
                    self.assertEqual(rows, int(artifact["rows"]))


if __name__ == "__main__":
    unittest.main()
