from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]


class TestInstallationContract(unittest.TestCase):
    def test_python_floor_supports_the_audit_hash_api(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        versions = SpecifierSet(project["requires-python"])
        self.assertNotIn("3.10", versions)
        self.assertIn("3.11", versions)
        self.assertIn("3.12", versions)

    def test_both_installation_routes_include_analysis_dependencies(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        packaged = {Requirement(value).name for value in project["dependencies"]}
        requirements = {
            Requirement(line).name
            for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("pandas", packaged)
        self.assertEqual(packaged, requirements)


if __name__ == "__main__":
    unittest.main()
