import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import spiral_hodge as hodge


class TestHFModelRefResolution(unittest.TestCase):
    def test_model_path_overrides_hub_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "local-gpt2"
            model_dir.mkdir()

            ref, is_local = hodge.resolve_hf_model_ref("gpt2", str(model_dir))

            self.assertEqual(ref, str(model_dir.resolve()))
            self.assertTrue(is_local)

    def test_model_argument_can_be_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model" / "gpt2"
            model_dir.mkdir(parents=True)

            ref, is_local = hodge.resolve_hf_model_ref(str(model_dir))

            self.assertEqual(ref, str(model_dir.resolve()))
            self.assertTrue(is_local)

    def test_missing_explicit_model_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing-gpt2"

            with self.assertRaisesRegex(FileNotFoundError, "Local model directory not found"):
                hodge.resolve_hf_model_ref("gpt2", str(missing))

    def test_bare_name_prefers_local_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "model" / "gpt2"
            model_dir.mkdir(parents=True)

            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                ref, is_local = hodge.resolve_hf_model_ref("gpt2")
            finally:
                os.chdir(old_cwd)

            self.assertEqual(ref, str(model_dir.resolve()))
            self.assertTrue(is_local)


class TestTextFilePathResolution(unittest.TestCase):
    def test_accepts_repo_name_prefixed_relative_path(self) -> None:
        path = hodge.resolve_text_file_path("./spiral-hodge/LICENSE")

        self.assertEqual(path, (hodge.SCRIPT_DIR / "LICENSE").resolve())

    def test_missing_text_file_suggests_shorter_relative_path(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "try --text-file ./missing.txt"):
            hodge.resolve_text_file_path("./spiral-hodge/missing.txt")


class TestSignedOrientationMetrics(unittest.TestCase):
    def test_trajectory_signed_circulation_flips_under_reversal(self) -> None:
        theta = np.linspace(0.0, 1.5 * np.pi, 12)
        coords = np.stack([np.cos(theta), np.sin(theta)], axis=1)[None, :, :]
        field = hodge.token_trajectory_field(coords, layer=0)
        rev_field = hodge.token_trajectory_field(coords[:, ::-1, :], layer=0)

        metric = hodge.signed_circulation_metrics(field.points, field.vectors)
        rev_metric = hodge.signed_circulation_metrics(rev_field.points, rev_field.vectors)

        self.assertGreater(metric["signed_circulation"], 0.0)
        self.assertAlmostEqual(metric["signed_circulation"], -rev_metric["signed_circulation"], places=10)
        self.assertAlmostEqual(
            metric["signed_circulation_alignment"],
            -rev_metric["signed_circulation_alignment"],
            places=10,
        )

    def test_spectral_signed_curl_flips_under_reversal(self) -> None:
        theta = np.linspace(0.0, 1.75 * np.pi, 16)
        coords = np.stack([np.cos(theta), np.sin(theta)], axis=1)[None, :, :]
        field = hodge.token_trajectory_field(coords, layer=0)
        rev_field = hodge.token_trajectory_field(coords[:, ::-1, :], layer=0)

        spec = hodge.vector_spectrum(field.points, field.vectors, modes=16, backend="direct")
        hspec = hodge.helmholtz_project_spectrum(spec)
        rev_spec = hodge.vector_spectrum(rev_field.points, rev_field.vectors, modes=16, backend="direct")
        rev_hspec = hodge.helmholtz_project_spectrum(rev_spec)

        metric = hodge.spectral_signed_curl_metrics(spec, hspec)
        rev_metric = hodge.spectral_signed_curl_metrics(rev_spec, rev_hspec)

        self.assertAlmostEqual(
            metric["signed_curl_alignment"],
            -rev_metric["signed_curl_alignment"],
            places=10,
        )
        self.assertAlmostEqual(
            metric["signed_vorticity_ratio"],
            -rev_metric["signed_vorticity_ratio"],
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
