import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import spiral_hodge as hodge
import spiral_hodge_report as report


def _jax_available() -> bool:
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except Exception:
        return False
    return True


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

    def test_trajectory_turning_flips_under_reversal(self) -> None:
        theta = np.linspace(0.0, 1.75 * np.pi, 16)
        coords = np.stack([np.cos(theta), np.sin(theta)], axis=1)[None, :, :]
        field = hodge.token_trajectory_field(coords, layer=0)
        rev_field = hodge.token_trajectory_field(coords[:, ::-1, :], layer=0)

        metric = hodge.trajectory_turning_metrics(field.vectors)
        rev_metric = hodge.trajectory_turning_metrics(rev_field.vectors)

        self.assertGreater(metric["signed_angle"], 0.0)
        self.assertAlmostEqual(metric["signed_angle"], -rev_metric["signed_angle"], places=10)
        self.assertAlmostEqual(metric["alignment"], -rev_metric["alignment"], places=10)

    def test_local_jacobian_vorticity_flips_under_reversal(self) -> None:
        theta = np.linspace(0.0, 1.75 * np.pi, 18)
        coords = np.stack([np.cos(theta), np.sin(theta)], axis=1)[None, :, :]
        field = hodge.token_trajectory_field(coords, layer=0)
        rev_field = hodge.token_trajectory_field(coords[:, ::-1, :], layer=0)

        metric = hodge.local_jacobian_vorticity_metrics(field.points, field.vectors, k_neighbors=6)
        rev_metric = hodge.local_jacobian_vorticity_metrics(rev_field.points, rev_field.vectors, k_neighbors=6)

        self.assertAlmostEqual(
            metric["signed_vorticity_mean"],
            -rev_metric["signed_vorticity_mean"],
            places=10,
        )
        self.assertAlmostEqual(metric["abs_vorticity_mean"], rev_metric["abs_vorticity_mean"], places=10)

    def test_spectral_curl_bands_partition_curl_energy(self) -> None:
        theta = np.linspace(0.0, 1.75 * np.pi, 16)
        coords = np.stack([np.cos(theta), np.sin(theta)], axis=1)[None, :, :]
        field = hodge.token_trajectory_field(coords, layer=0)

        spec = hodge.vector_spectrum(field.points, field.vectors, modes=16, backend="direct")
        hspec = hodge.helmholtz_project_spectrum(spec)
        bands = hodge.spectral_curl_band_metrics(spec, hspec)

        self.assertAlmostEqual(
            bands["curl_low_ratio"] + bands["curl_mid_ratio"] + bands["curl_high_ratio"],
            hspec.energy["curl_ratio"],
            places=12,
        )
        self.assertAlmostEqual(
            bands["curl_low_band_ratio"] + bands["curl_mid_band_ratio"] + bands["curl_high_band_ratio"],
            1.0,
            places=12,
        )


@unittest.skipUnless(_jax_available(), "JAX is not installed")
class TestJaxFourierBackend(unittest.TestCase):
    def _field(self) -> hodge.VectorFieldBundle:
        theta = np.linspace(0.0, 1.8 * np.pi, 18)
        radius = np.linspace(0.7, 1.2, theta.size)
        coords = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)[None, :, :]
        return hodge.token_trajectory_field(coords, layer=0)

    def test_jax_spectrum_matches_direct_backend(self) -> None:
        field = self._field()

        direct = hodge.vector_spectrum(field.points, field.vectors, modes=8, backend="direct")
        jax_spec = hodge.vector_spectrum(field.points, field.vectors, modes=8, backend="jax")

        self.assertEqual(jax_spec.backend, "jax")
        np.testing.assert_allclose(jax_spec.coeffs, direct.coeffs, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(jax_spec.power, direct.power, rtol=1e-4, atol=1e-5)

    def test_jax_signed_curl_matches_direct_backend(self) -> None:
        field = self._field()

        direct = hodge.vector_spectrum(field.points, field.vectors, modes=8, backend="direct")
        direct_metric = hodge.spectral_signed_curl_metrics(direct, hodge.helmholtz_project_spectrum(direct))
        jax_spec = hodge.vector_spectrum(field.points, field.vectors, modes=8, backend="jax")
        jax_metric = hodge.spectral_signed_curl_metrics(jax_spec, hodge.helmholtz_project_spectrum(jax_spec))

        np.testing.assert_allclose(
            jax_metric["signed_curl_alignment"],
            direct_metric["signed_curl_alignment"],
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            jax_metric["signed_vorticity_ratio"],
            direct_metric["signed_vorticity_ratio"],
            rtol=1e-4,
            atol=1e-5,
        )


class TestReportGeneration(unittest.TestCase):
    def _sample_rows(self):
        return [
            {
                "variant": "real",
                "layer": 0,
                "layers": 2,
                "tokens": 8,
                "dim": 4,
                "spectral_curl_ratio": 0.4,
                "hodge_curl_ratio": 0.2,
                "graph_high_freq_ratio": 0.1,
                "trajectory_signed_circulation_alignment": -0.2,
                "turning_alignment": -0.25,
                "local_signed_vorticity_ratio": -0.35,
                "spectral_curl_high_ratio": 0.04,
                "spectral_signed_curl_alignment": -0.3,
                "hodge_signed_curl_alignment": 0.01,
                "spectral_signed_vorticity_ratio": -0.4,
            },
            {
                "variant": "reverse_tokens",
                "layer": 0,
                "layers": 2,
                "tokens": 8,
                "dim": 4,
                "spectral_curl_ratio": 0.4,
                "hodge_curl_ratio": 0.2,
                "graph_high_freq_ratio": 0.1,
                "trajectory_signed_circulation_alignment": 0.2,
                "turning_alignment": 0.25,
                "local_signed_vorticity_ratio": 0.35,
                "spectral_curl_high_ratio": 0.04,
                "spectral_signed_curl_alignment": 0.3,
                "hodge_signed_curl_alignment": -0.01,
                "spectral_signed_vorticity_ratio": 0.4,
            },
        ]

    def test_reverse_diagnostics_detect_signed_cancellation(self) -> None:
        diagnostics = report.build_reverse_diagnostics(self._sample_rows())

        by_metric = {item["metric"]: item for item in diagnostics}
        self.assertEqual(
            by_metric["spectral_signed_vorticity_ratio"]["maxAbsRealPlusReverse"],
            0.0,
        )

    def test_build_report_html_embeds_payload_and_controls(self) -> None:
        html = report.build_report_html(
            self._sample_rows(),
            title="Test Spiral Hodge Report",
            csv_path=Path("layer_metrics.csv"),
        )

        self.assertIn("Test Spiral Hodge Report", html)
        self.assertIn("variantButtons", html)
        self.assertIn("spectral_signed_vorticity_ratio", html)
        self.assertIn("spectral_curl_high_ratio", html)
        self.assertIn("local_signed_vorticity_ratio", html)

    def test_write_report_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "layer_metrics.csv"
            csv_path.write_text(
                "variant,layer,layers,tokens,dim,spectral_curl_ratio,spectral_signed_vorticity_ratio\n"
                "real,0,1,8,4,0.5,-0.7\n",
                encoding="utf-8",
            )
            output = Path(td) / "report.html"

            report.write_report(metrics_path=csv_path, output_path=output, title="Temp Report")

            self.assertTrue(output.exists())
            self.assertIn("Temp Report", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
