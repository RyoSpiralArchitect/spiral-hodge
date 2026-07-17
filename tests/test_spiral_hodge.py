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


class TestHLTDGraphHodge(unittest.TestCase):
    def test_radius_filtration_uses_graph_median_edge_scale(self) -> None:
        points = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        edges = np.asarray(
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            dtype=int,
        )

        _C_open, triangles_open, _, diag_open = hodge.triangle_clique_radius_filtration(
            points,
            edges,
            radius_scale=1.2,
        )
        _C_mid, triangles_mid, _, diag_mid = hodge.triangle_clique_radius_filtration(
            points,
            edges,
            radius_scale=1.4,
        )
        C_full, triangles_full, _, diag_full = hodge.triangle_clique_radius_filtration(
            points,
            edges,
            radius_scale=np.inf,
        )

        self.assertEqual(len(triangles_open), 0)
        self.assertEqual(len(triangles_mid), 2)
        self.assertEqual(len(triangles_full), 4)
        self.assertEqual(C_full.shape, (6, 4))
        self.assertAlmostEqual(diag_open["graph_edge_length_median"], 1.5)
        self.assertAlmostEqual(diag_mid["triangle_fill_actual"], 0.5)
        self.assertTrue(np.isinf(diag_full["filtration_radius_scale_requested"]))

    def test_triangle_filtration_is_nested_and_matches_full_clique_complex(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        edges = np.asarray(
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            dtype=int,
        )

        C0, triangles0, _, diag0 = hodge.triangle_clique_filtration(
            points,
            edges,
            fill_fraction=0.0,
        )
        C50, triangles50, scores50, _ = hodge.triangle_clique_filtration(
            points,
            edges,
            fill_fraction=0.5,
        )
        C1, triangles1, scores1, diag1 = hodge.triangle_clique_filtration(
            points,
            edges,
            fill_fraction=1.0,
        )
        C_full, triangles_full = hodge.triangle_boundary_matrix_from_cliques(edges)

        self.assertEqual(C0.shape[1], 0)
        self.assertEqual(triangles0.shape, (0, 3))
        self.assertEqual(diag0["triangle_fill_actual"], 0.0)
        self.assertEqual(len(triangles50), 2)
        np.testing.assert_array_equal(triangles50, triangles1[:2])
        self.assertTrue(np.all(np.diff(scores1) >= 0.0))
        np.testing.assert_array_equal(scores50, scores1[:2])
        self.assertEqual(C1.shape, C_full.shape)
        self.assertEqual(set(map(tuple, triangles1)), set(map(tuple, triangles_full)))
        self.assertEqual(diag1["triangle_fill_actual"], 1.0)

    def test_incremental_boundary_basis_preserves_prefix_rank(self) -> None:
        edges = np.asarray([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)], dtype=int)
        triangles = np.asarray([(0, 1, 2), (0, 2, 3), (0, 1, 2)], dtype=int)
        C = hodge.triangle_boundary_matrix(edges, triangles)

        basis, rank_after, accepted = hodge.incremental_orthonormal_column_basis(C)
        limited_basis, limited_rank_after, limited_accepted = (
            hodge.incremental_orthonormal_column_basis(C, maximum_rank=1)
        )

        self.assertEqual(basis.shape, (5, 2))
        np.testing.assert_array_equal(rank_after, np.asarray([1, 2, 2]))
        np.testing.assert_array_equal(accepted, np.asarray([0, 1]))
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)
        self.assertEqual(limited_basis.shape, (5, 1))
        np.testing.assert_array_equal(limited_rank_after, np.asarray([1, 1, 1]))
        np.testing.assert_array_equal(limited_accepted, np.asarray([0]))

    def test_basis_hodge_projection_has_orthogonal_energy_closure(self) -> None:
        edges = np.asarray([(0, 1), (0, 2), (1, 2)], dtype=int)
        B = hodge.vertex_edge_incidence(3, edges)
        C, _ = hodge.triangle_boundary_matrix_from_cliques(edges)
        exact_basis, _, _ = hodge.incremental_orthonormal_column_basis(B.T)
        coexact_basis, _, _ = hodge.incremental_orthonormal_column_basis(C)
        flows = np.asarray([[1.0, -0.4, 0.7], [-0.5, 0.2, 1.3]])

        exact, coexact, harmonic, energies = hodge.hodge_decompose_graph_edge_flows_from_bases(
            flows,
            exact_basis,
            coexact_basis,
        )

        np.testing.assert_allclose(exact @ coexact.T, np.zeros((2, 2)), atol=1e-12)
        np.testing.assert_allclose(exact @ harmonic.T, np.zeros((2, 2)), atol=1e-12)
        np.testing.assert_allclose(coexact @ harmonic.T, np.zeros((2, 2)), atol=1e-12)
        for energy in energies:
            self.assertAlmostEqual(
                energy["total"],
                energy["exact"] + energy["coexact"] + energy["harmonic"],
                places=12,
            )
            self.assertLess(energy["reconstruction_error"], 1e-12)

    def test_matched_betti_hltd_selects_orthogonal_intermediate_complex(self) -> None:
        rng = np.random.default_rng(23)
        points = rng.normal(size=(5, 4))
        vectors = rng.normal(size=(5, 4))

        decomp, topology = hodge.hodge_latent_traversal_dynamics_matched_betti(
            points,
            vectors,
            k_neighbors=4,
            target_betti_1_fraction=0.5,
        )

        self.assertEqual(topology["complex_mode"], "matched_betti")
        self.assertEqual(topology["hodge_solver"], "orthogonal")
        self.assertEqual(topology["cycle_rank"], 6.0)
        self.assertEqual(topology["triangle_rank"], 3.0)
        self.assertEqual(topology["betti_1"], 3.0)
        self.assertEqual(topology["betti_1_fraction"], 0.5)
        self.assertEqual(len(decomp.triangles), int(topology["triangle_count"]))
        self.assertEqual(decomp.C.shape, (10, len(decomp.triangles)))
        self.assertAlmostEqual(
            decomp.energy["total"],
            decomp.energy["exact"] + decomp.energy["coexact"] + decomp.energy["harmonic"],
            places=12,
        )
        self.assertLess(abs(decomp.energy["exact_coexact_alignment"]), 1e-12)
        self.assertLess(abs(decomp.energy["exact_harmonic_alignment"]), 1e-12)
        self.assertLess(abs(decomp.energy["coexact_harmonic_alignment"]), 1e-12)
        np.testing.assert_allclose(decomp.B.T @ decomp.phi, decomp.exact, atol=1e-10)
        np.testing.assert_allclose(decomp.C @ decomp.psi, decomp.coexact, atol=1e-10)
        np.testing.assert_allclose(decomp.B @ decomp.harmonic, 0.0, atol=1e-10)
        np.testing.assert_allclose(decomp.C.T @ decomp.harmonic, 0.0, atol=1e-10)

    def test_harmonic_ring_transfers_to_coexact_when_faces_fill_it(self) -> None:
        edges = np.asarray([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)], dtype=int)
        B = hodge.vertex_edge_incidence(4, edges)
        ring_flow = np.asarray([1.0, 0.0, -1.0, 1.0, 1.0])
        C_open = hodge.triangle_boundary_matrix(edges, np.empty((0, 3), dtype=int))
        C_filled = hodge.triangle_boundary_matrix(
            edges,
            np.asarray([(0, 1, 2), (0, 2, 3)], dtype=int),
        )

        *_, open_energy = hodge.hodge_decompose_graph_edge_flow(
            ring_flow,
            B,
            C_open,
            ridge=1e-9,
        )
        *_, filled_energy = hodge.hodge_decompose_graph_edge_flow(
            ring_flow,
            B,
            C_filled,
            ridge=1e-9,
        )
        open_topology = hodge.hodge_complex_topology_diagnostics(4, edges, C_open)
        filled_topology = hodge.hodge_complex_topology_diagnostics(4, edges, C_filled)

        self.assertGreater(open_energy["harmonic_ratio"], 0.999999)
        self.assertGreater(filled_energy["coexact_ratio"], 0.999999)
        self.assertEqual(open_topology["betti_1"], 2.0)
        self.assertEqual(filled_topology["betti_1"], 0.0)

    def test_exact_flow_is_recovered_on_path_graph(self) -> None:
        edges = np.asarray([(0, 1), (1, 2)], dtype=int)
        B = hodge.vertex_edge_incidence(3, edges)
        C, triangles = hodge.triangle_boundary_matrix_from_cliques(edges)
        phi = np.asarray([0.0, 1.0, 3.0])
        flow = np.asarray(B.T @ phi).reshape(-1)

        exact, coexact, harmonic, *_rest, energy = hodge.hodge_decompose_graph_edge_flow(
            flow,
            B,
            C,
            ridge=1e-9,
        )

        self.assertEqual(triangles.shape, (0, 3))
        np.testing.assert_allclose(exact, flow, atol=1e-7)
        np.testing.assert_allclose(coexact, np.zeros_like(flow), atol=1e-7)
        np.testing.assert_allclose(harmonic, np.zeros_like(flow), atol=1e-6)
        self.assertGreater(energy["exact_ratio"], 0.999999)

    def test_triangle_boundary_flow_is_coexact(self) -> None:
        edges = np.asarray([(0, 1), (0, 2), (1, 2)], dtype=int)
        B = hodge.vertex_edge_incidence(3, edges)
        C, triangles = hodge.triangle_boundary_matrix_from_cliques(edges)
        flow = np.asarray(C[:, 0].toarray()).reshape(-1)

        exact, coexact, harmonic, *_rest, energy = hodge.hodge_decompose_graph_edge_flow(
            flow,
            B,
            C,
            ridge=1e-9,
        )

        np.testing.assert_array_equal(triangles, np.asarray([(0, 1, 2)], dtype=int))
        np.testing.assert_allclose(exact, np.zeros_like(flow), atol=1e-7)
        np.testing.assert_allclose(coexact, flow, atol=1e-7)
        np.testing.assert_allclose(harmonic, np.zeros_like(flow), atol=1e-6)
        self.assertGreater(energy["coexact_ratio"], 0.999999)

    def test_batch_hodge_matches_independent_single_flow_decompositions(self) -> None:
        edges = np.asarray([(0, 1), (0, 2), (1, 2)], dtype=int)
        B = hodge.vertex_edge_incidence(3, edges)
        C, _triangles = hodge.triangle_boundary_matrix_from_cliques(edges)
        flows = np.asarray(
            [
                [1.0, 0.5, -0.5],
                [-0.2, 0.7, 1.1],
            ]
        )

        exact, coexact, harmonic, phi, psi, energies = hodge.hodge_decompose_graph_edge_flows(
            flows,
            B,
            C,
            ridge=1e-7,
        )

        self.assertEqual(exact.shape, flows.shape)
        self.assertEqual(coexact.shape, flows.shape)
        self.assertEqual(harmonic.shape, flows.shape)
        self.assertEqual(phi.shape, (2, 3))
        self.assertEqual(psi.shape, (2, 1))
        for index, flow in enumerate(flows):
            single = hodge.hodge_decompose_graph_edge_flow(flow, B, C, ridge=1e-7)
            np.testing.assert_allclose(exact[index], single[0])
            np.testing.assert_allclose(coexact[index], single[1])
            np.testing.assert_allclose(harmonic[index], single[2])
            self.assertAlmostEqual(energies[index]["coexact_ratio"], single[-1]["coexact_ratio"])

    def test_node_vectors_from_edge_component_reconstructs_line_flow(self) -> None:
        points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        edges = np.asarray([(0, 1), (1, 2)], dtype=int)
        component = np.asarray([1.0, 1.0])

        vectors = hodge.node_vectors_from_edge_component(points, edges, component, ridge=1e-9)

        np.testing.assert_allclose(vectors[:, 0], np.ones(3), atol=1e-7)
        np.testing.assert_allclose(vectors[:, 1], np.zeros(3), atol=1e-7)

    def test_hltd_from_coordinates_exports_semantic_flow_energy(self) -> None:
        hidden = hodge.synthetic_hidden_states(layers=2, tokens=16, dim=12, seed=7).hidden
        coord = hodge.make_semantic_coordinates(hidden, n_components=4, verbose=False)

        decomp = hodge.hltd_from_coordinates(coord, layer=1, k_neighbors=4, ridge=1e-6)

        self.assertGreater(len(decomp.edges), 0)
        self.assertIn("semantic_flow_ratio", decomp.energy)
        self.assertTrue(np.isfinite(decomp.energy["semantic_flow_ratio"]))
        self.assertLess(decomp.energy["reconstruction_error"], 1e-9)

    def test_pca_chart_vectors_to_hidden_maps_differentials(self) -> None:
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(5)
        X = rng.normal(size=(20, 6))
        reducer = PCA(n_components=3, random_state=0).fit(X)
        chart_vectors = rng.normal(size=(4, 3))

        hidden_vectors = hodge.pca_chart_vectors_to_hidden(chart_vectors, reducer)

        self.assertEqual(hidden_vectors.shape, (4, 6))
        np.testing.assert_allclose(hidden_vectors, chart_vectors @ reducer.components_)

    def test_centered_hltd_energy_is_invariant_under_reversal(self) -> None:
        theta = np.linspace(0.0, 1.6 * np.pi, 18)
        coords = np.stack([np.cos(theta), np.sin(theta), 0.2 * theta], axis=1)[None, :, :]
        rev_coords = coords[:, ::-1, :]
        coord = hodge.CoordinateBundle(
            coords=coords,
            reducer=None,
            explained_variance_ratio=None,
            flattened_input_shape=(1, coords.shape[1], coords.shape[2]),
        )
        rev_coord = hodge.CoordinateBundle(
            coords=rev_coords,
            reducer=None,
            explained_variance_ratio=None,
            flattened_input_shape=(1, coords.shape[1], coords.shape[2]),
        )

        decomp = hodge.hltd_from_coordinates(
            coord,
            layer=0,
            k_neighbors=5,
            ridge=1e-9,
            vector_mode="centered",
        )
        rev_decomp = hodge.hltd_from_coordinates(
            rev_coord,
            layer=0,
            k_neighbors=5,
            ridge=1e-9,
            vector_mode="centered",
        )

        for key in ["total", "exact_ratio", "coexact_ratio", "harmonic_ratio", "semantic_flow_ratio"]:
            self.assertAlmostEqual(decomp.energy[key], rev_decomp.energy[key], places=8)

    def test_same_graph_reverse_hltd_diagnostic_is_exactly_stable(self) -> None:
        theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        vectors = np.stack([-np.sin(theta), np.cos(theta)], axis=1)

        decomp = hodge.hodge_latent_traversal_dynamics(points, vectors, k_neighbors=6, ridge=1e-9)
        metrics = hodge.hltd_same_graph_reverse_metrics(decomp, ridge=1e-9)

        self.assertLess(metrics["coexact_ratio_gap"], 1e-10)
        self.assertLess(metrics["semantic_flow_ratio_gap"], 1e-10)
        self.assertLess(metrics["total_ratio_gap"], 1e-10)
        if decomp.energy["coexact"] > 1e-9:
            self.assertAlmostEqual(metrics["coexact_alignment"], -1.0, places=7)

    def test_layer_metric_row_exports_same_graph_reverse_diagnostic(self) -> None:
        hidden = hodge.synthetic_hidden_states(layers=2, tokens=16, dim=12, seed=11).hidden
        coord = hodge.make_semantic_coordinates(hidden, n_components=4, verbose=False)

        result = hodge.analyze_layer_from_coordinates(
            coord,
            layer=1,
            do_fourier=False,
            do_graph=False,
            do_hodge=False,
            do_hltd=True,
            hltd_k_neighbors=4,
            hltd_ridge=1e-6,
            hltd_same_graph_reverse=True,
        )
        row = hodge.layer_metric_row(
            variant="real",
            layer=1,
            hidden_shape=hidden.shape,
            coord=coord,
            result=result,
        )

        self.assertIn("hltd_same_graph_reverse_coexact_ratio_gap", row)
        self.assertLess(row["hltd_same_graph_reverse_coexact_ratio_gap"], 1e-10)


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
