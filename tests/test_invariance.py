# ==============================================================================
# 中文概述：验证规范化后的周期图与预测对真空厚度、晶格设定、超胞、平移、原子顺序不变。
# English overview: Verify canonicalized graphs and predictions are invariant to vacuum
#                   thickness, lattice setting, supercell size, translation and atom order.
#
# 中文输入：一个小型 MXene 型 slab 及其等价表示。
# English inputs: A small MXene-like slab and its equivalent representations.
# 中文输出：确定性的 unittest 断言。
# English outputs: Deterministic unittest assertions.
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import unittest

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from nfe_model.canonical import canonicalize_slab, reduce_in_plane_basis
from nfe_model.data import REGRESSION_TARGETS, build_periodic_graph, collate_graphs, structure_to_graph
from nfe_model.model import AngularMoments, PeriodicNFEModel, SurfaceReadout


def base_structure() -> Structure:
    return Structure(
        Lattice.hexagonal(3.1, 30.0),
        ["F", "Ti", "C", "V", "O", "H"],
        [
            [1 / 3, 2 / 3, 0.42],
            [0.0, 0.0, 0.46],
            [1 / 3, 2 / 3, 0.50],
            [0.0, 0.0, 0.54],
            [1 / 3, 2 / 3, 0.58],
            [1 / 3, 2 / 3, 0.61],
        ],
    )


def variants(structure: Structure) -> dict[str, Structure]:
    matrix = np.asarray(structure.lattice.matrix)
    cart = structure.cart_coords
    result = {"as-is": structure}
    for c_new in (18.0, 25.0, 42.0):
        new_matrix = matrix.copy()
        new_matrix[2] = [0.0, 0.0, c_new]
        shifted = cart.copy()
        shifted[:, 2] += (c_new - structure.lattice.c) / 2.0
        result[f"c={c_new}"] = Structure(
            Lattice(new_matrix), [s.specie for s in structure], shifted, coords_are_cartesian=True
        )
    result["gamma=60"] = Structure(
        Lattice(np.array([matrix[0], matrix[0] + matrix[1], matrix[2]])),
        [s.specie for s in structure],
        cart,
        coords_are_cartesian=True,
        to_unit_cell=True,
    )
    supercell = structure.copy()
    supercell.make_supercell([2, 2, 1])
    result["2x2x1"] = supercell
    translated = structure.copy()
    translated.translate_sites(list(range(len(translated))), [0.37, 0.11, 0.23], frac_coords=True)
    result["translated"] = translated
    permuted = Structure(
        structure.lattice,
        [structure[i].specie for i in (5, 3, 1, 4, 0, 2)],
        [structure[i].frac_coords for i in (5, 3, 1, 4, 0, 2)],
    )
    result["permuted"] = permuted
    return result


def edge_signature(graph: dict) -> list[tuple]:
    """Basis- and permutation-invariant edge multiset: (Z_source, Z_destination, distance)."""
    _, distances, _ = PeriodicNFEModel.edge_geometry(
        graph["frac_pos"],
        graph["lattice"].unsqueeze(0),
        torch.zeros(len(graph["z"]), dtype=torch.long),
        graph["edge_index"],
        graph["edge_shift"],
    )
    z = graph["z"].numpy()
    edges = graph["edge_index"].numpy().T
    return sorted(
        (int(z[s]), int(z[d]), int(round(float(dist) * 1_000)))
        for (s, d), dist in zip(edges, distances.numpy())
    )


class CanonicalizationTest(unittest.TestCase):
    def test_reduction_gives_hexagonal_obtuse_setting(self) -> None:
        a = np.array([3.1, 0.0, 0.0])
        b = np.array([1.55, 3.1 * np.sqrt(3) / 2, 0.0])  # gamma = 60 setting
        ra, rb = reduce_in_plane_basis(a, b)
        cos_gamma = np.dot(ra, rb) / (np.linalg.norm(ra) * np.linalg.norm(rb))
        self.assertAlmostEqual(cos_gamma, -0.5, places=9)
        self.assertAlmostEqual(np.linalg.norm(ra), 3.1, places=9)
        self.assertAlmostEqual(np.linalg.norm(rb), 3.1, places=9)

    def test_all_variants_canonicalize_to_the_same_structure(self) -> None:
        from pymatgen.analysis.structure_matcher import StructureMatcher

        # safe_structure_match falls back to the pure-Python matcher where the
        # pymatgen Windows wheel's Cython LinearAssignment rejects int32 buffers.
        from nfe_model.strict_generation import safe_structure_match

        reference = canonicalize_slab(base_structure())
        self.assertAlmostEqual(reference.lattice.c, 30.0, places=9)
        self.assertAlmostEqual(reference.lattice.gamma, 120.0, places=6)
        self.assertEqual(len(reference), 6)
        matcher = StructureMatcher(
            ltol=0.01, stol=0.01, angle_tol=0.5, primitive_cell=False, scale=False, attempt_supercell=False
        )
        for name, variant in variants(base_structure()).items():
            canonical = canonicalize_slab(variant)
            with self.subTest(variant=name):
                self.assertEqual(len(canonical), 6)
                self.assertTrue(np.allclose(canonical.lattice.abc, reference.lattice.abc, atol=1e-6))
                self.assertTrue(np.allclose(canonical.lattice.angles, reference.lattice.angles, atol=1e-5))
                self.assertEqual(sorted(s.specie.symbol for s in canonical), sorted(s.specie.symbol for s in reference))
                # The reduced hexagonal basis is unique only up to the 60-degree
                # rotations of the lattice, so compare as crystals, not as arrays.
                self.assertTrue(safe_structure_match(matcher, canonical, reference), msg=name)
                z_reference = np.sort(reference.frac_coords[:, 2])
                z_canonical = np.sort(canonical.frac_coords[:, 2])
                self.assertTrue(np.allclose(z_canonical, z_reference, atol=1e-5), msg=name)

    def test_graphs_and_predictions_are_identical_across_variants(self) -> None:
        torch.manual_seed(3)
        model = PeriodicNFEModel(
            hidden_dim=32,
            vector_dim=8,
            num_layers=2,
            num_rbf=12,
            cutoff=6.0,
            global_features=11,
            num_regression_targets=len(REGRESSION_TARGETS),
        ).eval()
        graphs = {
            name: structure_to_graph(variant, 6.0, 36, identifier=name, canonicalize=True, complete_shells=True)
            for name, variant in variants(base_structure()).items()
        }
        reference = graphs["as-is"]
        reference_signature = edge_signature(reference)
        with torch.no_grad():
            reference_output = model(collate_graphs([dict(reference, targets=torch.zeros(len(REGRESSION_TARGETS)), target_mask=torch.ones(len(REGRESSION_TARGETS), dtype=torch.bool), label=-1)]))
        for name, graph in graphs.items():
            with self.subTest(variant=name):
                self.assertEqual(edge_signature(graph), reference_signature)
                self.assertTrue(torch.allclose(graph["global_features"], reference["global_features"], atol=1e-4))
                with torch.no_grad():
                    output = model(collate_graphs([dict(graph, targets=torch.zeros(len(REGRESSION_TARGETS)), target_mask=torch.ones(len(REGRESSION_TARGETS), dtype=torch.bool), label=-1)]))
                self.assertTrue(torch.allclose(output["class_logits"], reference_output["class_logits"], atol=1e-4))
                self.assertTrue(torch.allclose(output["regression_mean"], reference_output["regression_mean"], atol=1e-4))

    def test_complete_shells_never_splits_a_coordination_shell(self) -> None:
        structure = canonicalize_slab(base_structure())
        graph = build_periodic_graph(structure, 6.0, 12, complete_shells=True)
        _, distances, _ = PeriodicNFEModel.edge_geometry(
            graph["frac_pos"], graph["lattice"].unsqueeze(0), torch.zeros(len(structure), dtype=torch.long), graph["edge_index"], graph["edge_shift"]
        )
        destination = graph["edge_index"][1].numpy()
        quantized = np.rint(distances.numpy().astype(np.float64) * 1_000_000.0).astype(np.int64)
        for atom in range(len(structure)):
            kept = quantized[destination == atom]
            self.assertGreaterEqual(len(kept), 12)
            # Every neighbour at the largest kept distance must be present: rebuild with a
            # generous cap and compare the count of that shell.
            full = build_periodic_graph(structure, 6.0, 10_000, complete_shells=False)
            _, full_distances, _ = PeriodicNFEModel.edge_geometry(
                full["frac_pos"], full["lattice"].unsqueeze(0), torch.zeros(len(structure), dtype=torch.long), full["edge_index"], full["edge_shift"]
            )
            full_quantized = np.rint(full_distances.numpy().astype(np.float64) * 1_000_000.0).astype(np.int64)
            full_kept = full_quantized[full["edge_index"][1].numpy() == atom]
            last = kept.max()
            self.assertEqual(int(np.sum(kept == last)), int(np.sum(full_kept == last)))

    def test_model_without_global_features(self) -> None:
        model = PeriodicNFEModel(hidden_dim=32, vector_dim=8, num_layers=1, num_rbf=8, cutoff=6.0, global_features=0, num_regression_targets=len(REGRESSION_TARGETS)).eval()
        graph = structure_to_graph(base_structure(), 6.0, 36, canonicalize=True)
        batch = collate_graphs([dict(graph, targets=torch.zeros(len(REGRESSION_TARGETS)), target_mask=torch.ones(len(REGRESSION_TARGETS), dtype=torch.bool), label=-1)])
        with torch.no_grad():
            output = model(batch)
        self.assertEqual(tuple(output["class_logits"].shape), (1, 3))
        self.assertIsNone(model.global_encoder)


class SurfaceAndAngularTest(unittest.TestCase):
    """The two optional blocks must keep the symmetries the task has."""

    @staticmethod
    def _batch(structure: Structure) -> dict:
        graph = structure_to_graph(structure, 6.0, 36, canonicalize=True, complete_shells=True)
        return collate_graphs([
            dict(
                graph,
                targets=torch.zeros(len(REGRESSION_TARGETS)),
                target_mask=torch.ones(len(REGRESSION_TARGETS), dtype=torch.bool),
                label=-1,
            )
        ])

    def test_defaults_leave_the_released_network_unchanged(self) -> None:
        model = PeriodicNFEModel(hidden_dim=16, vector_dim=4, num_layers=1, num_rbf=8, global_features=0)
        self.assertIsNone(model.surface_readout)
        self.assertIsNone(model.angular)
        self.assertFalse(model.config["surface_readout"])
        self.assertEqual(model.config["angular_channels"], 0)

    def test_angular_moments_are_rotation_invariant(self) -> None:
        torch.manual_seed(0)
        module = AngularMoments(num_rbf=8, channels=4, max_order=3).eval()
        unit = torch.nn.functional.normalize(torch.randn(40, 3), dim=-1)
        radial = torch.rand(40, 8)
        destination = torch.randint(0, 5, (40,))
        angle = torch.tensor(0.7)
        rotation = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), torch.tensor(0.0)],
            [torch.sin(angle), torch.cos(angle), torch.tensor(0.0)],
            [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)],
        ])
        with torch.no_grad():
            plain = module(unit, radial, destination, 5)
            rotated = module(unit @ rotation.T, radial, destination, 5)
        torch.testing.assert_close(plain, rotated, rtol=1e-5, atol=1e-6)

    def test_surface_readout_is_invariant_to_flipping_the_slab(self) -> None:
        torch.manual_seed(0)
        module = SurfaceReadout(hidden_dim=6).eval()
        scalar = torch.randn(8, 6)
        height = torch.tensor([-3.0, -2.4, -0.8, -0.1, 0.1, 0.8, 2.4, 3.0])
        graph_index = torch.zeros(8, dtype=torch.long)
        order = torch.arange(7, -1, -1)
        with torch.no_grad():
            plain = module(scalar, height, graph_index, 1)
            flipped = module(scalar[order], -height[order], graph_index, 1)
        torch.testing.assert_close(plain, flipped, rtol=1e-5, atol=1e-6)

    def test_variants_run_and_stay_representation_invariant(self) -> None:
        for options in (
            {"surface_readout": True},
            {"angular_channels": 8},
            {"surface_readout": True, "angular_channels": 8},
        ):
            with self.subTest(**options):
                torch.manual_seed(0)
                model = PeriodicNFEModel(
                    hidden_dim=24, vector_dim=8, num_layers=2, num_rbf=8, cutoff=6.0,
                    global_features=0, num_regression_targets=len(REGRESSION_TARGETS),
                    dropout=0.0, **options,
                ).eval()
                with torch.no_grad():
                    reference = model(self._batch(base_structure()))["class_logits"]
                    for variant in variants(base_structure()).values():
                        other = model(self._batch(variant))["class_logits"]
                        torch.testing.assert_close(reference, other, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
