# ==============================================================================
# 中文概述：提供 NFE MXene 项目中的单一、可复用源码职责。
# English overview: Provide one reusable source-code responsibility in the NFE MXene project.
#
# 中文输入：请结合类型标注、命令行帮助和调用方查看输入。
# English inputs: Read type hints, CLI help, and callers for the expected inputs.
# 中文输出：返回值或生成文件由公开接口和命令行参数定义。
# English outputs: Return values or generated files are defined by public APIs and CLI arguments.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: SmokeTest
#
# Author: Ruck
# Generated: 2026-07-29 22:47:20 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import unittest

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from nfe_model.data import REGRESSION_TARGETS, build_periodic_graph, collate_graphs
from nfe_model.strict_generation import center_structure, prediction_matches_target
from nfe_model.surface_generator import SurfaceAwareTemplateFlow
from nfe_model.surface_generator_data import (
    SurfaceTemplateDataset,
    collate_surface_templates,
    lattice_normalizers,
    prepare_surface_generator_records,
)
from nfe_model.generator_data import (
    center_slab_fractional,
    center_slab_fractional_tensor,
    generation_batch,
    slab_center_fractional_z,
)
from nfe_model.metrics import classification_metrics
from nfe_model.model import PeriodicNFEModel
from nfe_model.surface_geometry import (
    BOTTOM,
    TOP,
    analyze_surface_geometry,
    validate_surface_topology,
)
from nfe_model.train_surface_generator import compute_flow_loss, flow_corruption


# 中文：顶层类 `SmokeTest`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `SmokeTest`; review type hints and callers before extending it.
class SmokeTest(unittest.TestCase):
    @staticmethod
    def structure() -> Structure:
        return Structure(
            Lattice.hexagonal(3.1, 18.0),
            ["Ti", "C", "O", "O"],
            [
                [0.0, 0.0, 0.50],
                [1 / 3, 2 / 3, 0.50],
                [0.0, 0.0, 0.43],
                [0.0, 0.0, 0.57],
            ],
        )

    @staticmethod
    def supervised_graph(structure: Structure) -> dict:
        graph = build_periodic_graph(structure, radius=5.0, max_neighbors=24)
        graph.update(
            {
                "targets": torch.zeros(len(REGRESSION_TARGETS)),
                "target_mask": torch.ones(len(REGRESSION_TARGETS), dtype=torch.bool),
                "label": 2,
            }
        )
        return graph

    def test_final_package_exports_surface_flow_only(self) -> None:
        # 中文：公开 API 只暴露最终 surface generator 表面流，不再暴露旧基础生成器。
        # English: The public API exposes the final surface-aware flow, not the legacy generator.
        import nfe_model

        self.assertIs(nfe_model.SurfaceAwareTemplateFlow, SurfaceAwareTemplateFlow)
        self.assertFalse(hasattr(nfe_model, "ConditionalCrystalFlow"))

    def test_structure_forward_backward(self) -> None:
        graph = self.supervised_graph(self.structure())
        batch = collate_graphs([graph, graph])
        model = PeriodicNFEModel(
            hidden_dim=48,
            vector_dim=16,
            num_layers=2,
            num_rbf=16,
            cutoff=5.0,
            global_features=11,
            num_regression_targets=len(REGRESSION_TARGETS),
        )
        output = model(batch)
        self.assertEqual(tuple(output["class_logits"].shape), (2, 3))
        self.assertEqual(
            tuple(output["regression_mean"].shape),
            (2, len(REGRESSION_TARGETS)),
        )
        loss = (
            output["class_logits"].square().mean()
            + output["regression_mean"].square().mean()
            + output["denoise_vector"].square().mean()
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_rotation_invariant_prediction(self) -> None:
        structure = self.structure()
        angle = np.deg2rad(37.0)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated = Structure(
            Lattice(np.asarray(structure.lattice.matrix) @ rotation.T),
            [site.specie for site in structure],
            structure.frac_coords,
        )
        original_batch = collate_graphs([self.supervised_graph(structure)])
        rotated_batch = collate_graphs([self.supervised_graph(rotated)])
        model = PeriodicNFEModel(
            hidden_dim=48,
            vector_dim=16,
            num_layers=2,
            num_rbf=16,
            cutoff=5.0,
            global_features=11,
            num_regression_targets=len(REGRESSION_TARGETS),
        ).eval()
        with torch.no_grad():
            original = model(original_batch)
            transformed = model(rotated_batch)
        self.assertTrue(
            torch.allclose(
                original["class_logits"],
                transformed["class_logits"],
                atol=2e-5,
                rtol=2e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                original["regression_mean"],
                transformed["regression_mean"],
                atol=2e-5,
                rtol=2e-5,
            )
        )

    def test_periodic_slab_centering(self) -> None:
        fractional = np.asarray(
            [
                [0.95, 0.05, 0.92],
                [0.05, 0.95, 0.02],
                [0.00, 0.00, 0.12],
            ]
        )
        centered = center_slab_fractional(fractional)
        self.assertAlmostEqual(
            slab_center_fractional_z(centered), 0.5, places=6
        )
        tensor = center_slab_fractional_tensor(
            torch.tensor(
                np.concatenate([fractional, fractional], axis=0),
                dtype=torch.float32,
            ),
            torch.tensor([0, 0, 0, 1, 1, 1]),
        ).numpy()
        self.assertAlmostEqual(
            slab_center_fractional_z(tensor[:3]), 0.5, places=5
        )
        self.assertAlmostEqual(
            slab_center_fractional_z(tensor[3:]), 0.5, places=5
        )
        shifted = self.structure().copy()
        shifted.translate_sites(
            range(len(shifted)), [0.17, 0.23, 0.31], frac_coords=True
        )
        recentered = center_structure(shifted)
        self.assertAlmostEqual(
            slab_center_fractional_z(recentered.frac_coords), 0.5, places=6
        )

    def test_all_class_metrics_and_strict_target_match(self) -> None:
        logits = np.asarray(
            [
                [8.0, 0.0, 0.0],
                [0.0, 8.0, 0.0],
                [0.0, 0.0, 8.0],
                [7.0, 1.0, 0.0],
                [0.0, 7.0, 1.0],
                [1.0, 0.0, 7.0],
            ]
        )
        labels = np.asarray([0, 1, 2, 0, 1, 2])
        metrics = classification_metrics(logits, labels)
        for label in ("low", "medium", "high"):
            self.assertEqual(metrics[f"{label}_f1"], 1.0)
            self.assertEqual(metrics[f"{label}_roc_auc"], 1.0)
            self.assertEqual(metrics[f"{label}_support"], 2.0)
        prediction = {
            "Predicted_NFE_Label": "medium",
            "Probability_Low": 0.05,
            "Probability_Medium": 0.90,
            "Probability_High": 0.05,
        }
        self.assertEqual(
            prediction_matches_target(prediction, "medium", 0.50),
            (True, 0.90),
        )
        self.assertEqual(
            prediction_matches_target(prediction, "high", 0.50),
            (False, 0.05),
        )

    def test_surface_geometry_recognizes_two_oh_terminations(self) -> None:
        z = np.asarray([1, 8, 41, 6, 23, 8, 1], dtype=np.int64)
        frac = np.asarray(
            [
                [0.00, 0.00, 0.384],
                [0.00, 0.00, 0.416],
                [0.00, 0.00, 0.460],
                [0.33, 0.33, 0.500],
                [0.00, 0.00, 0.540],
                [0.00, 0.00, 0.584],
                [0.00, 0.00, 0.616],
            ],
            dtype=np.float32,
        )
        lattice = np.diag([3.10, 3.10, 30.0]).astype(np.float32)
        result = analyze_surface_geometry(z, frac, lattice)
        self.assertEqual(len(result.oh_bonds), 2)
        self.assertEqual(int(np.sum(result.surface_side == BOTTOM)), 2)
        self.assertEqual(int(np.sum(result.surface_side == TOP)), 2)
        self.assertFalse(
            any(
                item.startswith("orphan_surface_hydrogen")
                for item in result.warnings
            )
        )

    def test_surface_geometry_flags_orphan_surface_hydrogen(self) -> None:
        z = np.asarray([1, 8, 8, 41, 6, 73, 1], dtype=np.int64)
        frac = np.asarray(
            [
                [0.00, 0.00, 0.376],
                [0.00, 0.00, 0.409],
                [0.33, 0.33, 0.458],
                [0.00, 0.00, 0.485],
                [0.33, 0.33, 0.528],
                [0.00, 0.00, 0.568],
                [0.00, 0.00, 0.624],
            ],
            dtype=np.float32,
        )
        lattice = np.diag([3.10, 3.10, 30.0]).astype(np.float32)
        result = analyze_surface_geometry(z, frac, lattice)
        self.assertIn("orphan_surface_hydrogen_top", result.warnings)

    def test_surface_topology_profile_accepts_double_oh(self) -> None:
        structure = Structure(
            Lattice.hexagonal(3.10, 30.0),
            ["H", "O", "Nb", "C", "V", "O", "H"],
            [
                [1 / 3, 2 / 3, 0.384],
                [1 / 3, 2 / 3, 0.416],
                [0.00, 0.00, 0.460],
                [1 / 3, 2 / 3, 0.500],
                [0.00, 0.00, 0.540],
                [1 / 3, 2 / 3, 0.584],
                [1 / 3, 2 / 3, 0.616],
            ],
        )
        profile = {
            "layer_count_distribution": {"5": 1, "6": 1, "7": 1},
            "termination_motif_top50": {"HO|HO": 1},
            "oh_bond_length_A": {"q01": 0.96, "q99": 1.00},
            "anchor_distance_A_by_pair": {},
        }
        valid, metrics = validate_surface_topology(structure, profile)
        self.assertTrue(valid, metrics)
        self.assertEqual(metrics["Termination_Motif"], "HO|HO")

    def test_surface_template_flow_backward(self) -> None:
        batch = generation_batch([[1, 8, 41, 6, 23, 8, 1]], torch.device("cpu"))
        batch.update(
            {
                "template_frac": torch.tensor(
                    [
                        [0.00, 0.00, 0.384],
                        [0.00, 0.00, 0.416],
                        [0.00, 0.00, 0.460],
                        [0.33, 0.33, 0.500],
                        [0.00, 0.00, 0.540],
                        [0.00, 0.00, 0.584],
                        [0.00, 0.00, 0.616],
                    ],
                    dtype=torch.float32,
                ),
                "surface_side": torch.tensor([-1, -1, 0, 0, 0, 1, 1]),
                "layer_position": torch.linspace(-1.0, 1.0, 7),
                "group_type": torch.tensor([3, 2, 0, 0, 0, 2, 3]),
                "adsorption_coordination": torch.tensor([3, 3, 0, 0, 0, 3, 3]),
            }
        )
        model = SurfaceAwareTemplateFlow(
            hidden_dim=48,
            vector_dim=16,
            num_layers=2,
            num_rbf=16,
            cutoff=12.0,
            max_neighbors=8,
            condition_dim=32,
        )
        frac = batch["template_frac"] + 0.01 * torch.randn(7, 3)
        lattice_state = torch.zeros(1, 6)
        lattice = torch.diag(torch.tensor([3.1, 3.1, 30.0])).unsqueeze(0)
        output = model(
            batch,
            frac,
            lattice_state,
            lattice,
            torch.tensor([0.5]),
            torch.tensor([2]),
            torch.tensor([0.85]),
        )
        loss = (
            output["coordinate_velocity_cart"].square().mean()
            + output["lattice_velocity"].square().mean()
        )
        loss.backward()
        self.assertEqual(tuple(output["coordinate_velocity_cart"].shape), (7, 3))
        self.assertTrue(torch.isfinite(loss))

    def test_surface_template_training_loss_backward(self) -> None:
        structures = []
        for metals in (("Nb", "V"), ("Ta", "W")):
            structures.append(
                Structure(
                    Lattice.hexagonal(3.10, 30.0),
                    ["H", "O", metals[0], "C", metals[1], "O", "H"],
                    [
                        [0.00, 0.00, 0.384],
                        [0.00, 0.00, 0.416],
                        [0.00, 0.00, 0.460],
                        [1 / 3, 2 / 3, 0.500],
                        [0.00, 0.00, 0.540],
                        [0.00, 0.00, 0.584],
                        [0.00, 0.00, 0.616],
                    ],
                )
            )
        records = []
        for index, structure in enumerate(structures):
            graph = build_periodic_graph(
                structure, radius=5.0, max_neighbors=24, identifier=str(index)
            )
            graph.update(
                {
                    "targets": torch.zeros(len(REGRESSION_TARGETS)),
                    "target_mask": torch.ones(
                        len(REGRESSION_TARGETS), dtype=torch.bool
                    ),
                    "label": 2,
                    "sample_weight": 1.0,
                    "split": "train",
                }
            )
            records.append(graph)
        prepared = prepare_surface_generator_records(records)
        normalizers = lattice_normalizers(prepared, [0, 1])
        dataset = SurfaceTemplateDataset(prepared, [0, 1], normalizers)
        batch = collate_surface_templates([dataset[0], dataset[1]])
        config = {
            "coordinate_weight": 1.0,
            "lattice_weight": 0.35,
            "repulsion_weight": 0.30,
            "endpoint_weight": 1.0,
            "pair_distance_weight": 0.80,
            "layer_order_weight": 0.50,
            "surface_anchor_weight": 0.80,
            "oh_geometry_weight": 1.50,
            "minimum_distance_factor": 0.70,
            "core_template_noise_A": 0.18,
            "surface_template_noise_A": 0.35,
            "hydrogen_template_noise_A": 0.16,
            "lattice_template_noise": 0.10,
        }
        flow = flow_corruption(batch, normalizers, config)
        model = SurfaceAwareTemplateFlow(
            hidden_dim=48,
            vector_dim=16,
            num_layers=2,
            num_rbf=16,
            cutoff=12.0,
            max_neighbors=8,
            condition_dim=32,
        )
        outputs = model(
            batch,
            flow["frac_pos"],
            flow["lattice_state"],
            flow["lattice"],
            flow["time"],
            batch["labels"],
            batch["scores"],
        )
        loss, components = compute_flow_loss(
            outputs, batch, flow, normalizers, config
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(np.isfinite(components["endpoint_rmse_A"]))
        self.assertGreaterEqual(components["surface_mae_A"], 0.0)


if __name__ == "__main__":
    unittest.main()
