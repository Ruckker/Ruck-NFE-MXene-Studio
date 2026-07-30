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
# - 主要接口 / Main APIs: ManifoldGenerationTest
#
# Author: Ruck
# Generated: 2026-07-30 01:12:13 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import random
import unittest

import numpy as np
from pymatgen.core import Lattice, Structure

from nfe_model.strict_generation import composition_key, layer_count
from nfe_model.manifold_generation import (
    mutate_template_to_unseen,
    project_structure_to_template_manifold,
)
from nfe_model.surface_geometry import validate_surface_topology


# 中文：顶层类 `ManifoldGenerationTest`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `ManifoldGenerationTest`; review type hints and callers before extending it.
class ManifoldGenerationTest(unittest.TestCase):
    @staticmethod
    def template() -> dict:
        lattice = Lattice.hexagonal(3.10, 30.0)
        return {
            "id": "trusted",
            "z": [9, 41, 6, 23, 9],
            "formula": "NbVCF2",
            "frac_pos": [
                [1 / 3, 2 / 3, 0.38],
                [0.00, 0.00, 0.44],
                [1 / 3, 2 / 3, 0.50],
                [0.00, 0.00, 0.56],
                [1 / 3, 2 / 3, 0.62],
            ],
            "lattice": np.asarray(lattice.matrix).tolist(),
            "surface_side": [-1, 0, 0, 0, 1],
            "layer_position": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "group_type": [1, 0, 0, 0, 1],
            "adsorption_coordination": [3, 0, 0, 0, 3],
            "termination_motif": "F|F",
            "label": 1,
            "score": 0.58,
        }

    def test_mutation_is_absent_from_forbidden_compositions(self) -> None:
        template = self.template()
        forbidden = {composition_key(template["z"])}
        random.seed(41)
        mutated = mutate_template_to_unseen(template, forbidden)
        self.assertNotIn(composition_key(mutated["z"]), forbidden)
        self.assertEqual(mutated["z"][0], template["z"][0])
        self.assertEqual(mutated["z"][2], template["z"][2])
        self.assertEqual(mutated["z"][4], template["z"][4])
        self.assertNotEqual(mutated["formula"], template["formula"])

    def test_projection_restores_layers_and_surface_topology(self) -> None:
        template = self.template()
        distorted = Structure(
            Lattice.hexagonal(3.45, 23.0),
            ["F", "Nb", "C", "V", "F"],
            [
                [0.48, 0.52, 0.475],
                [0.08, 0.07, 0.487],
                [0.43, 0.58, 0.500],
                [0.09, 0.10, 0.513],
                [0.52, 0.49, 0.525],
            ],
        )
        projected = project_structure_to_template_manifold(
            distorted, template
        )
        self.assertAlmostEqual(projected.lattice.c, 30.0, places=6)
        self.assertEqual(layer_count(projected), 5)
        profile = {
            "layer_count_distribution": {"5": 1, "6": 1, "7": 1},
            "termination_motif_top50": {"F|F": 1},
            "oh_bond_length_A": {"q01": 0.96, "q99": 1.00},
            "anchor_distance_A_by_pair": {},
        }
        valid, metrics = validate_surface_topology(projected, profile)
        self.assertTrue(valid, metrics)
        self.assertEqual(metrics["Surface_Layer_Count"], 5)
        self.assertEqual(metrics["Surface_Group_Count"], 2)


if __name__ == "__main__":
    unittest.main()
