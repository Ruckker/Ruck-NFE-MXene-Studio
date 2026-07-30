# ==============================================================================
# 中文概述：验证三维场景的原子、周期键、幽灵像和晶胞构建。
# English overview: Verify atom, periodic-bond, ghost-image, and unit-cell scene construction.
#
# 中文输入：小型 Pymatgen 测试结构。
# English inputs: Small Pymatgen test structures.
# 中文输出：确定性的 unittest 断言。
# English outputs: Deterministic unittest assertions.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: WindowsPreviewTest
#
# Author: Ruck
# Generated: 2026-07-30 08:20:42 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import unittest

import numpy as np
from pymatgen.core import Lattice, Structure

from app.windows.nfe_mxene_studio.structure_preview import (
    build_structure_scene,
    covalent_radius,
    element_color,
)


# 中文：顶层类 `WindowsPreviewTest`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `WindowsPreviewTest`; review type hints and callers before extending it.
class WindowsPreviewTest(unittest.TestCase):
    def test_scene_contains_atoms_bonds_and_cell(self) -> None:
        structure = Structure(
            Lattice.hexagonal(3.1, 24.0),
            ["F", "Ti", "C", "V", "O"],
            [
                [1 / 3, 2 / 3, 0.37],
                [0.0, 0.0, 0.43],
                [1 / 3, 2 / 3, 0.50],
                [0.0, 0.0, 0.57],
                [1 / 3, 2 / 3, 0.63],
            ],
        )
        scene = build_structure_scene(structure)
        self.assertEqual(len(scene.positions), 5)
        self.assertEqual(len(scene.symbols), 5)
        self.assertEqual(len(scene.colors), 5)
        self.assertEqual(len(scene.cell_segments), 12)
        self.assertGreater(len(scene.bonds), 0)
        self.assertTrue(np.all(scene.marker_sizes > 0))
        atom_minimum, atom_maximum = scene.atom_limits
        full_minimum, full_maximum = scene.full_limits
        self.assertTrue(np.all(atom_maximum > atom_minimum))
        self.assertTrue(np.all(full_maximum > full_minimum))

    def test_element_visual_properties_are_valid(self) -> None:
        for symbol in ("H", "C", "N", "O", "Ti", "Nb", "Ta", "Se", "Br"):
            color = element_color(symbol)
            self.assertRegex(color, r"^#[0-9A-Fa-f]{6}$")
            self.assertGreater(covalent_radius(symbol), 0)


if __name__ == "__main__":
    unittest.main()
