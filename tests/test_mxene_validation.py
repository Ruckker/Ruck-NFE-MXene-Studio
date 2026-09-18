# ==============================================================================
# 中文概述：验证 MXene 输入判定：样例与各种等价表示合法，训练域外的 MXene 合法但带提示，
#           体相、非碳/氮化物、异质元素、错误层序、游离端基、无序位点和坏文件均被拒绝；
#           并验证 Windows 导入筛选与弹窗文案。
# English overview: Verify the MXene input check: samples and equivalent representations pass, MXenes outside
#           the training domain pass with notes, and bulk crystals, non-carbide/nitride slabs, foreign
#           elements, wrong layer order, floating terminations, disordered sites and broken files are rejected;
#           also verify the Windows import screening and dialog texts.
#
# 中文输入：examples/structures 样例与代码内构造的结构。
# English inputs: Samples in examples/structures and structures built in code.
# 中文输出：unittest 断言。
# English outputs: Unittest assertions.
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from app.windows.nfe_mxene_studio.backend import (
    MISSING_INPUT_REASON,
    UNSUPPORTED_INPUT_REASON,
    rejection_dialog,
    screen_input_paths,
)
from nfe_model.mxene_validation import check_mxene_file, check_mxene_structure

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "examples" / "structures"
A_SITE, B_SITE, C_SITE = (0.0, 0.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0 / 3.0)


def slab(layers: list[tuple[str, tuple[float, float], float]], a: float = 3.07, c: float = 30.0) -> Structure:
    lattice = Lattice.hexagonal(a, c)
    species = [symbol for symbol, _site, _z in layers]
    coords = [[site[0], site[1], 0.5 + z / c] for _symbol, site, z in layers]
    return Structure(lattice, species, coords)


def ti2co2() -> Structure:
    return slab(
        [
            ("O", B_SITE, -2.15),
            ("Ti", A_SITE, -1.15),
            ("C", B_SITE, 0.0),
            ("Ti", C_SITE, 1.15),
            ("O", A_SITE, 2.15),
        ]
    )


class MXeneStructureCheckTest(unittest.TestCase):
    def assertInvalid(self, structure: Structure, fragment: str) -> None:
        check = check_mxene_structure(structure)
        self.assertFalse(check.valid, check)
        self.assertIn(fragment, check.message)

    def test_samples_are_valid_training_domain_mxenes(self) -> None:
        for path in sorted(SAMPLES.iterdir()):
            with self.subTest(path=path.name):
                check = check_mxene_file(path)
                self.assertTrue(check.valid, check.message)
                self.assertEqual(check.n, 1)
                self.assertTrue(check.in_training_domain, check.domain_notes)

    def test_equivalent_representations_stay_valid(self) -> None:
        base = Structure.from_file(SAMPLES / "sample_high_ZrTiHSNO.cif")
        matrix = np.array(base.lattice.matrix)
        cartesian = base.cart_coords
        species = [site.specie for site in base]
        supercell = base.copy()
        supercell.make_supercell([2, 2, 1])
        thin = matrix.copy()
        thin[2] = [0.0, 0.0, 15.0]
        shifted = cartesian.copy()
        shifted[:, 2] += (15.0 - base.lattice.c) / 2
        variants = {
            "2x2x1": supercell,
            "vacuum 15 A": Structure(Lattice(thin), species, shifted, coords_are_cartesian=True),
            "vacuum on a": Structure(Lattice(np.array([matrix[2], matrix[0], matrix[1]])), species, base.frac_coords[:, [2, 0, 1]]),
            "gamma 60": Structure(Lattice(np.array([matrix[0], matrix[0] + matrix[1], matrix[2]])), species, cartesian, coords_are_cartesian=True, to_unit_cell=True),
            "wrapped across c": Structure(base.lattice, species, np.mod(base.frac_coords + [0.0, 0.0, 0.5], 1.0)),
        }
        for name, structure in variants.items():
            with self.subTest(variant=name):
                check = check_mxene_structure(structure)
                self.assertTrue(check.valid, check.message)
                self.assertTrue(check.in_training_domain)

    def test_off_centre_relaxed_nitride_is_valid(self) -> None:
        # N only 0.30 A above the W plane, as in relaxed Cr-W-N training structures.
        structure = slab(
            [
                ("I", C_SITE, -2.54),
                ("W", A_SITE, -0.30),
                ("N", B_SITE, 0.0),
                ("Cr", C_SITE, 1.91),
                ("S", A_SITE, 3.07),
            ],
            a=2.9,
        )
        check = check_mxene_structure(structure)
        self.assertTrue(check.valid, check.message)
        self.assertTrue(check.in_training_domain, check.domain_notes)

    def test_mxenes_outside_training_domain_are_valid_with_notes(self) -> None:
        ti3c2o2 = slab(
            [
                ("O", B_SITE, -3.3),
                ("Ti", A_SITE, -2.3),
                ("C", B_SITE, -1.15),
                ("Ti", C_SITE, 0.0),
                ("C", A_SITE, 1.15),
                ("Ti", B_SITE, 2.3),
                ("O", C_SITE, 3.3),
            ]
        )
        check = check_mxene_structure(ti3c2o2)
        self.assertTrue(check.valid, check.message)
        self.assertEqual(check.n, 2)
        self.assertFalse(check.in_training_domain)
        self.assertTrue(any("n = 2" in note for note in check.domain_notes))
        bare = check_mxene_structure(slab([("Ti", A_SITE, -1.15), ("C", B_SITE, 0.0), ("Ti", C_SITE, 1.15)]))
        self.assertTrue(bare.valid, bare.message)
        self.assertIn("底面没有端基", bare.domain_notes)

    def test_bulk_crystal_is_rejected(self) -> None:
        bulk = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.33), ["Ti", "C"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.assertInvalid(bulk, "没有真空层")

    def test_non_mxene_slabs_are_rejected(self) -> None:
        mos2 = Structure(
            Lattice.hexagonal(3.16, 20.0),
            ["Mo", "S", "S"],
            [[0, 0, 0.5], [1 / 3, 2 / 3, 0.5 + 1.56 / 20], [1 / 3, 2 / 3, 0.5 - 1.56 / 20]],
        )
        self.assertInvalid(mos2, "没有 C 或 N")
        graphene = Structure(Lattice.hexagonal(2.46, 20.0), ["C", "C"], [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]])
        self.assertInvalid(graphene, "没有 MXene 的过渡金属")
        with_sodium = ti2co2()
        with_sodium.append("Na", [1 / 3, 2 / 3, 0.5 + 4.0 / 30.0])
        self.assertInvalid(with_sodium, "含有 MXene 以外的元素：Na")

    def test_wrong_stoichiometry_and_layer_order_are_rejected(self) -> None:
        ti2c2 = slab([("Ti", A_SITE, -1.7), ("C", B_SITE, -0.575), ("Ti", C_SITE, 0.575), ("C", A_SITE, 1.7)])
        self.assertInvalid(ti2c2, "不符合 M(n+1)X(n)")
        carbon_outside = slab([("Ti", A_SITE, -1.15), ("Ti", C_SITE, 1.15), ("C", B_SITE, 2.3)])
        self.assertInvalid(carbon_outside, "没有全部夹在金属层之间")
        oxygen_in_core = slab([("Ti", A_SITE, -1.15), ("C", B_SITE, 0.0), ("O", C_SITE, 0.2), ("Ti", C_SITE, 1.15)])
        self.assertInvalid(oxygen_in_core, "端基原子位于金属层之间")

    def test_floating_termination_and_flake_are_rejected(self) -> None:
        floating = ti2co2()
        floating.append("O", [1 / 3, 2 / 3, 0.5 + 7.0 / 30.0])
        self.assertInvalid(floating, "没有与表面金属成键")
        flake = Structure(
            Lattice.cubic(20.0),
            ["Ti", "C", "Ti"],
            [[10.0, 10.0, 8.85], [11.5, 10.9, 10.0], [10.0, 11.8, 11.15]],
            coords_are_cartesian=True,
        )
        self.assertInvalid(flake, "个方向上都有真空")

    def test_disordered_and_unreadable_inputs_are_rejected(self) -> None:
        disordered = Structure(
            Lattice.hexagonal(3.07, 30.0),
            [{"O": 0.5, "F": 0.5}, "Ti", "C", "Ti"],
            [[1 / 3, 2 / 3, 0.43], [0, 0, 0.46], [1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.54]],
        )
        self.assertInvalid(disordered, "无序位点")
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.cif"
            broken.write_text("this is not a CIF file\n", encoding="utf-8")
            check = check_mxene_file(broken)
            self.assertFalse(check.valid)
            self.assertIn("无法读取为晶体结构", check.message)


class InputScreeningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.low = self.directory / "sample_low_ScTaCSeBr.cif"
        self.high = self.directory / "sample_high_ZrTiHSNO.cif"
        shutil.copy2(SAMPLES / "sample_low_ScTaCSeBr.cif", self.low)
        shutil.copy2(SAMPLES / "sample_high_ZrTiHSNO.cif", self.high)
        self.bulk = self.directory / "bulk_TiC.cif"
        Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.33), ["Ti", "C"], [[0, 0, 0], [0.5, 0.5, 0.5]]).to(filename=str(self.bulk))
        self.text = self.directory / "notes.txt"
        self.text.write_text("not a structure", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_single_invalid_file_is_an_error_dialog(self) -> None:
        screening = screen_input_paths([self.bulk])
        self.assertEqual(screening.accepted, [])
        self.assertEqual(screening.considered, 1)
        kind, title, message = rejection_dialog(screening)
        self.assertEqual((kind, title), ("error", "输入文件不合法"))
        self.assertIn("bulk_TiC.cif", message)
        self.assertIn("没有真空层", message)

    def test_multiple_files_exclude_only_invalid_ones(self) -> None:
        screening = screen_input_paths([self.low, self.bulk, self.text, self.high])
        self.assertEqual([path.name for path in screening.accepted], [self.low.name, self.high.name])
        reasons = dict((path.name, reason) for path, reason in screening.rejected)
        self.assertEqual(set(reasons), {"bulk_TiC.cif", "notes.txt"})
        self.assertEqual(reasons["notes.txt"], UNSUPPORTED_INPUT_REASON)
        kind, title, message = rejection_dialog(screening)
        self.assertEqual((kind, title), ("warning", "已排除不合法的输入文件"))
        self.assertIn("其中 2 个不是合法的 MXene 结构", message)
        self.assertIn("其余 2 个已加入输入列表", message)

    def test_folder_skips_non_structure_files_and_reports_invalid_structures(self) -> None:
        screening = screen_input_paths([self.directory])
        self.assertEqual(len(screening.accepted), 2)
        self.assertEqual([path.name for path, _reason in screening.rejected], ["bulk_TiC.cif"])

    def test_all_invalid_multi_file_input_is_an_error(self) -> None:
        missing = self.directory / "missing.cif"
        screening = screen_input_paths([self.bulk, self.text, missing])
        kind, _title, message = rejection_dialog(screening)
        self.assertEqual(kind, "error")
        self.assertIn("全部不是合法的 MXene 结构", message)
        self.assertIn(MISSING_INPUT_REASON, dict((p.name, r) for p, r in screening.rejected)["missing.cif"])

    def test_duplicates_and_known_files_are_skipped_without_dialog(self) -> None:
        screening = screen_input_paths([self.low, self.low, self.high], known=[self.high.resolve()])
        self.assertEqual([path.name for path in screening.accepted], [self.low.name])
        self.assertEqual(screening.duplicates, 2)
        self.assertIsNone(rejection_dialog(screening))

    def test_dialog_truncates_long_lists(self) -> None:
        paths = []
        for index in range(14):
            path = self.directory / f"bad_{index:02d}.txt"
            path.write_text("x", encoding="utf-8")
            paths.append(path)
        kind, _title, message = rejection_dialog(screen_input_paths(paths), limit=12)
        self.assertEqual(kind, "error")
        self.assertIn("另有 2 个不合法文件", message)


if __name__ == "__main__":
    unittest.main()
