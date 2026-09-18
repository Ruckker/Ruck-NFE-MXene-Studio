# ==============================================================================
# 中文概述：验证 VASP 结果提取器的 PROCAR 自旋块解析与候选带选择。
# English overview: Verify PROCAR spin-block parsing and candidate-band selection in the extractor.
#
# 中文输入：合成的 ISPIN=2 PROCAR 文本与合成能带数据。
# English inputs: Synthetic ISPIN=2 PROCAR text and synthetic band data.
# 中文输出：确定性的 unittest 断言。
# English outputs: Deterministic unittest assertions.
#
# 关键约束 / Key invariants:
# - PROCAR 的自旋块以重复的 "# of k-points" 头行分隔，没有 "spin component" 行。
#   PROCAR spin blocks are separated by repeated "# of k-points" preambles; there is no "spin component" line.
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_nfe_dataset", ROOT / "data_tools" / "build_nfe_dataset.py"
)
extractor = importlib.util.module_from_spec(SPEC)
sys.modules["build_nfe_dataset"] = extractor
SPEC.loader.exec_module(extractor)


def _procar_block(values_by_band: list[list[float]], nkpoints: int) -> str:
    """One spin block: preamble, k-points, bands, two ions and a tot line per band."""
    lines = [
        f"# of k-points:   {nkpoints}         # of bands:   {len(values_by_band)}         # of ions:    2",
        "",
    ]
    for k in range(1, nkpoints + 1):
        lines.append(
            f" k-point     {k} :    0.00000000 0.00000000 0.00000000     weight = 0.50000000"
        )
        lines.append("")
        for band, values in enumerate(values_by_band, start=1):
            lines.append(f"band     {band} # energy   -1.00000000 # occ.  1.00000000")
            lines.append("")
            lines.append("ion      s     py     pz     px    dxy    dyz    dz2    dxz    dx2    tot")
            lines.append("    1  0.010  0.010  0.010  0.010  0.010  0.010  0.010  0.010  0.010  0.090")
            lines.append("    2  0.010  0.010  0.010  0.010  0.010  0.010  0.010  0.010  0.010  0.090")
            s, p, d, tot = values
            lines.append(
                f"tot    {s:.3f}  {p/3:.3f}  {p/3:.3f}  {p/3:.3f}  {d/5:.3f}  {d/5:.3f}  {d/5:.3f}  {d/5:.3f}  {d/5:.3f}  {tot:.3f}"
            )
            lines.append("")
    return "\n".join(lines) + "\n"


class ProcarSpinBlockTest(unittest.TestCase):
    def test_both_spin_blocks_are_parsed_separately(self) -> None:
        up = [[0.10, 0.30, 0.20, 0.60], [0.05, 0.06, 0.05, 0.16]]
        down = [[0.40, 0.30, 0.20, 0.90], [0.20, 0.09, 0.10, 0.39]]
        text = "PROCAR lm decomposed\n" + _procar_block(up, 2) + "\n" + _procar_block(down, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PROCAR"
            path.write_text(text, encoding="utf-8")
            projections = extractor.parse_gamma_projections(path, {1, 2})
        spins = sorted({key[0] for key in projections})
        self.assertEqual(spins, [0, 1])
        self.assertAlmostEqual(projections[(0, 1, 0)]["total"], 0.60, places=6)
        self.assertAlmostEqual(projections[(1, 1, 0)]["total"], 0.90, places=6)
        self.assertAlmostEqual(projections[(1, 2, 1)]["total"], 0.39, places=6)
        self.assertAlmostEqual(projections[(0, 2, 1)]["p"], 0.06, places=6)
        self.assertEqual(len(projections), 8)

    def test_spin_down_candidate_can_win(self) -> None:
        # M-K-Gamma-M path with five points per segment; Gamma sits at indices 9 and 10.
        segment = 5
        k_segment_1 = [np.array([0.5, 0.0, 0.0]) * (1 - t) + np.array([1 / 3, 1 / 3, 0.0]) * t for t in np.linspace(0, 1, segment)]
        k_segment_2 = [np.array([1 / 3, 1 / 3, 0.0]) * (1 - t) for t in np.linspace(0, 1, segment)]
        k_segment_3 = [np.array([0.5, 0.0, 0.0]) * t for t in np.linspace(0, 1, segment)]
        kpoints = np.asarray(k_segment_1 + k_segment_2 + k_segment_3)
        structure = Structure(Lattice.hexagonal(3.1, 30.0), ["Ti", "C"], [[0, 0, 0.5], [1 / 3, 2 / 3, 0.55]])
        reciprocal = np.asarray(structure.lattice.reciprocal_lattice.matrix)
        cart = kpoints @ reciprocal
        k2 = np.sum(cart**2, axis=1)
        alpha = extractor.H2_OVER_2ME_EV_A2  # free-electron mass
        efermi = 0.0
        # spin up: far above E_F; spin down: right at E_F with a clean parabola.
        energies = np.stack([2.5 + alpha * k2, -0.1 + alpha * k2])[:, :, None]
        occupations = np.where(energies < efermi, 1.0, 0.0)
        data = extractor.EigenData(
            nelect=10.0,
            kpoints=kpoints,
            weights=np.ones(len(kpoints)),
            energies=energies,
            occupations=occupations,
        )
        gamma = {2 * segment, 2 * segment + 1}
        projections = {}
        for spin, total in ((0, 0.20), (1, 0.15)):
            for k in gamma:
                projections[(spin, k, 0)] = {"s": 0.05, "p": 0.05, "d": 0.05, "total": total}
        result = extractor.nfe_band_features(data, projections, structure, efermi, fit_points=4)
        self.assertEqual(result["NFE_Candidate_Spin"], "down")
        self.assertEqual(result["NFE_Pseudo_Label"], "high")
        self.assertAlmostEqual(result["NFE_Effective_Mass_Geomean_me"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
