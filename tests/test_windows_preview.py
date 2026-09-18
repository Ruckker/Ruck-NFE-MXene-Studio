# ==============================================================================
# 中文概述：验证三维预览：场景构建（周期键、镜像、超胞、层、片层晶胞框）、渲染（各模式与主题、空状态）、
#           拾取、测量、视角与适配，以及 Tk 组件的点击选中、拖动旋转、滚轮缩放与导出。
# English overview: Verify the 3D preview: scene building (periodic bonds, images, supercells, layers, slab box),
#           rendering (modes, themes, empty state), picking, measurements, views and fitting, and the Tk widget's
#           click selection, drag rotation, wheel zoom and export.
#
# 中文输入：小型 Pymatgen 测试结构与 examples/structures 样例。
# English inputs: Small Pymatgen test structures and the samples in examples/structures.
# 中文输出：确定性的 unittest 断言。
# English outputs: Deterministic unittest assertions.
#
# Author: Ruck
# Generated: 2026-07-30 08:20:42 Asia/Shanghai；修订 / Revised: 2026-09-15
# ==============================================================================

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from pymatgen.core import Lattice, Structure

from app.windows.nfe_mxene_studio.structure_preview import build_structure_scene, covalent_radius, element_color
from app.windows.nfe_mxene_studio.structure_render import (
    Camera,
    RenderOptions,
    fit_camera,
    measurement_lines,
    pick_atom,
    render_scene,
    view_rotation,
)
from app.windows.nfe_mxene_studio.structure_scene import build_display_model

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "structures" / "sample_high_ZrTiHSNO.cif"


def small_mxene() -> Structure:
    return Structure(
        Lattice.hexagonal(3.1, 24.0),
        ["F", "Ti", "C", "V", "O"],
        [[1 / 3, 2 / 3, 0.37], [0.0, 0.0, 0.43], [1 / 3, 2 / 3, 0.50], [0.0, 0.0, 0.57], [1 / 3, 2 / 3, 0.63]],
    )


class WindowsPreviewTest(unittest.TestCase):
    def test_scene_contains_atoms_bonds_and_cell(self) -> None:
        scene = build_structure_scene(small_mxene())
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
        for symbol in ("H", "C", "N", "O", "Ti", "Nb", "Ta", "Se", "Br", "Mn", "Te", "Xe"):
            self.assertRegex(element_color(symbol), r"^#[0-9A-Fa-f]{6}$")
            self.assertGreater(covalent_radius(symbol), 0)


class DisplayModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.structure = Structure.from_file(SAMPLE)

    def test_supercell_images_layers_and_slab_box(self) -> None:
        single = build_display_model(self.structure, (1, 1), show_images=False)
        block = build_display_model(self.structure, (3, 2), show_images=False)
        with_images = build_display_model(self.structure, (1, 1), show_images=True)
        self.assertEqual(int((~single.is_image).sum()), len(self.structure))
        self.assertEqual(int((~block.is_image).sum()), 6 * len(self.structure))
        self.assertEqual(int(block.is_image.sum()), 0)
        self.assertGreater(int(with_images.is_image.sum()), 0)
        self.assertGreater(len(block.bonds), len(single.bonds))
        self.assertEqual(single.layer_count, 6)
        self.assertEqual(single.normal_axis, 2)
        self.assertAlmostEqual(single.thickness_A + single.vacuum_A, self.structure.lattice.c, places=3)
        slab_height = max(abs(float(np.dot(end - start, single.normal))) for start, end, kind in single.cell_edges if kind == "c")
        full_height = max(abs(float(np.dot(end - start, single.normal))) for start, end, kind in single.full_cell_edges if kind == "c")
        self.assertAlmostEqual(slab_height, single.thickness_A + 2.0, places=3)
        self.assertAlmostEqual(full_height, self.structure.lattice.c, places=3)
        # every bond joins atoms that are chemically close
        for first, second in block.bonds:
            self.assertLess(float(np.linalg.norm(block.positions[first] - block.positions[second])), 3.66)

    def test_slab_split_across_the_cell_boundary_is_made_contiguous(self) -> None:
        wrapped = Structure(self.structure.lattice, self.structure.species, np.mod(self.structure.frac_coords + [0.0, 0.0, 0.5], 1.0))
        model = build_display_model(wrapped, (1, 1), show_images=False)
        self.assertAlmostEqual(model.thickness_A, build_display_model(self.structure).thickness_A, places=3)


class RenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = build_display_model(Structure.from_file(SAMPLE), (2, 2), show_images=False)
        self.camera = Camera(rotation=view_rotation(self.model, "iso"))
        self.options = RenderOptions(title="sample")
        fit_camera(self.model, self.camera, 480, 320, self.options)

    def test_views_are_rotations(self) -> None:
        for name in ("iso", "top", "side_a", "side_b"):
            rotation = view_rotation(self.model, name)
            self.assertTrue(np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9), name)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)
        top = view_rotation(self.model, "top")
        self.assertTrue(np.allclose(top[2], self.model.normal))

    def test_render_modes_and_themes_produce_images(self) -> None:
        background = render_scene(None, Camera(), RenderOptions(), 480, 320, "draft").image
        for mode in ("ball", "space", "stick", "wire"):
            for theme in ("light", "dark", "white"):
                options = RenderOptions(mode=mode, theme=theme, show_labels=True, selection=(0, 1, 2))
                result = render_scene(self.model, self.camera, options, 480, 320, "draft")
                self.assertEqual(result.image.size, (480, 320))
                self.assertEqual(len(result.screen_xy), len(self.model.positions))
                if theme == "light":
                    changed = np.mean(np.any(np.asarray(result.image) != np.asarray(background), axis=-1))
                    self.assertGreater(changed, 0.02, mode)
        final = render_scene(self.model, self.camera, self.options, 480, 320, "final")
        self.assertEqual(final.image.size, (480, 320))
        exported = render_scene(self.model, self.camera, self.options, 480, 320, "final", pixel_ratio=2.0)
        self.assertEqual(exported.image.size, (960, 640))

    def test_fit_keeps_atoms_inside_the_canvas(self) -> None:
        result = render_scene(self.model, self.camera, self.options, 480, 320, "draft")
        xy, r = result.screen_xy, result.screen_r
        self.assertTrue(np.all(xy[:, 0] - r >= -1) and np.all(xy[:, 0] + r <= 481))
        self.assertTrue(np.all(xy[:, 1] - r >= -1) and np.all(xy[:, 1] + r <= 321))

    def test_pick_returns_the_nearest_atom(self) -> None:
        result = render_scene(self.model, self.camera, self.options, 480, 320, "draft")
        nearest = int(np.argmax(result.depth))
        self.assertEqual(pick_atom(result, *result.screen_xy[nearest]), nearest)
        self.assertIsNone(pick_atom(result, -500.0, -500.0))

    def test_measurements(self) -> None:
        structure = Structure(Lattice.cubic(20.0), ["O", "H", "H", "C"], [[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0.05, 0.05]], coords_are_cartesian=False)
        model = build_display_model(structure, (1, 1), show_images=False)
        lines = measurement_lines(model, (1, 0, 2, 3))
        self.assertIn("1.000 Å", lines[0])
        self.assertTrue(any("90.00°" in line for line in lines))
        self.assertTrue(any(line.startswith("二面角") for line in lines))


class PreviewWidgetTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import tkinter as tk

            self.root = tk.Tk()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"Tk unavailable: {exc}")
        self.root.withdraw()
        from app.windows.nfe_mxene_studio.structure_preview import StructurePreview3D

        self.preview = StructurePreview3D(self.root)
        self.preview.canvas.configure(width=640, height=360)
        self.preview.pack(fill="both", expand=True)
        self.root.update_idletasks()

    def tearDown(self) -> None:
        self.root.destroy()

    def test_click_drag_zoom_views_and_export(self) -> None:
        preview = self.preview
        preview.render_now("final")
        preview.set_path(SAMPLE)
        preview.render_now("final")
        self.assertEqual(preview.result.image.size, (640, 360))
        atom = int(np.argmax(preview.result.depth))
        x, y = preview.result.screen_xy[atom]
        event = SimpleNamespace(x=int(round(x)), y=int(round(y)), num=1, state=0, x_root=0, y_root=0)
        preview._on_press(event)
        preview._on_release(event)
        self.assertEqual(preview.selection, [atom])
        self.assertIn("笛卡尔坐标", preview.info_text.get("1.0", "end"))
        before = preview.camera.rotation.copy()
        preview._on_press(SimpleNamespace(x=100, y=100, num=1, state=0))
        preview._on_drag(SimpleNamespace(x=160, y=130))
        preview._on_release(SimpleNamespace(x=160, y=130, num=1, state=0))
        self.assertFalse(np.allclose(before, preview.camera.rotation))
        scale = preview.camera.scale
        preview._on_wheel(SimpleNamespace(x=320, y=180, delta=120))
        self.assertGreater(preview.camera.scale, scale)
        preview.set_view("top")
        self.assertTrue(np.allclose(preview.camera.rotation[2], preview.model.normal))
        preview.repeat_a_var.set("4")
        preview._repeat_changed()
        self.assertEqual(preview.model.repeat, (4, 3))
        preview.mode_var.set("空间填充")
        preview._option_changed(refit=True)
        with tempfile.TemporaryDirectory() as directory:
            path = preview.save_image(Path(directory) / "view.png", pixel_ratio=2.0)
            with Image.open(path) as image:
                self.assertEqual(image.size, (1280, 720))
        preview.clear()
        preview.render_now("draft")
        self.assertIsNone(preview.model)
        self.assertTrue(math.isfinite(preview.camera.scale))

    def test_narrow_hosts_wrap_the_toolbar_and_hide_the_panel(self) -> None:
        preview = self.preview
        preview._on_widget_resize(SimpleNamespace(widget=preview, width=560))
        self.assertEqual(int(preview.toolbar_secondary.grid_info()["row"]), 1)
        self.assertFalse(preview.panel_var.get())
        self.assertFalse(preview.panel.winfo_manager())
        preview._on_widget_resize(SimpleNamespace(widget=preview, width=1400))
        self.assertEqual(int(preview.toolbar_secondary.grid_info()["row"]), 0)
        self.assertTrue(preview.panel_var.get())
        self.assertEqual(preview.panel.winfo_manager(), "pack")
        preview.panel_var.set(False)
        preview._toggle_panel()
        preview._on_widget_resize(SimpleNamespace(widget=preview, width=1500))
        self.assertFalse(preview.panel_var.get(), "a manual choice must survive resizing")


if __name__ == "__main__":
    unittest.main()
