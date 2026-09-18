# ==============================================================================
# 中文概述：在真实 Tk 窗口中验证导入流程：单个非 MXene 文件弹出错误，多文件只排除不合法文件并弹出警告，
#           重复导入不弹窗。需要 tkinterdnd2 与桌面显示，缺少时自动跳过。
# English overview: Verify the import flow in a real Tk window: a single non-MXene file raises an error dialog,
#           multi-file input excludes only the invalid files with a warning, and re-importing shows no dialog.
#           Needs tkinterdnd2 and a desktop display; skipped otherwise.
#
# 中文输入：examples/structures 样例、岩盐 TiC 体相与文本文件。
# English inputs: Samples in examples/structures, bulk rock-salt TiC and a text file.
# 中文输出：unittest 断言（记录的弹窗与输入列表）。
# English outputs: Unittest assertions on recorded dialogs and the input list.
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from pymatgen.core import Lattice, Structure

try:
    import tkinter as tk

    import tkinterdnd2  # noqa: F401

    from app.windows.nfe_mxene_studio import app as app_module

    GUI_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - any missing GUI dependency skips the test
    GUI_IMPORT_ERROR = exc

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "examples" / "structures"


@unittest.skipIf(GUI_IMPORT_ERROR is not None, f"GUI stack unavailable: {GUI_IMPORT_ERROR}")
class WindowsInputValidationGuiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dialogs: list[tuple[str, str, str]] = []
        fake_messagebox = mock.Mock()
        fake_messagebox.showerror.side_effect = lambda title, message, **_: self.dialogs.append(("error", title, message))
        fake_messagebox.showwarning.side_effect = lambda title, message, **_: self.dialogs.append(("warning", title, message))
        fake_messagebox.showinfo.side_effect = lambda title, message, **_: self.dialogs.append(("info", title, message))
        self.patches = [
            mock.patch.object(app_module, "messagebox", fake_messagebox),
            mock.patch.object(app_module.NFEMXeneApp, "_initialize_engine", lambda self: None),
        ]
        for patch in self.patches:
            patch.start()
        try:
            self.window = app_module.NFEMXeneApp()
        except tk.TclError as exc:
            for patch in self.patches:
                patch.stop()
            self.skipTest(f"no Tk display: {exc}")
        self.window.withdraw()
        self.directory = Path(tempfile.mkdtemp())
        self.low = self.directory / "sample_low_ScTaCSeBr.cif"
        self.medium = self.directory / "sample_medium_TiNbCSeCl.cif"
        shutil.copy2(SAMPLES / "sample_low_ScTaCSeBr.cif", self.low)
        shutil.copy2(SAMPLES / "sample_medium_TiNbCSeCl.cif", self.medium)
        self.bulk = self.directory / "bulk_TiC.cif"
        Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.33), ["Ti", "C"], [[0, 0, 0], [0.5, 0.5, 0.5]]).to(filename=str(self.bulk))
        self.text = self.directory / "readme.txt"
        self.text.write_text("not a structure", encoding="utf-8")

    def tearDown(self) -> None:
        self.window.destroy()
        for patch in self.patches:
            patch.stop()
        shutil.rmtree(self.directory, ignore_errors=True)

    def drop(self, *paths: Path) -> None:
        # The worker thread marshals results with `after`, which Tk only accepts while the main
        # thread runs `mainloop`; start the import inside the loop and quit once it has finished.
        values = [str(path) for path in paths]
        deadline = time.time() + 120
        started: list[bool] = []

        def poll() -> None:
            if self.window.validating_inputs and time.time() < deadline:
                self.window.after(50, poll)
            else:
                self.window.after(50, self.window.quit)

        def start() -> None:
            started.append(True)
            self.window._add_inputs(values)
            self.window.after(50, poll)

        self.window.after(0, start)
        self.window.mainloop()
        self.window.update()
        self.assertTrue(started, "import was not started")
        self.assertFalse(self.window.validating_inputs, "validation did not finish")

    def test_single_invalid_file_shows_error_and_adds_nothing(self) -> None:
        self.drop(self.bulk)
        self.assertEqual(self.window.input_files, [])
        self.assertEqual(len(self.dialogs), 1)
        kind, title, message = self.dialogs[0]
        self.assertEqual((kind, title), ("error", "输入文件不合法"))
        self.assertIn("bulk_TiC.cif", message)
        self.assertEqual(str(self.window.predict_button["state"]), "normal")

    def test_multi_file_input_excludes_invalid_files_with_warning(self) -> None:
        self.drop(self.low, self.bulk, self.text, self.medium)
        self.assertEqual([path.name for path in self.window.input_files], [self.low.name, self.medium.name])
        self.assertEqual(self.window.input_list.size(), 2)
        self.assertEqual(len(self.dialogs), 1)
        kind, title, message = self.dialogs[0]
        self.assertEqual((kind, title), ("warning", "已排除不合法的输入文件"))
        self.assertIn("bulk_TiC.cif", message)
        self.assertIn("readme.txt", message)
        self.assertIn("排除 2 个不合法文件", self.window.status_var.get())

    def test_reimporting_valid_files_shows_no_dialog(self) -> None:
        self.drop(self.low)
        self.drop(self.low)
        self.assertEqual(len(self.window.input_files), 1)
        self.assertEqual(self.dialogs, [])
        self.assertIn("跳过 1 个已在列表中的文件", self.window.status_var.get())


if __name__ == "__main__":
    unittest.main()
