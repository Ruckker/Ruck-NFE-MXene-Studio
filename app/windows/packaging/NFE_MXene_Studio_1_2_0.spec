# 中文：Windows application 1.2.0 的 PyInstaller spec：复用生产 spec，只改产物名。
# English: PyInstaller spec for Windows application 1.2.0: reuse the production spec, rename the output.
# Author: Ruck
# Generated: 2026-09-15
# 1.2.0 与 1.1.1 的差别 / Changes since 1.1.1: 三维预览换为 NumPy + Pillow 软件渲染（structure_scene.py、
# structure_render.py），新增视角预设、显示模式、超胞、拾取测量与导出；模型权重不变。
from pathlib import Path


production_spec = Path(SPECPATH) / "NFE_MXene_Studio.spec"
source = production_spec.read_text(encoding="utf-8")
source = source.replace(
    'name="NFE_MXene_Studio"',
    'name="NFE_MXene_Studio_1_2_0"',
)
exec(compile(source, str(production_spec), "exec"), globals(), globals())
