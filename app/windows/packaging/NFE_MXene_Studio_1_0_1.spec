# 中文：Windows application 1.0.1 的 PyInstaller spec：复用生产 spec，只改产物名。
# English: PyInstaller spec for Windows application 1.0.1: reuse the production spec, rename the output.
# Author: Ruck
# Generated: 2026-09-15
# 1.0.1 与 1.0 的差别 / Changes since 1.0: 预测前输入规范化（真空/晶格设定/超胞/平移不变）、
# pymatgen Windows int64 补丁移入核心包、high-vs-rest 二分类输出、生成器无流基线、
# 骨架模板全在训练集时显式允许训练匹配。
from pathlib import Path


production_spec = Path(SPECPATH) / "NFE_MXene_Studio.spec"
source = production_spec.read_text(encoding="utf-8")
source = source.replace(
    'name="NFE_MXene_Studio"',
    'name="NFE_MXene_Studio_1_0_1"',
)
exec(compile(source, str(production_spec), "exec"), globals(), globals())
