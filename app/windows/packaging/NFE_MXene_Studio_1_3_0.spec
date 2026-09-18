# 中文：Windows application 1.3.0 的 PyInstaller spec：复用生产 spec，只改产物名。
# English: PyInstaller spec for Windows application 1.3.0: reuse the production spec, rename the output.
# Author: Ruck
# Generated: 2026-09-15
# 1.3.0 与 1.2.0 的差别 / Changes since 1.2.0: 内置生成器换为在 nfe-v1.1 标签上重训的表面生成器
# （models/mxene_generator.pt，父检查点 SHA256 a1d8eb5d…）；代码与预测器不变。
from pathlib import Path


production_spec = Path(SPECPATH) / "NFE_MXene_Studio.spec"
source = production_spec.read_text(encoding="utf-8")
source = source.replace(
    'name="NFE_MXene_Studio"',
    'name="NFE_MXene_Studio_1_3_0"',
)
exec(compile(source, str(production_spec), "exec"), globals(), globals())
