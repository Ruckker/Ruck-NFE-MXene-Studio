# 中文：Windows application 1.1.1 的 PyInstaller spec：复用生产 spec，只改产物名。
# English: PyInstaller spec for Windows application 1.1.1: reuse the production spec, rename the output.
# Author: Ruck
# Generated: 2026-09-15
# 1.1.1 与 1.1.0 的差别 / Changes since 1.1.0: 导入时检查输入是否为 MXene 片层（nfe_model.mxene_validation），
# 不合法文件单独排除并弹窗说明；模型权重与 1.1.0 相同。
from pathlib import Path


production_spec = Path(SPECPATH) / "NFE_MXene_Studio.spec"
source = production_spec.read_text(encoding="utf-8")
source = source.replace(
    'name="NFE_MXene_Studio"',
    'name="NFE_MXene_Studio_1_1_1"',
)
exec(compile(source, str(production_spec), "exec"), globals(), globals())
