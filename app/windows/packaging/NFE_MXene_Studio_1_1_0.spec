# 中文：Windows application 1.1.0 的 PyInstaller spec：复用生产 spec，只改产物名。
# English: PyInstaller spec for Windows application 1.1.0: reuse the production spec, rename the output.
# Author: Ruck
# Generated: 2026-09-15
# 1.1.0 与 1.0.1 的差别 / Changes since 1.0.1: 预测器换为在 nfe-v1.1 表（PROCAR 自旋修复）上以
# 规范化输入、完整壳层、无全局特征训练的 v1.1 检查点；生成器权重不变。
from pathlib import Path


production_spec = Path(SPECPATH) / "NFE_MXene_Studio.spec"
source = production_spec.read_text(encoding="utf-8")
source = source.replace(
    'name="NFE_MXene_Studio"',
    'name="NFE_MXene_Studio_1_1_0"',
)
exec(compile(source, str(production_spec), "exec"), globals(), globals())
