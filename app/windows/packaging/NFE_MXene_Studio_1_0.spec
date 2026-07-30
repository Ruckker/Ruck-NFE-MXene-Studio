# 中文：`NFE_MXene_Studio_1_0.spec` 的配置、依赖或构建说明。
# English: Configuration, dependency, or build instructions for `NFE_MXene_Studio_1_0.spec`.
# Author: Ruck
# Generated: 2026-07-30 08:28:48 Asia/Shanghai
# 提示 / Tip: 修改依赖、路径或阈值后，请重新运行测试与冒烟验证。
# Re-run tests and smoke validation after changing dependencies, paths, or thresholds.
# Final windowed 1.0 release with interactive CIF/POSCAR 3D previews.
from pathlib import Path


production_spec = Path(SPECPATH) / "NFE_MXene_Studio.spec"
source = production_spec.read_text(encoding="utf-8")
source = source.replace(
    'name="NFE_MXene_Studio"',
    'name="NFE_MXene_Studio_1_0"',
)
exec(compile(source, str(production_spec), "exec"), globals(), globals())
