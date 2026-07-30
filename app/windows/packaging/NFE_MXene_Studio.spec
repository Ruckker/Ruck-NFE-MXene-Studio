# 中文：`NFE_MXene_Studio.spec` 的配置、依赖或构建说明。
# English: Configuration, dependency, or build instructions for `NFE_MXene_Studio.spec`.
# Author: Ruck
# Generated: 2026-07-30 08:28:48 Asia/Shanghai
# 提示 / Tip: 修改依赖、路径或阈值后，请重新运行测试与冒烟验证。
# Re-run tests and smoke validation after changing dependencies, paths, or thresholds.
# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
)


APP_DIR = Path(SPECPATH).resolve()
ROOT = APP_DIR.parent

datas = [
    (
        str(APP_DIR / "models" / "nfe_predictor.pt"),
        "models",
    ),
    (
        str(APP_DIR / "models" / "mxene_generator.pt"),
        "models",
    ),
    (
        str(APP_DIR / "resources" / "surface_geometry_summary.json"),
        "resources",
    ),
    (
        str(APP_DIR / "resources" / "predictor_final_metrics.json"),
        "resources",
    ),
    (
        str(APP_DIR / "resources" / "generator_final_metrics.json"),
        "resources",
    ),
    (
        str(APP_DIR / "samples"),
        "samples",
    ),
]
datas += collect_data_files("chgnet")
datas += collect_data_files("pymatgen")
datas += collect_data_files("tkinterdnd2")

binaries = collect_dynamic_libs("chgnet")

hiddenimports = []
hiddenimports += [
    "backports",
    "backports.tarfile",
    "chgnet.graph.converter",
    "chgnet.graph.crystalgraph",
    "chgnet.graph.graph",
    "chgnet.model.dynamics",
    "chgnet.model.model",
    "matplotlib.backends.backend_tkagg",
    "mpl_toolkits.mplot3d",
    "pymatgen.analysis.structure_matcher",
    "pymatgen.io.cif",
    "pymatgen.io.vasp",
    "pymatgen.optimization.neighbors",
    "scipy.optimize",
    "scipy.spatial",
]

a = Analysis(
    [str(APP_DIR / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "Cython",
        "ase.test",
        "chgnet.trainer",
        "jupyter",
        "lightning",
        "notebook",
        "plotly",
        "pytorch_lightning",
        "pytest",
        "torch._dynamo",
        "torch._inductor",
        "torch.distributed.elastic",
        "torchmetrics",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NFE_MXene_Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="NFE_MXene_Studio",
)
