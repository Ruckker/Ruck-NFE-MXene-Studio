# ==============================================================================
# 中文概述：pymatgen Windows 轮子的 int32/int64 兼容补丁；规范化、原胞约化与结构匹配都依赖它。
# English overview: int32/int64 compatibility patch for pymatgen Windows wheels; canonicalization,
#                   primitive-cell reduction and structure matching all depend on it.
#
# 中文输入：无（导入时按平台自动应用，幂等）。
# English inputs: None (applied on import for win32, idempotent).
# 中文输出：修补后的 `find_points_in_spheres` 调用，pbc 数组显式为 int64。
# English outputs: Patched `find_points_in_spheres` calls with an explicit int64 pbc array.
#
# 关键约束 / Key invariants:
# - 64 位 Windows 上 `np.array(..., dtype=int)` 是 int32，而 pymatgen 的 Cython 扩展要求
#   `const int64_t`；Linux 不暴露这个问题。1.0 只在桌面后端修补了 `Structure.get_neighbor_list`，
#   `Lattice.get_points_in_sphere`（原胞约化、StructureMatcher 使用）仍会失败。
#   On 64-bit Windows `dtype=int` is int32 while pymatgen's Cython extension expects int64; 1.0 only
#   patched `Structure.get_neighbor_list` in the desktop backend.
# - 主要接口 / Main APIs: ensure_pymatgen_int64_compatibility
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import sys
from typing import Any

import numpy as np

_PATCHED = False


# 中文：顶层接口 `ensure_pymatgen_int64_compatibility`；把 pbc 强制为 int64 后再调用 Cython 内核。
# English: Top-level function `ensure_pymatgen_int64_compatibility`; cast pbc to int64 before the Cython kernel.
def ensure_pymatgen_int64_compatibility() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if sys.platform != "win32":
        _PATCHED = True
        return False
    try:
        from pymatgen.optimization import neighbors as neighbors_module
    except ImportError:
        _PATCHED = True
        return False
    original = getattr(neighbors_module, "find_points_in_spheres", None)
    if original is None or getattr(original, "_nfe_int64_wrapper", False):
        _PATCHED = True
        return original is not None

    def find_points_in_spheres_int64(*args: Any, **kwargs: Any) -> Any:
        if "pbc" in kwargs:
            kwargs["pbc"] = np.ascontiguousarray(kwargs["pbc"], dtype=np.int64)
        elif len(args) >= 4:
            args = (*args[:3], np.ascontiguousarray(args[3], dtype=np.int64), *args[4:])
        if "lattice" in kwargs:
            kwargs["lattice"] = np.ascontiguousarray(kwargs["lattice"], dtype=np.float64)
        return original(*args, **kwargs)

    find_points_in_spheres_int64._nfe_int64_wrapper = True  # type: ignore[attr-defined]
    neighbors_module.find_points_in_spheres = find_points_in_spheres_int64
    # Modules that imported the symbol by name keep their own reference.
    for module_name in ("pymatgen.core.lattice", "pymatgen.core.structure", "pymatgen.core.sites"):
        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = __import__(module_name, fromlist=["find_points_in_spheres"])
            except ImportError:
                continue
        if hasattr(module, "find_points_in_spheres"):
            module.find_points_in_spheres = find_points_in_spheres_int64
    _PATCHED = True
    return True


__all__ = ["ensure_pymatgen_int64_compatibility"]
