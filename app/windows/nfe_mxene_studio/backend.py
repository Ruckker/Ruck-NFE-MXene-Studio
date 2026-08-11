from __future__ import annotations

import sys
from pathlib import Path

import torch

from nfe_model.pair_symmetric_graph import install_pair_symmetric_graph_contract

# The desktop package must install the same periodic graph contract as the
# canonical CLI before importing the preserved GUI backend implementation.
install_pair_symmetric_graph_contract()

from nfe_model.data_v2 import torch_load_compat as _torch_load_compat  # noqa: E402
from nfe_model.predict_core import infer_chunk as _infer_chunk  # noqa: E402
from nfe_model.predict_guard import (  # noqa: E402
    guarded_build_periodic_graph as _guarded_build_periodic_graph,
    guarded_load_checkpoint_model as _guarded_load_checkpoint_model,
)
from nfe_model import strict_generation_formal as _strict_generation_formal  # noqa: E402

from . import backend_legacy as _legacy  # noqa: E402


def _load_windows_checkpoint(path: str | Path, device: torch.device):
    # A PyInstaller bundle has no Git worktree to compare at runtime. All other
    # scientific identity checks remain mandatory; source-tree execution keeps
    # the strict runtime Git equality check enabled.
    return _guarded_load_checkpoint_model(
        path,
        device,
        require_runtime_git_match=not bool(getattr(sys, "_MEIPASS", None)),
    )


# Patch only the runtime-contract dependencies. NFEEngine and all GUI-facing
# behavior remain byte-for-byte preserved in backend_legacy.py.
_legacy.build_periodic_graph = _guarded_build_periodic_graph
_legacy.torch_load_compat = _torch_load_compat
_legacy.infer_chunk = _infer_chunk
_legacy.load_checkpoint_model = _load_windows_checkpoint
_legacy.strict_generation = _strict_generation_formal
_legacy.manifold_generation.strict_generation = _strict_generation_formal

# Re-export the preserved public backend API so app.py and packaged
# `windows_app.backend` imports do not change.
for _name in dir(_legacy):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_legacy, _name)


def __getattr__(name: str):
    return getattr(_legacy, name)
