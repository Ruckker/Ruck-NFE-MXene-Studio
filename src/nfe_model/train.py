"""Audited public training API.

The historical implementation lives in :mod:`nfe_model.train_core`. Public data,
metric, and CLI training entrypoints expose the audited formal semantics. Internal
ablation machinery imports ``train_core`` explicitly and installs the same audit
patches before execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .train_core import *  # noqa: F401,F403
from . import data_v2 as _data_v2
from . import metrics_v2 as _metrics_v2
from . import train_core as _core
from .formal_config import validate_formal_config
from .provenance_v2 import assert_matching_experiment_protocol
from .train_audit_v2 import AuditedNFEDataset, audited_collate_graphs, install_audit_patches

NFEDataset = AuditedNFEDataset
collate_graphs = audited_collate_graphs
load_or_build_cache = _data_v2.load_or_build_cache
REGRESSION_TARGETS = _data_v2.REGRESSION_TARGETS
INDEX_TO_LABEL = _data_v2.INDEX_TO_LABEL
classification_metrics = _metrics_v2.classification_metrics
regression_metrics = _metrics_v2.regression_metrics
selection_score = _metrics_v2.selection_score


def _validated_cli_config(argv: Sequence[str] | None):
    args = _core.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = _core.resolve_config_paths(_core.load_config(config_path), config_path)
    validate_formal_config(config)
    return args, config


def _validate_resume_protocol(args, config) -> None:
    if not args.resume:
        return
    resume_path = Path(args.resume).resolve()
    expected_path = Path(config["training"]["checkpoint_dir"]).resolve() / "best.pt"
    if resume_path != expected_path:
        raise ValueError(
            "formal resume must continue the same run directory so checkpoint/history/provenance lineage "
            f"cannot be silently forked: resume={resume_path} expected={expected_path}"
        )
    checkpoint = _data_v2.torch_load_compat(resume_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"resume checkpoint is not a mapping: {resume_path}")
    assert_matching_experiment_protocol(checkpoint, config)


def main(argv: Sequence[str] | None = None) -> int:
    install_audit_patches(_core)
    args, config = _validated_cli_config(argv)
    _validate_resume_protocol(args, config)
    return _core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
