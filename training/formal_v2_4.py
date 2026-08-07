from __future__ import annotations

import runpy
import sys
from pathlib import Path

from nfe_model.pair_symmetric_graph import install_pair_symmetric_graph_contract


DEFAULT_CONFIG = "training/configs/nfe_predictor_v2_4.yaml"
ALIASES = {
    "train": "nfe_model.train",
    "ablation": "nfe_model.train_ablation_safe",
    "predict": "nfe_model.predict_formal",
    "baseline": "training.baselines.run",
    "official": "training.baselines.official.run",
    "baseline-summary": "training.baselines.summarize",
    "ablation-summary": "training.ablations.summarize",
    "cache-rebuild-audit": "training.evaluation.audit_cache_rebuild_integrity",
    "cache-sanity-audit": "training.evaluation.audit_cache_tensor_sanity",
    "split-duplicate-audit": "training.evaluation.audit_split_duplicates",
    "neighbor-symmetry-audit": "training.evaluation.audit_neighbor_pair_symmetry",
    "verified-queue": "training.evaluation.build_verified_review_queue_paper",
    "blind-verified": "training.evaluation.blind_verified_review_queue",
    "freeze-verified": "training.evaluation.freeze_verified_review_paper",
    "sign-predictions": "training.evaluation.sign_predictions_formal",
    "verified-evaluate": "training.evaluation.evaluate_verified_paper",
    "ood-manifest": "training.evaluation.build_ood_manifest",
    "ood-evaluate": "training.evaluation.formal_evaluate_slices",
    "paired-bootstrap": "training.evaluation.formal_multiseed_bootstrap_strict",
    "representation-audit": "training.evaluation.supercell_consistency",
    "checkpoint-audit": "training.evaluation.audit_checkpoint_integrity",
    "paper-preflight": "training.evaluation.paper_preflight_strict",
}
CONFIG_MODULES = {
    "nfe_model.train",
    "nfe_model.train_ablation_safe",
    "training.baselines.run",
    "training.baselines.official.run",
    "training.evaluation.audit_cache_rebuild_integrity",
    "training.evaluation.audit_cache_tensor_sanity",
    "training.evaluation.audit_split_duplicates",
    "training.evaluation.audit_neighbor_pair_symmetry",
    "training.evaluation.build_verified_review_queue_paper",
    "training.evaluation.sign_predictions_formal",
    "training.evaluation.build_ood_manifest",
}


def _usage() -> str:
    aliases = "\n".join(f"  {name:24s} -> {module}" for name, module in ALIASES.items())
    return (
        "Usage: python -m training.formal_v2_4 <alias-or-module> [arguments...]\n\n"
        "Canonical aliases:\n"
        f"{aliases}\n\n"
        "Any explicit Python module may also be supplied for development. Modules that consume the formal dataset "
        f"receive --config {DEFAULT_CONFIG} automatically unless --config is already present."
    )


def _inject_default_config(module: str, arguments: list[str]) -> list[str]:
    if module not in CONFIG_MODULES or "--config" in arguments:
        return arguments
    config = Path(DEFAULT_CONFIG)
    if not config.is_file():
        raise FileNotFoundError(
            f"formal v2.4 default configuration not found from current working directory: {config.resolve()}"
        )
    return ["--config", DEFAULT_CONFIG, *arguments]


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0
    requested = sys.argv[1]
    module = ALIASES.get(requested, requested)
    if not module or module.startswith("-"):
        raise ValueError(f"invalid formal module/alias: {requested!r}")

    # This must happen before importing the target module. Modules such as
    # train_audit_v2, predict_guard and benchmark summarizers capture data_v2
    # constants at import time.
    install_pair_symmetric_graph_contract()
    arguments = _inject_default_config(module, sys.argv[2:])
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
