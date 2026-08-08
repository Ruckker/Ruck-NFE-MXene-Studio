from __future__ import annotations

from pathlib import Path
from typing import Sequence

from . import train_ablation
from .utils import load_config


def _canonical_checkpoint_dir(args) -> Path:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    seed = int(args.seed if args.seed is not None else config["seed"])
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / "runs" / "ablations" / args.ablation / f"seed_{seed}").resolve()


def _assert_canonical_checkpoint_dir(args) -> None:
    if not args.checkpoint_dir:
        return
    observed = Path(args.checkpoint_dir).resolve()
    expected = _canonical_checkpoint_dir(args)
    if observed != expected:
        raise ValueError(
            "formal ablation output directory is fixed by ablation name and seed so independent "
            f"runs cannot overwrite/mix lineage: checkpoint_dir={observed} expected={expected}. "
            "Use nfe_model.train_ablation directly only for explicitly exploratory custom paths."
        )


def _selected_run_directory(args) -> Path:
    if getattr(args, "checkpoint_dir", None):
        return Path(args.checkpoint_dir).resolve()
    return _canonical_checkpoint_dir(args)


def _assert_same_run_resume(args) -> None:
    if not args.resume:
        return
    run_directory = _selected_run_directory(args)
    observed = Path(args.resume).resolve()
    if observed.parent != run_directory:
        raise ValueError(
            "formal ablation resume must continue from the same run directory so checkpoint/history/provenance "
            f"lineage cannot be silently forked: resume={observed} run_directory={run_directory}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = train_ablation.parse_args(argv)
    _assert_canonical_checkpoint_dir(args)
    _assert_same_run_resume(args)
    return train_ablation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
