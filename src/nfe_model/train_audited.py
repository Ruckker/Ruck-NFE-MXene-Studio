"""Backward-compatible alias for the formal audited predictor trainer.

Historically this module installed audit monkey patches directly on ``train_core``.
That bypassed newer public-entrypoint checks such as formal configuration
validation. Keep the import path for old launch scripts, but delegate all
execution to :mod:`nfe_model.train`, which owns the complete v2.2 contract.
"""

from __future__ import annotations

from typing import Sequence

from .train import main as _formal_main


def main(argv: Sequence[str] | None = None) -> int:
    return _formal_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
