from __future__ import annotations

from typing import Sequence

from . import train as _train
from .train_audit import install_audit_patches


def main(argv: Sequence[str] | None = None) -> int:
    install_audit_patches(_train)
    return _train.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
