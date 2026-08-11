#!/usr/bin/env python3
from __future__ import annotations

try:
    from .build_nfe_dataset_audit import *  # noqa: F401,F403
    from .build_nfe_dataset_audit import main
except ImportError:  # direct execution as `python data_tools/build_nfe_dataset.py`
    from build_nfe_dataset_audit import *  # type: ignore # noqa: F401,F403
    from build_nfe_dataset_audit import main  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())
