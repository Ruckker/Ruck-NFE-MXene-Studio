from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from . import summarize as base


def assert_complete_paper_ablation_set(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise RuntimeError("paper ablation matrix contains no audited result rows")
    expected = set(base.ABLATION_ORDER)
    observed = set(str(value) for value in frame["ablation"].tolist())
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise RuntimeError(
            "paper ablation matrix is incomplete or contaminated; "
            f"missing={missing} unexpected={unexpected}. Final ablation tables require the full preregistered set."
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = base.parse_args(argv)
    root = Path(args.runs_root).resolve()
    rows = base.load_runs(root)
    if not rows:
        raise SystemExit(f"no ablation final_metrics.json files found under {root}")
    frame = pd.DataFrame(rows)
    assert_complete_paper_ablation_set(frame)
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
