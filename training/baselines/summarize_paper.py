from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from . import summarize as base


EXPECTED_PAPER_MODELS = {
    "architecture": {
        "dummy",
        "xgboost",
        "cgcnn_controlled",
        "schnet_controlled",
        "angle_moment",
        "state_threebody",
        "painn",
    },
    "official-upstream": {
        "cgcnn_official",
        "schnet_official",
        "alignn_official",
        "m3gnet_official",
    },
    "full-system": {"ours_full"},
}


def assert_complete_paper_model_set(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise RuntimeError("paper benchmark contains no audited result rows")
    observed_tracks = set(str(value) for value in frame["track"].tolist())
    expected_tracks = set(EXPECTED_PAPER_MODELS)
    if observed_tracks != expected_tracks:
        raise RuntimeError(
            "paper benchmark track set is incomplete or contaminated: "
            f"observed={sorted(observed_tracks)} expected={sorted(expected_tracks)}"
        )
    for track, expected_models in EXPECTED_PAPER_MODELS.items():
        observed_models = set(
            str(value) for value in frame.loc[frame["track"] == track, "model"].tolist()
        )
        if observed_models != expected_models:
            missing = sorted(expected_models - observed_models)
            unexpected = sorted(observed_models - expected_models)
            raise RuntimeError(
                f"paper benchmark {track} model set mismatch; missing={missing} unexpected={unexpected}. "
                "A final table must not silently omit a preregistered baseline."
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = base.parse_args(argv)
    root = Path(args.results_root).resolve()
    rows = base.load_results(root)
    if not rows:
        raise SystemExit(
            f"no audited {base.BASELINE_RESULT_SCHEMA} files found under {root}"
        )
    frame = pd.DataFrame(rows)
    assert_complete_paper_model_set(frame)
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
