from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from nfe_model import data_v2, metrics_v2, provenance_v2

try:
    from . import common as _common
except ImportError:
    import common as _common

_common.load_or_build_cache = data_v2.load_or_build_cache
_common.classification_metrics = metrics_v2.classification_metrics
_common.regression_metrics = metrics_v2.regression_metrics
_common.selection_score = metrics_v2.selection_score
_common.build_provenance = provenance_v2.build_provenance

load_benchmark_data = _common.load_benchmark_data
manifest_frame = _common.manifest_frame


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the fixed v2 leakage-safe benchmark manifest.")
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument("--output", default="training/baselines/benchmark_split_manifest.csv")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_benchmark_data(args.config, rebuild_cache=args.rebuild_cache)
    frame = manifest_frame(data)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {len(frame)} rows to {output}")
    print(frame["Suggested_Split"].value_counts().to_dict())
    print(data.provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
