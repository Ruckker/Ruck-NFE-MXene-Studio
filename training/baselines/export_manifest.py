from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    from .common import load_benchmark_data, manifest_frame
except ImportError:
    from common import load_benchmark_data, manifest_frame


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the fixed leakage-safe NFE benchmark split manifest."
    )
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument(
        "--output", default="training/baselines/benchmark_split_manifest.csv"
    )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
