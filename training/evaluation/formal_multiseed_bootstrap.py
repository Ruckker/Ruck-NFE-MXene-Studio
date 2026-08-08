from __future__ import annotations

from typing import Any

import numpy as np

from . import formal_multiseed_bootstrap_core as _core


for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


def _observed_seed_mean_delta(
    pairs: list[tuple[int, Any, str, str]],
    metric: str,
) -> tuple[float, list[dict[str, object]]]:
    """Expose stable decimal seed deltas for the preregistered point estimate.

    The underlying MAE calculation is unchanged. Rounding only removes binary
    floating representation noise (for example 0.09999999999999998) from the
    serialized/tested seed-level deltas before their mean is taken.
    """
    _, rows = _core._observed_seed_mean_delta(pairs, metric)
    for row in rows:
        row["delta_a_minus_b"] = float(
            np.round(float(row["delta_a_minus_b"]), 12)
        )
    deltas = np.asarray(
        [float(row["delta_a_minus_b"]) for row in rows], dtype=float
    )
    return float(np.mean(deltas)), rows


def main() -> int:
    return _core.main()


def __getattr__(name: str):
    return getattr(_core, name)


if __name__ == "__main__":
    raise SystemExit(main())
