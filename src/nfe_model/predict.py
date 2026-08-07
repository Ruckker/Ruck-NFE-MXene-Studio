"""Audited public prediction API.

The historical implementation lives in :mod:`nfe_model.predict_core`.  Public
checkpoint loading and CLI inference always enforce the v2.1 data/graph/code
contract; internal legacy behavior is available only through the explicitly
named ``predict_core`` module.
"""

from __future__ import annotations

from .predict_core import *  # noqa: F401,F403
from .predict_guard import guarded_load_checkpoint_model, main

# Override the star-imported legacy loader in the public API.
load_checkpoint_model = guarded_load_checkpoint_model


if __name__ == "__main__":
    raise SystemExit(main())
