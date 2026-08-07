from __future__ import annotations

import sys
from collections import Counter

from nfe_model.prediction_manifest import (
    assert_same_prediction_data_identity,
    load_prediction_manifest,
)
from training.evaluation import formal_multiseed_bootstrap as bootstrap


def _values(flag: str) -> list[str]:
    if flag not in sys.argv:
        raise ValueError(f"strict multi-seed bootstrap requires {flag}")
    start = sys.argv.index(flag) + 1
    values = []
    for value in sys.argv[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    if not values:
        raise ValueError(f"{flag} requires one or more prediction files")
    return values


def _audit_side(paths: list[str], side: str) -> tuple[set[int], dict]:
    manifests = [load_prediction_manifest(path, expected_split="test") for path in paths]
    reference = manifests[0]
    for manifest in manifests[1:]:
        assert_same_prediction_data_identity(reference, manifest)
    identities = [manifest["run_identity"] for manifest in manifests]
    tracks = {str(identity.get("track", "")) for identity in identities}
    models = {str(identity.get("model", "")) for identity in identities}
    if len(tracks) != 1 or len(models) != 1 or "" in tracks or "" in models:
        raise RuntimeError(
            f"side {side} must contain one fixed non-empty track/model; tracks={tracks} models={models}"
        )
    seeds = [identity.get("seed") for identity in identities]
    if any(seed is None for seed in seeds):
        raise RuntimeError(f"side {side} manifests must record training seeds")
    seed_values = [int(seed) for seed in seeds]
    duplicates = [seed for seed, count in Counter(seed_values).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"side {side} repeats training seeds: {duplicates}")
    checkpoint_hashes = [str(identity.get("checkpoint_sha256", "")) for identity in identities]
    # Neural/model comparisons should be independent checkpoints. Deterministic
    # no-checkpoint baselines are not appropriate for the nested training-seed bootstrap.
    if any(not value for value in checkpoint_hashes):
        raise RuntimeError(
            f"side {side} contains prediction files without checkpoint hashes; "
            "use nested seed bootstrap only for independently trained checkpointed models"
        )
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise RuntimeError(f"side {side} reuses a checkpoint across multiple seed files")
    return set(seed_values), reference


def main() -> int:
    a_paths = _values("--a")
    b_paths = _values("--b")
    seeds_a, manifest_a = _audit_side(a_paths, "A")
    seeds_b, manifest_b = _audit_side(b_paths, "B")
    assert_same_prediction_data_identity(manifest_a, manifest_b)
    if seeds_a != seeds_b:
        raise RuntimeError(
            f"A/B nested bootstrap requires the same training seed set: A={sorted(seeds_a)} B={sorted(seeds_b)}"
        )
    return bootstrap.main()


if __name__ == "__main__":
    raise SystemExit(main())
