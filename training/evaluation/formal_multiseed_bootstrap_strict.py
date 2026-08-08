from __future__ import annotations

import sys
from collections import Counter

import pandas as pd

from nfe_model.prediction_manifest import (
    assert_same_prediction_data_identity,
    load_prediction_manifest,
)
from nfe_model.provenance_v2 import assert_matching_provenance
from training.baselines.common import load_benchmark_data
from training.evaluation import formal_multiseed_bootstrap as bootstrap
from training.evaluation.sign_predictions_formal import (
    _assert_exact_split_membership,
    _prediction_metrics,
)


EXPECTED_TRAINING_SEEDS = {2027, 2028, 2029, 2030, 2031}
EXPECTED_BOOTSTRAP_ITERATIONS = 5000
EXPECTED_BOOTSTRAP_RNG_SEED = 2027
FORMAL_TOLERANCE = 5e-6


def _model_key(manifest: dict) -> str:
    identity = manifest.get("run_identity", {})
    track = str(identity.get("track", "")).strip()
    model = str(identity.get("model", "")).strip()
    if not track or not model:
        raise RuntimeError("strict bootstrap manifest requires non-empty track/model identity")
    return f"{track}/{model}"


# Ordered A-minus-B comparisons. Direction is part of the preregistered estimand
# and must never be normalized through an unordered container such as frozenset.
PLANNED_COMPARISONS = {
    ("full-system/ours_full", "architecture/painn"),
    ("architecture/painn", "architecture/cgcnn_controlled"),
    ("architecture/painn", "architecture/schnet_controlled"),
    ("architecture/painn", "architecture/angle_moment"),
    ("architecture/painn", "architecture/state_threebody"),
    ("architecture/painn", "official-upstream/cgcnn_official"),
    ("architecture/painn", "official-upstream/schnet_official"),
    ("architecture/painn", "official-upstream/alignn_official"),
    ("architecture/painn", "official-upstream/m3gnet_official"),
    ("ablation/full", "ablation/no_denoise"),
    ("ablation/no_denoise", "ablation/no_vector"),
    ("ablation/full", "ablation/no_self_supervision"),
    ("ablation/no_self_supervision", "ablation/matched_supervision"),
    ("ablation/full", "ablation/no_global"),
}


def _values(flag: str) -> list[str]:
    if flag not in sys.argv:
        if any(token.startswith(flag + "=") for token in sys.argv):
            raise ValueError(
                f"strict multi-seed bootstrap requires {flag} followed by five file paths; "
                f"the {flag}=... form is intentionally unsupported for a multi-value option"
            )
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


def _single_option(flag: str, default: str) -> str:
    values: list[str] = []
    index = 0
    while index < len(sys.argv):
        token = sys.argv[index]
        if token == flag:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{flag} requires a value")
            values.append(sys.argv[index + 1])
            index += 2
            continue
        if token.startswith(flag + "="):
            values.append(token.split("=", 1)[1])
        index += 1
    if len(values) > 1:
        raise ValueError(f"{flag} may be supplied at most once")
    return values[0] if values else default


def _single_option_int(flag: str, default: int) -> int:
    value = _single_option(flag, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{flag} requires an integer") from exc


def _remove_option_with_value(arguments: list[str], flag: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == flag:
            if index + 1 >= len(arguments):
                raise ValueError(f"{flag} requires a value")
            index += 2
            continue
        if token.startswith(flag + "="):
            index += 1
            continue
        result.append(token)
        index += 1
    return result


def _audit_side(paths: list[str], side: str) -> tuple[set[int], dict, list[dict]]:
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
    if set(seed_values) != EXPECTED_TRAINING_SEEDS:
        raise RuntimeError(
            f"side {side} must use exactly preregistered seeds {sorted(EXPECTED_TRAINING_SEEDS)}; "
            f"observed={sorted(seed_values)}"
        )

    training_protocols = {
        str(identity.get("training_protocol_sha256", "")).strip()
        for identity in identities
    }
    if "" in training_protocols or len(training_protocols) != 1:
        raise RuntimeError(
            f"side {side} must use one non-empty training protocol across seeds: {training_protocols}"
        )
    model_protocols = [
        str(identity.get("model_protocol_sha256", "")).strip()
        for identity in identities
    ]
    nonempty_model_protocols = {value for value in model_protocols if value}
    if nonempty_model_protocols and (
        len(nonempty_model_protocols) != 1
        or any(not value for value in model_protocols)
    ):
        raise RuntimeError(
            f"side {side} mixes model protocol identities across seeds: {model_protocols}"
        )

    checkpoint_hashes = [str(identity.get("checkpoint_sha256", "")) for identity in identities]
    if any(len(value) != 64 for value in checkpoint_hashes):
        raise RuntimeError(
            f"side {side} contains missing/non-SHA256 checkpoint identities; "
            "nested training-seed bootstrap is reserved for independently checkpointed models"
        )
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise RuntimeError(f"side {side} reuses a checkpoint across multiple seed files")
    return set(seed_values), reference, manifests


def _assert_exact_files(paths: list[str], manifests: list[dict], data, side: str) -> None:
    for path, manifest in zip(paths, manifests):
        assert_matching_provenance(
            manifest["data_identity"],
            data.provenance,
            require_present=True,
            require_code_match=True,
        )
        frame = pd.read_csv(path)
        _prediction_metrics(frame)
        _assert_exact_split_membership(frame, data, "test", FORMAL_TOLERANCE)
        manifest_seed = int(manifest["run_identity"]["seed"])
        if manifest_seed not in EXPECTED_TRAINING_SEEDS:
            raise RuntimeError(f"side {side} contains unexpected seed {manifest_seed}")


def main() -> int:
    iterations = _single_option_int("--iterations", EXPECTED_BOOTSTRAP_ITERATIONS)
    rng_seed = _single_option_int("--seed", EXPECTED_BOOTSTRAP_RNG_SEED)
    minimum_seeds = _single_option_int("--minimum-training-seeds", len(EXPECTED_TRAINING_SEEDS))
    config_path = _single_option("--config", "training/configs/nfe_predictor.yaml")
    if iterations != EXPECTED_BOOTSTRAP_ITERATIONS:
        raise ValueError(
            f"paper bootstrap fixes --iterations={EXPECTED_BOOTSTRAP_ITERATIONS}; observed={iterations}"
        )
    if rng_seed != EXPECTED_BOOTSTRAP_RNG_SEED:
        raise ValueError(
            f"paper bootstrap fixes --seed={EXPECTED_BOOTSTRAP_RNG_SEED}; observed={rng_seed}"
        )
    if minimum_seeds != len(EXPECTED_TRAINING_SEEDS):
        raise ValueError(
            f"paper bootstrap fixes --minimum-training-seeds={len(EXPECTED_TRAINING_SEEDS)}"
        )

    a_paths = _values("--a")
    b_paths = _values("--b")
    seeds_a, manifest_a, manifests_a = _audit_side(a_paths, "A")
    seeds_b, manifest_b, manifests_b = _audit_side(b_paths, "B")
    assert_same_prediction_data_identity(manifest_a, manifest_b)
    if seeds_a != seeds_b:
        raise RuntimeError(
            f"A/B nested bootstrap requires the same training seed set: A={sorted(seeds_a)} B={sorted(seeds_b)}"
        )

    data = load_benchmark_data(config_path, rebuild_cache=False)
    _assert_exact_files(a_paths, manifests_a, data, "A")
    _assert_exact_files(b_paths, manifests_b, data, "B")

    comparison = (_model_key(manifest_a), _model_key(manifest_b))
    if comparison[0] == comparison[1]:
        raise ValueError("paper bootstrap requires two different model identities")
    if comparison not in PLANNED_COMPARISONS:
        raise ValueError(
            "requested directed model pair is not preregistered for formal paper inference: "
            f"{comparison[0]} -> {comparison[1]}. Use formal_multiseed_bootstrap.py for exploratory comparisons."
        )

    delegated = _remove_option_with_value(list(sys.argv), "--config")
    original = sys.argv
    try:
        sys.argv = delegated
        return bootstrap.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
