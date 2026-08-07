from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
import yaml

from nfe_model.provenance_v2 import git_repository_state
from training import formal_v2_4


PAPER_CONFIG = "training/configs/nfe_predictor_v2_4_paper_ready.yaml"
EXPECTED_SEEDS = (2027, 2028, 2029, 2030, 2031)
EXPECTED_BASELINE_TRACKS = {
    ("architecture", "dummy"),
    ("architecture", "xgboost"),
    ("architecture", "cgcnn_controlled"),
    ("architecture", "schnet_controlled"),
    ("architecture", "angle_moment"),
    ("architecture", "state_threebody"),
    ("architecture", "painn"),
    ("official-upstream", "cgcnn_official"),
    ("official-upstream", "schnet_official"),
    ("official-upstream", "alignn_official"),
    ("official-upstream", "m3gnet_official"),
    ("full-system", "ours_full"),
}
EXPECTED_ABLATIONS = {
    "full",
    "no_vector",
    "no_global",
    "no_masked_pretrain",
    "no_denoise",
    "no_self_supervision",
    "no_auxiliary_regression",
    "matched_supervision",
    "classification_only",
}
ALIASES = {
    **formal_v2_4.ALIASES,
    "baseline-summary": "training.baselines.summarize",
    "ablation-summary": "training.ablations.summarize",
    "generator-contract-audit": "training.evaluation.audit_generator_predictor_contract",
}
TRAINING_ALIASES = {"train", "ablation", "baseline", "official"}
CONFIG_ALIASES = {
    "train",
    "ablation",
    "baseline",
    "official",
    "cache-rebuild-audit",
    "cache-sanity-audit",
    "split-duplicate-audit",
    "neighbor-symmetry-audit",
    "verified-queue",
    "generator-contract-audit",
}

# Explicit scientific protocol registry. A clean commit alone is not enough:
# changing the YAML requires a deliberate code-level contract revision too.
EXPECTED_PAPER_VALUES: dict[tuple[str, ...], object] = {
    ("seed",): EXPECTED_SEEDS[0],
    ("data", "cache"): "../../cache/nfe_graphs_v2_4.pt",
    ("data", "radius"): 6.0,
    ("data", "max_neighbors"): 36,
    ("data", "max_cache_skip_fraction"): 0.0,
    ("model", "hidden_dim"): 192,
    ("model", "vector_dim"): 64,
    ("model", "num_layers"): 6,
    ("model", "num_rbf"): 48,
    ("model", "cutoff"): 6.0,
    ("model", "dropout"): 0.12,
    ("model", "max_atomic_number"): 118,
    ("model", "element_features"): 14,
    ("model", "global_features"): 11,
    ("training", "epochs"): 220,
    ("training", "pretrain_epochs"): 35,
    ("training", "batch_size_per_gpu"): 96,
    ("training", "grad_accum_steps"): 1,
    ("training", "learning_rate"): 3e-4,
    ("training", "min_learning_rate"): 5e-6,
    ("training", "weight_decay"): 1e-5,
    ("training", "warmup_epochs"): 8,
    ("training", "grad_clip"): 5.0,
    ("training", "amp"): True,
    ("training", "compile"): False,
    ("training", "early_stopping_patience"): 35,
    ("training", "checkpoint_dir"): "nfe_predictor_v2_4_paper_ready",
    ("loss", "class_weight"): 1.0,
    ("loss", "score_weight"): 1.5,
    ("loss", "auxiliary_weight"): 0.45,
    ("loss", "masked_atom_weight"): 0.35,
    ("loss", "denoise_weight"): 0.65,
    ("loss", "label_smoothing"): 0.04,
    ("loss", "mask_probability"): 0.15,
    ("loss", "coordinate_noise_min_A"): 0.01,
    ("loss", "coordinate_noise_max_A"): 0.15,
    ("inference", "mc_samples"): 30,
    ("inference", "confidence_level"): 0.90,
    ("inference", "embedding_bank_size"): 4096,
    ("generator_model", "hidden_dim"): 192,
    ("generator_model", "vector_dim"): 64,
    ("generator_model", "num_layers"): 6,
    ("generator_model", "num_rbf"): 64,
    ("generator_model", "cutoff"): 12.0,
    ("generator_model", "max_neighbors"): 24,
    ("generator_model", "dropout"): 0.10,
    ("generator_model", "max_atomic_number"): 118,
    ("generator_model", "element_features"): 14,
    ("generator_model", "condition_dim"): 128,
    ("generator_training", "epochs"): 320,
    ("generator_training", "batch_size_per_gpu"): 64,
    ("generator_training", "grad_accum_steps"): 1,
    ("generator_training", "learning_rate"): 2e-4,
    ("generator_training", "min_learning_rate"): 3e-6,
    ("generator_training", "weight_decay"): 1e-5,
    ("generator_training", "warmup_epochs"): 10,
    ("generator_training", "grad_clip"): 5.0,
    ("generator_training", "amp"): True,
    ("generator_training", "early_stopping_patience"): 45,
    ("generator_loss", "coordinate_weight"): 1.0,
    ("generator_loss", "lattice_weight"): 0.70,
    ("generator_loss", "repulsion_weight"): 0.12,
    ("generator_loss", "condition_dropout"): 0.15,
    ("generation", "sampling_steps"): 100,
    ("generation", "guidance_scale"): 2.5,
    ("generation", "oversample_factor"): 8,
    ("generation", "minimum_distance_factor"): 0.58,
    ("generation", "repair_steps"): 80,
    ("generation", "minimum_vacuum_A"): 15.0,
    ("generation", "maximum_slab_thickness_A"): 12.0,
    ("generation", "maximum_nearest_radius_ratio"): 1.85,
    ("generation", "minimum_atomic_layers"): 3,
}

# These options change the pre-registered optimization/capacity protocol. They
# are deliberately unavailable through the paper-ready dispatcher. Short smoke
# runs and exploratory overrides must use training.formal_v2_4 instead and must
# not be admitted to paper tables.
IMMUTABLE_TRAINING_OPTIONS = {
    "--config",
    "--epochs",
    "--batch-size",
    "--batch-size-per-gpu",
    "--learning-rate",
    "--min-learning-rate",
    "--warmup-epochs",
    "--weight-decay",
    "--patience",
    "--hidden-dim",
    "--layers",
    "--dropout",
    "--label-smoothing",
    "--no-amp",
    "--amp",
    "--device",
    "--seeds",
    "--rebuild-cache",
    "--allow-unverified-checkpoint",
}
IMMUTABLE_SUMMARY_OPTIONS = {
    "--minimum-full-seeds",
    "--minimum-model-seeds",
    "--minimum-seeds",
}


def _pytorch_world_size() -> int:
    raw = os.environ.get("WORLD_SIZE", "1").strip() or "1"
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid PyTorch WORLD_SIZE={raw!r}") from exc
    if value <= 0:
        raise ValueError(f"PyTorch WORLD_SIZE must be positive, got {value}")
    return value


def _assert_clean_git() -> None:
    state = git_repository_state()
    commit = str(state.get("git_commit", "unknown"))
    if state.get("git_dirty") is not False:
        raise RuntimeError("paper commands require a clean Git worktree before execution")
    if commit == "unknown" or len(commit) != 40:
        raise RuntimeError("paper commands require a resolvable 40-character Git commit")


def _assert_single_process_training(alias: str) -> None:
    if alias not in TRAINING_ALIASES:
        return
    world_size = _pytorch_world_size()
    if world_size != 1:
        raise RuntimeError(
            "paper optimization protocol fixes one Python training process per independent run; "
            f"observed WORLD_SIZE={world_size}. Run independent seeds/models concurrently across GPUs instead."
        )
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    if rank not in (None, "", "0") or local_rank not in (None, "", "0"):
        raise RuntimeError(
            "paper training must not run as a nonzero torchrun rank; launch one normal Python process per GPU"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "paper-ready training requires CUDA so AMP and the registered GPU optimization protocol are effective; "
            "use training.formal_v2_4 for CPU smoke tests"
        )


def _option_name(token: str) -> str | None:
    if not token.startswith("--"):
        return None
    return token.split("=", 1)[0]


def _option_value(arguments: list[str], name: str) -> str | None:
    found: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == name:
            if index + 1 >= len(arguments):
                raise ValueError(f"{name} requires a value")
            found.append(arguments[index + 1])
            index += 2
            continue
        if token.startswith(name + "="):
            found.append(token.split("=", 1)[1])
        index += 1
    if len(found) > 1:
        raise ValueError(f"{name} may be supplied at most once for paper-ready commands")
    return found[0] if found else None


def _reject_options(arguments: Iterable[str], forbidden: set[str], *, context: str) -> None:
    observed = sorted(
        {
            name
            for token in arguments
            if (name := _option_name(token)) is not None and name in forbidden
        }
    )
    if observed:
        raise ValueError(
            f"{context} is immutable for paper-ready runs; forbidden overrides={observed}. "
            "Use python -m training.formal_v2_4 for smoke/exploratory runs instead."
        )


def _nested_value(config: dict, path: tuple[str, ...]) -> object:
    value: object = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"paper-ready config is missing {'.'.join(path)}")
        value = value[key]
    return value


def _same_protocol_value(observed: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, float):
        try:
            return math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return observed == expected


def _validate_paper_config(config: dict) -> None:
    mismatches: list[str] = []
    for path, expected in EXPECTED_PAPER_VALUES.items():
        observed = _nested_value(config, path)
        if not _same_protocol_value(observed, expected):
            mismatches.append(
                f"{'.'.join(path)}={observed!r} (expected {expected!r})"
            )
    if mismatches:
        raise RuntimeError(
            "paper-ready YAML has drifted from the pre-registered protocol; "
            "revise the explicit protocol contract rather than silently changing the YAML: "
            + "; ".join(mismatches[:12])
        )


def _load_paper_config() -> dict:
    path = Path(PAPER_CONFIG)
    if not path.is_file():
        raise FileNotFoundError(f"paper-ready configuration not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"invalid paper-ready configuration: {path}")
    _validate_paper_config(config)
    return config


def _validate_ablation_seed(arguments: list[str]) -> None:
    value = _option_value(arguments, "--seed")
    if value is None:
        raise ValueError(
            "paper-ready ablation runs require an explicit --seed from the registered set "
            f"{EXPECTED_SEEDS}"
        )
    try:
        seed = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid paper ablation seed: {value!r}") from exc
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"paper ablation seed {seed} is outside registered set {EXPECTED_SEEDS}")


def _baseline_budget_args(alias: str, config: dict) -> list[str]:
    if alias not in {"baseline", "official"}:
        return []
    training = config["training"]
    model = config["model"]
    loss = config["loss"]
    values: list[tuple[str, object]] = [
        ("--epochs", training["epochs"]),
        ("--batch-size", training["batch_size_per_gpu"]),
        ("--learning-rate", training["learning_rate"]),
        ("--min-learning-rate", training["min_learning_rate"]),
        ("--warmup-epochs", training["warmup_epochs"]),
        ("--weight-decay", training["weight_decay"]),
        ("--patience", training["early_stopping_patience"]),
        ("--hidden-dim", model["hidden_dim"]),
        ("--layers", model["num_layers"]),
        ("--label-smoothing", loss["label_smoothing"]),
        ("--device", "cuda"),
        ("--seeds", ",".join(str(seed) for seed in EXPECTED_SEEDS)),
    ]
    if alias == "baseline":
        values.append(("--dropout", model["dropout"]))
    arguments: list[str] = []
    for name, value in values:
        arguments.extend([name, str(value)])
    if bool(training.get("amp", True)) is not True:
        raise RuntimeError("paper-ready architecture/official protocol requires training.amp=true")
    return arguments


def _baseline_summary_root(arguments: list[str]) -> Path:
    value = _option_value(arguments, "--results-root")
    return Path(value or "training/baselines/results").resolve()


def _ablation_summary_root(arguments: list[str]) -> Path:
    value = _option_value(arguments, "--runs-root")
    return Path(value or "runs/ablations").resolve()


def _assert_complete_baseline_results(arguments: list[str]) -> None:
    root = _baseline_summary_root(arguments)
    rows: list[tuple[str, str, int]] = []
    for path in sorted(root.glob("*/*/seed_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "nfe-baseline-result-2.2":
            continue
        try:
            seed = int(payload.get("seed"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"paper baseline result has invalid seed: {path}") from exc
        rows.append((str(payload.get("track", "")), str(payload.get("model", "")), seed))
    if not rows:
        raise RuntimeError(f"no paper baseline result rows found under {root}")
    observed_pairs = {(track, model) for track, model, _ in rows}
    missing = sorted(EXPECTED_BASELINE_TRACKS - observed_pairs)
    extra = sorted(observed_pairs - EXPECTED_BASELINE_TRACKS)
    if missing or extra:
        raise RuntimeError(
            "paper baseline roster mismatch: "
            f"missing={missing or 'none'} extra={extra or 'none'}"
        )
    for track, model in sorted(EXPECTED_BASELINE_TRACKS):
        seeds = tuple(sorted(seed for t, m, seed in rows if t == track and m == model))
        wanted = (EXPECTED_SEEDS[0],) if model == "dummy" else EXPECTED_SEEDS
        if seeds != wanted:
            raise RuntimeError(
                f"paper seed set mismatch for {track}/{model}: observed={seeds} expected={wanted}"
            )


def _assert_complete_ablation_results(arguments: list[str]) -> None:
    root = _ablation_summary_root(arguments)
    rows: list[tuple[str, int]] = []
    for path in sorted(root.glob("*/seed_*/final_metrics.json")):
        ablation = path.parents[1].name
        try:
            seed = int(path.parent.name.removeprefix("seed_"))
        except ValueError as exc:
            raise RuntimeError(f"paper ablation result has invalid seed directory: {path}") from exc
        rows.append((ablation, seed))
    if not rows:
        raise RuntimeError(f"no paper ablation result rows found under {root}")
    observed = {name for name, _ in rows}
    missing = sorted(EXPECTED_ABLATIONS - observed)
    extra = sorted(observed - EXPECTED_ABLATIONS)
    if missing or extra:
        raise RuntimeError(
            f"paper ablation roster mismatch: missing={missing or 'none'} extra={extra or 'none'}"
        )
    for ablation in sorted(EXPECTED_ABLATIONS):
        seeds = tuple(sorted(seed for name, seed in rows if name == ablation))
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"paper seed set mismatch for ablation {ablation}: observed={seeds} expected={EXPECTED_SEEDS}"
            )


def _usage() -> str:
    rows = "\n".join(f"  {alias:26s} -> {module}" for alias, module in ALIASES.items())
    return (
        "Usage: python -m training.paper <alias> [arguments...]\n\n"
        "This is the only paper-ready dispatcher. Arbitrary module passthrough is disabled.\n"
        f"Immutable config: {PAPER_CONFIG}\n"
        f"Registered seeds: {EXPECTED_SEEDS}\n"
        "Training requires CUDA and one Python process / one GPU; parallelize independent seeds/models across GPUs.\n\n"
        f"{rows}\n"
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0

    alias = sys.argv[1]
    if alias not in ALIASES:
        raise ValueError(
            f"{alias!r} is not a paper-ready alias. Arbitrary module passthrough is disabled.\n{_usage()}"
        )
    arguments = list(sys.argv[2:])

    _assert_clean_git()
    _assert_single_process_training(alias)
    if alias in TRAINING_ALIASES:
        _reject_options(arguments, IMMUTABLE_TRAINING_OPTIONS, context="paper training budget")
    elif any(_option_name(token) == "--config" for token in arguments):
        raise ValueError(
            f"paper-ready data identity is fixed to {PAPER_CONFIG}; --config overrides are forbidden"
        )
    if alias in {"baseline-summary", "ablation-summary"}:
        _reject_options(arguments, IMMUTABLE_SUMMARY_OPTIONS, context="paper seed-count gate")
    if alias == "ablation":
        _validate_ablation_seed(arguments)
    if alias == "baseline-summary":
        _assert_complete_baseline_results(arguments)
    if alias == "ablation-summary":
        _assert_complete_ablation_results(arguments)

    config = _load_paper_config()
    module = ALIASES[alias]
    fixed_arguments = _baseline_budget_args(alias, config)
    if alias in CONFIG_ALIASES:
        fixed_arguments = ["--config", PAPER_CONFIG, *fixed_arguments]

    # formal_v2_4 installs the pair-symmetric data/graph contract before the
    # target module is imported. We pass the explicit audited module here so the
    # lower-level development default config cannot replace PAPER_CONFIG.
    sys.argv = [sys.argv[0], module, *fixed_arguments, *arguments]
    formal_v2_4.DEFAULT_CONFIG = PAPER_CONFIG
    return formal_v2_4.main()


if __name__ == "__main__":
    raise SystemExit(main())
