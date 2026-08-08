from __future__ import annotations

import copy

import pytest
import torch

from nfe_model.ablation import AblationPeriodicNFEModel
from nfe_model.data_contract import data_implementation_sha256
from nfe_model.formal_config import validate_formal_config
from nfe_model.model import GaussianRBF, PeriodicNFEModel


def _config() -> dict:
    return {
        "seed": 2027,
        "data": {
            "table": "data.csv",
            "root": ".",
            "cache": "cache.pt",
            "radius": 6.0,
            "max_neighbors": 36,
            "max_cache_skip_fraction": 0.01,
        },
        "model": {
            "hidden_dim": 192,
            "vector_dim": 64,
            "num_layers": 6,
            "num_rbf": 48,
            "cutoff": 6.0,
            "dropout": 0.12,
            "max_atomic_number": 118,
            "element_features": 14,
            "global_features": 11,
        },
        "training": {
            "epochs": 220,
            "pretrain_epochs": 35,
            "batch_size_per_gpu": 96,
            "grad_accum_steps": 1,
            "learning_rate": 3e-4,
            "min_learning_rate": 5e-6,
            "weight_decay": 1e-5,
            "warmup_epochs": 8,
            "grad_clip": 5.0,
            "early_stopping_patience": 35,
        },
        "loss": {
            "class_weight": 1.0,
            "score_weight": 1.5,
            "auxiliary_weight": 0.45,
            "masked_atom_weight": 0.35,
            "denoise_weight": 0.65,
            "label_smoothing": 0.04,
            "mask_probability": 0.15,
            "coordinate_noise_min_A": 0.01,
            "coordinate_noise_max_A": 0.15,
        },
        "inference": {"mc_samples": 30, "confidence_level": 0.90},
    }


def test_formal_config_accepts_reference_contract() -> None:
    validate_formal_config(_config())


def test_formal_config_rejects_graph_model_cutoff_mismatch() -> None:
    config = copy.deepcopy(_config())
    config["model"]["cutoff"] = 5.5
    with pytest.raises(ValueError, match="data.radius == model.cutoff"):
        validate_formal_config(config)


def test_formal_config_rejects_unsafe_gradient_accumulation() -> None:
    config = copy.deepcopy(_config())
    config["training"]["grad_accum_steps"] = 2
    with pytest.raises(ValueError, match="grad_accum_steps == 1"):
        validate_formal_config(config)


def _batch(*, with_far_edges: bool, atomic_number: int = 6) -> dict[str, torch.Tensor]:
    if with_far_edges:
        edge_index = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
        edge_shift = torch.zeros((2, 3), dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_shift = torch.empty((0, 3), dtype=torch.float32)
    return {
        "z": torch.tensor([atomic_number, atomic_number], dtype=torch.long),
        "atom_features": torch.zeros((2, 14), dtype=torch.float32),
        "frac_pos": torch.tensor([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]], dtype=torch.float32),
        "lattice": torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]),
        "batch": torch.tensor([0, 0], dtype=torch.long),
        "edge_index": edge_index,
        "edge_shift": edge_shift,
        "global_features": torch.zeros((1, 11), dtype=torch.float32),
    }


def _model(model_cls=PeriodicNFEModel, **kwargs):
    torch.manual_seed(11)
    return model_cls(
        hidden_dim=24,
        vector_dim=8,
        num_layers=2,
        num_rbf=12,
        cutoff=1.0,
        dropout=0.0,
        max_atomic_number=118,
        element_features=14,
        global_features=11,
        num_regression_targets=10,
        **kwargs,
    ).eval()


def test_cutoff_envelope_is_exactly_zero_at_and_beyond_cutoff() -> None:
    rbf = GaussianRBF(8, 1.0)
    envelope = rbf.cutoff_envelope(torch.tensor([0.5, 1.0, 1.2]))
    assert envelope[0] > 0
    assert envelope[1].item() == 0.0
    assert envelope[2].item() == 0.0


def test_full_model_ignores_edges_beyond_model_cutoff() -> None:
    model = _model()
    with torch.no_grad():
        with_far = model(_batch(with_far_edges=True))
        without = model(_batch(with_far_edges=False))
    assert torch.allclose(with_far["class_logits"], without["class_logits"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        with_far["regression_mean"], without["regression_mean"], atol=1e-6, rtol=1e-6
    )


def test_scalar_ablation_ignores_edges_beyond_model_cutoff() -> None:
    model = _model(AblationPeriodicNFEModel, use_vector_features=False)
    with torch.no_grad():
        with_far = model(_batch(with_far_edges=True))
        without = model(_batch(with_far_edges=False))
    assert torch.allclose(with_far["class_logits"], without["class_logits"], atol=1e-6, rtol=1e-6)


def test_model_rejects_atomic_numbers_outside_vocabulary() -> None:
    model = _model()
    with pytest.raises(ValueError, match="atomic number outside model vocabulary"):
        model(_batch(with_far_edges=False, atomic_number=119))


def test_data_implementation_fingerprint_is_stable_sha256() -> None:
    first = data_implementation_sha256()
    second = data_implementation_sha256()
    assert first == second
    assert len(first) == 64
    int(first, 16)
