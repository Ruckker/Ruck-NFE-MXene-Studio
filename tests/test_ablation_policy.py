from __future__ import annotations

import torch

from nfe_model.train_ablation import _make_corrupt_structure, prepare_ablation


def _base_config() -> dict:
    return {
        "data": {},
        "model": {},
        "training": {"pretrain_epochs": 35},
        "loss": {
            "score_weight": 1.5,
            "auxiliary_weight": 0.45,
            "masked_atom_weight": 0.35,
            "denoise_weight": 0.65,
        },
        "inference": {},
    }


def test_classification_only_removes_all_auxiliary_objectives() -> None:
    config, behavior = prepare_ablation(_base_config(), "classification_only")
    assert config["training"]["pretrain_epochs"] == 0
    assert config["loss"]["score_weight"] == 0.0
    assert config["loss"]["auxiliary_weight"] == 0.0
    assert config["loss"]["masked_atom_weight"] == 0.0
    assert config["loss"]["denoise_weight"] == 0.0
    assert behavior["enable_masking"] is False
    assert behavior["enable_denoising"] is False
    assert all(not spec.main for spec in behavior["target_specs"])


def test_no_auxiliary_regression_keeps_only_nfe_score_main() -> None:
    config, behavior = prepare_ablation(_base_config(), "no_auxiliary_regression")
    assert config["loss"]["score_weight"] == 1.5
    assert config["loss"]["auxiliary_weight"] == 0.0
    assert behavior["target_specs"][0].main is True
    assert all(not spec.main for spec in behavior["target_specs"][1:])


def test_disabled_corruption_leaves_inputs_unchanged() -> None:
    batch = {
        "z": torch.tensor([6, 41, 8], dtype=torch.long),
        "frac_pos": torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=torch.float32,
        ),
        "lattice": torch.eye(3).unsqueeze(0) * 10.0,
        "batch": torch.zeros(3, dtype=torch.long),
    }
    corrupt = _make_corrupt_structure(enable_masking=False, enable_denoising=False)
    z, pos, mask, denoise = corrupt(
        batch,
        mask_probability=0.15,
        noise_min=0.01,
        noise_max=0.15,
    )
    assert torch.equal(z, batch["z"])
    assert torch.equal(pos, batch["frac_pos"])
    assert not torch.any(mask)
    assert torch.count_nonzero(denoise) == 0
