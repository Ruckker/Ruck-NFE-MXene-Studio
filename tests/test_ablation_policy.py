from __future__ import annotations

import torch

from nfe_model.train_core import compute_loss
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


def test_classification_only_removes_auxiliary_objectives_but_keeps_supervised_schedule() -> None:
    config, behavior = prepare_ablation(_base_config(), "classification_only")
    assert config["training"]["pretrain_epochs"] == 35
    assert config["loss"]["score_weight"] == 0.0
    assert config["loss"]["auxiliary_weight"] == 0.0
    assert config["loss"]["masked_atom_weight"] == 0.0
    assert config["loss"]["denoise_weight"] == 0.0
    assert behavior["enable_masking"] is False
    assert behavior["enable_denoising"] is False
    assert all(not spec.main for spec in behavior["target_specs"])
    assert config["ablation"]["supervised_weight_schedule"] == "retained_from_full"


def test_no_self_supervision_keeps_full_supervised_schedule_and_targets() -> None:
    config, behavior = prepare_ablation(_base_config(), "no_self_supervision")
    assert config["training"]["pretrain_epochs"] == 35
    assert config["loss"]["auxiliary_weight"] == 0.45
    assert config["loss"]["masked_atom_weight"] == 0.0
    assert config["loss"]["denoise_weight"] == 0.0
    assert behavior["enable_masking"] is False
    assert behavior["enable_denoising"] is False


def test_matched_supervision_keeps_early_supervised_weight_window() -> None:
    config, behavior = prepare_ablation(_base_config(), "matched_supervision")
    assert config["training"]["pretrain_epochs"] == 35
    assert config["loss"]["auxiliary_weight"] == 0.0
    assert config["loss"]["masked_atom_weight"] == 0.0
    assert config["loss"]["denoise_weight"] == 0.0
    assert behavior["target_specs"][0].main is True
    assert all(not spec.main for spec in behavior["target_specs"][1:])


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


def test_classification_only_ignores_nonfinite_disabled_head_outputs() -> None:
    outputs = {
        "class_logits": torch.tensor(
            [[1.0, 0.0, -1.0]], dtype=torch.float32, requires_grad=True
        ),
        "regression_mean": torch.full(
            (1, 2), float("inf"), dtype=torch.float32, requires_grad=True
        ),
        "regression_log_variance": torch.full(
            (1, 2), float("inf"), dtype=torch.float32, requires_grad=True
        ),
        "masked_atom_logits": torch.full(
            (1, 4), float("inf"), dtype=torch.float32, requires_grad=True
        ),
        "denoise_vector": torch.full(
            (1, 3), float("inf"), dtype=torch.float32, requires_grad=True
        ),
    }
    batch = {
        "labels": torch.tensor([0]),
        "sample_weights": torch.tensor([1.0]),
        "targets": torch.zeros(1, 2),
        "target_mask": torch.ones(1, 2, dtype=torch.bool),
        "z": torch.tensor([1]),
    }
    loss, components = compute_loss(
        outputs,
        batch,
        class_weights_tensor=torch.ones(3),
        target_weights=torch.zeros(2),
        loss_config={
            "class_weight": 1.0,
            "label_smoothing": 0.0,
            "masked_atom_weight": 0.0,
            "denoise_weight": 0.0,
        },
        masked_nodes=torch.zeros(1, dtype=torch.bool),
        denoise_target=torch.zeros(1, 3),
        pretraining=True,
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in components.values())
    assert torch.all(torch.isfinite(outputs["class_logits"].grad))
    for name in (
        "regression_mean",
        "regression_log_variance",
        "masked_atom_logits",
        "denoise_vector",
    ):
        assert torch.all(outputs[name].grad == 0), name
