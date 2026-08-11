from __future__ import annotations

import pytest
import torch

from nfe_model.checkpoint_contract import assert_checkpoint_internal_contract
from nfe_model.data_v2 import REGRESSION_TARGETS
from nfe_model.provenance_v2 import NORMALIZER_SCHEMA, tensor_mapping_sha256


def _checkpoint() -> dict:
    normalizers = {
        "target_median": torch.zeros(len(REGRESSION_TARGETS), dtype=torch.float32),
        "target_scale": torch.ones(len(REGRESSION_TARGETS), dtype=torch.float32),
        "global_median": torch.zeros(11, dtype=torch.float32),
        "global_scale": torch.ones(11, dtype=torch.float32),
    }
    return {
        "normalizers": normalizers,
        "model_config": {
            "num_regression_targets": len(REGRESSION_TARGETS),
            "global_features": 11,
        },
        "provenance": {
            "normalizer_schema": NORMALIZER_SCHEMA,
            "normalizer_sha256": tensor_mapping_sha256(
                normalizers, schema=NORMALIZER_SCHEMA
            ),
        },
    }


def test_checkpoint_internal_contract_accepts_consistent_normalizers() -> None:
    assert_checkpoint_internal_contract(_checkpoint())


def test_checkpoint_internal_contract_rejects_swapped_normalizer_tensor() -> None:
    checkpoint = _checkpoint()
    checkpoint["normalizers"]["target_scale"] = torch.full(
        (len(REGRESSION_TARGETS),), 2.0
    )
    with pytest.raises(ValueError, match="do not match checkpoint provenance"):
        assert_checkpoint_internal_contract(checkpoint)


def test_checkpoint_internal_contract_rejects_nonpositive_scale() -> None:
    checkpoint = _checkpoint()
    checkpoint["normalizers"]["global_scale"][0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        assert_checkpoint_internal_contract(checkpoint)
