from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from data_tools import build_nfe_dataset_audit as dataset_audit
from nfe_model.metrics_v2 import regression_metrics
from nfe_model.surface_generator_data import SurfaceTemplateDataset
from nfe_model import train_surface_generator_formal as generator_formal
from training.evaluation.audit_split_duplicates import _near_duplicate_review
from training.evaluation.paired_bootstrap import _values as paired_values
from training.evaluation import paper_preflight_strict


def test_formal_bootstrap_includes_high_enrichment_at_5pct() -> None:
    frame = pd.DataFrame(
        {
            "True_Label": ["low", "medium", "high", "low", "medium", "high"],
            "A_Probability_Low": [0.9, 0.1, 0.05, 0.8, 0.1, 0.05],
            "A_Probability_Medium": [0.05, 0.8, 0.05, 0.1, 0.8, 0.05],
            "A_Probability_High": [0.05, 0.1, 0.90, 0.1, 0.1, 0.90],
            "B_Probability_Low": [0.05, 0.1, 0.8, 0.9, 0.1, 0.8],
            "B_Probability_Medium": [0.05, 0.8, 0.1, 0.05, 0.8, 0.1],
            "B_Probability_High": [0.90, 0.1, 0.1, 0.05, 0.1, 0.1],
            "True_NFE_Pseudo_Score": [0.1, 0.5, 0.9, 0.2, 0.6, 0.8],
            "A_Predicted_NFE_Pseudo_Score": [0.1, 0.5, 0.9, 0.2, 0.6, 0.8],
            "B_Predicted_NFE_Pseudo_Score": [0.2, 0.4, 0.7, 0.3, 0.5, 0.6],
        }
    )
    values = paired_values(frame)
    assert "high_enrichment_at_5pct" in values
    assert np.isfinite(values["high_enrichment_at_5pct"])


def test_regression_metrics_report_support_and_ranges() -> None:
    prediction = np.asarray([[0.2], [0.6], [0.9]], dtype=float)
    target = np.asarray([[0.1], [0.5], [0.8]], dtype=float)
    mask = np.asarray([[True], [False], [True]])
    metrics = regression_metrics(prediction, target, mask, ["bounded"])
    assert metrics["bounded_support"] == 2.0
    assert metrics["bounded_target_min"] == pytest.approx(0.1)
    assert metrics["bounded_target_max"] == pytest.approx(0.8)
    assert metrics["bounded_prediction_min"] == pytest.approx(0.2)
    assert metrics["bounded_prediction_max"] == pytest.approx(0.9)


def _template_record(identifier: str, topology: str, value: float) -> dict:
    return {
        "id": identifier,
        "topology_key": (topology,),
        "frac_pos": torch.tensor([[value, 0.0, 0.0]], dtype=torch.float32),
        "lattice_params": torch.full((6,), value, dtype=torch.float32),
    }


def test_surface_generator_eval_uses_train_only_deterministic_template() -> None:
    records = [
        _template_record("train-template", "same", 0.1),
        _template_record("validation-target", "same", 0.2),
    ]
    normalizers = {
        "lattice_median": torch.zeros(6),
        "lattice_scale": torch.ones(6),
    }
    dataset = SurfaceTemplateDataset(
        records,
        [1],
        normalizers,
        template_indices=[0],
        deterministic_templates=True,
    )
    item = dataset[0]
    assert item["template_id"] == "train-template"
    assert torch.equal(item["template_frac"], records[0]["frac_pos"])


def test_surface_generator_eval_rejects_unseen_topology() -> None:
    records = [
        _template_record("train-template", "train", 0.1),
        _template_record("validation-target", "unseen", 0.2),
    ]
    normalizers = {
        "lattice_median": torch.zeros(6),
        "lattice_scale": torch.ones(6),
    }
    with pytest.raises(RuntimeError, match="no train-template support"):
        SurfaceTemplateDataset(
            records,
            [1],
            normalizers,
            template_indices=[0],
            deterministic_templates=True,
        )


def test_generator_distributed_eval_sampler_has_no_padding_duplicates(monkeypatch) -> None:
    monkeypatch.setattr(generator_formal.dist, "is_available", lambda: True)
    monkeypatch.setattr(generator_formal.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(generator_formal.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(generator_formal.dist, "get_world_size", lambda: 3)
    sampler = generator_formal.ExactDistributedEvalSampler(list(range(8)))
    assert list(sampler) == [1, 4, 7]
    assert len(sampler) == 3


def test_dataset_gamma_indices_must_really_be_gamma_vectors() -> None:
    band = SimpleNamespace(
        energies=np.zeros((1, 18, 1), dtype=float),
        kpoints=np.zeros((18, 3), dtype=float),
    )
    dataset_audit._assert_indexed_gamma_vectors(band)
    band.kpoints[11] = [0.01, 0.0, 0.0]
    with pytest.raises(ValueError, match="does not land on Gamma|does not land|Gamma k-vectors"):
        dataset_audit._assert_indexed_gamma_vectors(band)


def test_dataset_name_chemistry_includes_termination_elements() -> None:
    expected = dataset_audit._expected_name_elements(
        {
            "Name_Parse_OK": True,
            "Metal_Top": "Nb",
            "Metal_Bottom": "Nb",
            "X_Element": "C",
            "Termination_Top": "OH",
            "Termination_Bottom": "F",
        }
    )
    assert expected == {"Nb", "C", "O", "H", "F"}


def test_near_duplicate_candidates_require_disposition_closure(tmp_path) -> None:
    candidates = [
        {
            "fingerprint": "abc",
            "splits": ["train", "test"],
            "structures": ["a", "b"],
        }
    ]
    missing = tmp_path / "missing.json"
    closed, manifest = _near_duplicate_review(candidates, missing)
    assert closed is False
    assert manifest is None

    path = tmp_path / "dispositions.json"
    path.write_text(
        json.dumps(
            {
                "dispositions": {
                    "abc": {
                        "decision": "distinct_after_review",
                        "reviewer": "reviewer-1",
                        "rationale": "manual structure comparison confirms distinct slabs",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    closed, manifest = _near_duplicate_review(candidates, path)
    assert closed is True
    assert manifest is not None
    assert len(manifest["sha256"]) == 64


def test_paper_preflight_reconstructs_xgboost_state_from_artifacts(tmp_path) -> None:
    classifier = b"classifier-ubj-bytes"
    regressor = b"regressor-ubj-bytes"
    classifier_path = tmp_path / "xgboost_classifier.ubj"
    regressor_path = tmp_path / "xgboost_regressor.ubj"
    classifier_path.write_bytes(classifier)
    regressor_path.write_bytes(regressor)

    digest = hashlib.sha256()
    digest.update(b"xgboost-classifier\0")
    digest.update(classifier)
    digest.update(b"\0xgboost-regressor\0")
    digest.update(regressor)
    payload = {
        "details": {
            "score_regressor_fitted": True,
            "model_state_sha256": digest.hexdigest(),
            "booster_artifacts": {
                "format": "ubj",
                "classifier": {
                    "file": classifier_path.name,
                    "sha256": hashlib.sha256(classifier).hexdigest(),
                },
                "regressor": {
                    "file": regressor_path.name,
                    "sha256": hashlib.sha256(regressor).hexdigest(),
                },
            },
        }
    }
    result_path = tmp_path / "result.json"
    paper_preflight_strict._validate_xgboost_artifacts(payload, result_path)

    classifier_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="artifact SHA256 mismatch"):
        paper_preflight_strict._validate_xgboost_artifacts(payload, result_path)
