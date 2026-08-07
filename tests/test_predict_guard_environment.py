from __future__ import annotations

import pytest

from nfe_model import predict_guard


def _provenance(numpy: str = "2.1.0", pymatgen: str = "2025.1.1") -> dict:
    return {
        "runtime_environment": {
            "packages": {
                "numpy": numpy,
                "pymatgen": pymatgen,
            }
        }
    }


def test_feature_builder_environment_accepts_matching_versions(monkeypatch) -> None:
    monkeypatch.setattr(
        predict_guard,
        "runtime_environment",
        lambda: {"packages": {"numpy": "2.1.0", "pymatgen": "2025.1.1"}},
    )
    identity = predict_guard._assert_feature_builder_environment(_provenance())
    assert identity == (("numpy", "2.1.0"), ("pymatgen", "2025.1.1"))


def test_feature_builder_environment_rejects_pymatgen_drift(monkeypatch) -> None:
    monkeypatch.setattr(
        predict_guard,
        "runtime_environment",
        lambda: {"packages": {"numpy": "2.1.0", "pymatgen": "2026.1.0"}},
    )
    with pytest.raises(ValueError, match="pymatgen training=2025.1.1 runtime=2026.1.0"):
        predict_guard._assert_feature_builder_environment(_provenance())


def test_feature_builder_environment_rejects_unknown_version(monkeypatch) -> None:
    monkeypatch.setattr(
        predict_guard,
        "runtime_environment",
        lambda: {"packages": {"numpy": "unknown", "pymatgen": "2025.1.1"}},
    )
    with pytest.raises(ValueError, match="resolvable numpy version"):
        predict_guard._assert_feature_builder_environment(_provenance())
