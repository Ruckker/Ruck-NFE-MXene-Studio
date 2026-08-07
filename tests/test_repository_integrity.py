from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_no_connector_probe_artifact_is_committed() -> None:
    assert not (PROJECT_ROOT / "tmp_probe_should_not_exist").exists(), (
        "temporary connector probe file tmp_probe_should_not_exist must be removed before formal testing"
    )


def test_data_v2_was_not_replaced_by_patch_placeholder() -> None:
    path = PROJECT_ROOT / "src" / "nfe_model" / "data_v2.py"
    text = path.read_text(encoding="utf-8")
    assert "__PATCH_PLACEHOLDER__" not in text
    assert "def build_periodic_graph" in text
    assert "def load_or_build_cache" in text
