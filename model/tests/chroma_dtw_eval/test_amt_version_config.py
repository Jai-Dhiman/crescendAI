"""Locks the committed AMT-version config schema."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "model/config/amt_version.json"


def test_config_present_and_has_required_fields() -> None:
    assert CONFIG.exists(), f"committed AMT version config missing: {CONFIG}"
    body = json.loads(CONFIG.read_text())
    assert isinstance(body.get("checkpoint_hash"), str)
    assert len(body["checkpoint_hash"]) >= 16
    assert isinstance(body.get("parangonar_version"), str)
    assert body["parangonar_version"], "parangonar_version must be non-empty"
    assert isinstance(body.get("regen_source_default"), str)
    assert body["regen_source_default"], "regen_source_default must be non-empty"
