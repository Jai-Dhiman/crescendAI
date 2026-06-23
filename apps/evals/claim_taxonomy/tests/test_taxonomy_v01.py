from __future__ import annotations
import json
from pathlib import Path
import jsonschema

TAXONOMY_DIR = Path(__file__).resolve().parents[1]

def _load():
    taxonomy = json.loads((TAXONOMY_DIR / "claim_taxonomy.json").read_text())
    schema = json.loads((TAXONOMY_DIR / "claim_taxonomy.schema.json").read_text())
    return taxonomy, schema


def test_taxonomy_version_is_v02() -> None:
    taxonomy, _ = _load()
    assert taxonomy["taxonomy_version"] == "v0.2", (
        f"Expected v0.2, got {taxonomy['taxonomy_version']}"
    )


def test_localization_granularity_restriction_recorded() -> None:
    """GATE 1 (#95): bar-level localization failed -> the taxonomy records the
    restriction (single-bar inadmissible, region degraded, whole_piece reliable)."""
    taxonomy, _ = _load()
    lg = taxonomy["localization_granularity"]
    assert lg["gate_1_verdict"] == "FAIL_BAR_LEVEL"
    assert lg["tiers"]["single_bar"]["admissible"] is False
    assert lg["tiers"]["whole_piece"]["admissible"] is True
    assert lg["tiers"]["region"]["reliability"] == "degraded"
    assert lg["tiers"]["region"]["min_region_bars"] >= 2


def test_dynamics_is_active() -> None:
    taxonomy, _ = _load()
    dyn = taxonomy["dimensions"]["dynamics"]
    assert dyn["status"] == "active", f"Expected active, got {dyn['status']}"


def test_dynamics_has_all_active_fields() -> None:
    taxonomy, _ = _load()
    dyn = taxonomy["dimensions"]["dynamics"]
    for field in ("reference", "check", "tolerance", "reliability_tier", "measurement", "minimum_events"):
        assert field in dyn, f"dynamics missing field: {field}"
    assert dyn["tolerance"]["locked"] is False
    assert dyn["tolerance"]["calibration_source"] == "#65/M1 error-bar study"


def test_v01_taxonomy_validates_against_schema() -> None:
    taxonomy, schema = _load()
    jsonschema.validate(instance=taxonomy, schema=schema)


def test_three_active_dimensions() -> None:
    taxonomy, _ = _load()
    active = [k for k, v in taxonomy["dimensions"].items() if v["status"] == "active"]
    assert set(active) == {"timing", "pedaling", "dynamics"}, f"Active dims: {active}"


def test_signed_d_convention_doc_exists() -> None:
    doc = Path(__file__).resolve().parents[4] / "docs/model/claim-verifier-signed-d-conventions.md"
    assert doc.exists(), f"Signed-d convention doc not found at {doc}"
    content = doc.read_text()
    assert "timing" in content
    assert "pedaling" in content
    assert "dynamics" in content
    assert "Sign convention" in content
    assert "error_bar" in content
