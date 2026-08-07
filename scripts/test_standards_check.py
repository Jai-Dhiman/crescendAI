"""Behavior tests for the mechanical standards checks.

Run: uv run --no-project --with pytest pytest scripts/test_standards_check.py -q

Each check is exercised through its public entry point on a real file under the
repo root, in both directions: a violation is reported, and the corrected form
is not. Without the negative case a check that flags everything would pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import standards_check as sc  # noqa: E402

RULES = {
    "OPS-001": {"level": "MUST"},
    "OPS-002": {"level": "MUST"},
    "PY-001": {"level": "MUST"},
    "RES-001": {"level": "MUST"},
}


@pytest.fixture
def tmp_repo_file():
    """A file inside REPO_ROOT (the checks resolve paths relative to it)."""
    created = []

    def make(relative_path: str, content: str) -> Path:
        p = sc.REPO_ROOT / relative_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        created.append(p)
        return p

    yield make

    for p in created:
        p.unlink(missing_ok=True)
        if p.parent != sc.REPO_ROOT and not any(p.parent.iterdir()):
            p.parent.rmdir()


def test_bare_migrate_is_flagged(tmp_repo_file):
    f = tmp_repo_file("_tmp_standards/doc.md", "Run: cd apps/api && bun run migrate\n")
    findings = sc.check_bare_migrate([f], RULES)
    assert [x.rule_id for x in findings] == ["OPS-001"]


def test_migrate_with_explicit_database_url_is_clean(tmp_repo_file):
    f = tmp_repo_file(
        "_tmp_standards/doc.md",
        'Run: cd apps/api && DATABASE_URL="postgresql://localhost/crescendai_dev" '
        "bun run migrate\n",
    )
    assert sc.check_bare_migrate([f], RULES) == []


def test_uv_run_without_no_project_inside_a_project_is_flagged(tmp_repo_file):
    f = tmp_repo_file(
        "model/_tmp_standards/doc.py", '"""Run: uv run --with pytest pytest"""\n'
    )
    findings = sc.check_uv_run_isolation([f], RULES)
    assert [x.rule_id for x in findings] == ["OPS-002"]


def test_uv_run_with_no_project_is_clean(tmp_repo_file):
    f = tmp_repo_file(
        "model/_tmp_standards/doc.py",
        '"""Run: uv run --no-project --with pytest pytest"""\n',
    )
    assert sc.check_uv_run_isolation([f], RULES) == []


def test_uv_run_outside_any_uv_project_is_clean(tmp_repo_file):
    """The repo root has no pyproject.toml, so there is no venv to mutate."""
    f = tmp_repo_file(
        "_tmp_standards/doc.md", "Run: uv run --with pyyaml python x.py\n"
    )
    assert sc.check_uv_run_isolation([f], RULES) == []


def test_prose_describing_the_gotcha_is_not_flagged(tmp_repo_file):
    """Only commands in command position count; documentation of the rule does not."""
    f = tmp_repo_file(
        "model/_tmp_standards/doc.py",
        '"""Note: `uv run --with X` inside model/ rebuilds the shared venv."""\n',
    )
    assert sc.check_uv_run_isolation([f], RULES) == []


def test_cwd_relative_cli_default_is_flagged(tmp_repo_file):
    f = tmp_repo_file(
        "_tmp_standards/cli.py",
        'ap.add_argument("--root", type=Path, default=Path("data/evals"))\n',
    )
    findings = sc.check_cwd_relative_defaults([f], RULES)
    assert [x.rule_id for x in findings] == ["PY-001"]


def test_file_anchored_cli_default_is_clean(tmp_repo_file):
    f = tmp_repo_file(
        "_tmp_standards/cli.py",
        'ap.add_argument(\n'
        '    "--root", type=Path, default=Path(__file__).parent / "data/evals"\n'
        ')\n',
    )
    assert sc.check_cwd_relative_defaults([f], RULES) == []


@pytest.fixture
def reference_setup(tmp_path, monkeypatch):
    """Point RES-001 at a throwaway manifest + measurers dir.

    `substrate` is what the reference was calibrated on; `value` is what the manifest
    records;
    `code_value` is the literal actually in the module. Defaults are the clean,
    consistent case.
    """

    def make(
        *, substrate="transkun", value=0.8159, code_value=None, extra_module_line=""
    ):
        measurers = tmp_path / "measurers"
        measurers.mkdir(exist_ok=True)
        module = measurers / "articulation.py"
        module.write_text(
            f"REFERENCE_RATIO = {value if code_value is None else code_value}\n"
            f"{extra_module_line}",
            encoding="utf-8",
        )
        manifest = tmp_path / "reference_calibration.json"
        manifest.write_text(
            json.dumps({
                "active_substrate": "transkun",
                "constants": {
                    "REFERENCE_RATIO": {
                        "module": module.relative_to(sc.REPO_ROOT).as_posix()
                        if module.is_relative_to(sc.REPO_ROOT)
                        else str(module),
                        "value": value,
                        "calibrated_for": {"substrate": substrate, "corpus": "maestro"},
                    }
                },
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(sc, "REFERENCE_MANIFEST_PATH", manifest)
        monkeypatch.setattr(sc, "MEASURERS_DIR", measurers)
        monkeypatch.setattr(sc, "REPO_ROOT", tmp_path.parent)
        # the manifest stores module paths relative to REPO_ROOT
        data = json.loads(manifest.read_text())
        data["constants"]["REFERENCE_RATIO"]["module"] = module.relative_to(
            tmp_path.parent
        ).as_posix()
        manifest.write_text(json.dumps(data), encoding="utf-8")
        return manifest

    return make


def test_reference_calibrated_on_the_active_substrate_is_clean(reference_setup):
    reference_setup(substrate="transkun")
    assert sc.check_reference_calibration([], RULES) == []


def test_reference_calibrated_on_a_retired_substrate_is_flagged(reference_setup):
    """The FRONT 9 trap: aria-era reference, Transkun substrate, no loud failure."""
    reference_setup(substrate="aria-amt")
    findings = sc.check_reference_calibration([], RULES)
    assert len(findings) == 1
    assert findings[0].rule_id == "RES-001"
    assert "aria-amt" in findings[0].why and "transkun" in findings[0].why


def test_code_value_drifting_from_the_recorded_value_is_flagged(reference_setup):
    """Recalibrating in code without recording it is the same failure one step earlier.
    """
    reference_setup(value=0.8159, code_value=64.33)
    findings = sc.check_reference_calibration([], RULES)
    assert len(findings) == 1
    assert "64.33" in findings[0].why and "0.8159" in findings[0].why


def test_a_wrapped_reference_literal_is_reported_not_skipped(reference_setup):
    """A formatter can wrap an assignment across lines. The constant must still be SEEN
    and reported as uncheckable -- silently skipping it is the exact failure mode
    RES-001 exists to prevent."""
    manifest = reference_setup()
    module = manifest.parent / "measurers" / "articulation.py"
    module.write_text("REFERENCE_RATIO = (\n    0.8159\n)\n", encoding="utf-8")
    findings = sc.check_reference_calibration([], RULES)
    assert len(findings) == 1
    assert "plain float literal" in findings[0].why


def test_a_manifest_entry_missing_a_key_is_reported_not_raised(reference_setup):
    """The manifest is hand-edited. A KeyError escaping the check would abort every
    pre-push with a traceback instead of naming the problem."""
    manifest = reference_setup()
    data = json.loads(manifest.read_text())
    del data["constants"]["REFERENCE_RATIO"]["calibrated_for"]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    findings = sc.check_reference_calibration([], RULES)
    assert len(findings) == 1
    assert "calibrated_for" in findings[0].why


def test_an_unrecorded_reference_constant_is_flagged(reference_setup):
    """A new measurer reference with no provenance entry at all."""
    reference_setup(extra_module_line="REFERENCE_BRIGHTNESS = 12.5\n")
    findings = sc.check_reference_calibration([], RULES)
    assert len(findings) == 1
    assert "REFERENCE_BRIGHTNESS" in findings[0].excerpt


def test_live_manifest_matches_the_live_measurers():
    """The shipped manifest must describe the shipped code.

    REFERENCE_VELOCITY (dynamics) is the one remaining aria-era violation, held open
    at `approved` because shipping its recalibration changes production verdicts.
    REFERENCE_FRACTION was recalibrated in FRONT 10 -- the over-pedal scoping lift
    made the stale value newly harmful -- so it must NOT appear here. Anything else
    in this set means the manifest has drifted from the code.
    """
    rules = json.loads(sc.RULES_PATH.read_text(encoding="utf-8"))["rules"]
    findings = sc.check_reference_calibration([], rules)
    assert {f.excerpt.split(" =")[0] for f in findings} == {"REFERENCE_VELOCITY"}
    assert all("calibrated on 'aria-amt'" in f.why for f in findings)


def test_res_001_is_not_enforced_while_the_aria_references_stand():
    """Promotion-ramp guard: RES-001 must not block while known violations are live."""
    rules = json.loads(sc.RULES_PATH.read_text(encoding="utf-8"))["rules"]
    if sc.check_reference_calibration([], rules):
        assert rules["RES-001"]["status"] != "enforced"


def test_rules_index_and_style_guide_do_not_drift():
    """The live rules.json and TS_STYLE.md must agree. This is META-001 itself."""
    import json

    rules = json.loads(sc.RULES_PATH.read_text(encoding="utf-8"))["rules"]
    assert sc.check_rules_index_sync([], rules) == []


def test_every_declared_check_exists():
    """A rule naming a check that no longer exists would silently exit 2 in CI."""
    import json

    rules = json.loads(sc.RULES_PATH.read_text(encoding="utf-8"))["rules"]
    named = {r["check"] for r in rules.values() if r.get("check")}
    assert named <= set(sc.CHECKS)


def test_statuses_are_from_the_promotion_ramp():
    import json

    rules = json.loads(sc.RULES_PATH.read_text(encoding="utf-8"))["rules"]
    assert {r["status"] for r in rules.values()} <= {"proposed", "approved", "enforced"}
