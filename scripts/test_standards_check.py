"""Behavior tests for the mechanical standards checks.

Run: uv run --no-project --with pytest pytest scripts/test_standards_check.py -q

Each check is exercised through its public entry point on a real file under the
repo root, in both directions: a violation is reported, and the corrected form
is not. Without the negative case a check that flags everything would pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import standards_check as sc  # noqa: E402

RULES = {
    "OPS-001": {"level": "MUST"},
    "OPS-002": {"level": "MUST"},
    "PY-001": {"level": "MUST"},
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
