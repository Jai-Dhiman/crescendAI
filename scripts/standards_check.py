#!/usr/bin/env python3
"""Deterministic repo-hygiene checks for the rules in docs/standards/rules.json.

Only rules whose `check` field names a function here are mechanical. Everything
else in rules.json is enforced by /review, which cites the same rule ids.

Usage:
    python3 scripts/standards_check.py --staged   # staged files (pre-commit)
    python3 scripts/standards_check.py --all      # every tracked file
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = REPO_ROOT / "docs" / "standards" / "rules.json"
TS_STYLE_PATH = REPO_ROOT / "apps" / "api" / "TS_STYLE.md"

TEXT_SUFFIXES = {
    ".md", ".py", ".ts", ".tsx", ".js", ".sh",
    ".just", ".toml", ".yaml", ".yml", ".json",
}
TEXT_NAMES = {"justfile", "makefile"}  # compared lowercased

# This file's fixtures deliberately contain violations of every rule below, so
# scanning it would report the test data as findings.
FIXTURE_EXEMPT = {"scripts/test_standards_check.py"}


class Finding:
    def __init__(
        self, rule_id: str, level: str, path: str, line: int, excerpt: str, why: str
    ):
        self.rule_id = rule_id
        self.level = level
        self.path = path
        self.line = line
        self.excerpt = excerpt
        self.why = why

    def render(self) -> str:
        return (
            f"[{self.rule_id}] ({self.level}) {self.path}:{self.line}\n"
            f"    {self.excerpt.strip()[:140]}\n"
            f"    {self.why}"
        )


def tracked_files(mode: str) -> list[Path]:
    if mode == "staged":
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
    else:
        cmd = ["git", "ls-files"]
    out = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    paths = []
    for name in out.splitlines():
        p = REPO_ROOT / name
        if name in FIXTURE_EXEMPT:
            continue
        if (p.suffix in TEXT_SUFFIXES or p.name.lower() in TEXT_NAMES) and p.is_file():
            paths.append(p)
    return paths


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []


# --- mechanical checks -------------------------------------------------------

MIGRATE_RE = re.compile(r"bun\s+run\s+migrate\b")
# apps/api/package.json defines the script itself; the justfile recipe is the
# production path and is human-lit by policy.
# Compared lowercased; the tracked file is "Justfile".
MIGRATE_EXEMPT = {"apps/api/package.json", "justfile"}


def check_bare_migrate(files: list[Path], rules: dict) -> list[Finding]:
    rule = rules["OPS-001"]
    findings = []
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.lower() in MIGRATE_EXEMPT:
            continue
        for i, line in enumerate(read_lines(path), 1):
            m = MIGRATE_RE.search(line)
            if m and not in_prose(line, m.start()) and "DATABASE_URL" not in line:
                findings.append(
                    Finding(
                        "OPS-001",
                        rule["level"],
                        rel,
                        i,
                        line,
                        "bare `bun run migrate` targets the hosted production "
                        "database; "
                        'prefix DATABASE_URL="postgresql://.../crescendai_dev".',
                    )
                )
    return findings


# A copy-pasteable command is a footgun; prose *about* the rule is not. The
# discriminator is the character immediately before the match: quoting it
# ("`uv run --with X` mutates the venv") marks it as a mention, not an
# instruction. Anchoring on shell separators instead would miss the common
# docstring form `"""Run: uv run --with pytest ..."""`.
QUOTE_CHARS = "`'\""
UV_RUN_RE = re.compile(r"(uv\s+run\b[^\n&|;`'\"]*)")
CD_RE = re.compile(r"\bcd\s+([\w./\-]+)")


def in_prose(line: str, start: int) -> bool:
    return start > 0 and line[start - 1] in QUOTE_CHARS


def uv_project_dirs() -> set[str]:
    """Directories where `uv run` finds a project whose .venv it would sync."""
    out = subprocess.run(
        ["git", "ls-files", "*pyproject.toml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {Path(name).parent.as_posix() for name in out.splitlines()}


def _under_project(rel_dir: str, projects: set[str]) -> bool:
    parts = Path(rel_dir).as_posix()
    return any(parts == p or parts.startswith(p + "/") for p in projects)


def check_uv_run_isolation(files: list[Path], rules: dict) -> list[Finding]:
    rule = rules["OPS-002"]
    projects = uv_project_dirs()
    findings = []
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        for i, line in enumerate(read_lines(path), 1):
            m = UV_RUN_RE.search(line)
            if not m or in_prose(line, m.start(1)):
                continue
            cmd = m.group(1)
            # The rule only bites where uv actually resolves a project: either the
            # command cds into one, or the file itself lives inside one.
            cd_match = CD_RE.search(line[: m.start(1)])
            run_dir = cd_match.group(1) if cd_match else str(Path(rel).parent)
            if not _under_project(run_dir, projects):
                continue
            if "--with" in cmd and "--no-project" not in cmd:
                findings.append(
                    Finding(
                        "OPS-002",
                        rule["level"],
                        rel,
                        i,
                        line,
                        "`uv run --with X` from inside a project rebuilds the "
                        "shared .venv; "
                        "add --no-project for one-off tools.",
                    )
                )
    return findings


# A quoted relative path containing a separator, used as an argparse default.
ARGPARSE_DEFAULT_RE = re.compile(
    r"""default\s*=\s*(?:Path\()?["'](?!/)[\w.\-]+/[\w./\-*]+["']"""
)
ANCHORED_TOKENS = ("__file__", "REPO_ROOT", "PROJECT_ROOT", "_HERE", "MODULE_DIR")


def check_cwd_relative_defaults(files: list[Path], rules: dict) -> list[Finding]:
    rule = rules["PY-001"]
    findings = []
    for path in files:
        if path.suffix != ".py":
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for i, line in enumerate(read_lines(path), 1):
            if not ARGPARSE_DEFAULT_RE.search(line):
                continue
            if any(tok in line for tok in ANCHORED_TOKENS):
                continue
            findings.append(
                Finding(
                    "PY-001",
                    rule["level"],
                    rel,
                    i,
                    line,
                    "CLI default paths must be anchored to __file__; "
                    "`just` recipes shift CWD.",
                )
            )
    return findings


REFERENCE_MANIFEST_PATH = (
    REPO_ROOT / "apps" / "evals" / "claim_taxonomy" / "reference_calibration.json"
)
MEASURERS_DIR = (
    REPO_ROOT / "apps" / "evals" / "claim_taxonomy" / "verifier" / "measurers"
)
# `NAME = 12.34` at module level. Only float literals: a reference is a corpus median.
REFERENCE_ASSIGN_RE = re.compile(
    r"^(REFERENCE_[A-Z_]+)\s*=\s*(-?\d+(?:\.\d+)?)\s*(?:#.*)?$"
)


def _reference_literals(module_path: Path) -> dict[str, tuple[float, int]]:
    """Every module-level REFERENCE_* float in a measurer, with its line number."""
    found = {}
    for i, line in enumerate(read_lines(module_path), 1):
        m = REFERENCE_ASSIGN_RE.match(line)
        if m:
            found[m.group(1)] = (float(m.group(2)), i)
    return found


def check_reference_calibration(_files: list[Path], rules: dict) -> list[Finding]:
    """A whole_piece reference must be calibrated on the ACTIVE transcription substrate.

    A reference is a corpus-median anchor measured on one (transcriber x corpus) pair.
    Swap
    either and nothing breaks loudly: d stays computable, the router stays confident,
    and the
    faithfulness rate silently collapses. #101 FRONT 9 measured 0.979 -> 0.236 for
    dynamics on
    one stale line; FRONT 10 replicated it at 0.950 -> 0.438 for articulation. This
    check makes
    that failure mechanical in three directions:

      1. the literal in the measurer matches the value recorded in the manifest,
      2. the manifest's calibrated_for.substrate is the active substrate,
      3. no REFERENCE_* constant exists in a measurer without a manifest entry.
    """
    rule = rules["RES-001"]
    rid, level = "RES-001", rule["level"]
    rel_manifest = REFERENCE_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix()
    if not REFERENCE_MANIFEST_PATH.is_file():
        return [Finding(rid, level, rel_manifest, 0, "(missing)",
                        "the reference calibration manifest is missing; every measurer "
                        "reference is unrecorded.")]

    manifest = json.loads(REFERENCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    active = manifest["active_substrate"]
    constants = manifest["constants"]
    findings = []

    for name, entry in constants.items():
        module = REPO_ROOT / entry["module"]
        if not module.is_file():
            findings.append(Finding(rid, level, entry["module"], 0, name,
                                    f"{name} names a module that does not exist."))
            continue
        literals = _reference_literals(module)
        if name not in literals:
            findings.append(Finding(rid, level, entry["module"], 0, name,
                                    f"{name} is recorded in {rel_manifest} but is not "
                                    f"assigned a float literal in the module."))
            continue
        value, line = literals[name]
        if value != entry["value"]:
            findings.append(Finding(
                rid, level, entry["module"], line, f"{name} = {value}",
                f"the code says {value} but {rel_manifest} records {entry['value']}. "
                "Recalibrating means editing both, so the provenance cannot drift."))
        calibrated = entry["calibrated_for"]["substrate"]
        if calibrated != active:
            findings.append(Finding(
                rid, level, entry["module"], line, f"{name} = {value}",
                f"calibrated on '{calibrated}' but the active substrate is '{active}'. "
                "A stale reference does not fail loudly -- it silently collapses the "
                "faithfulness rate (#101 FRONT 9: dynamics 0.979 -> 0.236)."))

    for module in sorted(MEASURERS_DIR.glob("*.py")):
        rel = module.relative_to(REPO_ROOT).as_posix()
        for name, (value, line) in _reference_literals(module).items():
            if name not in constants:
                findings.append(Finding(
                    rid, level, rel, line, f"{name} = {value}",
                    f"{name} is a substrate-calibrated reference with no entry in "
                    f"{rel_manifest}; its calibration provenance is unrecorded."))
    return findings


SLUG_RE = re.compile(r"\[(TS-[A-Z]+-\d{3})\]")


def check_rules_index_sync(_files: list[Path], rules: dict) -> list[Finding]:
    """Every TS-* rule must be defined in TS_STYLE.md, and vice versa."""
    declared = {rid for rid in rules if rid.startswith("TS-")}
    in_doc = set(SLUG_RE.findall(TS_STYLE_PATH.read_text(encoding="utf-8")))
    findings = []
    for rid in sorted(declared - in_doc):
        findings.append(
            Finding(
                "META-001",
                "MUST",
                "docs/standards/rules.json",
                0,
                rid,
                f"{rid} is in rules.json but has no [{rid}] marker "
                "in apps/api/TS_STYLE.md.",
            )
        )
    for rid in sorted(in_doc - declared):
        findings.append(
            Finding(
                "META-001",
                "MUST",
                "apps/api/TS_STYLE.md",
                0,
                rid,
                f"{rid} is marked in TS_STYLE.md but missing from "
                "docs/standards/rules.json.",
            )
        )
    return findings


CHECKS = {
    "bare_migrate": check_bare_migrate,
    "uv_run_isolation": check_uv_run_isolation,
    "cwd_relative_defaults": check_cwd_relative_defaults,
    "rules_index_sync": check_rules_index_sync,
    "reference_calibration": check_reference_calibration,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true", help="check staged files only")
    group.add_argument("--all", action="store_true", help="check every tracked file")
    args = parser.parse_args()

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))["rules"]
    files = tracked_files("staged" if args.staged else "all")

    findings: list[Finding] = []
    for rule_id, rule in rules.items():
        check_name = rule.get("check")
        if not check_name:
            continue
        if check_name not in CHECKS:
            print(
                f"standards_check: rule {rule_id} names unknown check {check_name!r}",
                file=sys.stderr,
            )
            return 2
        findings.extend(CHECKS[check_name](files, rules))

    def is_blocking(f: Finding) -> bool:
        return rules.get(f.rule_id, {}).get("status") == "enforced"

    blocking = [f for f in findings if is_blocking(f)]
    advisory = [f for f in findings if not is_blocking(f)]

    for f in advisory:
        print(f"advisory {f.render()}\n")
    for f in blocking:
        print(f"BLOCKING {f.render()}\n")

    if blocking:
        print(
            f"standards_check: {len(blocking)} blocking finding(s). "
            "See docs/standards/rules.json."
        )
        return 1
    if advisory:
        print(f"standards_check: {len(advisory)} advisory finding(s), 0 blocking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
