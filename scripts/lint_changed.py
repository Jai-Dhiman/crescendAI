#!/usr/bin/env python3
"""Lint staged files, but block only on findings that sit on changed lines.

`.githooks/pre-commit` used to run `biome check` and `ruff check` over whole
staged files. With `just lint-api` red across 161 files (#142), that blocked
nearly every commit to a busy file on debt it did not introduce, so
`--no-verify` became the default and the gate stopped meaning anything (#147).

This filters each finding against the lines the commit actually touched:

  blocking  -- the finding is on a line in the staged diff
  advisory  -- everything else; printed so debt stays visible, never blocks

Whole-file findings need care. Biome reports `format` at line 0 and
`assist/source/organizeImports` at line 1, neither of which is a real location.
For `format` we reformat the file and block only if the reformatted lines
intersect the changed set. `organizeImports` is genuinely file-wide -- an import
block is either sorted or not -- so it is always advisory.

Usage: lint_changed.py            # staged files (pre-commit)
       lint_changed.py --debug    # also print the changed-line map
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Biome categories that carry no usable line number.
FORMAT_CATEGORY = "format"
IMPORTS_CATEGORY = "assist/source/organizeImports"

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Finding:
    tool: str
    path: str
    line: int | None
    code: str
    severity: str
    message: str


def parse_changed_lines(diff_text: str) -> set[int]:
    """Return the new-file line numbers touched by a unified diff.

    Only hunk headers are read, so this works with `-U0` output regardless of
    content. A hunk `@@ -a,b +c,d @@` covers new-file lines c .. c+d-1; `d`
    defaults to 1 when absent, and `d == 0` (pure deletion) contributes nothing.
    """
    changed: set[int] = set()
    for raw in diff_text.splitlines():
        m = HUNK_RE.match(raw)
        if not m:
            continue
        start = int(m.group(1))
        count = 1 if m.group(2) is None else int(m.group(2))
        changed.update(range(start, start + count))
    return changed


def staged_changed_lines(path: str) -> set[int]:
    out = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return parse_changed_lines(out)


def partition(
    findings: list[Finding], changed: dict[str, set[int]]
) -> tuple[list[Finding], list[Finding]]:
    """Split findings into (blocking, advisory).

    A finding blocks only when it names a line the commit changed. Findings with
    no line (whole-file) are advisory; callers resolve `format` separately,
    because only they can reformat the file to find real line numbers.
    """
    blocking: list[Finding] = []
    advisory: list[Finding] = []
    for f in findings:
        if f.line is not None and f.line in changed.get(f.path, set()):
            blocking.append(f)
        else:
            advisory.append(f)
    return blocking, advisory


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def reformatted_lines(app_dir: Path, rel_path: str) -> set[int]:
    """Lines that differ between the file and what `biome format` would write.

    Biome's `format` diagnostic has no line number, so this recovers one. An
    empty set means the file's formatting problems (if any) are outside the
    changed region -- or that formatting is clean.

    Must use `--stdin-file-path`: `biome format <path>` prints a human summary
    ("Checked 1 file..."), not the formatted source. Diffing against that
    summary marks every line as reformatted and blocks every commit.
    """
    current_text = (app_dir / rel_path).read_text(encoding="utf-8")
    proc = subprocess.run(
        ["bunx", "biome", "format", f"--stdin-file-path={rel_path}"],
        cwd=app_dir,
        input=current_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if not proc.stdout:
        return set()
    current = current_text.splitlines()
    formatted = proc.stdout.splitlines()
    lines: set[int] = set()
    matcher = difflib.SequenceMatcher(a=current, b=formatted, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        # i1/i2 index `current`; +1 converts to 1-based line numbers. A pure
        # insertion (i1 == i2) still implicates the line it lands on.
        lines.update(range(i1 + 1, max(i2, i1 + 1) + 1))
    return lines


def biome_findings(app: str, rel_files: list[str]) -> list[Finding]:
    app_dir = ROOT / "apps" / app
    proc = _run(["bunx", "biome", "check", "--reporter=json", *rel_files], app_dir)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(
            f"lint_changed: could not parse biome JSON for apps/{app}",
            file=sys.stderr,
        )
        print(proc.stdout[:500] or proc.stderr[:500], file=sys.stderr)
        raise

    out: list[Finding] = []
    for d in payload.get("diagnostics", []):
        loc = d.get("location") or {}
        rel = loc.get("path")
        if not rel:
            continue
        repo_path = f"apps/{app}/{rel}"
        category = d.get("category", "")
        line = ((loc.get("start") or {}).get("line")) or None

        if category == IMPORTS_CATEGORY:
            line = None  # file-wide by nature; never blocking
        elif category == FORMAT_CATEGORY:
            # Recover real line numbers by reformatting.
            for n in reformatted_lines(app_dir, rel):
                out.append(
                    Finding(
                        "biome", repo_path, n, category, "error", "needs formatting"
                    )
                )
            continue

        out.append(
            Finding(
                "biome",
                repo_path,
                line,
                category,
                d.get("severity", "info"),
                # Biome puts the human text in `message`; there is no
                # `description` key, so reading one yields blank output.
                str(d.get("message") or "").strip(),
            )
        )
    return out


def ruff_findings(files: list[str]) -> list[Finding]:
    ruff = shutil.which("ruff")
    if ruff is None:
        print("lint_changed: ruff not on PATH; skipping python lint", file=sys.stderr)
        return []
    proc = _run(
        [
            ruff,
            "check",
            "--config",
            "model/pyproject.toml",
            "--output-format",
            "json",
            *files,
        ],
        ROOT,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print("lint_changed: could not parse ruff JSON", file=sys.stderr)
        print(proc.stdout[:500] or proc.stderr[:500], file=sys.stderr)
        raise

    out: list[Finding] = []
    for d in payload:
        rel = d.get("filename", "")
        try:
            rel = str(Path(rel).resolve().relative_to(ROOT))
        except ValueError:
            pass
        out.append(
            Finding(
                "ruff",
                rel,
                (d.get("location") or {}).get("row"),
                d.get("code") or "",
                "error",
                (d.get("message") or "").strip(),
            )
        )
    return out


def staged_files() -> list[str]:
    out = _run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"], ROOT
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug", action="store_true", help="print the changed-line map"
    )
    args = parser.parse_args()

    files = staged_files()
    if not files:
        return 0

    findings: list[Finding] = []
    for app in ("api", "web"):
        prefix = f"apps/{app}/"
        rel = [
            f[len(prefix) :]
            for f in files
            if f.startswith(prefix) and f.endswith((".ts", ".tsx", ".js", ".jsx"))
        ]
        if rel:
            findings.extend(biome_findings(app, rel))

    pyfiles = [f for f in files if f.endswith(".py")]
    if pyfiles:
        findings.extend(ruff_findings(pyfiles))

    if not findings:
        return 0

    changed = {f: staged_changed_lines(f) for f in {fi.path for fi in findings}}
    if args.debug:
        for path, lines in sorted(changed.items()):
            print(f"lint_changed: {path} changed lines -> {sorted(lines)}")

    blocking, advisory = partition(findings, changed)

    if advisory:
        print(
            f"lint_changed: {len(advisory)} pre-existing finding(s) "
            "on untouched lines (not blocking)"
        )

    if not blocking:
        return 0

    print("")
    print("lint_changed: findings on lines this commit changed:")
    for f in sorted(blocking, key=lambda x: (x.path, x.line or 0)):
        print(f"  {f.path}:{f.line}  [{f.code}] {f.message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
