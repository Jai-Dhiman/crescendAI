from __future__ import annotations

import re
import subprocess

import pytest

from shared.provenance import RunProvenance, make_run_provenance


def test_make_run_provenance_returns_all_fields() -> None:
    prov = make_run_provenance()
    assert isinstance(prov, RunProvenance)
    assert prov.run_id
    assert prov.git_sha
    assert isinstance(prov.git_dirty, bool)


def test_run_id_is_filesystem_safe() -> None:
    prov = make_run_provenance()
    # No characters that would break a filename on macOS/Linux
    assert re.match(r"^[A-Za-z0-9._\-]+$", prov.run_id), f"unsafe run_id: {prov.run_id}"


def test_run_id_includes_suffix_when_given() -> None:
    prov = make_run_provenance(suffix="candidate-42")
    assert "candidate-42" in prov.run_id


def test_git_sha_is_hex_string_when_git_available() -> None:
    prov = make_run_provenance()
    if prov.git_sha == "unknown":
        pytest.skip("git not available in this environment")
    assert re.match(r"^[0-9a-f]{7,40}$", prov.git_sha), f"not a git sha: {prov.git_sha}"


def test_falls_back_gracefully_when_git_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git binary not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    prov = make_run_provenance()
    assert prov.git_sha == "unknown"
    assert prov.git_dirty is True
    assert prov.run_id  # still produces an id


def test_two_calls_close_in_time_have_distinct_ids_when_suffix_differs() -> None:
    a = make_run_provenance(suffix="a")
    b = make_run_provenance(suffix="b")
    assert a.run_id != b.run_id
