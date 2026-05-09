"""Pipeline B+ runner: replays positive successful tool calls, classifies follow-up."""
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.stage0.run_continuation import run as run_continuation


class _ScriptedClient:
    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.model = "fake/qwen"

    def complete(self, user: str, system: str = "", max_tokens: int = 0) -> str:
        import re
        m = re.search(r"<case_id>(.+?)</case_id>", user)
        return self.replies.get(m.group(1) if m else "", "")


def _write_cases(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"case_id": "p_search_01", "expected_call": True, "expected_tool": "search_catalog",
         "category": None, "briefing": {"prompt": "find Chopin"}},
        {"case_id": "p_search_02", "expected_call": True, "expected_tool": "search_catalog",
         "category": None, "briefing": {"prompt": "find Liszt"}},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_tool_runs(path: Path) -> None:
    rows = [
        {"case_id": "p_search_01", "expected_call": True, "called": True, "tool_name": "search_catalog",
         "arguments": {"query": "Chopin"}, "discipline_correct": True, "format_valid": True,
         "raw_response": '<tool_call>{"name":"search_catalog","arguments":{"query":"Chopin"}}</tool_call>',
         "category": None, "error": ""},
        {"case_id": "p_search_02", "expected_call": True, "called": False, "tool_name": None,
         "arguments": None, "discipline_correct": False, "format_valid": None,
         "raw_response": "I don't know.", "category": None, "error": ""},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_only_replays_successful_positive_calls(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    runs = tmp_path / "tool_runs.jsonl"
    out = tmp_path / "cont.jsonl"
    _write_cases(cases)
    _write_tool_runs(runs)
    client = _ScriptedClient({
        "p_search_01": "Great -- try Chopin's Ballade No. 1, focusing on the second theme around bar 68 for that singing tone.",
    })
    stats = run_continuation(
        tool_runs_path=runs, cases_path=cases,
        teacher_client=client, out_path=out,
    )
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["case_id"] == "p_search_01"
    assert rows[0]["category"] == "clean"
    assert rows[0]["is_degenerate"] is False
    assert stats.processed == 1


def test_classifies_refusal(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    runs = tmp_path / "tool_runs.jsonl"
    out = tmp_path / "cont.jsonl"
    _write_cases(cases)
    _write_tool_runs(runs)
    client = _ScriptedClient({"p_search_01": "I cannot continue. I am unable to help with this request."})
    run_continuation(
        tool_runs_path=runs, cases_path=cases,
        teacher_client=client, out_path=out,
    )
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows[0]["category"] == "refusal"
    assert rows[0]["is_degenerate"] is True


def test_resume_skips_completed_cases(tmp_path: Path) -> None:
    """Re-running continuation must not duplicate rows for already-completed cases."""
    cases = tmp_path / "cases.jsonl"
    runs = tmp_path / "tool_runs.jsonl"
    out = tmp_path / "cont.jsonl"
    _write_cases(cases)
    _write_tool_runs(runs)

    client = _ScriptedClient({
        "p_search_01": "Great -- try Chopin's Ballade No. 1 for that singing tone.",
    })
    stats1 = run_continuation(
        tool_runs_path=runs, cases_path=cases,
        teacher_client=client, out_path=out,
    )
    assert stats1.processed == 1 and stats1.skipped == 0

    # Second run with a fresh client; p_search_01 is already done.
    stats2 = run_continuation(
        tool_runs_path=runs, cases_path=cases,
        teacher_client=_ScriptedClient({}), out_path=out,
    )
    assert stats2.processed == 0 and stats2.skipped == 1
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1, "must not append a duplicate row"
