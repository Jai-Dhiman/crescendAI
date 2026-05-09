"""Pipeline B runner: scoring + per-row category propagation + resume."""
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.stage0.run_tool_probe import run as run_tool_probe

_SCHEMAS = {
    "search_catalog": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    "create_exercise": {
        "type": "object",
        "properties": {"skill": {"type": "string"}, "exercises": {"type": "array"}},
        "required": ["skill", "exercises"],
    },
}


class _ScriptedClient:
    def __init__(self, script: dict[str, str]) -> None:
        self.script = script
        self.model = "fake/qwen"
        self.calls = 0

    def complete(self, user: str, system: str = "", max_tokens: int = 0) -> str:
        self.calls += 1
        # The user prompt embeds the case_id in <case_id>...</case_id>.
        import re
        m = re.search(r"<case_id>(.+?)</case_id>", user)
        if not m:
            return ""
        return self.script.get(m.group(1), "")


def _write_cases(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"case_id": "p_search_01", "expected_call": True, "expected_tool": "search_catalog", "category": None,
         "briefing": {"prompt": "student: 'find me Chopin pieces'"}},
        {"case_id": "n_chitchat_01", "expected_call": False, "expected_tool": None, "category": "chitchat",
         "briefing": {"prompt": "student: 'thanks!'"}},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_runner_scores_each_case_and_propagates_category(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases)
    sys_prompt = tmp_path / "tool_probe_system.txt"
    sys_prompt.write_text("system\n<schemas/>")
    client = _ScriptedClient({
        "p_search_01": '<tool_call>{"name":"search_catalog","arguments":{"query":"Chopin"}}</tool_call>',
        "n_chitchat_01": "You're welcome!",
    })
    out = tmp_path / "tool_runs.jsonl"
    stats = run_tool_probe(
        cases_path=cases, system_prompt_path=sys_prompt,
        schemas=_SCHEMAS, teacher_client=client, out_path=out,
    )
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 2
    p = next(r for r in rows if r["case_id"] == "p_search_01")
    n = next(r for r in rows if r["case_id"] == "n_chitchat_01")
    assert p["called"] is True and p["discipline_correct"] is True
    assert n["called"] is False and n["category"] == "chitchat" and n["discipline_correct"] is True
    assert stats.processed == 2 and stats.errors == 0


def test_runner_resumes_skipping_completed(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases)
    sys_prompt = tmp_path / "tool_probe_system.txt"
    sys_prompt.write_text("x")
    client = _ScriptedClient({
        "p_search_01": '<tool_call>{"name":"search_catalog","arguments":{"query":"Chopin"}}</tool_call>',
        "n_chitchat_01": "ok",
    })
    out = tmp_path / "tool_runs.jsonl"
    run_tool_probe(cases_path=cases, system_prompt_path=sys_prompt, schemas=_SCHEMAS,
                   teacher_client=client, out_path=out)
    assert client.calls == 2
    stats = run_tool_probe(cases_path=cases, system_prompt_path=sys_prompt, schemas=_SCHEMAS,
                          teacher_client=_ScriptedClient({}), out_path=out)
    assert stats.processed == 0 and stats.skipped == 2


def test_errored_rows_are_not_re_appended_on_retry(tmp_path: Path) -> None:
    """Errored rows stay present and are NOT duplicated on retry."""
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases)
    sys_prompt = tmp_path / "tool_probe_system.txt"
    sys_prompt.write_text("x")
    out = tmp_path / "tool_runs.jsonl"

    class _FailingClient:
        model = "fake/qwen"

        def complete(self, user: str, system: str = "", max_tokens: int = 0) -> str:
            raise RuntimeError("transient upstream failure")

    run_tool_probe(
        cases_path=cases, system_prompt_path=sys_prompt, schemas=_SCHEMAS,
        teacher_client=_FailingClient(), out_path=out,
    )
    rows_after_first = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows_after_first) == 2
    assert all(r["error"] for r in rows_after_first)

    # Retry with a healthy client: errored rows must NOT be re-processed,
    # and the file must remain at 2 rows (no duplicates).
    healthy = _ScriptedClient({
        "p_search_01": '<tool_call>{"name":"search_catalog","arguments":{"query":"Chopin"}}</tool_call>',
        "n_chitchat_01": "ok",
    })
    stats = run_tool_probe(
        cases_path=cases, system_prompt_path=sys_prompt, schemas=_SCHEMAS,
        teacher_client=healthy, out_path=out,
    )
    rows_after_second = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows_after_second) == 2
    assert stats.processed == 0 and stats.skipped == 2
