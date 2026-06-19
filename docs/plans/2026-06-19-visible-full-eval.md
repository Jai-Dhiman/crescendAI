# Visible Full Local Eval Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** A single `just e2e-full-session` command runs a full local eval session in a headed browser, seeds canary memory facts, confirms recall in live chat, and prints a per-criterion PASS/FAIL report.
**Spec:** docs/specs/2026-06-19-visible-full-eval-design.md
**Style:** Follow CLAUDE.md (bun not npm; uv not pip; explicit exceptions not fallbacks; no emojis; surgical additive changes only; `bun run test` not `bun test`).

---

## Task Groups

```
Group A (parallel): Task 1, Task 2
Group B (sequential, depends on A): Task 3
Group C (sequential, depends on B): Task 4
```

- **Group A** [SHIPS INDEPENDENTLY]: offline unit-testable building blocks — SQL builder in `memory_seeder.py` and the `assistant-message` testid in `ChatMessages.tsx`.
- **Group B**: `run_chat_turns()` in `ui_verifier.py` (depends on the `assistant-message` testid being in place from Task 2).
- **Group C**: `e2e_full_session.py` orchestrator + `just e2e-full-session` recipe (depends on all modules existing).

---

## Task 1: `memory_seeder.py` — canary fact SQL builder (offline unit-testable)

**Group:** A (parallel with Task 2)

**Behavior being verified:** `build_insert_rows(student_id, tokens)` returns a list of dicts with the required non-null columns, one per token, using the `CANARY_` prefix convention.

**Interface under test:** `memory_seeder.build_insert_rows(student_id: str, tokens: list[str]) -> list[dict]`

**Files:**
- Create: `apps/evals/memory_seeder.py`
- Create: `apps/evals/tests/test_memory_seeder.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_memory_seeder.py
"""Offline unit tests for memory_seeder — no DB or live services required."""
from __future__ import annotations

import pytest


def test_build_insert_rows_returns_one_row_per_token():
    from memory_seeder import build_insert_rows

    rows = build_insert_rows(
        student_id="student-abc",
        tokens=["CANARY_RACHMANINOFF_ETUDE", "CANARY_LEFT_HAND_WEAKNESS"],
    )

    assert len(rows) == 2
    for row in rows:
        assert row["student_id"] == "student-abc"
        assert row["fact_text"].startswith("CANARY_")
        assert row["fact_type"] in ("technical_observation", "repertoire_context", "student_goal")
        assert "valid_at" in row
        assert row["confidence"] in ("high", "medium", "low")
        assert row["evidence"]
        assert row["source_type"]


def test_build_insert_rows_embeds_token_in_fact_text():
    from memory_seeder import build_insert_rows

    rows = build_insert_rows(
        student_id="s1",
        tokens=["CANARY_RACHMANINOFF_ETUDE", "CANARY_LEFT_HAND_WEAKNESS"],
    )

    texts = [r["fact_text"] for r in rows]
    assert any("CANARY_RACHMANINOFF_ETUDE" in t for t in texts)
    assert any("CANARY_LEFT_HAND_WEAKNESS" in t for t in texts)


def test_build_insert_rows_required_columns_present():
    from memory_seeder import build_insert_rows

    required = {
        "student_id", "fact_text", "fact_type",
        "valid_at", "confidence", "evidence", "source_type",
    }
    rows = build_insert_rows("s1", ["CANARY_X"])
    assert required.issubset(rows[0].keys())
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest tests/test_memory_seeder.py -v 2>&1 | head -30
```

Expected: FAIL — `ModuleNotFoundError: No module named 'memory_seeder'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/memory_seeder.py
"""Seed and clean up canary synthesized_facts for the debug user.

Interface:
    seed_canary_facts(student_id, db_dsn, wrangler_url) -> CanarySeed
    cleanup_canary_facts(student_id, db_dsn) -> int  (rows deleted)
    build_insert_rows(student_id, tokens) -> list[dict]  (offline-testable)

Hides: psycopg2 connection lifecycle, INSERT SQL, DELETE-on-prefix cleanup,
       student_id resolution via /api/auth/debug.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Canary prefix convention — seeder matches this for cleanup.
CANARY_PREFIX = "CANARY_"

# Fixed canary token pair used by e2e_full_session.
CANARY_TOKENS = [
    "CANARY_RACHMANINOFF_ETUDE",
    "CANARY_LEFT_HAND_WEAKNESS",
]

# Fact template: each token is embedded in a human-readable sentence so the
# teacher LLM sees it as meaningful context and is likely to echo it.
_TOKEN_TEMPLATES = {
    "CANARY_RACHMANINOFF_ETUDE": (
        "Student is currently preparing the CANARY_RACHMANINOFF_ETUDE passage "
        "and struggles with maintaining tempo under pressure.",
        "repertoire_context",
    ),
    "CANARY_LEFT_HAND_WEAKNESS": (
        "Persistent CANARY_LEFT_HAND_WEAKNESS observed across three sessions: "
        "left hand consistently trails right by 30-50ms in fast passages.",
        "technical_observation",
    ),
}

_DEFAULT_TEMPLATE = (
    "Student note: {token} — a canary marker for automated recall verification.",
    "student_goal",
)


@dataclass
class CanarySeed:
    """Outcome of a successful seed operation."""
    student_id: str
    tokens: list[str] = field(default_factory=list)
    rows_inserted: int = 0


def build_insert_rows(student_id: str, tokens: list[str]) -> list[dict[str, Any]]:
    """Build INSERT row dicts for synthesized_facts — no DB required.

    Each row contains all required non-null columns. The token string is
    embedded verbatim in fact_text so keyword-search assertions work.

    Args:
        student_id: The student UUID to seed facts for.
        tokens: List of canary token strings (must contain the token literally).

    Returns:
        List of dicts ready for psycopg2 executemany (column -> value).
    """
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []

    for token in tokens:
        if token in _TOKEN_TEMPLATES:
            fact_text, fact_type = _TOKEN_TEMPLATES[token]
        else:
            template, fact_type = _DEFAULT_TEMPLATE
            fact_text = template.format(token=token)

        rows.append({
            "id": str(uuid.uuid4()),
            "student_id": student_id,
            "fact_text": fact_text,
            "fact_type": fact_type,
            "valid_at": now,
            "confidence": "high",
            "evidence": f"Automated canary seed for e2e recall verification ({token})",
            "source_type": "eval_seed",
        })

    return rows


def seed_canary_facts(
    student_id: str,
    db_dsn: str,
    tokens: list[str] | None = None,
) -> CanarySeed:
    """Insert canary synthesized_facts into crescendai_dev for student_id.

    Removes any previous canary rows (fact_text LIKE 'CANARY_%') for this
    student first to avoid accumulating stale rows across runs.

    Args:
        student_id: Student UUID (stable for debug@crescend.ai across restarts).
        db_dsn: PostgreSQL DSN, e.g. postgresql://jdhiman:postgres@localhost:5432/crescendai_dev
        tokens: Canary token strings to seed (defaults to CANARY_TOKENS).

    Returns:
        CanarySeed with student_id, tokens, and rows_inserted count.

    Raises:
        RuntimeError: If psycopg2 connection or INSERT fails.
    """
    import psycopg2  # type: ignore[import]

    effective_tokens = tokens if tokens is not None else CANARY_TOKENS

    try:
        conn = psycopg2.connect(db_dsn)
    except Exception as exc:
        raise RuntimeError(f"Cannot connect to DB at {db_dsn}: {exc}") from exc

    try:
        with conn:
            with conn.cursor() as cur:
                # Remove stale canary rows for this student.
                cur.execute(
                    "DELETE FROM synthesized_facts "
                    "WHERE student_id = %s AND fact_text LIKE %s",
                    (student_id, f"{CANARY_PREFIX}%"),
                )

                # Insert fresh rows.
                rows = build_insert_rows(student_id, effective_tokens)
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO synthesized_facts
                            (id, student_id, fact_text, fact_type,
                             valid_at, confidence, evidence, source_type)
                        VALUES
                            (%(id)s, %(student_id)s, %(fact_text)s, %(fact_type)s,
                             %(valid_at)s, %(confidence)s, %(evidence)s, %(source_type)s)
                        """,
                        row,
                    )
    except Exception as exc:
        raise RuntimeError(f"Failed to seed canary facts: {exc}") from exc
    finally:
        conn.close()

    return CanarySeed(
        student_id=student_id,
        tokens=effective_tokens,
        rows_inserted=len(effective_tokens),
    )


def cleanup_canary_facts(student_id: str, db_dsn: str) -> int:
    """Delete all canary rows for student_id. Returns count deleted.

    Raises:
        RuntimeError: If DB connection or DELETE fails.
    """
    import psycopg2  # type: ignore[import]

    try:
        conn = psycopg2.connect(db_dsn)
    except Exception as exc:
        raise RuntimeError(f"Cannot connect to DB at {db_dsn}: {exc}") from exc

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM synthesized_facts "
                    "WHERE student_id = %s AND fact_text LIKE %s",
                    (student_id, f"{CANARY_PREFIX}%"),
                )
                return cur.rowcount
    except Exception as exc:
        raise RuntimeError(f"Failed to cleanup canary facts: {exc}") from exc
    finally:
        conn.close()


def get_debug_student_id(wrangler_url: str) -> str:
    """Fetch the debug user's student_id from the API.

    Calls POST /api/auth/debug and reads the student_id from the response body.
    The debug user is stable across wrangler dev restarts for crescendai_dev.

    Raises:
        RuntimeError: If the API is unreachable or returns a non-200 status.
    """
    import requests

    try:
        resp = requests.post(f"{wrangler_url}/api/auth/debug", timeout=10)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"API not reachable at {wrangler_url}. Run `just dev` first."
        ) from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"POST /api/auth/debug returned {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    student_id = data.get("user", {}).get("id") or data.get("studentId") or data.get("id")
    if not student_id:
        raise RuntimeError(
            f"Could not extract student_id from /api/auth/debug response: {data}"
        )
    return student_id
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest tests/test_memory_seeder.py -v 2>&1 | tail -15
```

Expected: PASS — all 3 tests green.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-70-visible-full-eval && git add apps/evals/memory_seeder.py apps/evals/tests/test_memory_seeder.py && git commit -m "feat(evals): memory_seeder — canary fact SQL builder + offline unit tests (#70)"
```

---

## Task 2: `ChatMessages.tsx` — add `data-testid="assistant-message"` to regular assistant bubbles

**Group:** A (parallel with Task 1)

**Behavior being verified:** A regular assistant chat message (role=assistant, messageType not "synthesis") renders a `[data-testid=assistant-message]` element; synthesis messages keep their existing `synthesis-message` testid.

**Interface under test:** `ChatMessages` React component render output — queried via `screen.getByTestId("assistant-message")`.

**Files:**
- Modify: `apps/web/src/components/ChatMessages.tsx`
- Modify: `apps/web/src/components/ChatMessages.test.tsx`

- [ ] **Step 1: Write the failing test**

Add this describe block at the end of `apps/web/src/components/ChatMessages.test.tsx`:

```typescript
describe("ChatMessages — assistant-message testid", () => {
  it("renders data-testid=assistant-message on a plain assistant reply", async () => {
    vi.resetModules();
    vi.doMock("./Artifact", () => ({ Artifact: () => null }));
    vi.doMock("./MessageContent", () => ({
      MessageContent: ({ content }: { content: string }) =>
        React.createElement("div", { "data-testid": "message-content" }, content),
    }));
    vi.doMock("./ToolCallBar", () => ({ ToolCallBar: () => null }));

    const { ChatMessages } = await import("./ChatMessages");
    const message = {
      id: "msg-assistant",
      role: "assistant" as const,
      content: "Here is my feedback on your playing.",
      createdAt: new Date().toISOString(),
      // No messageType — plain assistant reply, not synthesis
    };
    render(React.createElement(ChatMessages, { messages: [message] }));
    expect(document.querySelector("[data-testid='assistant-message']")).not.toBeNull();
  });

  it("does NOT add assistant-message testid to a synthesis message", async () => {
    vi.resetModules();
    vi.doMock("./Artifact", () => ({ Artifact: () => null }));
    vi.doMock("./MessageContent", () => ({
      MessageContent: ({ content }: { content: string }) =>
        React.createElement("div", { "data-testid": "message-content" }, content),
    }));
    vi.doMock("./ToolCallBar", () => ({ ToolCallBar: () => null }));

    const { ChatMessages } = await import("./ChatMessages");
    const message = {
      id: "msg-synth",
      role: "assistant" as const,
      content: "Your phrasing needs work.",
      createdAt: new Date().toISOString(),
      messageType: "synthesis" as const,
      // No components — so ReflectionMessage branch is not taken
    };
    render(React.createElement(ChatMessages, { messages: [message] }));
    // synthesis-message testid should be present
    expect(document.querySelector("[data-testid='synthesis-message']")).not.toBeNull();
    // assistant-message testid must NOT be present on a synthesis message
    expect(document.querySelector("[data-testid='assistant-message']")).toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test -- --reporter=verbose src/components/ChatMessages.test.tsx 2>&1 | tail -20
```

Expected: FAIL — `expected null not to be null` (assistant-message testid absent).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/components/ChatMessages.tsx`, locate the final `return` inside `MessageBubble` (the non-synthesis, non-session-lifecycle, non-user branch) at line 228. Change the outer `<div>` from:

```typescript
	return (
		<div className="flex justify-start animate-fade-in" data-testid={message.messageType === "synthesis" ? "synthesis-message" : undefined}>
```

to:

```typescript
	return (
		<div
			className="flex justify-start animate-fade-in"
			data-testid={
				message.messageType === "synthesis"
					? "synthesis-message"
					: message.role === "assistant"
						? "assistant-message"
						: undefined
			}
		>
```

No other lines change. This is additive: synthesis messages keep `synthesis-message`; assistant non-synthesis messages get `assistant-message`; user messages and dividers stay untagged.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test -- --reporter=verbose src/components/ChatMessages.test.tsx 2>&1 | tail -20
```

Expected: PASS — all tests in ChatMessages.test.tsx green (existing 3 + new 2 = 5 total).

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-70-visible-full-eval && git add apps/web/src/components/ChatMessages.tsx apps/web/src/components/ChatMessages.test.tsx && git commit -m "feat(web): add data-testid=assistant-message to regular assistant bubbles (#70)"
```

---

## Task 3: `ui_verifier.py` — add `run_chat_turns()` (depends on Group A)

**Group:** B (sequential, depends on Task 2 for the `assistant-message` testid)

**Behavior being verified:** `run_chat_turns` drives a list of text turns through the chat UI: fills the textarea, presses Enter, waits (bounded) for a new assistant-message to appear with stable text, and returns the reply texts.

**Interface under test:** `run_chat_turns(page, turns, reply_timeout_ms) -> list[ChatTurnResult]`

**Files:**
- Modify: `apps/evals/ui_verifier.py`

Note: `run_chat_turns` takes a live Playwright `Page` object. The contract test below uses a minimal stub `Page` to verify the function's argument handling and return type without requiring a browser. Live behavior is verified only during the full e2e run.

- [ ] **Step 1: Write the failing test**

Add `apps/evals/tests/test_run_chat_turns.py`:

```python
"""Offline contract test for run_chat_turns() — no live browser required.

Verifies that run_chat_turns() returns one ChatTurnResult per input turn with
the expected field names, using a lightweight stub Page that simulates a new
assistant-message appearing after each Enter press.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch


def _make_stub_page(reply_texts: list[str]):
    """Build a minimal Page stub that simulates new assistant-message elements appearing."""
    page = MagicMock()

    # Simulate textarea fill + Enter
    page.fill = MagicMock()
    page.press = MagicMock()

    # Track call count for query_selector_all to simulate reply appearing
    call_counts: dict[str, int] = {}

    def _query_all(selector: str):
        call_counts[selector] = call_counts.get(selector, 0) + 1
        count = call_counts[selector]
        # After enough polls, return one element per turn that has been "sent"
        num_replies = min(count // 2, len(reply_texts))
        elements = []
        for i in range(num_replies):
            el = MagicMock()
            el.inner_text.return_value = reply_texts[i]
            el.get_attribute.return_value = None  # no streaming class
            elements.append(el)
        return elements

    page.query_selector_all = _query_all
    page.wait_for_selector = MagicMock()
    return page


def test_run_chat_turns_returns_one_result_per_turn():
    from ui_verifier import run_chat_turns

    stub_page = _make_stub_page(["Reply to hello.", "Reply about music."])

    results = run_chat_turns(
        page=stub_page,
        turns=["hello", "tell me about music"],
        reply_timeout_ms=5000,
    )

    assert len(results) == 2


def test_run_chat_turns_result_has_required_fields():
    from ui_verifier import run_chat_turns, ChatTurnResult

    stub_page = _make_stub_page(["Reply one.", "Reply two."])

    results = run_chat_turns(
        page=stub_page,
        turns=["turn one", "turn two"],
        reply_timeout_ms=5000,
    )

    for r in results:
        assert isinstance(r, ChatTurnResult)
        assert isinstance(r.turn_text, str)
        assert isinstance(r.reply_text, str)
        assert isinstance(r.elapsed_ms, int)
        assert r.elapsed_ms >= 0


def test_run_chat_turns_fills_textarea_for_each_turn():
    from ui_verifier import run_chat_turns

    stub_page = _make_stub_page(["r1", "r2"])

    run_chat_turns(
        page=stub_page,
        turns=["hello", "goodbye"],
        reply_timeout_ms=5000,
    )

    # fill called once per turn with the textarea selector and turn text
    assert stub_page.fill.call_count == 2
    fill_texts = [c.args[1] for c in stub_page.fill.call_args_list]
    assert "hello" in fill_texts
    assert "goodbye" in fill_texts
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest tests/test_run_chat_turns.py -v 2>&1 | head -20
```

Expected: FAIL — `ImportError: cannot import name 'run_chat_turns' from 'ui_verifier'` or `ImportError: cannot import name 'ChatTurnResult' from 'ui_verifier'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `apps/evals/ui_verifier.py` (after the existing imports and before `VerificationResult`):

```python
import time as _time

@dataclass
class ChatTurnResult:
    """Result of a single scripted chat turn."""
    turn_text: str
    reply_text: str
    elapsed_ms: int
```

Then add the function after the existing `_save_screenshot` helper:

```python
_ASSISTANT_MSG_SELECTOR = "[data-testid='assistant-message']"
_CHAT_TEXTAREA_SELECTOR = "textarea"
# Wait for the streaming class to disappear and text to stabilise for this many ms.
_STABILITY_WAIT_MS = 600
_POLL_INTERVAL_MS = 250


def run_chat_turns(
    page: "Page",
    turns: list[str],
    reply_timeout_ms: int = 45000,
) -> list[ChatTurnResult]:
    """Drive scripted chat turns through the web UI and collect reply texts.

    For each turn:
      1. Fill the chat textarea with the turn text.
      2. Press Enter.
      3. Poll (bounded by reply_timeout_ms) until a new assistant-message
         element appears whose inner_text has been stable for _STABILITY_WAIT_MS.
      4. Record the reply text and elapsed time.

    The textarea selector is ``textarea`` (ChatInput renders exactly one).
    Assistant replies carry ``data-testid="assistant-message"`` (added in Task 2).

    Args:
        page: Playwright Page object, already on the conversation URL.
        turns: List of message texts to send in order.
        reply_timeout_ms: Hard upper bound per turn (raises RuntimeError on breach).

    Returns:
        List of ChatTurnResult, one per turn, in order.

    Raises:
        RuntimeError: If a reply does not arrive within reply_timeout_ms for any turn.
    """
    results: list[ChatTurnResult] = []
    baseline_count = len(page.query_selector_all(_ASSISTANT_MSG_SELECTOR))

    for turn_text in turns:
        # Fill textarea and submit.
        page.fill(_CHAT_TEXTAREA_SELECTOR, turn_text)
        page.press(_CHAT_TEXTAREA_SELECTOR, "Enter")

        start_ms = int(_time.monotonic() * 1000)
        deadline_ms = start_ms + reply_timeout_ms
        expected_count = baseline_count + 1
        last_text = ""
        stable_since_ms: int | None = None

        while True:
            now_ms = int(_time.monotonic() * 1000)
            if now_ms >= deadline_ms:
                raise RuntimeError(
                    f"Chat turn timed out after {reply_timeout_ms}ms waiting for reply to: "
                    f"{turn_text!r}"
                )

            elements = page.query_selector_all(_ASSISTANT_MSG_SELECTOR)
            if len(elements) >= expected_count:
                # Read the newest element's text.
                latest = elements[-1]
                current_text = (latest.inner_text() or "").strip()

                if current_text and current_text == last_text:
                    # Text has not changed — check if stable for enough time.
                    if stable_since_ms is None:
                        stable_since_ms = now_ms
                    elif now_ms - stable_since_ms >= _STABILITY_WAIT_MS:
                        # Stable and non-empty: turn complete.
                        elapsed = now_ms - start_ms
                        results.append(ChatTurnResult(
                            turn_text=turn_text,
                            reply_text=current_text,
                            elapsed_ms=elapsed,
                        ))
                        baseline_count = len(elements)
                        break
                else:
                    # Text changed — reset stability clock.
                    last_text = current_text
                    stable_since_ms = None

            _time.sleep(_POLL_INTERVAL_MS / 1000)

    return results
```

Also add `ChatTurnResult` to the module's public name if an `__all__` is added (no `__all__` currently, so no change needed).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest tests/test_run_chat_turns.py -v 2>&1 | tail -15
```

Expected: PASS — all 3 tests green.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-70-visible-full-eval && git add apps/evals/ui_verifier.py apps/evals/tests/test_run_chat_turns.py && git commit -m "feat(evals): ui_verifier.run_chat_turns — bounded scripted chat turn driver (#70)"
```

---

## Task 4: `e2e_full_session.py` orchestrator + `just e2e-full-session` recipe (depends on Group B)

**Group:** C (sequential, depends on all of Group A + Group B)

**Behavior being verified:** `FullSessionReport` correctly reflects the per-criterion outcome flags, and `format_report()` produces a non-empty string containing "PASS", "FAIL", or "SKIP" per criterion — verifiable offline without live services.

**Interface under test:** `FullSessionReport` dataclass + `format_report(report) -> str`; the full `run()` function is live-gated only.

**Files:**
- Create: `apps/evals/e2e_full_session.py`
- Modify: `justfile`
- Create: `apps/evals/tests/test_full_session_report.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_full_session_report.py
"""Offline unit tests for FullSessionReport and format_report() — no live services."""
from __future__ import annotations


def test_format_report_contains_all_criteria_labels():
    from e2e_full_session import FullSessionReport, format_report

    report = FullSessionReport(
        conversation_id="conv-abc",
        criteria_a=True,
        criteria_b_headline=True,
        criteria_b_components=True,
        criteria_c=None,
        criteria_e_recall=True,
        criteria_e_tokens_found=["CANARY_RACHMANINOFF_ETUDE"],
        criteria_e_tokens_missing=[],
        criteria_f_tool_outcome="TEXT_ONLY",
        errors=[],
    )

    text = format_report(report)

    assert "(a)" in text
    assert "(b)" in text
    assert "(e)" in text
    assert "(f)" in text
    assert "PASS" in text
    assert "TEXT_ONLY" in text


def test_format_report_shows_skip_for_none_criteria_c():
    from e2e_full_session import FullSessionReport, format_report

    report = FullSessionReport(
        conversation_id="conv-xyz",
        criteria_a=True,
        criteria_b_headline=True,
        criteria_b_components=True,
        criteria_c=None,
        criteria_e_recall=True,
        criteria_e_tokens_found=["T1"],
        criteria_e_tokens_missing=[],
        criteria_f_tool_outcome="TOOL_RENDERED",
        errors=[],
    )

    text = format_report(report)
    assert "SKIP" in text  # criteria_c is None -> SKIP


def test_full_session_report_overall_false_when_criteria_e_fails():
    from e2e_full_session import FullSessionReport

    report = FullSessionReport(
        conversation_id="conv-fail",
        criteria_a=True,
        criteria_b_headline=True,
        criteria_b_components=True,
        criteria_c=None,
        criteria_e_recall=False,   # recall failed
        criteria_e_tokens_found=[],
        criteria_e_tokens_missing=["CANARY_RACHMANINOFF_ETUDE", "CANARY_LEFT_HAND_WEAKNESS"],
        criteria_f_tool_outcome="TEXT_ONLY",
        errors=[],
    )

    assert report.overall is False


def test_full_session_report_overall_true_when_all_pass():
    from e2e_full_session import FullSessionReport

    report = FullSessionReport(
        conversation_id="conv-ok",
        criteria_a=True,
        criteria_b_headline=True,
        criteria_b_components=True,
        criteria_c=None,
        criteria_e_recall=True,
        criteria_e_tokens_found=["T1", "T2"],
        criteria_e_tokens_missing=[],
        criteria_f_tool_outcome="TEXT_ONLY",
        errors=[],
    )

    assert report.overall is True
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest tests/test_full_session_report.py -v 2>&1 | head -20
```

Expected: FAIL — `ModuleNotFoundError: No module named 'e2e_full_session'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# apps/evals/e2e_full_session.py
"""Full-session visible eval orchestrator (issue #70).

Runs a complete local session end-to-end with a headed browser and scripted
chat turns that verify memory recall. Prints a per-criterion PASS/FAIL report.

Usage:
    cd apps/evals
    uv run python -m e2e_full_session [--recording <wav>] [--piece-slug <slug>]

Key flags:
    --headless              Run browser headlessly (default: False — visible window)
    --conversation-id ID    Skip drive_persisted(); use existing conversation
    --no-seed               Skip canary fact seeding
    --max-chunks N          Max WebM chunks (default: 6)
    --timeout SECS          Per-event WS timeout (default: 120)
    --screenshot-dir PATH   Dir to save per-phase screenshots (default: /tmp/e2e-full)
    --wrangler-url URL      API URL (default: http://localhost:8787)
    --web-url URL           Web URL (default: http://localhost:3000)
    --db-dsn DSN            Postgres DSN for seeding (default: postgresql://jdhiman:postgres@localhost:5432/crescendai_dev)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Report types (offline-testable — no imports from live modules at module level)
# ---------------------------------------------------------------------------

ToolOutcome = Literal["TOOL_RENDERED", "TEXT_ONLY", "UNKNOWN"]


@dataclass
class FullSessionReport:
    """Per-criterion outcomes for the full session eval run."""
    conversation_id: str
    # (a) V6 artifact rendered (synthesis-message in DOM)
    criteria_a: bool
    # (b) Headline match + components
    criteria_b_headline: bool
    criteria_b_components: bool
    # (c) Exercise confirm flow (None = no prescription)
    criteria_c: bool | None
    # (e) Memory recall — canary tokens found in chat reply
    criteria_e_recall: bool
    criteria_e_tokens_found: list[str]
    criteria_e_tokens_missing: list[str]
    # (f) Tool action outcome (non-fatal)
    criteria_f_tool_outcome: ToolOutcome
    errors: list[str] = field(default_factory=list)

    @property
    def overall(self) -> bool:
        """PASS iff (a), (b-headline), (b-components), and (e) all pass; (f) is non-fatal."""
        if self.errors:
            return False
        return (
            self.criteria_a
            and self.criteria_b_headline
            and self.criteria_b_components
            and self.criteria_e_recall
        )


def format_report(report: FullSessionReport) -> str:
    """Render a human-readable per-criterion report string."""
    def _yn(v: bool | None, skip_label: str = "SKIP") -> str:
        if v is None:
            return skip_label
        return "PASS" if v else "FAIL"

    lines = [
        "",
        "=" * 65,
        "FULL SESSION EVAL REPORT",
        "=" * 65,
        f"Conversation: {report.conversation_id}",
        "",
        f"(a) V6 artifact rendered (synthesis in DOM):     {_yn(report.criteria_a)}",
        f"(b) Headline match (DOM == WS text):             {_yn(report.criteria_b_headline)}",
        f"(b) Components rendered (cards present):         {_yn(report.criteria_b_components)}",
    ]

    if report.criteria_c is None:
        lines.append("(c) Exercise confirm flow:                       SKIP (no prescription)")
    else:
        lines.append(f"(c) Exercise confirm flow:                       {_yn(report.criteria_c)}")

    recall_detail = ""
    if report.criteria_e_tokens_found:
        recall_detail = f" found={report.criteria_e_tokens_found}"
    if report.criteria_e_tokens_missing:
        recall_detail += f" missing={report.criteria_e_tokens_missing}"
    lines.append(
        f"(e) Memory recall — canary tokens in reply:      {_yn(report.criteria_e_recall)}{recall_detail}"
    )
    lines.append(
        f"(f) Tool action (ExerciseSetCard):               {report.criteria_f_tool_outcome} (non-fatal)"
    )

    if report.errors:
        lines.append("")
        lines.append("Errors:")
        for err in report.errors:
            lines.append(f"  - {err}")

    lines.append("")
    lines.append(f"OVERALL: {'PASS' if report.overall else 'FAIL'}")
    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat turn scripts
# ---------------------------------------------------------------------------

RECALL_TURN = "What do you know about me and what am I currently preparing?"
TOOL_TURN = "Give me a left-hand drill for bars 1-4."

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDING = (
    REPO_ROOT
    / "model"
    / "data"
    / "evals"
    / "practice_eval"
    / "nocturne_op9no2"
    / "audio"
    / "_aySCutsVVQ.wav"
)
DEFAULT_PIECE_SLUG = "nocturne_op9no2"
DEFAULT_API_DIR = REPO_ROOT / "apps" / "api"
DEFAULT_DB_DSN = "postgresql://jdhiman:postgres@localhost:5432/crescendai_dev"
DEFAULT_SCREENSHOT_DIR = Path("/tmp/e2e-full")


def run(
    recording: Path,
    piece_slug: str,
    wrangler_url: str = "http://localhost:8787",
    web_url: str = "http://localhost:3000",
    api_dir: Path | None = None,
    db_dsn: str = DEFAULT_DB_DSN,
    screenshot_dir: Path = DEFAULT_SCREENSHOT_DIR,
    max_chunks: int = 6,
    timeout_per_event: float = 120.0,
    headless: bool = False,
    slow_mo: int = 700,
    conversation_id: str | None = None,
    skip_seed: bool = False,
) -> int:
    """Run the full visible session eval. Returns 0 on PASS, 1 on FAIL."""
    from memory_seeder import (
        CANARY_TOKENS,
        CanarySeed,
        get_debug_student_id,
        seed_canary_facts,
    )
    from shared.local_session import check_services, drive_persisted
    from ui_verifier import (
        VerificationResult,
        _auth_browser,
        _component_testid,
        _save_screenshot,
        run_chat_turns,
        verify_ui,
    )
    from playwright.sync_api import sync_playwright

    effective_api_dir = api_dir if api_dir is not None else DEFAULT_API_DIR
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # --- Pre-flight ---
    print(f"[full-eval] Health check: {wrangler_url}")
    try:
        check_services(wrangler_url)
    except RuntimeError as exc:
        print(f"ERROR: services not ready: {exc}", file=sys.stderr)
        print("Run `just dev` and `just seed-fingerprint` first.", file=sys.stderr)
        return 1

    # --- Step 1: Seed canary facts ---
    canary: CanarySeed | None = None
    if not skip_seed:
        print("[full-eval] Fetching debug student_id from API...")
        try:
            student_id = get_debug_student_id(wrangler_url)
        except RuntimeError as exc:
            print(f"ERROR: cannot get debug student_id: {exc}", file=sys.stderr)
            return 1

        print(f"[full-eval] Seeding canary facts for student_id={student_id}")
        try:
            canary = seed_canary_facts(student_id=student_id, db_dsn=db_dsn)
        except RuntimeError as exc:
            print(f"ERROR: canary seed failed: {exc}", file=sys.stderr)
            return 1
        print(f"[full-eval] Seeded {canary.rows_inserted} canary rows: {canary.tokens}")
    else:
        print("[full-eval] Skipping canary seed (--no-seed).")

    # --- Step 2: Drive recording (or skip if --conversation-id provided) ---
    if conversation_id is None:
        if not recording.exists():
            print(f"ERROR: recording not found: {recording}", file=sys.stderr)
            return 1

        print(f"[full-eval] Driving recording: {recording.name} (piece={piece_slug})")
        try:
            from e2e_ui_session import _lowest_dim
            cap = drive_persisted(
                recording=recording,
                piece_slug=piece_slug,
                wrangler_url=wrangler_url,
                api_dir=effective_api_dir,
                timeout_per_event=timeout_per_event,
                max_chunks=max_chunks,
            )
        except RuntimeError as exc:
            print(f"ERROR: drive_persisted() failed: {exc}", file=sys.stderr)
            return 1

        conversation_id = cap.conversation_id
        expected_headline = cap.headline_text
        component_types = [c.get("type", "") for c in cap.components]
        lowest_dim = _lowest_dim(cap.chunk_scores)
        has_prescription = cap.prescribed_exercise is not None
        is_fallback = cap.is_fallback

        print(f"[full-eval] Session: {cap.session_id}")
        print(f"[full-eval] Conversation: {conversation_id}")
        print(f"[full-eval] Headline: {expected_headline!r}")
        print(f"[full-eval] is_fallback: {is_fallback}")

        if is_fallback:
            print("ERROR: synthesis is_fallback=true — V6 artifact not produced.", file=sys.stderr)
            return 1
    else:
        print(f"[full-eval] Using existing conversation_id: {conversation_id} (skipping recording)")
        expected_headline = ""
        component_types = []
        lowest_dim = None
        has_prescription = False

    # --- Step 3: Open headed browser, run synthesis assertions + chat turns ---
    errors: list[str] = []
    criteria_a = False
    criteria_b_headline = False
    criteria_b_components = False
    criteria_c: bool | None = None
    criteria_e_recall = False
    tokens_found: list[str] = []
    tokens_missing: list[str] = []
    criteria_f_outcome: ToolOutcome = "UNKNOWN"

    nav_timeout_ms = 15000
    synthesis_timeout_ms = 30000
    reply_timeout_ms = 45000

    print(f"[full-eval] Opening {'headed' if not headless else 'headless'} browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, slow_mo=slow_mo if not headless else 0)
        context = browser.new_context()
        page = context.new_page()

        try:
            # Authenticate
            try:
                _auth_browser(page, wrangler_url)
            except RuntimeError as exc:
                errors.append(f"Auth failed: {exc}")
                return _finish(
                    conversation_id, criteria_a, criteria_b_headline, criteria_b_components,
                    criteria_c, criteria_e_recall, tokens_found, tokens_missing,
                    criteria_f_outcome, errors, page, screenshot_dir,
                )

            # Navigate to conversation
            conv_url = f"{web_url}/app/c/{conversation_id}"
            print(f"[full-eval] Navigating to: {conv_url}")
            try:
                page.goto(conv_url, timeout=nav_timeout_ms)
            except Exception as exc:
                errors.append(f"Navigation failed: {exc}")
                return _finish(
                    conversation_id, criteria_a, criteria_b_headline, criteria_b_components,
                    criteria_c, criteria_e_recall, tokens_found, tokens_missing,
                    criteria_f_outcome, errors, page, screenshot_dir,
                )

            # Wait for synthesis message (criterion a)
            if expected_headline:
                try:
                    page.wait_for_selector("[data-testid='synthesis-message']", timeout=synthesis_timeout_ms)
                    criteria_a = True
                except Exception as exc:
                    errors.append(f"synthesis-message not found within {synthesis_timeout_ms}ms: {exc}")

                if criteria_a:
                    # Criterion b-headline
                    hl_el = page.query_selector("[data-testid='synthesis-headline']")
                    if hl_el:
                        dom_text = (hl_el.inner_text() or "").strip()
                        criteria_b_headline = dom_text == expected_headline.strip()
                        if not criteria_b_headline:
                            errors.append(
                                f"Headline mismatch.\n  Expected: {expected_headline!r}\n  Got: {dom_text!r}"
                            )
                    else:
                        errors.append("synthesis-headline not found")

                    # Criterion b-components
                    criteria_b_components = True
                    renderable = [t for t in component_types if t not in ("pending_exercise", "search_catalog_result")]
                    for ctype in renderable:
                        testid = _component_testid(ctype)
                        if testid and not page.query_selector(f"[data-testid='{testid}']"):
                            errors.append(f"Component card not found: type={ctype} testid={testid}")
                            criteria_b_components = False

                    # Criterion c — exercise confirm flow
                    if has_prescription:
                        btn = page.query_selector("[data-testid='confirm-exercise-button']")
                        if btn:
                            try:
                                btn.click()
                                page.wait_for_selector("[data-testid='exercise-set-card']", timeout=10000)
                                criteria_c = True
                            except Exception as exc:
                                errors.append(f"Confirm flow failed: {exc}")
                                criteria_c = False
                        else:
                            errors.append("confirm-exercise-button not found (prescription expected)")
                            criteria_c = False
            else:
                # No expected headline (--conversation-id path) — skip a/b/c checks
                criteria_a = True
                criteria_b_headline = True
                criteria_b_components = True

            # Save phase-1 screenshot
            _save_screenshot(page, screenshot_dir / "phase1_synthesis.png")

            # --- Step 4: Scripted chat turns ---
            print(f"[full-eval] Running chat turn (recall): {RECALL_TURN!r}")
            try:
                recall_results = run_chat_turns(
                    page=page,
                    turns=[RECALL_TURN],
                    reply_timeout_ms=reply_timeout_ms,
                )
                recall_reply = recall_results[0].reply_text if recall_results else ""
                print(f"[full-eval] Recall reply ({recall_results[0].elapsed_ms}ms): {recall_reply[:120]!r}...")
            except RuntimeError as exc:
                errors.append(f"Recall turn failed: {exc}")
                recall_reply = ""

            # Check canary tokens
            effective_tokens = canary.tokens if canary else CANARY_TOKENS
            recall_lower = recall_reply.lower()
            for token in effective_tokens:
                if token.lower() in recall_lower:
                    tokens_found.append(token)
                else:
                    tokens_missing.append(token)
            criteria_e_recall = len(tokens_found) > 0 and len(tokens_missing) == 0

            _save_screenshot(page, screenshot_dir / "phase2_recall.png")

            # --- Step 5: Tool action turn ---
            print(f"[full-eval] Running chat turn (tool): {TOOL_TURN!r}")
            try:
                run_chat_turns(
                    page=page,
                    turns=[TOOL_TURN],
                    reply_timeout_ms=reply_timeout_ms,
                )
                tool_card = page.query_selector("[data-testid='exercise-set-card']")
                criteria_f_outcome = "TOOL_RENDERED" if tool_card else "TEXT_ONLY"
                print(f"[full-eval] Tool outcome: {criteria_f_outcome}")
            except RuntimeError as exc:
                errors.append(f"Tool turn failed (non-fatal): {exc}")
                criteria_f_outcome = "UNKNOWN"

            _save_screenshot(page, screenshot_dir / "phase3_tool.png")

        except Exception as exc:
            errors.append(f"Unexpected error: {exc}")
            _save_screenshot(page, screenshot_dir / "phase_error.png")
        finally:
            context.close()
            browser.close()

    return _finish(
        conversation_id, criteria_a, criteria_b_headline, criteria_b_components,
        criteria_c, criteria_e_recall, tokens_found, tokens_missing,
        criteria_f_outcome, errors, None, screenshot_dir,
    )


def _finish(
    conversation_id: str,
    criteria_a: bool,
    criteria_b_headline: bool,
    criteria_b_components: bool,
    criteria_c: bool | None,
    criteria_e_recall: bool,
    tokens_found: list[str],
    tokens_missing: list[str],
    criteria_f_outcome: "ToolOutcome",
    errors: list[str],
    page: object | None,
    screenshot_dir: Path,
) -> int:
    from ui_verifier import _save_screenshot
    if page is not None:
        _save_screenshot(page, screenshot_dir / "phase_error.png")  # type: ignore[arg-type]

    report = FullSessionReport(
        conversation_id=conversation_id,
        criteria_a=criteria_a,
        criteria_b_headline=criteria_b_headline,
        criteria_b_components=criteria_b_components,
        criteria_c=criteria_c,
        criteria_e_recall=criteria_e_recall,
        criteria_e_tokens_found=tokens_found,
        criteria_e_tokens_missing=tokens_missing,
        criteria_f_tool_outcome=criteria_f_outcome,
        errors=errors,
    )
    print(format_report(report))

    if screenshot_dir.exists():
        print(f"Screenshots: {screenshot_dir}/")

    return 0 if report.overall else 1


def _cli() -> None:
    parser = argparse.ArgumentParser(description="CrescendAI full visible session eval")
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--piece-slug", default=DEFAULT_PIECE_SLUG)
    parser.add_argument("--wrangler-url", default="http://localhost:8787")
    parser.add_argument("--web-url", default="http://localhost:3000")
    parser.add_argument("--api-dir", type=Path, default=None)
    parser.add_argument("--db-dsn", default=DEFAULT_DB_DSN, help="Postgres DSN for canary seeding")
    parser.add_argument("--screenshot-dir", type=Path, default=DEFAULT_SCREENSHOT_DIR)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-event WS timeout (seconds)")
    parser.add_argument("--headless", action="store_true", help="Run browser headlessly (default: visible)")
    parser.add_argument("--slow-mo", type=int, default=700, help="Slow motion ms (headed mode only)")
    parser.add_argument("--conversation-id", default=None, help="Skip recording; use existing conversation")
    parser.add_argument("--no-seed", action="store_true", help="Skip canary fact seeding")
    args = parser.parse_args()

    sys.exit(run(
        recording=args.recording,
        piece_slug=args.piece_slug,
        wrangler_url=args.wrangler_url,
        web_url=args.web_url,
        api_dir=args.api_dir,
        db_dsn=args.db_dsn,
        screenshot_dir=args.screenshot_dir,
        max_chunks=args.max_chunks,
        timeout_per_event=args.timeout,
        headless=args.headless,
        slow_mo=args.slow_mo,
        conversation_id=args.conversation_id,
        skip_seed=args.no_seed,
    ))


if __name__ == "__main__":
    _cli()
```

Then add the `just e2e-full-session` recipe to `justfile` immediately after the existing `e2e-ui-session` recipe:

```
# Full visible session eval: seeds canary memory facts, drives a real recording,
# opens a headed browser, runs scripted chat turns (recall + tool action), and
# prints a per-criterion PASS/FAIL report.
# Requires: just dev (MuQ:8000 + AMT:8001 + API:8787 + web:3000) + just seed-fingerprint.
# Default recording: model/data/evals/practice_eval/nocturne_op9no2/audio/_aySCutsVVQ.wav
e2e-full-session recording=("model/data/evals/practice_eval/nocturne_op9no2/audio/_aySCutsVVQ.wav") piece="nocturne_op9no2":
    cd apps/evals && uv run python -m e2e_full_session \
        --recording "../../{{recording}}" \
        --piece-slug "{{piece}}" \
        --screenshot-dir /tmp/e2e-full
```

Also add `psycopg2-binary` to `apps/evals/pyproject.toml` dependencies (required by `memory_seeder.py`). Locate the `dependencies = [` block and append `"psycopg2-binary>=2.9.0",`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest tests/test_full_session_report.py -v 2>&1 | tail -15
```

Expected: PASS — all 4 tests green.

Run all offline eval tests together to check for regressions:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest tests/test_memory_seeder.py tests/test_run_chat_turns.py tests/test_full_session_report.py -v 2>&1 | tail -20
```

Expected: PASS — all tests in the three new test files pass.

Run the web component tests:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/web && bun run test -- src/components/ChatMessages.test.tsx 2>&1 | tail -15
```

Expected: PASS — all 5 ChatMessages tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-70-visible-full-eval && git add apps/evals/e2e_full_session.py apps/evals/pyproject.toml apps/evals/tests/test_full_session_report.py justfile && git commit -m "feat(evals): e2e_full_session orchestrator + just e2e-full-session recipe (#70)"
```

---

## Live Gate (operator-only, requires full stack)

After all tasks are committed, the operator runs:

```bash
# 1. Start the full stack (in separate terminals or via tmux)
just dev           # MuQ:8000, AMT:8001, API:8787, web:3000
just seed-fingerprint

# 2. Run the full visible session eval
just e2e-full-session
```

Expected output includes a headed Chromium window opening, the synthesis appearing, two chat turns typed visibly, and a printed report ending with `OVERALL: PASS`.

Screenshots land in `/tmp/e2e-full/`:
- `phase1_synthesis.png` — after synthesis renders
- `phase2_recall.png` — after recall reply arrives
- `phase3_tool.png` — after tool turn reply arrives

To re-run chat only (after a session is already in DB):

```bash
cd apps/evals && uv run python -m e2e_full_session \
    --conversation-id <existing-conv-id> \
    --no-seed
```

---

## psycopg2-binary dependency note

`memory_seeder.py` uses `psycopg2` for direct Postgres access. The binary wheel (`psycopg2-binary`) is self-contained and does not require a system libpq install. Add it to `apps/evals/pyproject.toml` in the `dependencies` list during Task 4 Step 3. Run `cd apps/evals && uv lock` after editing `pyproject.toml` to regenerate the lockfile.

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv add psycopg2-binary
```

This updates both `pyproject.toml` and `uv.lock` atomically. Stage both files in the Task 4 commit.
