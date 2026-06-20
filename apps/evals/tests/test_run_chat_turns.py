"""Offline contract test for run_chat_turns() — no live browser required.

Verifies that run_chat_turns() returns one ChatTurnResult per input turn with
the expected field names, using a lightweight stub Page that simulates a new
assistant-message appearing after each Enter press.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parents[1]))


def _make_stub_page(reply_texts: list[str]):
    """Build a minimal Page stub simulating assistant-message elements appearing.

    Simulates turn-by-turn behavior:
    - Baseline call (before any fill/press) returns 0 elements.
    - After each press(), the next poll returns a new element with stable text.
    """
    page = MagicMock()
    page.fill = MagicMock()
    page.wait_for_selector = MagicMock()

    # State: tracks how many "turns" have been submitted via press()
    state = {"turns_submitted": 0, "call_index": 0}

    def _press(selector: str, key: str):
        if key == "Enter":
            state["turns_submitted"] += 1

    page.press = _press

    def _query_all(selector: str):
        state["call_index"] += 1
        # Return as many stable elements as turns submitted so far (capped by reply_texts)
        num = min(state["turns_submitted"], len(reply_texts))
        elements = []
        for i in range(num):
            el = MagicMock()
            el.inner_text.return_value = reply_texts[i]
            elements.append(el)
        return elements

    page.query_selector_all = _query_all
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

    assert stub_page.fill.call_count == 2
    fill_texts = [c.args[1] for c in stub_page.fill.call_args_list]
    assert "hello" in fill_texts
    assert "goodbye" in fill_texts


def test_run_chat_turns_timeout_raises_runtime_error():
    from ui_verifier import run_chat_turns

    # Stub that never returns any elements — will hit deadline
    stub_page = MagicMock()
    stub_page.fill = MagicMock()
    stub_page.press = MagicMock()
    stub_page.query_selector_all = MagicMock(return_value=[])

    try:
        run_chat_turns(
            page=stub_page,
            turns=["hello"],
            reply_timeout_ms=200,  # very short deadline
        )
        assert False, "Expected RuntimeError not raised"
    except RuntimeError as exc:
        assert "timed out" in str(exc).lower()


def test_run_chat_turns_returns_reply_text():
    from ui_verifier import run_chat_turns

    stub_page = _make_stub_page(["The teacher says practice slowly."])

    results = run_chat_turns(
        page=stub_page,
        turns=["What should I practice?"],
        reply_timeout_ms=5000,
    )

    assert len(results) == 1
    assert "slowly" in results[0].reply_text
