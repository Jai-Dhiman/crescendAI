"""Offline unit tests for FullSessionReport and format_report() — no live services."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


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
    assert "SKIP" in text


def test_full_session_report_overall_false_when_criteria_e_fails():
    from e2e_full_session import FullSessionReport

    report = FullSessionReport(
        conversation_id="conv-fail",
        criteria_a=True,
        criteria_b_headline=True,
        criteria_b_components=True,
        criteria_c=None,
        criteria_e_recall=False,
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


def test_full_session_report_tool_text_only_is_nonfatal():
    """TEXT_ONLY tool outcome must not flip overall to FAIL."""
    from e2e_full_session import FullSessionReport

    report = FullSessionReport(
        conversation_id="conv-tool",
        criteria_a=True,
        criteria_b_headline=True,
        criteria_b_components=True,
        criteria_c=None,
        criteria_e_recall=True,
        criteria_e_tokens_found=["T1"],
        criteria_e_tokens_missing=[],
        criteria_f_tool_outcome="TEXT_ONLY",
        errors=[],
    )

    assert report.overall is True


def test_full_session_report_overall_false_on_errors():
    from e2e_full_session import FullSessionReport

    report = FullSessionReport(
        conversation_id="conv-err",
        criteria_a=True,
        criteria_b_headline=True,
        criteria_b_components=True,
        criteria_c=None,
        criteria_e_recall=True,
        criteria_e_tokens_found=["T1"],
        criteria_e_tokens_missing=[],
        criteria_f_tool_outcome="TOOL_RENDERED",
        errors=["Auth failed unexpectedly"],
    )

    assert report.overall is False
