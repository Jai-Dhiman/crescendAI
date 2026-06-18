"""Offline unit tests for the e2e UI session orchestrator (issue #68, Task 7).

Tests cover:
- _lowest_dim() correct dimension name from chunk scores
- e2e_ui_session.run() pre-flight: missing recording raises a non-zero exit
- e2e_ui_session.run() pre-flight: services unreachable returns non-zero exit
- ui_verifier.VerificationResult.passed logic

No live services required — drive_persisted and verify_ui are mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2]))

from e2e_ui_session import _lowest_dim
from ui_verifier import VerificationResult


# ---------------------------------------------------------------------------
# _lowest_dim
# ---------------------------------------------------------------------------

def test_lowest_dim_returns_correct_name():
    """_lowest_dim returns the dimension name with the lowest mean score."""
    # dims: dynamics, timing, pedaling, articulation, phrasing, interpretation
    chunk_scores = [
        [0.9, 0.8, 0.3, 0.7, 0.6, 0.5],  # pedaling=0.3 in row 0
        [0.8, 0.7, 0.4, 0.6, 0.5, 0.6],  # pedaling=0.4 in row 1
    ]
    # means: dynamics=0.85, timing=0.75, pedaling=0.35, articulation=0.65, phrasing=0.55, interpretation=0.55
    assert _lowest_dim(chunk_scores) == "pedaling"


def test_lowest_dim_returns_none_for_empty():
    """_lowest_dim returns None when chunk_scores is empty."""
    assert _lowest_dim([]) is None


def test_lowest_dim_single_chunk():
    """_lowest_dim works with a single chunk."""
    scores = [[0.5, 0.2, 0.8, 0.9, 0.6, 0.7]]
    assert _lowest_dim(scores) == "timing"


def test_lowest_dim_six_equal_scores():
    """_lowest_dim returns the first dimension when all are equal."""
    scores = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
    assert _lowest_dim(scores) == "dynamics"


def test_lowest_dim_single_dim_rows():
    """_lowest_dim handles rows shorter than 6 dimensions."""
    scores = [[0.3], [0.4]]
    assert _lowest_dim(scores) == "dynamics"


# ---------------------------------------------------------------------------
# VerificationResult.passed
# ---------------------------------------------------------------------------

def test_verification_result_passed_all_true():
    r = VerificationResult(
        conversation_id="c",
        criteria_a_v6_artifact=True,
        criteria_b_headline_match=True,
        criteria_b_components_rendered=True,
        criteria_c_confirm_flow=None,
        criteria_d_dimension_in_headline=None,
        screenshot_path=None,
    )
    assert r.passed is True


def test_verification_result_fails_if_headline_mismatch():
    r = VerificationResult(
        conversation_id="c",
        criteria_a_v6_artifact=True,
        criteria_b_headline_match=False,
        criteria_b_components_rendered=True,
        criteria_c_confirm_flow=None,
        criteria_d_dimension_in_headline=None,
        screenshot_path=None,
    )
    assert r.passed is False


def test_verification_result_fails_if_has_errors():
    r = VerificationResult(
        conversation_id="c",
        criteria_a_v6_artifact=True,
        criteria_b_headline_match=True,
        criteria_b_components_rendered=True,
        criteria_c_confirm_flow=None,
        criteria_d_dimension_in_headline=None,
        screenshot_path=None,
        errors=["something failed"],
    )
    assert r.passed is False


def test_verification_result_fails_if_a_false():
    r = VerificationResult(
        conversation_id="c",
        criteria_a_v6_artifact=False,
        criteria_b_headline_match=True,
        criteria_b_components_rendered=True,
        criteria_c_confirm_flow=None,
        criteria_d_dimension_in_headline=None,
        screenshot_path=None,
    )
    assert r.passed is False


# ---------------------------------------------------------------------------
# e2e_ui_session.run() pre-flight checks (offline)
# ---------------------------------------------------------------------------

def test_run_fails_if_recording_missing(tmp_path):
    """run() returns 1 if the recording file does not exist."""
    import e2e_ui_session
    missing = tmp_path / "no_such_file.wav"
    ret = e2e_ui_session.run(
        recording=missing,
        piece_slug="nocturne_op9no2",
    )
    assert ret == 1


@patch("e2e_ui_session.check_services")
def test_run_fails_if_services_unreachable(mock_check, tmp_path):
    """run() returns 1 if check_services raises RuntimeError."""
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"fake")
    mock_check.side_effect = RuntimeError("API not reachable")

    import e2e_ui_session
    ret = e2e_ui_session.run(
        recording=wav,
        piece_slug="nocturne_op9no2",
    )
    assert ret == 1


@patch("e2e_ui_session.verify_ui")
@patch("e2e_ui_session.drive_persisted")
@patch("e2e_ui_session.check_services")
def test_run_passes_when_mocked_success(mock_check, mock_drive, mock_verify, tmp_path):
    """run() returns 0 when drive_persisted and verify_ui both succeed."""
    from shared.local_session import PersistedSessionCapture

    wav = tmp_path / "test.wav"
    wav.write_bytes(b"fake")

    mock_check.return_value = None
    mock_drive.return_value = PersistedSessionCapture(
        session_id="sess-1",
        conversation_id="conv-1",
        recording=wav,
        piece_slug="nocturne_op9no2",
        headline_text="Beautiful phrasing today.",
        components=[],
        is_fallback=False,
        piece_identification=None,
        prescribed_exercise=None,
        chunk_scores=[[0.7, 0.6, 0.8, 0.5, 0.9, 0.4]],
    )
    mock_verify.return_value = VerificationResult(
        conversation_id="conv-1",
        criteria_a_v6_artifact=True,
        criteria_b_headline_match=True,
        criteria_b_components_rendered=True,
        criteria_c_confirm_flow=None,
        criteria_d_dimension_in_headline=True,
        screenshot_path=None,
    )

    import e2e_ui_session
    ret = e2e_ui_session.run(
        recording=wav,
        piece_slug="nocturne_op9no2",
    )
    assert ret == 0


@patch("e2e_ui_session.verify_ui")
@patch("e2e_ui_session.drive_persisted")
@patch("e2e_ui_session.check_services")
def test_run_returns_1_if_is_fallback(mock_check, mock_drive, mock_verify, tmp_path):
    """run() returns 1 and skips verification if synthesis is_fallback=true."""
    from shared.local_session import PersistedSessionCapture

    wav = tmp_path / "test.wav"
    wav.write_bytes(b"fake")

    mock_check.return_value = None
    mock_drive.return_value = PersistedSessionCapture(
        session_id="s",
        conversation_id="c",
        recording=wav,
        piece_slug="slug",
        headline_text="Fallback text",
        components=[],
        is_fallback=True,  # V6 failed
        piece_identification=None,
        prescribed_exercise=None,
        chunk_scores=[],
    )

    import e2e_ui_session
    ret = e2e_ui_session.run(recording=wav, piece_slug="slug")
    assert ret == 1
    mock_verify.assert_not_called()
