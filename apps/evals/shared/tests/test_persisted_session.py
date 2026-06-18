"""Unit tests for drive_persisted() and PersistedSessionCapture (issue #68, Task 1+2).

All tests are offline — no live services required.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from shared.local_session import (
    PersistedSessionCapture,
    drive_persisted,
)


# ---------------------------------------------------------------------------
# Task 1: PersistedSessionCapture structure
# ---------------------------------------------------------------------------

def test_persisted_session_capture_fields():
    """PersistedSessionCapture has the required fields."""
    cap = PersistedSessionCapture(
        session_id="sess-1",
        conversation_id="conv-1",
        recording=Path("dummy.wav"),
        piece_slug="nocturne_op9no2",
        headline_text="Your phrasing was beautiful.",
        components=[],
        is_fallback=False,
        piece_identification={"pieceId": "chopin.nocturne_op9no2", "confidence": 0.95},
        prescribed_exercise=None,
        chunk_scores=[[0.7, 0.6, 0.8, 0.5, 0.9, 0.4]],
    )
    assert cap.session_id == "sess-1"
    assert cap.conversation_id == "conv-1"
    assert cap.headline_text == "Your phrasing was beautiful."
    assert cap.is_fallback is False
    assert cap.piece_identification is not None
    assert cap.chunk_scores == [[0.7, 0.6, 0.8, 0.5, 0.9, 0.4]]


def test_persisted_session_capture_with_prescription():
    """PersistedSessionCapture holds prescribed_exercise when present."""
    prescription = {"exerciseId": "ex-1", "focusDimension": "dynamics", "previewTitle": "Softer!"}
    cap = PersistedSessionCapture(
        session_id="s",
        conversation_id="c",
        recording=Path("x.wav"),
        piece_slug="slug",
        headline_text="great",
        components=[{"type": "pending_exercise", "config": prescription}],
        is_fallback=False,
        piece_identification=None,
        prescribed_exercise=prescription,
        chunk_scores=[],
    )
    assert cap.prescribed_exercise is not None
    assert cap.prescribed_exercise["focusDimension"] == "dynamics"


def test_drive_persisted_importable():
    """drive_persisted is importable and has the right signature."""
    sig = inspect.signature(drive_persisted)
    assert "recording" in sig.parameters
    assert "piece_slug" in sig.parameters
    assert "wrangler_url" in sig.parameters
    assert "timeout_per_event" in sig.parameters
    assert "max_chunks" in sig.parameters


# ---------------------------------------------------------------------------
# Task 2: WS message parsing — mocked end-to-end
# ---------------------------------------------------------------------------

def _make_synthesis_event(
    headline: str = "Focus on your dynamics.",
    components: list | None = None,
    is_fallback: bool = False,
) -> dict:
    return {
        "type": "synthesis",
        "text": headline,
        "components": components or [],
        "isFallback": is_fallback,
    }


def _make_piece_id_event(piece_id: str = "chopin.nocturne_op9no2") -> dict:
    return {"type": "piece_identified", "pieceId": piece_id, "confidence": 0.92}


def _make_chunk_processed_event(scores: list | None = None) -> dict:
    evt: dict = {"type": "chunk_processed"}
    if scores:
        evt["scores"] = scores
    return evt


@patch("shared.local_session._get_debug_auth")
@patch("shared.local_session._upload_chunk_to_r2")
@patch("shared.local_session._slice_to_webm_chunks")
@patch("shared.local_session.asyncio.run")
def test_drive_persisted_builds_capture_from_synthesis(
    mock_asyncio_run,
    mock_slice,
    mock_upload,
    mock_get_auth,
    tmp_path,
):
    """drive_persisted correctly builds PersistedSessionCapture from synthesis WS event."""
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"fake audio")

    # Mock auth session
    mock_auth = MagicMock()
    mock_auth.headers = {}
    mock_auth.cookies = {}
    start_resp = MagicMock()
    start_resp.status_code = 201
    start_resp.json.return_value = {"sessionId": "sess-abc", "conversationId": "conv-xyz"}
    mock_auth.post.return_value = start_resp
    mock_get_auth.return_value = mock_auth

    # Mock chunk slicing — one fake chunk
    fake_chunk = tmp_path / "chunk_000.webm"
    fake_chunk.write_bytes(b"chunk")
    mock_slice.return_value = [fake_chunk]

    # Mock asyncio.run to return the WS result dict
    synth_event = _make_synthesis_event("Focus on your dynamics.")
    piece_evt = _make_piece_id_event()
    mock_asyncio_run.return_value = {
        "synthesis": synth_event,
        "piece_identification": piece_evt,
        "chunk_scores": [[0.7, 0.6, 0.8, 0.5, 0.9, 0.4]],
    }

    cap = drive_persisted(wav, "nocturne_op9no2")

    assert isinstance(cap, PersistedSessionCapture)
    assert cap.session_id == "sess-abc"
    assert cap.conversation_id == "conv-xyz"
    assert cap.headline_text == "Focus on your dynamics."
    assert cap.is_fallback is False
    assert cap.piece_identification == {"pieceId": "chopin.nocturne_op9no2", "confidence": 0.92}
    assert cap.chunk_scores == [[0.7, 0.6, 0.8, 0.5, 0.9, 0.4]]
    mock_upload.assert_called_once()


@patch("shared.local_session._get_debug_auth")
@patch("shared.local_session._upload_chunk_to_r2")
@patch("shared.local_session._slice_to_webm_chunks")
@patch("shared.local_session.asyncio.run")
def test_drive_persisted_extracts_prescription(
    mock_asyncio_run,
    mock_slice,
    mock_upload,
    mock_get_auth,
    tmp_path,
):
    """drive_persisted extracts prescribed_exercise from pending_exercise component."""
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"fake audio")

    mock_auth = MagicMock()
    mock_auth.headers = {}
    mock_auth.cookies = {}
    start_resp = MagicMock()
    start_resp.status_code = 201
    start_resp.json.return_value = {"sessionId": "s1", "conversationId": "c1"}
    mock_auth.post.return_value = start_resp
    mock_get_auth.return_value = mock_auth

    fake_chunk = tmp_path / "chunk_000.webm"
    fake_chunk.write_bytes(b"x")
    mock_slice.return_value = [fake_chunk]

    prescription_config = {
        "exerciseId": "ex-001",
        "focusDimension": "dynamics",
        "previewTitle": "Play softer in mm. 5-8",
    }
    synth_event = _make_synthesis_event(
        headline="Work on your dynamics.",
        components=[{"type": "pending_exercise", "config": prescription_config}],
    )
    mock_asyncio_run.return_value = {
        "synthesis": synth_event,
        "piece_identification": None,
        "chunk_scores": [],
    }

    cap = drive_persisted(wav, "nocturne_op9no2")

    assert cap.prescribed_exercise is not None
    assert cap.prescribed_exercise["focusDimension"] == "dynamics"
    assert cap.prescribed_exercise["exerciseId"] == "ex-001"


@patch("shared.local_session._get_debug_auth")
def test_drive_persisted_raises_if_session_start_fails(mock_get_auth, tmp_path):
    """drive_persisted raises RuntimeError if /api/practice/start fails."""
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"x")

    mock_auth = MagicMock()
    mock_auth.headers = {}
    mock_auth.cookies = {}
    start_resp = MagicMock()
    start_resp.status_code = 500
    start_resp.text = "Internal Server Error"
    mock_auth.post.return_value = start_resp
    mock_get_auth.return_value = mock_auth

    with pytest.raises(RuntimeError, match="Failed to start practice session"):
        drive_persisted(wav, "nocturne_op9no2")


@patch("shared.local_session._get_debug_auth")
def test_drive_persisted_raises_if_conversation_id_missing(mock_get_auth, tmp_path):
    """drive_persisted raises RuntimeError if conversationId absent from start response."""
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"x")

    mock_auth = MagicMock()
    mock_auth.headers = {}
    mock_auth.cookies = {}
    start_resp = MagicMock()
    start_resp.status_code = 201
    start_resp.json.return_value = {"sessionId": "sess-abc"}  # missing conversationId
    start_resp.text = '{"sessionId":"sess-abc"}'
    mock_auth.post.return_value = start_resp
    mock_get_auth.return_value = mock_auth

    with pytest.raises(RuntimeError, match="missing conversationId"):
        drive_persisted(wav, "nocturne_op9no2")


def test_persisted_no_eval_params_in_ws_url():
    """The WS URL built by _drive_persisted_async must NOT contain eval params."""
    # This is a structural check — we inspect the source code to verify the contract.
    import inspect as _inspect
    import shared.local_session as _mod
    src = _inspect.getsource(_mod._drive_persisted_async)
    assert "eval=true" not in src
    assert "evalStudentId" not in src
    assert "x-eval-secret" not in src
    assert "conversationId" in src
