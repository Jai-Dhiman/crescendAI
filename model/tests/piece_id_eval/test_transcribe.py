# model/tests/piece_id_eval/test_transcribe.py
"""Verify ensure_amt_notes against a stub AMT HTTP server.

Pattern copied from tests/chroma_dtw_eval/test_amt_regen.py.
"""
from __future__ import annotations

import http.server
import json
import socketserver
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from piece_id_eval.transcribe import ensure_amt_notes


def _write_stub_wav(path: Path, duration_seconds: float = 3.0, sr: int = 16000) -> None:
    """Write a silent mono 16kHz WAV fixture."""
    samples = np.zeros(int(duration_seconds * sr), dtype=np.float32)
    sf.write(str(path), samples, sr)


_CANNED_NOTES = [
    {"onset": 0.0, "offset": 0.5, "pitch": 60, "velocity": 80},
    {"onset": 0.5, "offset": 1.0, "pitch": 62, "velocity": 75},
    {"onset": 1.0, "offset": 1.5, "pitch": 64, "velocity": 70},
]


class _StubAmtHandler(http.server.BaseHTTPRequestHandler):
    call_count: int = 0

    def log_message(self, *a, **k) -> None:  # silence
        pass

    def do_POST(self) -> None:  # noqa: N802
        _StubAmtHandler.call_count += 1
        n = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(n)
        body = json.dumps({"midi_notes": _CANNED_NOTES}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_amt_server():
    _StubAmtHandler.call_count = 0
    srv = socketserver.TCPServer(("127.0.0.1", 0), _StubAmtHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{port}/transcribe"
    srv.shutdown()
    srv.server_close()


def test_ensure_amt_notes_writes_file(tmp_path: Path, stub_amt_server: str) -> None:
    audio = tmp_path / "test.wav"
    _write_stub_wav(audio)
    out = tmp_path / "amt_notes.json"

    result = ensure_amt_notes(audio, out, amt_url=stub_amt_server)

    assert result == out
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) > 0
    assert "onset" in data[0] and "pitch" in data[0]


def test_ensure_amt_notes_is_idempotent(tmp_path: Path, stub_amt_server: str) -> None:
    audio = tmp_path / "test.wav"
    _write_stub_wav(audio)
    out = tmp_path / "amt_notes.json"

    ensure_amt_notes(audio, out, amt_url=stub_amt_server)
    calls_after_first = _StubAmtHandler.call_count

    # Second call must NOT hit the server again.
    ensure_amt_notes(audio, out, amt_url=stub_amt_server)
    assert _StubAmtHandler.call_count == calls_after_first, (
        "ensure_amt_notes made an HTTP request on second call (not idempotent)"
    )


def test_ensure_amt_notes_force_retranscribes(tmp_path: Path, stub_amt_server: str) -> None:
    audio = tmp_path / "test.wav"
    _write_stub_wav(audio)
    out = tmp_path / "amt_notes.json"

    ensure_amt_notes(audio, out, amt_url=stub_amt_server)
    calls_after_first = _StubAmtHandler.call_count

    ensure_amt_notes(audio, out, amt_url=stub_amt_server, force=True)
    assert _StubAmtHandler.call_count > calls_after_first, (
        "force=True did not retranscribe"
    )


def test_ensure_amt_notes_raises_on_missing_audio(tmp_path: Path, stub_amt_server: str) -> None:
    with pytest.raises(FileNotFoundError):
        ensure_amt_notes(
            tmp_path / "missing.wav",
            tmp_path / "out.json",
            amt_url=stub_amt_server,
        )
