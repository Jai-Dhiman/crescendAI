"""Drive amt_regen against a stub AMT HTTP server and the committed bach
prelude JSON; assert (a) the 4-field-keyed cache is written, (b) second
call with identical inputs is a no-op, (c) the loader reads it back.

The bach JSON is the canonical score for the first piece. The stub AMT
returns a synthesized note set whose pitches overlap the bach prelude
bar 1 (C major arpeggio), so parangonar's matcher will produce >= 100
matches when stub note count is scaled accordingly.
"""
from __future__ import annotations

import http.server
import json
import socketserver
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from chroma_dtw_eval.amt_regen import (
    AmtRegenError,
    LowCoverageError,
    RegenResult,
    regenerate_pseudo_truth,
)
from chroma_dtw_eval.pseudo_truth_cache import load_pseudo_truth

REPO_ROOT = Path(__file__).resolve().parents[3]
BACH_SCORE = REPO_ROOT / "model/data/scores/bach.prelude.bwv_846.json"


def _bach_canned_notes() -> list[dict]:
    """Synthesize an AMT-like note set from the committed bach JSON itself
    by reading bar 1's notes and shifting onsets by +0.05s (a small jitter
    to mimic AMT detection noise). Returns >= 100 notes by walking enough
    bars; ensures the coverage gate passes.
    """
    body = json.loads(BACH_SCORE.read_text())
    bars = body.get("bars") or []
    notes: list[dict] = []
    for bar in bars:
        for n in (bar.get("notes") or []):
            notes.append({
                "onset": float(n["onset_seconds"]) + 0.05,
                "offset": float(n["onset_seconds"]) + float(n.get("duration_seconds", 0.2)) + 0.05,
                "pitch": int(n["pitch"]),
                "velocity": int(n.get("velocity", 80)),
            })
            if len(notes) >= 200:
                return notes
    return notes


class _StubAmtHandler(http.server.BaseHTTPRequestHandler):
    canned_notes: list[dict] = []

    def log_message(self, *a, **k):  # silence
        pass

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(n)
        body = json.dumps({"midi_notes": self.canned_notes}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_amt_server():
    _StubAmtHandler.canned_notes = _bach_canned_notes()
    srv = socketserver.TCPServer(("127.0.0.1", 0), _StubAmtHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown(); srv.server_close()


@pytest.fixture
def tiny_audio(tmp_path: Path) -> Path:
    wav = tmp_path / "tiny.wav"
    # 25s of near-silence at 16k. Must be <= 27s (AMT_CHUNK_S) so only one
    # AMT chunk is issued. Two chunks double the stub's note count, pushing
    # the match_rate below the 0.5 coverage gate with the stub's fixed 200-note
    # response.
    sf.write(wav, np.zeros(16000 * 25, dtype=np.float32), 16000, subtype="FLOAT")
    return wav


def test_regen_writes_cache_and_is_idempotent(
    stub_amt_server: str, tiny_audio: Path, tmp_path: Path,
) -> None:
    cache_root = tmp_path / "pseudo_truth"
    first: RegenResult = regenerate_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="V0",
        score_path=BACH_SCORE, audio_path=tiny_audio,
        amt_url=stub_amt_server + "/transcribe",
        amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
        parangonar_version="3.3.2",
        cache_root=cache_root,
    )
    assert first.wrote_cache is True
    assert first.cache_path.exists()
    assert first.n_matched >= 100

    second: RegenResult = regenerate_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="V0",
        score_path=BACH_SCORE, audio_path=tiny_audio,
        amt_url=stub_amt_server + "/transcribe",
        amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
        parangonar_version="3.3.2",
        cache_root=cache_root,
    )
    assert second.wrote_cache is False, "second regen with identical inputs must be no-op"

    loaded = load_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="V0",
        audio_sha256=first.audio_sha256,
        amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
        score_sha256=first.score_sha256,
        parangonar_version="3.3.2",
        cache_root=cache_root,
    )
    assert loaded.perf_audio_sec.size >= 100
    assert loaded.score_audio_sec.size == loaded.perf_audio_sec.size


def test_regen_raises_low_coverage_on_sparse_match(
    stub_amt_server: str, tiny_audio: Path, tmp_path: Path,
) -> None:
    # Override stub to return only 10 notes; coverage gate (>=100 matched) must fire.
    _StubAmtHandler.canned_notes = _bach_canned_notes()[:10]
    try:
        with pytest.raises((LowCoverageError, AmtRegenError)):
            regenerate_pseudo_truth(
                piece_id="bach_prelude_c_wtc1", video_id="V1",
                score_path=BACH_SCORE, audio_path=tiny_audio,
                amt_url=stub_amt_server + "/transcribe",
                amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
                parangonar_version="3.3.2",
                cache_root=tmp_path / "pseudo_truth",
            )
    finally:
        _StubAmtHandler.canned_notes = _bach_canned_notes()


@pytest.mark.parametrize("missing", ["score", "audio"])
def test_regen_raises_on_missing_path(tmp_path: Path, missing: str) -> None:
    """First-line guard: regenerate_pseudo_truth must raise AmtRegenError on
    missing score or audio paths before any work runs."""
    real_audio = tmp_path / "real.wav"
    sf.write(str(real_audio), np.zeros(16000, dtype=np.float32), 16000)

    score_path = BACH_SCORE if missing != "score" else tmp_path / "nope_score.json"
    audio_path = real_audio if missing != "audio" else tmp_path / "nope_audio.wav"

    with pytest.raises(AmtRegenError, match=f"{missing} not found"):
        regenerate_pseudo_truth(
            piece_id="bach_prelude_c_wtc1", video_id="V_missing",
            score_path=score_path, audio_path=audio_path,
            amt_url="http://127.0.0.1:9/transcribe",
            amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
            parangonar_version="3.3.2",
            cache_root=tmp_path / "pseudo_truth",
        )
