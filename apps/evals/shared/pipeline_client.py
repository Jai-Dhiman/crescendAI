"""WebSocket client for sending eval chunks to the local Rust pipeline.

Connects to wrangler dev, creates a practice session, sends pre-computed
inference results via eval_chunk messages, captures synthesis output and
accumulator state for offline analysis.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlparse

import asyncio
import requests
import websockets


@dataclass
class PipelineObservation:
    """A lightweight observation returned during accumulation."""
    text: str
    dimension: str
    framing: str
    chunk_index: int
    score: float
    baseline: float
    reasoning_trace: str
    is_fallback: bool = False
    raw_message: dict = field(default_factory=dict)


@dataclass
class PieceIdentification:
    """Result of automatic piece identification."""
    piece_id: str
    confidence: float
    method: str
    notes_consumed: int = 0


@dataclass
class SynthesisResult:
    """Session synthesis output from the teacher LLM."""
    text: str
    is_fallback: bool
    eval_context: dict = field(default_factory=dict)


@dataclass
class SessionResult:
    """Result of running a full recording through the pipeline."""
    session_id: str
    recording_id: str
    observations: list[PipelineObservation]
    chunk_responses: list[dict]
    errors: list[str]
    duration_ms: int
    piece_identification: PieceIdentification | None = None
    synthesis: SynthesisResult | None = None
    # Efficiency metadata
    chunk_send_duration_ms: int = 0
    synthesis_latency_ms: int = 0


def _get_debug_auth(wrangler_url: str) -> requests.Session:
    """Authenticate via debug endpoint, return a session with auth cookie."""
    session = requests.Session()
    resp = session.post(f"{wrangler_url}/api/auth/debug", timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Debug auth failed: {resp.status_code} {resp.text}")
    # The debug endpoint sets an HttpOnly cookie + returns token in body
    data = resp.json()
    token = data.get("token", "")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    # Verify the token actually works before returning
    verify = session.get(f"{wrangler_url}/api/auth/me", timeout=5)
    if verify.status_code != 200:
        raise RuntimeError(f"Auth verification failed: {verify.status_code} {verify.text}")

    return session


# Cache auth session across recordings (re-auth once per eval run)
_auth_session: requests.Session | None = None


def _get_auth_session(wrangler_url: str) -> requests.Session:
    global _auth_session
    if _auth_session is None:
        _auth_session = _get_debug_auth(wrangler_url)
    return _auth_session


def _reset_auth_session() -> None:
    """Reset cached auth session (e.g., on 401 response)."""
    global _auth_session
    _auth_session = None


async def run_recording(
    wrangler_url: str,
    recording_cache: dict,
    student_id: str = "eval-student-001",
    piece_query: str | None = None,
) -> SessionResult:
    """Run a cached recording through the full pipeline via wrangler dev.

    Sends all chunks as eval_chunk messages, then triggers end_session
    to capture the synthesis output. The pipeline routes through the
    production accumulation path (no legacy per-observation LLM calls).
    """
    recording_id = recording_cache["recording_id"]
    chunks = recording_cache["chunks"]
    start = time.time()

    # 0. Get authenticated session
    auth = _get_auth_session(wrangler_url)

    # 1. Create practice session via HTTP
    resp = auth.post(
        f"{wrangler_url}/api/practice/start",
        json={},
        timeout=10,
    )
    if resp.status_code == 401:
        # Token may have expired -- re-auth and retry once
        _reset_auth_session()
        auth = _get_auth_session(wrangler_url)
        resp = auth.post(
            f"{wrangler_url}/api/practice/start",
            json={},
            timeout=10,
        )
    if resp.status_code != 200:
        return SessionResult(
            session_id="",
            recording_id=recording_id,
            observations=[],
            chunk_responses=[],
            errors=[f"Failed to start session: {resp.status_code} {resp.text}"],
            duration_ms=0,
        )

    # Use the server-generated session ID (not a client UUID)
    session_id = resp.json().get("sessionId", str(uuid.uuid4()))

    # 2. Connect WebSocket (localhost only -- eval never targets remote)
    if "localhost" not in wrangler_url and "127.0.0.1" not in wrangler_url:
        raise ValueError(f"Eval pipeline client only connects to localhost, got: {wrangler_url}")

    parsed = urlparse(wrangler_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}/api/practice/ws/{session_id}"

    observations: list[PipelineObservation] = []
    chunk_responses: list[dict] = []
    errors: list[str] = []

    # Pass auth token to WebSocket via cookie header
    ws_headers = {}
    token = auth.headers.get("Authorization", "")
    if token:
        ws_headers["Authorization"] = token
    # Also forward cookies (HttpOnly auth cookie)
    cookie_str = "; ".join(f"{k}={v}" for k, v in auth.cookies.items())
    if cookie_str:
        ws_headers["Cookie"] = cookie_str

    piece_id_result: PieceIdentification | None = None
    synthesis_result: SynthesisResult | None = None
    chunk_send_start = 0
    chunk_send_end = 0
    synthesis_request_time = 0

    try:
        async with websockets.connect(ws_url, additional_headers=ws_headers) as ws:
            # 2.5. Set piece query if provided (via WS message, matching web client)
            if piece_query:
                await ws.send(json.dumps({"type": "set_piece", "query": piece_query}))

            # 3. Send each chunk as eval_chunk
            chunk_send_start = time.time()
            for chunk in chunks:
                msg = {
                    "type": "eval_chunk",
                    "chunk_index": chunk["chunk_index"],
                    "predictions": chunk["predictions"],
                    "midi_notes": chunk.get("midi_notes", []),
                    "pedal_events": chunk.get("pedal_events", []),
                }
                await ws.send(json.dumps(msg))

                # Collect responses with timeout
                try:
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        response = json.loads(raw)
                        msg_type = response.get("type", "")

                        if msg_type == "chunk_processed":
                            chunk_responses.append(response)
                            break
                        elif msg_type == "observation":
                            observations.append(_parse_observation(response))
                        elif msg_type == "piece_identified":
                            piece_id_result = _parse_piece_id(response)
                        elif msg_type == "mode_change":
                            pass  # captured in accumulator state
                        elif msg_type == "error":
                            errors.append(response.get("message", "unknown error"))
                            break
                except asyncio.TimeoutError:
                    errors.append(f"Timeout waiting for chunk {chunk['chunk_index']}")

            chunk_send_end = time.time()

            # 4. Send end_session and wait for synthesis
            synthesis_request_time = time.time()
            await ws.send(json.dumps({"type": "end_session"}))

            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
                    response = json.loads(raw)
                    msg_type = response.get("type", "")

                    if msg_type == "synthesis":
                        synthesis_result = SynthesisResult(
                            text=response.get("text", ""),
                            is_fallback=response.get("is_fallback", False),
                            eval_context=response.get("eval_context", {}),
                        )
                    elif msg_type == "observation":
                        observations.append(_parse_observation(response))
                    elif msg_type == "piece_identified":
                        piece_id_result = _parse_piece_id(response)
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

    except Exception as e:
        errors.append(f"WebSocket error: {e}")

    if synthesis_result is None and not errors:
        errors.append("No synthesis received before connection closed")

    duration_ms = int((time.time() - start) * 1000)
    chunk_send_duration_ms = int((chunk_send_end - chunk_send_start) * 1000) if chunk_send_start else 0
    synthesis_latency_ms = (
        int((time.time() - synthesis_request_time) * 1000)
        if synthesis_request_time and synthesis_result
        else 0
    )

    return SessionResult(
        session_id=session_id,
        recording_id=recording_id,
        observations=observations,
        chunk_responses=chunk_responses,
        errors=errors,
        duration_ms=duration_ms,
        piece_identification=piece_id_result,
        synthesis=synthesis_result,
        chunk_send_duration_ms=chunk_send_duration_ms,
        synthesis_latency_ms=synthesis_latency_ms,
    )


def _parse_observation(response: dict) -> PipelineObservation:
    """Parse a WebSocket observation message into a PipelineObservation."""
    return PipelineObservation(
        text=response.get("text", ""),
        dimension=response.get("dimension", ""),
        framing=response.get("framing", ""),
        chunk_index=response.get("chunk_index", 0),
        score=response.get("score", 0.0),
        baseline=response.get("baseline", 0.0),
        reasoning_trace=response.get("reasoning_trace", ""),
        is_fallback=response.get("is_fallback", False),
        raw_message=response,
    )


def _parse_piece_id(response: dict) -> PieceIdentification:
    """Parse a piece_identified WebSocket message."""
    return PieceIdentification(
        piece_id=response.get("pieceId", ""),
        confidence=response.get("confidence", 0.0),
        method=response.get("method", ""),
        notes_consumed=response.get("notesConsumed", 0),
    )
