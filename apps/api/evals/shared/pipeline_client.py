"""WebSocket client for sending eval chunks to the local Rust pipeline.

Connects to wrangler dev, creates a practice session, sends pre-computed
inference results via eval_chunk messages, and collects observations.
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
    """An observation returned by the pipeline."""
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
class SessionResult:
    """Result of running a full recording through the pipeline."""
    session_id: str
    recording_id: str
    observations: list[PipelineObservation]
    chunk_responses: list[dict]
    errors: list[str]
    duration_ms: int


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
    return session


# Cache auth session across recordings (re-auth once per eval run)
_auth_session: requests.Session | None = None


def _get_auth_session(wrangler_url: str) -> requests.Session:
    global _auth_session
    if _auth_session is None:
        _auth_session = _get_debug_auth(wrangler_url)
    return _auth_session


async def run_recording(
    wrangler_url: str,
    recording_cache: dict,
    student_id: str = "eval-student-001",
    piece_query: str | None = None,
) -> SessionResult:
    """Run a cached recording through the full pipeline via wrangler dev."""
    recording_id = recording_cache["recording_id"]
    chunks = recording_cache["chunks"]
    start = time.time()

    # 0. Get authenticated session
    auth = _get_auth_session(wrangler_url)

    # 1. Create practice session via HTTP
    session_id = str(uuid.uuid4())
    resp = auth.post(
        f"{wrangler_url}/api/practice/start",
        json={"session_id": session_id, "student_id": student_id, "piece_query": piece_query},
        timeout=10,
    )
    if resp.status_code != 200:
        return SessionResult(
            session_id=session_id,
            recording_id=recording_id,
            observations=[],
            chunk_responses=[],
            errors=[f"Failed to start session: {resp.status_code} {resp.text}"],
            duration_ms=0,
        )

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

    try:
        async with websockets.connect(ws_url, additional_headers=ws_headers) as ws:
            # 3. Send each chunk as eval_chunk
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
                        elif msg_type == "error":
                            errors.append(response.get("message", "unknown error"))
                            break
                except asyncio.TimeoutError:
                    errors.append(f"Timeout waiting for chunk {chunk['chunk_index']}")

            # 4. Wait briefly for trailing observations
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    response = json.loads(raw)
                    if response.get("type") == "observation":
                        observations.append(_parse_observation(response))
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

    except Exception as e:
        errors.append(f"WebSocket error: {e}")

    duration_ms = int((time.time() - start) * 1000)
    return SessionResult(
        session_id=session_id,
        recording_id=recording_id,
        observations=observations,
        chunk_responses=chunk_responses,
        errors=errors,
        duration_ms=duration_ms,
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
