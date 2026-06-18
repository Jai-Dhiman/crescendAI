# Persisted-Session E2E UI Test Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Drive one real recording through the live local pipeline as a non-eval session, persist synthesis to Postgres, and assert rendered output in the web UI via Python Playwright.
**Spec:** docs/specs/2026-06-17-e2e-ui-session-design.md
**Style:** Follow apps/api/TS_STYLE.md for any TS touched; Python: explicit exceptions, no silent fallbacks, uv (not pip), no emojis.

---

## Prerequisites (build agent must verify before starting)

- `just dev` is running (MuQ:8000, AMT:8001, API:8787, Web:3000)
- `just seed-fingerprint` has been run (populates local R2 `fingerprint/v2/piece_index.json`)
- `apps/evals/.venv` has Playwright installed (`playwright>=1.58.0` is in pyproject.toml)
- Playwright browsers are installed: `cd apps/evals && uv run playwright install chromium`
- Recording exists: `model/data/evals/practice_eval/nocturne_op9no2/audio/_aySCutsVVQ.wav`

---

## Task Groups

```
Group A (parallel): Task 1, Task 2
Group B (sequential, depends on A): Task 3
Group C (sequential, depends on B): Task 4
Group D (sequential, depends on C): Task 5
Group D2 (sequential, depends on D — Task 6 imports e2e.ui_verifier created in Task 5): Task 6
Group E (sequential, depends on D): Task 7
```

---

## Classifier: offline vs live-stack

| Task | Requires live stack? |
|------|---------------------|
| Task 1 (PersistedSessionCapture + drive_persisted stub) | No |
| Task 2 (offline unit test for capture-parsing) | No |
| Task 3 (drive_persisted live run) | Yes — needs `just dev` + MuQ + AMT |
| Task 4 (CORS + cookie validation) | Yes — needs `just dev` + web:3000 |
| Task 5 (UIVerifier render assertions) | Yes — needs full stack + conversation from Task 3 |
| Task 6 (confirm->reveal flow) | Yes — needs full stack + prescription from Task 3 |
| Task 7 (orchestrator + just recipe) | No (wires Task 3+5+6); live stack for final smoke run |

---

## Task 1: Add `PersistedSessionCapture` dataclass and `drive_persisted()` to `local_session.py`

**Group:** A (parallel with Task 2)
**Requires live stack:** No

**Behavior being verified:** `drive_persisted` is importable with the correct signature and `PersistedSessionCapture` is constructable with the expected fields; `__all__` exports both.

**Interface under test:** `from shared.local_session import drive_persisted, PersistedSessionCapture`

**Files:**
- Modify: `apps/evals/shared/local_session.py`
- Test: `apps/evals/pipeline/exercise_routing/tests/test_local_session_smoke.py` (extend existing file)

- [ ] **Step 1: Write the failing test**

Extend `apps/evals/pipeline/exercise_routing/tests/test_local_session_smoke.py` — append at the end of the file:

```python
def test_persisted_session_capture_importable():
    """PersistedSessionCapture must be importable and constructable with all fields."""
    from shared.local_session import PersistedSessionCapture
    cap = PersistedSessionCapture(
        conversation_id="conv-123",
        session_id="sess-456",
        synthesis_text="Focus on dynamics.",
        components=[{"type": "score_highlight", "config": {}}],
        chunk_scores=[{"dynamics": 0.7, "timing": 0.6, "pedaling": 0.5,
                       "articulation": 0.8, "phrasing": 0.6, "interpretation": 0.7}],
        piece_identification={"pieceId": "chopin.nocturne_op9no2", "confidence": 0.95},
    )
    assert cap.conversation_id == "conv-123"
    assert cap.session_id == "sess-456"
    assert cap.synthesis_text == "Focus on dynamics."
    assert len(cap.components) == 1
    assert len(cap.chunk_scores) == 1
    assert cap.piece_identification is not None


def test_drive_persisted_callable():
    """drive_persisted is importable and has the expected signature."""
    import inspect
    from shared.local_session import drive_persisted
    sig = inspect.signature(drive_persisted)
    assert "recording" in sig.parameters
    assert "piece_slug" in sig.parameters
    assert "wrangler_url" in sig.parameters
    assert "timeout_per_event" in sig.parameters
    assert "max_chunks" in sig.parameters


def test_local_session_all_exports():
    """__all__ exports PersistedSessionCapture and drive_persisted."""
    import shared.local_session as m
    assert "PersistedSessionCapture" in m.__all__
    assert "drive_persisted" in m.__all__
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest pipeline/exercise_routing/tests/test_local_session_smoke.py::test_persisted_session_capture_importable -x -q 2>&1 | tail -10
```

Expected: FAIL — `ImportError: cannot import name 'PersistedSessionCapture' from 'shared.local_session'`

- [ ] **Step 3: Implement the minimum to make the tests pass**

Add to `apps/evals/shared/local_session.py` — insert immediately after the existing `__all__` line (line 28) and before `CHUNK_SECONDS`:

```python
__all__ = ["SessionCapture", "drive", "check_services", "read_eval_secret",
           "PersistedSessionCapture", "drive_persisted"]
```

(Replace the existing `__all__` line entirely.)

Then insert the `PersistedSessionCapture` dataclass and the `drive_persisted` / `_drive_persisted_async` functions after the `_upload_chunk_to_r2` function (after line 121). Add these imports at the top of the file (after the existing imports block):

```python
from dataclasses import dataclass, field
```

Add the dataclass right after the `_upload_chunk_to_r2` function:

```python
@dataclass
class PersistedSessionCapture:
    """Output of drive_persisted(): one WAV through the non-eval chunk_ready path.

    conversation_id is guaranteed non-empty; isFallback=True raises before this
    is returned. chunk_scores accumulates one entry per chunk_processed event.
    """
    conversation_id: str
    session_id: str
    synthesis_text: str
    components: list[dict]
    chunk_scores: list[dict] = field(default_factory=list)
    piece_identification: dict | None = None


async def _drive_persisted_async(
    recording: Path,
    piece_slug: str,
    session_id: str,
    conversation_id: str,
    r2_keys: list[str],
    wrangler_url: str,
    timeout_per_event: float,
) -> dict:
    """Non-eval async WS driver. Returns {synthesis_event, piece_identification, chunk_scores}."""
    parsed = urlparse(wrangler_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    # Non-eval: no ?eval=true, no evalStudentId — conversationId forwarded for persistence gate
    ws_url = (
        f"{ws_scheme}://{parsed.netloc}/api/practice/ws/{session_id}"
        f"?conversationId={conversation_id}"
    )

    from shared.pipeline_client import _get_debug_auth
    auth = _get_debug_auth(wrangler_url)
    headers: dict[str, str] = {}
    token = auth.headers.get("Authorization", "")
    if token:
        headers["Authorization"] = token
    cookie_str = "; ".join(f"{k}={v}" for k, v in auth.cookies.items())
    if cookie_str:
        headers["Cookie"] = cookie_str
    # No x-eval-secret header — non-eval path

    piece_identification: dict | None = None
    synthesis_event: dict | None = None
    chunk_scores: list[dict] = []

    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        await ws.send(json.dumps({"type": "set_piece", "query": piece_slug}))

        for idx, r2_key in enumerate(r2_keys):
            await ws.send(json.dumps({"type": "chunk_ready", "index": idx, "r2Key": r2_key}))
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_per_event)
                    evt = json.loads(raw)
                    etype = evt.get("type")
                    if etype == "chunk_processed":
                        chunk_scores.append({
                            "dynamics": evt.get("scores", {}).get("dynamics", 0.0),
                            "timing": evt.get("scores", {}).get("timing", 0.0),
                            "pedaling": evt.get("scores", {}).get("pedaling", 0.0),
                            "articulation": evt.get("scores", {}).get("articulation", 0.0),
                            "phrasing": evt.get("scores", {}).get("phrasing", 0.0),
                            "interpretation": evt.get("scores", {}).get("interpretation", 0.0),
                        })
                        break
                    elif etype == "piece_identified":
                        piece_identification = {
                            "pieceId": evt.get("pieceId", ""),
                            "confidence": evt.get("confidence", 0.0),
                        }
                    elif etype == "error":
                        raise RuntimeError(
                            f"DO returned error during chunk {idx}: {evt.get('message')}"
                        )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Timeout after {timeout_per_event}s waiting for chunk_processed "
                    f"for chunk {idx} of {recording}. Is MuQ:8000 warm?"
                )
            except websockets.exceptions.ConnectionClosed as exc:
                raise RuntimeError(
                    f"Connection closed waiting for chunk_processed for chunk {idx}: {exc}"
                ) from exc

        await ws.send(json.dumps({"type": "end_session"}))

        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout_per_event)
                evt = json.loads(raw)
                etype = evt.get("type")
                if etype == "synthesis":
                    synthesis_event = evt
                    break
                elif etype == "piece_identified":
                    piece_identification = {
                        "pieceId": evt.get("pieceId", ""),
                        "confidence": evt.get("confidence", 0.0),
                    }
                elif etype == "error":
                    raise RuntimeError(f"DO returned error during synthesis: {evt.get('message')}")
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Timeout after {timeout_per_event}s waiting for synthesis for {recording}."
            )
        except websockets.exceptions.ConnectionClosed:
            pass  # Server closed connection after synthesis; check synthesis_event below

    if synthesis_event is None:
        raise RuntimeError(f"No synthesis event received for {recording}")
    return {
        "synthesis": synthesis_event,
        "piece_identification": piece_identification,
        "chunk_scores": chunk_scores,
    }


def drive_persisted(
    recording: Path,
    piece_slug: str,
    wrangler_url: str = "http://localhost:8787",
    api_dir: Path = DEFAULT_API_DIR,
    timeout_per_event: float = 180.0,
    max_chunks: int = 6,
) -> PersistedSessionCapture:
    """Drive one WAV as a non-eval session; return a PersistedSessionCapture.

    Unlike drive(), this runs without ?eval=true so the DO's conversationId
    persistence gate fires and the synthesis message is written to Postgres.
    The WS payload contains only {text, components, isFallback} with NO eval_context.

    Raises RuntimeError if:
    - services are not reachable
    - /start returns no conversationId
    - isFallback is True in the synthesis event
    - no synthesis event is received within timeout
    """
    from shared.pipeline_client import _get_debug_auth

    auth = _get_debug_auth(wrangler_url)
    resp = auth.post(
        f"{wrangler_url}/api/practice/start",
        json={},
        timeout=10,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to start practice session: {resp.status_code} {resp.text}"
        )
    body = resp.json()
    session_id = body.get("sessionId")
    conversation_id = body.get("conversationId")
    if not session_id:
        raise RuntimeError(f"POST /api/practice/start missing sessionId: {resp.text}")
    if not conversation_id:
        raise RuntimeError(
            f"POST /api/practice/start missing conversationId: {resp.text}. "
            "Persistence gate requires conversationId; cannot proceed."
        )

    with tempfile.TemporaryDirectory(prefix="crescend-e2e-") as tmp:
        tmp_dir = Path(tmp)
        chunks = _slice_to_webm_chunks(recording, tmp_dir, max_chunks)

        r2_keys: list[str] = []
        for idx, chunk_path in enumerate(chunks):
            r2_key = f"sessions/{session_id}/chunks/{idx}.webm"
            _upload_chunk_to_r2(api_dir, r2_key, chunk_path)
            r2_keys.append(r2_key)

        result = asyncio.run(
            _drive_persisted_async(
                recording=recording,
                piece_slug=piece_slug,
                session_id=session_id,
                conversation_id=conversation_id,
                r2_keys=r2_keys,
                wrangler_url=wrangler_url,
                timeout_per_event=timeout_per_event,
            )
        )

    synth = result["synthesis"]
    if synth.get("isFallback") is True:
        raise RuntimeError(
            f"V6 synthesis returned isFallback=True for {recording}. "
            "Check DO logs for v6 phase_error. glm@WorkersAI must be reachable."
        )

    return PersistedSessionCapture(
        conversation_id=conversation_id,
        session_id=session_id,
        synthesis_text=synth.get("text", ""),
        components=synth.get("components", []),
        chunk_scores=result["chunk_scores"],
        piece_identification=result.get("piece_identification"),
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest pipeline/exercise_routing/tests/test_local_session_smoke.py -x -q 2>&1 | tail -10
```

Expected: PASS — all smoke tests green including the 3 new ones.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-68-e2e-ui-session && git add apps/evals/shared/local_session.py apps/evals/pipeline/exercise_routing/tests/test_local_session_smoke.py && git commit -m "feat(e2e): add PersistedSessionCapture + drive_persisted() to local_session

Additive — drive() and SessionCapture are unchanged. drive_persisted() runs
the non-eval path (?conversationId=..., no ?eval=true, no x-eval-secret) so
the DO persistence gate fires and synthesis is written to Postgres.

Closes #68 (partial)"
```

---

## Task 2: Offline unit test for `drive_persisted` capture-parsing with mocked WS

**Group:** A (parallel with Task 1)
**Requires live stack:** No

**Behavior being verified:** `drive_persisted` correctly parses a synthesis WS event (text, components) and accumulates chunk_processed scores when the WS exchange succeeds; raises RuntimeError when isFallback=True.

**Interface under test:** `drive_persisted(recording, piece_slug, ...)` — behavior observable through the returned `PersistedSessionCapture` fields and raised exceptions.

**Files:**
- Create: `apps/evals/e2e/__init__.py`
- Create: `apps/evals/e2e/tests/__init__.py`
- Create: `apps/evals/e2e/tests/test_drive_persisted_capture.py`

- [ ] **Step 1: Write the failing test**

Create `apps/evals/e2e/__init__.py` (empty):
```python
```

Create `apps/evals/e2e/tests/__init__.py` (empty):
```python
```

Create `apps/evals/e2e/tests/test_drive_persisted_capture.py`:

```python
"""Offline unit tests for drive_persisted() capture-parsing.

Uses unittest.mock to replace websockets.connect and requests.Session so no
live services are required. Exercises the public interface (PersistedSessionCapture
fields + raised exceptions) not internal implementation details.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[4]))

from shared.local_session import PersistedSessionCapture, drive_persisted


FAKE_WAV = Path(__file__).parent / "fixtures" / "fake.wav"
SYNTHESIS_EVENT = {
    "type": "synthesis",
    "text": "Your dynamics were expressive in bars 1-4.",
    "components": [
        {"type": "score_highlight", "config": {"highlights": [{"bars": [1, 4], "dimension": "dynamics"}]}},
    ],
    "isFallback": False,
}
CHUNK_PROCESSED_EVENT = {
    "type": "chunk_processed",
    "index": 0,
    "scores": {
        "dynamics": 0.72,
        "timing": 0.65,
        "pedaling": 0.55,
        "articulation": 0.80,
        "phrasing": 0.60,
        "interpretation": 0.70,
    },
}
PIECE_ID_EVENT = {
    "type": "piece_identified",
    "pieceId": "chopin.nocturne_op9no2",
    "confidence": 0.94,
}


def _make_mock_ws_messages(*messages):
    """Return an async iterator over the given JSON-encoded messages."""
    encoded = [json.dumps(m) for m in messages]
    idx = 0

    async def recv():
        nonlocal idx
        if idx >= len(encoded):
            raise Exception("No more messages")
        msg = encoded[idx]
        idx += 1
        return msg

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=cm)
    cm.__aexit__ = AsyncMock(return_value=False)
    cm.send = AsyncMock()
    cm.recv = recv
    return cm


@pytest.fixture()
def mock_start_response():
    """Mock POST /api/practice/start returning sessionId + conversationId."""
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {
        "sessionId": "sess-test-001",
        "conversationId": "conv-test-abc",
    }
    mock_resp.text = '{"sessionId":"sess-test-001","conversationId":"conv-test-abc"}'
    return mock_resp


def test_drive_persisted_parses_synthesis_and_chunk_scores(tmp_path, mock_start_response):
    """drive_persisted returns PersistedSessionCapture with correct text, components, chunk_scores."""
    # Create a minimal fake WAV (ffmpeg will be mocked)
    fake_wav = tmp_path / "fake.wav"
    fake_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    fake_chunk = tmp_path / "chunk_000.webm"
    fake_chunk.write_bytes(b"\x1aE\xdf\xa3")  # minimal WebM magic bytes

    mock_ws = _make_mock_ws_messages(
        CHUNK_PROCESSED_EVENT,
        SYNTHESIS_EVENT,
    )

    mock_session = MagicMock()
    mock_session.post.return_value = mock_start_response
    mock_session.headers = {}
    mock_session.cookies = {}

    with (
        patch("shared.local_session._slice_to_webm_chunks", return_value=[fake_chunk]),
        patch("shared.local_session._upload_chunk_to_r2"),
        patch("shared.pipeline_client._get_debug_auth", return_value=mock_session),
        patch("websockets.connect", return_value=mock_ws),
    ):
        result = drive_persisted(
            recording=fake_wav,
            piece_slug="chopin.nocturne_op9no2",
            wrangler_url="http://localhost:8787",
            api_dir=tmp_path,
        )

    assert isinstance(result, PersistedSessionCapture)
    assert result.conversation_id == "conv-test-abc"
    assert result.session_id == "sess-test-001"
    assert result.synthesis_text == "Your dynamics were expressive in bars 1-4."
    assert len(result.components) == 1
    assert result.components[0]["type"] == "score_highlight"
    assert len(result.chunk_scores) == 1
    assert result.chunk_scores[0]["dynamics"] == pytest.approx(0.72)
    assert result.chunk_scores[0]["timing"] == pytest.approx(0.65)


def test_drive_persisted_raises_on_is_fallback(tmp_path, mock_start_response):
    """drive_persisted raises RuntimeError when synthesis isFallback=True."""
    fake_wav = tmp_path / "fake.wav"
    fake_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    fake_chunk = tmp_path / "chunk_000.webm"
    fake_chunk.write_bytes(b"\x1aE\xdf\xa3")

    fallback_synthesis = {**SYNTHESIS_EVENT, "isFallback": True}
    mock_ws = _make_mock_ws_messages(CHUNK_PROCESSED_EVENT, fallback_synthesis)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_start_response
    mock_session.headers = {}
    mock_session.cookies = {}

    with (
        patch("shared.local_session._slice_to_webm_chunks", return_value=[fake_chunk]),
        patch("shared.local_session._upload_chunk_to_r2"),
        patch("shared.pipeline_client._get_debug_auth", return_value=mock_session),
        patch("websockets.connect", return_value=mock_ws),
    ):
        with pytest.raises(RuntimeError, match="isFallback=True"):
            drive_persisted(
                recording=fake_wav,
                piece_slug="chopin.nocturne_op9no2",
                wrangler_url="http://localhost:8787",
                api_dir=tmp_path,
            )


def test_drive_persisted_raises_when_conversation_id_missing(tmp_path):
    """drive_persisted raises RuntimeError when /start returns no conversationId."""
    fake_wav = tmp_path / "fake.wav"
    fake_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    no_conv_resp = MagicMock()
    no_conv_resp.status_code = 201
    no_conv_resp.json.return_value = {"sessionId": "sess-test-001"}
    no_conv_resp.text = '{"sessionId":"sess-test-001"}'

    mock_session = MagicMock()
    mock_session.post.return_value = no_conv_resp
    mock_session.headers = {}
    mock_session.cookies = {}

    with patch("shared.pipeline_client._get_debug_auth", return_value=mock_session):
        with pytest.raises(RuntimeError, match="missing conversationId"):
            drive_persisted(
                recording=fake_wav,
                piece_slug="chopin.nocturne_op9no2",
                wrangler_url="http://localhost:8787",
                api_dir=tmp_path,
            )


def test_drive_persisted_captures_piece_identification(tmp_path, mock_start_response):
    """drive_persisted captures piece_identification from piece_identified WS event."""
    fake_wav = tmp_path / "fake.wav"
    fake_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    fake_chunk = tmp_path / "chunk_000.webm"
    fake_chunk.write_bytes(b"\x1aE\xdf\xa3")

    mock_ws = _make_mock_ws_messages(
        CHUNK_PROCESSED_EVENT,
        PIECE_ID_EVENT,
        SYNTHESIS_EVENT,
    )

    mock_session = MagicMock()
    mock_session.post.return_value = mock_start_response
    mock_session.headers = {}
    mock_session.cookies = {}

    with (
        patch("shared.local_session._slice_to_webm_chunks", return_value=[fake_chunk]),
        patch("shared.local_session._upload_chunk_to_r2"),
        patch("shared.pipeline_client._get_debug_auth", return_value=mock_session),
        patch("websockets.connect", return_value=mock_ws),
    ):
        result = drive_persisted(
            recording=fake_wav,
            piece_slug="chopin.nocturne_op9no2",
            wrangler_url="http://localhost:8787",
            api_dir=tmp_path,
        )

    assert result.piece_identification is not None
    assert result.piece_identification["pieceId"] == "chopin.nocturne_op9no2"
    assert result.piece_identification["confidence"] == pytest.approx(0.94)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest e2e/tests/test_drive_persisted_capture.py -x -q 2>&1 | tail -15
```

Expected: FAIL — `ImportError: cannot import name 'drive_persisted' from 'shared.local_session'` (Task 1 not yet complete in a fresh agent; if running after Task 1, this test will fail because `drive_persisted` doesn't exist yet in the Task 1 stub version, OR because the mock wiring test finds a real bug in the implementation).

**Note for build agent:** Tasks 1 and 2 run in parallel. If Task 1 completes first, the expected failure reason changes to a logic assertion error. In either case, Step 4 (PASS) validates correctness.

- [ ] **Step 3: Implement the minimum to make the tests pass**

The implementation lives in Task 1. This task adds only the test file. After Task 1 is merged, run Step 4.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest e2e/tests/test_drive_persisted_capture.py -v 2>&1 | tail -20
```

Expected: PASS — 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-68-e2e-ui-session && git add apps/evals/e2e/__init__.py apps/evals/e2e/tests/__init__.py apps/evals/e2e/tests/test_drive_persisted_capture.py && git commit -m "test(e2e): offline unit tests for drive_persisted capture-parsing

Mocked WS + requests — no live services required. Verifies text/component
parsing, isFallback guard, missing-conversationId guard, piece_id capture.

Refs #68"
```

---

## Task 3: Live integration — `drive_persisted` drives nocturne through full pipeline

**Group:** B (depends on Group A)
**Requires live stack:** Yes — `just dev` + `just seed-fingerprint` must be running

**Behavior being verified:** `drive_persisted` with the real nocturne recording returns a `PersistedSessionCapture` with non-empty `synthesis_text`, at least one component, and `isFallback` never raised (confirming V6 on glm-4.7-flash@WorkersAI produced a real artifact).

**Interface under test:** `drive_persisted(recording=Path("...nocturne.wav"), piece_slug="chopin.nocturne_op9no2")` — the public function, end-to-end.

**Files:**
- Create: `apps/evals/e2e/tests/test_drive_persisted_live.py`

**Note:** This test is marked `@pytest.mark.integration` and skipped in offline CI. It is the live gate that confirms the pipeline works.

- [ ] **Step 1: Write the failing test**

Create `apps/evals/e2e/tests/test_drive_persisted_live.py`:

```python
"""Live integration test for drive_persisted().

Requires `just dev` + `just seed-fingerprint`. Run with:
  cd apps/evals && uv run python -m pytest e2e/tests/test_drive_persisted_live.py -v -m integration

Skipped unless CRESCEND_LIVE_STACK=1 environment variable is set.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[4]))

LIVE_STACK = os.environ.get("CRESCEND_LIVE_STACK", "") == "1"
NOCTURNE_WAV = (
    Path(__file__).resolve().parents[5]
    / "model" / "data" / "evals" / "practice_eval"
    / "nocturne_op9no2" / "audio" / "_aySCutsVVQ.wav"
)


@pytest.mark.integration
@pytest.mark.skipif(not LIVE_STACK, reason="Set CRESCEND_LIVE_STACK=1 and run `just dev` first")
def test_drive_persisted_nocturne_real_pipeline():
    """Drive the nocturne WAV as a non-eval session; verify V6 synthesis persisted."""
    from shared.local_session import PersistedSessionCapture, drive_persisted, check_services

    if not NOCTURNE_WAV.exists():
        pytest.skip(f"Nocturne WAV not found at {NOCTURNE_WAV}")

    check_services("http://localhost:8787")

    result = drive_persisted(
        recording=NOCTURNE_WAV,
        piece_slug="chopin.nocturne_op9no2",
        max_chunks=4,  # ~1 minute of audio; enough for V6 to fire
    )

    assert isinstance(result, PersistedSessionCapture)
    assert result.conversation_id, "conversation_id must be non-empty (persistence gate)"
    assert result.session_id, "session_id must be non-empty"
    assert result.synthesis_text, "synthesis_text must be non-empty (V6 must have fired)"
    assert len(result.chunk_scores) > 0, "at least one chunk must have been processed"
    # isFallback guard is enforced by drive_persisted itself — if we reach here it's False

    # Print for build agent to inspect (prescription present or not)
    has_pending = any(c.get("type") == "pending_exercise" for c in result.components)
    print(f"\nconversation_id: {result.conversation_id}")
    print(f"synthesis_text[:120]: {result.synthesis_text[:120]}")
    print(f"components: {[c.get('type') for c in result.components]}")
    print(f"has_pending_exercise: {has_pending}")
    print(f"chunk_scores count: {len(result.chunk_scores)}")
    if result.piece_identification:
        print(f"piece_identified: {result.piece_identification}")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest e2e/tests/test_drive_persisted_live.py -v -m integration 2>&1 | tail -10
```

Expected: SKIP (if `CRESCEND_LIVE_STACK` not set) or FAIL if set but services not running.

To run for real (with services up):
```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && CRESCEND_LIVE_STACK=1 uv run python -m pytest e2e/tests/test_drive_persisted_live.py -v -m integration -s 2>&1 | tail -30
```

Expected: FAIL before implementation — impossible since Task 1 provides `drive_persisted`. If it fails with `RuntimeError: isFallback=True`, the glm model is not reachable; check `apps/api/.dev.vars` for `AI_GATEWAY_ENDPOINT` and `AI_GATEWAY_TOKEN`. If it fails with `RuntimeError: missing conversationId`, the `/start` route has regressed.

- [ ] **Step 3: Implement**

No additional implementation needed — all code is in Task 1. This task's job is to run the live test and confirm the pipeline works. If the live test reveals bugs, fix them in `local_session.py` (same commit).

**Build agent action:** Run the live test with `CRESCEND_LIVE_STACK=1`. Capture the printed `conversation_id` — it is needed by Tasks 5 and 6. Save it to `apps/evals/results/e2e-session-latest.json`:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && CRESCEND_LIVE_STACK=1 uv run python -m pytest e2e/tests/test_drive_persisted_live.py -v -m integration -s 2>&1
```

**Also empirically confirm** whether the nocturne recording produces a `pending_exercise` component by inspecting the printed `components` list. This determines whether Tasks 5 and 6 can test the confirm flow unconditionally or must use the conditional path.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && CRESCEND_LIVE_STACK=1 uv run python -m pytest e2e/tests/test_drive_persisted_live.py -v -m integration -s 2>&1 | tail -20
```

Expected: PASS — `PersistedSessionCapture` with non-empty `conversation_id`, `synthesis_text`, and at least one entry in `chunk_scores`.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-68-e2e-ui-session && git add apps/evals/e2e/tests/test_drive_persisted_live.py && git commit -m "test(e2e): live integration test for drive_persisted nocturne pipeline

Marked @pytest.mark.integration; skipped without CRESCEND_LIVE_STACK=1.
Verifies V6 synthesis is non-fallback and conversation_id is non-empty.

Refs #68"
```

---

## Task 4: CORS + cookie validation — confirm browser can authenticate and fetch conversation

**Group:** C (depends on Group B — needs a live conversation_id from Task 3)
**Requires live stack:** Yes — `just dev` + web:3000

**Behavior being verified:** A Playwright browser context that POSTs to `http://localhost:8787/api/auth/debug` can subsequently fetch `GET http://localhost:8787/api/conversations/<conversation_id>` and receive a 200 with the persisted conversation (not 401 or 404), confirming the CORS + cookie + ownership chain works before writing the full UI verifier.

**Interface under test:** `playwright.sync_api` — `browser.new_context()`, `context.request.post(...)`, `context.request.get(...)`.

**Files:**
- Create: `apps/evals/e2e/tests/test_cors_cookie_validation.py`

- [ ] **Step 1: Write the failing test**

Create `apps/evals/e2e/tests/test_cors_cookie_validation.py`:

```python
"""Validate CORS + cookie chain: browser POST /api/auth/debug -> GET /api/conversations/:id.

Requires live stack: `just dev` + CRESCEND_LIVE_STACK=1.
Also requires a conversation_id from a drive_persisted() run (Task 3).
Pass it via CRESCEND_TEST_CONV_ID env var.

Run:
  CRESCEND_LIVE_STACK=1 CRESCEND_TEST_CONV_ID=<uuid> \\
    uv run python -m pytest e2e/tests/test_cors_cookie_validation.py -v -m integration -s
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[4]))

LIVE_STACK = os.environ.get("CRESCEND_LIVE_STACK", "") == "1"
CONV_ID = os.environ.get("CRESCEND_TEST_CONV_ID", "")


@pytest.mark.integration
@pytest.mark.skipif(
    not LIVE_STACK or not CONV_ID,
    reason="Set CRESCEND_LIVE_STACK=1 and CRESCEND_TEST_CONV_ID=<uuid> with `just dev` running",
)
def test_browser_auth_and_conversation_fetch():
    """Browser context POSTs debug auth then GETs the conversation — must be 200."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(base_url="http://localhost:8787")

        # Step 1: authenticate in-browser
        auth_resp = context.request.post("/api/auth/debug")
        assert auth_resp.status == 200, (
            f"Debug auth failed: {auth_resp.status} {auth_resp.text()[:200]}\n"
            "Is `just dev` running?"
        )

        # Step 2: fetch the conversation (same debug user owns it)
        conv_resp = context.request.get(f"/api/conversations/{CONV_ID}")
        if conv_resp.status == 404:
            pytest.fail(
                f"Conversation {CONV_ID} not found (404). "
                "This means the identity/ownership chain is broken. "
                "Possible causes: (1) CORS SameSite cookie not forwarded cross-origin; "
                "(2) conversation_id belongs to a different user than debug@crescend.ai; "
                "(3) persistence gate did not fire (check drive_persisted logs). "
                "Fallback: add a dev-only login affordance to the web app."
            )
        if conv_resp.status == 401:
            pytest.fail(
                f"Unauthorized (401) fetching conversation {CONV_ID}. "
                "The auth cookie was not sent with the conversations request. "
                "Check better-auth SameSite setting and CORS credentials config."
            )
        assert conv_resp.status == 200, (
            f"Unexpected status {conv_resp.status} fetching conversation {CONV_ID}: "
            f"{conv_resp.text()[:200]}"
        )

        data = conv_resp.json()
        assert "messages" in data, f"Response missing 'messages': {str(data)[:200]}"
        synthesis_messages = [
            m for m in data.get("messages", [])
            if m.get("messageType") == "synthesis"
        ]
        assert len(synthesis_messages) >= 1, (
            f"No synthesis message found in conversation {CONV_ID}. "
            f"Messages: {[m.get('messageType') for m in data.get('messages', [])]}"
        )

        browser.close()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest e2e/tests/test_cors_cookie_validation.py -v -m integration 2>&1 | tail -5
```

Expected: SKIP (missing env vars). With env vars set but before live run, fails with a meaningful error.

- [ ] **Step 3: Implement**

No new implementation code. This test either passes immediately (CORS + cookie chain works) or reveals a failure that must be fixed before Task 5.

**If the test fails with 401 or 404:** The build agent must investigate the CORS/cookie issue:

1. Check `apps/api/src/index.ts` CORS config — confirm `credentials: true` and `origin: "http://localhost:3000"` are present (they are in the current code).
2. Check better-auth cookie `SameSite` attribute. Run:
   ```bash
   curl -si -X POST http://localhost:8787/api/auth/debug | grep -i set-cookie
   ```
   If `SameSite=Strict` or `SameSite=None` without `Secure` appears, the fallback is needed.
3. **Fallback if CORS fails:** Add a `data-testid="debug-login-btn"` button to the web app's root that POSTs `/api/auth/debug` when `import.meta.env.DEV` is true. The UIVerifier then clicks it instead of using `context.request.post`. File: `apps/web/src/routes/__root.tsx` or `apps/web/src/routes/index.tsx` (additive, conditional on `DEV`).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
CRESCEND_LIVE_STACK=1 CRESCEND_TEST_CONV_ID=<conversation_id_from_task3> \
  cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest e2e/tests/test_cors_cookie_validation.py -v -m integration -s 2>&1 | tail -20
```

Expected: PASS — 200 response with at least one synthesis message.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-68-e2e-ui-session && git add apps/evals/e2e/tests/test_cors_cookie_validation.py && git commit -m "test(e2e): CORS + cookie chain validation for browser -> API conversation fetch

Playwright browser context POSTs /api/auth/debug then GETs /api/conversations/:id.
Fails with diagnostic message if CORS, SameSite, or ownership chain is broken.

Refs #68"
```

---

## Task 5: `UIVerifier` — render-only assertions (headline + component cards)

**Group:** D (depends on Group C — CORS validated, live conversation available)
**Requires live stack:** Yes — `just dev` + web:3000

**Behavior being verified:** After `verify_session_ui` is called with a `PersistedSessionCapture`, the web app shows the synthesis headline text and renders at least one card per non-pending, non-search component; screenshot is saved.

**Interface under test:** `verify_session_ui(capture, ...) -> UIAssertionResult` with `passed=True`, `headline_matched=True`, `component_count_matched=True`.

**Files:**
- Create: `apps/evals/e2e/ui_verifier.py`
- Create: `apps/evals/e2e/tests/test_ui_verifier_live.py`
- Conditionally modify (additive only): `apps/web/src/components/ReflectionMessage.tsx`, `apps/web/src/components/cards/ExerciseSetCard.tsx`, `apps/web/src/components/cards/ScoreHighlightCard.tsx`, `apps/web/src/components/InlineCard.tsx` — add `data-testid` attributes IF no stable text/role selector exists.

- [ ] **Step 1: Write the failing test**

Create `apps/evals/e2e/tests/test_ui_verifier_live.py`:

```python
"""Live test for UIVerifier render-only assertions.

Requires `just dev` + CRESCEND_LIVE_STACK=1 + a drive_persisted() result.
Run:
  CRESCEND_LIVE_STACK=1 CRESCEND_TEST_CONV_ID=<uuid> \\
  CRESCEND_TEST_SYNTHESIS_TEXT="<first 60 chars of synthesis text>" \\
  CRESCEND_TEST_COMPONENT_COUNT=<N> \\
    uv run python -m pytest e2e/tests/test_ui_verifier_live.py -v -m integration -s
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[4]))

LIVE_STACK = os.environ.get("CRESCEND_LIVE_STACK", "") == "1"
CONV_ID = os.environ.get("CRESCEND_TEST_CONV_ID", "")
SYNTHESIS_TEXT = os.environ.get("CRESCEND_TEST_SYNTHESIS_TEXT", "")
COMPONENT_COUNT = int(os.environ.get("CRESCEND_TEST_COMPONENT_COUNT", "0"))


@pytest.mark.integration
@pytest.mark.skipif(
    not LIVE_STACK or not CONV_ID,
    reason="Set CRESCEND_LIVE_STACK=1 CRESCEND_TEST_CONV_ID=<uuid> with `just dev` running",
)
def test_ui_verifier_render_assertions():
    """verify_session_ui returns UIAssertionResult with headline_matched + component_count_matched."""
    from shared.local_session import PersistedSessionCapture
    from e2e.ui_verifier import UIAssertionResult, verify_session_ui

    # Build a minimal PersistedSessionCapture from env (real data from Task 3)
    capture = PersistedSessionCapture(
        conversation_id=CONV_ID,
        session_id="test-session",
        synthesis_text=SYNTHESIS_TEXT,
        components=[{"type": "score_highlight"}] * COMPONENT_COUNT,  # shape only; DOM is ground truth
        chunk_scores=[],
    )

    result = verify_session_ui(
        capture=capture,
        web_url="http://localhost:3000",
        api_url="http://localhost:8787",
        headed=False,
    )

    assert isinstance(result, UIAssertionResult)
    assert result.screenshot_path, "Screenshot must always be saved"
    assert Path(result.screenshot_path).exists(), f"Screenshot file missing: {result.screenshot_path}"
    assert result.headline_matched, (
        f"Headline mismatch. Expected text containing: {SYNTHESIS_TEXT[:80]!r}. "
        f"Error: {result.error}"
    )
    assert result.component_count_matched, (
        f"Component count mismatch. Expected {COMPONENT_COUNT} renderable cards. "
        f"Error: {result.error}"
    )
    assert result.passed, f"UIAssertionResult.passed=False. Error: {result.error}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest e2e/tests/test_ui_verifier_live.py -v -m integration 2>&1 | tail -5
```

Expected: SKIP or FAIL — `ImportError: No module named 'e2e.ui_verifier'`.

- [ ] **Step 3: Implement `apps/evals/e2e/ui_verifier.py`**

```python
"""UIVerifier: Playwright-based DOM assertion for persisted session conversations.

verify_session_ui() hides all browser lifecycle, auth, navigation, wait strategy,
DOM selector logic, and screenshot IO behind a single function that returns
UIAssertionResult. Raises RuntimeError only for programmer errors (wrong types);
all runtime failures are captured in UIAssertionResult.error.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Component types that produce a renderable card (excludes pending_exercise,
# search_catalog_result which have no InlineCard renderer).
RENDERABLE_COMPONENT_TYPES = {
    "exercise_set",
    "score_highlight",
    "segment_loop",
    "play_passage",
}


@dataclass
class UIAssertionResult:
    """Result of verify_session_ui(). All runtime failures are captured here, not raised."""
    passed: bool
    headline_matched: bool
    component_count_matched: bool
    confirm_flow_ran: bool
    confirm_flow_passed: bool | None
    screenshot_path: str
    error: str | None = None


def verify_session_ui(
    capture: "PersistedSessionCapture",  # type: ignore[name-defined]
    web_url: str = "http://localhost:3000",
    api_url: str = "http://localhost:8787",
    headed: bool = False,
    screenshot_dir: Path = RESULTS_DIR,
) -> UIAssertionResult:
    """Navigate to the persisted conversation and assert the web UI rendered it correctly.

    Phase 1: authenticate in browser context, navigate to /app/c/<conversation_id>,
    wait for synthesis message, assert headline text + component card count.
    Phase 2 (conditional): if pending_exercise component present, click Confirm,
    wait for ExerciseSetCard to appear, assert reveal.

    Returns UIAssertionResult. Never raises for runtime assertion failures.
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    screenshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = str(screenshot_dir / f"e2e-ui-session-{timestamp}.png")

    headline_matched = False
    component_count_matched = False
    confirm_flow_ran = False
    confirm_flow_passed: bool | None = None
    error: str | None = None

    renderable_components = [
        c for c in capture.components
        if isinstance(c, dict) and c.get("type") in RENDERABLE_COMPONENT_TYPES
    ]
    has_pending = any(
        isinstance(c, dict) and c.get("type") == "pending_exercise"
        for c in capture.components
    )
    expected_card_count = len(renderable_components)

    page = None  # Guard for the screenshot-recovery block if an exception fires early.

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not headed)
            context = browser.new_context(base_url=api_url)

            # Authenticate in-browser as debug@crescend.ai (same user that owns the conversation)
            auth_resp = context.request.post("/api/auth/debug")
            if auth_resp.status != 200:
                raise RuntimeError(
                    f"Browser debug auth failed: {auth_resp.status} {auth_resp.text()[:200]}"
                )

            # CRITICAL: the page MUST be created in `context`, not via browser.new_page().
            # browser.new_page() creates a page in the DEFAULT context, which does NOT
            # hold the auth cookie set by context.request.post("/api/auth/debug"). A page
            # in the default context hits 401 on GET /api/auth/session and redirects to
            # /signin, so the conversation never renders. context.new_page() shares the
            # cookie jar with the authenticated request, so the page is authenticated.
            page = context.new_page()

            # Navigate to the conversation
            nav_url = f"{web_url}/app/c/{capture.conversation_id}"
            page.goto(nav_url, wait_until="networkidle", timeout=30_000)

            # Wait for synthesis message bubble to contain the headline text.
            # The synthesis message renders as a <p> with class text-body-sm inside
            # the message bubble (ReflectionMessage.reflectionText or MessageContent).
            # Strategy: wait for any element containing the first 60 chars of the headline.
            headline_prefix = capture.synthesis_text[:60]
            try:
                page.wait_for_function(
                    f"() => document.body.innerText.includes({headline_prefix!r})",
                    timeout=15_000,
                )
                headline_matched = True
            except PWTimeout:
                error = (
                    f"Headline text not found in DOM after 15s. "
                    f"Expected text containing: {headline_prefix!r}. "
                    f"Page URL: {page.url}. "
                    f"Check: (1) conversation_id correct? (2) auth cookie forwarded? "
                    f"(3) React hydrated?"
                )

            # Count renderable component cards.
            # ExerciseSetCard: has h4 with targetSkill text (inside bg-surface-card div)
            # ScoreHighlightCard: look for 'Score Highlight' label
            # PlayPassageCard: look for 'Play Passage' label
            # SegmentLoopArtifactCard: look for segment loop card
            # Fallback selector: count [data-testid^="inline-card-"] if build adds testid hooks.
            #
            # Build note: if no stable selector exists for a card type, add:
            #   data-testid="inline-card-<type>" to the outer div of each card component.
            # This is the PREFERRED selector for Playwright robustness.
            # Check if testid hooks exist:
            testid_cards = page.locator("[data-testid^='inline-card-']")
            if expected_card_count == 0:
                component_count_matched = True
            elif testid_cards.count() >= expected_card_count:
                component_count_matched = True
            else:
                # Fallback: count known card text patterns
                score_highlight_count = page.locator("text=Score Highlight").count()
                play_passage_count = page.locator("text=Play Passage").count()
                exercise_set_count = page.locator(".bg-surface-card").count()
                segment_loop_count = page.locator("text=Loop").count()
                found_count = max(
                    score_highlight_count + play_passage_count + segment_loop_count,
                    exercise_set_count,
                )
                component_count_matched = found_count >= expected_card_count
                if not component_count_matched:
                    card_error = (
                        f"Expected {expected_card_count} renderable cards; "
                        f"found {found_count} via fallback selectors. "
                        f"Build may need to add data-testid='inline-card-<type>' "
                        f"to ExerciseSetCard, ScoreHighlightCard, PlayPassageCard, SegmentLoopArtifactCard."
                    )
                    error = f"{error or ''}\n{card_error}".strip()

            # Phase 2: confirm flow (conditional)
            if has_pending:
                confirm_flow_ran = True
                try:
                    # ReflectionMessage renders a "Confirm" button (text-based selector is stable)
                    confirm_btn = page.locator("button", has_text="Confirm").first
                    confirm_btn.wait_for(state="visible", timeout=10_000)
                    confirm_btn.click()

                    # Wait for ExerciseSetCard to appear (exercise_set card reveals after assign)
                    # Either via data-testid or the bg-surface-card class with exercise content
                    try:
                        page.wait_for_selector(
                            "[data-testid='inline-card-exercise_set'], .bg-surface-card",
                            timeout=15_000,
                        )
                        confirm_flow_passed = True
                    except PWTimeout:
                        confirm_flow_passed = False
                        confirm_error = (
                            "ExerciseSetCard did not appear after clicking Confirm within 15s. "
                            "Check POST /api/exercises/assign response."
                        )
                        error = f"{error or ''}\n{confirm_error}".strip()
                except PWTimeout:
                    confirm_flow_passed = False
                    confirm_error = "Confirm button not found within 10s."
                    error = f"{error or ''}\n{confirm_error}".strip()

            page.screenshot(path=screenshot_path, full_page=True)
            browser.close()

    except Exception as exc:
        error = f"{error or ''}\nUnexpected error: {exc}".strip()
        # Still try to take screenshot if a page was created. `page` is initialized to
        # None before the try block, so this guard never raises NameError even when the
        # exception fired before context.new_page() ran.
        if page is not None:
            try:
                page.screenshot(path=screenshot_path, full_page=True)
            except Exception:
                pass

    passed = (
        headline_matched
        and component_count_matched
        and (not confirm_flow_ran or confirm_flow_passed is True)
        and error is None
    )

    return UIAssertionResult(
        passed=passed,
        headline_matched=headline_matched,
        component_count_matched=component_count_matched,
        confirm_flow_ran=confirm_flow_ran,
        confirm_flow_passed=confirm_flow_passed,
        screenshot_path=screenshot_path,
        error=error,
    )
```

**Build note on selectors:** Before committing, the build agent must check whether `data-testid` hooks are needed:

1. Check if any card component (ExerciseSetCard, ScoreHighlightCard, PlayPassageCard, SegmentLoopArtifactCard) already has `data-testid` attributes: `grep -r "data-testid" apps/web/src/components/cards/`
2. If none exist and the fallback text selectors are unreliable, add `data-testid="inline-card-exercise_set"` to the outermost `<div>` of `ExerciseSetCard`, `data-testid="inline-card-score_highlight"` to `ScoreHighlightCard`, etc. These are purely additive and have zero behavioral impact.
3. Add `data-testid="inline-card-exercise_set"` to `ExerciseSetCard`'s outermost div — in `apps/web/src/components/cards/ExerciseSetCard.tsx`, the outer div at line 219 (the `bg-surface-card` div):

```tsx
<div data-testid="inline-card-exercise_set" className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3">
```

4. Similarly for ScoreHighlightCard, PlayPassageCard, SegmentLoopArtifactCard — add `data-testid="inline-card-<type>"` to each outer container div.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
CRESCEND_LIVE_STACK=1 \
CRESCEND_TEST_CONV_ID=<conversation_id_from_task3> \
CRESCEND_TEST_SYNTHESIS_TEXT="<first_60_chars_of_synthesis>" \
CRESCEND_TEST_COMPONENT_COUNT=<count_of_renderable_components> \
  cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest e2e/tests/test_ui_verifier_live.py -v -m integration -s 2>&1 | tail -20
```

Expected: PASS — `UIAssertionResult.passed=True`, screenshot saved to `apps/evals/results/`.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-68-e2e-ui-session && git add apps/evals/e2e/ui_verifier.py apps/evals/e2e/tests/test_ui_verifier_live.py && git commit -m "feat(e2e): UIVerifier — Playwright render + confirm assertions for persisted sessions

verify_session_ui() hides browser auth/nav/wait/DOM/screenshot behind one function.
Headline text match + component card count + conditional confirm->reveal flow.
Screenshot always saved to apps/evals/results/.

Refs #68"
```

If `data-testid` hooks were added to web components:
```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-68-e2e-ui-session && git add apps/web/src/components/cards/ apps/evals/e2e/ui_verifier.py apps/evals/e2e/tests/test_ui_verifier_live.py && git commit -m "feat(e2e): UIVerifier + data-testid hooks for Playwright card selectors

Additive data-testid attributes on ExerciseSetCard, ScoreHighlightCard,
PlayPassageCard, SegmentLoopArtifactCard outer divs — no behavioral change.
verify_session_ui() uses testid selectors with text-based fallback.

Refs #68"
```

---

## Task 6: Confirm flow — verify prescription exists before asserting confirm->reveal

**Group:** D2 (sequential, depends on D — Task 6 imports `e2e.ui_verifier` which is created in Task 5; it CANNOT run in parallel with Task 5)
**Requires live stack:** Yes

**Behavior being verified:** When `has_pending_exercise=True` (i.e., the nocturne recording produced a prescription), clicking Confirm in the ReflectionMessage component causes the ExerciseSetCard to appear; when `has_pending_exercise=False`, the confirm flow is gracefully skipped.

**Interface under test:** `verify_session_ui(capture, ...)` — the conditional path in `UIAssertionResult.confirm_flow_ran` and `confirm_flow_passed`.

**Files:**
- Extend: `apps/evals/e2e/tests/test_ui_verifier_live.py` — add second test function

**Dependency note (build agent):** Task 6 depends on `apps/evals/e2e/ui_verifier.py` from Task 5. Do NOT dispatch Task 6 in parallel with Task 5 — `from e2e.ui_verifier import verify_session_ui` will hit `ModuleNotFoundError` if Task 5 has not committed. Run Task 5 to completion (commit), then dispatch Task 6.

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/e2e/tests/test_ui_verifier_live.py`:

```python
HAS_PENDING = os.environ.get("CRESCEND_TEST_HAS_PENDING", "") == "1"


@pytest.mark.integration
@pytest.mark.skipif(
    not LIVE_STACK or not CONV_ID or not HAS_PENDING,
    reason=(
        "Set CRESCEND_LIVE_STACK=1 CRESCEND_TEST_CONV_ID=<uuid> "
        "CRESCEND_TEST_HAS_PENDING=1 with `just dev` running"
    ),
)
def test_ui_verifier_confirm_flow():
    """When pending_exercise component present, confirm->reveal flow runs and passes."""
    from shared.local_session import PersistedSessionCapture
    from e2e.ui_verifier import verify_session_ui

    capture = PersistedSessionCapture(
        conversation_id=CONV_ID,
        session_id="test-session",
        synthesis_text=SYNTHESIS_TEXT,
        components=[
            {"type": "pending_exercise", "config": {"exerciseId": "ex-123", "focusDimension": "dynamics"}},
            {"type": "score_highlight", "config": {}},
        ],
        chunk_scores=[],
    )

    result = verify_session_ui(
        capture=capture,
        web_url="http://localhost:3000",
        api_url="http://localhost:8787",
        headed=False,
    )

    assert result.confirm_flow_ran, "Confirm flow must run when pending_exercise is present"
    assert result.confirm_flow_passed is True, (
        f"Confirm flow failed. Error: {result.error}. "
        f"Screenshot: {result.screenshot_path}"
    )
    assert result.passed, f"Overall assertion failed. Error: {result.error}"


@pytest.mark.integration
@pytest.mark.skipif(
    not LIVE_STACK or not CONV_ID,
    reason="Set CRESCEND_LIVE_STACK=1 CRESCEND_TEST_CONV_ID=<uuid> with `just dev` running",
)
def test_ui_verifier_no_pending_exercise_skips_confirm_flow():
    """When no pending_exercise component, confirm_flow_ran=False is not a failure."""
    from shared.local_session import PersistedSessionCapture
    from e2e.ui_verifier import verify_session_ui

    capture = PersistedSessionCapture(
        conversation_id=CONV_ID,
        session_id="test-session",
        synthesis_text=SYNTHESIS_TEXT,
        components=[{"type": "score_highlight", "config": {}}],
        chunk_scores=[],
    )

    result = verify_session_ui(
        capture=capture,
        web_url="http://localhost:3000",
        api_url="http://localhost:8787",
        headed=False,
    )

    assert not result.confirm_flow_ran, "confirm_flow_ran must be False when no pending_exercise"
    assert result.confirm_flow_passed is None, "confirm_flow_passed must be None when flow not ran"
    assert result.headline_matched, f"Headline must still match. Error: {result.error}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest e2e/tests/test_ui_verifier_live.py::test_ui_verifier_confirm_flow -v -m integration 2>&1 | tail -5
```

Expected: SKIP (env vars not set) or FAIL (ui_verifier not yet created, if parallel with Task 5).

- [ ] **Step 3: Implement**

Implementation is in `ui_verifier.py` from Task 5. The confirm flow logic is already included there. This task validates the conditional branching.

- [ ] **Step 4: Run test — verify it PASSES**

For the case where prescription IS present (set `CRESCEND_TEST_HAS_PENDING=1` based on Task 3 empirical result):
```bash
CRESCEND_LIVE_STACK=1 \
CRESCEND_TEST_CONV_ID=<conversation_id_from_task3> \
CRESCEND_TEST_SYNTHESIS_TEXT="<first_60_chars>" \
CRESCEND_TEST_HAS_PENDING=1 \
  cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest e2e/tests/test_ui_verifier_live.py::test_ui_verifier_confirm_flow -v -m integration -s 2>&1 | tail -10
```

For the case where prescription is NOT present:
```bash
CRESCEND_LIVE_STACK=1 \
CRESCEND_TEST_CONV_ID=<conversation_id_from_task3> \
CRESCEND_TEST_SYNTHESIS_TEXT="<first_60_chars>" \
  cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest e2e/tests/test_ui_verifier_live.py::test_ui_verifier_no_pending_exercise_skips_confirm_flow -v -m integration -s 2>&1 | tail -10
```

Expected: PASS in both cases.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-68-e2e-ui-session && git add apps/evals/e2e/tests/test_ui_verifier_live.py && git commit -m "test(e2e): confirm->reveal flow assertions + graceful skip when no prescription

Two paths: (1) pending_exercise present -> Confirm clicked -> ExerciseSetCard revealed;
(2) no pending_exercise -> confirm_flow_ran=False, still asserts headline + cards.

Refs #68"
```

---

## Task 7: Orchestrator CLI + `just e2e-ui-session` recipe

**Group:** E (depends on Group D — all assertions in place)
**Requires live stack:** No for writing; Yes for the final smoke run.

**Behavior being verified:** `python -m e2e.ui_session` runs Phase 1 (`drive_persisted`) then Phase 2 (`verify_session_ui`), prints a structured PASS/FAIL summary, saves screenshot, and exits 0 on PASS / 1 on FAIL.

**Interface under test:** CLI exit code and printed output — observable through `subprocess.run(["uv", "run", "python", "-m", "e2e.ui_session"])`.

**Files:**
- Create: `apps/evals/e2e/ui_session.py`
- Modify: `justfile` — add `e2e-ui-session` recipe

- [ ] **Step 1: Write the failing test**

There is no offline unit test for the orchestrator (it requires live services). The test is the `just` recipe smoke run described in Step 4. Instead, verify the module is importable and the CLI argument parser works:

Append to `apps/evals/e2e/tests/test_drive_persisted_capture.py`:

```python
def test_orchestrator_module_importable():
    """ui_session orchestrator module must be importable without side effects."""
    import importlib
    # Should not raise even without live services
    mod = importlib.import_module("e2e.ui_session")
    assert hasattr(mod, "main"), "ui_session must expose a main() function"
    assert hasattr(mod, "DEFAULT_RECORDING"), "ui_session must expose DEFAULT_RECORDING constant"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest e2e/tests/test_drive_persisted_capture.py::test_orchestrator_module_importable -x -q 2>&1 | tail -5
```

Expected: FAIL — `ModuleNotFoundError: No module named 'e2e.ui_session'`

- [ ] **Step 3: Implement `apps/evals/e2e/ui_session.py` and justfile recipe**

Create `apps/evals/e2e/ui_session.py`:

```python
"""E2E UI session orchestrator.

Phase 1: drive_persisted() — runs one WAV through the non-eval pipeline.
Phase 2: verify_session_ui() — asserts the web UI rendered the persisted synthesis.

Usage:
  cd apps/evals && uv run python -m e2e.ui_session [--headed] [--recording PATH] [--piece-slug SLUG]

Prerequisites:
  - `just dev` running (MuQ:8000, AMT:8001, API:8787, Web:3000)
  - `just seed-fingerprint` run (populates piece_index.json in local R2)
  - Playwright browsers installed: `uv run playwright install chromium`
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DEFAULT_RECORDING = (
    Path(__file__).resolve().parents[5]
    / "model" / "data" / "evals" / "practice_eval"
    / "nocturne_op9no2" / "audio" / "_aySCutsVVQ.wav"
)
DEFAULT_PIECE_SLUG = "chopin.nocturne_op9no2"
DEFAULT_WEB_URL = "http://localhost:3000"
DEFAULT_API_URL = "http://localhost:8787"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    """Run the full e2e UI session test. Returns exit code (0=PASS, 1=FAIL)."""
    parser = argparse.ArgumentParser(description="E2E persisted session UI test")
    parser.add_argument(
        "--recording",
        type=Path,
        default=DEFAULT_RECORDING,
        help=f"Path to WAV recording (default: {DEFAULT_RECORDING})",
    )
    parser.add_argument(
        "--piece-slug",
        default=DEFAULT_PIECE_SLUG,
        help=f"Piece slug for set_piece query (default: {DEFAULT_PIECE_SLUG})",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run Playwright in headed (visible) mode for debugging",
    )
    parser.add_argument(
        "--web-url",
        default=DEFAULT_WEB_URL,
        help=f"Web app URL (default: {DEFAULT_WEB_URL})",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=4,
        help="Maximum number of audio chunks to process (default: 4 = ~1 min)",
    )
    args = parser.parse_args()

    print(f"[e2e] Starting persisted session e2e test")
    print(f"[e2e] Recording: {args.recording}")
    print(f"[e2e] Piece slug: {args.piece_slug}")
    print(f"[e2e] API: {args.api_url} | Web: {args.web_url}")

    if not args.recording.exists():
        print(f"[e2e] FAIL: Recording not found: {args.recording}", file=sys.stderr)
        return 1

    from shared.local_session import PersistedSessionCapture, check_services, drive_persisted
    from e2e.ui_verifier import verify_session_ui

    # --- Phase 1: Service health check ---
    print(f"[e2e] Phase 1a: Checking services...")
    try:
        check_services(args.api_url)
    except RuntimeError as exc:
        print(f"[e2e] FAIL: Service check failed: {exc}", file=sys.stderr)
        return 1
    print(f"[e2e] Services reachable.")

    # --- Phase 1b: drive_persisted ---
    print(f"[e2e] Phase 1b: Driving recording through non-eval pipeline...")
    try:
        capture = drive_persisted(
            recording=args.recording,
            piece_slug=args.piece_slug,
            wrangler_url=args.api_url,
            max_chunks=args.max_chunks,
        )
    except RuntimeError as exc:
        print(f"[e2e] FAIL: drive_persisted raised: {exc}", file=sys.stderr)
        return 1

    print(f"[e2e] Session complete.")
    print(f"[e2e]   conversation_id: {capture.conversation_id}")
    print(f"[e2e]   session_id:      {capture.session_id}")
    print(f"[e2e]   synthesis_text:  {capture.synthesis_text[:100]!r}...")
    print(f"[e2e]   components:      {[c.get('type') for c in capture.components]}")
    print(f"[e2e]   chunk_scores:    {len(capture.chunk_scores)} chunks")
    if capture.piece_identification:
        print(f"[e2e]   piece_id:        {capture.piece_identification}")

    has_pending = any(c.get("type") == "pending_exercise" for c in capture.components)
    print(f"[e2e]   has_pending_exercise: {has_pending}")

    # Save session metadata for debugging
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = RESULTS_DIR / "e2e-session-latest.json"
    meta_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "conversation_id": capture.conversation_id,
        "session_id": capture.session_id,
        "synthesis_text": capture.synthesis_text,
        "components": capture.components,
        "has_pending_exercise": has_pending,
        "chunk_scores_count": len(capture.chunk_scores),
        "piece_identification": capture.piece_identification,
    }, indent=2))
    print(f"[e2e] Session metadata saved to {meta_path}")

    # --- Phase 2: UI verification ---
    print(f"[e2e] Phase 2: Verifying web UI rendered the persisted conversation...")
    result = verify_session_ui(
        capture=capture,
        web_url=args.web_url,
        api_url=args.api_url,
        headed=args.headed,
        screenshot_dir=RESULTS_DIR,
    )

    print(f"[e2e] Screenshot: {result.screenshot_path}")
    print(f"[e2e] headline_matched:       {result.headline_matched}")
    print(f"[e2e] component_count_matched: {result.component_count_matched}")
    print(f"[e2e] confirm_flow_ran:        {result.confirm_flow_ran}")
    print(f"[e2e] confirm_flow_passed:     {result.confirm_flow_passed}")

    if result.error:
        print(f"[e2e] Errors:\n{result.error}", file=sys.stderr)

    if result.passed:
        print(f"[e2e] PASS")
        return 0
    else:
        print(f"[e2e] FAIL", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

Add to `justfile` (after the `eval-e2e` recipe):

```makefile
# Run persisted-session e2e UI test (requires `just dev` + `just seed-fingerprint`).
# Drives one real WAV through MuQ+AMT+V6 as a non-eval session, persists to Postgres,
# then asserts web UI rendered the synthesis correctly via Playwright.
# Pass --headed for a visible browser, --max-chunks N to limit audio duration.
e2e-ui-session *ARGS:
    cd apps/evals && uv run python -m e2e.ui_session {{ARGS}}
```

- [ ] **Step 4: Run test — verify it PASSES**

First, offline:
```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m pytest e2e/tests/test_drive_persisted_capture.py::test_orchestrator_module_importable -x -q 2>&1 | tail -5
```
Expected: PASS

Then, full live smoke (with services running):
```bash
cd /Users/jdhiman/Documents/crescendai && just e2e-ui-session 2>&1
```
Expected: Prints `[e2e] PASS` and exits 0.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-68-e2e-ui-session && git add apps/evals/e2e/ui_session.py justfile apps/evals/e2e/tests/test_drive_persisted_capture.py && git commit -m "feat(e2e): orchestrator CLI + just e2e-ui-session recipe

Thin CLI wires drive_persisted() -> verify_session_ui(). Saves session metadata
to apps/evals/results/e2e-session-latest.json and screenshot to
apps/evals/results/e2e-ui-session-<timestamp>.png.
Exit 0 = PASS, exit 1 = FAIL with diagnostic output.

Closes #68"
```

---

## Verification Checklist

After all tasks complete, verify:

- [ ] `cd apps/evals && uv run python -m pytest pipeline/exercise_routing/tests/test_local_session_smoke.py -v` — all original + 3 new tests PASS
- [ ] `cd apps/evals && uv run python -m pytest e2e/tests/test_drive_persisted_capture.py -v` — 4 offline + 1 import test PASS
- [ ] `just e2e-ui-session` exits 0 with `[e2e] PASS` (requires `just dev` + `just seed-fingerprint`)
- [ ] Screenshot saved to `apps/evals/results/e2e-ui-session-<timestamp>.png`
- [ ] `apps/evals/results/e2e-session-latest.json` contains non-empty `conversation_id` and `synthesis_text`
- [ ] `drive()` and `SessionCapture` in `local_session.py` are unchanged (no regression to existing eval path)
