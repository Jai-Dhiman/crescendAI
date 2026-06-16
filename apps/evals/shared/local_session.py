"""Deep driver: one WAV -> SessionCapture over the real chunk_ready path.

Hides WS connect, eval-identity headers, ffmpeg chunking, local R2 upload,
chunk_ready message loop, synthesis event parsing, and eval_context deserialization.
The caller hands over a WAV path and piece slug; the caller receives a SessionCapture.

drive() raises RuntimeError if services are unavailable — the health check fires
before any WS connection attempt.

SessionCapture is defined in pipeline.exercise_routing.score and re-exported here
so callers can import it from either location without duplication.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import websockets

from pipeline.exercise_routing.score import SessionCapture  # single definition
from shared.pipeline_client import _get_auth_session

# Re-export so callers can do `from shared.local_session import SessionCapture`
__all__ = ["SessionCapture", "drive", "check_services", "read_eval_secret"]

CHUNK_SECONDS = 15
R2_BUCKET = "crescendai-bucket"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_API_DIR = REPO_ROOT / "apps" / "api"
DEFAULT_DEV_VARS = DEFAULT_API_DIR / ".dev.vars"


def read_eval_secret(dev_vars: Path = DEFAULT_DEV_VARS) -> str:
    """Read EVAL_SHARED_SECRET from apps/api/.dev.vars. Raises if absent or empty."""
    if not dev_vars.exists():
        raise FileNotFoundError(
            f"apps/api/.dev.vars not found at {dev_vars}. "
            "Cannot read EVAL_SHARED_SECRET."
        )
    for line in dev_vars.read_text().splitlines():
        line = line.strip()
        if line.startswith("EVAL_SHARED_SECRET="):
            secret = line.split("=", 1)[1].strip().strip('"').strip("'")
            if not secret:
                raise ValueError("EVAL_SHARED_SECRET is empty in .dev.vars")
            return secret
    raise KeyError("EVAL_SHARED_SECRET not present in .dev.vars")


def check_services(wrangler_url: str) -> None:
    """Raise RuntimeError if the API or MuQ/AMT services are not reachable.

    Called before any recording is processed so the operator gets a single clear
    error rather than 22 per-recording timeouts.
    """
    import requests
    try:
        resp = requests.get(f"{wrangler_url}/health", timeout=5)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"API not reachable at {wrangler_url}/health. "
            "Run `just dev` (or `just api`) before running the eval harness."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(
            f"API health check timed out at {wrangler_url}/health after 5s. "
            "Ensure `just dev` is running and responsive."
        ) from exc
    if resp.status_code != 200:
        raise RuntimeError(
            f"API health check failed: {resp.status_code}. "
            "Ensure `just dev` is running and `just seed-fingerprint` has been run."
        )


def _slice_to_webm_chunks(wav: Path, out_dir: Path, max_chunks: int) -> list[Path]:
    """ffmpeg-segment WAV into 15s Opus/WebM independently-decodable chunks."""
    pattern = str(out_dir / "chunk_%03d.webm")
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(wav),
            "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
            "-c:a", "libopus", "-b:a", "96k", "-ac", "1",
            pattern,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {wav}:\n{result.stderr}")
    chunks = sorted(out_dir.glob("chunk_*.webm"))
    if not chunks:
        raise RuntimeError(f"ffmpeg produced no chunks from {wav}")
    return chunks[:max_chunks]


def _upload_chunk_to_r2(api_dir: Path, r2_key: str, file_path: Path) -> None:
    """Write one chunk into local R2 at the key the DO reads (chunk_ready path)."""
    result = subprocess.run(
        [
            "wrangler", "r2", "object", "put",
            f"{R2_BUCKET}/{r2_key}",
            f"--file={file_path}",
            "--local",
        ],
        cwd=api_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"wrangler r2 put failed for {r2_key}:\n{result.stderr}\n"
            "Is `just dev` (or `just api`) running?"
        )


async def _drive_async(
    recording: Path,
    piece_slug: str,
    session_id: str,
    r2_keys: list[str],
    wrangler_url: str,
    eval_secret: str,
    timeout_per_event: float,
) -> dict:
    """Internal async driver. Returns raw event dict with synthesis and piece_identification."""
    parsed = urlparse(wrangler_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = (
        f"{ws_scheme}://{parsed.netloc}/api/practice/ws/{session_id}"
        f"?eval=true&evalStudentId=eval-routing-harness"
    )

    auth = _get_auth_session(wrangler_url)
    headers = {"x-eval-secret": eval_secret}
    token = auth.headers.get("Authorization", "")
    if token:
        headers["Authorization"] = token
    cookie_str = "; ".join(f"{k}={v}" for k, v in auth.cookies.items())
    if cookie_str:
        headers["Cookie"] = cookie_str

    piece_identification: dict | None = None
    synthesis_event: dict | None = None

    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        await ws.send(json.dumps({"type": "set_piece", "query": piece_slug}))

        for idx, r2_key in enumerate(r2_keys):
            await ws.send(json.dumps({"type": "chunk_ready", "index": idx, "r2Key": r2_key}))
            # Wait for chunk_processed before sending next chunk
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_per_event)
                    evt = json.loads(raw)
                    etype = evt.get("type")
                    if etype == "chunk_processed":
                        break
                    elif etype == "piece_identified":
                        piece_identification = evt
                    elif etype == "error":
                        raise RuntimeError(
                            f"DO returned error during chunk {idx}: {evt.get('message')}"
                        )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Timeout after {timeout_per_event}s waiting for chunk_processed "
                    f"for chunk {idx} of {recording}. Is MuQ:8000 warm?"
                )

        await ws.send(json.dumps({"type": "end_session"}))

        # Wait for synthesis
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout_per_event)
                evt = json.loads(raw)
                etype = evt.get("type")
                if etype == "synthesis":
                    synthesis_event = evt
                    break
                elif etype == "piece_identified":
                    piece_identification = evt
                elif etype == "error":
                    raise RuntimeError(f"DO returned error: {evt.get('message')}")
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Timeout after {timeout_per_event}s waiting for synthesis "
                f"for {recording}. Is MuQ:8000 warm?"
            )
        except websockets.exceptions.ConnectionClosed:
            pass  # Server closed connection after synthesis; check synthesis_event below

    if synthesis_event is None:
        raise RuntimeError(f"No synthesis received for {recording}")

    return {"synthesis": synthesis_event, "piece_identification": piece_identification}


def drive(
    recording: Path,
    piece_slug: str,
    wrangler_url: str = "http://localhost:8787",
    api_dir: Path = DEFAULT_API_DIR,
    eval_secret: str | None = None,
    timeout_per_event: float = 120.0,
    max_chunks: int = 6,
) -> SessionCapture:
    """Drive one WAV through the real chunk_ready path; return a SessionCapture.

    Raises RuntimeError if services are unreachable, ffmpeg fails, wrangler
    r2 put fails, or no synthesis is received within timeout.
    """
    import requests as _requests

    if eval_secret is None:
        eval_secret = read_eval_secret()

    auth = _get_auth_session(wrangler_url)
    resp = auth.post(
        f"{wrangler_url}/api/practice/start",
        json={},
        headers={"x-eval-secret": eval_secret},
        timeout=10,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to start practice session: {resp.status_code} {resp.text}"
        )
    session_id = resp.json().get("sessionId")
    if not session_id:
        raise RuntimeError(
            f"POST /api/practice/start response missing sessionId: {resp.text}"
        )

    with tempfile.TemporaryDirectory(prefix="crescend-routing-eval-") as tmp:
        tmp_dir = Path(tmp)
        chunks = _slice_to_webm_chunks(recording, tmp_dir, max_chunks)

        r2_keys: list[str] = []
        for idx, chunk_path in enumerate(chunks):
            r2_key = f"sessions/{session_id}/chunks/{idx}.webm"
            _upload_chunk_to_r2(api_dir, r2_key, chunk_path)
            r2_keys.append(r2_key)

        result = asyncio.run(
            _drive_async(
                recording=recording,
                piece_slug=piece_slug,
                session_id=session_id,
                r2_keys=r2_keys,
                wrangler_url=wrangler_url,
                eval_secret=eval_secret,
                timeout_per_event=timeout_per_event,
            )
        )

    synth = result["synthesis"]
    piece_id_evt = result["piece_identification"]
    eval_ctx = synth.get("eval_context", {})

    piece_identification = None
    if piece_id_evt:
        piece_identification = {
            "pieceId": piece_id_evt.get("pieceId", ""),
            "confidence": piece_id_evt.get("confidence", 0.0),
        }

    teaching_moments = eval_ctx.get("teaching_moments", [])
    baselines = eval_ctx.get("baselines", {})
    prescribed_exercise = eval_ctx.get("prescribed_exercise")

    # dominant_dimension: evalContext has teaching_moments but no artifact.dominant_dimension.
    # Use the top teaching moment's dimension as the proxy (matches score.py scoring logic).
    dominant_dimension = None
    if teaching_moments:
        dominant_dimension = teaching_moments[0].get("dimension")

    return SessionCapture(
        session_id=session_id,
        recording=recording,
        piece_slug=piece_slug,
        teaching_moments=teaching_moments,
        baselines=baselines,
        piece_identification=piece_identification,
        piece_resolved=piece_identification is not None,
        dominant_dimension=dominant_dimension,
        prescribed_exercise=prescribed_exercise,
        synthesis_text=synth.get("text", ""),
    )
