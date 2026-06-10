#!/usr/bin/env python3
"""Local end-to-end piece-ID verification through the REAL chunk_ready path.

Unlike the eval harness (which injects pre-computed notes via `eval_chunk`), this
driver exercises the PRODUCTION audio pipeline exactly as the web/iOS clients do:

    WAV slice -> POST /api/practice/chunk (R2) -> WS chunk_ready -> DO fetch audio
    -> local MuQ (:8000) + AMT (:8001) -> perfNotes -> accumulateAndIdentify
    -> buffer >= 30 -> fetch fingerprint/v2 -> wasm.identifyPiece -> lock
    -> piece_identified WS  -> (next chunk) loadScoreContext -> alignChunkChroma
    -> chunk_bar_map WS

Observable proof:
  * piece_identified  -> the certified gate locked the piece (correct pieceId).
  * chunk_bar_map     -> chroma-DTW score-following engaged against the locked
                         piece's score context (only emitted post-lock, >= chunk 1).

The driver STOPS before end_session, so the teacher-LLM synthesis never runs
(no Anthropic spend). It asserts nothing about synthesis.

Prereqs (see justfile):
    just amt &  just muq &  just api          # 8001 / 8000 / 8787
    just fingerprint && just seed-fingerprint # fingerprint/v2 in local R2
    just seed-score-json                      # scores/v1/{pieceId}.json in local R2
Env:
    EVAL_SHARED_SECRET must match apps/api/.dev.vars (the DO eval-identity
    override is fail-closed and rejects an empty/wrong secret).

Usage:
    EVAL_SHARED_SECRET=... uv run --with soundfile --with numpy --with requests \
        --with websockets python apps/evals/piece_id_e2e.py \
        --wav model/data/evals/practice_eval/fur_elise/audio/wRgMrEvH01E.wav \
        --expect beethoven.fur_elise --chunks 3

    # negative (leave-one-out index seeded first): expect no lock
    ... python apps/evals/piece_id_e2e.py --wav <same> --expect-unknown --chunks 3
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
from urllib.parse import urlparse

import requests
import soundfile as sf
import websockets

CHUNK_SECONDS = 15  # production MuQ chunk size (apps/web MediaRecorder timeslice)


def _slice_wav(path: str, n_chunks: int, skip_seconds: float) -> list[bytes]:
    """Slice a WAV into N consecutive CHUNK_SECONDS windows, each re-encoded as WAV.

    Skips the first `skip_seconds` (intro applause/silence often transcribes to junk).
    Raises if the file is too short for the requested number of chunks.
    """
    audio, sr = sf.read(path)
    start = int(skip_seconds * sr)
    span = int(CHUNK_SECONDS * sr)
    needed = start + span * n_chunks
    if len(audio) < needed:
        raise ValueError(
            f"{path}: {len(audio)/sr:.1f}s is too short for {n_chunks} x {CHUNK_SECONDS}s "
            f"after a {skip_seconds:.0f}s skip (need {needed/sr:.0f}s)."
        )
    chunks: list[bytes] = []
    for i in range(n_chunks):
        seg = audio[start + i * span : start + (i + 1) * span]
        buf = io.BytesIO()
        sf.write(buf, seg, sr, format="WAV")
        chunks.append(buf.getvalue())
    return chunks


def _auth(base_url: str) -> requests.Session:
    """Local debug auth: returns a requests.Session carrying the auth cookie."""
    s = requests.Session()
    resp = s.post(f"{base_url}/api/auth/debug", timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"debug auth failed: {resp.status_code} {resp.text[:200]}")
    return s


async def run(
    base_url: str,
    wav: str,
    n_chunks: int,
    skip_seconds: float,
    secret: str,
    student_id: str,
) -> dict:
    if not secret:
        raise RuntimeError(
            "EVAL_SHARED_SECRET is not set (export it to match apps/api/.dev.vars)."
        )
    if "localhost" not in base_url and "127.0.0.1" not in base_url:
        raise ValueError(f"refusing to target non-local host: {base_url}")

    chunks = _slice_wav(wav, n_chunks, skip_seconds)
    auth = _auth(base_url)

    resp = auth.post(
        f"{base_url}/api/practice/start",
        json={},
        headers={"x-eval-secret": secret},
        timeout=15,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"start failed: {resp.status_code} {resp.text[:200]}")
    session_id = resp.json()["sessionId"]

    # Upload every chunk to R2 via the real /chunk route (writes the same key the
    # DO reads: sessions/{id}/chunks/{i}.webm). WAV bytes under a .webm key is fine
    # -- MuQ/AMT sniff the container from content, not the extension.
    for i, audio in enumerate(chunks):
        up = auth.post(
            f"{base_url}/api/practice/chunk",
            params={"sessionId": session_id, "chunkIndex": i},
            data=audio,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )
        if up.status_code != 200:
            raise RuntimeError(f"chunk {i} upload failed: {up.status_code} {up.text[:200]}")

    parsed = urlparse(base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = (
        f"{ws_scheme}://{parsed.netloc}/api/practice/ws/{session_id}"
        f"?eval=true&evalStudentId={student_id}"
    )
    ws_headers = {"x-eval-secret": secret}
    cookie = "; ".join(f"{k}={v}" for k, v in auth.cookies.items())
    if cookie:
        ws_headers["Cookie"] = cookie

    piece_identified: dict | None = None
    bar_maps: list[dict] = []
    chunk_processed: list[int] = []
    errors: list[str] = []

    try:
        async with websockets.connect(ws_url, additional_headers=ws_headers) as ws:
            for i in range(len(chunks)):
                r2_key = f"sessions/{session_id}/chunks/{i}.webm"
                await ws.send(json.dumps({"type": "chunk_ready", "index": i, "r2Key": r2_key}))
                print(f"  -> sent chunk_ready index={i}", flush=True)
                # Drain until this chunk's terminal chunk_processed; capture piece_identified
                # and chunk_bar_map that arrive alongside it.
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=120.0)
                    except asyncio.TimeoutError:
                        errors.append(f"timeout waiting on chunk {i}")
                        break
                    msg = json.loads(raw)
                    t = msg.get("type", "")
                    print(f"  <- {t} {json.dumps({k: v for k, v in msg.items() if k != 'type'})[:160]}", flush=True)
                    if t == "piece_identified" and piece_identified is None:
                        piece_identified = msg
                    elif t == "chunk_bar_map":
                        bar_maps.append(msg)
                    elif t == "chunk_processed":
                        chunk_processed.append(msg.get("index", i))
                        break
                    elif t == "error":
                        errors.append(msg.get("message", "unknown ws error"))
                        break
            # STOP here -- deliberately no end_session, so synthesis never fires.
    except websockets.exceptions.ConnectionClosed as e:
        errors.append(
            f"ws closed: code={e.code} reason={e.reason!r} "
            f"(after {len(chunk_processed)} chunk_processed)"
        )

    return {
        "session_id": session_id,
        "chunks_sent": len(chunks),
        "chunks_processed": chunk_processed,
        "piece_identified": piece_identified,
        "bar_maps": bar_maps,
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--base-url", default="http://localhost:8787")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--skip-seconds", type=float, default=8.0)
    ap.add_argument("--student-id", default="e2e-piece-id-001")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--expect", help="expected locked pieceId (positive case)")
    group.add_argument(
        "--expect-unknown",
        action="store_true",
        help="assert NO lock (off-catalog / leave-one-out negative case)",
    )
    args = ap.parse_args()

    result = asyncio.run(
        run(
            base_url=args.base_url,
            wav=args.wav,
            n_chunks=args.chunks,
            skip_seconds=args.skip_seconds,
            secret=os.environ.get("EVAL_SHARED_SECRET", ""),
            student_id=args.student_id,
        )
    )

    print(json.dumps(result, indent=2))

    pid = result["piece_identified"]
    locked_id = pid.get("pieceId") if pid else None

    ok = True
    if args.expect_unknown:
        if pid is not None:
            print(f"\nFAIL: expected NO lock, but locked onto {locked_id!r} "
                  f"(confidence={pid.get('confidence')})")
            ok = False
        else:
            print("\nPASS: stayed unknown (no false lock).")
    else:
        if pid is None:
            print(f"\nFAIL: expected lock onto {args.expect!r}, but stayed unknown.")
            ok = False
        elif locked_id != args.expect:
            print(f"\nFAIL: locked onto {locked_id!r}, expected {args.expect!r}.")
            ok = False
        else:
            print(f"\nPASS: locked onto {locked_id!r} "
                  f"(confidence={pid.get('confidence'):.4f}, method={pid.get('method')}).")
            if result["bar_maps"]:
                bm = result["bar_maps"][0]
                print(f"PASS: score-following engaged -- {len(result['bar_maps'])} "
                      f"chunk_bar_map event(s), first bars [{bm.get('bar_min')}, {bm.get('bar_max')}].")
            else:
                print("WARN: locked, but no chunk_bar_map seen "
                      "(score JSON not seeded, or <2 chunks, or alignment failed).")
                ok = False

    if result["errors"]:
        print(f"\nerrors: {result['errors']}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
