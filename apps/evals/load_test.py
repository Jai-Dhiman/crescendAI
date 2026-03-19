# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "websockets>=13.0",
#     "httpx>=0.27.0",
#     "rich>=13.0",
# ]
# ///
"""Load test: long recording through inference + API pipeline.

Takes an audio file, chunks it into 15s segments, runs local MuQ+AMT
inference on each chunk, then feeds results through the API's Durable Object
practice session via WebSocket eval_chunk messages.

Two modes:
  --mode inference-only   Just run local inference, report per-chunk timings + scores
  --mode full-pipeline    Inference + feed through API via WebSocket (requires wrangler dev)

Usage:
    # Inference only (no API needed):
    cd apps/evals
    CRESCEND_DEVICE=mps uv run python load_test.py /path/to/audio.wav

    # Full pipeline (start wrangler dev + local_server first):
    cd apps/evals
    CRESCEND_DEVICE=mps uv run python load_test.py /path/to/audio.wav --mode full-pipeline

    # With piece context (enables score following + Tier 1 analysis):
    cd apps/evals
    uv run python load_test.py /path/to/audio.wav --mode full-pipeline --piece "Chopin Nocturne Op. 9 No. 2"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
from paths import INFERENCE_DIR

sys.path.insert(0, str(INFERENCE_DIR))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    index: int
    inference_time_ms: int = 0
    predictions: dict = field(default_factory=dict)
    note_count: int = 0
    pedal_event_count: int = 0
    midi_notes: list = field(default_factory=list)
    pedal_events: list = field(default_factory=list)
    # API pipeline results (full-pipeline mode only)
    api_process_time_ms: int = 0
    api_scores: dict = field(default_factory=dict)
    observation: str | None = None
    observation_dimension: str | None = None
    observation_framing: str | None = None
    bar_range: list | None = None
    error: str | None = None


@dataclass
class SessionReport:
    audio_file: str
    audio_duration_s: float
    chunk_count: int
    mode: str
    piece: str | None
    chunks: list[ChunkResult] = field(default_factory=list)
    total_inference_time_ms: int = 0
    total_api_time_ms: int = 0
    observations: list[dict] = field(default_factory=list)
    session_summary: dict | None = None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_local_inference(audio_file: str) -> tuple[list[ChunkResult], float]:
    """Chunk audio and run MuQ + AMT on each chunk locally."""
    from audio_chunker import chunk_audio_file
    from constants import PERCEPIANO_DIMENSIONS
    from models.inference import extract_muq_embeddings, predict_with_ensemble
    from models.loader import _resolve_device, get_model_cache
    from models.transcription import TranscriptionError, TranscriptionModel

    checkpoint_dir = Path(INFERENCE_DIR).parents[1] / "model" / "data" / "checkpoints" / "model_improvement" / "A1"

    print(f"Loading models...")
    device = os.environ.get("CRESCEND_DEVICE", "auto")

    cache = get_model_cache()
    cache.initialize(device=device, checkpoint_dir=checkpoint_dir)
    print(f"  MuQ + {len(cache.muq_heads)} prediction heads loaded")

    resolved_device = str(_resolve_device(device))
    try:
        transcription = TranscriptionModel(device=resolved_device)
    except RuntimeError as e:
        if "mps" in str(e).lower():
            print(f"  AMT failed on {resolved_device}, falling back to CPU")
            transcription = TranscriptionModel(device="cpu")
        else:
            raise
    print(f"  AMT loaded")

    print(f"\nChunking {audio_file}...")
    chunks = chunk_audio_file(audio_file, max_duration=3600)
    audio_duration = len(chunks) * 15.0  # approximate
    print(f"  {len(chunks)} chunks ({audio_duration:.0f}s)")

    results = []
    for i, audio in enumerate(chunks):
        t0 = time.time()

        # MuQ
        embeddings = extract_muq_embeddings(audio, cache)
        predictions = predict_with_ensemble(embeddings, cache)
        pred_dict = {
            dim: round(float(predictions[j]), 4)
            for j, dim in enumerate(PERCEPIANO_DIMENSIONS)
        }

        # AMT
        midi_notes = []
        pedal_events = []
        try:
            midi_notes, pedal_events = transcription.transcribe(audio, 24000)
        except TranscriptionError as e:
            print(f"  Chunk {i}: AMT failed: {e}")

        elapsed_ms = int((time.time() - t0) * 1000)

        result = ChunkResult(
            index=i,
            inference_time_ms=elapsed_ms,
            predictions=pred_dict,
            note_count=len(midi_notes),
            pedal_event_count=len(pedal_events),
            midi_notes=midi_notes,
            pedal_events=pedal_events,
        )
        results.append(result)

        # Print progress
        scores_str = "  ".join(f"{d[:3]}={v:.2f}" for d, v in pred_dict.items())
        print(f"  [{i+1}/{len(chunks)}] {elapsed_ms}ms | {scores_str} | {len(midi_notes)} notes")

    return results, audio_duration


# ---------------------------------------------------------------------------
# API pipeline (WebSocket)
# ---------------------------------------------------------------------------

async def run_api_pipeline(
    chunk_results: list[ChunkResult],
    api_base: str,
    piece: str | None,
) -> tuple[list[dict], dict | None]:
    """Feed pre-computed inference results through the API via WebSocket."""
    import httpx
    import websockets

    # 1. Auth (debug login)
    print(f"\nAuthenticating with {api_base}...")
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{api_base}/api/auth/debug")
        if resp.status_code != 200:
            raise RuntimeError(f"Debug auth failed: {resp.status_code} {resp.text}")
        cookie = resp.headers.get("set-cookie", "")
        if not cookie:
            raise RuntimeError("No session cookie returned from debug auth")
        # Extract just the cookie value
        session_cookie = cookie.split(";")[0]
        print(f"  Authenticated (cookie: {session_cookie[:40]}...)")

    # 2. Start session
    async with httpx.AsyncClient(headers={"Cookie": session_cookie}) as client:
        resp = await client.post(f"{api_base}/api/practice/start")
        if resp.status_code != 200:
            raise RuntimeError(f"Start session failed: {resp.status_code} {resp.text}")
        session_id = resp.json()["sessionId"]
        print(f"  Session: {session_id}")

    # 3. Connect WebSocket
    ws_scheme = "ws" if "localhost" in api_base or "127.0.0.1" in api_base else "wss"
    ws_host = api_base.replace("http://", "").replace("https://", "")
    ws_url = f"{ws_scheme}://{ws_host}/api/practice/ws/{session_id}"

    print(f"  Connecting to {ws_url}...")
    ws = await websockets.connect(
        ws_url,
        additional_headers={"Cookie": session_cookie},
    )

    # Wait for connected message
    msg = json.loads(await ws.recv())
    if msg.get("type") != "connected":
        raise RuntimeError(f"Expected 'connected', got: {msg}")
    print(f"  WebSocket connected")

    # 4. Set piece if provided
    if piece:
        await ws.send(json.dumps({"type": "set_piece", "query": piece}))
        msg = json.loads(await ws.recv())
        print(f"  Piece set: {msg.get('query', 'unknown')}")

    # 5. Feed chunks via eval_chunk
    observations = []
    print(f"\nFeeding {len(chunk_results)} chunks through pipeline...")

    for cr in chunk_results:
        eval_msg = {
            "type": "eval_chunk",
            "chunk_index": cr.index,
            "predictions": cr.predictions,
            "midi_notes": cr.midi_notes,
            "pedal_events": cr.pedal_events,
        }

        t0 = time.time()
        await ws.send(json.dumps(eval_msg))

        # Collect responses until we get chunk_processed
        while True:
            try:
                resp_text = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except asyncio.TimeoutError:
                cr.error = "timeout waiting for chunk_processed"
                print(f"  [{cr.index}] TIMEOUT")
                break

            resp = json.loads(resp_text)
            msg_type = resp.get("type")

            if msg_type == "chunk_processed":
                cr.api_process_time_ms = int((time.time() - t0) * 1000)
                cr.api_scores = resp.get("scores", {})
                scores_str = "  ".join(
                    f"{d[:3]}={v:.2f}" for d, v in cr.api_scores.items() if isinstance(v, (int, float))
                )
                print(f"  [{cr.index+1}/{len(chunk_results)}] {cr.api_process_time_ms}ms | {scores_str}")
                break

            elif msg_type == "observation":
                obs = {
                    "chunk_index": cr.index,
                    "text": resp.get("text", ""),
                    "dimension": resp.get("dimension", ""),
                    "framing": resp.get("framing", ""),
                    "bar_range": resp.get("barRange"),
                }
                observations.append(obs)
                cr.observation = resp.get("text")
                cr.observation_dimension = resp.get("dimension")
                cr.observation_framing = resp.get("framing")
                cr.bar_range = resp.get("barRange")
                print(f"    OBSERVATION [{resp.get('dimension')}]: {resp.get('text', '')[:80]}...")
                # Keep reading -- chunk_processed should follow

            elif msg_type == "error":
                cr.error = resp.get("message", "unknown error")
                print(f"  [{cr.index}] ERROR: {cr.error}")
                break

    # 6. End session
    print(f"\nEnding session...")
    await ws.send(json.dumps({"type": "end_session"}))

    session_summary = None
    try:
        while True:
            resp_text = await asyncio.wait_for(ws.recv(), timeout=30.0)
            resp = json.loads(resp_text)
            if resp.get("type") == "session_summary":
                session_summary = resp
                print(f"  Session summary: {resp.get('total_chunks', '?')} chunks, "
                      f"{resp.get('inference_failures', 0)} failures, "
                      f"{len(resp.get('observations', []))} observations")
                break
            elif resp.get("type") == "observation":
                obs = {
                    "text": resp.get("text", ""),
                    "dimension": resp.get("dimension", ""),
                    "framing": resp.get("framing", ""),
                }
                observations.append(obs)
                print(f"    OBSERVATION [{resp.get('dimension')}]: {resp.get('text', '')[:80]}...")
    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
        print(f"  Session ended (connection closed)")

    await ws.close()
    return observations, session_summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(report: SessionReport) -> None:
    """Print a formatted report to terminal."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Header
    console.print(f"\n{'='*70}")
    console.print(f"  LOAD TEST REPORT", style="bold")
    console.print(f"{'='*70}")
    console.print(f"  File:     {report.audio_file}")
    console.print(f"  Duration: {report.audio_duration_s:.0f}s ({report.chunk_count} chunks)")
    console.print(f"  Mode:     {report.mode}")
    if report.piece:
        console.print(f"  Piece:    {report.piece}")

    # Inference timing
    inf_times = [c.inference_time_ms for c in report.chunks if c.inference_time_ms > 0]
    if inf_times:
        console.print(f"\n  INFERENCE")
        console.print(f"    Total:   {sum(inf_times)}ms ({sum(inf_times)/1000:.1f}s)")
        console.print(f"    Average: {sum(inf_times)//len(inf_times)}ms per chunk")
        console.print(f"    Min:     {min(inf_times)}ms")
        console.print(f"    Max:     {max(inf_times)}ms")
        rtf = (sum(inf_times) / 1000) / report.audio_duration_s
        console.print(f"    RTF:     {rtf:.2f}x (real-time factor)")

    # API timing
    api_times = [c.api_process_time_ms for c in report.chunks if c.api_process_time_ms > 0]
    if api_times:
        console.print(f"\n  API PIPELINE")
        console.print(f"    Total:   {sum(api_times)}ms ({sum(api_times)/1000:.1f}s)")
        console.print(f"    Average: {sum(api_times)//len(api_times)}ms per chunk")
        console.print(f"    Min:     {min(api_times)}ms")
        console.print(f"    Max:     {max(api_times)}ms")

    # Per-chunk scores table
    console.print()
    table = Table(title="Per-Chunk Results")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Inf (ms)", justify="right")
    table.add_column("API (ms)", justify="right")
    table.add_column("Dyn", justify="right")
    table.add_column("Tim", justify="right")
    table.add_column("Ped", justify="right")
    table.add_column("Art", justify="right")
    table.add_column("Phr", justify="right")
    table.add_column("Int", justify="right")
    table.add_column("Notes", justify="right")
    table.add_column("Observation", max_width=40)

    for c in report.chunks:
        scores = c.predictions or c.api_scores
        obs_text = ""
        if c.observation:
            obs_text = f"[{c.observation_dimension}] {c.observation[:35]}..."
        elif c.error:
            obs_text = f"ERR: {c.error[:35]}"

        table.add_row(
            str(c.index),
            str(c.inference_time_ms) if c.inference_time_ms else "-",
            str(c.api_process_time_ms) if c.api_process_time_ms else "-",
            f"{scores.get('dynamics', 0):.2f}",
            f"{scores.get('timing', 0):.2f}",
            f"{scores.get('pedaling', 0):.2f}",
            f"{scores.get('articulation', 0):.2f}",
            f"{scores.get('phrasing', 0):.2f}",
            f"{scores.get('interpretation', 0):.2f}",
            str(c.note_count),
            obs_text,
        )

    console.print(table)

    # Observations
    if report.observations:
        console.print(f"\n  OBSERVATIONS ({len(report.observations)} total)")
        for i, obs in enumerate(report.observations):
            console.print(f"\n  [{i+1}] [{obs.get('dimension', '?')}] ({obs.get('framing', '?')})")
            if obs.get("bar_range"):
                console.print(f"      Bars: {obs['bar_range']}")
            console.print(f"      {obs.get('text', '')}")

    console.print(f"\n{'='*70}\n")


def save_report(report: SessionReport, output_path: str) -> None:
    """Save report as JSON for later analysis."""
    data = {
        "audio_file": report.audio_file,
        "audio_duration_s": report.audio_duration_s,
        "chunk_count": report.chunk_count,
        "mode": report.mode,
        "piece": report.piece,
        "total_inference_time_ms": sum(c.inference_time_ms for c in report.chunks),
        "total_api_time_ms": sum(c.api_process_time_ms for c in report.chunks),
        "chunks": [
            {
                "index": c.index,
                "inference_time_ms": c.inference_time_ms,
                "api_process_time_ms": c.api_process_time_ms,
                "predictions": c.predictions,
                "note_count": c.note_count,
                "pedal_event_count": c.pedal_event_count,
                "api_scores": c.api_scores,
                "observation": c.observation,
                "observation_dimension": c.observation_dimension,
                "observation_framing": c.observation_framing,
                "bar_range": c.bar_range,
                "error": c.error,
            }
            for c in report.chunks
        ],
        "observations": report.observations,
        "session_summary": report.session_summary,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Report saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Load test: long recording through inference + API pipeline"
    )
    parser.add_argument("audio_file", help="Path to audio file (WAV, MP3, WebM, etc.)")
    parser.add_argument(
        "--mode",
        choices=["inference-only", "full-pipeline"],
        default="inference-only",
        help="inference-only: local MuQ+AMT only. full-pipeline: inference + API WebSocket.",
    )
    parser.add_argument(
        "--piece",
        help="Piece name for score following (e.g. 'Chopin Nocturne Op. 9 No. 2')",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8787",
        help="API base URL (default: http://localhost:8787)",
    )
    parser.add_argument(
        "--output",
        help="Save JSON report to this path (default: reports/load_test_<timestamp>.json)",
    )
    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        print(f"Error: {args.audio_file} not found")
        sys.exit(1)

    # Run inference
    chunk_results, audio_duration = run_local_inference(args.audio_file)

    report = SessionReport(
        audio_file=args.audio_file,
        audio_duration_s=audio_duration,
        chunk_count=len(chunk_results),
        mode=args.mode,
        piece=args.piece,
        chunks=chunk_results,
    )

    # Optionally feed through API
    if args.mode == "full-pipeline":
        observations, session_summary = await run_api_pipeline(
            chunk_results, args.api_base, args.piece,
        )
        report.observations = observations
        report.session_summary = session_summary

    # Report
    print_report(report)

    # Save
    output_path = args.output
    if not output_path:
        reports_dir = Path(__file__).parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(reports_dir / f"load_test_{timestamp}.json")
    save_report(report, output_path)


if __name__ == "__main__":
    asyncio.run(main())
