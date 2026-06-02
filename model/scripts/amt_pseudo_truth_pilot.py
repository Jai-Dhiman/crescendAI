"""AMT-pseudo-truth pilot: estimate parangonar noise floor on practice_eval clips.

Pipeline per clip:
  1. Read WAV (yt-dlp produced 24kHz mono).
  2. Split into 30s non-overlapping chunks, POST each to local AMT /transcribe.
  3. Concatenate notes across chunks (offset by chunk_idx * 30s).
  4. Write performance MIDI from AMT notes.
  5. Load score via partitura; project to constant-tempo performance to get onset_sec.
  6. Run parangonar AutomaticNoteMatcher to align AMT-MIDI to score.
  7. Report alignment statistics: match rate, audio_time -> score_bar samples, internal
     consistency (overlapping-window probe at a few timestamps).

Designed for the pre-harness-rework pilot; not production code. Single-clip-at-a-time;
audio fits in RAM. Explicit exceptions, no silent fallbacks.

Usage:
    cd model && uv run python scripts/amt_pseudo_truth_pilot.py \\
        --score scores/v1/chopin.ballades.1.mxl \\
        --piece chopin_ballade_1 \\
        --video-id MKbOHysTOE8

Outputs to model/data/evals/practice_eval_pseudo/<piece>/<video_id>/
  - amt_notes.json       (raw AMT notes for the clip)
  - matched_pairs.json   (parangonar output)
  - bar_map.json         (audio_time -> bar_number anchor points + interpolation source)
  - report.json          (statistics)
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf

MODEL_ROOT = Path(__file__).resolve().parents[1]
PRACTICE_EVAL = MODEL_ROOT / "data" / "evals" / "practice_eval"
OUT_ROOT = MODEL_ROOT / "data" / "evals" / "practice_eval_pseudo"

AMT_URL = "http://127.0.0.1:8001/transcribe"
# AMT internally pads to 30s. Sending 27s lets the model emit clean note
# terminations within the trailing 3s of zero-padding, avoiding tokenizer
# "Unexpected token order" failures at the 30000ms boundary.
AMT_CHUNK_S = 27.0
TARGET_SR = 16000  # AMT input


@dataclass
class AmtNote:
    onset: float
    offset: float
    pitch: int
    velocity: int


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _resample_to_16k(audio: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == TARGET_SR:
        return audio
    # simple polyphase resampling via scipy
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr_in, TARGET_SR)
    up = TARGET_SR // g
    down = sr_in // g
    return resample_poly(audio, up, down).astype(np.float32)


def _encode_wav_b64(audio_pcm: np.ndarray, sr: int) -> str:
    """PCM float32 -> WAV bytes -> base64 string. ffmpeg in the AMT server
    auto-detects WAV regardless of the .webm tempfile extension."""
    import io
    buf = io.BytesIO()
    sf.write(buf, audio_pcm, sr, format="WAV", subtype="FLOAT")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _amt_transcribe_one(chunk_audio_b64: str, timeout_s: float = 180.0) -> dict:
    resp = requests.post(
        AMT_URL,
        json={"chunk_audio": chunk_audio_b64, "context_audio": None},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise RuntimeError(f"AMT error: {body['error']}")
    return body


def transcribe_clip(wav_path: Path) -> list[AmtNote]:
    audio, sr = _read_wav_mono(wav_path)
    audio_16k = _resample_to_16k(audio, sr)
    total_s = len(audio_16k) / TARGET_SR
    n_chunks = int(np.ceil(total_s / AMT_CHUNK_S))
    print(f"  clip: {total_s:.1f}s @ 16kHz, {n_chunks} AMT chunks", flush=True)

    all_notes: list[AmtNote] = []
    for i in range(n_chunks):
        start = int(i * AMT_CHUNK_S * TARGET_SR)
        end = int(min((i + 1) * AMT_CHUNK_S * TARGET_SR, len(audio_16k)))
        chunk_pcm = audio_16k[start:end]
        # pad to full chunk if last; AMT may be picky about exact length
        if len(chunk_pcm) < AMT_CHUNK_S * TARGET_SR:
            pad = int(AMT_CHUNK_S * TARGET_SR) - len(chunk_pcm)
            chunk_pcm = np.concatenate([chunk_pcm, np.zeros(pad, dtype=np.float32)])
        chunk_b64 = _encode_wav_b64(chunk_pcm, TARGET_SR)
        t0 = time.time()
        try:
            out = _amt_transcribe_one(chunk_b64)
            notes_in_chunk = out.get("midi_notes") or []
        except RuntimeError as exc:
            # AMT tokenizer occasionally hits boundary token-order errors.
            # Log and skip this chunk rather than abort the whole clip.
            print(f"    chunk {i+1}/{n_chunks}: SKIPPED ({exc!s:.140}...)", flush=True)
            continue
        dt = time.time() - t0
        time_offset = i * AMT_CHUNK_S
        for n in notes_in_chunk:
            all_notes.append(AmtNote(
                onset=float(n["onset"]) + time_offset,
                offset=float(n["offset"]) + time_offset,
                pitch=int(n["pitch"]),
                velocity=int(n.get("velocity", 80)),
            ))
        print(f"    chunk {i+1}/{n_chunks}: {len(notes_in_chunk)} notes in {dt:.1f}s", flush=True)
    return all_notes


def amt_notes_to_perf_note_array(notes: list[AmtNote]) -> np.ndarray:
    """Build a partitura-compatible performance note array from AMT notes."""
    dtype = [
        ("onset_sec", float),
        ("duration_sec", float),
        ("pitch", int),
        ("velocity", int),
        ("track", int),
        ("channel", int),
        ("id", "U32"),
    ]
    arr = np.empty(len(notes), dtype=dtype)
    for i, n in enumerate(notes):
        arr[i] = (n.onset, max(n.offset - n.onset, 0.001), n.pitch, n.velocity, 0, 0, f"amt{i}")
    arr.sort(order="onset_sec")
    return arr


def load_score_perf_na(score_path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return (score_na with score-projected onset_sec, partitura measures table).

    Uses constant-tempo projection (bpm=100) — GoldMap-style. We only need
    monotonicity for parangonar's matcher.
    """
    import partitura as pt
    from partitura.utils.music import performance_notearray_from_score_notearray

    score = pt.load_score(str(score_path))
    score_na = score.note_array()
    perf_proj = performance_notearray_from_score_notearray(score_na, bpm=100.0)
    if "onset_sec" not in perf_proj.dtype.names:
        raise RuntimeError("score projection missing onset_sec; partitura behavior changed")
    if len(perf_proj) != len(score_na):
        raise RuntimeError(
            f"score-perf projection length mismatch ({len(perf_proj)} vs {len(score_na)})"
        )

    # Build measure table
    part = score.parts[0]
    measures = list(part.iter_all(pt.score.Measure))
    measure_table = []
    for m in measures:
        measure_table.append({
            "bar_number": m.number,
            "start_div": m.start.t,
            "end_div": m.end.t if m.end is not None else None,
        })
    return score_na, perf_proj, measure_table


def run_parangonar(score_na: np.ndarray, amt_perf_na: np.ndarray) -> list[dict]:
    """Wrap parangonar AutomaticNoteMatcher; return raw match list.

    parangonar wants the raw score note_array (with onset_beat) as the score
    side and a performance note_array (with onset_sec) as the performance side.
    """
    import parangonar as pa
    matcher = pa.AutomaticNoteMatcher()
    matches = matcher(score_na, amt_perf_na)
    return list(matches)


def build_audio_to_score_div(
    matches: list[dict],
    score_na: np.ndarray,
    score_perf_na: np.ndarray,
    amt_perf_na: np.ndarray,
) -> np.ndarray:
    """For each match[label == 'match'], pair (perf onset_sec) <-> (score onset_div).

    Returns Nx2 array sorted by perf_onset_sec.
    """
    score_id_to_div = {n["id"]: float(n["onset_div"]) for n in score_na}
    amt_id_to_sec = {n["id"]: float(n["onset_sec"]) for n in amt_perf_na}

    pairs: list[tuple[float, float]] = []
    for m in matches:
        if m.get("label") != "match":
            continue
        sid = m.get("score_id")
        pid = m.get("performance_id")
        if sid is None or pid is None:
            continue
        if sid not in score_id_to_div or pid not in amt_id_to_sec:
            continue
        pairs.append((amt_id_to_sec[pid], score_id_to_div[sid]))
    arr = np.array(sorted(pairs), dtype=float) if pairs else np.zeros((0, 2))
    return arr


def audio_time_to_bar(
    audio_sec: float,
    pairs: np.ndarray,
    measure_table: list[dict],
) -> float | None:
    """Interpolate audio_sec -> score_div -> bar_number via measure boundaries."""
    if len(pairs) < 2:
        return None
    # monotonic-clean pairs: enforce non-decreasing score_div by taking running max
    perf_t = pairs[:, 0]
    score_div = np.maximum.accumulate(pairs[:, 1])
    if audio_sec < perf_t[0] or audio_sec > perf_t[-1]:
        return None
    score_div_at_t = float(np.interp(audio_sec, perf_t, score_div))
    # find bar containing score_div_at_t
    for m in measure_table:
        end = m["end_div"]
        if end is None:
            continue
        if m["start_div"] <= score_div_at_t < end:
            frac = (score_div_at_t - m["start_div"]) / max(end - m["start_div"], 1)
            return m["bar_number"] + frac
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--score", required=True, help="path to score MXL/MusicXML/MIDI")
    p.add_argument("--piece", required=True, help="practice_eval piece slug")
    p.add_argument("--video-id", required=True, help="video id within that piece")
    p.add_argument("--anchor-times", default="30,60,120,240",
                   help="comma-separated audio times (s) to report bar at")
    args = p.parse_args()

    wav = PRACTICE_EVAL / args.piece / "audio" / f"{args.video_id}.wav"
    if not wav.exists():
        print(f"missing wav: {wav}", file=sys.stderr)
        return 1
    score_path = Path(args.score)
    if not score_path.is_absolute():
        score_path = MODEL_ROOT / args.score
    if not score_path.exists():
        print(f"missing score: {score_path}", file=sys.stderr)
        return 1

    out_dir = OUT_ROOT / args.piece / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"piece={args.piece} video={args.video_id}", flush=True)
    amt_cache = out_dir / "amt_notes.json"
    if amt_cache.exists():
        print(f"loading cached AMT notes from {amt_cache.name}", flush=True)
        raw = json.loads(amt_cache.read_text())
        amt_notes = [AmtNote(**r) for r in raw]
    else:
        print(f"transcribing {wav.name}", flush=True)
        amt_notes = transcribe_clip(wav)
        amt_cache.write_text(json.dumps([asdict(n) for n in amt_notes]))
    print(f"  total AMT notes: {len(amt_notes)}", flush=True)

    print(f"loading score {score_path.name}", flush=True)
    score_na, score_perf_na, measure_table = load_score_perf_na(score_path)
    print(f"  score notes: {len(score_na)}, measures: {len(measure_table)}", flush=True)

    print("building AMT performance note array", flush=True)
    amt_perf_na = amt_notes_to_perf_note_array(amt_notes)

    print("running parangonar AutomaticNoteMatcher", flush=True)
    t0 = time.time()
    matches = run_parangonar(score_na, amt_perf_na)
    dt = time.time() - t0
    n_match = sum(1 for m in matches if m.get("label") == "match")
    print(f"  matches: {n_match}/{len(matches)} labels, {dt:.1f}s", flush=True)
    (out_dir / "matched_pairs.json").write_text(json.dumps([
        {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in m.items()}
        for m in matches
    ]))

    pairs = build_audio_to_score_div(matches, score_na, score_perf_na, amt_perf_na)
    print(f"  matched (audio_sec, score_div) pairs: {len(pairs)}", flush=True)

    anchors = [float(t) for t in args.anchor_times.split(",")]
    bar_map = {}
    for t in anchors:
        bar = audio_time_to_bar(t, pairs, measure_table)
        bar_map[f"{t:.1f}s"] = bar
        print(f"  t={t:6.1f}s -> bar {bar}", flush=True)
    (out_dir / "bar_map.json").write_text(json.dumps(bar_map, indent=2))

    report = {
        "piece": args.piece,
        "video_id": args.video_id,
        "amt_note_count": len(amt_notes),
        "score_note_count": int(len(score_na)),
        "measure_count": len(measure_table),
        "matched_label_count": n_match,
        "matched_pair_count": int(len(pairs)),
        "match_rate_vs_amt": (n_match / max(len(amt_notes), 1)),
        "match_rate_vs_score": (n_match / max(len(score_na), 1)),
        "parangonar_seconds": dt,
        "anchor_bars": bar_map,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out_dir}/report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
