"""AMT regen orchestrator: practice audio -> AMT -> parangonar -> pseudo-truth cache.

Single-tempo scores ONLY in this rework (see spec "Variable-tempo score support (future)").
Loads the score JSON directly (no partitura). Score onset_seconds is the
score-audio-time axis under the constant-tempo identity.

Idempotent: re-running with identical 4-field cache key is a no-op.
Explicit exceptions on AMT failures, low coverage, and variable-tempo scores.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthMismatchError,
    PseudoTruthMissingError,
    PseudoTruthPayload,
    cache_path,
    load_pseudo_truth,
    write_pseudo_truth,
)

AMT_CHUNK_S = 27.0
TARGET_SR = 16000
RETRY_LIMIT = 2

# Aria-AMT emits sustained/held notes as many short same-pitch re-onsets (~50%
# of raw notes are a same-pitch repeat within 80ms of a prior one). Merge those
# into one note; the window is far below the legitimate repeated-16th spacing on
# these pieces (>=190ms at performance tempo), so real repeats survive.
DEDUP_WINDOW_S = 0.08

# Distributional acceptance gate for the perf->score time-map. A usable map needs
# enough monotonic anchors that SPAN the audio without a large blind gap -- NOT a
# high match *count*. (matched/amt_notes is poisoned by AMT over-transcription;
# matched/score_notes wrongly rejects clean-but-sparse maps -- e.g. the invention
# aligns 0/202 non-monotonic, 95% span, yet only 44% of score notes.)
MIN_ANCHORS = 100
MIN_SPAN_FRACTION = 0.85
MAX_ANCHOR_GAP_S = 8.0

# Default paths anchored to THIS module's location, never relative to CWD.
_MODULE_DIR = Path(__file__).resolve()
DEFAULT_AMT_URL = os.environ.get("AMT_URL", "http://127.0.0.1:8001/transcribe")
DEFAULT_AMT_VERSION_CONFIG = _MODULE_DIR.parents[2] / "config/amt_version.json"
DEFAULT_PRACTICE_ROOT = _MODULE_DIR.parents[2] / "data/evals/practice_eval"
DEFAULT_CACHE_ROOT = _MODULE_DIR.parents[2] / "data/evals/pseudo_truth"
DEFAULT_SCORE_ROOT = _MODULE_DIR.parents[2] / "data/scores"
DEFAULT_SCORE_BY_PIECE = {
    "bach_prelude_c_wtc1": DEFAULT_SCORE_ROOT / "bach.prelude.bwv_846.json",
    "bach_invention_1": DEFAULT_SCORE_ROOT / "bach.inventions.1.json",
}


class AmtRegenError(RuntimeError):
    pass


class LowCoverageError(AmtRegenError):
    pass


@dataclass
class RegenResult:
    wrote_cache: bool
    cache_path: Path
    audio_sha256: str
    score_sha256: str
    n_amt_notes: int
    n_matched: int


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _read_wav_16k_mono(audio_path: Path) -> np.ndarray:
    y, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, TARGET_SR)
        y = resample_poly(y, TARGET_SR // g, sr // g).astype(np.float32)
    return y


def _encode_chunk_b64(pcm: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, pcm, TARGET_SR, format="WAV", subtype="FLOAT")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _post_chunk(amt_url: str, pcm: np.ndarray) -> dict:
    """POST one chunk to AMT with retries. RequestException -> AmtRegenError
    after RETRY_LIMIT attempts. A documented 200-with-error-body (tokenizer
    boundary bug) is the only condition that signals skip-this-chunk.
    """
    last_exc: Exception | None = None
    for attempt in range(RETRY_LIMIT + 1):
        try:
            r = requests.post(
                amt_url,
                json={"chunk_audio": _encode_chunk_b64(pcm), "context_audio": None},
                timeout=180,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last_exc = exc
            continue
    raise AmtRegenError(
        f"AMT POST to {amt_url} failed after {RETRY_LIMIT + 1} attempts: {last_exc}"
    ) from last_exc


def _transcribe_clip_with_pedals(
    audio_16k: np.ndarray, amt_url: str
) -> tuple[list[dict], list[dict]]:
    """Transcribe a clip in AMT_CHUNK_S chunks, threading BOTH notes and pedal events.

    The AMT server (apps/inference/amt/server.py) returns ``midi_notes`` and
    ``pedal_events`` with times relative to each posted chunk; both receive the same
    ``offset = i * AMT_CHUNK_S`` so they share one clip-relative time axis.

    pedal_events schema (server-emitted, passed through unchanged): ``{"time": sec, "value": 0|127}``.
    Returns (notes, pedal_events). A chunk body lacking ``pedal_events`` (older server /
    error path) contributes no pedal events rather than raising.
    """
    n_chunks = max(1, int(np.ceil(len(audio_16k) / (AMT_CHUNK_S * TARGET_SR))))
    chunk_len = int(AMT_CHUNK_S * TARGET_SR)
    all_notes: list[dict] = []
    all_pedals: list[dict] = []
    for i in range(n_chunks):
        start = i * chunk_len
        end = min(start + chunk_len, len(audio_16k))
        pcm = audio_16k[start:end]
        if len(pcm) < chunk_len:
            pcm = np.concatenate([pcm, np.zeros(chunk_len - len(pcm), dtype=np.float32)])
        body = _post_chunk(amt_url, pcm)
        if "error" in body:
            # Documented tokenizer-boundary failure mode; skip this one chunk.
            continue
        offset = i * AMT_CHUNK_S
        for n in body.get("midi_notes") or []:
            all_notes.append({
                "onset": float(n["onset"]) + offset,
                "offset": float(n["offset"]) + offset,
                "pitch": int(n["pitch"]),
                "velocity": int(n.get("velocity", 80)),
            })
        for p in body.get("pedal_events") or []:
            all_pedals.append({
                "time": float(p["time"]) + offset,
                "value": int(p["value"]),
            })
    return all_notes, all_pedals


def _transcribe_clip(audio_16k: np.ndarray, amt_url: str) -> list[dict]:
    """Notes-only transcription (backward-compatible wrapper around the pedal-aware path)."""
    notes, _ = _transcribe_clip_with_pedals(audio_16k, amt_url)
    return notes


def _load_bach_json_score(score_path: Path) -> tuple[np.ndarray, list[dict], str, float]:
    """Load a score JSON into (score_na, measure_table, sha256, beat_sec).

    Variable-tempo and non-4/4 scores ARE supported (#98): the score-time axis comes
    from the precomputed per-bar ``start_seconds`` / per-note ``onset_seconds``, and
    beat positions come from MIDI ticks (``onset_tick / ticks_per_quarter``), which are
    metric and tempo-independent. ``beat_sec`` is the nominal first-tempo quarter
    duration -- only used to seed the perf-side beat scale; parangonar warps tempo.

    score_na fields: ("onset_sec", float), ("onset_beat", float),
                     ("pitch", int), ("duration_sec", float),
                     ("duration_beat", float), ("id", "U32").
    """
    score_sha256 = _sha256_file(score_path)
    body = json.loads(score_path.read_text())
    tempos = body.get("tempo_markings") or []
    if not tempos:
        raise AmtRegenError(f"no tempo markings in {score_path}")
    bars = body.get("bars") or []
    ts_list = body.get("time_signatures") or []
    ts = ts_list[0] if ts_list else {}
    numerator = int(ts.get("numerator", 4))
    denominator = int(ts.get("denominator", 4))
    if len(bars) >= 2:
        # ticks_per_quarter (PPQ) from bar geometry, meter-correct for any time sig:
        # ticks_per_bar = ppq * 4 * numerator/denominator
        #   => ppq = ticks_per_bar * denominator / (4 * numerator).
        # For 4/4 this reduces to ticks_per_bar / 4, identical to the prior loader.
        ticks_per_bar = int(bars[1]["start_tick"]) - int(bars[0]["start_tick"])
        ticks_per_quarter = int(round(ticks_per_bar * denominator / (4 * numerator)))
    else:
        ticks_per_quarter = 480
    if ticks_per_quarter <= 0:
        raise AmtRegenError(f"could not infer ticks_per_quarter from {score_path}")

    # Nominal quarter duration from the FIRST tempo marking (perf-side seed only).
    bpm = float(tempos[0].get("bpm") or (60_000_000 / tempos[0]["tempo_usec"]))
    beat_sec = 60.0 / bpm

    rows = []
    nid = 0
    for bar in bars:
        for n in (bar.get("notes") or []):
            onset_sec = float(n["onset_seconds"])
            onset_beat = float(n["onset_tick"]) / ticks_per_quarter
            dur_sec = float(n.get("duration_seconds", 0.001))
            # Tempo-independent beat duration from ticks; falls back to dur_sec/beat_sec
            # only if duration_ticks is absent. Equal to the old value at constant tempo.
            dur_ticks = n.get("duration_ticks")
            dur_beat = (
                float(dur_ticks) / ticks_per_quarter
                if dur_ticks is not None
                else dur_sec / beat_sec
            )
            rows.append((
                onset_sec,
                onset_beat,
                int(n["pitch"]),
                dur_sec,
                dur_beat,
                f"s{nid}",
            ))
            nid += 1
    if not rows:
        raise AmtRegenError(f"no notes found in score: {score_path}")
    dtype = [
        ("onset_sec", float), ("onset_beat", float), ("pitch", int),
        ("duration_sec", float), ("duration_beat", float), ("id", "U32"),
    ]
    score_na = np.array(rows, dtype=dtype)
    score_na.sort(order="onset_sec")

    measure_table = [
        {"bar_number": int(b["bar_number"]),
         "start_sec": float(b["start_seconds"]),
         "start_tick": int(b["start_tick"])}
        for b in bars
    ]
    return score_na, measure_table, score_sha256, beat_sec


def _dedup_amt_notes(notes: list[dict], window_s: float = DEDUP_WINDOW_S) -> list[dict]:
    """Merge same-pitch re-onsets within `window_s`: keep the earliest onset and
    extend its offset to cover the merged run. Removes the Aria-AMT note-repetition
    artifact (a held note emitted as many short onsets) without touching real
    repeated notes, which are spaced well above the window on these pieces.
    """
    by_pitch = sorted(notes, key=lambda n: (n["pitch"], n["onset"]))
    out: list[dict] = []
    last_for_pitch: dict[int, dict] = {}
    for n in by_pitch:
        pitch = int(n["pitch"])
        prev = last_for_pitch.get(pitch)
        if prev is not None and float(n["onset"]) - float(prev["onset"]) < window_s:
            prev["offset"] = max(float(prev["offset"]), float(n["offset"]))
            continue
        merged = dict(n)
        out.append(merged)
        last_for_pitch[pitch] = merged
    out.sort(key=lambda n: n["onset"])
    return out


def _amt_to_perf_na(notes: list[dict], beat_sec: float = 1.0) -> np.ndarray:
    """Convert raw AMT note dicts to the structured note array parangonar expects.

    parangonar requires both 'duration_beat' and 'onset_beat' for its piano-roll
    computation; onset_beat = 0.0 causes zero matches. We use beat_sec=1.0
    (onset_beat == onset_sec) rather than the score's tempo: the perf runs
    1.5-2.2x slower than the score's nominal tempo, so scaling by the SCORE beat
    mis-initializes the matcher. Sec-as-beat removes that wrong assumption and
    empirically lifts score alignment coverage (prelude 0.41 -> 0.59).
    """
    dtype = [
        ("onset_sec", float), ("onset_beat", float),
        ("duration_sec", float), ("duration_beat", float),
        ("pitch", int), ("velocity", int), ("id", "U32"),
    ]
    arr = np.empty(len(notes), dtype=dtype)
    for i, n in enumerate(notes):
        onset_sec = float(n["onset"])
        dur_sec = max(float(n["offset"]) - onset_sec, 0.001)
        arr[i] = (
            onset_sec,
            onset_sec / beat_sec,  # approximate beat position under constant tempo
            dur_sec,
            dur_sec / beat_sec,
            int(n["pitch"]),
            int(n.get("velocity", 80)),
            f"p{i}",
        )
    arr.sort(order="onset_sec")
    return arr


def _match(score_na: np.ndarray, perf_na: np.ndarray) -> list[dict]:
    import parangonar as pa
    matcher = pa.AutomaticNoteMatcher()
    return list(matcher(score_na, perf_na))


def _build_pairs(
    score_na: np.ndarray, amt_perf_na: np.ndarray, matches: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (perf_audio_sec, score_audio_sec) pairs from label=='match'
    entries; sort by perf time; enforce monotone running-max on score axis.
    """
    score_id_to_audio_sec = {str(s["id"]): float(s["onset_sec"]) for s in score_na}
    perf_id_to_audio_sec = {str(n["id"]): float(n["onset_sec"]) for n in amt_perf_na}
    pairs: list[tuple[float, float]] = []
    for entry in matches:
        if entry.get("label") != "match":
            continue
        s_id = str(entry.get("score_id"))
        p_id = str(entry.get("performance_id"))
        if s_id in score_id_to_audio_sec and p_id in perf_id_to_audio_sec:
            pairs.append((perf_id_to_audio_sec[p_id], score_id_to_audio_sec[s_id]))
    if not pairs:
        raise AmtRegenError("parangonar produced zero matches; cannot build pseudo-truth")
    pairs.sort()
    perf_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    score_arr = np.array([p[1] for p in pairs], dtype=np.float64)
    score_arr = np.maximum.accumulate(score_arr)
    return perf_arr, score_arr


def regenerate_pseudo_truth(
    piece_id: str,
    video_id: str,
    *,
    score_path: Path,
    audio_path: Path,
    amt_url: str,
    amt_checkpoint_hash: str,
    parangonar_version: str,
    cache_root: Path,
    force: bool = False,
) -> RegenResult:
    if not score_path.exists():
        raise AmtRegenError(f"score not found: {score_path}")
    if not audio_path.exists():
        raise AmtRegenError(f"audio not found: {audio_path}")
    audio_sha256 = _sha256_file(audio_path)
    score_sha256 = _sha256_file(score_path)

    # Idempotence check. Catch the SPECIFIC exception classes; let any other
    # error propagate (CLAUDE.md "no catch-all" rule).
    if not force:
        try:
            load_pseudo_truth(
                piece_id, video_id,
                audio_sha256=audio_sha256,
                amt_checkpoint_hash=amt_checkpoint_hash,
                score_sha256=score_sha256,
                parangonar_version=parangonar_version,
                cache_root=cache_root,
            )
            return RegenResult(
                wrote_cache=False,
                cache_path=cache_path(cache_root, piece_id, video_id),
                audio_sha256=audio_sha256,
                score_sha256=score_sha256,
                n_amt_notes=0, n_matched=0,
            )
        except PseudoTruthMissingError:
            pass  # regen below
        except PseudoTruthMismatchError:
            pass  # regen below

    audio_16k = _read_wav_16k_mono(audio_path)
    amt_notes = _transcribe_clip(audio_16k, amt_url)
    if not amt_notes:
        raise AmtRegenError(f"AMT returned zero notes for {audio_path}")

    score_na, measure_table, score_sha256_2, _beat_sec = _load_bach_json_score(score_path)
    if score_sha256 != score_sha256_2:
        raise AmtRegenError(
            f"score file mutated during regen: {score_path} "
            f"(first sha={score_sha256!r}, second sha={score_sha256_2!r})"
        )

    deduped_notes = _dedup_amt_notes(amt_notes)
    amt_perf_na = _amt_to_perf_na(deduped_notes)
    matches = _match(score_na, amt_perf_na)
    perf_arr, score_arr = _build_pairs(score_na, amt_perf_na, matches)

    # Distributional acceptance: enough monotonic anchors (score_arr is already
    # running-max'd in _build_pairs), spanning the audio without a large blind gap.
    audio_dur = len(audio_16k) / TARGET_SR
    n_anchors = int(perf_arr.size)
    if n_anchors >= 2 and audio_dur > 0:
        span_fraction = float(perf_arr.max() - perf_arr.min()) / audio_dur
        max_gap = float(np.max(np.diff(np.sort(perf_arr))))
    else:
        span_fraction = 0.0
        max_gap = float("inf")
    if (
        n_anchors < MIN_ANCHORS
        or span_fraction < MIN_SPAN_FRACTION
        or max_gap > MAX_ANCHOR_GAP_S
    ):
        raise LowCoverageError(
            f"unusable pseudo-truth time-map: anchors={n_anchors} "
            f"(min {MIN_ANCHORS}), span_fraction={span_fraction:.2f} "
            f"(min {MIN_SPAN_FRACTION}), max_gap={max_gap:.2f}s "
            f"(max {MAX_ANCHOR_GAP_S}s); raw_amt={len(amt_notes)}, "
            f"deduped_amt={len(deduped_notes)}, score_notes={len(score_na)}"
        )

    config_body = (
        json.loads(DEFAULT_AMT_VERSION_CONFIG.read_text())
        if DEFAULT_AMT_VERSION_CONFIG.exists() else {}
    )
    regen_source = config_body.get("regen_source_default", "local:aria-amt")
    payload = PseudoTruthPayload(
        perf_audio_sec=perf_arr,
        score_audio_sec=score_arr,
        measure_table=measure_table,
        audio_sha256=audio_sha256,
        amt_checkpoint_hash=amt_checkpoint_hash,
        score_sha256=score_sha256,
        parangonar_version=parangonar_version,
        regen_source=regen_source,
    )
    out = write_pseudo_truth(piece_id, video_id, payload, cache_root)
    return RegenResult(
        wrote_cache=True,
        cache_path=out,
        audio_sha256=audio_sha256,
        score_sha256=score_sha256,
        n_amt_notes=len(amt_notes),
        n_matched=int(score_arr.size),
    )


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chroma_dtw_eval.amt_regen")
    p.add_argument("--piece", required=True)
    p.add_argument("--video-id", required=True)
    p.add_argument("--score", type=Path, default=None)
    p.add_argument("--audio", type=Path, default=None)
    p.add_argument("--amt-url", default=DEFAULT_AMT_URL)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--config", type=Path, default=DEFAULT_AMT_VERSION_CONFIG)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    score = args.score or DEFAULT_SCORE_BY_PIECE.get(args.piece)
    if score is None:
        raise AmtRegenError(
            f"no default score for piece {args.piece!r}; pass --score explicitly"
        )
    audio = args.audio or (DEFAULT_PRACTICE_ROOT / args.piece / "audio" / f"{args.video_id}.wav")
    config_body = json.loads(args.config.read_text())
    res = regenerate_pseudo_truth(
        piece_id=args.piece, video_id=args.video_id,
        score_path=score, audio_path=audio,
        amt_url=args.amt_url,
        amt_checkpoint_hash=config_body["checkpoint_hash"],
        parangonar_version=config_body["parangonar_version"],
        cache_root=args.cache_root, force=args.force,
    )
    print(json.dumps({
        "wrote_cache": res.wrote_cache,
        "cache_path": str(res.cache_path),
        "audio_sha256": res.audio_sha256,
        "score_sha256": res.score_sha256,
        "n_amt_notes": res.n_amt_notes,
        "n_matched": res.n_matched,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
