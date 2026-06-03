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

# Default paths anchored to THIS module's location, never relative to CWD.
_MODULE_DIR = Path(__file__).resolve()
DEFAULT_AMT_URL = os.environ.get("AMT_URL", "http://127.0.0.1:8001/transcribe")
DEFAULT_AMT_VERSION_CONFIG = _MODULE_DIR.parents[2] / "config/amt_version.json"
DEFAULT_PRACTICE_ROOT = _MODULE_DIR.parents[2] / "data/evals/practice_eval"
DEFAULT_CACHE_ROOT = _MODULE_DIR.parents[2] / "data/evals/pseudo_truth"
DEFAULT_SCORE_ROOT = _MODULE_DIR.parents[2] / "data/scores"
DEFAULT_SCORE_BY_PIECE = {
    "bach_prelude_c_wtc1": DEFAULT_SCORE_ROOT / "bach.prelude.bwv_846.json",
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


def _transcribe_clip(audio_16k: np.ndarray, amt_url: str) -> list[dict]:
    n_chunks = max(1, int(np.ceil(len(audio_16k) / (AMT_CHUNK_S * TARGET_SR))))
    chunk_len = int(AMT_CHUNK_S * TARGET_SR)
    all_notes: list[dict] = []
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
    return all_notes


def _load_bach_json_score(score_path: Path) -> tuple[np.ndarray, list[dict], str, float]:
    """Load a single-tempo score JSON (bach prelude format).

    Returns (score_na, measure_table, score_sha256).
    score_na fields: ("onset_sec", float), ("onset_beat", float),
                     ("pitch", int), ("duration_sec", float),
                     ("duration_beat", float), ("id", "U32").
    Constant-tempo identity: onset_sec IS the score-audio-time axis.
    """
    score_sha256 = _sha256_file(score_path)
    body = json.loads(score_path.read_text())
    tempos = body.get("tempo_markings") or []
    if len(tempos) != 1:
        raise AmtRegenError(
            f"variable-tempo scores not supported in this rework; got "
            f"{len(tempos)} tempo markings in {score_path}. "
            f"See spec section 'Variable-tempo score support (future)'."
        )
    bars = body.get("bars") or []
    if len(bars) >= 2:
        # Infer ticks_per_beat from bar geometry. Assert 4/4; fail loud otherwise.
        ts_list = body.get("time_signatures") or []
        ts = ts_list[0] if ts_list else {}
        beats_per_bar = int(ts.get("numerator", 4))
        if beats_per_bar != 4:
            raise AmtRegenError(
                f"non-4/4 scores not supported in this rework; got "
                f"time_signature numerator={beats_per_bar} in {score_path}"
            )
        ticks_per_beat = (int(bars[1]["start_tick"]) - int(bars[0]["start_tick"])) // beats_per_bar
    else:
        ticks_per_beat = 480
    if ticks_per_beat <= 0:
        raise AmtRegenError(f"could not infer ticks_per_beat from {score_path}")

    # BPM from the single tempo marking (microseconds per beat -> BPM).
    bpm = float(tempos[0].get("bpm") or (60_000_000 / tempos[0]["tempo_usec"]))
    beat_sec = 60.0 / bpm

    rows = []
    nid = 0
    for bar in bars:
        for n in (bar.get("notes") or []):
            onset_sec = float(n["onset_seconds"])
            onset_beat = float(n["onset_tick"]) / ticks_per_beat
            dur_sec = float(n.get("duration_seconds", 0.001))
            dur_beat = dur_sec / beat_sec
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


def _amt_to_perf_na(notes: list[dict], beat_sec: float = 0.5) -> np.ndarray:
    """Convert raw AMT note dicts to the structured note array parangonar expects.

    parangonar requires both 'duration_beat' and 'onset_beat' for its piano-roll
    computation. We derive both from the score's constant tempo (beat_sec).
    Setting onset_beat = onset_sec / beat_sec gives parangonar accurate beat
    positions for its DTW initialization; using 0.0 causes zero matches.
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

    score_na, measure_table, score_sha256_2, beat_sec = _load_bach_json_score(score_path)
    if score_sha256 != score_sha256_2:
        raise AmtRegenError(
            f"score file mutated during regen: {score_path} "
            f"(first sha={score_sha256!r}, second sha={score_sha256_2!r})"
        )

    amt_perf_na = _amt_to_perf_na(amt_notes, beat_sec=beat_sec)
    matches = _match(score_na, amt_perf_na)
    perf_arr, score_arr = _build_pairs(score_na, amt_perf_na, matches)

    if score_arr.size < 100 or score_arr.size / max(len(amt_notes), 1) < 0.5:
        raise LowCoverageError(
            f"insufficient match coverage: matched={score_arr.size}, "
            f"amt_notes={len(amt_notes)}, match_rate="
            f"{score_arr.size / max(len(amt_notes), 1):.3f}"
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
