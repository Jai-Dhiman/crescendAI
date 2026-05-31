"""parangonar-based audio-to-score ground truth map for the gold-truth slice.

MAESTRO audio to MAESTRO MIDI is zero-error (Disklavier simultaneous capture);
(n)ASAP gives MIDI-to-score at ~6ms via parangonar's AutomaticNoteMatcher; the
composition is audio_seconds -> MIDI_seconds (identity) -> score_seconds -> score_frame.

Score-side seconds come from projecting the score note array through partitura's
performance_notearray_from_score_notearray at a fixed reference tempo. This
turns score beats into a monotone seconds timeline that is consistent across the
piece -- parangonar's match pairs (by score id) then give us perf_seconds and
score_seconds pairs that the GoldMap interpolates over.

Cache keyed by (midi sha256, score sha256). Stored as JSON (NOT pickle -- JSON
round-trips across Python minor versions and removes the pickle code-exec risk).
No silent fallbacks -- raises GoldMapMissingDataError when inputs are missing.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parangonar as pa
import partitura
from partitura.utils.music import performance_notearray_from_score_notearray


class GoldMapMissingDataError(FileNotFoundError):
    pass


@dataclass
class GoldMap:
    """Performance-time-seconds to score-frame-index lookup.

    The map is stored as two parallel arrays: perf_seconds (monotone) and
    score_seconds (in the score timeline). audio_seconds_to_score_frame
    interpolates and converts to a frame index at the requested rate.
    """

    perf_seconds: np.ndarray
    score_seconds: np.ndarray

    def audio_seconds_to_score_frame(self, t: float, frame_rate_hz: float) -> int:
        if t < self.perf_seconds[0]:
            t = float(self.perf_seconds[0])
        if t > self.perf_seconds[-1]:
            t = float(self.perf_seconds[-1])
        score_t = float(np.interp(t, self.perf_seconds, self.score_seconds))
        return int(round(score_t * frame_rate_hz))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_gold_map(midi_path: Path, score_path: Path, cache_root: Path) -> GoldMap:
    if not midi_path.exists():
        raise GoldMapMissingDataError(f"midi not found: {midi_path}")
    if not score_path.exists():
        raise GoldMapMissingDataError(f"score not found: {score_path}")
    cache_root.mkdir(parents=True, exist_ok=True)
    key = f"{_sha(midi_path)}_{_sha(score_path)}.json"
    cache_file = cache_root / key
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return GoldMap(
            np.asarray(data["perf_seconds"], dtype=np.float64),
            np.asarray(data["score_seconds"], dtype=np.float64),
        )

    perf = partitura.load_performance_midi(midi_path)
    score = partitura.load_score(score_path)
    perf_na = perf.note_array()
    score_na = score.note_array()

    # Project score beats to seconds via a synthetic constant-tempo performance.
    # 100 bpm is the partitura default; the absolute scale cancels out when
    # interpolating perf_seconds -> score_seconds -> score_frame.
    score_perf_na = performance_notearray_from_score_notearray(score_na, bpm=100.0)
    if "onset_sec" not in score_perf_na.dtype.names:
        raise GoldMapMissingDataError(
            "performance_notearray_from_score_notearray did not produce onset_sec -- "
            f"got fields {score_perf_na.dtype.names}"
        )

    # Defensive row-alignment assertion: partitura must preserve row order.
    # Both arrays should have the same length and matching id fields.
    if len(score_na) != len(score_perf_na):
        raise GoldMapMissingDataError(
            f"score_na row count ({len(score_na)}) != score_perf_na row count "
            f"({len(score_perf_na)}) -- partitura row order assumption violated"
        )
    if "id" in score_na.dtype.names and "id" in score_perf_na.dtype.names:
        mismatched = np.sum(score_na["id"] != score_perf_na["id"])
        if mismatched > 0:
            raise GoldMapMissingDataError(
                f"{mismatched} id mismatches between score_na and score_perf_na -- "
                "partitura row alignment assumption violated; update this code"
            )

    # parangonar 3.x: AutomaticNoteMatcher is exposed at top-level.
    # Calling signature: matcher(score_note_array, performance_note_array) -> List[Dict].
    matcher = pa.AutomaticNoteMatcher()
    alignment = matcher(score_na, perf_na)

    # Build lookups keyed by note id.
    # Score-side seconds come from the projected array (parallel to score_na by index).
    score_id_to_sec: dict[str, float] = {
        str(s["id"]): float(p["onset_sec"])
        for s, p in zip(score_na, score_perf_na)
    }
    perf_id_to_sec: dict[str, float] = {
        str(n["id"]): float(n["onset_sec"]) for n in perf_na
    }

    pairs: list[tuple[float, float]] = []
    for entry in alignment:
        if entry.get("label") != "match":
            continue
        s_id = str(entry.get("score_id"))
        p_id = str(entry.get("performance_id"))
        if s_id in score_id_to_sec and p_id in perf_id_to_sec:
            pairs.append((perf_id_to_sec[p_id], score_id_to_sec[s_id]))

    if not pairs:
        raise GoldMapMissingDataError(
            f"parangonar produced no match pairs for {midi_path} / {score_path}"
        )

    pairs.sort()
    perf_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    score_arr = np.array([p[1] for p in pairs], dtype=np.float64)
    cache_file.write_text(
        json.dumps(
            {
                "perf_seconds": perf_arr.tolist(),
                "score_seconds": score_arr.tolist(),
            }
        )
    )
    return GoldMap(perf_arr, score_arr)
