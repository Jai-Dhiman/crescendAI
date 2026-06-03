"""On-disk cache for AMT-derived pseudo-truth alignment of practice audio
to score (single-tempo identity: score_audio_sec is score seconds directly).

Keyed by the 4-tuple (audio_sha256, amt_checkpoint_hash, score_sha256,
parangonar_version). Read-only at eval time; written only by amt_regen.
Explicit exceptions on missing files and any-field mismatch -- no silent
fallbacks. JSON on disk (not pickle) for forward-compatibility.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class PseudoTruthMissingError(FileNotFoundError):
    pass


class PseudoTruthMismatchError(ValueError):
    pass


@dataclass
class PseudoTruthPayload:
    perf_audio_sec: np.ndarray
    score_audio_sec: np.ndarray
    measure_table: list[dict]
    audio_sha256: str
    amt_checkpoint_hash: str
    score_sha256: str
    parangonar_version: str
    regen_source: str


@dataclass
class PseudoTruth:
    perf_audio_sec: np.ndarray
    score_audio_sec: np.ndarray
    measure_table: list[dict]
    audio_sha256: str
    amt_checkpoint_hash: str
    score_sha256: str
    parangonar_version: str

    def audio_sec_to_score_sec(self, t: float) -> float:
        if self.perf_audio_sec.size < 2:
            raise PseudoTruthMismatchError("perf_audio_sec must have >= 2 anchors")
        return float(np.interp(t, self.perf_audio_sec, self.score_audio_sec))

    def score_sec_to_audio_sec(self, s: float) -> float:
        if self.score_audio_sec.size < 2:
            raise PseudoTruthMismatchError("score_audio_sec must have >= 2 anchors")
        return float(np.interp(s, self.score_audio_sec, self.perf_audio_sec))


def cache_path(cache_root: Path, piece_id: str, video_id: str) -> Path:
    """PUBLIC. Used by amt_regen and chunk_sampler to locate cache files."""
    return cache_root / piece_id / f"{video_id}.json"


def write_pseudo_truth(
    piece_id: str,
    video_id: str,
    payload: PseudoTruthPayload,
    cache_root: Path,
) -> Path:
    if payload.perf_audio_sec.shape != payload.score_audio_sec.shape:
        raise PseudoTruthMismatchError(
            f"shape mismatch: perf {payload.perf_audio_sec.shape} vs score {payload.score_audio_sec.shape}"
        )
    out = cache_path(cache_root, piece_id, video_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "audio_sha256": payload.audio_sha256,
        "amt_checkpoint_hash": payload.amt_checkpoint_hash,
        "score_sha256": payload.score_sha256,
        "parangonar_version": payload.parangonar_version,
        "regen_source": payload.regen_source,
        "perf_audio_sec": payload.perf_audio_sec.tolist(),
        "score_audio_sec": payload.score_audio_sec.tolist(),
        "measure_table": payload.measure_table,
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body))
    tmp.replace(out)
    return out


def load_pseudo_truth(
    piece_id: str,
    video_id: str,
    *,
    audio_sha256: str,
    amt_checkpoint_hash: str,
    score_sha256: str,
    parangonar_version: str,
    cache_root: Path,
) -> PseudoTruth:
    path = cache_path(cache_root, piece_id, video_id)
    if not path.exists():
        raise PseudoTruthMissingError(
            f"pseudo-truth cache missing for {piece_id}/{video_id}: {path}"
        )
    body = json.loads(path.read_text())
    for field, expected in (
        ("audio_sha256", audio_sha256),
        ("amt_checkpoint_hash", amt_checkpoint_hash),
        ("score_sha256", score_sha256),
        ("parangonar_version", parangonar_version),
    ):
        actual = body.get(field)
        if actual != expected:
            raise PseudoTruthMismatchError(
                f"{field} mismatch for {piece_id}/{video_id}: "
                f"requested {expected!r}, cached {actual!r}"
            )
    return PseudoTruth(
        perf_audio_sec=np.asarray(body["perf_audio_sec"], dtype=np.float64),
        score_audio_sec=np.asarray(body["score_audio_sec"], dtype=np.float64),
        measure_table=body["measure_table"],
        audio_sha256=body["audio_sha256"],
        amt_checkpoint_hash=body["amt_checkpoint_hash"],
        score_sha256=body["score_sha256"],
        parangonar_version=body["parangonar_version"],
    )
