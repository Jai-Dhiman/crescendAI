"""BundleExtractor: produce a unified per-clip measurement bundle from AMT + parangonar.

Reuses chroma_dtw_eval.amt_regen internals for AMT transcription and parangonar alignment.
Adds CC64 sustain-pedal capture (pedal_events: [] if AMT server does not expose CC data).
Output: bundle JSON at bundle_root/piece_id/video_id.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chroma_dtw_eval.amt_regen import (
    _dedup_amt_notes,
    _transcribe_clip,
    _read_wav_16k_mono,
    _amt_to_perf_na,
    _load_bach_json_score,
    _match,
    _build_pairs,
    DEFAULT_AMT_URL,
    DEFAULT_AMT_VERSION_CONFIG,
)

BUNDLE_SCHEMA_VERSION = "v1"


class BundleExtractionError(RuntimeError):
    pass


def _bundle_path(bundle_root: Path, piece_id: str, video_id: str) -> Path:
    return bundle_root / piece_id / f"{video_id}.json"


def extract_bundle(
    piece_id: str,
    video_id: str,
    *,
    audio_path: Path,
    score_path: Path,
    cache_root: Path,
    bundle_root: Path,
    amt_url: str = DEFAULT_AMT_URL,
    force: bool = False,
) -> Path:
    """Produce bundle JSON at bundle_root/piece_id/video_id.json. Returns path.

    Idempotent: returns existing path if bundle exists and force=False.
    Raises BundleExtractionError on AMT or alignment failure.
    """
    out_path = _bundle_path(bundle_root, piece_id, video_id)
    if not force and out_path.exists():
        return out_path

    if not audio_path.exists():
        raise BundleExtractionError(f"audio not found: {audio_path}")
    if not score_path.exists():
        raise BundleExtractionError(f"score not found: {score_path}")

    config_body = (
        json.loads(DEFAULT_AMT_VERSION_CONFIG.read_text())
        if DEFAULT_AMT_VERSION_CONFIG.exists() else {}
    )
    amt_checkpoint_hash = config_body.get("checkpoint_hash", "unknown")
    parangonar_version = config_body.get("parangonar_version", "unknown")

    audio_16k = _read_wav_16k_mono(audio_path)
    amt_notes = _transcribe_clip(audio_16k, amt_url)
    if not amt_notes:
        raise BundleExtractionError(f"AMT returned zero notes for {audio_path}")

    score_na, measure_table, score_sha256, beat_sec = _load_bach_json_score(score_path)
    deduped_notes = _dedup_amt_notes(amt_notes)

    # CC64 pedal events: AMT server currently returns notes only.
    # pedal_events is [] until the AMT endpoint exposes MIDI CC.
    pedal_events: list[dict] = []

    amt_perf_na = _amt_to_perf_na(deduped_notes, beat_sec)
    matches = _match(score_na, amt_perf_na)
    perf_arr, score_arr = _build_pairs(score_na, amt_perf_na, matches)

    bundle = {
        "piece_id": piece_id,
        "video_id": video_id,
        "audio_path": str(audio_path.resolve()),
        "notes": deduped_notes,
        "pedal_events": pedal_events,
        "measure_table": measure_table,
        "anchors": {
            "perf_audio_sec": perf_arr.tolist(),
            "score_audio_sec": score_arr.tolist(),
        },
        "substrate_versions": {
            "amt_checkpoint_hash": amt_checkpoint_hash,
            "parangonar_version": parangonar_version,
            "bundle_schema": BUNDLE_SCHEMA_VERSION,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(bundle))
    tmp.replace(out_path)
    return out_path
