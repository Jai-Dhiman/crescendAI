"""sample_chunks emits ChunkManifestEntry with real per-chunk audio_sha256
computed from the chunk's audio bytes, not the whole file."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from chroma_dtw_eval.chunk_sampler import (
    ChunkManifestEntry,
    sample_chunks,
)
from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthPayload, write_pseudo_truth,
)


def _stage(corpus_root: Path) -> None:
    piece_dir = corpus_root / "practice_eval" / "bach_prelude_c_wtc1"
    (piece_dir / "audio").mkdir(parents=True, exist_ok=True)
    # Deterministic non-silent audio so per-chunk sha256 differs.
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(16000 * 90).astype(np.float32) * 0.05
    sf.write(piece_dir / "audio" / "VID0.wav", audio, 16000, subtype="FLOAT")
    (piece_dir / "candidates.yaml").write_text(yaml.safe_dump({
        "piece": "bach_prelude_c_wtc1",
        "recordings": [
            {"video_id": "VID0", "approved": True, "downloaded": True},
            {"video_id": "VID1", "approved": False, "downloaded": True},
        ],
    }))
    cache_root = corpus_root / "pseudo_truth"
    write_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="VID0",
        payload=PseudoTruthPayload(
            perf_audio_sec=np.array([0.0, 90.0], dtype=np.float64),
            score_audio_sec=np.array([0.0, 90.0], dtype=np.float64),
            measure_table=[],
            audio_sha256="a" * 16, amt_checkpoint_hash="b" * 16,
            score_sha256="c" * 16, parangonar_version="3.3.2",
            regen_source="local:test",
        ),
        cache_root=cache_root,
    )


def test_sample_chunks_emits_manifest_with_real_chunk_sha256(tmp_path: Path) -> None:
    _stage(tmp_path)
    entries = sample_chunks(
        corpus_root=tmp_path,
        pseudo_truth_root=tmp_path / "pseudo_truth",
        seed=0,
    )
    assert entries, "expected at least one entry"
    assert all(isinstance(e, ChunkManifestEntry) for e in entries)
    assert all(e.piece == "bach_prelude_c_wtc1" for e in entries)
    assert all(e.video_id == "VID0" for e in entries), "unapproved VID1 must be excluded"
    # audio_sha256 is per-chunk (depends on start/end), so distinct chunks
    # at different offsets have distinct hashes.
    hashes = {e.audio_sha256 for e in entries}
    assert len(hashes) >= 2, "per-chunk audio_sha256 should differ across chunks"
    # Verify one chunk's hash matches independently-computed sha256 of the
    # actual audio slice between its start and end.
    e0 = entries[0]
    y, sr = sf.read(tmp_path / "practice_eval" / "bach_prelude_c_wtc1" / "audio" / "VID0.wav",
                    dtype="float32")
    start_frame = int(round(e0.start_audio_sec * sr))
    end_frame = int(round(e0.end_audio_sec * sr))
    slice_bytes = y[start_frame:end_frame].tobytes()
    expected = hashlib.sha256(slice_bytes).hexdigest()[:16]
    assert e0.audio_sha256 == expected


def test_sample_chunks_writes_committed_manifest(tmp_path: Path) -> None:
    _stage(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    sample_chunks(
        corpus_root=tmp_path,
        pseudo_truth_root=tmp_path / "pseudo_truth",
        seed=0,
        manifest_out=manifest_path,
    )
    assert manifest_path.exists()
    import json
    body = json.loads(manifest_path.read_text())
    assert isinstance(body, list)
    assert body
    first = body[0]
    for key in ("piece", "video_id", "start_audio_sec", "end_audio_sec", "audio_sha256"):
        assert key in first
