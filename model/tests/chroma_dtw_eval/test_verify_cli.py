"""End-to-end verify CLI behavior.

(a) Smoke test against staged manifest + pseudo-truth (--skip-dtw):
    asserts CLI exits 0, prints one float, writes sidecar with
    error_seconds_distribution + tolerance_sensitivity, AND emits stderr
    WARNING when fewer than 2 pieces are in the manifest.

(b) Numerical correctness test: fabricate 3 practice chunks where the
    pseudo-truth is identity-linear over 60s; assert each chunk's
    error_seconds in the sidecar matches the analytic expectation within
    0.1s.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from chroma_dtw_eval.chunk_sampler import ChunkManifestEntry
from chroma_dtw_eval.pseudo_truth_cache import (
    PseudoTruthPayload, write_pseudo_truth,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "model"


def _stage(tmp_path: Path, *, audio_sha256_for_chunks: list[str]) -> tuple[Path, Path, Path]:
    evals = tmp_path / "evals"
    piece_dir = evals / "practice_eval" / "bach_prelude_c_wtc1"
    (piece_dir / "audio").mkdir(parents=True)
    rng = np.random.default_rng(0)
    sf.write(
        piece_dir / "audio" / "VID0.wav",
        rng.standard_normal(16000 * 60).astype(np.float32) * 0.05,
        16000, subtype="FLOAT",
    )
    # Identity-linear pseudo-truth: 60s perf == 60s score.
    cache_root = evals / "pseudo_truth"
    write_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="VID0",
        payload=PseudoTruthPayload(
            perf_audio_sec=np.linspace(0.0, 60.0, 61, dtype=np.float64),
            score_audio_sec=np.linspace(0.0, 60.0, 61, dtype=np.float64),
            measure_table=[],
            audio_sha256=audio_sha256_for_chunks[0],
            amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
            score_sha256="deadbeefdeadbeef",
            parangonar_version="3.3.2",
            regen_source="test",
        ),
        cache_root=cache_root,
    )
    # Manifest with three chunks; each chunk's audio_sha256 is the value
    # the pseudo-truth was keyed by (so cache lookup succeeds).
    manifest = [
        {
            "piece": "bach_prelude_c_wtc1",
            "video_id": "VID0",
            "start_audio_sec": float(start),
            "end_audio_sec": float(start + 15.0),
            "audio_sha256": audio_sha256_for_chunks[i],
            "position_bucket": bucket,
        }
        for i, (start, bucket) in enumerate([(0.0, "intro"), (20.0, "middle"), (40.0, "late")])
    ]
    manifest_path = evals / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "primary": 0.0,
        "guards": {"g1": 100.0, "g2": 0.0, "g3": 100.0, "g4": 0.0, "g5": 100.0},
    }))
    # Write a fake score JSON sitting where verify expects it; its sha256
    # must equal the value written into the pseudo-truth above
    # ("deadbeefdeadbeef"). We compute the sha256 from the actual file
    # contents and then patch the cache to match.
    score_dir = evals / "scores"
    score_dir.mkdir()
    score_path = score_dir / "bach.prelude.bwv_846.json"
    score_path.write_text(json.dumps({"tempo_markings": [{"bpm": 120}], "bars": []}))
    import hashlib
    real_sha = hashlib.sha256(score_path.read_bytes()).hexdigest()[:16]
    # Re-write cache with the real score_sha256.
    write_pseudo_truth(
        piece_id="bach_prelude_c_wtc1", video_id="VID0",
        payload=PseudoTruthPayload(
            perf_audio_sec=np.linspace(0.0, 60.0, 61, dtype=np.float64),
            score_audio_sec=np.linspace(0.0, 60.0, 61, dtype=np.float64),
            measure_table=[],
            audio_sha256=audio_sha256_for_chunks[0],
            amt_checkpoint_hash="aria_amt_v1_pilot_2026_06_01",
            score_sha256=real_sha,
            parangonar_version="3.3.2",
            regen_source="test",
        ),
        cache_root=cache_root,
    )
    return evals, manifest_path, baseline


def test_skip_dtw_smoke_emits_one_float_sidecar_and_warning(tmp_path: Path) -> None:
    # When all three chunks share the same audio_sha256, the cache lookup
    # for each will succeed against the same cached pseudo-truth (real
    # corpus would have distinct shas; smoke uses one).
    sha = "abcd123456789012"
    evals, manifest_path, baseline = _stage(tmp_path, audio_sha256_for_chunks=[sha, sha, sha])
    sidecar = tmp_path / "sidecar.json"
    res = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify",
         "--baseline", str(baseline),
         "--manifest", str(manifest_path),
         "--sidecar", str(sidecar),
         "--corpus-root", str(evals),
         "--skip-dtw"],
        capture_output=True, text=True, timeout=120,
        cwd=MODEL_DIR,
    )
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    lines = [ln for ln in res.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == 1
    float(lines[0])
    body = json.loads(sidecar.read_text())
    assert set(body["guards"].keys()) == {"g1", "g2", "g3", "g4", "g5"}
    assert "error_seconds_distribution" in body
    assert "tolerance_sensitivity" in body
    for k in ("0.5", "1.0", "1.5", "2.0", "3.0"):
        assert k in body["tolerance_sensitivity"]
    # Manifest has only one piece -> stderr WARNING.
    assert "WARNING" in res.stderr and "piece" in res.stderr


def test_skip_dtw_numerical_error_within_0p1s(tmp_path: Path) -> None:
    """With identity-linear pseudo-truth and --skip-dtw, the synthetic
    predicted_score_sec equals chunk_start_audio_sec, so error_seconds
    should be 0 within 0.1s for every chunk.
    """
    sha = "abcd123456789012"
    evals, manifest_path, baseline = _stage(tmp_path, audio_sha256_for_chunks=[sha, sha, sha])
    sidecar = tmp_path / "sidecar.json"
    res = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify",
         "--baseline", str(baseline),
         "--manifest", str(manifest_path),
         "--sidecar", str(sidecar),
         "--corpus-root", str(evals),
         "--skip-dtw"],
        capture_output=True, text=True, timeout=120,
        cwd=MODEL_DIR,
    )
    assert res.returncode == 0, res.stderr
    body = json.loads(sidecar.read_text())
    dist = body["error_seconds_distribution"]
    assert dist["max"] <= 0.1
    assert dist["mean"] <= 0.1


def test_skip_dtw_not_in_help_output() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify", "--help"],
        capture_output=True, text=True, timeout=10, cwd=MODEL_DIR,
    )
    assert res.returncode == 0
    assert "--skip-dtw" not in res.stdout
