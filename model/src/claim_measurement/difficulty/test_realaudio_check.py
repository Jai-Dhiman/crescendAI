"""Tests for realaudio_check (#149 / #138 Phase 1) -- the real-audio second
gate: MIDI drift, resumable transcription, and per-fold audio scoring.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

import numpy as np
import pytest

from claim_measurement.difficulty.bakeoff_npz import write_embedding_npz
from claim_measurement.difficulty.realaudio_check import build_wav_manifest, midi_drift


def test_midi_drift_computes_note_count_delta_and_onset_f1_with_tolerance_matching():
    reference = [{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
                 {"pitch": 64, "onset": 0.5, "offset": 1.0, "velocity": 80}]

    identical = midi_drift(reference, reference, onset_tolerance=0.05)
    assert identical == {"note_count_delta": 0, "onset_f1": 1.0}

    candidate = [
        # onset shifted past tolerance
        {"pitch": 60, "onset": 0.20, "offset": 0.5, "velocity": 80},
        {"pitch": 64, "onset": 0.5, "offset": 1.0, "velocity": 80},
        {"pitch": 67, "onset": 2.0, "offset": 2.5, "velocity": 80},  # extra note
    ]
    degraded = midi_drift(reference, candidate, onset_tolerance=0.05)
    assert degraded["note_count_delta"] == 1
    # tp=1, precision=1/3, recall=1/2
    assert degraded["onset_f1"] == pytest.approx(2 / 5)


from claim_measurement.difficulty.realaudio_check import main


def test_transcribe_stage_skips_pieces_whose_cache_file_already_exists(tmp_path):
    wav_manifest = tmp_path / "wav_manifest.json"
    wav_manifest.write_text(json.dumps([
        {"seg_id": "already_done", "wav_path": str(tmp_path / "a.wav")},
        {"seg_id": "new_piece", "wav_path": str(tmp_path / "b.wav")},
    ]))
    out_dir = tmp_path / "cache"
    out_dir.mkdir()
    (out_dir / "already_done.json").write_text(json.dumps({"notes": [], "pedals": []}))

    calls = []

    def fake_transcriber(wav_path):
        calls.append(wav_path)
        return ([{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80}], [])

    exit_code = main(["--wav-manifest", str(wav_manifest), "--out-dir", str(out_dir)],
                      transcriber=fake_transcriber)

    assert exit_code == 0
    # only the not-yet-cached piece was transcribed
    assert calls == [tmp_path / "b.wav"]
    cached = json.loads((out_dir / "new_piece.json").read_text())
    assert cached["notes"][0]["pitch"] == 60


from claim_measurement.difficulty.realaudio_check import score_audio_subset
from claim_measurement.difficulty.bakeoff_cv import composer_disjoint_folds


def _fold_signal_embeddings(y, composers, seg_ids, audio_subset_ids, n_folds,
                             seed, rng):
    """Build a per-fold embedding fixture where fold f's matrix carries the
    y signal ONLY in column f (every other column is pure noise). This is
    deliberately NON-identical across folds: a ridge model fit on the WRONG
    fold's matrix cannot recover y through it, because that fold's
    informative column is elsewhere. audio_embeddings mirrors this -- each
    subset piece's vector carries its signal in ITS OWN true-fold column
    (looked up via the same composer_disjoint_folds used inside
    score_audio_subset) and zeros elsewhere, matching emb_by_fold's shape.
    A bug that scores every piece through one fixed fold's model then reads
    a pure-noise column and the correlation collapses."""
    test_folds = composer_disjoint_folds(composers, n_folds, seed)
    fold_of_idx = {i: f for f, idx in enumerate(test_folds) for i in idx}

    emb_by_fold = {
        f: np.column_stack([
            (y + rng.normal(scale=0.01, size=len(y))) if c == f
            else rng.normal(size=len(y))
            for c in range(n_folds)
        ]).astype(np.float32)
        for f in range(n_folds)
    }

    audio_embeddings = {}
    for i, seg_id in enumerate(seg_ids):
        if seg_id in audio_subset_ids:
            vec = np.zeros(n_folds, dtype=np.float32)
            vec[fold_of_idx[i]] = y[i] + 0.05
            audio_embeddings[seg_id] = vec

    return emb_by_fold, audio_embeddings


def test_score_audio_subset_reports_matched_symbolic_and_audio_tau_c():
    rng = np.random.default_rng(2026)
    n = 60
    # distinct -> vacuous disjointness
    composers = np.array([f"composer_{i}" for i in range(n)])
    y = rng.integers(0, 11, size=n).astype(float)
    seg_ids = [f"p{i:03d}" for i in range(n)]

    audio_subset = set(seg_ids[:20])
    emb_by_fold, audio_embeddings = _fold_signal_embeddings(
        y, composers, seg_ids, audio_subset, n_folds=5, seed=2026, rng=rng)

    # unused by this test's assertions
    features37_x = rng.normal(size=(n, 5)).astype(np.float32)

    result = score_audio_subset(emb_by_fold, audio_embeddings, features37_x, y,
                                 composers, seg_ids, n_folds=5, seed=2026)

    assert result["n"] == 20
    assert result["audio_tau_c"] > 0.9
    assert result["symbolic_tau_c"] > 0.9


def test_score_audio_subset_reports_features37_gate_paired_against_audio():
    rng = np.random.default_rng(2026)
    n = 60
    # distinct -> vacuous disjointness
    composers = np.array([f"composer_{i}" for i in range(n)])
    y = rng.integers(0, 11, size=n).astype(float)
    seg_ids = [f"p{i:03d}" for i in range(n)]

    audio_subset = set(seg_ids[:20])
    emb_by_fold, audio_embeddings = _fold_signal_embeddings(
        y, composers, seg_ids, audio_subset, n_folds=5, seed=2026, rng=rng)

    # A deliberately weak features37 stand-in (heavy noise on top of y) so the
    # near-perfect audio arm clearly beats it -- this fixture only needs to
    # prove the gate computes and pairs correctly, not that any real numbers hold.
    features37_x = np.column_stack([y + rng.normal(scale=4.0, size=n),
                                     rng.normal(size=(n, 4))]).astype(np.float32)

    result = score_audio_subset(emb_by_fold, audio_embeddings, features37_x, y,
                                 composers, seg_ids, n_folds=5, seed=2026)

    assert result["n"] == 20
    assert result["audio_tau_c"] > result["features37_tau_c"]
    assert result["delta_vs_features37"] > 0
    assert result["ci_lo_vs_features37"] > 0  # SIG on this fixture
    assert (result["ci_lo_vs_features37"] <= result["delta_vs_features37"]
            <= result["ci_hi_vs_features37"])


def test_build_wav_manifest_lists_only_the_eval_pieces_that_have_a_wav(tmp_path):
    """709 of the 900 eval pieces have a local WAV. A piece without one is
    omitted, never pointed at a missing file -- the gate's n must be the real
    audio that exists."""
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    (wav_dir / "p001.wav").write_bytes(b"RIFF")
    (wav_dir / "p003.wav").write_bytes(b"RIFF")

    entries = build_wav_manifest(["p001", "p002", "p003"], wav_dir)

    assert entries == [
        {"seg_id": "p001", "wav_path": str(wav_dir / "p001.wav")},
        {"seg_id": "p003", "wav_path": str(wav_dir / "p003.wav")},
    ]


def test_write_wav_manifest_mode_takes_its_seg_ids_from_features37(tmp_path):
    features37_dir = tmp_path / "features37"
    features37_dir.mkdir()
    for i, seg_id in enumerate(["p002", "p001", "p003"]):
        write_embedding_npz(features37_dir / f"{seg_id}.npz",
                            {"raw37": np.arange(37, dtype=np.float32)},
                            grade=i, composer_id=i)
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    (wav_dir / "p001.wav").write_bytes(b"RIFF")
    (wav_dir / "p003.wav").write_bytes(b"RIFF")
    manifest_path = tmp_path / "audio_wav_manifest.json"

    calls = []

    def fake_transcriber(wav_path):
        calls.append(wav_path)
        return ([], [])

    exit_code = main(["--write-wav-manifest", str(manifest_path),
                      "--wav-dir", str(wav_dir),
                      "--features37-dir", str(features37_dir)],
                     transcriber=fake_transcriber)

    assert exit_code == 0
    assert not calls  # manifest generation never transcribes
    assert [e["seg_id"] for e in json.loads(manifest_path.read_text())] == [
        "p001", "p003"]
