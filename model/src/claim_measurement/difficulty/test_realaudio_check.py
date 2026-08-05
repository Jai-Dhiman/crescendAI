"""Tests for realaudio_check (#149 / #138 Phase 1) -- the real-audio second
gate: MIDI drift, resumable transcription, and per-fold audio scoring.

Run: cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov
"""
import json

import numpy as np
import pytest

from claim_measurement.difficulty.realaudio_check import midi_drift


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


def test_score_audio_subset_reports_matched_symbolic_and_audio_tau_c():
    rng = np.random.default_rng(2026)
    n = 60
    # distinct -> vacuous disjointness
    composers = np.array([f"composer_{i}" for i in range(n)])
    y = rng.integers(0, 11, size=n).astype(float)
    seg_ids = [f"p{i:03d}" for i in range(n)]

    emb_by_fold = {
        f: np.column_stack([y, rng.normal(size=(n, 2)) * 0.01]).astype(np.float32)
        for f in range(5)
    }
    audio_subset = set(seg_ids[:20])
    audio_embeddings = {
        # 3 columns to match emb_by_fold's 3-column shape (y + 2 noise cols) --
        # score_audio_subset scores this row through the SAME ridge model fit
        # on emb_by_fold[fold], so the feature count must match exactly.
        seg_id: np.array([y[i] + 0.05, 0.0, 0.0], dtype=np.float32)
        for i, seg_id in enumerate(seg_ids) if seg_id in audio_subset
    }

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

    emb_by_fold = {
        f: np.column_stack([y, rng.normal(size=(n, 2)) * 0.01]).astype(np.float32)
        for f in range(5)
    }
    # A deliberately weak features37 stand-in (heavy noise on top of y) so the
    # near-perfect audio arm clearly beats it -- this fixture only needs to
    # prove the gate computes and pairs correctly, not that any real numbers hold.
    features37_x = np.column_stack([y + rng.normal(scale=4.0, size=n),
                                     rng.normal(size=(n, 4))]).astype(np.float32)
    audio_subset = set(seg_ids[:20])
    audio_embeddings = {
        seg_id: np.array([y[i] + 0.05, 0.0, 0.0], dtype=np.float32)
        for i, seg_id in enumerate(seg_ids) if seg_id in audio_subset
    }

    result = score_audio_subset(emb_by_fold, audio_embeddings, features37_x, y,
                                 composers, seg_ids, n_folds=5, seed=2026)

    assert result["n"] == 20
    assert result["audio_tau_c"] > result["features37_tau_c"]
    assert result["delta_vs_features37"] > 0
    assert result["ci_lo_vs_features37"] > 0  # SIG on this fixture
    assert (result["ci_lo_vs_features37"] <= result["delta_vs_features37"]
            <= result["ci_hi_vs_features37"])
