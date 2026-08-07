# model/tests/follower_eval/test_calibration_recall.py
"""Unit tests for the G-OOD-0 calibration-recall gate (#148).

Hermetic: no audio, no transcriber weights, no parangonar. The parangonar call
is a single injected seam (``_match``), so every verdict path can be driven to a
known answer -- which matters more here than elsewhere, because this gate's
FAIL verdict is what stops a physical recording session.
"""

from __future__ import annotations

import json

import pytest
from follower_eval import calibration_recall as cr


def _notes(pitches, step=0.5):
    return [
        {"onset": i * step, "offset": i * step + 0.4, "pitch": p, "velocity": 70}
        for i, p in enumerate(pitches)
    ]


def _matcher(pairs):
    """A fake parangonar returning exactly ``pairs`` of (score_idx, perf_idx)."""

    def match(score_na, perf_na):
        return [
            {
                "label": "match",
                "score_id": str(score_na[s]["id"]),
                "performance_id": str(perf_na[p]["id"]),
            }
            for s, p in pairs
        ]

    return match


# --- note arrays ------------------------------------------------------------


def test_note_array_sorts_chords_by_pitch_so_both_streams_agree():
    """Two notes at the same onset must come out in the same order on the score
    side and the transcription side, or the timing-free arm splits every chord
    on arrival order rather than on content."""
    arr = cr._note_array(
        [
            {"onset": 0.0, "offset": 1.0, "pitch": 67},
            {"onset": 0.0, "offset": 1.0, "pitch": 60},
        ],
        "s",
    )
    assert [int(p) for p in arr["pitch"]] == [60, 67]


def test_jittered_chord_notes_still_come_out_in_pitch_order():
    """The regression that mattered. Transcribed onsets carry ~20 ms of jitter,
    so a chord's two notes no longer share an onset and a plain (onset, pitch)
    sort returns them in arrival order. Measured, that put the timing-free arm
    at 0.75 on a PERFECT transcription, which would make FAIL_MATCHER
    unreachable and misattribute every failure to the channel."""
    score = cr._note_array(
        [
            {"onset": 0.0, "offset": 1.0, "pitch": 72},
            {"onset": 0.0, "offset": 1.0, "pitch": 60},
        ],
        "s",
    )
    jittered = cr._note_array(
        [
            {"onset": 0.021, "offset": 1.0, "pitch": 72},
            {"onset": 0.033, "offset": 1.0, "pitch": 60},
        ],
        "p",
    )
    assert [int(p) for p in score["pitch"]] == [int(p) for p in jittered["pitch"]]
    assert cr.sequence_matched(
        [int(p) for p in score["pitch"]], [int(p) for p in jittered["pitch"]]
    ) == 2


def test_notes_further_apart_than_the_chord_window_keep_their_time_order():
    """Grouping must not reach across a real melodic step, or the arm would
    reorder a descending line into an ascending one and lose it."""
    arr = cr._note_array(
        [
            {"onset": 0.0, "offset": 0.1, "pitch": 72},
            {"onset": 0.5, "offset": 0.6, "pitch": 60},
        ],
        "p",
    )
    assert [int(p) for p in arr["pitch"]] == [72, 60]


def test_note_array_refuses_an_empty_stream():
    with pytest.raises(cr.CalibrationRecallError, match="zero notes"):
        cr._note_array([], "p")


def test_note_array_uses_seconds_as_beats():
    """parangonar mis-initializes when the score's nominal tempo is used, since
    a calibration take runs slower than the render (chroma_dtw_eval's measured
    finding). onset_beat must therefore equal onset_sec."""
    arr = cr._note_array(_notes([60, 62]), "p")
    assert list(arr["onset_beat"]) == list(arr["onset_sec"])


# --- arm 1: parangonar ------------------------------------------------------


def test_duplicate_score_matches_cannot_push_recall_above_one():
    """parangonar can pair the same score note twice. Counting entries rather
    than distinct score ids would report recall > 1.0 and read as a pass."""
    score = cr._note_array(_notes([60, 62]), "s")
    perf = cr._note_array(_notes([60, 60]), "p")
    matched = cr.parangonar_matched_score_ids(
        score, perf, _matcher([(0, 0), (0, 1)])
    )
    assert matched == {"s0"}


def test_unknown_ids_from_the_matcher_are_ignored():
    score = cr._note_array(_notes([60]), "s")
    perf = cr._note_array(_notes([60]), "p")

    def match(_s, _p):
        return [{"label": "match", "score_id": "s99", "performance_id": "p0"}]

    assert cr.parangonar_matched_score_ids(score, perf, match) == set()


def test_non_match_labels_do_not_count():
    score = cr._note_array(_notes([60, 62]), "s")
    perf = cr._note_array(_notes([60, 62]), "p")

    def match(s, p):
        return [
            {"label": "match", "score_id": "s0", "performance_id": "p0"},
            {"label": "deletion", "score_id": "s1", "performance_id": "p1"},
        ]

    assert cr.parangonar_matched_score_ids(score, perf, match) == {"s0"}


# --- arm 2: timing-free -----------------------------------------------------


def test_timing_free_arm_ignores_tempo_entirely():
    """The whole point of this arm: the same pitches played at any speed, with
    any rubato, must score 1.0. A clock-based arm would not."""
    assert cr.sequence_matched([60, 62, 64], [60, 62, 64]) == 3


def test_timing_free_arm_counts_a_dropped_note():
    assert cr.sequence_matched([60, 62, 64], [60, 64]) == 2


def test_timing_free_arm_is_monotone_not_a_multiset():
    """A bag-of-pitches count would score a scrambled transcription perfectly.
    Requiring a common SUBSEQUENCE means order still has to hold."""
    assert cr.sequence_matched([60, 62, 64], [64, 62, 60]) == 1


def test_timing_free_arm_ignores_spurious_extra_notes():
    """Room noise and the clap slates transcribe as notes. They are not recall
    misses, so inserting them must not lower this arm."""
    assert cr.sequence_matched([60, 62], [99, 60, 98, 62, 97]) == 2


def test_timing_free_arm_refuses_rather_than_hangs_on_a_huge_pair():
    big = [60] * 4000
    with pytest.raises(cr.CalibrationRecallError, match="timing-free arm refused"):
        cr.sequence_matched(big, big)


# --- matcher floor ----------------------------------------------------------


def test_matcher_floor_is_one_for_a_perfect_matcher():
    notes = _notes([60, 62, 64])
    floor = cr.matcher_floor(notes, _matcher([(0, 0), (1, 1), (2, 2)]))
    assert floor == pytest.approx(1.0)


def test_matcher_floor_exposes_matcher_loss_with_no_audio_involved():
    """Score against itself, so a miss here cannot be the channel, the piano or
    the performer -- it is the aligner, and it is the ceiling every take in the
    session is measured against."""
    notes = _notes([60, 62, 64, 65])
    assert cr.matcher_floor(notes, _matcher([(0, 0), (1, 1)])) == pytest.approx(0.5)


# --- verdicts ---------------------------------------------------------------


def test_pass_needs_the_real_pipeline_arm_not_the_timing_free_one():
    v, _ = cr.verdict(recall_parangonar=0.97, recall_sequence=0.99, floor=1.0)
    assert v == "PASS"


def test_a_good_channel_behind_a_bad_matcher_is_fail_matcher():
    """The notes are in the transcription and the aligner is losing them. The
    issue's remedy ('mic placement or the retrofit') is the wrong action here,
    so the verdict must not read the same as a channel failure."""
    v, why = cr.verdict(recall_parangonar=0.80, recall_sequence=0.98, floor=1.0)
    assert v == "FAIL_MATCHER"
    assert "microphone will not move this number" in why


def test_both_arms_low_is_fail_channel():
    v, why = cr.verdict(recall_parangonar=0.70, recall_sequence=0.72, floor=1.0)
    assert v == "FAIL_CHANNEL"
    assert "Phase 2 stops" in why


def test_a_matcher_floor_below_the_bar_makes_the_gate_uninformative():
    """Same precedent this issue already set for G-OOD-6: a gate whose control
    cannot separate the arms reports that it cannot discriminate, rather than a
    pass or a fail nobody can interpret."""
    v, why = cr.verdict(recall_parangonar=0.99, recall_sequence=0.99, floor=0.80)
    assert v == "UNINFORMATIVE"
    assert "do not report a pass or a fail" in why


def test_the_uninformative_check_outranks_a_would_be_pass():
    """A broken matcher that happens to score high on a take must not launder
    itself into a PASS."""
    assert cr.verdict(1.0, 1.0, floor=0.5)[0] == "UNINFORMATIVE"


def test_verdicts_use_only_the_one_declared_bar():
    """No second threshold is introduced anywhere: a value exactly at the bar
    passes, and one just under it does not."""
    assert cr.verdict(cr.G_OOD_0_BAR, 0.0, 1.0)[0] == "PASS"
    assert cr.verdict(cr.G_OOD_0_BAR - 1e-9, 0.0, 1.0)[0] == "FAIL_CHANNEL"


# --- manifest guards --------------------------------------------------------


def _manifest(tmp_path, **overrides):
    body = {
        "take_id": "cal_01",
        "reference_channel": "ref",
        "channels": {"ref": "ref.wav"},
        "behavior": "calibration",
        "score_midi": "score.mid",
    }
    body.update(overrides)
    (tmp_path / "ref.wav").write_bytes(b"")
    p = tmp_path / "take.json"
    p.write_text(json.dumps(body))
    return p


def test_a_practice_take_is_refused_not_scored(tmp_path):
    """Scoring G-OOD-0 on a take full of deliberate wrong notes would fail the
    gate on the performance and cancel Phase 2 for a reason that has nothing to
    do with the reference channel."""
    p = _manifest(tmp_path, behavior="practice")
    with pytest.raises(cr.CalibrationRecallError, match="not 'calibration'"):
        cr.reference_channel_path(p)


def test_a_take_with_no_declared_behavior_is_refused(tmp_path):
    body = json.loads(_manifest(tmp_path).read_text())
    del body["behavior"]
    (tmp_path / "take.json").write_text(json.dumps(body))
    with pytest.raises(cr.CalibrationRecallError, match="not 'calibration'"):
        cr.reference_channel_path(tmp_path / "take.json")


def test_a_take_with_no_known_score_is_refused(tmp_path):
    body = json.loads(_manifest(tmp_path).read_text())
    del body["score_midi"]
    (tmp_path / "take.json").write_text(json.dumps(body))
    with pytest.raises(cr.CalibrationRecallError, match="nothing to be recalled"):
        cr.reference_channel_path(tmp_path / "take.json")


def test_the_reference_channel_is_the_one_the_manifest_names(tmp_path):
    p = _manifest(
        tmp_path,
        reference_channel="dpa",
        channels={"dpa": "ref.wav", "pos1_phone": "a.wav"},
    )
    assert cr.reference_channel_path(p).name == "ref.wav"


def test_a_missing_score_midi_raises(tmp_path):
    with pytest.raises(cr.CalibrationRecallError, match="score MIDI missing"):
        cr.load_score_notes(tmp_path / "nope.mid")


def test_run_refuses_an_empty_session():
    with pytest.raises(cr.CalibrationRecallError, match="no calibration manifests"):
        cr.run([], lambda _w: ([], []))


# --- end to end -------------------------------------------------------------


def _session(tmp_path, pitches):
    """A minimal on-disk calibration take: a real WAV, a real score MIDI, a
    manifest. Only the transcriber is faked."""
    import subprocess

    import numpy as np
    import partitura as pa

    d = tmp_path / "cal_01"
    d.mkdir()
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i",
         "anullsrc=r=48000:cl=mono", "-t", "1", str(d / "ref.wav")],
        capture_output=True, check=True,
    )
    na = np.array(
        [(i * 0.5, 0.4, p, 70, f"n{i}") for i, p in enumerate(pitches)],
        dtype=[("onset_sec", float), ("duration_sec", float),
               ("pitch", int), ("velocity", int), ("id", "U32")],
    )
    pa.save_performance_midi(pa.performance.PerformedPart.from_note_array(na),
                             str(d / "score.mid"))
    (d / "take.json").write_text(json.dumps({
        "take_id": "cal_01", "reference_channel": "ref",
        "channels": {"ref": "ref.wav"},
        "behavior": "calibration", "score_midi": "score.mid",
    }))
    return d / "take.json"


def test_run_scores_a_real_session_end_to_end(tmp_path):
    """Exercises the parts the unit tests cannot: ffmpeg downsampling to the
    transcriber's 16 kHz, manifest path resolution, pooling and the verdict."""
    pitches = [60, 62, 64, 65, 67, 69, 71, 72] * 4
    manifest = _session(tmp_path, pitches)

    def transcribe(wav):
        assert wav.exists()
        return ([{"onset": i * 0.8, "offset": i * 0.8 + 0.5, "pitch": p,
                  "velocity": 70} for i, p in enumerate(pitches)], [])

    result = cr.run([manifest], transcribe)
    assert result["n_takes"] == 1
    assert result["n_score_notes"] == len(pitches)
    assert result["matcher_floor"] == pytest.approx(1.0)
    assert result["recall_parangonar"] >= cr.G_OOD_0_BAR
    assert result["verdict"] == "PASS"
    assert cr._format(result).count("G-OOD-0") == 1


def test_run_fails_the_gate_when_the_channel_loses_notes(tmp_path):
    """The consequential path: a reference channel that drops a third of the
    notes must stop Phase 2, not average itself into a pass."""
    pitches = [60, 62, 64, 65, 67, 69, 71, 72] * 4
    manifest = _session(tmp_path, pitches)
    kept = [p for i, p in enumerate(pitches) if i % 3]

    def transcribe(_wav):
        return ([{"onset": i * 0.8, "offset": i * 0.8 + 0.5, "pitch": p,
                  "velocity": 70} for i, p in enumerate(kept)], [])

    result = cr.run([manifest], transcribe)
    assert result["verdict"] == "FAIL_CHANNEL"
    assert result["recall_parangonar"] < cr.G_OOD_0_BAR


def test_run_raises_when_the_transcriber_returns_nothing(tmp_path):
    """A dead channel is an error, not a recall of 0.0 -- reporting it as a
    number would put a fabricated measurement into the gate table."""
    manifest = _session(tmp_path, [60, 62, 64])
    with pytest.raises(cr.CalibrationRecallError, match="returned zero notes"):
        cr.run([manifest], lambda _w: ([], []))
