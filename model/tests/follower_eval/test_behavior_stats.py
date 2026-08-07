# model/tests/follower_eval/test_behavior_stats.py
"""Unit tests for the G-OOD-6 behavior statistics (issue #148).

These pin the two things the gate actually depends on: that each statistic
measures the behavior it claims to (constructed clips with known behavior), and
that the corpus denominator comes from the build manifests rather than a
directory glob."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from follower_eval import behavior_stats as bs


def _notes(onsets, pitch=60):
    return [
        {"onset": t, "offset": t + 0.2, "pitch": pitch, "velocity": 64} for t in onsets
    ]


# --- chord collapsing -------------------------------------------------------


def test_chord_events_collapses_simultaneous_notes():
    ev = bs.chord_events([(0.0, 60), (0.01, 64), (0.02, 67), (1.0, 72)])
    assert len(ev) == 2
    assert ev[0] == (0.0, (60, 64, 67))
    assert ev[1] == (1.0, (72,))


def test_chord_events_does_not_chain_a_rolled_chord_unboundedly():
    # Each note is within CHORD_SEC of the previous, but the span is 0.16s.
    # Greedy-from-event-start must break it up rather than absorb the chain.
    ev = bs.chord_events([(0.0, 60), (0.04, 62), (0.08, 64), (0.12, 65), (0.16, 67)])
    assert len(ev) > 1


# --- pause / rate statistics ------------------------------------------------


def test_pause_and_duration_on_a_clip_with_two_known_stops():
    # 20 events: steady 0.5s, then a 5s stop, then steady, then a 3s stop.
    times = [i * 0.5 for i in range(8)]
    times += [times[-1] + 5.0 + i * 0.5 for i in range(8)]
    times += [times[-1] + 3.0 + i * 0.5 for i in range(4)]
    b = bs.clip_behavior("p", "c", _notes(times))
    assert b.n_events == 20
    assert b.active_duration_s == pytest.approx(times[-1] - times[0])
    assert b.longest_pause_s == pytest.approx(5.0)
    # exactly two gaps >= PAUSE_SEC, expressed per minute
    assert b.pause_rate_per_min == pytest.approx(2 * 60.0 / b.active_duration_s)


def test_timing_statistics_are_missing_not_zero_below_the_floor():
    b = bs.clip_behavior("p", "c", _notes([0.0, 0.5, 1.0]))
    assert b.n_events == 3
    assert b.active_duration_s is None
    assert b.pause_rate_per_min is None
    assert b.longest_pause_s is None


# --- repetition -------------------------------------------------------------


def test_repeat_frac_zero_when_no_passage_recurs():
    notes = [
        {"onset": i * 0.5, "offset": i * 0.5 + 0.2, "pitch": 40 + i, "velocity": 64}
        for i in range(16)
    ]
    assert (
        bs.repeat_event_frac(
            [
                p
                for _, p in bs.chord_events(
                    [(float(n["onset"]), int(n["pitch"])) for n in notes]
                )
            ]
        )
        == 0.0
    )


def test_repeat_frac_detects_a_replayed_passage():
    # An 8-note phrase played twice: every event sits in a recurring n-gram.
    phrase = [40, 42, 44, 45, 47, 49, 51, 52]
    seq = [(p,) for p in phrase + phrase]
    assert bs.repeat_event_frac(seq) == pytest.approx(1.0)


def test_repeat_frac_is_timing_invariant():
    """A repeat played at a different tempo is still a repeat -- the statistic
    reads pitch-set sequence only, so it must not move."""
    phrase = [(p,) for p in [40, 42, 44, 45, 47, 49, 51, 52]]
    assert bs.repeat_event_frac(phrase * 2) == pytest.approx(1.0)


def test_repeat_frac_missing_when_clip_too_short_to_hold_two_occurrences():
    assert bs.repeat_event_frac([(60,)] * (bs.MIN_EVENTS_REPEAT - 1)) is None


# --- tempo jitter -----------------------------------------------------------


def test_jitter_is_zero_for_a_metronomic_clip():
    assert bs.local_tempo_jitter([i * 0.5 for i in range(30)]) == pytest.approx(0.0)


def test_jitter_rises_with_unevenness():
    steady = [i * 0.5 for i in range(30)]
    noisy, t = [], 0.0
    for step in [0.42, 0.55, 0.47, 0.61, 0.39, 0.52, 0.58, 0.44, 0.63, 0.41] * 3:
        noisy.append(t)
        t += step
    assert bs.local_tempo_jitter(noisy) > bs.local_tempo_jitter(steady)


def test_jitter_catches_alternating_unevenness():
    """Regression: a MEDIAN local reference scores a strictly alternating
    long-short clip at exactly 0.0 -- the window majority alternates with the
    values, so every IOI equals its own reference. Uneven hands and dotted-
    rhythm drift are alternating patterns, so that blind spot would hide one of
    the commonest amateur behaviors. The geometric-mean reference must see it."""
    uneven, t = [], 0.0
    for i in range(30):
        uneven.append(t)
        t += 0.35 if i % 2 else 0.65
    assert bs.local_tempo_jitter(uneven) > 0.3


def test_jitter_ignores_pauses_so_it_does_not_duplicate_pause_rate():
    """A metronomic clip interrupted by long stops is still steady WITHIN
    phrases; charging the stops here would make jitter redundant with
    pause_rate_per_min."""
    times = [i * 0.5 for i in range(15)]
    times += [times[-1] + 8.0 + i * 0.5 for i in range(15)]
    assert bs.local_tempo_jitter(times) == pytest.approx(0.0)


def test_jitter_is_tempo_invariant():
    slow = [i * 1.0 for i in range(30)]
    fast = [i * 0.25 for i in range(30)]
    assert bs.local_tempo_jitter(slow) == pytest.approx(bs.local_tempo_jitter(fast))


def test_jitter_absorbs_a_gradual_ritardando_as_local_tempo():
    """A smooth slowdown is a tempo change, not unsteadiness."""
    t, step, times = 0.0, 0.30, []
    for _ in range(40):
        times.append(t)
        t += step
        step *= 1.03
    assert bs.local_tempo_jitter(times) < 0.10


def test_jitter_missing_without_enough_continuous_playing():
    # Every gap is a pause -> no continuous IOIs at all.
    assert bs.local_tempo_jitter([i * 5.0 for i in range(20)]) is None


# --- corpus denominator -----------------------------------------------------


def _write_bundle(root: Path, piece: str, vid: str, n: int = 20):
    p = root / piece / f"{vid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"notes": _notes([i * 0.5 for i in range(n)])}))
    return p


def _write_manifest(root: Path, name: str, rows: list[tuple[str, str, str]]):
    (root / name).write_text(
        json.dumps(
            {"outcomes": [{"piece": p, "video_id": v, "status": s} for p, v, s in rows]}
        )
    )


def test_corpus_is_the_manifest_not_the_directory(tmp_path: Path):
    """The live bundles dir holds more files than the corpus. A glob would
    silently inflate the denominator and move every quantile."""
    _write_bundle(tmp_path, "fur_elise", "aaa")
    _write_bundle(tmp_path, "fur_elise", "bbb")
    _write_bundle(tmp_path, "fur_elise", "leftover_from_an_earlier_build")
    _write_manifest(
        tmp_path,
        "_manifest_g0.json",
        [("fur_elise", "aaa", "ok"), ("fur_elise", "bbb", "ok")],
    )

    clips = bs.corpus_clips(tmp_path)
    assert [c[1] for c in clips] == ["aaa", "bbb"]


def test_corpus_dedupes_across_manifests_and_excludes_failures(tmp_path: Path):
    _write_bundle(tmp_path, "fur_elise", "aaa")
    _write_bundle(tmp_path, "fur_elise", "bbb")
    _write_manifest(
        tmp_path,
        "_manifest_g0.json",
        [("fur_elise", "aaa", "ok"), ("fur_elise", "ccc", "download_fail")],
    )
    _write_manifest(
        tmp_path,
        "_manifest_g1.json",
        [
            ("fur_elise", "aaa", "skip"),
            ("fur_elise", "bbb", "ok"),
            ("fur_elise", "ddd", "transcribe_fail"),
        ],
    )

    assert [c[1] for c in bs.corpus_clips(tmp_path)] == ["aaa", "bbb"]


def test_corpus_raises_when_a_manifest_member_is_missing_from_disk(tmp_path: Path):
    _write_bundle(tmp_path, "fur_elise", "aaa")
    _write_manifest(
        tmp_path,
        "_manifest_g0.json",
        [("fur_elise", "aaa", "ok"), ("fur_elise", "gone", "ok")],
    )
    with pytest.raises(bs.BehaviorStatsError, match="absent from disk"):
        bs.corpus_clips(tmp_path)


def test_corpus_raises_without_manifests(tmp_path: Path):
    _write_bundle(tmp_path, "fur_elise", "aaa")
    with pytest.raises(bs.BehaviorStatsError, match="no _\\*manifest"):
        bs.corpus_clips(tmp_path)


def test_run_raises_on_an_empty_notes_bundle(tmp_path: Path):
    (tmp_path / "fur_elise").mkdir(parents=True)
    (tmp_path / "fur_elise" / "aaa.json").write_text(json.dumps({"notes": []}))
    _write_manifest(tmp_path, "_manifest_g0.json", [("fur_elise", "aaa", "ok")])
    with pytest.raises(bs.BehaviorStatsError, match="no 'notes'"):
        bs.run(tmp_path)


# --- summary / gate ---------------------------------------------------------


def test_summarize_reports_n_per_statistic_with_missing_counted(tmp_path: Path):
    rich = bs.clip_behavior("p", "long", _notes([i * 0.5 for i in range(40)]))
    poor = bs.clip_behavior("p", "short", _notes([0.0, 0.5, 1.0]))
    s = bs.summarize([rich, poor])
    assert s["active_duration_s"]["n"] == 1
    assert s["active_duration_s"]["n_missing"] == 1
    assert s["local_tempo_jitter"]["n"] == 1


def test_quantile_matches_linear_interpolation():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert bs._quantile(vals, 0.25) == pytest.approx(1.75)
    assert bs._quantile(vals, 0.5) == pytest.approx(2.5)
    assert bs._quantile(vals, 0.75) == pytest.approx(3.25)


def test_inside_iqr_is_inclusive_of_the_bounds():
    corpus = {"x": {"iqr": [1.0, 3.0]}}
    assert bs.inside_iqr("x", 1.0, corpus)
    assert bs.inside_iqr("x", 3.0, corpus)
    assert bs.inside_iqr("x", 2.0, corpus)
    assert not bs.inside_iqr("x", 3.001, corpus)


def test_score_arm_counts_statistics_inside_the_corpus_band():
    corpus = {
        "statistics": {n: {"iqr": [0.0, 1000.0], "n": 5} for n in bs.STATISTIC_NAMES}
    }
    arm = [
        bs.clip_behavior("p", f"c{i}", _notes([j * 0.5 for j in range(30)]))
        for i in range(3)
    ]
    scored = bs.score_arm(arm, corpus)
    assert scored["n_clips"] == 3
    assert scored["n_inside"] == 6 and scored["passes_bar"]


def test_score_arm_fails_when_medians_fall_outside():
    corpus = {
        "statistics": {n: {"iqr": [1e6, 2e6], "n": 5} for n in bs.STATISTIC_NAMES}
    }
    arm = [bs.clip_behavior("p", "c", _notes([j * 0.5 for j in range(30)]))]
    assert bs.score_arm(arm, corpus)["n_inside"] == 0


def test_separation_auc_is_half_for_identical_populations():
    pop = [
        bs.clip_behavior("p", f"c{i}", _notes([j * 0.5 for j in range(30)]))
        for i in range(4)
    ]
    auc = bs.separation_auc(pop, pop)
    assert auc["local_tempo_jitter"] == pytest.approx(0.5)


def test_separation_auc_saturates_for_disjoint_populations():
    short = [
        bs.clip_behavior("p", f"s{i}", _notes([j * 0.5 for j in range(10)]))
        for i in range(4)
    ]
    long = [
        bs.clip_behavior("p", f"l{i}", _notes([j * 0.5 for j in range(200)]))
        for i in range(4)
    ]
    assert bs.separation_auc(long, short)["active_duration_s"] == pytest.approx(1.0)


def test_run_is_deterministic(tmp_path: Path):
    for vid in ("aaa", "bbb", "ccc"):
        _write_bundle(tmp_path, "fur_elise", vid, n=30)
    _write_manifest(
        tmp_path,
        "_manifest_g0.json",
        [("fur_elise", v, "ok") for v in ("ccc", "aaa", "bbb")],
    )
    a, b = bs.run(tmp_path), bs.run(tmp_path)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert [c["clip"] for c in a["per_clip"]] == ["aaa", "bbb", "ccc"]
