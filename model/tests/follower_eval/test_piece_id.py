# model/tests/follower_eval/test_piece_id.py
"""Unit tests for the piece-ID stage's own logic (issue #133). The follower and
score loaders are covered elsewhere; here we pin the ngram shortlist, the verify
window, and the coverage*confidence decision on constructed inputs."""
from __future__ import annotations

from follower_bench.segments import PerfNote

from follower_eval import piece_id as pid


def _pn(onset: float, pitch: int) -> PerfNote:
    return PerfNote(onset=onset, offset=onset + 0.2, pitch=pitch, velocity=60)


def _cand(sid, cov, conf, votes=0, source="ngram") -> pid.Candidate:
    return pid.Candidate(score_id=sid, source=source, ngram_votes=votes,
                         coverage=cov, confidence=conf)


def test_ngram_shortlist_votes_and_rank():
    # index maps a pitch-trigram -> [(score_id, pos), ...]
    idx = {
        "60,62,64": [["scoreA", 0], ["scoreB", 5]],
        "62,64,65": [["scoreA", 1]],
    }
    perf = [_pn(0, 60), _pn(1, 62), _pn(2, 64), _pn(3, 65)]  # trigrams 60,62,64 and 62,64,65
    ranked = pid.ngram_shortlist(perf, idx, k=5)
    assert ranked[0] == ("scoreA", 2)   # matched both trigrams
    assert ("scoreB", 1) in ranked


def test_window_takes_mid_slice():
    perf = [_pn(float(i), 60 + (i % 5)) for i in range(100)]  # onsets 0..99
    win = pid._window(perf, window_sec=10.0)
    # starts at 25% in (~t=24.75) and spans 10s -> onsets ~25..34
    assert all(24.0 <= n.onset <= 36.0 for n in win)
    assert 8 <= len(win) <= 14


def test_window_tiny_input_falls_back():
    perf = [_pn(0, 60), _pn(1, 61)]
    win = pid._window(perf, window_sec=10.0)
    assert len(win) >= 2   # never returns fewer than what's there


def test_decide_accepts_clear_winner():
    # right score: high cov AND high conf; wrong: covers but low conf
    cands = sorted([_cand("right", 0.62, 0.84), _cand("wrong", 0.51, 0.06)],
                   key=pid._decision_score, reverse=True)
    assert pid.decide(cands) == "right"


def test_decide_confirms_label_over_high_coverage_wrong():
    # Bach case: label covers 0.99/0.97; a wrong ngram cand covers 0.70 but conf 0.41
    cands = sorted([_cand("bach.prelude.bwv_846", 0.99, 0.97, source="label"),
                    _cand("beethoven.sonata", 0.70, 0.41)],
                   key=pid._decision_score, reverse=True)
    assert pid.decide(cands) == "bach.prelude.bwv_846"


def test_decide_abstains_on_low_confidence():
    # all candidates cover moderately but none is confident -> abstain
    cands = sorted([_cand("a", 0.70, 0.30), _cand("b", 0.66, 0.28)],
                   key=pid._decision_score, reverse=True)
    assert pid.decide(cands) == "ABSTAIN"


def test_decide_abstains_on_thin_margin():
    # two confident, near-tied candidates (ambiguous) -> abstain
    cands = sorted([_cand("a", 0.80, 0.80), _cand("b", 0.78, 0.80)],
                   key=pid._decision_score, reverse=True)
    assert pid.decide(cands) == "ABSTAIN"


def test_decide_empty():
    assert pid.decide([]) == "ABSTAIN"
