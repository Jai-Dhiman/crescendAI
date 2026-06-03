"""Verify pre-registered KILL/TUNE/PROCEED rule."""
from piece_id_eval.decision import decide


def test_kill_when_dtw_below_threshold() -> None:
    # DTW ceiling recall@10 < 0.70 -> KILL regardless of indexable
    assert decide(dtw_recall10=0.60, best_indexable_recall10=0.95, open_set_ok_flag=True) == "KILL"


def test_kill_at_exact_boundary() -> None:
    # Strictly less than 0.70; 0.699 -> KILL
    assert decide(dtw_recall10=0.699, best_indexable_recall10=0.90, open_set_ok_flag=True) == "KILL"


def test_proceed_when_all_criteria_met() -> None:
    # DTW >= 0.70, indexable >= 0.85, open_set_ok -> PROCEED
    assert decide(dtw_recall10=0.80, best_indexable_recall10=0.85, open_set_ok_flag=True) == "PROCEED"


def test_tune_when_dtw_ok_but_indexable_low() -> None:
    # DTW >= 0.70, indexable < 0.85 -> TUNE
    assert decide(dtw_recall10=0.75, best_indexable_recall10=0.80, open_set_ok_flag=True) == "TUNE"


def test_tune_when_dtw_ok_indexable_ok_but_open_set_fails() -> None:
    # DTW >= 0.70, indexable >= 0.85, but open-set not ok -> TUNE
    assert decide(dtw_recall10=0.80, best_indexable_recall10=0.90, open_set_ok_flag=False) == "TUNE"


def test_proceed_at_exact_indexable_boundary() -> None:
    # Exactly 0.85 should qualify as PROCEED if open_set_ok
    assert decide(dtw_recall10=0.70, best_indexable_recall10=0.85, open_set_ok_flag=True) == "PROCEED"


def test_tune_when_dtw_exactly_at_boundary() -> None:
    # Exactly 0.70 survives the KILL check; indexable low -> TUNE
    assert decide(dtw_recall10=0.70, best_indexable_recall10=0.60, open_set_ok_flag=True) == "TUNE"
