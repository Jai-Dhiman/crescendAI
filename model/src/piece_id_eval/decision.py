"""Pre-registered KILL / TUNE / PROCEED gate for the piece-ID feasibility harness.

Rule (pre-registered before any real data is collected):
  KILL    if DtwCeilingMatcher recall@10 < 0.70
  PROCEED if some indexable matcher recall@10 >= 0.85
           AND open_set_ok (FA <= 0.10 at TA >= 0.75)
  TUNE    otherwise
"""
from __future__ import annotations

_DTW_KILL_THRESHOLD = 0.70
_INDEXABLE_PROCEED_THRESHOLD = 0.85


def decide(
    dtw_recall10: float,
    best_indexable_recall10: float,
    open_set_ok_flag: bool,
) -> str:
    """Return 'KILL', 'PROCEED', or 'TUNE' based on pre-registered thresholds.

    Args:
        dtw_recall10: recall@10 of DtwCeilingMatcher (the discrimination ceiling).
        best_indexable_recall10: max recall@10 across ChordNgramMatcher and TwoDFTMatcher.
        open_set_ok_flag: True iff an open-set threshold exists with FA<=0.10, TA>=0.75.

    Returns:
        'KILL' | 'PROCEED' | 'TUNE'
    """
    if dtw_recall10 < _DTW_KILL_THRESHOLD:
        return "KILL"
    if best_indexable_recall10 >= _INDEXABLE_PROCEED_THRESHOLD and open_set_ok_flag:
        return "PROCEED"
    return "TUNE"
