"""Key estimation via Krumhansl-Schmuckler profiles.

Derives the best-matching key from a ScoreData pitch-class histogram by
correlating against all 24 major+minor Krumhansl profiles.  Used as the
expected_key in the self-consistency gate for bulk ingests where no
hand-authored key label is available.
"""

from __future__ import annotations

from score_library.schema import ScoreData
from score_library.validate import KRUMHANSL_MAJOR, KRUMHANSL_MINOR, _pearson

_TONIC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def estimate_key(score: ScoreData) -> str:
    """Return the best-fitting Krumhansl key for the score as 'X major|minor'.

    Builds a pitch-class histogram over all notes in the score, then correlates
    it against all 24 major + 24 minor Krumhansl profiles (rotated per tonic)
    and returns the argmax key label.

    The returned string is in the same format expected by validate_score's
    key_agreement check (e.g. 'C major', 'F# minor').
    """
    pc_counts = [0] * 12
    for bar in score.bars:
        for note in bar.notes:
            pc_counts[note.pitch % 12] += 1

    total = sum(pc_counts)
    if total == 0:
        return "C major"

    histogram = [c / total for c in pc_counts]

    best_key = "C major"
    best_corr = float("-inf")

    for tonic_pc, tonic_name in enumerate(_TONIC_NAMES):
        major_profile = [KRUMHANSL_MAJOR[(pc - tonic_pc) % 12] for pc in range(12)]
        corr = _pearson(histogram, major_profile)
        if corr > best_corr:
            best_corr = corr
            best_key = f"{tonic_name} major"

        minor_profile = [KRUMHANSL_MINOR[(pc - tonic_pc) % 12] for pc in range(12)]
        corr = _pearson(histogram, minor_profile)
        if corr > best_corr:
            best_corr = corr
            best_key = f"{tonic_name} minor"

    return best_key
