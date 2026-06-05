# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: notes.py
- REPO_ROOT uses parents[3] (blocker-1 fix from 49d39948; original plan had parents[4]).
- load_score_notes uses bars[].notes[] with onset_seconds/duration_seconds, verified against bach.prelude.bwv_846.json.
- 6 tests pass.

## Task 2: transcribe.py
- Reuses _read_wav_16k_mono and _transcribe_clip from chroma_dtw_eval.amt_regen as private imports.
- Default path anchor uses _MODULE_DIR.parents[1] (= model/) per blocker-2 fix from 49d39948.
- 4 tests pass.

## Task 3: windowing.py
- Short-recording fallback returns [notes] (single window), not n_starts copies.
- Empty notes returns [[]] (plan gap, not tested, matches spec intent).
- 6 tests pass.

## Task 4: note_chroma.py
- Key-dependent (no OTI). pitch % 12 -> bin 0-11.
- Per-frame L2 normalization; silent frames left as zeros.
- 7 tests pass.

## Task 5: corruption.py
- Three modes applied in order: deletion, jitter, insertion. Output sorted by onset.
- Insertion pitch clamped [21, 108]; onset_shift in [-1.0, 1.0].
- 6 tests pass.

## Task 6: open_set.py
- best_point returns qualifying point with lowest FA (not highest TA).
- 6 tests pass.

## Task 7: decision.py
- Docstring updated: FA <= 0.10 @ TA >= 0.75 -> FA <= 0.05 @ TA >= 0.60.
- Args docstring updated to reference note-based indexable matchers.
- Logic was already correct from 49d39948 blocker fix.
- 8 tests pass.

## Task 8: matchers/base.py + NoteChromaMatcher (C1)
- base.py: removed numpy import, rank() signature changed np.ndarray -> list[Note].
- NoteChromaMatcher: pre-computes chroma vectors at construction; cosine via np.dot.
- test_matchers.py: replaced old chroma-based tests with note-based suite.
- 3 tests pass.

## Task 9: landmark.py (C2)
- Ordinal tokens (pc_anchor, interval, ordinal_gap); interval clamped [-12, 12]; MAX_GAP=5.
- Inverted index deduplicates tokens per piece at build time (seen set).
- 4 tests pass.

## Task 10: dtw_ceiling.py (C3 note-based)
- Complete replacement of old chroma-DTW implementation.
- Subsequence sliding-window pitch DTW; normalized by query length; negated to score.
- 5 tests pass.

## Task 11: chroma_seq_dtw.py (C4)
- Subsequence DTW on chroma_sequence (12, T) at frame_seconds=0.5.
- Column-wise Euclidean; sliding window; normalized by Q frames.
- 5 tests pass.

## Task 12: bakeoff.py + dead-file deletions
- Open-set oracle uses NoteChromaMatcher (cosine in [-1,1]) not DtwCeilingMatcher (scores <= 0). This is blocker-3 fix.
- Threshold sweep [0.00..1.00] at 0.01 intervals (101 points).
- Trackio block catches (ImportError, RuntimeError) and logs to stderr; no silent swallow.
- Matcher .name read from existing matchers list; no redundant instantiations.
- Deleted 12 dead files: chord_ngram.py, twodft.py, query_chroma.py, score_chroma.py, query_set.py, cli.py, report.py, and their 5 test counterparts.
- 6 tests pass (including CLI smoke and open_set_ok oracle correctness).

## Final suite: 76 passed in 2.47s
