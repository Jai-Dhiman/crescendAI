---
name: align-performance-to-score
tier: atom
description: |
  Aligns AMT-transcribed performance midi_notes to a score MIDI via DTW on (onset, pitch). fires when any timing molecule needs per-note score correspondence. fires when cross-modal contradiction check needs aligned context. fires when bar-range slicing requires score-anchored bar timings. fires when score-following bar regressions are computed. fires when exercise-proposal anchors a drill to specific bars. does NOT fire when score MIDI is not available. does NOT call other skills.
dimensions: [timing, articulation]
reads:
  signals: 'AMT performance midi_notes (pitch, onset_ms, duration_ms) and reference score MIDI (pitch, expected_onset_ms)'
  artifacts: []
writes: 'scalar:Alignment = { perf_index: number, score_index: number, expected_onset_ms: number, bar: number }[]'
depends_on: []
---

## When-to-fire
Caller passes performance midi_notes and reference score MIDI for a piece. Atom returns the per-performance-note alignment to score notes (DTW-best path on onset+pitch joint cost).

## When-NOT-to-fire
Do not invoke when the score MIDI is not loaded for the piece. Do not invoke on freely improvisatory passages.

## Procedure
1. Build cost matrix: cost(perf_i, score_j) = |perf_i.onset - score_j.expected_onset_normalized| + 100 * (perf_i.pitch != score_j.pitch).
2. Run DTW with monotonic constraint to find best alignment path.
3. For each performance note, emit { perf_index, score_index, expected_onset_ms, bar }.
4. Drop performance notes whose alignment cost exceeds threshold 500 (mark as unaligned: score_index = -1).

## Concrete example
Input: performance notes [pitch=60 onset=1000, pitch=64 onset=1500], score notes [pitch=60 expected=1000 bar=12, pitch=64 expected=1450 bar=12].
Output: [{perf_index:0, score_index:0, expected_onset_ms:1000, bar:12}, {perf_index:1, score_index:1, expected_onset_ms:1450, bar:12}].

## Post-conditions
Returned list has one entry per performance note. score_index is -1 for unaligned notes; expected_onset_ms is null for unaligned notes; bar is the score bar number for aligned notes.
