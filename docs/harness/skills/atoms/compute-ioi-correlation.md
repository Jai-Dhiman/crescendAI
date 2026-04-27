---
name: compute-ioi-correlation
tier: atom
description: |
  Computes Pearson correlation between performance inter-onset intervals and score-expected inter-onset intervals over a bar range. fires when tempo-stability triage runs. fires when rubato-coaching distinguishes intentional from unintended deviation. fires when phrasing-arc-analysis weighs agogic structure. fires when cross-modal contradiction check needs a structured timing scalar. fires when weekly review tracks ioi-coherence trend. does NOT fire when fewer than 4 aligned notes exist in the range. does NOT call other skills.
dimensions: [timing, phrasing]
reads:
  signals: 'AMT performance midi_notes (onset_ms) and score-aligned expected_onset_ms within bar_range'
  artifacts: []
writes: 'scalar:number = Pearson r in [-1, 1]; null if fewer than 4 aligned notes'
depends_on: []
---

## When-to-fire
Caller passes performance midi_notes with score-aligned expected_onset_ms for a bar range. Atom returns Pearson r between adjacent-note IOIs.

## When-NOT-to-fire
Do not invoke when there are fewer than 4 aligned notes (correlation is unreliable). Do not invoke when score alignment is missing.

## Procedure
1. Compute performance IOIs: ioi_perf[i] = perf[i+1].onset - perf[i].onset.
2. Compute score IOIs: ioi_score[i] = score[i+1].expected_onset - score[i].expected_onset.
3. If len(ioi_perf) < 3, return null.
4. Return Pearson r between the two arrays.

## Concrete example
Input: 5 aligned notes with performance IOIs [400, 410, 390, 420] and score IOIs [400, 400, 400, 400] (rigid metronome).
Output: ~0.0 (low correlation -- performer rubato is uncorrelated with score timing).

## Post-conditions
Returned value is null OR a finite number in [-1, 1].
