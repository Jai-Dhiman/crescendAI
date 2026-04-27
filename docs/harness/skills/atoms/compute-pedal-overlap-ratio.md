---
name: compute-pedal-overlap-ratio
tier: atom
description: |
  Computes the fraction of note duration covered by sustain pedal (CC64 >= 64) within a bar range. fires when pedal triage runs. fires when cross-modal contradiction check inspects pedaling. fires when score-aligned pedaling is audited. fires when dynamic-range audit needs pedal context. fires when pedal_isolation exercise prerequisites are checked. does NOT fire when AMT pedal CC is missing. does NOT call other skills.
dimensions: [pedaling]
reads:
  signals: 'AMT midi_notes (onset_ms, duration_ms) and AMT pedal CC64 timeline within bar_range'
  artifacts: []
writes: 'scalar:number = fraction in [0, 1]'
depends_on: []
---

## When-to-fire
Caller passes a bar_range with AMT midi_notes and AMT pedal CC64 timeline. Atom returns the time-weighted fraction of note duration during which the sustain pedal is depressed.

## When-NOT-to-fire
Do not invoke on bar ranges where AMT pedal CC is missing or where midi_notes is empty (returns 0 in both, but the caller should handle the missing-data case explicitly).

## Procedure
1. For each note in the range, compute its [onset_ms, onset_ms + duration_ms] interval.
2. Compute total note duration: sum over notes of duration_ms.
3. Compute pedaled note duration: for each note, integrate the time during its interval where CC64 >= 64.
4. Return pedaled / total. If total == 0, return 0.

## Concrete example
Input: bar_range=[12,16], two notes (onset 0ms dur 1000ms, onset 500ms dur 500ms), pedal depressed [200ms, 800ms].
Output: 0.6 -- note1 has 600ms pedaled of 1000ms; note2 has 300ms pedaled of 500ms; total pedaled=900ms / total=1500ms = 0.6.

## Post-conditions
Returned value is a number in [0, 1]. Returns 0 when there are no notes in the range.
