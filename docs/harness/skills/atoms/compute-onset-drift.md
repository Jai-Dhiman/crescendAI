---
name: compute-onset-drift
tier: atom
description: |
  Computes per-note millisecond drift between performance onset and score-aligned expected onset. fires when timing molecules need drift signal. fires when rubato coaching evaluates intentional vs unintended deviation. fires when tempo stability triage runs. fires when cross-modal contradiction check needs timing evidence. fires when phrasing arc analysis weighs agogic accents. does NOT fire when score alignment is missing. does NOT call other skills.
dimensions: [timing]
reads:
  signals: 'AMT midi_notes (pitch, onset_ms) and score-alignment (per-note expected_onset_ms) within bar_range'
  artifacts: []
writes: 'scalar:OnsetDrift = { note_index: number, drift_ms: number, signed: number }[]'
depends_on: []
---

## When-to-fire
Caller passes a bar_range, performance midi_notes, and score-alignment for that range. Atom returns per-note drift values.

## When-NOT-to-fire
Do not invoke when score-alignment is unavailable (no score MIDI). Do not invoke on freely improvisatory passages where there is no notated reference.

## Procedure
1. For each performance note, look up its score-aligned expected_onset_ms.
2. drift_ms = abs(performance_onset - expected_onset).
3. signed = performance_onset - expected_onset (negative = early, positive = late).
4. Return ordered list keyed by note_index.

## Concrete example
Input: bar_range=[12,12], performance onsets [1000, 1500, 2000], score expected [1000, 1450, 2050].
Output: [{note_index:0, drift_ms:0, signed:0}, {note_index:1, drift_ms:50, signed:50}, {note_index:2, drift_ms:50, signed:-50}].

## Post-conditions
Returned list has one entry per performance note in bar_range. drift_ms is non-negative; signed may be negative (early), zero (on time), or positive (late).
