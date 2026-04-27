---
name: compute-key-overlap-ratio
tier: atom
description: |
  Computes the average ratio of (note_off_ms - next_note_on_ms) to total note duration, an articulation proxy where high values indicate legato and low values indicate staccato. fires when articulation-clarity check runs. fires when cross-modal contradiction check needs an AMT-side articulation scalar. fires when phrasing-arc-analysis distinguishes legato vs detached phrases. fires when piece-onboarding compares articulation against reference style. fires when exercise-proposal needs articulation prerequisite check. does NOT fire when fewer than 3 consecutive notes exist. does NOT call other skills.
dimensions: [articulation]
reads:
  signals: 'AMT midi_notes (onset_ms, duration_ms) for adjacent notes within bar_range'
  artifacts: []
writes: 'scalar:number = mean overlap ratio; positive = legato, near-zero = detache, negative = staccato (gap)'
depends_on: []
---

## When-to-fire
Caller passes a sequence of consecutive monophonic notes. Atom returns the mean per-pair overlap ratio.

## When-NOT-to-fire
Do not invoke on polyphonic passages without monophonic projection (caller must reduce to a single voice first). Do not invoke when fewer than 3 notes are present.

## Procedure
1. For each adjacent pair (note_i, note_i+1), compute overlap = (note_i.onset + note_i.duration) - note_i+1.onset.
2. Normalize: pair_ratio = overlap / note_i.duration.
3. Return mean pair_ratio across all pairs.

## Concrete example
Input: notes [(0, 500), (450, 500), (950, 500)] (50ms overlap each).
Output: 0.10 (mild legato).

## Post-conditions
Returned value is a finite number. Positive values indicate overlap (legato); zero indicates detache; negative indicates gap (staccato).
