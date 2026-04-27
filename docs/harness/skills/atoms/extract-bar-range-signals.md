---
name: extract-bar-range-signals
tier: atom
description: |
  Slices the enrichment cache to return all signals (MuQ scores, AMT midi_notes, score-alignment, pedal CC) overlapping a bar range. fires when any molecule needs a unified view of signals over a passage. fires when cross-modal contradiction check needs all extractions for one slice. fires when phrasing-arc-analysis pulls a phrase. fires when exercise-proposal anchors a drill to specific bars. fires when bar-analyzer-style aggregation runs. does NOT fire across non-contiguous bar ranges (caller should call once per contiguous slice). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'enrichment cache entries (MuQ-quality, AMT-transcription, score-alignment, pedal CC) keyed by chunk_id with bar timing metadata'
  artifacts: []
writes: 'scalar:SignalBundle = { muq_scores: number[6][], midi_notes: Note[], pedal_cc: CcEvent[], alignment: Alignment[] }'
depends_on: []
---

## When-to-fire
Caller passes a session_id and bar_range. Atom returns the union of all overlapping enrichment cache entries projected to the bar range.

## When-NOT-to-fire
Do not invoke for non-contiguous ranges. Do not invoke when cache entries for the session have not finished writing (caller awaits write barrier).

## Procedure
1. For session_id, list all chunks whose bar coverage overlaps bar_range.
2. For each chunk, fetch its MuQ scores, AMT midi_notes (filtered to bars in range), pedal CC events (filtered), and score-alignment (filtered).
3. Concatenate, dedupe by (chunk_id, signal_type), and return the SignalBundle.

## Concrete example
Input: session_id='sess_42', bar_range=[12,16]. Three chunks overlap: chunk_a covers bars 10-14, chunk_b covers bars 14-18, chunk_c covers bars 18-22.
Output: muq_scores from chunks a+b only; midi_notes from a+b filtered to bars 12-16; etc.

## Post-conditions
All notes in midi_notes have onset_ms within bar_range; all alignment entries have bar in bar_range; muq_scores has one 6-vector per overlapping chunk.
