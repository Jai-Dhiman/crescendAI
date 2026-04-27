---
name: compute-velocity-curve
tier: atom
description: |
  Computes per-bar mean MIDI velocity across a bar range. fires when a molecule asks for velocity contour. fires when computing dynamic range. fires when phrasing arc analysis runs. fires when voicing diagnosis runs. fires when dynamic range audit runs. does NOT fire on raw audio (use MuQ scores instead). does NOT call other skills.
dimensions: [dynamics, phrasing]
reads:
  signals: 'AMT midi_notes (pitch, onset_ms, velocity) within bar_range; bar timing from score-alignment'
  artifacts: []
writes: 'scalar:VelocityCurve = { bar: number, mean_velocity: number, p90_velocity: number }[]'
depends_on: []
---

## When-to-fire
Caller passes a bar_range and AMT midi_notes for that range. Atom computes one curve point per bar.

## When-NOT-to-fire
Do not invoke on audio segments lacking AMT transcription; the atom assumes midi_notes is present and well-formed.

## Procedure
1. Group midi_notes by bar using bar timing from score-alignment.
2. For each bar, compute mean and p90 of the velocity field across all notes whose onset falls in that bar.
3. Return ordered list of { bar, mean_velocity, p90_velocity }.

## Concrete example
Input: bar_range=[12,14], midi_notes spanning bars 12-14 with velocities 60, 65, 70 (bar 12), 80, 85 (bar 13), 50, 55, 60 (bar 14).
Output: [{bar:12, mean_velocity:65, p90_velocity:69}, {bar:13, mean_velocity:82.5, p90_velocity:84.5}, {bar:14, mean_velocity:55, p90_velocity:59}].

## Post-conditions
Returned list has exactly bar_range[1] - bar_range[0] + 1 entries; each entry's mean_velocity and p90_velocity are in [0, 127]; entries are ordered by bar ascending.
