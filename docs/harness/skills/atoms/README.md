# Atoms

Single-purpose, near-deterministic building blocks. Atoms do not call other skills. One file per atom. See `../README.md` for three-tier overview.

**Final size:** 15 atoms.

## Computational primitives
- `compute-velocity-curve` -- per-bar mean MIDI velocity
- `compute-pedal-overlap-ratio` -- fraction of note duration covered by sustain pedal
- `compute-onset-drift` -- per-note ms drift from score-aligned expected onset
- `compute-dimension-delta` -- z-score of MuQ dimension vs baseline or cohort
- `compute-ioi-correlation` -- Pearson r between performer and score IOIs
- `compute-key-overlap-ratio` -- mean note overlap, articulation proxy
- `detect-passage-repetition` -- in-session bar-range repetition detection
- `prioritize-diagnoses` -- ranking policy for DiagnosisArtifact lists

## Retrieval primitives
- `fetch-student-baseline` -- per-dimension rolling mean+stddev for student
- `fetch-reference-percentile` -- cohort percentile rank for a dimension score
- `fetch-similar-past-observation` -- nearest prior diagnosis match for student+context
- `fetch-session-history` -- prior sessions in date window with synthesis+diagnoses

## Signal pipeline primitives
- `align-performance-to-score` -- DTW alignment of performance midi_notes to score
- `classify-stop-moment` -- logistic regression on MuQ scores -> stop probability
- `extract-bar-range-signals` -- enrichment cache slice for a bar range

Each atom is narrow, deterministic on a given input, makes no calls to other skills, and is independently testable. Each atom's contract lives in its own markdown file in this directory.
