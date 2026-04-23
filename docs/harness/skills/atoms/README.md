# Atoms

Single-purpose, near-deterministic building blocks. Atoms do not call other skills. One file per atom. See `../README.md` for three-tier overview.

**Target size:** ~10-15 atoms.

Populated by V5 work. Initial candidate list (names subject to refinement during decomposition):

- `compute-velocity-curve`
- `compute-pedal-overlap-ratio`
- `compute-onset-drift`
- `compute-dimension-delta`
- `fetch-student-baseline`
- `fetch-reference-percentile`
- `fetch-similar-past-observation`
- `align-performance-to-score`
- `classify-stop-moment`
- `extract-bar-range-signals`
- `compute-ioi-correlation` (inter-onset interval correlation)
- `compute-key-overlap-ratio` (articulation proxy)
- `detect-passage-repetition`

Each atom must be: narrow, deterministic on a given input, no calls to other skills, independently testable.
