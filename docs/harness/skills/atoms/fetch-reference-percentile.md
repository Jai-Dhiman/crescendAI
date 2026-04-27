---
name: fetch-reference-percentile
tier: atom
description: |
  Fetches the cohort percentile rank of a MuQ dimension score for a given piece, level, and dimension. fires when no per-student baseline exists. fires when piece-onboarding compares to similarly-leveled cohort. fires when a diagnosis molecule needs a population-level reference. fires when dynamic-range audit grounds expectations. fires when articulation-clarity check needs a normative band. does NOT fire for student-personalized comparisons (use fetch-student-baseline). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'precomputed cohort percentile tables keyed by (piece_id | piece_level, dimension, percentile)'
  artifacts: []
writes: 'scalar:number = percentile in [0, 100] of the input score against the cohort'
depends_on: []
---

## When-to-fire
Caller passes a dimension, MuQ score, and either piece_id or piece_level. Atom returns the percentile rank in the cohort table.

## When-NOT-to-fire
Do not invoke when no cohort table exists for the requested key. Do not invoke for student-personalized comparisons.

## Procedure
1. Look up cohort table for (piece_id, dimension); fall back to (piece_level, dimension) if specific piece is missing.
2. Find the percentile bucket containing the input score using linear interpolation between adjacent percentile values.
3. Return percentile in [0, 100].

## Concrete example
Input: dimension='dynamics', score=0.72, piece_level='advanced'.
Output: 64 -- the score sits at the 64th percentile of advanced-level cohort dynamics scores.

## Post-conditions
Returned value is in [0, 100]. Returns 50 by convention when the input score equals the cohort median.
