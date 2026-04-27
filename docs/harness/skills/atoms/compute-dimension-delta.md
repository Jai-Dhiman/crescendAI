---
name: compute-dimension-delta
tier: atom
description: |
  Computes z-score delta between a performance MuQ dimension score and the student's baseline (or, if no baseline, the cohort percentile). fires when any diagnosis molecule needs to know whether a dimension regressed. fires when cross-modal contradiction check needs MuQ-side magnitude. fires when prioritize-diagnoses ranks by severity. fires when session synthesis aggregates dimension performance. fires when weekly review compares against past weeks. does NOT fire on raw audio. does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'MuQ 6-dim score for current chunk, plus reference baseline mean+stddev (per dimension) for either student or cohort'
  artifacts: []
writes: 'scalar:number = signed z-score; negative = below baseline, positive = above'
depends_on: []
---

## When-to-fire
Caller passes a dimension name, current MuQ score for that dimension, and a baseline {mean, stddev}. Atom returns (current - mean) / stddev.

## When-NOT-to-fire
Do not invoke when baseline stddev == 0 (returns 0 by convention, but caller should treat as no signal). Do not invoke when the dimension name is not one of the 6 teacher-grounded dimensions.

## Procedure
1. Validate dimension is one of [dynamics, timing, pedaling, articulation, phrasing, interpretation].
2. If baseline.stddev == 0, return 0.
3. Return (current - baseline.mean) / baseline.stddev.

## Concrete example
Input: dimension='pedaling', current=0.42, baseline={mean: 0.65, stddev: 0.10}.
Output: -2.3 (significant regression below baseline).

## Post-conditions
Returned value is a finite number; negative indicates below baseline, positive above. Returns 0 when stddev is 0.
