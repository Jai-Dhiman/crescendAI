---
name: classify-stop-moment
tier: atom
description: |
  Returns probability that a teacher would stop the student at this audio chunk, given MuQ 6-dim scores. fires for every 15s audio chunk during live-practice-companion. fires for chunk selection in session-synthesis. fires when picking the dominant-issue chunk for cross-modal contradiction check. fires when ranking chunks for weekly review surfacing. fires when piece-onboarding picks the most-stoppable demonstration chunk. does NOT fire on raw audio (run MuQ first). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'MuQ 6-dim quality scores for one audio chunk (vector of 6 floats in [0, 1])'
  artifacts: []
writes: 'scalar:number = stop probability in [0, 1]'
depends_on: []
---

## When-to-fire
Caller passes a MuQ 6-dim score vector for one chunk. Atom returns the logistic-regression-derived probability that a teacher would stop here.

## When-NOT-to-fire
Do not invoke without MuQ scores. Do not invoke on chunks shorter than the model's expected window (15s).

## Procedure
1. Apply pre-trained logistic regression coefficients (loaded from STOP classifier weights, see apps/api/src/wasm/score-analysis/src/stop.rs for the implementation reference).
2. Return sigmoid(coefficients dot scores + intercept).

## Concrete example
Input: scores=[0.4, 0.3, 0.2, 0.5, 0.4, 0.4] (low pedaling).
Output: 0.78 (high stop probability).

## Post-conditions
Returned value is in [0, 1]. Output is deterministic given the same input vector and the same loaded coefficients.
