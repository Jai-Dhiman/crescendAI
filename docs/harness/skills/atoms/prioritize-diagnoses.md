---
name: prioritize-diagnoses
tier: atom
description: |
  Ranks a list of DiagnosisArtifacts by severity, then confidence, then dimension priority. fires when session-synthesis selects top-N focus areas. fires when weekly-review surfaces dominant patterns. fires when live-practice-companion picks which diagnosis to surface first. fires when exercise-proposal needs diagnosis ranking to pick which to address. fires when prioritize ranking is needed for any compound aggregation. does NOT fire on a single diagnosis (no ordering needed). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'list of DiagnosisArtifact objects passed in by caller (not from cache)'
  artifacts: []
writes: 'scalar:RankedDiagnosisList = DiagnosisArtifact[] (input list reordered, no mutation)'
depends_on: []
---

## When-to-fire
Caller passes an array of DiagnosisArtifacts. Atom returns them reordered by composite priority.

## When-NOT-to-fire
Do not invoke on an empty list (return empty). Do not invoke on a single diagnosis (no work).

## Procedure
1. Compute priority key for each: (severity_rank, confidence_rank, dimension_priority).
   - severity_rank: significant=3, moderate=2, minor=1.
   - confidence_rank: high=3, medium=2, low=1.
   - dimension_priority: pedaling=6, timing=5, dynamics=4, phrasing=3, articulation=2, interpretation=1.
2. Sort descending by tuple (severity_rank, confidence_rank, dimension_priority).
3. finding_type='strength' diagnoses sort to the END regardless of severity (strengths are not focus_areas).

## Concrete example
Input: three diagnoses -- A (severity:moderate, confidence:high, dim:timing), B (severity:significant, confidence:medium, dim:articulation), C (severity:minor, confidence:high, dim:pedaling, finding_type:strength).
Output: [B, A, C] -- B first (significant), A second (moderate>minor), C last (strength regardless).

## Post-conditions
Output length equals input length; output contains exactly the same artifact objects (referential equality preserved); strengths appear last.
