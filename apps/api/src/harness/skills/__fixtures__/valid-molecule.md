---
name: fixture-molecule
tier: molecule
description: |
  A fixture molecule. fires when test runs. fires under validator. fires for fixture purposes.
  fires only in tests. fires when fixture loads. does NOT fire in prod. does NOT call other molecules.
dimensions: [pedaling, timing]
reads:
  signals: 'fixture signals'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [fixture-atom]
---

## When-to-fire
Fixture content.

## When-NOT-to-fire
Fixture content.

## Procedure
Fixture content.

## Concrete example
Fixture content.

## Post-conditions
Fixture content.
