---
name: fixture-compound
tier: compound
description: |
  A fixture compound. fires on OnFixtureHook. fires when test loads. fires for fixture purposes.
  fires under validator. fires in unit tests. does NOT fire in prod. does NOT call other compounds.
dimensions: [pedaling]
reads:
  signals: 'fixture signals'
  artifacts: [DiagnosisArtifact]
writes: SynthesisArtifact
depends_on: [fixture-molecule]
triggered_by: OnFixtureHook
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
