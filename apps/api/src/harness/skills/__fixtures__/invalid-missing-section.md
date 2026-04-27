---
name: invalid-missing-procedure
tier: atom
description: |
  Missing the Procedure section. fires never. fires under test. fires for negative case.
  fires for validator. fires in unit tests. does NOT fire in prod. does NOT call other skills.
dimensions: [timing]
reads:
  signals: 'fixture signal'
  artifacts: []
writes: 'scalar:number'
depends_on: []
---

## When-to-fire
Fixture content.

## When-NOT-to-fire
Fixture content.

## Concrete example
Fixture content.

## Post-conditions
Fixture content.
