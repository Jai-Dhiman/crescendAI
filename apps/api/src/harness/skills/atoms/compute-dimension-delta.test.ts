import { test, expect } from 'vitest'
import { computeDimensionDelta } from './compute-dimension-delta'

test('computeDimensionDelta: concrete example from spec returns z-score -2.3', async () => {
  // (0.42 - 0.65) / 0.10 = -2.3
  const result = await computeDimensionDelta.invoke({
    dimension: 'pedaling',
    current: 0.42,
    baseline: { mean: 0.65, stddev: 0.10 },
  })
  expect(result).toBeCloseTo(-2.3, 5)
})
