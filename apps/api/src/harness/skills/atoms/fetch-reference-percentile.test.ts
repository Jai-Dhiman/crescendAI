import { test, expect } from 'vitest'
import { fetchReferencePercentile } from './fetch-reference-percentile'

test('fetchReferencePercentile: score 0.72 between p60=0.70 and p70=0.75 returns ~64', async () => {
  // interpolation: p = 60 + (0.72 - 0.70) / (0.75 - 0.70) * (70 - 60) = 60 + 0.4 * 10 = 64
  const cohort_table = [
    { p: 0, value: 0.30 },
    { p: 50, value: 0.62 },
    { p: 60, value: 0.70 },
    { p: 70, value: 0.75 },
    { p: 90, value: 0.85 },
    { p: 100, value: 1.00 },
  ]
  const result = await fetchReferencePercentile.invoke({
    dimension: 'dynamics',
    score: 0.72,
    cohort_table,
  })
  expect(result).toBeCloseTo(64, 0)
})
