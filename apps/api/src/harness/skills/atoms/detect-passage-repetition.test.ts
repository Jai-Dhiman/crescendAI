import { test, expect } from 'vitest'
import { detectPassageRepetition } from './detect-passage-repetition'

test('detectPassageRepetition: three plays of bars 12-16 are detected, one-off bars 20-24 are not', async () => {
  const attempts = [
    { bar_range: [12, 16] as [number, number], time_ms: 0 },
    { bar_range: [20, 24] as [number, number], time_ms: 30000 },
    { bar_range: [12, 16] as [number, number], time_ms: 60000 },
    { bar_range: [12, 16] as [number, number], time_ms: 75000 },
  ]
  const result = await detectPassageRepetition.invoke({ attempts }) as { bar_range: [number, number]; attempt_count: number; first_attempt_ms: number; last_attempt_ms: number }[]
  expect(result).toHaveLength(1)
  expect(result[0].bar_range).toEqual([12, 16])
  expect(result[0].attempt_count).toBe(3)
  expect(result[0].first_attempt_ms).toBe(0)
  expect(result[0].last_attempt_ms).toBe(75000)
})
