import { test, expect } from 'vitest'
import { computeOnsetDrift } from './compute-onset-drift'

test('computeOnsetDrift: concrete example from spec produces correct per-note drift', async () => {
  const notes = [
    { onset_ms: 1000, expected_onset_ms: 1000 },
    { onset_ms: 1500, expected_onset_ms: 1450 },
    { onset_ms: 2000, expected_onset_ms: 2050 },
  ]
  const result = await computeOnsetDrift.invoke({ notes }) as { note_index: number; drift_ms: number; signed: number }[]
  expect(result).toHaveLength(3)
  expect(result[0]).toEqual({ note_index: 0, drift_ms: 0, signed: 0 })
  expect(result[1]).toEqual({ note_index: 1, drift_ms: 50, signed: 50 })
  expect(result[2]).toEqual({ note_index: 2, drift_ms: 50, signed: -50 })
})
