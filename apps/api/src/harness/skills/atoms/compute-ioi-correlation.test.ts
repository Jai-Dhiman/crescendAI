import { test, expect } from 'vitest'
import { computeIoiCorrelation } from './compute-ioi-correlation'

test('computeIoiCorrelation: rubato performance against rigid score returns r near 0', async () => {
  // 5 notes: perf IOIs [400,410,390,420], score IOIs [400,400,400,400]
  // Pearson r between [400,410,390,420] and [400,400,400,400]:
  // score IOIs are constant -> zero variance -> r = 0 by definition (or near 0 numerically)
  const notes = [
    { onset_ms: 0, expected_onset_ms: 0 },
    { onset_ms: 400, expected_onset_ms: 400 },
    { onset_ms: 810, expected_onset_ms: 800 },
    { onset_ms: 1200, expected_onset_ms: 1200 },
    { onset_ms: 1620, expected_onset_ms: 1600 },
  ]
  const result = await computeIoiCorrelation.invoke({ notes })
  expect(result).not.toBeNull()
  expect(Math.abs(result as number)).toBeLessThan(0.1)
})
