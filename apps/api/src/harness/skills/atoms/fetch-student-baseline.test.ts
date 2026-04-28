import { test, expect } from 'vitest'
import { fetchStudentBaseline } from './fetch-student-baseline'

test('fetchStudentBaseline: eight sessions with mean 0.65 and stddev 0.10 match spec example', async () => {
  // [0.55, 0.75, 0.55, 0.75, 0.55, 0.75, 0.55, 0.75] → mean=0.65, stddev=0.10
  const session_means = [0.55, 0.75, 0.55, 0.75, 0.55, 0.75, 0.55, 0.75]
  const result = await fetchStudentBaseline.invoke({ dimension: 'pedaling', session_means }) as { dimension: string; mean: number; stddev: number; n_sessions: number } | null
  expect(result).not.toBeNull()
  expect(result!.dimension).toBe('pedaling')
  expect(result!.mean).toBeCloseTo(0.65, 5)
  expect(result!.stddev).toBeCloseTo(0.10, 5)
  expect(result!.n_sessions).toBe(8)
})

test('fetchStudentBaseline: fewer than 3 sessions returns null', async () => {
  const result = await fetchStudentBaseline.invoke({ dimension: 'dynamics', session_means: [0.5, 0.6] })
  expect(result).toBeNull()
})
