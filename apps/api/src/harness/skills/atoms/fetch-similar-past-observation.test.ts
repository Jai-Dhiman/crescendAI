import { test, expect } from 'vitest'
import { fetchSimilarPastObservation } from './fetch-similar-past-observation'

test('fetchSimilarPastObservation: matching piece_id and same bar_range returns similarity 1.0', async () => {
  const now_ms = 1_714_003_200_000
  const past_diagnoses = [
    {
      artifact_id: 'diag:abc789',
      session_id: 'sess_31',
      created_at: now_ms - 5 * 86_400_000, // 5 days ago
      primary_dimension: 'pedaling',
      piece_id: 'chopin_op23',
      bar_range: [12, 16] as [number, number],
    },
    {
      artifact_id: 'diag:xyz000',
      session_id: 'sess_10',
      created_at: now_ms - 30 * 86_400_000, // 30 days ago
      primary_dimension: 'pedaling',
      piece_id: 'beethoven_op27', // different piece
      bar_range: [20, 24] as [number, number], // different bars
    },
  ]
  const result = await fetchSimilarPastObservation.invoke({
    dimension: 'pedaling',
    piece_id: 'chopin_op23',
    bar_range: [12, 16],
    past_diagnoses,
    now_ms,
  }) as { artifact_id: string; days_ago: number; similarity_score: number } | null
  expect(result).not.toBeNull()
  expect(result!.artifact_id).toBe('diag:abc789')
  expect(result!.similarity_score).toBe(1.0) // 0.5 * piece_match(1) + 0.5 * bar_overlap(1)
  expect(result!.days_ago).toBeCloseTo(5, 0)
})
