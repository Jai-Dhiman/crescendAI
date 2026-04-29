import { test, expect } from 'vitest'
import { articulationClarityCheck } from './articulation-clarity-check'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('articulationClarityCheck: staccato score with legato execution (mismatch=8/8) and z=-1.75 returns issue/moderate', async () => {
  const barsData = Array.from({ length: 8 }, (_, barIdx) => ({
    bar: 5 + barIdx,
    notes: [
      { onset_ms: barIdx * 1000,       duration_ms: 450 },
      { onset_ms: barIdx * 1000 + 333, duration_ms: 450 },
      { onset_ms: barIdx * 1000 + 666, duration_ms: 450 },
    ],
  }))
  const input = {
    bar_range: [5, 12] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c2', 'cache:amt:s1:c2'],
    muq_scores: [0.54, 0.48, 0.46, 0.38, 0.52, 0.51],
    mono_notes_per_bar: barsData,
    score_articulation_per_bar: barsData.map(b => ({ bar: b.bar, articulation: 'staccato' as const })),
    cohort_table_articulation: [{ p: 50, value: 0.52 }, { p: 84, value: 0.60 }],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await articulationClarityCheck.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('articulation')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('moderate')
})

test('articulationClarityCheck: z above neutral threshold returns neutral', async () => {
  const input = {
    bar_range: [5, 12] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c2'],
    muq_scores: [0.54, 0.48, 0.46, 0.52, 0.52, 0.51],
    mono_notes_per_bar: [],
    score_articulation_per_bar: [],
    cohort_table_articulation: [{ p: 50, value: 0.52 }, { p: 84, value: 0.60 }],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await articulationClarityCheck.invoke(input) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})
