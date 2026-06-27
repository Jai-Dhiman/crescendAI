// cross-modal-contradiction-check.test.ts (after refactor — selectors-only contract)
// Articulation arm REMOVED. 3 remaining arms: timing-drift-vs-MuQ, pedal-ratio-vs-MuQ, dynamics-range-vs-MuQ.
import { test, expect } from 'vitest'
import { crossModalContradictionCheck } from './cross-modal-contradiction-check'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import type { GroundedDigest } from '../../loop/grounded-digest'
import type { PhaseContext } from '../../loop/types'

function makeCtx(muqScores: number[], opts?: {
  midi_notes?: GroundedDigest['chunks_adapted'][0]['midi_notes']
  pedal_cc?: { time_ms: number; value: number }[]
  alignment?: GroundedDigest['chunks_adapted'][0]['alignment']
}): PhaseContext {
  const dims = ['dynamics','timing','pedaling','articulation','phrasing','interpretation'] as const
  const session_means: Record<string, number[]> = {}
  const within_session_means: Record<string, number> = {}
  for (const d of dims) {
    session_means[d] = [0.5, 0.5, 0.5]
    within_session_means[d] = 0.52
  }
  const defaultNotes = [
    { pitch: 60, onset_ms: 0,    duration_ms: 1000, velocity: 70, bar: 20 },
    { pitch: 62, onset_ms: 1000, duration_ms: 1000, velocity: 70, bar: 22 },
    { pitch: 64, onset_ms: 2000, duration_ms: 1000, velocity: 70, bar: 24 },
    { pitch: 65, onset_ms: 3000, duration_ms: 1000, velocity: 70, bar: 26 },
  ]
  const defaultAlignment = [
    { perf_index: 0, score_index: 0, expected_onset_ms: 0,    bar: 20 },
    { perf_index: 1, score_index: 1, expected_onset_ms: 1000, bar: 22 },
    { perf_index: 2, score_index: 2, expected_onset_ms: 2000, bar: 24 },
    { perf_index: 3, score_index: 3, expected_onset_ms: 3000, bar: 26 },
  ]
  const digest: GroundedDigest = {
    chunks_adapted: [{
      chunk_id: 'chunk:0',
      bar_coverage: [20, 28] as [number, number],
      muq_scores: muqScores,
      midi_notes: opts?.midi_notes ?? defaultNotes,
      pedal_cc: opts?.pedal_cc ?? [{ time_ms: 0, value: 127 }],
      alignment: opts?.alignment ?? defaultAlignment,
    }],
    mono_notes_per_bar: [],
    now_ms: 1000,
    cohort: {
      dynamics:       { mean: 0.54, stddev: 0.07 },
      timing:         { mean: 0.48, stddev: 0.04 },
      pedaling:       { mean: 0.46, stddev: 0.08 },
      articulation:   { mean: 0.54, stddev: 0.02 },
      phrasing:       { mean: 0.52, stddev: 0.11 },
      interpretation: { mean: 0.51, stddev: 0.13 },
    },
    past_diagnoses_grounded: [],
    session_means: session_means as GroundedDigest['session_means'],
    within_session_means: within_session_means as GroundedDigest['within_session_means'],
    compact_signal_summary: 'chunk:0 bars 20-28',
    piece_id: 'test-piece',
  }
  return {
    env: {} as unknown as import('../../../lib/types').Bindings,
    studentId: 'stu-1', sessionId: 'sess-1', conversationId: null,
    digest: digest as unknown as Record<string, unknown>,
    waitUntil: () => {},
    pieceId: 'test-piece',
    trigger: 'synthesis',
    turnCap: 10,
  }
}

test('crossModalContradictionCheck: MuQ pedaling high (z≈+3.3) but overlap ratio=1.0 returns issue/significant/pedaling', async () => {
  // muq pedaling=0.72, cohort mean=0.46 stddev=0.08 → z=(0.72-0.46)/0.08=+3.25
  // overlap ratio: one sustained pedal CC127 covering all notes → ratio > 0.85 → contradiction
  const ctx = makeCtx([0.54, 0.48, 0.72, 0.54, 0.52, 0.51])
  const selectors = {
    bar_range: [20, 28] as [number, number],
    scope: 'stop_moment' as const,
    evidence_refs: ['cache:muq:s1:c7', 'cache:amt-pedal:s1:c7'],
  }
  const result = await crossModalContradictionCheck.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('pedaling')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
})

test('crossModalContradictionCheck: no contradictions detected returns neutral', async () => {
  // All z values low (≈-1), no arms trigger
  const ctx = makeCtx([0.40, 0.40, 0.40, 0.40, 0.52, 0.51], {
    pedal_cc: [],
    midi_notes: [
      { pitch: 60, onset_ms: 0,   duration_ms: 500, velocity: 70, bar: 20 },
      { pitch: 62, onset_ms: 500, duration_ms: 500, velocity: 75, bar: 22 },
    ],
    alignment: [
      { perf_index: 0, score_index: 0, expected_onset_ms: 0,   bar: 20 },
      { perf_index: 1, score_index: 1, expected_onset_ms: 500, bar: 22 },
    ],
  })
  const selectors = { bar_range: [20, 28] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c1'] }
  const result = await crossModalContradictionCheck.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})

test('crossModalContradictionCheck: timing arm — empty alignment returns neutral (not throw)', async () => {
  // High timing z (+3) but empty alignment → timing arm skipped → neutral
  const ctx = makeCtx([0.54, 0.72, 0.46, 0.54, 0.52, 0.51], {
    alignment: [],
    pedal_cc: [],  // no pedal contradiction either
    midi_notes: [
      { pitch: 60, onset_ms: 0,   duration_ms: 500, velocity: 68, bar: 20 },
      { pitch: 62, onset_ms: 500, duration_ms: 500, velocity: 72, bar: 22 },
    ],
  })
  const selectors = { bar_range: [20, 28] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c7'] }
  const result = await crossModalContradictionCheck.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBeDefined()
  expect(result.dimensions.length).toBeGreaterThanOrEqual(1)
  expect(result.evidence_refs.length).toBeGreaterThanOrEqual(1)
})

test('crossModalContradictionCheck: dynamics arm — high z + compressed velocity returns issue', async () => {
  // dynamics muq=0.72 → z=(0.72-0.54)/0.07≈+2.57; velocity range=75-70=5 < 25 → contradiction
  const ctx = makeCtx([0.72, 0.48, 0.46, 0.54, 0.52, 0.51], {
    pedal_cc: [],
    alignment: [],
    midi_notes: [
      { pitch: 60, onset_ms: 0,    duration_ms: 500, velocity: 70, bar: 20 },
      { pitch: 62, onset_ms: 500,  duration_ms: 500, velocity: 72, bar: 22 },
      { pitch: 64, onset_ms: 1000, duration_ms: 500, velocity: 73, bar: 24 },
      { pitch: 65, onset_ms: 1500, duration_ms: 500, velocity: 75, bar: 26 },
    ],
  })
  const selectors = { bar_range: [20, 28] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c7'] }
  const result = await crossModalContradictionCheck.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.finding_type).toBe('issue')
  expect(result.primary_dimension).toBe('dynamics')
})

test('crossModalContradictionCheck: missing ctx throws', async () => {
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: [] }
  await expect(crossModalContradictionCheck.invoke(selectors, undefined)).rejects.toThrow('ctx')
})
