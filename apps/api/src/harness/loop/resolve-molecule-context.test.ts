// resolve-molecule-context.test.ts
import { test, expect } from 'vitest'
import { resolveMoleculeContext } from './resolve-molecule-context'
import type { GroundedDigest } from './grounded-digest'

function makeDigest(sessionMeansCounts: Record<string, number>): GroundedDigest {
  const dims = ['dynamics', 'timing', 'pedaling', 'articulation', 'phrasing', 'interpretation'] as const
  const session_means: Record<string, number[]> = {}
  for (const d of dims) {
    const n = sessionMeansCounts[d] ?? 0
    session_means[d] = Array.from({ length: n }, () => 0.5)
  }
  const within_session_means: Record<string, number> = {}
  for (const d of dims) within_session_means[d] = 0.52
  return {
    chunks_adapted: [
      {
        chunk_id: 'chunk:0',
        bar_coverage: [1, 8] as [number, number],
        muq_scores: [0.55, 0.48, 0.46, 0.54, 0.52, 0.51],
        midi_notes: [
          { pitch: 60, onset_ms: 0, duration_ms: 500, velocity: 70, bar: 2 },
          { pitch: 62, onset_ms: 500, duration_ms: 500, velocity: 70, bar: 3 },
        ],
        pedal_cc: [],
        alignment: [
          { perf_index: 0, score_index: 0, expected_onset_ms: 0, bar: 2 },
          { perf_index: 1, score_index: 1, expected_onset_ms: 500, bar: 3 },
        ],
      },
    ],
    mono_notes_per_bar: [],
    now_ms: 5000,
    cohort: {
      dynamics:       { mean: 0.55, stddev: 0.10 },
      timing:         { mean: 0.48, stddev: 0.12 },
      pedaling:       { mean: 0.46, stddev: 0.12 },
      articulation:   { mean: 0.54, stddev: 0.11 },
      phrasing:       { mean: 0.52, stddev: 0.11 },
      interpretation: { mean: 0.51, stddev: 0.13 },
    },
    past_diagnoses_grounded: [],
    session_means: session_means as GroundedDigest['session_means'],
    within_session_means: within_session_means as GroundedDigest['within_session_means'],
    compact_signal_summary: 'chunk:0 bars 1-8 muq=[0.55,0.48,0.46,0.54,0.52,0.51]',
    piece_id: 'test-piece',
  }
}

test('resolveMoleculeContext: bar_range=[2,3] filters midi_notes to bars 2-3 only', async () => {
  const ctx = await resolveMoleculeContext(makeDigest({ dynamics: 3 }), [2, 3])
  expect(ctx.bundle.midi_notes.every((n) => n.bar !== undefined && n.bar >= 2 && n.bar <= 3)).toBe(true)
})

test('resolveMoleculeContext: bar_range=null returns all notes (full-session bundle)', async () => {
  const ctx = await resolveMoleculeContext(makeDigest({ dynamics: 3 }), null)
  expect(ctx.bundle.midi_notes.length).toBe(2)
})

test('resolveMoleculeContext: session_means length>=3 produces multi-session baseline', async () => {
  const ctx = await resolveMoleculeContext(makeDigest({ dynamics: 3, timing: 3 }), [2, 3])
  // fetchStudentBaseline returns non-null when n>=3
  expect(ctx.baseline.dynamics.n_sessions).toBeGreaterThanOrEqual(3)
})

test('resolveMoleculeContext: session_means length<3 synthesises within-session baseline (not null)', async () => {
  const ctx = await resolveMoleculeContext(makeDigest({ dynamics: 1 }), [2, 3])
  // Must not be null; falls back to within_session_means
  expect(ctx.baseline.dynamics).toBeDefined()
  expect(ctx.baseline.dynamics.n_sessions).toBe(1)
  expect(ctx.baseline.dynamics.mean).toBeCloseTo(0.52)
})

test('resolveMoleculeContext: cohort, past_diagnoses, piece_id, now_ms forwarded from digest', async () => {
  const ctx = await resolveMoleculeContext(makeDigest({}), null)
  expect(ctx.cohort.dynamics.mean).toBeCloseTo(0.55)
  expect(ctx.past_diagnoses).toEqual([])
  expect(ctx.piece_id).toBe('test-piece')
  expect(ctx.now_ms).toBe(5000)
})
