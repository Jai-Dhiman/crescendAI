// grounded-digest.test.ts
import { test, expect } from 'vitest'
import { buildGroundedDigest } from './grounded-digest'
import type { SynthesisInput } from '../../services/teacher'

// NOTE: fixture matches production COHORT_TABLES shape exactly — p25/p50/p75/p90 only.
// p84 is NOT present in production; stddev uses (p75 - p50) per Blocker 2 resolution.
const COHORT_TABLES: Record<string, { p: number; value: number }[]> = {
  dynamics:       [{ p: 25, value: 0.38 }, { p: 50, value: 0.55 }, { p: 75, value: 0.70 }, { p: 90, value: 0.82 }],
  timing:         [{ p: 25, value: 0.34 }, { p: 50, value: 0.48 }, { p: 75, value: 0.63 }, { p: 90, value: 0.77 }],
  pedaling:       [{ p: 25, value: 0.32 }, { p: 50, value: 0.46 }, { p: 75, value: 0.61 }, { p: 90, value: 0.75 }],
  articulation:   [{ p: 25, value: 0.37 }, { p: 50, value: 0.54 }, { p: 75, value: 0.68 }, { p: 90, value: 0.80 }],
  phrasing:       [{ p: 25, value: 0.36 }, { p: 50, value: 0.52 }, { p: 75, value: 0.66 }, { p: 90, value: 0.79 }],
  interpretation: [{ p: 25, value: 0.35 }, { p: 50, value: 0.51 }, { p: 75, value: 0.65 }, { p: 90, value: 0.78 }],
}

function makeInput(): SynthesisInput {
  return {
    studentId: 'stu-1',
    conversationId: null,
    sessionDurationMs: 30000,
    practicePattern: 'test',
    topMoments: [],
    drillingRecords: [],
    pieceMetadata: null,
    enrichedChunks: [
      {
        chunkIndex: 0,
        muq_scores: [0.55, 0.48, 0.46, 0.54, 0.52, 0.51],
        midi_notes: [
          { pitch: 60, onset_ms: 0,    duration_ms: 500, velocity: 70 },
          { pitch: 62, onset_ms: 500,  duration_ms: 500, velocity: 70 },
        ],
        pedal_cc: [{ time_ms: 100, value: 127 }],
        // alignment is always [] in production (barMapAlignments never populated).
        // bar injection uses bar_coverage[0] as a chunk-level approximation.
        alignment: [],
        bar_coverage: [1, 4],
      },
    ],
    baselines: null,
    sessionHistory: [],
    pastDiagnoses: [
      // id and pieceId are added to PastDiagnosisRecord in C3/C4; cast for now.
      {
        id: 'diag-uuid-1',
        sessionId: 'sess-prior',
        primaryDimension: 'pedaling',
        barRangeStart: 1, barRangeEnd: 4,
        artifactJson: {},
        createdAt: new Date(0).toISOString(),
        pieceId: 'piece-1',
      } as import('../../services/teacher').PastDiagnosisRecord,
    ],
    pieceId: 'piece-1',
    referenceMode: null,
  }
}

test('buildGroundedDigest: bar assigned from chunk bar_coverage[0] (chunk-level approximation)', async () => {
  // alignment is always [] in production; bar is derived from the chunk's bar_coverage.
  // Every note in a chunk gets bar = bar_coverage[0].
  const mockDb = {
    select: () => ({ from: () => ({ where: () => ({ groupBy: () => ({ orderBy: () => ({ limit: () => Promise.resolve([]) }) }) }) }) }),
  } as unknown as import('../../db').Db
  const digest = await buildGroundedDigest(makeInput(), { db: mockDb, studentId: 'stu-1' }, COHORT_TABLES)
  const note = digest.chunks_adapted[0].midi_notes[0]
  // bar_coverage = [1, 4] → every note in this chunk gets bar = 1
  expect(note.bar).toBe(1)
  const note2 = digest.chunks_adapted[0].midi_notes[1]
  expect(note2.bar).toBe(1)
})

test('buildGroundedDigest: chunk_id is chunk:0 for chunkIndex 0', async () => {
  const mockDb = {
    select: () => ({ from: () => ({ where: () => ({ groupBy: () => ({ orderBy: () => ({ limit: () => Promise.resolve([]) }) }) }) }) }),
  } as unknown as import('../../db').Db
  const digest = await buildGroundedDigest(makeInput(), { db: mockDb, studentId: 'stu-1' }, COHORT_TABLES)
  expect(digest.chunks_adapted[0].chunk_id).toBe('chunk:0')
})

test('buildGroundedDigest: cohort mean=p50, stddev=max(0.01,p75-p50)', async () => {
  // Production COHORT_TABLES has p25/p50/p75/p90 only — no p84.
  // stddev proxy = max(0.01, p75 - p50). For dynamics: p75=0.70, p50=0.55 => 0.15.
  const mockDb = {
    select: () => ({ from: () => ({ where: () => ({ groupBy: () => ({ orderBy: () => ({ limit: () => Promise.resolve([]) }) }) }) }) }),
  } as unknown as import('../../db').Db
  const digest = await buildGroundedDigest(makeInput(), { db: mockDb, studentId: 'stu-1' }, COHORT_TABLES)
  expect(digest.cohort.dynamics.mean).toBeCloseTo(0.55)
  expect(digest.cohort.dynamics.stddev).toBeCloseTo(0.15) // p75=0.70, p50=0.55 => 0.15
})

test('buildGroundedDigest: within_session_means computed from chunk muq_scores', async () => {
  const mockDb = {
    select: () => ({ from: () => ({ where: () => ({ groupBy: () => ({ orderBy: () => ({ limit: () => Promise.resolve([]) }) }) }) }) }),
  } as unknown as import('../../db').Db
  const digest = await buildGroundedDigest(makeInput(), { db: mockDb, studentId: 'stu-1' }, COHORT_TABLES)
  // single chunk, dim 0 (dynamics) = 0.55
  expect(digest.within_session_means.dynamics).toBeCloseTo(0.55)
})

test('buildGroundedDigest: past_diagnoses_grounded reshapes id+pieceId', async () => {
  const mockDb = {
    select: () => ({ from: () => ({ where: () => ({ groupBy: () => ({ orderBy: () => ({ limit: () => Promise.resolve([]) }) }) }) }) }),
  } as unknown as import('../../db').Db
  const digest = await buildGroundedDigest(makeInput(), { db: mockDb, studentId: 'stu-1' }, COHORT_TABLES)
  expect(digest.past_diagnoses_grounded[0].artifact_id).toBe('diag-uuid-1')
  expect(digest.past_diagnoses_grounded[0].piece_id).toBe('piece-1')
})

test('buildGroundedDigest: compact_signal_summary is a short non-empty string', async () => {
  const mockDb = {
    select: () => ({ from: () => ({ where: () => ({ groupBy: () => ({ orderBy: () => ({ limit: () => Promise.resolve([]) }) }) }) }) }),
  } as unknown as import('../../db').Db
  const digest = await buildGroundedDigest(makeInput(), { db: mockDb, studentId: 'stu-1' }, COHORT_TABLES)
  expect(typeof digest.compact_signal_summary).toBe('string')
  expect(digest.compact_signal_summary.length).toBeGreaterThan(10)
})

test('buildGroundedDigest: DB query failure throws (not silently ignored)', async () => {
  const badDb = {
    select: () => { throw new Error('db connection refused') },
  } as unknown as import('../../db').Db
  await expect(buildGroundedDigest(makeInput(), { db: badDb, studentId: 'stu-1' }, COHORT_TABLES))
    .rejects.toThrow('db connection refused')
})
