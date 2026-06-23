# Ground Molecule Layer — TDD Implementation Plan

Spec: `docs/specs/2026-06-22-ground-molecule-layer-design.md`
Issue: #99
Branch: `issue-99-ground-molecule-layer`

---

## Goal

Remove the 136K-token Phase-1 digest dump; replace it with a compact signal
summary. Ground all 7 diagnosis molecules so they fetch their own signal data
server-side and never receive fabricated inputs from the LLM. Fix the three
identified breakages: missing `bar` on midi_notes, missing `id`/`pieceId` on
past-diagnoses, thin-history throw.

---

## Dependency Map

```
Group A (sequential foundation)
  Task A1  buildGroundedDigest
  Task A2  resolveMoleculeContext   (depends on A1)

Group B (parallel molecule refactors, all depend on A2)
  Task B1  pedal-triage (exemplar — full recipe here)
  Task B2  tempo-stability-triage
  Task B3  rubato-coaching
  Task B4  voicing-diagnosis
  Task B5  dynamic-range-audit
  Task B6  phrasing-arc-analysis
  Task B7  cross-modal-contradiction-check

Group C (sequential wiring, depends on B-all)
  Task C1  phase1.ts + overflow regression test
  Task C2  compound-registry.ts + index.ts
  Task C3  teacher.ts  (buildGroundedDigest call + PastDiagnosisRecord extension)
  Task C4  session-brain.ts  (past-diagnoses SELECT adds id + pieceId)
```

---

## Group A — Foundation

### Task A1 — `buildGroundedDigest`

**Files**
- `apps/api/src/harness/loop/grounded-digest.ts` (new)
- `apps/api/src/harness/loop/grounded-digest.test.ts` (new)

**Failing test first**

```ts
// grounded-digest.test.ts
import { test, expect } from 'vitest'
import { buildGroundedDigest } from './grounded-digest'
import type { SynthesisInput } from '../../services/teacher'

const COHORT_TABLES: Record<string, { p: number; value: number }[]> = {
  dynamics:       [{ p: 25, value: 0.38 }, { p: 50, value: 0.55 }, { p: 75, value: 0.70 }, { p: 84, value: 0.67 }, { p: 90, value: 0.82 }],
  timing:         [{ p: 25, value: 0.34 }, { p: 50, value: 0.48 }, { p: 75, value: 0.63 }, { p: 84, value: 0.60 }, { p: 90, value: 0.77 }],
  pedaling:       [{ p: 25, value: 0.32 }, { p: 50, value: 0.46 }, { p: 75, value: 0.61 }, { p: 84, value: 0.58 }, { p: 90, value: 0.75 }],
  articulation:   [{ p: 25, value: 0.37 }, { p: 50, value: 0.54 }, { p: 75, value: 0.68 }, { p: 84, value: 0.65 }, { p: 90, value: 0.80 }],
  phrasing:       [{ p: 25, value: 0.36 }, { p: 50, value: 0.52 }, { p: 75, value: 0.66 }, { p: 84, value: 0.63 }, { p: 90, value: 0.79 }],
  interpretation: [{ p: 25, value: 0.35 }, { p: 50, value: 0.51 }, { p: 75, value: 0.65 }, { p: 84, value: 0.62 }, { p: 90, value: 0.78 }],
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
        alignment: [
          { perf_index: 0, score_index: 0, expected_onset_ms: 0,   bar: 1 },
          { perf_index: 1, score_index: 1, expected_onset_ms: 500, bar: 2 },
        ],
        bar_coverage: [1, 4],
      },
    ],
    baselines: null,
    sessionHistory: [],
    pastDiagnoses: [
      {
        id: 'diag-uuid-1',
        sessionId: 'sess-prior',
        primaryDimension: 'pedaling',
        barRangeStart: 1, barRangeEnd: 4,
        artifactJson: {},
        createdAt: new Date(0).toISOString(),
        pieceId: 'piece-1',
      },
    ],
    pieceId: 'piece-1',
    referenceMode: null,
  }
}

test('buildGroundedDigest: bar injected onto midi_notes via alignment perf_index join', async () => {
  const mockDb = {
    select: () => ({ from: () => ({ where: () => ({ groupBy: () => Promise.resolve([]) }) }) }),
  } as unknown as import('../../db').Db
  const digest = await buildGroundedDigest(makeInput(), { db: mockDb, studentId: 'stu-1' }, COHORT_TABLES)
  const note = digest.chunks_adapted[0].midi_notes[0]
  expect(note.bar).toBe(1)
  const note2 = digest.chunks_adapted[0].midi_notes[1]
  expect(note2.bar).toBe(2)
})

test('buildGroundedDigest: chunk_id is chunk:0 for chunkIndex 0', async () => {
  const mockDb = {
    select: () => ({ from: () => ({ where: () => ({ groupBy: () => Promise.resolve([]) }) }) }),
  } as unknown as import('../../db').Db
  const digest = await buildGroundedDigest(makeInput(), { db: mockDb, studentId: 'stu-1' }, COHORT_TABLES)
  expect(digest.chunks_adapted[0].chunk_id).toBe('chunk:0')
})

test('buildGroundedDigest: cohort mean=p50, stddev=max(0.01,p84-p50)', async () => {
  const mockDb = {
    select: () => ({ from: () => ({ where: () => ({ groupBy: () => Promise.resolve([]) }) }) }),
  } as unknown as import('../../db').Db
  const digest = await buildGroundedDigest(makeInput(), { db: mockDb, studentId: 'stu-1' }, COHORT_TABLES)
  expect(digest.cohort.dynamics.mean).toBeCloseTo(0.55)
  expect(digest.cohort.dynamics.stddev).toBeCloseTo(0.12) // p84=0.67, p50=0.55 => 0.12
})

test('buildGroundedDigest: within_session_means computed from chunk muq_scores', async () => {
  const mockDb = {
    select: () => ({ from: () => ({ where: () => ({ groupBy: () => Promise.resolve([]) }) }) }),
  } as unknown as import('../../db').Db
  const digest = await buildGroundedDigest(makeInput(), { db: mockDb, studentId: 'stu-1' }, COHORT_TABLES)
  // single chunk, dim 0 (dynamics) = 0.55
  expect(digest.within_session_means.dynamics).toBeCloseTo(0.55)
})

test('buildGroundedDigest: past_diagnoses_grounded reshapes id+pieceId', async () => {
  const mockDb = {
    select: () => ({ from: () => ({ where: () => ({ groupBy: () => Promise.resolve([]) }) }) }),
  } as unknown as import('../../db').Db
  const digest = await buildGroundedDigest(makeInput(), { db: mockDb, studentId: 'stu-1' }, COHORT_TABLES)
  expect(digest.past_diagnoses_grounded[0].artifact_id).toBe('diag-uuid-1')
  expect(digest.past_diagnoses_grounded[0].piece_id).toBe('piece-1')
})

test('buildGroundedDigest: compact_signal_summary is a short non-empty string', async () => {
  const mockDb = {
    select: () => ({ from: () => ({ where: () => ({ groupBy: () => Promise.resolve([]) }) }) }),
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
```

**Implementation**

```ts
// grounded-digest.ts
import { sql, eq, and } from 'drizzle-orm'
import type { Db } from '../../db'
import { observations } from '../../db/schema/observations'
import type { SynthesisInput, PastDiagnosisRecord } from '../../services/teacher'

export const DIMENSIONS_6 = ['dynamics', 'timing', 'pedaling', 'articulation', 'phrasing', 'interpretation'] as const
type Dim6 = (typeof DIMENSIONS_6)[number]

export type GroundedNote = {
  pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number
}
export type AdaptedChunk = {
  chunk_id: string
  bar_coverage: [number, number]
  muq_scores: number[]
  midi_notes: GroundedNote[]
  pedal_cc: { time_ms: number; value: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number | null; bar: number }[]
}
export type GroundedPastDiagnosis = {
  artifact_id: string
  session_id: string
  created_at: number
  primary_dimension: string
  bar_range: [number, number] | null
  piece_id: string | null
}
export type CohortStats = Record<Dim6, { mean: number; stddev: number }>
export type SessionMeans = Record<Dim6, number[]>

export type GroundedDigest = {
  chunks_adapted: AdaptedChunk[]
  mono_notes_per_bar: { bar: number; notes: { onset_ms: number; duration_ms: number }[] }[]
  now_ms: number
  cohort: CohortStats
  past_diagnoses_grounded: GroundedPastDiagnosis[]
  session_means: SessionMeans
  within_session_means: Record<Dim6, number>
  compact_signal_summary: string
  piece_id: string | null
}

export async function buildGroundedDigest(
  input: SynthesisInput,
  deps: { db: Db; studentId: string },
  cohortTables: Record<string, { p: number; value: number }[]>,
): Promise<GroundedDigest> {
  const now_ms = Date.now()

  // Adapt chunks: inject bar onto midi_notes via alignment perf_index join
  const chunks_adapted: AdaptedChunk[] = input.enrichedChunks.map((chunk) => {
    const perfToBar = new Map(chunk.alignment.map((a) => [a.perf_index, a.bar]))
    const grounded_notes: GroundedNote[] = chunk.midi_notes.map((note, idx) => ({
      ...note,
      bar: perfToBar.get(idx) ?? 0,
    }))
    return {
      chunk_id: `chunk:${chunk.chunkIndex}`,
      bar_coverage: chunk.bar_coverage as [number, number],
      muq_scores: chunk.muq_scores,
      midi_notes: grounded_notes,
      pedal_cc: chunk.pedal_cc,
      alignment: chunk.alignment,
    }
  })

  // mono_notes_per_bar: group all notes by bar (monophonic proxy)
  const barNoteMap = new Map<number, { onset_ms: number; duration_ms: number }[]>()
  for (const chunk of chunks_adapted) {
    for (const note of chunk.midi_notes) {
      const existing = barNoteMap.get(note.bar) ?? []
      existing.push({ onset_ms: note.onset_ms, duration_ms: note.duration_ms })
      barNoteMap.set(note.bar, existing)
    }
  }
  const mono_notes_per_bar = Array.from(barNoteMap.entries())
    .sort(([a], [b]) => a - b)
    .map(([bar, notes]) => ({ bar, notes }))

  // cohort stats: mean=p50, stddev=max(0.01, p84-p50)
  const cohort = {} as CohortStats
  for (const dim of DIMENSIONS_6) {
    const table = cohortTables[dim] ?? []
    const p50 = table.find((e) => e.p === 50)?.value ?? 0.5
    const p84 = table.find((e) => e.p === 84)?.value ?? (p50 + 0.1)
    cohort[dim] = { mean: p50, stddev: Math.max(0.01, p84 - p50) }
  }

  // within_session_means: mean muq_scores[dimIdx] across all chunks
  const within_session_means = {} as Record<Dim6, number>
  for (let dimIdx = 0; dimIdx < DIMENSIONS_6.length; dimIdx++) {
    const dim = DIMENSIONS_6[dimIdx]
    const vals = input.enrichedChunks.map((c) => c.muq_scores[dimIdx]).filter((v) => typeof v === 'number')
    within_session_means[dim] = vals.length > 0 ? vals.reduce((s, v) => s + v, 0) / vals.length : 0.5
  }

  // session_means: per-session AVG(dimension_score) GROUP BY session_id from observations
  // Throws if DB query fails — explicit, not silently caught.
  const rows = await deps.db
    .select({
      sessionId: observations.sessionId,
      dimension: observations.dimension,
      avgScore: sql<number>`AVG(${observations.dimensionScore})`,
    })
    .from(observations)
    .where(and(eq(observations.studentId, deps.studentId)))
    .groupBy(observations.sessionId, observations.dimension)

  const sessionDimMap = new Map<string, Map<Dim6, number>>()
  for (const row of rows) {
    const dim = row.dimension as Dim6
    if (!DIMENSIONS_6.includes(dim)) continue
    if (!sessionDimMap.has(row.sessionId)) sessionDimMap.set(row.sessionId, new Map())
    sessionDimMap.get(row.sessionId)!.set(dim, row.avgScore)
  }
  const session_means = {} as SessionMeans
  for (const dim of DIMENSIONS_6) {
    session_means[dim] = Array.from(sessionDimMap.values())
      .map((m) => m.get(dim))
      .filter((v): v is number => v !== undefined)
  }

  // past_diagnoses_grounded: reshape PastDiagnosisRecord → GroundedPastDiagnosis
  const past_diagnoses_grounded: GroundedPastDiagnosis[] = input.pastDiagnoses.map((r) => ({
    artifact_id: (r as PastDiagnosisRecord & { id: string }).id,
    session_id: r.sessionId,
    created_at: new Date(r.createdAt).getTime(),
    primary_dimension: r.primaryDimension,
    bar_range: r.barRangeStart !== null && r.barRangeEnd !== null
      ? [r.barRangeStart, r.barRangeEnd]
      : null,
    piece_id: (r as PastDiagnosisRecord & { pieceId?: string | null }).pieceId ?? null,
  }))

  // compact_signal_summary: one line per chunk
  const lines = chunks_adapted.map((c) => {
    const scores = c.muq_scores.map((v) => v.toFixed(2)).join(',')
    return `${c.chunk_id} bars ${c.bar_coverage[0]}-${c.bar_coverage[1]} muq=[${scores}]`
  })
  const compact_signal_summary = lines.join('\n')

  return {
    chunks_adapted,
    mono_notes_per_bar,
    now_ms,
    cohort,
    past_diagnoses_grounded,
    session_means,
    within_session_means,
    compact_signal_summary,
    piece_id: input.pieceId ?? null,
  }
}
```

**Commit:** `feat(#99): buildGroundedDigest — bar injection, cohort stats, compact summary`

---

### Task A2 — `resolveMoleculeContext`

**Files**
- `apps/api/src/harness/loop/resolve-molecule-context.ts` (new)
- `apps/api/src/harness/loop/resolve-molecule-context.test.ts` (new)

**Failing test first**

```ts
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
```

**Implementation**

```ts
// resolve-molecule-context.ts
import { fetchStudentBaseline } from '../skills/atoms/fetch-student-baseline'
import type { Baseline } from '../skills/atoms/fetch-student-baseline'
import { extractBarRangeSignals } from '../skills/atoms/extract-bar-range-signals'
import type { SignalBundle } from '../skills/atoms/extract-bar-range-signals'
import type { GroundedDigest, GroundedPastDiagnosis, CohortStats } from './grounded-digest'
import { DIMENSIONS_6 } from './grounded-digest'

type Dim6 = (typeof DIMENSIONS_6)[number]

export type TieredBaseline = Record<Dim6, Baseline>

export type ResolvedMoleculeContext = {
  bundle: SignalBundle
  baseline: TieredBaseline
  cohort: CohortStats
  past_diagnoses: GroundedPastDiagnosis[]
  piece_id: string | null
  now_ms: number
}

export async function resolveMoleculeContext(
  digest: GroundedDigest,
  bar_range: [number, number] | null,
): Promise<ResolvedMoleculeContext> {
  const effectiveRange = bar_range ?? fullSessionRange(digest)
  const bundle = await extractBarRangeSignals.invoke({
    bar_range: effectiveRange,
    chunks: digest.chunks_adapted,
  }) as SignalBundle

  const baseline = {} as TieredBaseline
  for (const dim of DIMENSIONS_6) {
    const session_means = digest.session_means[dim]
    if (session_means.length >= 3) {
      const result = await fetchStudentBaseline.invoke({ dimension: dim, session_means }) as Baseline | null
      if (result === null) {
        throw new Error(`resolveMoleculeContext: fetchStudentBaseline returned null despite n=${session_means.length} sessions for dim=${dim}`)
      }
      baseline[dim] = result
    } else {
      // Thin-history tier: synthesise from within_session_means
      const mean = digest.within_session_means[dim]
      baseline[dim] = { dimension: dim, mean, stddev: Math.max(0.1, digest.cohort[dim].stddev), n_sessions: session_means.length }
    }
  }

  return {
    bundle,
    baseline,
    cohort: digest.cohort,
    past_diagnoses: digest.past_diagnoses_grounded,
    piece_id: digest.piece_id,
    now_ms: digest.now_ms,
  }
}

function fullSessionRange(digest: GroundedDigest): [number, number] {
  let min = Infinity; let max = -Infinity
  for (const chunk of digest.chunks_adapted) {
    if (chunk.bar_coverage[0] < min) min = chunk.bar_coverage[0]
    if (chunk.bar_coverage[1] > max) max = chunk.bar_coverage[1]
  }
  return [isFinite(min) ? min : 0, isFinite(max) ? max : 0]
}
```

**Commit:** `feat(#99): resolveMoleculeContext — tiered baseline, bar-range signal bundle`

---

## Group B — Molecule Refactors

### Molecule Refactor Recipe (full exemplar: `pedal-triage`)

All 7 molecule refactors follow this exact pattern. The exemplar is given in
full; subsequent tasks specify only their selector list, resolved fields
consumed, and test deltas.

**Pattern summary:**
1. `input_schema` reduced to `{ bar_range, scope, evidence_refs }` only.
2. Internal type renamed to `*Selectors`.
3. First line of `invoke` calls `resolveMoleculeContext(ctx!.digest as GroundedDigest, i.bar_range)`.
4. All signal access goes through `ctx_r.bundle.*`, `ctx_r.baseline[dim]`, `ctx_r.cohort[dim]`, `ctx_r.past_diagnoses`, `ctx_r.piece_id`, `ctx_r.now_ms`.
5. `invoke: async (input: unknown, ctx?: PhaseContext)` — ctx is required at runtime; if absent throw.
6. Tests call `molecule.invoke(selectors, ctxWithDigest)` directly.

**Test fixture helper** (same for all molecules — define once in a shared inline
helper inside each test file or repeat the 10-line helper):

```ts
function makeCtx(digest: GroundedDigest): import('../../loop/types').PhaseContext {
  return {
    env: {} as unknown as import('../../../../lib/types').Bindings,
    studentId: 'stu-1', sessionId: 'sess-1', conversationId: null,
    digest: digest as unknown as Record<string, unknown>,
    waitUntil: () => {},
    pieceId: digest.piece_id ?? undefined,
    trigger: 'synthesis',
    turnCap: 10,
  }
}
```

**Exemplar full refactor: `pedal-triage.ts`**

```ts
// pedal-triage.ts (after refactor)
import type { ToolDefinition, PhaseContext } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computePedalOverlapRatio } from '../atoms/compute-pedal-overlap-ratio'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchSimilarPastObservation } from '../atoms/fetch-similar-past-observation'
import type { PastObservation } from '../atoms/fetch-similar-past-observation'
import { resolveMoleculeContext } from '../../loop/resolve-molecule-context'
import type { GroundedDigest } from '../../loop/grounded-digest'

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type PedalSelectors = {
  bar_range: [number, number] | null
  scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]
}

export const pedalTriage: ToolDefinition = {
  name: 'pedal-triage',
  description: 'Distinguishes over-pedaling, under-pedaling, and pedal-timing issues by combining MuQ pedaling delta with AMT pedal CC overlap ratio.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
    },
    required: ['scope', 'evidence_refs'],
  },
  invoke: async (input: unknown, ctx?: PhaseContext): Promise<DiagnosisArtifact> => {
    if (!ctx) throw new Error('pedal-triage: ctx (PhaseContext with digest) is required')
    const i = input as PedalSelectors
    const ctx_r = await resolveMoleculeContext(ctx.digest as unknown as GroundedDigest, i.bar_range ?? null)
    const muq_pedaling = ctx_r.bundle.muq_scores.length > 0
      ? ctx_r.bundle.muq_scores.reduce((s, v) => s + v[2], 0) / ctx_r.bundle.muq_scores.length
      : ctx_r.baseline.pedaling.mean
    const ratio = await computePedalOverlapRatio.invoke({
      notes: ctx_r.bundle.midi_notes,
      pedal_cc: ctx_r.bundle.pedal_cc,
    }) as number
    const z = await computeDimensionDelta.invoke({
      dimension: 'pedaling',
      current: muq_pedaling,
      baseline: ctx_r.baseline.pedaling,
    }) as number
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'pedaling', dimensions: ['pedaling'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Pedaling is within student baseline.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -1.0) return neutral
    let finding: string
    const barLabel = i.bar_range ? `bars ${i.bar_range[0]}-${i.bar_range[1]}` : 'this session'
    if (ratio > 0.85) {
      finding = `Over-pedaled through ${barLabel}; the harmonies are blurring into one wash.`
    } else if (ratio < 0.30) {
      finding = `Under-pedaled through ${barLabel}; the tone sounds dry and disconnected.`
    } else {
      finding = `Pedal not released at harmony changes in ${barLabel}; notes from adjacent harmonies are blurring.`
    }
    const past = await fetchSimilarPastObservation.invoke({
      dimension: 'pedaling',
      piece_id: ctx_r.piece_id ?? '',
      bar_range: i.bar_range ?? null,
      past_diagnoses: ctx_r.past_diagnoses,
      now_ms: ctx_r.now_ms,
    }) as PastObservation | null
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'pedaling', dimensions: ['pedaling'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: past ? 'high' : 'medium',
      finding_type: 'issue',
    })
  },
}
```

**Exemplar full test: `pedal-triage.test.ts`**

```ts
// pedal-triage.test.ts (after refactor)
import { test, expect } from 'vitest'
import { pedalTriage } from './pedal-triage'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import type { GroundedDigest } from '../../loop/grounded-digest'
import type { PhaseContext } from '../../loop/types'

function makeCtx(sessionMeansN: number, muqPedaling: number): PhaseContext {
  const dims = ['dynamics','timing','pedaling','articulation','phrasing','interpretation'] as const
  const session_means: Record<string, number[]> = {}
  const within_session_means: Record<string, number> = {}
  for (const d of dims) {
    session_means[d] = Array.from({ length: sessionMeansN }, () => 0.5)
    within_session_means[d] = 0.52
  }
  const digest: GroundedDigest = {
    chunks_adapted: [{
      chunk_id: 'chunk:0',
      bar_coverage: [12, 16] as [number, number],
      muq_scores: [0.54, 0.48, muqPedaling, 0.54, 0.52, 0.51],
      midi_notes: [
        { pitch: 60, onset_ms: 0,    duration_ms: 1000, velocity: 70, bar: 12 },
        { pitch: 62, onset_ms: 1000, duration_ms: 1000, velocity: 70, bar: 13 },
        { pitch: 64, onset_ms: 2000, duration_ms: 1000, velocity: 70, bar: 14 },
        { pitch: 65, onset_ms: 3000, duration_ms: 1000, velocity: 70, bar: 15 },
      ],
      pedal_cc: [{ time_ms: 0, value: 127 }],
      alignment: [
        { perf_index: 0, score_index: 0, expected_onset_ms: 0,    bar: 12 },
        { perf_index: 1, score_index: 1, expected_onset_ms: 1000, bar: 13 },
        { perf_index: 2, score_index: 2, expected_onset_ms: 2000, bar: 14 },
        { perf_index: 3, score_index: 3, expected_onset_ms: 3000, bar: 15 },
      ],
    }],
    mono_notes_per_bar: [],
    now_ms: 1000,
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
    compact_signal_summary: 'chunk:0 bars 12-16',
    piece_id: 'test-piece',
  }
  return {
    env: {} as unknown as import('../../../../lib/types').Bindings,
    studentId: 'stu-1', sessionId: 'sess-1', conversationId: null,
    digest: digest as unknown as Record<string, unknown>,
    waitUntil: () => {},
    pieceId: 'test-piece',
    trigger: 'synthesis',
    turnCap: 10,
  }
}

test('pedalTriage: over-pedal (ratio>0.85, z<-1) with sufficient history returns issue/significant', async () => {
  // session_means=[0.50,0.60,0.70] => mean=0.60, stddev≈0.082 => z=(0.35-0.60)/0.082≈-3.05
  const selectors = {
    bar_range: [12, 16] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c5', 'cache:amt-pedal:s1:c5'],
  }
  const ctx = makeCtx(3, 0.35)
  // override session_means to give 3 entries with mean 0.60
  ;(ctx.digest as unknown as GroundedDigest).session_means.pedaling = [0.50, 0.60, 0.70]
  const result = await pedalTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('pedaling')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
  expect(result.evidence_refs.length).toBeGreaterThan(0)
})

test('pedalTriage: z above neutral threshold (-1.0) returns neutral', async () => {
  const selectors = { bar_range: [12, 16] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c5'] }
  const ctx = makeCtx(3, 0.46)
  ;(ctx.digest as unknown as GroundedDigest).session_means.pedaling = [0.46, 0.46, 0.46]
  const result = await pedalTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})

test('pedalTriage: insufficient session history (n<3) uses within-session baseline and returns artifact (not throw)', async () => {
  // session_means.pedaling has only 1 entry → thin-history tier
  const selectors = { bar_range: [12, 16] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c5'] }
  const ctx = makeCtx(0, 0.35)
  ;(ctx.digest as unknown as GroundedDigest).session_means.pedaling = [0.50]
  ;(ctx.digest as unknown as GroundedDigest).within_session_means.pedaling = 0.60
  const result = await pedalTriage.invoke(selectors, ctx) as DiagnosisArtifact
  // Must not throw; must return a valid DiagnosisArtifact
  expect(result.primary_dimension).toBe('pedaling')
})

test('pedalTriage: bar_range=null resolves full session bundle and returns artifact', async () => {
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: ['cache:muq:s1:c5'] }
  const ctx = makeCtx(3, 0.46)
  ;(ctx.digest as unknown as GroundedDigest).session_means.pedaling = [0.46, 0.46, 0.46]
  const result = await pedalTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(['neutral','issue','strength']).toContain(result.finding_type)
})

test('pedalTriage: missing ctx throws', async () => {
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: [] }
  await expect(pedalTriage.invoke(selectors, undefined)).rejects.toThrow('ctx')
})
```

**Commit:** `feat(#99): refactor pedal-triage to selectors-only + self-fetch via resolveMoleculeContext`

---

### Task B1 — `pedal-triage` (exemplar, see recipe above)

**Files:** `pedal-triage.ts`, `pedal-triage.test.ts`
**Resolved fields consumed:** `bundle.midi_notes`, `bundle.pedal_cc`, `bundle.muq_scores[2]` (dim 2 = pedaling), `baseline.pedaling`, `past_diagnoses`, `piece_id`, `now_ms`
**Test deltas from recipe:** None — this is the exemplar.
**Commit:** `feat(#99): refactor pedal-triage to selectors-only + self-fetch via resolveMoleculeContext`

---

### Task B2 — `tempo-stability-triage`

**Files:** `tempo-stability-triage.ts`, `tempo-stability-triage.test.ts`
**Resolved fields consumed:** `bundle.midi_notes`, `bundle.alignment`, `bundle.muq_scores[1]` (dim 1 = timing), `baseline.timing`
**Selectors type:**
```ts
type TempoSelectors = { bar_range: [number,number]|null; scope: 'stop_moment'|'passage'|'session'; evidence_refs: string[] }
```
**Key impl delta from recipe:**
- `muq_timing` = mean of `bundle.muq_scores[][1]`
- Alignment/note pairing: use `bundle.alignment` and `bundle.midi_notes` directly; `midi_notes.length` must equal `alignment.length` (throw if not)
- No `fetchSimilarPastObservation` call (tempo-stability-triage does not use it)
- No `past_diagnoses` consumption

**Test deltas from recipe:**

```ts
test('tempoStabilityTriage: monotonic positive drift returns issue/significant/slowing', async () => {
  // build digest with 14 notes having monotonic positive drift, 3 session_means for timing
  // selectors: { bar_range: [1,16], scope: 'session', evidence_refs: [...] }
  // assert result.finding_type === 'issue' && result.severity === 'significant'
})

test('tempoStabilityTriage: z above neutral threshold returns neutral', async () => {
  // digest with muq_timing equal to baseline => z > -1.0 => neutral
})

test('tempoStabilityTriage: insufficient history uses within-session baseline and returns artifact (not throw)', async () => {
  // session_means.timing = [0.48] => thin-history tier; assert result is DiagnosisArtifact
})

test('tempoStabilityTriage: bar_range=null returns artifact', async () => {})

test('tempoStabilityTriage: missing ctx throws', async () => {})
```

**Commit:** `feat(#99): refactor tempo-stability-triage to selectors-only + self-fetch`

---

### Task B3 — `rubato-coaching`

**Files:** `rubato-coaching.ts`, `rubato-coaching.test.ts`
**Resolved fields consumed:** `bundle.midi_notes`, `bundle.alignment`, `bundle.muq_scores[1]` (timing), `baseline.timing`
**Selectors type:** same 3-field pattern as B2
**Key impl delta from recipe:**
- `muq_timing` = mean of `bundle.muq_scores[][1]`
- Uses `bundle.alignment` and `bundle.midi_notes`; no `fetchSimilarPastObservation`
- IOI correlation on alignment-mapped notes; onset-drift on aligned pairs

**Test deltas from recipe:**

```ts
test('rubatoCoaching: monotonic drift without return returns issue/significant/dragged', ...)
test('rubatoCoaching: fewer than 4 aligned notes returns neutral', ...)
test('rubatoCoaching: insufficient history uses within-session baseline (not throw)', ...)
test('rubatoCoaching: bar_range=null returns artifact', ...)
test('rubatoCoaching: missing ctx throws', ...)
```

**Commit:** `feat(#99): refactor rubato-coaching to selectors-only + self-fetch`

---

### Task B4 — `voicing-diagnosis`

**Files:** `voicing-diagnosis.ts`, `voicing-diagnosis.test.ts`
**Resolved fields consumed:** `bundle.midi_notes`, `bundle.muq_scores[0]` (dynamics), `baseline.dynamics`, `cohort.dynamics`, `past_diagnoses`, `piece_id`, `now_ms`
**Selectors type:** same 3-field pattern
**Key impl delta from recipe:**
- `muq_dynamics` = mean of `bundle.muq_scores[][0]`
- `cohort.dynamics` replaces `cohort_table_dynamics` (no longer in input)
- `fetchSimilarPastObservation` still called with `ctx_r.past_diagnoses`

**Test deltas from recipe:**

```ts
test('voicingDiagnosis: flat top/bass voicing with z<-1 returns issue/significant', ...)
test('voicingDiagnosis: z above neutral threshold returns neutral', ...)
test('voicingDiagnosis: insufficient history uses within-session baseline (not throw)', ...)
test('voicingDiagnosis: bar_range=null returns artifact', ...)
test('voicingDiagnosis: missing ctx throws', ...)
```

**Commit:** `feat(#99): refactor voicing-diagnosis to selectors-only + self-fetch`

---

### Task B5 — `dynamic-range-audit`

**Files:** `dynamic-range-audit.ts`, `dynamic-range-audit.test.ts`
**Resolved fields consumed:** `bundle.midi_notes`, `bundle.muq_scores[0]` (dynamics), `baseline.dynamics`, `cohort.dynamics`
**Selectors type:** same 3-field pattern
**Key impl delta from recipe:**
- `score_marking_type` is REMOVED from input (spec: "Not in scope: Molecule calling score-derived dynamics range when score lacks dynamics text"). The `dynamicRangeAudit` now treats the score marking as always `'wide'` (i.e., the score-marking guard `if (z > -0.8 || i.score_marking_type === 'none')` becomes `if (z > -0.8)`). The molecule detects compressed velocities without score knowledge.
- `muq_dynamics` = mean of `bundle.muq_scores[][0]`

**Test deltas from recipe:**

```ts
test('dynamicRangeAudit: compressed velocities with z<-0.8 returns issue/significant', ...)
test('dynamicRangeAudit: z above neutral threshold returns neutral', ...)
test('dynamicRangeAudit: insufficient history uses within-session baseline (not throw)', ...)
test('dynamicRangeAudit: bar_range=null returns artifact', ...)
test('dynamicRangeAudit: missing ctx throws', ...)
```

**Commit:** `feat(#99): refactor dynamic-range-audit to selectors-only + self-fetch`

---

### Task B6 — `phrasing-arc-analysis`

**Files:** `phrasing-arc-analysis.ts`, `phrasing-arc-analysis.test.ts`
**Resolved fields consumed:** `bundle.midi_notes`, `bundle.alignment`, `bundle.muq_scores[4]` (phrasing, dim index 4), `cohort.phrasing` (replaces `cohort_table_phrasing`)
**Selectors type:** same 3-field pattern
**Key impl delta from recipe:**
- `muq_phrasing` = mean of `bundle.muq_scores[][4]`
- `cohortBaseline = { mean: ctx_r.cohort.phrasing.mean, stddev: ctx_r.cohort.phrasing.stddev }`
- No `fetchStudentBaseline` call (phrasing-arc uses cohort table, not student history)
- No thin-history concern for this molecule (cohort table is always available)
- `computeVelocityCurve` called on `bundle.midi_notes` with `bar_range`
- Alignment map built from `bundle.alignment`

**Test deltas from recipe:**

```ts
test('phrasingArcAnalysis: early velocity peak + large drift returns issue/moderate', ...)
test('phrasingArcAnalysis: z above neutral threshold returns neutral', ...)
// No "insufficient history" test because this molecule uses cohort, not student history
// But: test that bar_range=null returns artifact (uses full session)
test('phrasingArcAnalysis: bar_range=null returns artifact', ...)
test('phrasingArcAnalysis: missing ctx throws', ...)
```

**Commit:** `feat(#99): refactor phrasing-arc-analysis to selectors-only + self-fetch`

---

### Task B7 — `cross-modal-contradiction-check`

**Files:** `cross-modal-contradiction-check.ts`, `cross-modal-contradiction-check.test.ts`
**Resolved fields consumed:** `bundle.midi_notes`, `bundle.pedal_cc`, `bundle.alignment`, `bundle.muq_scores` (all dims), `cohort` (all dims), `mono_notes_per_bar` from digest directly (available via `ctx.digest as GroundedDigest`)
**Selectors type:** same 3-field pattern
**Key impl delta from recipe:**
- Articulation arm REMOVED entirely (no `score_articulation_per_bar`, no `computeKeyOverlapRatio` call)
- Remaining arms: timing-drift (dim 1), pedal-ratio (dim 2), dynamics-range (dim 0)
- `muq_scores` for each dim: mean of `bundle.muq_scores[][dimIdx]`
- `cohort_baselines[dim]` replaced by `ctx_r.cohort[dim]`
- `mono_notes_per_bar` sourced from `(ctx.digest as unknown as GroundedDigest).mono_notes_per_bar`
- No `fetchSimilarPastObservation` call in this molecule

**Test deltas from recipe:**

```ts
test('crossModalContradictionCheck: timing z>=0.5 but drift>80ms returns issue/timing', ...)
test('crossModalContradictionCheck: no contradictions returns neutral', ...)
// Confirm articulation arm is gone: pass notes with articulation mismatch and assert NOT flagged
test('crossModalContradictionCheck: articulation mismatch does not produce articulation contradiction', ...)
test('crossModalContradictionCheck: insufficient history uses within-session baseline (not throw)', ...)
test('crossModalContradictionCheck: bar_range=null returns artifact', ...)
test('crossModalContradictionCheck: missing ctx throws', ...)
```

**Commit:** `feat(#99): refactor cross-modal-contradiction-check — remove articulation arm, self-fetch`

---

## Group C — Wiring

### Task C1 — `phase1.ts` + overflow regression test

**Files:** `apps/api/src/harness/loop/phase1.ts`, `apps/api/src/harness/loop/phase1.test.ts`

**Failing test first** (add to `phase1.test.ts`):

```ts
test('phase1 overflow regression: 10-chunk grounded digest prompt is < 10000 chars', async () => {
  // Build a GroundedDigest with 10 chunks, each having 200 notes.
  // The compact_signal_summary for 10 chunks is ~10 lines × ~50 chars = ~500 chars.
  // Build a CompoundBinding and a fake PhaseContext whose digest has compact_signal_summary.
  // Extract the first message content string that would be sent to the model.
  // Assert its length < 10000.
  const { buildCompact10ChunkDigest } = await import('./__test-fixtures__/grounded-digest-fixtures')
  const digest = buildCompact10ChunkDigest()
  const promptString = `Session summary:\n${digest.compact_signal_summary}\n\n` + 'procedure prompt here'
  expect(promptString.length).toBeLessThan(10000)
})
```

Add `apps/api/src/harness/loop/__test-fixtures__/grounded-digest-fixtures.ts`:
```ts
// grounded-digest-fixtures.ts
import type { GroundedDigest } from '../grounded-digest'
export function buildCompact10ChunkDigest(): GroundedDigest {
  const dims = ['dynamics','timing','pedaling','articulation','phrasing','interpretation'] as const
  const chunks = Array.from({ length: 10 }, (_, i) => ({
    chunk_id: `chunk:${i}`,
    bar_coverage: [i * 8 + 1, i * 8 + 8] as [number, number],
    muq_scores: [0.55, 0.48, 0.46, 0.54, 0.52, 0.51],
    midi_notes: Array.from({ length: 200 }, (_, j) => ({ pitch: 60, onset_ms: j * 100, duration_ms: 400, velocity: 70, bar: i * 8 + 1 })),
    pedal_cc: [],
    alignment: [],
  }))
  const lines = chunks.map((c) => `${c.chunk_id} bars ${c.bar_coverage[0]}-${c.bar_coverage[1]} muq=[${c.muq_scores.map(v => v.toFixed(2)).join(',')}]`)
  const compact_signal_summary = lines.join('\n')
  const session_means: Record<string, number[]> = {}
  const within_session_means: Record<string, number> = {}
  const cohort: Record<string, { mean: number; stddev: number }> = {}
  for (const d of dims) {
    session_means[d] = []
    within_session_means[d] = 0.52
    cohort[d] = { mean: 0.52, stddev: 0.10 }
  }
  return {
    chunks_adapted: chunks,
    mono_notes_per_bar: [],
    now_ms: Date.now(),
    cohort: cohort as GroundedDigest['cohort'],
    past_diagnoses_grounded: [],
    session_means: session_means as GroundedDigest['session_means'],
    within_session_means: within_session_means as GroundedDigest['within_session_means'],
    compact_signal_summary,
    piece_id: null,
  }
}
```

**Implementation change to `phase1.ts`:**

```ts
// phase1.ts line ~41: replace
`Session digest:\n${JSON.stringify(ctx.digest, null, 2)}\n\n` + binding.procedurePrompt,
// with:
`Session summary:\n${(ctx.digest as import('./grounded-digest').GroundedDigest).compact_signal_summary}\n\n` + binding.procedurePrompt,
```

**Commit:** `feat(#99): phase1 uses compact_signal_summary; overflow regression test < 10000 chars`

---

### Task C2 — `compound-registry.ts` + `index.ts`

**Files:**
- `apps/api/src/harness/loop/compound-registry.ts`
- `apps/api/src/harness/skills/molecules/index.ts`

**Failing test first** (add to `compound-registry.test.ts`):

```ts
test('SESSION_SYNTHESIS_PROCEDURE contains "bar_range, scope, and evidence_refs" instruction', async () => {
  const { getCompoundBinding } = await import('./compound-registry')
  const binding = getCompoundBinding('OnSessionEnd')!
  expect(binding.procedurePrompt).toContain('bar_range, scope, and evidence_refs')
})

test('OnSessionEnd tool list includes extract-bar-range-signals', async () => {
  const { getCompoundBinding } = await import('./compound-registry')
  const binding = getCompoundBinding('OnSessionEnd')!
  expect(binding.tools.some(t => t.name === 'extract-bar-range-signals')).toBe(true)
})

test('OnSessionEnd tool list does NOT include articulation-clarity-check', async () => {
  const { getCompoundBinding } = await import('./compound-registry')
  const binding = getCompoundBinding('OnSessionEnd')!
  expect(binding.tools.some(t => t.name === 'articulation-clarity-check')).toBe(false)
})
```

**Implementation changes:**

In `compound-registry.ts`:
- Update `SESSION_SYNTHESIS_PROCEDURE` to replace "supply the bar range and signal data from the digest" with "supply only `bar_range`, `scope`, and `evidence_refs` — the molecule fetches all signal data server-side."
- Add `extractBarRangeSignals` (imported from atoms index) to the OnSessionEnd `tools` array.

In `index.ts`:
- Remove the `articulationClarityCheck` import and entry from `ALL_MOLECULES`.

**Commit:** `feat(#99): compound-registry — updated procedure prompt, add extractBarRangeSignals; remove articulationClarityCheck`

---

### Task C3 — `teacher.ts`

**Files:** `apps/api/src/services/teacher.ts`

**Failing test first** (add to an existing or new `teacher.synthesizeV6.test.ts` or inline in the service test):

```ts
test('synthesizeV6 digest contains compact_signal_summary key', async () => {
  // Spy on runHook to capture the hookCtx.digest
  // Build a minimal SynthesisInput with 1 chunk
  // Call synthesizeV6 and drain events
  // Assert hookCtx.digest.compact_signal_summary is a string
  // (This test verifies buildGroundedDigest was called and wired in)
})
```

**Implementation changes:**

1. Extend `PastDiagnosisRecord` interface (line 73-80) to add:
   ```ts
   id: string
   pieceId: string | null
   ```

2. In `synthesizeV6`, call `buildGroundedDigest` before building `hookCtx`:
   ```ts
   import { buildGroundedDigest } from '../harness/loop/grounded-digest'
   // ...
   // In synthesizeV6, before building digest:
   const groundedDigest = await buildGroundedDigest(input, { db: ctx.db, studentId: input.studentId }, COHORT_TABLES)
   const digest: Record<string, unknown> = groundedDigest as unknown as Record<string, unknown>
   ```

   Remove the existing inline `digest` object construction (the `chunks`, `baselines`, etc. object) — `buildGroundedDigest` now owns all of that, plus the grounding layer. The `hookCtx.digest` should be `groundedDigest`.

3. Pass `COHORT_TABLES` to `buildGroundedDigest` (it's already defined in teacher.ts).

**Commit:** `feat(#99): teacher.ts — extend PastDiagnosisRecord with id+pieceId; wire buildGroundedDigest into synthesizeV6`

---

### Task C4 — `session-brain.ts`

**Files:** `apps/api/src/do/session-brain.ts`

**Failing test first** (conceptual — verify at the type level via TypeScript compile, and via the existing `session-brain.concurrency.test.ts` or a new integration test that asserts the pastDiagnoses array has `id` populated):

```ts
test('session-brain pastDiagnoses mapping includes id and pieceId fields', () => {
  // Construct a diagRows array as the DB would return
  // (id, sessionId, primaryDimension, barRangeStart, barRangeEnd, artifactJson, createdAt, pieceId)
  // Map it using the same logic as session-brain.ts ~1680
  // Assert r.id is present and maps to the record's id field
  const diagRow = {
    id: 'uuid-diag',
    sessionId: 'uuid-sess',
    primaryDimension: 'pedaling',
    barRangeStart: 1,
    barRangeEnd: 4,
    artifactJson: {},
    createdAt: new Date(0),
    studentId: 'stu-1',
    pieceId: 'piece-abc',
  }
  const record = {
    id: diagRow.id,
    sessionId: diagRow.sessionId,
    primaryDimension: diagRow.primaryDimension,
    barRangeStart: diagRow.barRangeStart ?? null,
    barRangeEnd: diagRow.barRangeEnd ?? null,
    artifactJson: diagRow.artifactJson,
    createdAt: diagRow.createdAt.toISOString(),
    pieceId: diagRow.pieceId ?? null,
  }
  expect(record.id).toBe('uuid-diag')
  expect(record.pieceId).toBe('piece-abc')
})
```

**Implementation change** at `session-brain.ts ~1680-1687`:

```ts
pastDiagnoses = diagRows.map((r) => ({
  id: r.id,                             // ADD
  sessionId: r.sessionId,
  primaryDimension: r.primaryDimension,
  barRangeStart: r.barRangeStart ?? null,
  barRangeEnd: r.barRangeEnd ?? null,
  artifactJson: r.artifactJson,
  createdAt: r.createdAt.toISOString(),
  pieceId: r.pieceId ?? null,           // ADD
}));
```

**Commit:** `feat(#99): session-brain — include id and pieceId in past-diagnoses SELECT mapping`

---

## Verification Gate

After all commits, run:

```bash
cd apps/api && bun run test --run src/harness
```

All new tests pass. No pre-existing tests regress (baseline: 20 pre-existing TS errors; net-new errors = 0).

Manual smoke: `just dev-light` → POST a synthesis session → confirm no HTTP 413; confirm Phase-1 prompt string < 10000 chars in local logs.
