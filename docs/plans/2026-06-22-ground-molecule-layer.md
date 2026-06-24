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

  // Adapt chunks: assign bar to midi_notes from chunk's bar_coverage (chunk-level approximation).
  // NOTE: alignment is always [] in production (barMapAlignments never populated in session-brain.ts).
  // Per-note bar from the WASM BarMap is a follow-up; for now every note in a chunk gets
  // bar = bar_coverage[0]. When bar_coverage is null/missing, notes contribute bar = 0 and
  // are effectively excluded from bar-range filtering.
  const chunks_adapted: AdaptedChunk[] = input.enrichedChunks.map((chunk) => {
    const chunkBar = chunk.bar_coverage != null ? chunk.bar_coverage[0] : 0
    const grounded_notes: GroundedNote[] = chunk.midi_notes.map((note) => ({
      ...note,
      bar: chunkBar,
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

  // cohort stats: mean=p50, stddev=max(0.01, p75-p50).
  // Production COHORT_TABLES has p25/p50/p75/p90 only — p84 is absent.
  // p75 is the standard one-sigma proxy available in the production table.
  const cohort = {} as CohortStats
  for (const dim of DIMENSIONS_6) {
    const table = cohortTables[dim] ?? []
    const p50 = table.find((e) => e.p === 50)?.value ?? 0.5
    const p75 = table.find((e) => e.p === 75)?.value ?? (p50 + 0.1)
    cohort[dim] = { mean: p50, stddev: Math.max(0.01, p75 - p50) }
  }

  // within_session_means: mean muq_scores[dimIdx] across all chunks
  const within_session_means = {} as Record<Dim6, number>
  for (let dimIdx = 0; dimIdx < DIMENSIONS_6.length; dimIdx++) {
    const dim = DIMENSIONS_6[dimIdx]
    const vals = input.enrichedChunks.map((c) => c.muq_scores[dimIdx]).filter((v) => typeof v === 'number')
    within_session_means[dim] = vals.length > 0 ? vals.reduce((s, v) => s + v, 0) / vals.length : 0.5
  }

  // session_means: per-session AVG(dimension_score) from last 10 sessions ordered by recency.
  // LIMIT caps the scan: at 6 dims × 10 sessions = 60 rows max. Throws if DB fails.
  const rows = await deps.db
    .select({
      sessionId: observations.sessionId,
      dimension: observations.dimension,
      avgScore: sql<number>`AVG(${observations.dimensionScore})`,
    })
    .from(observations)
    .where(and(eq(observations.studentId, deps.studentId)))
    .groupBy(observations.sessionId, observations.dimension)
    .orderBy(sql`MAX(${observations.createdAt}) DESC`)
    .limit(10 * 6)

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

  // NOTE: the 6 per-dimension baseline computations may be awaited together via Promise.all
  // rather than 6 sequential awaits. fetchStudentBaseline is a pure computation (no I/O),
  // so parallelising is straightforward and avoids 6 sequential microtask round-trips.
  const baselineEntries = await Promise.all(
    DIMENSIONS_6.map(async (dim) => {
      const session_means = digest.session_means[dim]
      if (session_means.length >= 3) {
        const result = await fetchStudentBaseline.invoke({ dimension: dim, session_means }) as Baseline | null
        if (result === null) {
          throw new Error(`resolveMoleculeContext: fetchStudentBaseline returned null despite n=${session_means.length} sessions for dim=${dim}`)
        }
        return [dim, result] as const
      } else {
        // Thin-history tier: synthesise from within_session_means
        const mean = digest.within_session_means[dim]
        return [dim, { dimension: dim, mean, stddev: Math.max(0.1, digest.cohort[dim].stddev), n_sessions: session_means.length }] as const
      }
    })
  )
  const baseline = Object.fromEntries(baselineEntries) as TieredBaseline

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
- REMOVE the `midi_notes.length !== alignment.length` throw (line 47 of current file). After bar-range filtering both arrays are independently filtered and are not guaranteed co-length. When `bundle.alignment` is empty or insufficient for IOI computation, return a NEUTRAL `DiagnosisArtifact` (grounded, no fabrication). Timing/IOI arms are data-limited until real alignment is populated (follow-up to the bar-precision follow-up in the spec).
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

test('tempoStabilityTriage: empty/sparse alignment returns neutral (not throw)', async () => {
  // chunk has midi_notes but alignment = [] (production reality)
  // selectors: { bar_range: [1,4], scope: 'session', evidence_refs: ['cache:muq:s1:c0'] }
  // assert: does NOT throw; returns DiagnosisArtifact with finding_type 'neutral'
  const ctx = makeCtx(3, 0.48)
  ;(ctx.digest as unknown as GroundedDigest).chunks_adapted[0].alignment = []
  const selectors = { bar_range: [1, 4] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c0'] }
  const result = await tempoStabilityTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
  expect(result.primary_dimension).toBe('timing')
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
- REMOVE any `midi_notes.length !== alignment.length` throw guard. When `bundle.alignment` is empty or insufficient (< 4 correlated pairs), return a NEUTRAL `DiagnosisArtifact` — not a throw. Timing/IOI arms are data-limited until real alignment is populated (follow-up in spec).

**Test deltas from recipe:**

```ts
test('rubatoCoaching: monotonic drift without return returns issue/significant/dragged', ...)
test('rubatoCoaching: fewer than 4 aligned notes returns neutral', ...)
test('rubatoCoaching: empty/sparse alignment returns neutral (not throw)', async () => {
  // alignment = [] => IOI computation impossible => neutral result, no throw
  const ctx = makeCtx(3, 0.48)
  ;(ctx.digest as unknown as GroundedDigest).chunks_adapted[0].alignment = []
  const selectors = { bar_range: [1, 4] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c0'] }
  const result = await rubatoCoaching.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})
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
- REMOVE any `midi_notes.length !== alignment.length` throw guard in the timing arm. When `bundle.alignment` is empty or insufficient for onset drift computation, the timing arm returns no contradiction (skips silently) — overall molecule still returns a valid `DiagnosisArtifact` (neutral if no other arms fire).

**Test deltas from recipe:**

```ts
test('crossModalContradictionCheck: timing z>=0.5 but drift>80ms returns issue/timing', ...)
test('crossModalContradictionCheck: no contradictions returns neutral', ...)
// Confirm articulation arm is gone: pass notes with articulation mismatch and assert NOT flagged
test('crossModalContradictionCheck: articulation mismatch does not produce articulation contradiction', ...)
test('crossModalContradictionCheck: empty/sparse alignment does not throw — timing arm skipped, returns artifact', async () => {
  // alignment = [] => timing arm has no data => skips; pedal and dynamics arms can still run
  const ctx = makeCtx(3, [0.85, 0.35, 0.46, 0.54, 0.52, 0.51])
  ;(ctx.digest as unknown as GroundedDigest).chunks_adapted[0].alignment = []
  const selectors = { bar_range: [1, 4] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c0'] }
  const result = await crossModalContradictionCheck.invoke(selectors, ctx) as DiagnosisArtifact
  // No throw; result is a valid DiagnosisArtifact
  expect(['neutral', 'issue', 'strength']).toContain(result.finding_type)
})
test('crossModalContradictionCheck: insufficient history uses within-session baseline (not throw)', ...)
test('crossModalContradictionCheck: bar_range=null returns artifact', ...)
test('crossModalContradictionCheck: missing ctx throws', ...)
```

**Commit:** `feat(#99): refactor cross-modal-contradiction-check — remove articulation arm, self-fetch`

---

## Group C — Wiring

### Task C1 — `phase1.ts` + overflow regression test

**Files:** `apps/api/src/harness/loop/phase1.ts`, `apps/api/src/harness/loop/phase1.test.ts`

**Implementation change to `phase1.ts` first:**

Extract the Phase-1 user message construction into an exported helper so the test can call it directly:

```ts
// In phase1.ts — add this exported helper alongside the existing runPhase1 function:
import type { GroundedDigest } from './grounded-digest'

export function buildPhase1UserMessage(digest: GroundedDigest, procedurePrompt: string): string {
  return `Session summary:\n${digest.compact_signal_summary}\n\n${procedurePrompt}`
}
```

Then in `runPhase1`, replace the raw digest dump:
```ts
// Before (line ~41):
// `Session digest:\n${JSON.stringify(ctx.digest, null, 2)}\n\n` + binding.procedurePrompt
// After:
buildPhase1UserMessage(ctx.digest as unknown as GroundedDigest, binding.procedurePrompt)
```

**Failing test first** (add to `phase1.test.ts`):

```ts
import { buildPhase1UserMessage } from './phase1'
import { buildCompact10ChunkDigest } from './__test-fixtures__/grounded-digest-fixtures'

test('phase1 overflow regression: buildPhase1UserMessage with 10-chunk digest produces < 10000 chars', () => {
  // Uses the ACTUAL buildPhase1UserMessage from phase1.ts — not a hand-built string.
  // A 10-chunk compact_signal_summary is ~10 lines × ~50 chars = ~500 chars.
  // Asserts the path phase1.ts uses for the LLM user message stays well under context limits.
  const digest = buildCompact10ChunkDigest()
  const result = buildPhase1UserMessage(digest, 'procedure prompt here')
  expect(result.length).toBeLessThan(10000)
  // Also assert the old raw midi_notes JSON is NOT present:
  expect(result).not.toContain('"midi_notes"')
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

**Also update these catalog test files** (removing `articulationClarityCheck` from `ALL_MOLECULES` breaks them without these changes):

In `apps/api/src/harness/skills/molecules/index.test.ts`:
- Change `expect(ALL_MOLECULES).toHaveLength(8)` → `expect(ALL_MOLECULES).toHaveLength(7)`
- Change `expect(new Set(names).size).toBe(8)` → `expect(new Set(names).size).toBe(7)`
- In the second test, remove `'articulation-clarity-check'` from the `expected` array (7 names remain)

In `apps/api/src/harness/skills/__catalog__/readme-molecules.test.ts`:
- Remove `"articulation-clarity-check"` from the `FINAL_MOLECULES` array (7 entries remain)

In `docs/harness/skills/molecules/README.md`:
- Change `**Final size:** 8 molecules.` → `**Final size:** 7 molecules.`
- Remove the `- \`articulation-clarity-check\`` bullet from the Diagnosis molecules list

**Commit:** `feat(#99): compound-registry — updated procedure prompt, add extractBarRangeSignals; remove articulationClarityCheck`

---

### Task C3 — `teacher.ts`

**Files:** `apps/api/src/services/teacher.ts`

**Failing test first** (add to `compound-registry.test.ts` — these are behavior tests on the registry, not internal-state spying):

```ts
import { test, expect } from 'vitest'

test('OnSessionEnd procedurePrompt instructs supply of bar_range/scope/evidence_refs only (not signal data)', async () => {
  const { getCompoundBinding } = await import('./compound-registry')
  const binding = getCompoundBinding('OnSessionEnd')!
  // Confirm the new instruction is present
  expect(binding.procedurePrompt).toContain('bar_range, scope, and evidence_refs')
  // Confirm the old instruction to supply signal data is gone
  expect(binding.procedurePrompt).not.toContain('signal data from the digest')
})

test('OnSessionEnd tool list includes extract-bar-range-signals', async () => {
  const { getCompoundBinding } = await import('./compound-registry')
  const binding = getCompoundBinding('OnSessionEnd')!
  expect(binding.tools.some((t) => t.name === 'extract-bar-range-signals')).toBe(true)
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

**Files:** `apps/api/src/do/session-brain.ts`, `apps/api/src/do/session-brain.test.ts` (new or existing)

**Implementation change first** — extract the reshape into an exported pure function so it is directly testable:

```ts
// In session-brain.ts — add this exported helper near the pastDiagnoses mapping (~line 1675):
import type { PastDiagnosisRecord } from '../services/teacher'

export function toPastDiagnosisRecord(row: {
  id: string
  sessionId: string
  primaryDimension: string
  barRangeStart: number | null
  barRangeEnd: number | null
  artifactJson: unknown
  createdAt: Date
  pieceId: string | null | undefined
}): PastDiagnosisRecord {
  return {
    id: row.id,
    sessionId: row.sessionId,
    primaryDimension: row.primaryDimension,
    barRangeStart: row.barRangeStart ?? null,
    barRangeEnd: row.barRangeEnd ?? null,
    artifactJson: row.artifactJson as Record<string, unknown>,
    createdAt: row.createdAt.toISOString(),
    pieceId: row.pieceId ?? null,
  }
}
```

Then at `session-brain.ts ~1680-1687`, replace the inline mapping:
```ts
// Before:
pastDiagnoses = diagRows.map((r) => ({
  sessionId: r.sessionId,
  primaryDimension: r.primaryDimension,
  barRangeStart: r.barRangeStart ?? null,
  barRangeEnd: r.barRangeEnd ?? null,
  artifactJson: r.artifactJson,
  createdAt: r.createdAt.toISOString(),
}));
// After:
pastDiagnoses = diagRows.map(toPastDiagnosisRecord);
```

**Failing test first** (import and call the extracted function — not a shape copy):

```ts
// session-brain.test.ts (new file or add to existing)
import { test, expect } from 'vitest'
import { toPastDiagnosisRecord } from './session-brain'

test('toPastDiagnosisRecord: maps id, pieceId, and createdAt ISO string from DB row', () => {
  const row = {
    id: 'uuid-diag',
    sessionId: 'uuid-sess',
    primaryDimension: 'pedaling',
    barRangeStart: 1,
    barRangeEnd: 4,
    artifactJson: { severity: 'moderate' },
    createdAt: new Date(0),
    pieceId: 'piece-abc',
  }
  const record = toPastDiagnosisRecord(row)
  expect(record.id).toBe('uuid-diag')
  expect(record.pieceId).toBe('piece-abc')
  expect(record.createdAt).toBe(new Date(0).toISOString())
  expect(record.barRangeStart).toBe(1)
  expect(record.barRangeEnd).toBe(4)
})

test('toPastDiagnosisRecord: null pieceId and null barRange pass through as null', () => {
  const row = {
    id: 'uuid-diag-2',
    sessionId: 'uuid-sess-2',
    primaryDimension: 'dynamics',
    barRangeStart: null,
    barRangeEnd: null,
    artifactJson: {},
    createdAt: new Date(1000),
    pieceId: null,
  }
  const record = toPastDiagnosisRecord(row)
  expect(record.pieceId).toBeNull()
  expect(record.barRangeStart).toBeNull()
  expect(record.barRangeEnd).toBeNull()
})
```

**Commit:** `feat(#99): session-brain — extract toPastDiagnosisRecord with id+pieceId; wire into pastDiagnoses mapping`

---

## Verification Gate

After all commits, run:

```bash
cd apps/api && bun run test --run src/harness
```

All new tests pass. No pre-existing tests regress (baseline: 20 pre-existing TS errors; net-new errors = 0).

Manual smoke: `just dev-light` → POST a synthesis session → confirm no HTTP 413; confirm Phase-1 prompt string < 10000 chars in local logs.

---

## Challenge Review

### CEO Pass

**Premise:** Correct and urgent. `phase1.ts:41` is confirmed to `JSON.stringify(ctx.digest, null, 2)` the full digest including raw `midi_notes`, `pedal_cc`, and `alignment` arrays for every chunk. The 136K-token overflow is real and reproducible at 10 chunks. The root problem (LLM receiving fabricated inputs because molecules had fat input_schemas the LLM was expected to populate) is also verified in the current molecule code. This is exactly the right thing to fix.

**Direct path:** Yes. The two new deep modules (`buildGroundedDigest`, `resolveMoleculeContext`) are the minimum surface to fix all three breakages cleanly. Simpler alternatives (e.g. truncating the JSON, or passing data via tool results instead of the digest) would leave molecules fabricating inputs. The spec chose the correct path.

**Alternatives:** Spec documents the design choice implicitly (compact summary vs. raw dump, server-side fetch vs. LLM-supplied data) but does not write down rejected alternatives. Not blocking for a pre-beta codebase.

**Twelve-month alignment:**
```
CURRENT STATE                          THIS PLAN                         12-MONTH IDEAL
Molecules get LLM-fabricated           Molecules self-fetch via           Molecules are fully
inputs; digest is 136K raw             resolveMoleculeContext;            autonomous units with
JSON; thin-history throws;             compact summary in prompt;         real baselines, bar-level
bar field missing on notes.            7 molecules grounded.              signal, and instrument
                                                                         adaptation.
```
This plan moves cleanly toward the ideal. No tech debt is introduced.

---

### Engineering Pass

**Architecture — data flow:**
```
SynthesisInput
  → buildGroundedDigest(input, {db, studentId}, COHORT_TABLES)
      ├── bar inject (alignment.perf_index join)  ← [BLOCKER B1 below]
      ├── DB query: AVG(dimensionScore) GROUP BY (sessionId, dimension)
      ├── cohort stats (p50/p84 arithmetic)       ← [BLOCKER B2 below]
      └── compact_signal_summary string
  → hookCtx.digest = groundedDigest
  → phase1.ts: compact_signal_summary in prompt
  → LLM calls molecule.invoke({bar_range, scope, evidence_refs}, ctx)
  → resolveMoleculeContext(ctx.digest, bar_range)
      ├── extractBarRangeSignals.invoke({bar_range, chunks})  ← [BLOCKER B3 below]
      └── tiered baseline (n>=3 → fetchStudentBaseline, else within-session)
  → molecule logic on ctx_r.bundle.*
```

---

### Blockers

**[BLOCKER] (confidence: 10/10) — Bar injection join is always a no-op: `alignment` is always `[]` in `EnrichedChunk`.**

Verified in `session-brain.ts`: `barMapAlignments` is declared at lines 745 and 1220 as an empty array `[]` and is never pushed to. There is no code path that populates it before it is passed to `toEnrichedChunk`. The `alignment` field in every stored `EnrichedChunk` is therefore always `[]`. Consequently, `buildGroundedDigest`'s `perfToBar = new Map(chunk.alignment.map(...))` builds an empty map, and every `midi_notes[idx]` gets `bar = perfToBar.get(idx) ?? 0` — i.e. `bar = 0` for every note. The spec's core promise ("bar injected onto midi_notes via alignment perf_index join") is structurally impossible with the current data. The plan must either (a) populate `barMapAlignments` from the WASM BarMap result that already exists in the chunk_ready path, or (b) derive bar from `bar_coverage` as a chunk-level approximation. The plan cannot proceed as written because the grounding is a no-op.

**[BLOCKER] (confidence: 9/10) — `p84` percentile does not exist in production `COHORT_TABLES`; the stddev formula silently falls back to `p50 + 0.1` in production.**

Verified: `COHORT_TABLES` in `teacher.ts` (lines 976–982) has percentile entries for `p: 25`, `p: 50`, `p: 75`, and `p: 90` only — no `p: 84`. The plan's `buildGroundedDigest` implementation looks up `table.find((e) => e.p === 84)`, which returns `undefined`, and then falls back to `p50 + 0.1`. The test fixture *does* include `p84` entries so the tests will pass, but in production `cohort[dim].stddev = 0.10` for all dimensions regardless of actual spread. The `p75` percentile is present and is the standard one-sigma proxy (p84 ≈ mean+1σ for normal distributions is approximate anyway). The plan should either add `p: 84` entries to `COHORT_TABLES` in `teacher.ts`, or change the formula to use `p75` (which exists). As written: tests green, production silently wrong.

**[BLOCKER] (confidence: 9/10) — `tempo-stability-triage` (B2) will throw on real bar-filtered data because `bundle.midi_notes.length !== bundle.alignment.length` after `extractBarRangeSignals` filtering.**

Verified: the current `tempo-stability-triage.ts` line 47 already throws if `midi_notes.length !== alignment.length`. After the refactor, `resolveMoleculeContext` calls `extractBarRangeSignals` which independently bar-filters `midi_notes` (by `n.bar >= lo && n.bar <= hi`) and `alignment` (by `a.bar >= lo && a.bar <= hi`). These two filtered arrays are not guaranteed to be the same length — a note with `bar = 0` (the fallback from BLOCKER B1) would be excluded from midi_notes but alignment entries for that note would also be excluded, which happens to make lengths equal only if the no-op bar=0 case hits uniformly. More importantly, if BLOCKER B1 is fixed and real bar values are present, notes outside the bar_range are excluded from midi_notes but alignment entries for those same notes are also excluded independently — they coincide only when every note in the chunk has a matching alignment entry. The plan's instruction "midi_notes.length must equal alignment.length (throw if not)" creates a guaranteed production throw whenever the filtered lists differ. The plan must specify that the length guard is removed, or that the note-alignment join uses `perf_index` from `bundle.alignment` rather than positional indexing.

**[BLOCKER] (confidence: 8/10) — Task C4's test is a shape test that would pass without the implementation.**

The C4 test (plan lines 1082–1111) manually constructs a `diagRow` object and a `record` object inline in the test body, asserting `record.id === 'uuid-diag'`. This does not call any function from `session-brain.ts`. It does not import `session-brain.ts`. It would pass identically whether or not the `session-brain.ts` mapping at line 1680 is changed. This is a copying-the-code-into-the-test pattern, not a behavior test. Required fix: test must call a function exported from `session-brain.ts` (or the DO itself via a stub harness) with a mock DB row and assert the `PastDiagnosisRecord` returned by the actual mapping logic includes `id` and `pieceId`. Alternatively, verify at TypeScript compile time only (no runtime test needed for a 2-line field addition), but then delete the test rather than keep a vacuous one.

**[BLOCKER] (confidence: 8/10) — Two existing catalog tests assert `ALL_MOLECULES` has 8 entries and specifically names `articulation-clarity-check`. Removing it in C2 will break both.**

Verified: `molecules/index.test.ts` asserts `ALL_MOLECULES.toHaveLength(8)` and explicitly lists `articulation-clarity-check` as one of the 8 expected names. `readme-molecules.test.ts` asserts that `docs/harness/skills/molecules/README.md` lists all 8 final molecules by name, including `articulation-clarity-check`. The plan's C2 task removes `articulationClarityCheck` from `ALL_MOLECULES` and from `index.ts` but does not update these two tests or the README. Both will fail at the verification gate. The plan must include: (1) update `molecules/index.test.ts` to expect 7 molecules and drop the name; (2) update `readme-molecules.test.ts` `FINAL_MOLECULES` list; (3) update `docs/harness/skills/molecules/README.md`.

---

### Risks

**[RISK] (confidence: 8/10) — The `session_means` DB query has no LIMIT and no time-window filter; it will scan all observations for the student across all time.**

The spec says "last-10-session means" but the plan's SQL implementation (`AVG(dimensionScore) GROUP BY (sessionId, dimension)` with only a `studentId` WHERE clause) returns averages for every session the student has ever had. For pre-beta this is negligible, but the session count grows unboundedly and the `idx_observations_student` index is on `(studentId, createdAt)` — the query has no `createdAt` bound, so it can't use the index efficiently as data grows. Consider adding `.orderBy(observations.createdAt, 'desc').limit(10 * 6)` or a subquery to cap the scan to the last 10 sessions.

**[RISK] (confidence: 8/10) — `cross-modal-contradiction-check` (B7) carries the existing perf-index-vs-array-index bug into the refactored version.**

The current `cross-modal-contradiction-check.ts` line 62 builds `alignMap` from `alignment.perf_index` but then looks up entries with array-position `idx` from `midi_notes.map((n, idx) => ...)`. After bar-range filtering, `idx` (position in filtered array) no longer equals `perf_index`. The plan says to use `bundle.alignment` and `bundle.midi_notes` directly without specifying a fix to this join. The refactored B7 molecule will silently fail to match most notes (getting `undefined` from `alignMap.get(idx)`) and the `filter` for `expected_onset_ms !== null` will drop them. The timing contradiction arm will almost never fire. Fix: join on `note.bar` (which is now present) or build the note map by `perf_index` from the grounded notes.

**[RISK] (confidence: 7/10) — `rubato-coaching` (B3) carries the same index-vs-perf_index mismatch for its alignment join.**

`rubato-coaching.ts` line 50: `perfNoteMap = new Map(i.midi_notes.map((n, idx) => [idx, n.onset_ms]))`. After the refactor, `i.midi_notes` becomes `bundle.midi_notes` (bar-filtered), so positional `idx` no longer corresponds to `perf_index`. The `correlationNotes` built from `i.alignment.map(a => ({ onset_ms: perfNoteMap.get(a.perf_index) ?? ... }))` will miss most entries and fall back to `a.expected_onset_ms` for almost all notes, making the IOI correlation measure the score timing, not the performance. Fallback to neutral is the result rather than a throw, so this degrades silently. Fix: use `perf_index` in both maps consistently.

**[RISK] (confidence: 7/10) — Task C1's overflow test does not test `phase1.ts`; it is a vacuous string-length check.**

The test (plan lines 929–940) manually constructs a `promptString` from a fixture and asserts its length. It never calls `runPhase1`, never exercises `phase1.ts` line 41, and would pass regardless of whether `phase1.ts` is changed. It catches nothing about the actual runtime behavior of `phase1.ts` with a grounded digest. This is a shape/proxy test. Consider: import `runPhase1`, stub the `callModel` call (Cloudflare Workers test infra already supports this pattern), and assert that the first message's content does not include `midi_notes` as a substring.

**[RISK] (confidence: 7/10) — `resolveMoleculeContext` calls `fetchStudentBaseline.invoke(...)` once per dimension (6 calls) inside `buildGroundedDigest` via the tiered baseline loop. Each `invoke` is `async`. For a new student (n < 3 for all 6 dims), all 6 calls short-circuit immediately. But for a returning student with n >= 3, this is 6 sequential async calls per synthesis. Given `fetchStudentBaseline` is a pure computation (no I/O — it just does arithmetic on the `session_means` array), wrapping it in sequential `await` is unnecessary overhead. Not a correctness issue, but flag for the builder to run the 6 calls in `Promise.all` or, preferably, inline the computation directly rather than calling through `ToolDefinition.invoke`.**

**[RISK] (confidence: 6/10) — Task C3's test is described as conceptual ("Spy on runHook...") with no concrete implementation provided.**

The plan's C3 test sketch says "Spy on runHook to capture the hookCtx.digest" but provides no actual test code — only a comment describing what to verify. The builder will have to invent the test from scratch, which means it may end up being a mock-heavy internal-state test rather than a behavior test. Recommend: make the test concrete in the plan, or accept that C3's only coverage is the TypeScript compile check on `PastDiagnosisRecord` extension.

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `alignment` in `EnrichedChunk` contains populated `bar` entries | RISKY | Verified: `barMapAlignments` is always `[]`; bar injection is a no-op |
| `COHORT_TABLES` contains `p: 84` entries | RISKY | Verified: only `p25, p50, p75, p90` present; no `p84` |
| `bundle.midi_notes.length === bundle.alignment.length` after bar-range filtering | RISKY | Not guaranteed; both are independently filtered |
| `fetchStudentBaseline` returns non-null when `session_means.length >= 3` | SAFE | Verified in `fetch-student-baseline.ts` line 37 |
| `extractBarRangeSignals.invoke({bar_range, chunks})` is safe to call directly (not just as LLM tool) | SAFE | Verified: `invoke` is a plain async function, no LLM-context dependency |
| Removing `articulationClarityCheck` from `ALL_MOLECULES` requires no catalog test updates | RISKY | Verified: two catalog tests hard-code the count 8 and the name |
| The `observations` table has columns `studentId`, `sessionId`, `dimension`, `dimensionScore` | SAFE | Verified in `observations.ts` schema |
| `ctx.digest` in `PhaseContext` accepts `GroundedDigest` at runtime | SAFE | `digest: Record<string, unknown>` — structural typing permits it |
| The existing `session-brain.ts` past-diagnoses query selects `diagnosisArtifacts.id` | SAFE | Verified: `diagnosisArtifacts.id` column exists (uuid, primaryKey) |
| `diagnosisArtifacts` table has `pieceId` column | SAFE | Verified in `diagnosis-artifacts.ts` schema |
| `tempo-stability-triage` length guard survives bar-range filtering | RISKY | After filtering, `midi_notes.length !== alignment.length` is likely |
| Production `p84` fallback `= p50 + 0.1` is close enough to actual `p84 - p50` | VALIDATE | For a normal-ish distribution `p84 ≈ p50 + σ` — depends on actual score spread |

---

### Summary

[BLOCKER] count: 5
[RISK]    count: 6
[QUESTION] count: 0

VERDICT: NEEDS_REWORK — Five blockers must be resolved before execution: (1) `alignment` is always `[]` making bar injection a structural no-op; (2) `p84` is absent from production `COHORT_TABLES` making stddev silently wrong; (3) `tempo-stability-triage` will throw on real bar-filtered data due to length mismatch; (4) Task C4's test is a shape test with zero coupling to the implementation; (5) two catalog tests hard-code `articulation-clarity-check` and count=8, which C2 breaks without updating them.

---

## Challenge Review — Loop 2 (post-commit f7dd15a2)

**Scope:** Re-review after fix pass that resolved 5 prior blockers in the plan document. All source files verified against actual code. No source files were modified by f7dd15a2 — all fixes are plan-text changes only (test fixtures, implementation snippets, task descriptions). The build agent will implement from this updated plan.

### Blocker Re-Verification

**Prior Blocker 1 — Bar injection always no-op: RESOLVED in plan.**

Verified: `session-brain.ts` still initializes `barMapAlignments = []` at lines 745 and 1220 and never pushes to it — `alignment` in every `EnrichedChunk` is still always `[]` in production. The plan now correctly documents this and the A1 implementation uses `chunk.bar_coverage[0]` for all notes in a chunk (chunk-level approximation). The A1 test fixture has `alignment: []` with assertion `note.bar === bar_coverage[0]`. The spec adds a follow-up note. This is an accurate and complete fix.

**Prior Blocker 2 — `p84` absent from production COHORT_TABLES: RESOLVED in plan.**

Verified: `COHORT_TABLES` in `teacher.ts` still has only `p25/p50/p75/p90` entries. The updated A1 plan now uses `p75` (not `p84`) for stddev: `stddev = max(0.01, p75 - p50)`. The test fixture is updated to `p25/p50/p75/p90` shape and the expected value is `0.15` (not `0.12`). Correct.

**Prior Blocker 3 — length-guard throws in B2/B3/B7: RESOLVED in plan.**

Verified: `tempo-stability-triage.ts` still has the throw at line 47 in source. The plan now explicitly instructs removing this guard in tasks B2, B3, and B7, and provides explicit "empty alignment → neutral, not throw" tests for all three. This is the correct fix; the build agent will implement it.

**Prior Blocker 4 — C4 test is a vacuous shape test: RESOLVED in plan.**

The updated C4 task now extracts `toPastDiagnosisRecord()` as an exported pure function from `session-brain.ts` and the test calls it directly (`import { toPastDiagnosisRecord } from './session-brain'`). Verified: `session-brain.ts` currently has exports at lines 71, 128, 178, 198, 210 — all are pure exported functions, confirming the pattern is established. The test now exercises real implementation code. Correct.

**Prior Blocker 5 — Two catalog tests break when articulationClarityCheck removed: RESOLVED in plan.**

Verified: `molecules/index.test.ts` still asserts `toHaveLength(8)` and names `articulation-clarity-check`. `readme-molecules.test.ts` still lists 8 molecules in `FINAL_MOLECULES`. The updated C2 task now explicitly instructs updating both tests (8→7, removing the name) and the README. Correct.

**Prior Concern (C1 overflow test vacuous): RESOLVED in plan.**

The updated C1 task extracts `buildPhase1UserMessage()` as an exported helper from `phase1.ts` and the test calls it with the 10-chunk fixture, asserting `length < 10000` and absence of `"midi_notes"`. Verified: `phase1.ts` does not currently export `buildPhase1UserMessage` — the build agent will add it. The test now exercises a real exported function from `phase1.ts`. Correct.

**Prior Concern (sequential fetchStudentBaseline calls): RESOLVED in plan.**

The updated A2 `resolveMoleculeContext` uses `Promise.all` for the 6 per-dimension baseline calls. Correct.

**Prior Concern (session_means query unbounded): RESOLVED in plan.**

The updated A1 `buildGroundedDigest` query adds `.orderBy(sql\`MAX(${observations.createdAt}) DESC\`).limit(10 * 6).groupBy(...)`. See new risk below.

---

### Remaining Issues

**[BLOCKER] (confidence: 9/10) — Drizzle 0.38 query chain in `buildGroundedDigest` has invalid method order: `.orderBy().limit().groupBy()` is not permitted.**

Verified: Drizzle 0.38's builder enforces a strict chaining order; `.groupBy()` must precede `.orderBy()` and `.limit()`. After `.limit()` the builder's TypeScript type excludes `.groupBy()`, so this will be a compile error when the builder writes the code. The correct order is `.where(...).groupBy(observations.sessionId, observations.dimension).orderBy(sql\`MAX(${observations.createdAt}) DESC\`).limit(10 * 6)`. Additionally, using `MAX(createdAt)` in an `ORDER BY` without it appearing in the `SELECT` list may produce a database error depending on the Postgres mode (strict `GROUP BY` conformance). The fix: move `.groupBy()` before `.orderBy()` and `.limit()`. This is a new issue introduced by the Concern 1 fix.

**[RISK] (confidence: 8/10) — `buildGroundedDigest`'s `past_diagnoses_grounded` uses unsafe type casts for `id` and `pieceId` that produce `undefined` until C3+C4 complete.**

The A1 implementation accesses `(r as PastDiagnosisRecord & { id: string }).id` and `(r as ... & { pieceId?: string | null }).pieceId`. The current `PastDiagnosisRecord` interface (`teacher.ts` lines 73–80) has neither field. The current `session-brain.ts` pastDiagnoses mapping (lines 1679–1686) omits both fields. So during any intermediate commit (A1 done, C3/C4 not yet done), production calls will produce `artifact_id: undefined` and `piece_id: null` for every past diagnosis. This is acceptable only if the build agent completes all tasks in one atomic session and doesn't ship partial commits to production — which matches the plan's pre-beta, local-first model. Flag for the builder: do not verify the past-diagnoses path until after C3+C4 are committed.

**[RISK] (confidence: 7/10) — `cross-modal-contradiction-check` (B7) perf_index vs array-index mismatch is still unaddressed in the plan.**

The existing `cross-modal-contradiction-check.ts` builds `alignMap` from `alignment.perf_index` but looks up by array position from `midi_notes.map((n, idx) => ...)`. After the refactor, `bundle.midi_notes` is bar-filtered, so positional `idx` no longer equals `perf_index`. The plan says "use `bundle.alignment` and `bundle.midi_notes` directly" but does not specify fixing this join. The timing contradiction arm will silently miss most notes. Non-blocking because the fallback is neutral (no throw), but the timing arm will be functionally dead. The builder should join on `note.bar` or carry `perf_index` through `GroundedNote`.

**[RISK] (confidence: 7/10) — `rubato-coaching` (B3) same perf_index vs positional-index mismatch.**

`rubato-coaching.ts` line 50 builds `perfNoteMap = new Map(i.midi_notes.map((n, idx) => [idx, n.onset_ms]))`. After the refactor, `i.midi_notes` = `bundle.midi_notes` (bar-filtered), so positional `idx` doesn't correspond to `perf_index`. Most `alignment.map(a => perfNoteMap.get(a.perf_index))` lookups return `undefined`, falling back to `a.expected_onset_ms` — which measures score timing, not performance. Degrades silently to neutral. Unaddressed in the plan. Non-blocking because alignment is always `[]` in production currently anyway.

**[OBS] — The A1 test mock for the DB query is `select().from().where().orderBy().limit().groupBy()` — consistent with the plan's implementation chain order. Once the Drizzle chain order is fixed (groupBy before orderBy+limit), the mock chain must also be updated to match. The mock is a stub and will work regardless of chain order, but keep them consistent to avoid confusion.**

**[OBS] — `fetchStudentBaseline.invoke()` returns `null` (not throws) when `n < 3`. The `resolveMoleculeContext` plan already handles this via the `if (session_means.length >= 3)` guard before calling it, with `throw` if it unexpectedly returns null despite n >= 3. This is correct: the null return path is guarded before invocation.**

---

### Presumption Inventory — Loop 2

| Assumption | Verdict | Reason |
|---|---|---|
| Drizzle 0.38 permits `.orderBy().limit().groupBy()` chain | RISKY | Drizzle enforces strict builder order; `groupBy` must precede `orderBy`+`limit` |
| `toPastDiagnosisRecord` can be exported from `session-brain.ts` (a DO class file) | SAFE | Verified: existing exports at lines 71, 128, 178, 198, 210 confirm the pattern |
| `buildPhase1UserMessage` export from `phase1.ts` won't create a circular import | SAFE | `phase1.ts` imports `gateway-client`, `middleware`, `route-model`, `types` — no digest module yet |
| All 5 prior blockers are fully resolved by the plan text changes | SAFE | Each verified above against actual source; plan text corrections are accurate |
| `PastDiagnosisRecord` type cast `(r as ... & { id: string }).id` is safe once C3+C4 complete | SAFE | C3 adds `id`+`pieceId` to the interface; C4 adds them to the mapping; sequencing is correct |
| `extractBarRangeSignals` is in the atoms `index.ts` and can be imported from there | SAFE | Verified: exported at atoms/index.ts line 38 |

---

### Summary — Loop 2

[BLOCKER] count: 1
[RISK]    count: 4
[QUESTION] count: 0

VERDICT: PROCEED_WITH_CAUTION — One remaining blocker: fix the Drizzle chain order in A1's `buildGroundedDigest` DB query from `.where().orderBy().limit().groupBy()` to `.where().groupBy().orderBy().limit()`. All 5 prior blockers are correctly resolved in the plan. Risks to monitor during build: (1) Drizzle chain fix must not re-break the mock; (2) B7 and B3 perf_index/positional-index mismatch will leave timing arms functionally dead but non-crashing; (3) intermediate commits leave `past_diagnoses_grounded.artifact_id = undefined` until C3+C4 complete — do not test this path until after C4 commit.
