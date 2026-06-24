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
      bar_coverage: (chunk.bar_coverage ?? [0, 0]) as [number, number],
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
  // Chain order: .where().groupBy().orderBy().limit() — Drizzle enforces this order.
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
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  }

  // past_diagnoses_grounded: reshape PastDiagnosisRecord → GroundedPastDiagnosis
  // NOTE: PastDiagnosisRecord does not yet have `id` or `pieceId` fields (added in C3/C4).
  // Cast to access them — they will be present at runtime once session-brain.ts is updated.
  const past_diagnoses_grounded: GroundedPastDiagnosis[] = input.pastDiagnoses.map((r) => ({
    artifact_id: (r as PastDiagnosisRecord & { id?: string }).id ?? '',
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
