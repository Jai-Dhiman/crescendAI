import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeKeyOverlapRatio } from '../atoms/compute-key-overlap-ratio'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type ArticulationInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  mono_notes_per_bar: { bar: number; notes: { onset_ms: number; duration_ms: number }[] }[]
  score_articulation_per_bar: { bar: number; articulation: 'slur' | 'staccato' | 'detache' }[]
  cohort_table_articulation: { p: number; value: number }[]
  piece_id: string; now_ms: number
}

export const articulationClarityCheck: ToolDefinition = {
  name: 'articulation-clarity-check',
  description: 'Identifies execution mismatches between notated articulation (slurs vs staccato) and observed key-overlap behavior.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      mono_notes_per_bar: { type: 'array', items: { type: 'object' } },
      score_articulation_per_bar: { type: 'array', items: { type: 'object' } },
      cohort_table_articulation: { type: 'array', items: { type: 'object' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'mono_notes_per_bar', 'score_articulation_per_bar', 'cohort_table_articulation', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as ArticulationInput
    const p50 = i.cohort_table_articulation.find(e => e.p === 50)?.value ?? 0.52
    const p84 = i.cohort_table_articulation.find(e => e.p === 84)?.value ?? 0.60
    const cohortBaseline = { mean: p50, stddev: Math.max(0.01, p84 - p50) }
    const z = await computeDimensionDelta.invoke({ dimension: 'articulation', current: i.muq_scores[DIM.articulation], baseline: cohortBaseline }) as number
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'articulation', dimensions: ['articulation'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Articulation execution matches score markings.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -0.8) return neutral
    const articMap = new Map(i.score_articulation_per_bar.map(a => [a.bar, a.articulation]))
    let staccatoAsMiscount = 0
    let legatoAsMiscount = 0
    let totalBars = 0

    for (const barData of i.mono_notes_per_bar) {
      if (barData.notes.length < 3) continue
      totalBars++
      const ratio = await computeKeyOverlapRatio.invoke({ notes: barData.notes }) as number
      const scoreArt = articMap.get(barData.bar) ?? 'detache'
      if (scoreArt === 'staccato' && ratio > 0) staccatoAsMiscount++
      else if (scoreArt === 'slur' && ratio < 0) legatoAsMiscount++
    }

    const mismatchCount = staccatoAsMiscount + legatoAsMiscount
    if (totalBars === 0 || mismatchCount / totalBars < 0.5) return neutral
    const dominantMismatch: 'legato_as_staccato' | 'staccato_as_legato' =
      legatoAsMiscount > staccatoAsMiscount ? 'legato_as_staccato' : 'staccato_as_legato'
    const finding = dominantMismatch === 'staccato_as_legato'
      ? `The staccato bars ${i.bar_range[0]}-${i.bar_range[1]} are sustaining into each other; the notes are blurring rather than separating.`
      : `The slurred bars ${i.bar_range[0]}-${i.bar_range[1]} are detaching between notes; the legato line is breaking up.`
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'articulation', dimensions: ['articulation'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'high', finding_type: 'issue',
    })
  },
}
