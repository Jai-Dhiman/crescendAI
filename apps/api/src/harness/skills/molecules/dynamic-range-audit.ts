import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeVelocityCurve } from '../atoms/compute-velocity-curve'
import type { VelocityCurve } from '../atoms/compute-velocity-curve'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchStudentBaseline } from '../atoms/fetch-student-baseline'
import type { Baseline } from '../atoms/fetch-student-baseline'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type DynamicInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  score_marking_type: 'wide' | 'medium' | 'narrow' | 'none'
  session_means_dynamics: number[]; cohort_table_dynamics: { p: number; value: number }[]
  piece_id: string; now_ms: number
}

export const dynamicRangeAudit: ToolDefinition = {
  name: 'dynamic-range-audit',
  description: 'Compares velocity range used in performance against the dynamic range asked for by the score.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      score_marking_type: { type: 'string', enum: ['wide', 'medium', 'narrow', 'none'] },
      session_means_dynamics: { type: 'array', items: { type: 'number' } },
      cohort_table_dynamics: { type: 'array', items: { type: 'object' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'score_marking_type', 'session_means_dynamics', 'cohort_table_dynamics', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as DynamicInput
    const curve = await computeVelocityCurve.invoke({ bar_range: i.bar_range, notes: i.midi_notes }) as VelocityCurve[]
    const baseline = await fetchStudentBaseline.invoke({ dimension: 'dynamics', session_means: i.session_means_dynamics }) as Baseline | null
    if (!baseline) throw new Error('dynamic-range-audit: insufficient session history for dynamics baseline (need >= 3 sessions)')
    const z = await computeDimensionDelta.invoke({ dimension: 'dynamics', current: i.muq_scores[DIM.dynamics], baseline }) as number
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'dynamics', dimensions: ['dynamics'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Dynamic range is adequate for the score markings.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -0.8 || i.score_marking_type === 'none') return neutral
    const maxMean = Math.max(...curve.map(c => c.mean_velocity))
    const minMean = Math.min(...curve.map(c => c.mean_velocity))
    const observedRange = maxMean - minMean
    if (observedRange >= 30) return neutral
    const finding = `The dynamic range across bars ${i.bar_range[0]}-${i.bar_range[1]} is only ${Math.round(observedRange)} velocity points; the score asks for much more contrast here.`
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'dynamics', dimensions: ['dynamics'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'high', finding_type: 'issue',
    })
  },
}
