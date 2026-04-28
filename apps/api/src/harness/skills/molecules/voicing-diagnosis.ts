import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeVelocityCurve } from '../atoms/compute-velocity-curve'
import type { VelocityCurve } from '../atoms/compute-velocity-curve'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchStudentBaseline } from '../atoms/fetch-student-baseline'
import type { Baseline } from '../atoms/fetch-student-baseline'
import { fetchReferencePercentile } from '../atoms/fetch-reference-percentile'
import { fetchSimilarPastObservation } from '../atoms/fetch-similar-past-observation'
import type { PastObservation } from '../atoms/fetch-similar-past-observation'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type VoicingInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  session_means_dynamics: number[]
  cohort_table_dynamics: { p: number; value: number }[]
  past_diagnoses: { artifact_id: string; session_id: string; created_at: number; primary_dimension: string; bar_range: [number,number]|null; piece_id: string }[]
  piece_id: string; now_ms: number
}

function projectVoices(notes: { pitch: number; velocity: number; bar: number }[], bar: number) {
  const bn = notes.filter(n => n.bar === bar)
  if (bn.length === 0) return null
  const sorted = [...bn].sort((a, b) => b.pitch - a.pitch)
  const k = Math.max(1, Math.floor(sorted.length * 0.25))
  const topMean = sorted.slice(0, k).reduce((s, n) => s + n.velocity, 0) / k
  const bassMean = sorted.slice(-k).reduce((s, n) => s + n.velocity, 0) / k
  return { topMean, bassMean }
}

export const voicingDiagnosis: ToolDefinition = {
  name: 'voicing-diagnosis',
  description: 'Diagnoses melody/accompaniment voicing imbalance in homophonic textures. Returns DiagnosisArtifact with primary_dimension "dynamics".',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      session_means_dynamics: { type: 'array', items: { type: 'number' } },
      cohort_table_dynamics: { type: 'array', items: { type: 'object' } },
      past_diagnoses: { type: 'array', items: { type: 'object' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'session_means_dynamics', 'cohort_table_dynamics', 'past_diagnoses', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as VoicingInput
    const baseline = await fetchStudentBaseline.invoke({ dimension: 'dynamics', session_means: i.session_means_dynamics }) as Baseline | null
    if (!baseline) throw new Error('voicing-diagnosis: insufficient session history for dynamics baseline (need >= 3 sessions)')
    const z = await computeDimensionDelta.invoke({ dimension: 'dynamics', current: i.muq_scores[DIM.dynamics], baseline }) as number
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'dynamics', dimensions: ['dynamics', 'phrasing'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Voicing balance is within student baseline.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -1.0) return neutral
    const bars = [...new Set(i.midi_notes.map(n => n.bar))].sort((a, b) => a - b)
    let flatCount = 0
    for (const bar of bars) {
      const p = projectVoices(i.midi_notes, bar)
      if (p && Math.abs(p.topMean - p.bassMean) < 5) flatCount++
    }
    if (bars.length === 0 || flatCount / bars.length < 0.6) return neutral
    await fetchReferencePercentile.invoke({ dimension: 'dynamics', score: i.muq_scores[DIM.dynamics], cohort_table: i.cohort_table_dynamics })
    const past = await fetchSimilarPastObservation.invoke({ dimension: 'dynamics', piece_id: i.piece_id, bar_range: i.bar_range, past_diagnoses: i.past_diagnoses, now_ms: i.now_ms }) as PastObservation | null
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'dynamics', dimensions: ['dynamics', 'phrasing'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Melody and accompaniment are voiced almost equally; the top line is not coming through.',
      confidence: past ? 'high' : 'medium',
      finding_type: 'issue',
    })
  },
}
