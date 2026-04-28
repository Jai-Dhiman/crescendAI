import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computePedalOverlapRatio } from '../atoms/compute-pedal-overlap-ratio'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchStudentBaseline } from '../atoms/fetch-student-baseline'
import type { Baseline } from '../atoms/fetch-student-baseline'
import { fetchSimilarPastObservation } from '../atoms/fetch-similar-past-observation'
import type { PastObservation } from '../atoms/fetch-similar-past-observation'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type PedalInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  pedal_cc: { time_ms: number; value: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number | null; bar: number }[]
  harmony_changes: { bar: number; time_ms: number }[]
  session_means_pedaling: number[]
  past_diagnoses: { artifact_id: string; session_id: string; created_at: number; primary_dimension: string; bar_range: [number,number]|null; piece_id: string }[]
  piece_id: string; now_ms: number
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
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      pedal_cc: { type: 'array', items: { type: 'object' } },
      alignment: { type: 'array', items: { type: 'object' } },
      harmony_changes: { type: 'array', items: { type: 'object' } },
      session_means_pedaling: { type: 'array', items: { type: 'number' } },
      past_diagnoses: { type: 'array', items: { type: 'object' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'pedal_cc', 'alignment', 'harmony_changes', 'session_means_pedaling', 'past_diagnoses', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as PedalInput
    const ratio = await computePedalOverlapRatio.invoke({ notes: i.midi_notes, pedal_cc: i.pedal_cc }) as number
    const baseline = await fetchStudentBaseline.invoke({ dimension: 'pedaling', session_means: i.session_means_pedaling }) as Baseline | null
    if (!baseline) throw new Error('pedal-triage: insufficient session history for pedaling baseline (need >= 3 sessions)')
    const z = await computeDimensionDelta.invoke({ dimension: 'pedaling', current: i.muq_scores[DIM.pedaling], baseline }) as number
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'pedaling', dimensions: ['pedaling'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Pedaling is within student baseline.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -1.0) return neutral
    let finding: string
    if (ratio > 0.85) {
      finding = `Over-pedaled through bars ${i.bar_range[0]}-${i.bar_range[1]}; the harmonies are blurring into one wash.`
    } else if (ratio < 0.30) {
      finding = `Under-pedaled through bars ${i.bar_range[0]}-${i.bar_range[1]}; the tone sounds dry and disconnected.`
    } else {
      finding = `Pedal not released at harmony changes in bars ${i.bar_range[0]}-${i.bar_range[1]}; notes from adjacent harmonies are blurring.`
    }
    const past = await fetchSimilarPastObservation.invoke({ dimension: 'pedaling', piece_id: i.piece_id, bar_range: i.bar_range, past_diagnoses: i.past_diagnoses, now_ms: i.now_ms }) as PastObservation | null
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
