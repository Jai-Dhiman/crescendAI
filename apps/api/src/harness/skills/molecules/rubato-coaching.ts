import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeOnsetDrift } from '../atoms/compute-onset-drift'
import type { OnsetDrift } from '../atoms/compute-onset-drift'
import { computeIoiCorrelation } from '../atoms/compute-ioi-correlation'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchStudentBaseline } from '../atoms/fetch-student-baseline'
import type { Baseline } from '../atoms/fetch-student-baseline'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type RubatoInput = {
  bar_range: [number, number]
  scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]
  muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number | null; bar: number }[]
  session_means_timing: number[]
  piece_id: string
  now_ms: number
}

export const rubatoCoaching: ToolDefinition = {
  name: 'rubato-coaching',
  description: 'Distinguishes intentional returned rubato from uncompensated drift using IOI correlation and net phrase-end drift.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      alignment: { type: 'array', items: { type: 'object' } },
      session_means_timing: { type: 'array', items: { type: 'number' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'alignment', 'session_means_timing', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as RubatoInput
    const alignMap = new Map(i.alignment.map(a => [a.perf_index, a.expected_onset_ms]))
    const correlationNotes = i.midi_notes.map((n, idx) => ({ onset_ms: n.onset_ms, expected_onset_ms: alignMap.get(idx) ?? null }))
    const r = await computeIoiCorrelation.invoke({ notes: correlationNotes }) as number | null
    const baseline = await fetchStudentBaseline.invoke({ dimension: 'timing', session_means: i.session_means_timing }) as Baseline | null
    if (!baseline) throw new Error('rubato-coaching: insufficient session history for timing baseline (need >= 3 sessions)')
    const z = await computeDimensionDelta.invoke({ dimension: 'timing', current: i.muq_scores[DIM.timing], baseline }) as number
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'timing', dimensions: ['timing', 'phrasing', 'interpretation'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Timing is within baseline or rubato resolves cleanly.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -0.8 || (r !== null && r >= 0.3)) return neutral
    const alignedNotes = i.midi_notes
      .map((n, idx) => ({ onset_ms: n.onset_ms, expected_onset_ms: alignMap.get(idx) }))
      .filter((n): n is { onset_ms: number; expected_onset_ms: number } => n.expected_onset_ms !== null)
    const drift = await computeOnsetDrift.invoke({ notes: alignedNotes }) as OnsetDrift[]
    if (drift.length === 0) return neutral
    const lastSigned = drift[drift.length - 1].signed
    if (Math.abs(lastSigned) <= 50) {
      return DiagnosisArtifactSchema.parse({
        primary_dimension: 'timing', dimensions: ['timing', 'phrasing', 'interpretation'],
        severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
        evidence_refs: i.evidence_refs,
        one_sentence_finding: 'The phrase stretched and came back; rubato is well-shaped.',
        confidence: 'medium', finding_type: 'strength',
      })
    }
    const meanSigned = drift.reduce((s, d) => s + d.signed, 0) / drift.length
    const subtype = meanSigned < 0 ? 'rushed' : 'dragged'
    const finding = subtype === 'dragged'
      ? `The rubato through bars ${i.bar_range[0]}-${i.bar_range[1]} stretched without coming back; the phrase loses its shape.`
      : `The rubato through bars ${i.bar_range[0]}-${i.bar_range[1]} rushed without settling; the phrase ends too abruptly.`
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'timing', dimensions: ['timing', 'phrasing', 'interpretation'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'medium', finding_type: 'issue',
    })
  },
}
