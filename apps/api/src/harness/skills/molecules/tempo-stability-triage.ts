import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeIoiCorrelation } from '../atoms/compute-ioi-correlation'
import { computeOnsetDrift } from '../atoms/compute-onset-drift'
import type { OnsetDrift } from '../atoms/compute-onset-drift'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchStudentBaseline } from '../atoms/fetch-student-baseline'
import type { Baseline } from '../atoms/fetch-student-baseline'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type TempoInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number; bar: number }[]
  session_means_timing: number[]
  piece_id: string; now_ms: number
}

export const tempoStabilityTriage: ToolDefinition = {
  name: 'tempo-stability-triage',
  description: 'Distinguishes tempo drift, intentional rubato, and loss of pulse by checking monotonic drift and IOI correlation.',
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
    const i = input as TempoInput
    if (i.midi_notes.length !== i.alignment.length) {
      throw new Error(`tempo-stability-triage: midi_notes length (${i.midi_notes.length}) does not match alignment length (${i.alignment.length})`)
    }
    // Use perf_index from alignment to look up note onset_ms — avoids array-position assumption
    const perfNoteMap = new Map(i.midi_notes.map((n, perfIdx) => [perfIdx, n.onset_ms]))
    const correlationNotes = i.alignment.map(a => ({
      onset_ms: perfNoteMap.get(a.perf_index) ?? a.expected_onset_ms,
      expected_onset_ms: a.expected_onset_ms,
    }))
    const r = await computeIoiCorrelation.invoke({ notes: correlationNotes }) as number | null
    const baseline = await fetchStudentBaseline.invoke({ dimension: 'timing', session_means: i.session_means_timing }) as Baseline | null
    if (!baseline) throw new Error('tempo-stability-triage: insufficient session history for timing baseline (need >= 3 sessions)')
    const z = await computeDimensionDelta.invoke({ dimension: 'timing', current: i.muq_scores[DIM.timing], baseline }) as number
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'timing', dimensions: ['timing'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Tempo is stable or within acceptable rubato range.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -1.0 || (r !== null && r >= 0.4)) return neutral
    const driftNotes = i.alignment.map(a => {
      const onset = perfNoteMap.get(a.perf_index)
      if (onset === undefined) throw new Error(`tempo-stability-triage: no note found for perf_index ${a.perf_index}`)
      return { onset_ms: onset, expected_onset_ms: a.expected_onset_ms }
    })
    const drift = await computeOnsetDrift.invoke({ notes: driftNotes }) as OnsetDrift[]
    if (drift.length === 0) return neutral
    const positiveCount = drift.filter(d => d.signed > 0).length
    const negativeCount = drift.filter(d => d.signed < 0).length
    const dominantFraction = Math.max(positiveCount, negativeCount) / drift.length
    if (dominantFraction < 0.8) return neutral
    const subtype: 'slowing' | 'rushing' = positiveCount > negativeCount ? 'slowing' : 'rushing'
    const finding = subtype === 'slowing'
      ? `The pulse slowed gradually across bars ${i.bar_range[0]}-${i.bar_range[1]}; the tempo is drifting under the beat.`
      : `The pulse rushed across bars ${i.bar_range[0]}-${i.bar_range[1]}; the tempo is running ahead of the beat.`
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'timing', dimensions: ['timing'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'high', finding_type: 'issue',
    })
  },
}
