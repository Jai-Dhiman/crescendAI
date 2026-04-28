import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeVelocityCurve } from '../atoms/compute-velocity-curve'
import type { VelocityCurve } from '../atoms/compute-velocity-curve'
import { computeOnsetDrift } from '../atoms/compute-onset-drift'
import type { OnsetDrift } from '../atoms/compute-onset-drift'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchReferencePercentile } from '../atoms/fetch-reference-percentile'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type PhrasingInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number | null; bar: number }[]
  cohort_table_phrasing: { p: number; value: number }[]
  piece_id: string; now_ms: number
}

export const phrasingArcAnalysis: ToolDefinition = {
  name: 'phrasing-arc-analysis',
  description: 'Assesses dynamic and timing arc shape across a complete phrase by detecting peak position and drift convergence.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      alignment: { type: 'array', items: { type: 'object' } },
      cohort_table_phrasing: { type: 'array', items: { type: 'object' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'alignment', 'cohort_table_phrasing', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as PhrasingInput
    const curve = await computeVelocityCurve.invoke({ bar_range: i.bar_range, notes: i.midi_notes }) as VelocityCurve[]
    const p50 = i.cohort_table_phrasing.find(e => e.p === 50)?.value ?? 0.52
    const p84 = i.cohort_table_phrasing.find(e => e.p === 84)?.value ?? 0.60
    const cohortBaseline = { mean: p50, stddev: Math.max(0.01, p84 - p50) }
    const z = await computeDimensionDelta.invoke({ dimension: 'phrasing', current: i.muq_scores[DIM.phrasing], baseline: cohortBaseline }) as number
    await fetchReferencePercentile.invoke({ dimension: 'phrasing', score: i.muq_scores[DIM.phrasing], cohort_table: i.cohort_table_phrasing })
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'phrasing', dimensions: ['phrasing', 'dynamics'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Phrase shape is within cohort norms.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -0.8) return neutral
    if (curve.length === 0) return neutral
    const maxVel = Math.max(...curve.map(c => c.mean_velocity))
    const peakIndex = curve.findIndex(c => c.mean_velocity === maxVel)
    const nearPeak = curve.filter(c => maxVel - c.mean_velocity < 5)
    const flatOrMulti = peakIndex === 0 || peakIndex === curve.length - 1 || nearPeak.length > 1
    const alignMap = new Map(i.alignment.map(a => [a.perf_index, a.expected_onset_ms]))
    const alignedNotes = i.midi_notes
      .map((n, idx) => ({ onset_ms: n.onset_ms, expected_onset_ms: alignMap.get(idx) }))
      .filter((n): n is { onset_ms: number; expected_onset_ms: number } => n.expected_onset_ms !== null && n.expected_onset_ms !== undefined)
    const drift = await computeOnsetDrift.invoke({ notes: alignedNotes }) as OnsetDrift[]
    const lastDrift = drift.length > 0 ? Math.abs(drift[drift.length - 1].signed) : 0
    if (!flatOrMulti && lastDrift <= 50) {
      return DiagnosisArtifactSchema.parse({
        primary_dimension: 'phrasing', dimensions: ['phrasing', 'dynamics'],
        severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
        evidence_refs: i.evidence_refs,
        one_sentence_finding: 'The phrase has a clear arc with a well-placed peak.',
        confidence: 'medium', finding_type: 'strength',
      })
    }
    const peakBar = curve[peakIndex]?.bar ?? i.bar_range[0]
    const finding = `The phrase peaks at bar ${peakBar} instead of the expected middle; the climax of the line is arriving too early.`
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'phrasing', dimensions: ['phrasing', 'dynamics'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'medium', finding_type: 'issue',
    })
  },
}
