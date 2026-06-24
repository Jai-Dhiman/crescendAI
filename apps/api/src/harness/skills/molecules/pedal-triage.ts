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
