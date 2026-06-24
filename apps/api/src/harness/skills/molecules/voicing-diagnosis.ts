// voicing-diagnosis.ts (after refactor — selectors-only + self-fetch via resolveMoleculeContext)
import type { ToolDefinition, PhaseContext } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchSimilarPastObservation } from '../atoms/fetch-similar-past-observation'
import type { PastObservation } from '../atoms/fetch-similar-past-observation'
import { resolveMoleculeContext } from '../../loop/resolve-molecule-context'
import type { GroundedDigest } from '../../loop/grounded-digest'

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type VoicingSelectors = {
  bar_range: [number, number] | null
  scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]
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
    },
    required: ['scope', 'evidence_refs'],
  },
  invoke: async (input: unknown, ctx?: PhaseContext): Promise<DiagnosisArtifact> => {
    if (!ctx) throw new Error('voicing-diagnosis: ctx (PhaseContext with digest) is required')
    const i = input as VoicingSelectors
    const ctx_r = await resolveMoleculeContext(ctx.digest as unknown as GroundedDigest, i.bar_range ?? null)

    const muq_dynamics = ctx_r.bundle.muq_scores.length > 0
      ? ctx_r.bundle.muq_scores.reduce((s, row) => s + row[0], 0) / ctx_r.bundle.muq_scores.length
      : ctx_r.baseline.dynamics.mean

    const z = await computeDimensionDelta.invoke({
      dimension: 'dynamics',
      current: muq_dynamics,
      baseline: ctx_r.baseline.dynamics,
    }) as number

    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'dynamics', dimensions: ['dynamics', 'phrasing'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range ?? null,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Voicing balance is within student baseline.',
      confidence: 'low', finding_type: 'neutral',
    })

    if (z > -1.0) return neutral

    const bars = [...new Set(ctx_r.bundle.midi_notes.map(n => n.bar))].sort((a, b) => (a ?? 0) - (b ?? 0))
    let flatCount = 0
    for (const bar of bars) {
      const p = projectVoices(ctx_r.bundle.midi_notes as { pitch: number; velocity: number; bar: number }[], bar as number)
      if (p && Math.abs(p.topMean - p.bassMean) < 5) flatCount++
    }
    if (bars.length === 0 || flatCount / bars.length < 0.6) return neutral

    const past = await fetchSimilarPastObservation.invoke({
      dimension: 'dynamics',
      piece_id: ctx_r.piece_id ?? '',
      bar_range: i.bar_range ?? null,
      past_diagnoses: ctx_r.past_diagnoses,
      now_ms: ctx_r.now_ms,
    }) as PastObservation | null

    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'dynamics', dimensions: ['dynamics', 'phrasing'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range ?? null,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Melody and accompaniment are voiced almost equally; the top line is not coming through.',
      confidence: past ? 'high' : 'medium',
      finding_type: 'issue',
    })
  },
}
