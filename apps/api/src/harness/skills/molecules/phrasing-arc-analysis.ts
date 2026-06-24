// phrasing-arc-analysis.ts (after refactor — selectors-only + self-fetch via resolveMoleculeContext)
// Cohort phrasing stats come from rc.cohort.phrasing (mean/stddev). Empty alignment -> neutral guard applied.
import type { ToolDefinition, PhaseContext } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeVelocityCurve } from '../atoms/compute-velocity-curve'
import type { VelocityCurve } from '../atoms/compute-velocity-curve'
import { computeOnsetDrift } from '../atoms/compute-onset-drift'
import type { OnsetDrift } from '../atoms/compute-onset-drift'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { resolveMoleculeContext } from '../../loop/resolve-molecule-context'
import type { GroundedDigest } from '../../loop/grounded-digest'

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type PhrasingSelectors = {
  bar_range: [number, number] | null
  scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]
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
    },
    required: ['scope', 'evidence_refs'],
  },
  invoke: async (input: unknown, ctx?: PhaseContext): Promise<DiagnosisArtifact> => {
    if (!ctx) throw new Error('phrasing-arc-analysis: ctx (PhaseContext with digest) is required')
    const i = input as PhrasingSelectors
    const ctx_r = await resolveMoleculeContext(ctx.digest as unknown as GroundedDigest, i.bar_range ?? null)

    const muq_phrasing = ctx_r.bundle.muq_scores.length > 0
      ? ctx_r.bundle.muq_scores.reduce((s, row) => s + row[4], 0) / ctx_r.bundle.muq_scores.length
      : ctx_r.cohort.phrasing.mean

    // Use cohort phrasing stats as the comparison baseline
    const cohortBaseline = ctx_r.cohort.phrasing
    const z = await computeDimensionDelta.invoke({
      dimension: 'phrasing',
      current: muq_phrasing,
      baseline: cohortBaseline,
    }) as number

    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'phrasing', dimensions: ['phrasing', 'dynamics'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range ?? null,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Phrase shape is within cohort norms.',
      confidence: 'low', finding_type: 'neutral',
    })

    if (z > -0.8) return neutral

    const effectiveBarRange = i.bar_range ?? (
      ctx_r.bundle.midi_notes.length > 0
        ? [
            Math.min(...ctx_r.bundle.midi_notes.map(n => n.bar ?? 0)),
            Math.max(...ctx_r.bundle.midi_notes.map(n => n.bar ?? 0)),
          ] as [number, number]
        : [0, 0] as [number, number]
    )

    const curve = await computeVelocityCurve.invoke({
      bar_range: effectiveBarRange,
      notes: ctx_r.bundle.midi_notes,
    }) as VelocityCurve[]

    if (curve.length === 0) return neutral

    const maxVel = Math.max(...curve.map(c => c.mean_velocity))
    const peakIndex = curve.findIndex(c => c.mean_velocity === maxVel)
    const nearPeak = curve.filter(c => maxVel - c.mean_velocity < 5)
    const flatOrMulti = peakIndex === 0 || peakIndex === curve.length - 1 || nearPeak.length > 1

    // Guard: empty alignment means timing drift arm is data-limited — skip drift check
    let lastDrift = 0
    if (ctx_r.bundle.alignment.length > 0) {
      const alignMap = new Map(ctx_r.bundle.alignment.map(a => [a.perf_index, a.expected_onset_ms]))
      const alignedNotes = ctx_r.bundle.midi_notes
        .map((n, idx) => ({ onset_ms: n.onset_ms, expected_onset_ms: alignMap.get(idx) }))
        .filter((n): n is { onset_ms: number; expected_onset_ms: number } =>
          n.expected_onset_ms !== null && n.expected_onset_ms !== undefined)
      const drift = await computeOnsetDrift.invoke({ notes: alignedNotes }) as OnsetDrift[]
      lastDrift = drift.length > 0 ? Math.abs(drift[drift.length - 1].signed) : 0
    }

    if (!flatOrMulti && lastDrift <= 50) {
      return DiagnosisArtifactSchema.parse({
        primary_dimension: 'phrasing', dimensions: ['phrasing', 'dynamics'],
        severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range ?? null,
        evidence_refs: i.evidence_refs,
        one_sentence_finding: 'The phrase has a clear arc with a well-placed peak.',
        confidence: 'medium', finding_type: 'strength',
      })
    }

    const peakBar = curve[peakIndex]?.bar ?? (i.bar_range ? i.bar_range[0] : 0)
    let finding: string
    if (peakIndex === curve.length - 1) {
      finding = `The phrase peaks at bar ${peakBar} at the very end; the climax arrives too late, leaving no room for the line to resolve.`
    } else {
      finding = `The phrase peaks at bar ${peakBar} instead of the expected middle; the climax of the line is arriving too early.`
    }

    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'phrasing', dimensions: ['phrasing', 'dynamics'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range ?? null,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'medium', finding_type: 'issue',
    })
  },
}
