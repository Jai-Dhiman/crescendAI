// dynamic-range-audit.ts (after refactor — selectors-only + self-fetch via resolveMoleculeContext)
// score_marking_type REMOVED: not available from digest. The molecule now compares observed
// velocity range against cohort/baseline only. Previously the score_marking_type==='none' branch
// returned neutral unconditionally; that branch is dropped — the baseline+range check is
// sufficient to distinguish issues without score metadata.
import type { ToolDefinition, PhaseContext } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeVelocityCurve } from '../atoms/compute-velocity-curve'
import type { VelocityCurve } from '../atoms/compute-velocity-curve'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { resolveMoleculeContext } from '../../loop/resolve-molecule-context'
import type { GroundedDigest } from '../../loop/grounded-digest'

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type DynamicSelectors = {
  bar_range: [number, number] | null
  scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]
}

export const dynamicRangeAudit: ToolDefinition = {
  name: 'dynamic-range-audit',
  description: 'Compares velocity range used in performance against cohort/baseline dynamics. score_marking_type is not available and is not used.',
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
    if (!ctx) throw new Error('dynamic-range-audit: ctx (PhaseContext with digest) is required')
    const i = input as DynamicSelectors
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
      primary_dimension: 'dynamics', dimensions: ['dynamics'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range ?? null,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Dynamic range is adequate.',
      confidence: 'low', finding_type: 'neutral',
    })

    if (z > -0.8) return neutral

    // Determine the effective bar_range for the velocity curve call
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

    const maxMean = Math.max(...curve.map(c => c.mean_velocity))
    const minMean = Math.min(...curve.map(c => c.mean_velocity))
    const observedRange = maxMean - minMean

    if (observedRange >= 30) return neutral

    const barLabel = i.bar_range ? `bars ${i.bar_range[0]}-${i.bar_range[1]}` : 'this session'
    const finding = `The dynamic range across ${barLabel} is only ${Math.round(observedRange)} velocity points; the playing is too compressed.`

    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'dynamics', dimensions: ['dynamics'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range ?? null,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'high', finding_type: 'issue',
    })
  },
}
