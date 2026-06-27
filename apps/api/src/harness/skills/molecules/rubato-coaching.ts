// rubato-coaching.ts (after refactor — selectors-only + self-fetch via resolveMoleculeContext)
import type { ToolDefinition, PhaseContext } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeOnsetDrift } from '../atoms/compute-onset-drift'
import type { OnsetDrift } from '../atoms/compute-onset-drift'
import { computeIoiCorrelation } from '../atoms/compute-ioi-correlation'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { resolveMoleculeContext } from '../../loop/resolve-molecule-context'
import type { GroundedDigest } from '../../loop/grounded-digest'

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type RubatoSelectors = {
  bar_range: [number, number] | null
  scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]
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
    },
    required: ['scope', 'evidence_refs'],
  },
  invoke: async (input: unknown, ctx?: PhaseContext): Promise<DiagnosisArtifact> => {
    if (!ctx) throw new Error('rubato-coaching: ctx (PhaseContext with digest) is required')
    const i = input as RubatoSelectors
    const ctx_r = await resolveMoleculeContext(ctx.digest as unknown as GroundedDigest, i.bar_range ?? null)

    const muq_timing = ctx_r.bundle.muq_scores.length > 0
      ? ctx_r.bundle.muq_scores.reduce((s, row) => s + row[1], 0) / ctx_r.bundle.muq_scores.length
      : ctx_r.baseline.timing.mean

    const z = await computeDimensionDelta.invoke({
      dimension: 'timing',
      current: muq_timing,
      baseline: ctx_r.baseline.timing,
    }) as number

    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'timing', dimensions: ['timing', 'phrasing', 'interpretation'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range ?? null,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Timing is within baseline or rubato resolves cleanly.',
      confidence: 'low', finding_type: 'neutral',
    })

    // Guard: empty alignment means IOI/drift arms are data-limited — return neutral, not throw
    if (ctx_r.bundle.alignment.length === 0) return neutral

    const perfNoteMap = new Map(ctx_r.bundle.midi_notes.map((n, idx) => [idx, n.onset_ms]))
    const alignMap = new Map(ctx_r.bundle.alignment.map(a => [a.perf_index, a.expected_onset_ms]))

    const correlationNotes = ctx_r.bundle.alignment.map(a => ({
      onset_ms: perfNoteMap.get(a.perf_index) ?? a.expected_onset_ms ?? 0,
      expected_onset_ms: a.expected_onset_ms,
    }))

    const r = await computeIoiCorrelation.invoke({ notes: correlationNotes }) as number | null

    if (r === null) return neutral
    if (z > -0.8 || r >= 0.3) return neutral

    const alignedNotes = ctx_r.bundle.midi_notes
      .map((n, idx) => ({ onset_ms: n.onset_ms, expected_onset_ms: alignMap.get(idx) }))
      .filter((n): n is { onset_ms: number; expected_onset_ms: number } => n.expected_onset_ms !== null && n.expected_onset_ms !== undefined)

    const drift = await computeOnsetDrift.invoke({ notes: alignedNotes }) as OnsetDrift[]
    if (drift.length === 0) return neutral

    const lastSigned = drift[drift.length - 1].signed
    if (Math.abs(lastSigned) <= 50) {
      return DiagnosisArtifactSchema.parse({
        primary_dimension: 'timing', dimensions: ['timing', 'phrasing', 'interpretation'],
        severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range ?? null,
        evidence_refs: i.evidence_refs,
        one_sentence_finding: 'The phrase stretched and came back; rubato is well-shaped.',
        confidence: 'medium', finding_type: 'strength',
      })
    }

    const meanSigned = drift.reduce((s, d) => s + d.signed, 0) / drift.length
    const subtype = meanSigned < 0 ? 'rushed' : 'dragged'
    const barLabel = i.bar_range ? `bars ${i.bar_range[0]}-${i.bar_range[1]}` : 'this session'
    const finding = subtype === 'dragged'
      ? `The rubato through ${barLabel} stretched without coming back; the phrase loses its shape.`
      : `The rubato through ${barLabel} rushed without settling; the phrase ends too abruptly.`

    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'timing', dimensions: ['timing', 'phrasing', 'interpretation'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range ?? null,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'medium', finding_type: 'issue',
    })
  },
}
