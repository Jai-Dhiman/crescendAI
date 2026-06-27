import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeIoiCorrelation } from '../atoms/compute-ioi-correlation'
import { computeOnsetDrift } from '../atoms/compute-onset-drift'
import type { OnsetDrift } from '../atoms/compute-onset-drift'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { resolveMoleculeContext } from '../../loop/resolve-molecule-context'
import type { GroundedDigest } from '../../loop/grounded-digest'

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type TempoSelectors = {
  bar_range: [number, number] | null
  scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]
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
    },
    required: ['scope', 'evidence_refs'],
  },
  invoke: async (input: unknown, ctx?: import('../../loop/types').PhaseContext): Promise<DiagnosisArtifact> => {
    if (!ctx) throw new Error('tempo-stability-triage: ctx (PhaseContext with digest) is required')
    const i = input as TempoSelectors
    const ctx_r = await resolveMoleculeContext(ctx.digest as unknown as GroundedDigest, i.bar_range ?? null)

    const muq_timing = ctx_r.bundle.muq_scores.length > 0
      ? ctx_r.bundle.muq_scores.reduce((s, row) => s + row[1], 0) / ctx_r.bundle.muq_scores.length
      : ctx_r.baseline.timing.mean

    const z = await computeDimensionDelta.invoke({
      dimension: 'timing',
      current: muq_timing,
      baseline: ctx_r.baseline.timing,
    }) as number

    const rangeLabel = i.bar_range ? `bars ${i.bar_range[0]}-${i.bar_range[1]}` : 'this session'

    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'timing', dimensions: ['timing'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range ?? null,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Tempo is stable or within acceptable rubato range.',
      confidence: 'low', finding_type: 'neutral',
    })

    // Guard: empty alignment means IOI/drift arms are data-limited — return neutral, not throw
    if (ctx_r.bundle.alignment.length === 0) return neutral

    // Build perf note map keyed by perf_index for O(1) lookup
    const perfNoteMap = new Map(ctx_r.bundle.midi_notes.map((n, perfIdx) => [perfIdx, n.onset_ms]))

    const correlationNotes = ctx_r.bundle.alignment.map(a => ({
      onset_ms: perfNoteMap.get(a.perf_index) ?? a.expected_onset_ms ?? 0,
      expected_onset_ms: a.expected_onset_ms,
    }))

    const r = await computeIoiCorrelation.invoke({ notes: correlationNotes }) as number | null

    if (z > -1.0 || (r !== null && r >= 0.4)) return neutral

    const driftNotes = ctx_r.bundle.alignment
      .filter(a => a.expected_onset_ms !== null)
      .map(a => {
        const onset = perfNoteMap.get(a.perf_index)
        if (onset === undefined) return null
        return { onset_ms: onset, expected_onset_ms: a.expected_onset_ms as number }
      })
      .filter((n): n is { onset_ms: number; expected_onset_ms: number } => n !== null)

    const drift = await computeOnsetDrift.invoke({ notes: driftNotes }) as OnsetDrift[]
    if (drift.length === 0) return neutral

    const positiveCount = drift.filter(d => d.signed > 0).length
    const negativeCount = drift.filter(d => d.signed < 0).length
    const dominantFraction = Math.max(positiveCount, negativeCount) / drift.length
    if (dominantFraction < 0.8) return neutral

    const subtype: 'slowing' | 'rushing' = positiveCount > negativeCount ? 'slowing' : 'rushing'
    const finding = subtype === 'slowing'
      ? `The pulse slowed gradually across ${rangeLabel}; the tempo is drifting under the beat.`
      : `The pulse rushed across ${rangeLabel}; the tempo is running ahead of the beat.`

    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'timing', dimensions: ['timing'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range ?? null,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'high', finding_type: 'issue',
    })
  },
}
