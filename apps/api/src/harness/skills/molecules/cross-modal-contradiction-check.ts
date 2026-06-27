// cross-modal-contradiction-check.ts (after refactor — selectors-only + self-fetch via resolveMoleculeContext)
// Articulation arm REMOVED: requires score articulation per bar, not available from digest.
// Remaining 3 score-independent arms:
//   1. timing-drift-vs-MuQ: z>=+0.5 AND mean onset drift >80ms (empty alignment → skip, not throw)
//   2. pedal-ratio-vs-MuQ: z>=+0.5 AND overlap ratio <0.30 OR >0.85
//   3. dynamics-range-vs-MuQ: z>=+0.5 AND velocity range <25
import type { ToolDefinition, PhaseContext } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeOnsetDrift } from '../atoms/compute-onset-drift'
import type { OnsetDrift } from '../atoms/compute-onset-drift'
import { computePedalOverlapRatio } from '../atoms/compute-pedal-overlap-ratio'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { resolveMoleculeContext } from '../../loop/resolve-molecule-context'
import type { GroundedDigest } from '../../loop/grounded-digest'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

type CrossModalSelectors = {
  bar_range: [number, number] | null
  scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]
}

export const crossModalContradictionCheck: ToolDefinition = {
  name: 'cross-modal-contradiction-check',
  description: 'Flags cases where MuQ dimension scores and AMT-derived structural features disagree on a passage. Articulation arm removed (requires score data). Checks: timing-drift, pedal-ratio, dynamics-range.',
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
    if (!ctx) throw new Error('cross-modal-contradiction-check: ctx (PhaseContext with digest) is required')
    const i = input as CrossModalSelectors
    const ctx_r = await resolveMoleculeContext(ctx.digest as unknown as GroundedDigest, i.bar_range ?? null)

    const muq_scores = ctx_r.bundle.muq_scores.length > 0
      ? ctx_r.bundle.muq_scores[0]  // use first chunk's scores; all within bar_range
      : null

    async function dimZ(dimName: keyof typeof DIM): Promise<number> {
      const current = muq_scores ? muq_scores[DIM[dimName]] : ctx_r.cohort[dimName].mean
      return await computeDimensionDelta.invoke({
        dimension: dimName,
        current,
        baseline: ctx_r.cohort[dimName],
      }) as number
    }

    const contradictions: { dimension: string; delta: number; finding: string }[] = []

    // Arm 1: timing-drift-vs-MuQ: z>=+0.5 AND mean onset drift >80ms
    // Guard: empty alignment -> skip (not throw)
    const timingZ = await dimZ('timing')
    if (timingZ >= 0.5 && ctx_r.bundle.alignment.length >= 2) {
      const perfNoteMap = new Map(ctx_r.bundle.midi_notes.map((n, idx) => [idx, n.onset_ms]))
      const driftNotes = ctx_r.bundle.alignment
        .filter(a => a.expected_onset_ms !== null)
        .map(a => {
          const onset = perfNoteMap.get(a.perf_index)
          if (onset === undefined) return null
          return { onset_ms: onset, expected_onset_ms: a.expected_onset_ms as number }
        })
        .filter((n): n is { onset_ms: number; expected_onset_ms: number } => n !== null)
      if (driftNotes.length >= 2) {
        const drift = await computeOnsetDrift.invoke({ notes: driftNotes }) as OnsetDrift[]
        const meanDrift = drift.reduce((s, d) => s + d.drift_ms, 0) / drift.length
        if (meanDrift > 80) {
          contradictions.push({
            dimension: 'timing',
            delta: Math.abs(timingZ),
            finding: `MuQ rates timing clean here, but the onsets drifted ${Math.round(meanDrift)}ms on average.`,
          })
        }
      }
    }

    // Arm 2: pedal-ratio-vs-MuQ: z>=+0.5 AND overlap ratio <0.30 OR >0.85
    const pedalZ = await dimZ('pedaling')
    if (pedalZ >= 0.5) {
      const ratio = await computePedalOverlapRatio.invoke({
        notes: ctx_r.bundle.midi_notes,
        pedal_cc: ctx_r.bundle.pedal_cc,
      }) as number
      if (ratio < 0.30 || ratio > 0.85) {
        contradictions.push({
          dimension: 'pedaling',
          delta: Math.abs(pedalZ),
          finding: `MuQ rates pedaling clean here, but the pedal overlap ratio was ${(ratio * 100).toFixed(0)}% -- the model and the score disagree.`,
        })
      }
    }

    // Arm 3: dynamics-range-vs-MuQ: z>=+0.5 AND velocity range <25
    const dynamicsZ = await dimZ('dynamics')
    if (dynamicsZ >= 0.5 && ctx_r.bundle.midi_notes.length >= 2) {
      const vels = ctx_r.bundle.midi_notes.map(n => n.velocity)
      const velRange = Math.max(...vels) - Math.min(...vels)
      if (velRange < 25) {
        contradictions.push({
          dimension: 'dynamics',
          delta: Math.abs(dynamicsZ),
          finding: `MuQ rates dynamics clean here, but the velocity range across the passage is only ${velRange} -- compressed.`,
        })
      }
    }

    if (contradictions.length === 0) {
      return DiagnosisArtifactSchema.parse({
        primary_dimension: 'dynamics',
        dimensions: ['dynamics'],
        severity: 'minor', scope: i.scope, bar_range: i.bar_range ?? null,
        evidence_refs: i.evidence_refs,
        one_sentence_finding: 'No cross-modal contradictions detected.',
        confidence: 'high', finding_type: 'neutral',
      })
    }

    const winner = contradictions.reduce((best, c) => c.delta > best.delta ? c : best)
    const dims = [...new Set(contradictions.map(c => c.dimension))] as (keyof typeof DIM)[]
    return DiagnosisArtifactSchema.parse({
      primary_dimension: winner.dimension,
      dimensions: dims,
      severity: 'significant',
      scope: i.scope, bar_range: i.bar_range ?? null,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: winner.finding,
      confidence: 'high', finding_type: 'issue',
    })
  },
}
