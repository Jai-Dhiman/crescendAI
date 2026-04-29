import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeOnsetDrift } from '../atoms/compute-onset-drift'
import type { OnsetDrift } from '../atoms/compute-onset-drift'
import { computePedalOverlapRatio } from '../atoms/compute-pedal-overlap-ratio'
import { computeKeyOverlapRatio } from '../atoms/compute-key-overlap-ratio'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

type CrossModalInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]
  muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  pedal_cc: { time_ms: number; value: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number | null; bar: number }[]
  mono_notes_per_bar: { bar: number; notes: { onset_ms: number; duration_ms: number }[] }[]
  score_articulation_per_bar: { bar: number; articulation: 'slur' | 'staccato' | 'detache' }[]
  cohort_baselines: { [dim: string]: { mean: number; stddev: number } }
  piece_id: string; now_ms: number
}

export const crossModalContradictionCheck: ToolDefinition = {
  name: 'cross-modal-contradiction-check',
  description: 'Flags cases where MuQ dimension scores and AMT-derived structural features disagree on a passage.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      pedal_cc: { type: 'array', items: { type: 'object' } },
      alignment: { type: 'array', items: { type: 'object' } },
      mono_notes_per_bar: { type: 'array', items: { type: 'object' } },
      score_articulation_per_bar: { type: 'array', items: { type: 'object' } },
      cohort_baselines: { type: 'object' },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'pedal_cc', 'alignment', 'mono_notes_per_bar', 'score_articulation_per_bar', 'cohort_baselines', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as CrossModalInput

    async function dimZ(dimName: string): Promise<number> {
      if (!(dimName in DIM)) return 0
      const bl = i.cohort_baselines[dimName]
      if (!bl) throw new Error(`cross-modal-contradiction-check: missing cohort_baseline for dimension "${dimName}"`)
      return await computeDimensionDelta.invoke({ dimension: dimName, current: i.muq_scores[DIM[dimName as keyof typeof DIM]], baseline: bl }) as number
    }

    const contradictions: { dimension: string; delta: number; finding: string }[] = []

    // timing pair: z >= +0.5 AND mean onset drift > 80ms
    const timingZ = await dimZ('timing')
    if (timingZ >= 0.5) {
      const alignMap = new Map(i.alignment.map(a => [a.perf_index, a.expected_onset_ms]))
      const driftNotes = i.midi_notes
        .map((n, idx) => ({ onset_ms: n.onset_ms, expected_onset_ms: alignMap.get(idx) }))
        .filter((n): n is { onset_ms: number; expected_onset_ms: number } => n.expected_onset_ms !== null)
      if (driftNotes.length >= 2) {
        const drift = await computeOnsetDrift.invoke({ notes: driftNotes }) as OnsetDrift[]
        const meanDrift = drift.reduce((s, d) => s + d.drift_ms, 0) / drift.length
        if (meanDrift > 80) {
          contradictions.push({ dimension: 'timing', delta: Math.abs(timingZ), finding: `MuQ rates timing clean here, but the onsets drifted ${Math.round(meanDrift)}ms on average.` })
        }
      }
    }

    // pedaling pair: z >= +0.5 AND overlap ratio < 0.30 OR > 0.85
    const pedalZ = await dimZ('pedaling')
    if (pedalZ >= 0.5) {
      const ratio = await computePedalOverlapRatio.invoke({ notes: i.midi_notes, pedal_cc: i.pedal_cc }) as number
      if (ratio < 0.30 || ratio > 0.85) {
        contradictions.push({ dimension: 'pedaling', delta: Math.abs(pedalZ), finding: `MuQ rates pedaling clean here, but the pedal overlap ratio was ${(ratio * 100).toFixed(0)}% -- the model and the score disagree.` })
      }
    }

    // articulation pair: z >= +0.5 AND key overlap direction opposes score in >= 50% bars
    const articulationZ = await dimZ('articulation')
    if (articulationZ >= 0.5 && i.mono_notes_per_bar.length >= 2) {
      const articMap = new Map(i.score_articulation_per_bar.map(a => [a.bar, a.articulation]))
      let artMismatch = 0; let artTotal = 0
      for (const bd of i.mono_notes_per_bar) {
        if (bd.notes.length < 3) continue
        artTotal++
        const ratio = await computeKeyOverlapRatio.invoke({ notes: bd.notes }) as number
        const sa = articMap.get(bd.bar) ?? 'detache'
        if ((sa === 'staccato' && ratio >= 0) || (sa === 'slur' && ratio <= 0)) artMismatch++
      }
      if (artTotal > 0 && artMismatch / artTotal >= 0.5) {
        contradictions.push({ dimension: 'articulation', delta: Math.abs(articulationZ), finding: `MuQ rates articulation clean here, but the key-overlap direction contradicts the score markings in ${artMismatch}/${artTotal} bars.` })
      }
    }

    // dynamics pair: z >= +0.5 AND velocity range < 25
    const dynamicsZ = await dimZ('dynamics')
    if (dynamicsZ >= 0.5 && i.midi_notes.length >= 2) {
      const vels = i.midi_notes.map(n => n.velocity)
      const velRange = Math.max(...vels) - Math.min(...vels)
      if (velRange < 25) {
        contradictions.push({ dimension: 'dynamics', delta: Math.abs(dynamicsZ), finding: `MuQ rates dynamics clean here, but the velocity range across the passage is only ${velRange} -- compressed.` })
      }
    }

    if (contradictions.length === 0) {
      return DiagnosisArtifactSchema.parse({
        primary_dimension: 'dynamics',
        dimensions: ['dynamics'],
        severity: 'minor', scope: i.scope, bar_range: i.bar_range,
        evidence_refs: i.evidence_refs,
        one_sentence_finding: 'No cross-modal contradictions detected.',
        confidence: 'high', finding_type: 'neutral',
      })
    }

    const winner = contradictions.reduce((best, c) => c.delta > best.delta ? c : best)
    const dims = contradictions.map(c => c.dimension) as (keyof typeof DIM)[]
    return DiagnosisArtifactSchema.parse({
      primary_dimension: winner.dimension,
      dimensions: [...new Set(dims)],
      severity: 'significant',
      scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: winner.finding,
      confidence: 'high', finding_type: 'issue',
    })
  },
}
