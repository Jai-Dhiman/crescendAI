import type { GroundedDigest } from '../grounded-digest'

export function buildCompact10ChunkDigest(): GroundedDigest {
  const dims = ['dynamics', 'timing', 'pedaling', 'articulation', 'phrasing', 'interpretation'] as const
  const chunks = Array.from({ length: 10 }, (_, i) => ({
    chunk_id: `chunk:${i}`,
    bar_coverage: [i * 8 + 1, i * 8 + 8] as [number, number],
    muq_scores: [0.55, 0.48, 0.46, 0.54, 0.52, 0.51],
    midi_notes: Array.from({ length: 200 }, (_, j) => ({ pitch: 60, onset_ms: j * 100, duration_ms: 400, velocity: 70, bar: i * 8 + 1 })),
    pedal_cc: [],
    alignment: [],
  }))
  const lines = chunks.map((c) => `${c.chunk_id} bars ${c.bar_coverage[0]}-${c.bar_coverage[1]} muq=[${c.muq_scores.map(v => v.toFixed(2)).join(',')}]`)
  const compact_signal_summary = lines.join('\n')
  const session_means: Record<string, number[]> = {}
  const within_session_means: Record<string, number> = {}
  const cohort: Record<string, { mean: number; stddev: number }> = {}
  for (const d of dims) {
    session_means[d] = []
    within_session_means[d] = 0.52
    cohort[d] = { mean: 0.52, stddev: 0.10 }
  }
  return {
    chunks_adapted: chunks,
    mono_notes_per_bar: [],
    now_ms: Date.now(),
    cohort: cohort as GroundedDigest['cohort'],
    past_diagnoses_grounded: [],
    session_means: session_means as GroundedDigest['session_means'],
    within_session_means: within_session_means as GroundedDigest['within_session_means'],
    compact_signal_summary,
    piece_id: null,
  }
}
