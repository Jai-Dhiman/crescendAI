// resolve-molecule-context.ts
import { fetchStudentBaseline } from '../skills/atoms/fetch-student-baseline'
import type { Baseline } from '../skills/atoms/fetch-student-baseline'
import { extractBarRangeSignals } from '../skills/atoms/extract-bar-range-signals'
import type { SignalBundle } from '../skills/atoms/extract-bar-range-signals'
import type { GroundedDigest, GroundedPastDiagnosis, CohortStats } from './grounded-digest'
import { DIMENSIONS_6 } from './grounded-digest'

type Dim6 = (typeof DIMENSIONS_6)[number]

export type TieredBaseline = Record<Dim6, Baseline>

export type ResolvedMoleculeContext = {
  bundle: SignalBundle
  baseline: TieredBaseline
  cohort: CohortStats
  past_diagnoses: GroundedPastDiagnosis[]
  piece_id: string | null
  now_ms: number
}

export async function resolveMoleculeContext(
  digest: GroundedDigest,
  bar_range: [number, number] | null,
): Promise<ResolvedMoleculeContext> {
  const effectiveRange = bar_range ?? fullSessionRange(digest)
  const bundle = await extractBarRangeSignals.invoke({
    bar_range: effectiveRange,
    chunks: digest.chunks_adapted,
  }) as SignalBundle

  // The 6 per-dimension baseline computations run via Promise.all.
  // fetchStudentBaseline is a pure computation (no I/O), so parallelising avoids
  // 6 sequential microtask round-trips.
  const baselineEntries = await Promise.all(
    DIMENSIONS_6.map(async (dim) => {
      const session_means = digest.session_means[dim]
      if (session_means.length >= 3) {
        const result = await fetchStudentBaseline.invoke({ dimension: dim, session_means }) as Baseline | null
        if (result === null) {
          throw new Error(`resolveMoleculeContext: fetchStudentBaseline returned null despite n=${session_means.length} sessions for dim=${dim}`)
        }
        return [dim, result] as const
      } else {
        // Thin-history tier: synthesise from within_session_means
        const mean = digest.within_session_means[dim]
        return [dim, { dimension: dim, mean, stddev: Math.max(0.1, digest.cohort[dim].stddev), n_sessions: session_means.length }] as const
      }
    })
  )
  const baseline = Object.fromEntries(baselineEntries) as TieredBaseline

  return {
    bundle,
    baseline,
    cohort: digest.cohort,
    past_diagnoses: digest.past_diagnoses_grounded,
    piece_id: digest.piece_id,
    now_ms: digest.now_ms,
  }
}

function fullSessionRange(digest: GroundedDigest): [number, number] {
  let min = Infinity; let max = -Infinity
  for (const chunk of digest.chunks_adapted) {
    if (chunk.bar_coverage[0] < min) min = chunk.bar_coverage[0]
    if (chunk.bar_coverage[1] > max) max = chunk.bar_coverage[1]
  }
  return [isFinite(min) ? min : 0, isFinite(max) ? max : 0]
}
