import type { ToolDefinition } from '../../loop/types'
import { type DiagnosisArtifact, DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import { z } from 'zod'

const SEVERITY_RANK: Record<string, number> = { significant: 3, moderate: 2, minor: 1 }
const CONFIDENCE_RANK: Record<string, number> = { high: 3, medium: 2, low: 1 }
const DIMENSION_PRIORITY: Record<string, number> = {
  pedaling: 6, timing: 5, dynamics: 4, phrasing: 3, articulation: 2, interpretation: 1,
}

export const prioritizeDiagnoses: ToolDefinition = {
  name: 'prioritize-diagnoses',
  description:
    'Ranks DiagnosisArtifacts by (severity_rank DESC, confidence_rank DESC, dimension_priority DESC). Diagnoses with finding_type "strength" sort to the end regardless of severity.',
  input_schema: {
    type: 'object',
    properties: {
      diagnoses: {
        type: 'array',
        items: { type: 'object' },
        description: 'DiagnosisArtifact objects to rank.',
      },
    },
    required: ['diagnoses'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact[]> => {
    const { diagnoses } = input as { diagnoses: unknown[] }
    // parse() validates and returns new objects; extra fields not in schema are stripped
    const parsed = z.array(DiagnosisArtifactSchema).parse(diagnoses)
    return [...parsed].sort((a, b) => {
      const aStrength = a.finding_type === 'strength' ? 1 : 0
      const bStrength = b.finding_type === 'strength' ? 1 : 0
      if (aStrength !== bStrength) return aStrength - bStrength
      const aSev = SEVERITY_RANK[a.severity] ?? 0
      const bSev = SEVERITY_RANK[b.severity] ?? 0
      if (aSev !== bSev) return bSev - aSev
      const aConf = CONFIDENCE_RANK[a.confidence] ?? 0
      const bConf = CONFIDENCE_RANK[b.confidence] ?? 0
      if (aConf !== bConf) return bConf - aConf
      const aDim = DIMENSION_PRIORITY[a.primary_dimension] ?? 0
      const bDim = DIMENSION_PRIORITY[b.primary_dimension] ?? 0
      return bDim - aDim
    })
  },
}
