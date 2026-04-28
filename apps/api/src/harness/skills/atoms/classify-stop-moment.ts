import type { ToolDefinition } from '../../loop/types'

// Coefficients from apps/api/src/wasm/score-analysis/src/stop.rs
// Dimension order: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
const SCALER_MEAN = [0.5450, 0.4848, 0.4594, 0.5369, 0.5188, 0.5064]
const SCALER_STD  = [0.0689, 0.0388, 0.0791, 0.0154, 0.0186, 0.0555]
const WEIGHTS     = [-0.5266, 0.3681, -0.5483, 0.4884, 0.2427, -0.1541]
const BIAS        = 0.1147

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

export const classifyStopMoment: ToolDefinition = {
  name: 'classify-stop-moment',
  description:
    'Returns the probability [0,1] that a teacher would stop the student at this 15s chunk, given MuQ 6-dim scores. Input order: [dynamics, timing, pedaling, articulation, phrasing, interpretation]. Uses StandardScaler + logistic regression from the STOP classifier (LOVO CV AUC 0.649).',
  input_schema: {
    type: 'object',
    properties: {
      scores: {
        type: 'array',
        items: { type: 'number', minimum: 0, maximum: 1 },
        minItems: 6,
        maxItems: 6,
        description: 'MuQ 6-dim quality scores: [dynamics, timing, pedaling, articulation, phrasing, interpretation]',
      },
    },
    required: ['scores'],
  },
  invoke: async (input: unknown): Promise<number> => {
    const { scores } = input as { scores: number[] }
    if (!Array.isArray(scores) || scores.length !== 6) {
      throw new Error('classify-stop-moment: scores must be an array of exactly 6 numbers')
    }
    let logit = BIAS
    for (let i = 0; i < 6; i++) {
      const scaled = (scores[i] - SCALER_MEAN[i]) / SCALER_STD[i]
      logit += scaled * WEIGHTS[i]
    }
    return sigmoid(logit)
  },
}
