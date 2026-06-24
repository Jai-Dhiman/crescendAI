import type { ToolDefinition } from '../../loop/types'
import { voicingDiagnosis } from './voicing-diagnosis'
import { pedalTriage } from './pedal-triage'
import { rubatoCoaching } from './rubato-coaching'
import { phrasingArcAnalysis } from './phrasing-arc-analysis'
import { tempoStabilityTriage } from './tempo-stability-triage'
import { dynamicRangeAudit } from './dynamic-range-audit'
import { crossModalContradictionCheck } from './cross-modal-contradiction-check'
export const ALL_MOLECULES: ToolDefinition[] = [
  voicingDiagnosis,
  pedalTriage,
  rubatoCoaching,
  phrasingArcAnalysis,
  tempoStabilityTriage,
  dynamicRangeAudit,
  crossModalContradictionCheck,
]
