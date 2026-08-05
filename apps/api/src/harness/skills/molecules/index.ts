import type { ToolDefinition } from "../../loop/types";
import { crossModalContradictionCheck } from "./cross-modal-contradiction-check";
import { dynamicRangeAudit } from "./dynamic-range-audit";
import { pedalTriage } from "./pedal-triage";
import { phrasingArcAnalysis } from "./phrasing-arc-analysis";
import { rubatoCoaching } from "./rubato-coaching";
import { tempoStabilityTriage } from "./tempo-stability-triage";
import { voicingDiagnosis } from "./voicing-diagnosis";
export const ALL_MOLECULES: ToolDefinition[] = [
	voicingDiagnosis,
	pedalTriage,
	rubatoCoaching,
	phrasingArcAnalysis,
	tempoStabilityTriage,
	dynamicRangeAudit,
	crossModalContradictionCheck,
];
