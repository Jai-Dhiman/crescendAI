import type { ToolDefinition } from "../../loop/types";
import { alignPerformanceToScore } from "./align-performance-to-score";
import { classifyStopMoment } from "./classify-stop-moment";
import { computeDimensionDelta } from "./compute-dimension-delta";
import { computeIoiCorrelation } from "./compute-ioi-correlation";
import { computeKeyOverlapRatio } from "./compute-key-overlap-ratio";
import { computeOnsetDrift } from "./compute-onset-drift";
import { computePedalOverlapRatio } from "./compute-pedal-overlap-ratio";
import { computeVelocityCurve } from "./compute-velocity-curve";
import { detectPassageRepetition } from "./detect-passage-repetition";
import { extractBarRangeSignals } from "./extract-bar-range-signals";
import { fetchReferencePercentile } from "./fetch-reference-percentile";
import { fetchSessionHistory } from "./fetch-session-history";
import { fetchSimilarPastObservation } from "./fetch-similar-past-observation";
import { fetchStudentBaseline } from "./fetch-student-baseline";
import { prioritizeDiagnoses } from "./prioritize-diagnoses";

export type { Alignment } from "./align-performance-to-score";
export type { OnsetDrift } from "./compute-onset-drift";
export type { CcEvent } from "./compute-pedal-overlap-ratio";
export type { VelocityCurve } from "./compute-velocity-curve";
export type { MidiNote, SignalBundle } from "./extract-bar-range-signals";
export type { RepetitionEntry } from "./detect-passage-repetition";
export type { Baseline } from "./fetch-student-baseline";
export type { SessionHistory } from "./fetch-session-history";
export type { PastObservation } from "./fetch-similar-past-observation";

export const ALL_ATOMS: ToolDefinition[] = [
	alignPerformanceToScore,
	classifyStopMoment,
	computeDimensionDelta,
	computeIoiCorrelation,
	computeKeyOverlapRatio,
	computeOnsetDrift,
	computePedalOverlapRatio,
	computeVelocityCurve,
	detectPassageRepetition,
	extractBarRangeSignals,
	fetchReferencePercentile,
	fetchSessionHistory,
	fetchSimilarPastObservation,
	fetchStudentBaseline,
	prioritizeDiagnoses,
];
