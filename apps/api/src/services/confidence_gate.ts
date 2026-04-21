import { DIMS_6 } from "../lib/dims";
import type { Dimension } from "../lib/dims";
import type { MuqConfidences, MuqScores } from "./inference";

const DEFAULT_SIGMA_THRESHOLD = 0.15;

export interface GateResult {
	surfaced: Partial<Record<Dimension, number>>;
	suppressed: Dimension[];
}

export function applyConfidenceGate(
	scores: MuqScores,
	confidences: MuqConfidences,
	threshold: number = DEFAULT_SIGMA_THRESHOLD,
): GateResult {
	const surfaced: Partial<Record<Dimension, number>> = {};
	const suppressed: Dimension[] = [];

	for (const dim of DIMS_6) {
		if (confidences[dim] < threshold) {
			surfaced[dim] = scores[dim];
		} else {
			suppressed.push(dim);
		}
	}

	return { surfaced, suppressed };
}
