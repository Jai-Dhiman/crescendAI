import type { Dimension } from "../lib/dims";
import { DIMS_6 } from "../lib/dims";
import type { DimScores } from "./scoring";
import type { BaselineState } from "./student-baseline";

export interface MarkCandidate {
	dimension: Dimension;
	taxonomy: "needs_work" | "strong";
	atSeconds: number;
}

export interface ProposeMarksInput {
	atSeconds: number;
	scores: DimScores;
	baseline: BaselineState | null;
	sessionSamples: Partial<Record<Dimension, readonly number[]>>;
	emittedDimensions: ReadonlySet<Dimension>;
	firePersistence: number;
}

export interface ProposeMarksResult {
	candidates: MarkCandidate[];
}

const MIN_SAMPLES_FOR_COLD_START_SPREAD = 3;
const COLD_START_BAND_MULTIPLE = 1.5;

function median(values: readonly number[]): number {
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 === 0
		? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2
		: (sorted[mid] ?? 0);
}

function medianAbsoluteDeviation(
	values: readonly number[],
	centre: number,
): number {
	return median(values.map((v) => Math.abs(v - centre)));
}

interface Band {
	center: number;
	halfWidth: number;
}

/**
 * Band for a dimension: the student's persisted BaselineState when
 * initialized, otherwise the within-session median +/- a robust spread —
 * the same cold-start-equals-fallback path as 02-pipeline.md section 7.
 * Returns null when there is not enough evidence to judge (uninitialized
 * baseline AND fewer than 3 session samples, or zero spread).
 */
function bandFor(
	dim: Dimension,
	baseline: BaselineState | null,
	samples: readonly number[],
): Band | null {
	const dimState = baseline?.dimensions[dim];
	if (dimState?.initialized) {
		return { center: dimState.longMean, halfWidth: dimState.halfWidth };
	}
	if (samples.length < MIN_SAMPLES_FOR_COLD_START_SPREAD) return null;
	const center = median(samples);
	const spread = medianAbsoluteDeviation(samples, center);
	if (spread <= 0) return null;
	return { center, halfWidth: COLD_START_BAND_MULTIPLE * spread };
}

function trailingOutOfBandCount(
	samples: readonly number[],
	band: Band,
): number {
	let count = 0;
	for (let i = samples.length - 1; i >= 0; i--) {
		const s = samples[i];
		if (s === undefined || Math.abs(s - band.center) <= band.halfWidth) break;
		count++;
	}
	return count;
}

/**
 * Pure per-chunk mark gate. No I/O, no clock. Requires
 * `firePersistence`-consecutive trailing out-of-band samples in a
 * dimension's session history, dedupes against `emittedDimensions`
 * (a dimension fires at most once per session), and yields at most one
 * candidate per call (DIMS_6 order breaks ties).
 */
export function proposeMarks(input: ProposeMarksInput): ProposeMarksResult {
	const candidates: MarkCandidate[] = [];
	for (const dim of DIMS_6) {
		if (candidates.length > 0) break;
		if (input.emittedDimensions.has(dim)) continue;
		const samples = input.sessionSamples[dim] ?? [];
		const band = bandFor(dim, input.baseline, samples);
		if (band === null) continue;
		const streak = trailingOutOfBandCount(samples, band);
		if (streak >= input.firePersistence) {
			const latest = samples[samples.length - 1] ?? input.scores[dim];
			candidates.push({
				dimension: dim,
				taxonomy: latest < band.center ? "needs_work" : "strong",
				atSeconds: input.atSeconds,
			});
		}
	}
	return { candidates };
}
