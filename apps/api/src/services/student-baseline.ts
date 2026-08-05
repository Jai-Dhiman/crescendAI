import { z } from "zod";
import { DIMS_6, type Dimension } from "../lib/dims";

// ---------------------------------------------------------------------------
// student-baseline — the single gate deciding whether a deviation in a
// student's playing is worth marking. See docs/specs/2026-08-05-student-
// baseline-gate-design.md for the full design rationale.
// ---------------------------------------------------------------------------

export type Lifecycle = "absent" | "active" | "improving" | "resolved";

export interface SessionSamples {
	/** ISO 8601 timestamp for this session. */
	timestamp: string;
	/** Raw per-dimension sample scores observed during this session. */
	scores: Partial<Record<Dimension, readonly number[]>>;
}

export interface BaselineConfig {
	shortHalfLifeSessions: number;
	longHalfLifeSessions: number;
	minBandSdFraction: number;
	firePersistence: number;
	improvingPersistence: number;
	retirePersistence: number;
	promotionDistinctWeeks: number;
	maxWithinSessionContribution: number;
	minSamplesForSpread: number;
	deviantSampleMultiple: number;
	confidenceProvisionalUpdates: number;
	confidenceEstablishedUpdates: number;
}

export const DEFAULT_BASELINE_CONFIG: BaselineConfig = {
	shortHalfLifeSessions: 4,
	longHalfLifeSessions: 20,
	minBandSdFraction: 0.2,
	firePersistence: 3,
	improvingPersistence: 2,
	retirePersistence: 3,
	promotionDistinctWeeks: 2,
	maxWithinSessionContribution: 3,
	minSamplesForSpread: 3,
	deviantSampleMultiple: 1.5,
	confidenceProvisionalUpdates: 3,
	confidenceEstablishedUpdates: 8,
};

const DimensionStateSchema = z.object({
	lifecycle: z.enum(["absent", "active", "improving", "resolved"]),
	longMean: z.number(),
	longSd: z.number(),
	shortMean: z.number(),
	noiseFloor: z.number(),
	halfWidth: z.number(),
	consecutiveOutOfBand: z.number().int().min(0),
	consecutiveInBand: z.number().int().min(0),
	promoted: z.boolean(),
	evidenceWeeks: z.array(z.string()),
	initialized: z.boolean(),
	updateCount: z.number().int().min(0),
	// Never gates -- see docs/specs/2026-08-05-student-baseline-gate-design.md's
	// "Divergence from 03-memory-system.md" section. Purely a display hint so
	// the teacher can frame early marks as exploratory prose.
	confidence: z.enum(["exploratory", "provisional", "established"]),
});

// An explicit object (one key per DIMS_6 entry), not z.record: z.record infers
// values as possibly-undefined on access, which would force every caller and
// every internal read to null-check a key that this module always populates.
export const BaselineStateSchema = z.object({
	lastSessionTimestamp: z.string().nullable(),
	dimensions: z.object({
		dynamics: DimensionStateSchema,
		timing: DimensionStateSchema,
		pedaling: DimensionStateSchema,
		articulation: DimensionStateSchema,
		phrasing: DimensionStateSchema,
		interpretation: DimensionStateSchema,
	}),
});

export type BaselineState = z.infer<typeof BaselineStateSchema>;
export type DimensionBaselineState = z.infer<typeof DimensionStateSchema>;

/** A fresh baseline: every dimension absent, no evidence folded in yet. */
export function initialBaselineState(): BaselineState {
	const dimensions = {} as Record<Dimension, DimensionBaselineState>;
	for (const dim of DIMS_6) {
		dimensions[dim] = {
			lifecycle: "absent",
			longMean: 0,
			longSd: 0,
			shortMean: 0,
			noiseFloor: 0,
			halfWidth: 0,
			consecutiveOutOfBand: 0,
			consecutiveInBand: 0,
			promoted: false,
			evidenceWeeks: [],
			initialized: false,
			updateCount: 0,
			confidence: "exploratory",
		};
	}
	return { lastSessionTimestamp: null, dimensions };
}

/**
 * Pure fold: (state, session) -> state. No clock, no randomness, no I/O.
 * NOTE: this is a deliberately minimal first cut -- it establishes the call
 * shape and return type only. It does not yet validate input (Tasks 2-5) or
 * fold any evidence (Tasks 6+); every dimension simply passes through
 * unchanged.
 */
function validateSession(state: BaselineState, session: SessionSamples): void {
	const timestampMs = Date.parse(session.timestamp);
	if (Number.isNaN(timestampMs)) {
		throw new Error(
			`updateBaseline: unparseable timestamp "${session.timestamp}"`,
		);
	}
	if (state.lastSessionTimestamp !== null) {
		const lastMs = Date.parse(state.lastSessionTimestamp);
		if (timestampMs < lastMs) {
			throw new Error(
				`updateBaseline: session timestamp ${session.timestamp} precedes last folded session ${state.lastSessionTimestamp}`,
			);
		}
	}
	for (const [dimension, samples] of Object.entries(session.scores)) {
		if (!(DIMS_6 as readonly string[]).includes(dimension)) {
			throw new Error(`updateBaseline: unknown dimension "${dimension}"`);
		}
		for (const score of samples ?? []) {
			if (!Number.isFinite(score)) {
				throw new Error(
					`updateBaseline: non-finite score ${score} for dimension "${dimension}"`,
				);
			}
		}
	}
}

function median(values: readonly number[]): number {
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 === 0
		? (sorted[mid - 1] + sorted[mid]) / 2
		: sorted[mid];
}

/** Median absolute deviation — robust to the minority-outlier case this gate targets. */
function medianAbsoluteDeviation(
	values: readonly number[],
	centre: number,
): number {
	return median(values.map((v) => Math.abs(v - centre)));
}

/**
 * NOTE: deliberately crude for this task -- any deviant sample fires
 * immediately, ignoring persistence. Task 7 replaces this with the real
 * FIRE_PERSISTENCE-threshold counter mechanism.
 */
function foldDimension(
	prior: DimensionBaselineState,
	samples: readonly number[],
	config: BaselineConfig,
): DimensionBaselineState {
	const sessionCentre = median(samples);
	let withinSessionDeviants = 0;
	if (samples.length >= config.minSamplesForSpread) {
		const mad = medianAbsoluteDeviation(samples, sessionCentre);
		if (mad > 0) {
			const threshold = config.deviantSampleMultiple * mad;
			for (const s of samples) {
				if (Math.abs(s - sessionCentre) > threshold) withinSessionDeviants += 1;
			}
		}
	}
	const lifecycle = withinSessionDeviants > 0 ? "active" : prior.lifecycle;
	return { ...prior, lifecycle };
}

export function updateBaseline(
	state: BaselineState,
	session: SessionSamples,
	config: BaselineConfig = DEFAULT_BASELINE_CONFIG,
): BaselineState {
	validateSession(state, session);
	const dimensions = { ...state.dimensions };
	for (const [dimension, samples] of Object.entries(session.scores)) {
		if (!samples || samples.length === 0) continue;
		const dim = dimension as Dimension;
		dimensions[dim] = foldDimension(dimensions[dim], samples, config);
	}
	return { lastSessionTimestamp: session.timestamp, dimensions };
}
