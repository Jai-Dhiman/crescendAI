import type { Dimension } from "../lib/dims";
import type { PerfNote, PerfPedalEvent } from "./inference";

export interface MpmFeatureSet {
	windowIndex: number;
	localTempoBpm: number;
	tempoDeviationPct: number;
	velocityMean: number;
	velocityRange: number;
	velocitySlope: number;
	articulationRatio: number;
	pedalOverlapFraction: number;
}

export interface MpmFeatureInput {
	windowIndex: number;
	perfNotes: readonly PerfNote[];
	perfPedal: readonly PerfPedalEvent[];
	bars?: readonly [number, number];
}

function zeroSet(windowIndex: number): MpmFeatureSet {
	return {
		windowIndex,
		localTempoBpm: 0,
		tempoDeviationPct: 0,
		velocityMean: 0,
		velocityRange: 0,
		velocitySlope: 0,
		articulationRatio: 0,
		pedalOverlapFraction: 0,
	};
}

/**
 * Deterministic per-window MPM-style features over transcribed notes. The
 * `bars` field is accepted but unused today — the follower issue adds
 * per-bar granularity through this same field without an interface change.
 */
export function extractMpmFeatures(input: MpmFeatureInput): MpmFeatureSet[] {
	const { windowIndex, perfNotes } = input;
	if (perfNotes.length < 2) {
		return [zeroSet(windowIndex)];
	}
	return [zeroSet(windowIndex)]; // extended in Task 5
}

export function evidenceFor(_set: MpmFeatureSet, dimension: Dimension): string {
	return `${dimension} placeholder`; // implemented in Task 5
}
