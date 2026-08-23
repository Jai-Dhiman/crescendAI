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

function mean(values: readonly number[]): number {
	if (values.length === 0) return 0;
	return values.reduce((sum, v) => sum + v, 0) / values.length;
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
	const { windowIndex, perfNotes, perfPedal } = input;
	if (perfNotes.length < 2) {
		return [zeroSet(windowIndex)];
	}

	const sorted = [...perfNotes].sort((a, b) => a.onset - b.onset);
	const iois: number[] = [];
	for (let i = 1; i < sorted.length; i++) {
		const prev = sorted[i - 1];
		const cur = sorted[i];
		if (prev && cur) iois.push(cur.onset - prev.onset);
	}
	const ioiMean = mean(iois);
	const localTempoBpm = ioiMean > 0 ? 60 / ioiMean : 0;

	const mid = Math.floor(iois.length / 2);
	const firstHalf = iois.slice(0, mid);
	const secondHalf = iois.slice(mid);
	const firstBpmMean = mean(firstHalf);
	const secondBpmMean = mean(secondHalf);
	const firstBpm = firstBpmMean > 0 ? 60 / firstBpmMean : 0;
	const secondBpm = secondBpmMean > 0 ? 60 / secondBpmMean : 0;
	const tempoDeviationPct =
		firstBpm > 0 ? ((secondBpm - firstBpm) / firstBpm) * 100 : 0;

	const velocities = sorted.map((n) => n.velocity);
	const velocityMean = mean(velocities);
	const velocityRange = Math.max(...velocities) - Math.min(...velocities);
	const velMid = Math.floor(velocities.length / 2);
	const velocityFirst = mean(velocities.slice(0, velMid));
	const velocitySecond = mean(velocities.slice(velMid));
	const velocitySlope = velocitySecond - velocityFirst;

	const articulationRatios: number[] = [];
	for (let i = 1; i < sorted.length; i++) {
		const ioi = iois[i - 1];
		const note = sorted[i - 1];
		if (ioi !== undefined && ioi > 0 && note) {
			articulationRatios.push((note.offset - note.onset) / ioi);
		}
	}
	const articulationRatio = mean(articulationRatios);

	const pedalOn = perfPedal.filter((p) => p.value >= 64).length;
	const pedalOverlapFraction =
		perfPedal.length > 0 ? pedalOn / perfPedal.length : 0;

	return [
		{
			windowIndex,
			localTempoBpm,
			tempoDeviationPct,
			velocityMean,
			velocityRange,
			velocitySlope,
			articulationRatio,
			pedalOverlapFraction,
		},
	];
}

export function evidenceFor(set: MpmFeatureSet, dimension: Dimension): string {
	switch (dimension) {
		case "timing": {
			const direction = set.tempoDeviationPct >= 0 ? "up" : "down";
			return `local tempo ${direction} ${Math.abs(Math.round(set.tempoDeviationPct))}%, uneven note lengths`;
		}
		case "dynamics":
			return `velocity range ${Math.round(set.velocityRange)}, mean ${Math.round(set.velocityMean)}`;
		case "pedaling":
			return `pedal overlap ${Math.round(set.pedalOverlapFraction * 100)}%`;
		case "articulation":
			return `articulation ratio ${set.articulationRatio.toFixed(2)}`;
		case "phrasing": {
			const direction = set.velocitySlope >= 0 ? "rising" : "falling";
			return `velocity slope ${direction} ${Math.abs(Math.round(set.velocitySlope))}`;
		}
		case "interpretation":
			return `tempo ${Math.round(set.localTempoBpm)} bpm, velocity mean ${Math.round(set.velocityMean)}`;
	}
}
