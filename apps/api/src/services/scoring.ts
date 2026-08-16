import type { PerfNote, PerfPedalEvent } from "./inference";

export interface DimScores {
	dynamics: number;
	timing: number;
	pedaling: number;
	articulation: number;
	phrasing: number;
	interpretation: number;
}

function clamp01(x: number): number {
	if (Number.isNaN(x)) return 0.5;
	return Math.max(0, Math.min(1, x));
}

function mean(values: readonly number[]): number {
	if (values.length === 0) return 0;
	return values.reduce((a, b) => a + b, 0) / values.length;
}

function stdev(values: readonly number[], centre: number): number {
	if (values.length === 0) return 0;
	return Math.sqrt(mean(values.map((v) => (v - centre) ** 2)));
}

/**
 * Deterministic pure-TS mock 6-dim scorer. Behind the exact interface the
 * future MoonBeam container call will implement (issue #162 spec) — callers
 * cannot tell the difference. No network, no randomness, no clock.
 */
export function scoreChunk(
	perfNotes: readonly PerfNote[],
	perfPedal: readonly PerfPedalEvent[],
): DimScores {
	if (perfNotes.length === 0) {
		return {
			dynamics: 0.5,
			timing: 0.5,
			pedaling: 0.5,
			articulation: 0.5,
			phrasing: 0.5,
			interpretation: 0.5,
		};
	}

	const sorted = [...perfNotes].sort((a, b) => a.onset - b.onset);
	const velocities = sorted.map((n) => n.velocity);
	const dynamics = clamp01(mean(velocities) / 127);

	const iois: number[] = [];
	for (let i = 1; i < sorted.length; i++) {
		const prev = sorted[i - 1];
		const cur = sorted[i];
		if (prev && cur) iois.push(cur.onset - prev.onset);
	}
	const ioiMean = mean(iois);
	const ioiStd = stdev(iois, ioiMean);
	const timingCv = ioiMean > 0 ? ioiStd / ioiMean : 0;
	const timing = clamp01(1 - timingCv);

	const pedalOn = perfPedal.filter((p) => p.value >= 64).length;
	const pedaling = clamp01(
		perfPedal.length > 0 ? pedalOn / perfPedal.length : 0.5,
	);

	const articulationRatios: number[] = [];
	for (let i = 1; i < sorted.length; i++) {
		const ioi = iois[i - 1];
		const note = sorted[i - 1];
		if (ioi !== undefined && ioi > 0 && note) {
			articulationRatios.push((note.offset - note.onset) / ioi);
		}
	}
	const articulation = clamp01(
		articulationRatios.length > 0 ? mean(articulationRatios) : 0.5,
	);

	const velocityRange =
		velocities.length > 0
			? Math.max(...velocities) - Math.min(...velocities)
			: 0;
	const phrasing = clamp01(velocityRange / 127);

	const interpretation = clamp01(
		mean([dynamics, timing, pedaling, articulation, phrasing]),
	);

	return { dynamics, timing, pedaling, articulation, phrasing, interpretation };
}
