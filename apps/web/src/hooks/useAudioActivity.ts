import { useEffect, useRef, useState } from "react";
import { createLogger } from "../lib/logger";

const log = createLogger("AudioActivity");

/** Spectral energy threshold (0-1). Piano audio should comfortably exceed this. */
export const ENERGY_THRESHOLD = 0.04;

/** Frames above threshold before onset triggers (~150ms at 60fps) */
const ONSET_FRAMES = 4;

/** Milliseconds below threshold before offset triggers.
 * Piano has natural 2-3s gaps between phrases -- 5s avoids false offsets. */
const OFFSET_MS = 5000;

/**
 * Spectral gate thresholds.
 *
 * Gate logic (OR -- generous to maintain high recall):
 *   isPianoLike = energy > ENERGY_THRESHOLD AND (flatness < FLATNESS_MAX OR centroid < CENTROID_MAX)
 *
 * Tuned from Python harness on 40-clip labeled dataset (20 piano, 5 quiet piano,
 * 5 speech, 5 noise, 5 mixed). Two-feature gate achieved F1=0.91, recall=1.0,
 * precision=0.83.
 *
 * Spectral flatness: geometric/arithmetic mean of linear magnitude spectrum.
 *   Piano (tonal): ~0.001-0.20, Speech: ~0.01-0.07, Noise (broadband): ~0.3-0.8
 *   Computed on linear magnitudes from getFloatFrequencyData (dB -> linear conversion).
 *
 * Spectral centroid: weighted mean frequency of spectrum.
 *   Piano: ~300-2400 Hz, Speech: ~1600-3200 Hz, Fan/AC: ~5000+ Hz
 *   Centroid transfers near-perfectly between librosa and AnalyserNode (rho=0.99).
 */
const FLATNESS_MAX = 0.15;
const CENTROID_MAX_HZ = 2500;

export interface AudioActivityState {
	isPlaying: boolean;
	energy: number;
}

/** Compute spectral flatness from dB-scaled frequency data.
 * Converts dB -> linear magnitude, then computes geometric/arithmetic mean ratio. */
function computeSpectralFlatness(dbData: Float32Array, binCount: number): number {
	let logSum = 0;
	let linSum = 0;
	let count = 0;
	for (let i = 0; i < binCount; i++) {
		const db = dbData[i];
		if (db < -100) continue;
		const linear = Math.pow(10, db / 20);
		logSum += Math.log(linear + 1e-20);
		linSum += linear;
		count++;
	}
	if (count === 0 || linSum < 1e-20) return 1.0;
	const geoMean = Math.exp(logSum / count);
	const arithMean = linSum / count;
	return geoMean / arithMean;
}

/** Compute spectral centroid from dB-scaled frequency data.
 * Returns weighted mean frequency in Hz. */
function computeSpectralCentroid(
	dbData: Float32Array,
	binCount: number,
	sampleRate: number,
): number {
	let weightedSum = 0;
	let totalWeight = 0;
	const nyquist = sampleRate / 2;
	for (let i = 0; i < binCount; i++) {
		const db = dbData[i];
		const linear = Math.pow(10, db / 20);
		const freq = (i / binCount) * nyquist;
		weightedSum += freq * linear;
		totalWeight += linear;
	}
	if (totalWeight < 1e-20) return 0;
	return weightedSum / totalWeight;
}

export function useAudioActivity(
	analyserNodeRef: React.RefObject<AnalyserNode | null>,
): AudioActivityState {
	const [isPlaying, setIsPlaying] = useState(false);
	const [energy, setEnergy] = useState(0);

	const isPlayingRef = useRef(false);
	const onsetCountRef = useRef(0);
	const offsetStartRef = useRef<number | null>(null);
	const rafRef = useRef<number>(0);
	const lastLogTimeRef = useRef(0);
	const gatePassRef = useRef(0);
	const gateRejectRef = useRef(0);

	useEffect(() => {
		const byteArrayRef: { current: Uint8Array<ArrayBuffer> | null } = { current: null };
		const floatArrayRef: { current: Float32Array<ArrayBuffer> | null } = { current: null };

		function tick(timestamp: number) {
			rafRef.current = requestAnimationFrame(tick);

			const analyser = analyserNodeRef.current;
			if (!analyser) return;

			const binCount = analyser.frequencyBinCount;

			if (!byteArrayRef.current || byteArrayRef.current.length !== binCount) {
				byteArrayRef.current = new Uint8Array(binCount) as Uint8Array<ArrayBuffer>;
			}
			if (!floatArrayRef.current || floatArrayRef.current.length !== binCount) {
				floatArrayRef.current = new Float32Array(binCount) as Float32Array<ArrayBuffer>;
			}

			// Compute spectral energy
			analyser.getByteFrequencyData(byteArrayRef.current);
			let sum = 0;
			for (let i = 0; i < byteArrayRef.current.length; i++) {
				sum += byteArrayRef.current[i];
			}
			const rawEnergy = sum / (byteArrayRef.current.length * 255);

			setEnergy(rawEnergy);

			// Gate decision: energy must pass first, then spectral features
			let aboveThreshold: boolean;
			if (rawEnergy > ENERGY_THRESHOLD) {
				analyser.getFloatFrequencyData(floatArrayRef.current);
				const flatness = computeSpectralFlatness(floatArrayRef.current, binCount);
				const centroid = computeSpectralCentroid(
					floatArrayRef.current,
					binCount,
					analyser.context.sampleRate,
				);
				// OR logic: either low flatness (tonal) or low centroid (piano range) -> pass
				const isPianoLike = flatness < FLATNESS_MAX || centroid < CENTROID_MAX_HZ;
				aboveThreshold = isPianoLike;

				if (isPianoLike) {
					gatePassRef.current++;
				} else {
					gateRejectRef.current++;
				}
			} else {
				aboveThreshold = false;
			}

			if (!isPlayingRef.current) {
				if (aboveThreshold) {
					onsetCountRef.current++;
					if (onsetCountRef.current >= ONSET_FRAMES) {
						isPlayingRef.current = true;
						setIsPlaying(true);
						onsetCountRef.current = 0;
						offsetStartRef.current = null;
						log.log(
							`PLAYING detected (pass=${gatePassRef.current} reject=${gateRejectRef.current})`,
						);
					}
				} else {
					onsetCountRef.current = 0;
					if (timestamp - lastLogTimeRef.current > 1000) {
						lastLogTimeRef.current = timestamp;
					}
				}
			} else {
				if (!aboveThreshold) {
					if (offsetStartRef.current === null) {
						offsetStartRef.current = timestamp;
					}
					const silenceDuration = timestamp - offsetStartRef.current;
					if (silenceDuration >= OFFSET_MS) {
						isPlayingRef.current = false;
						setIsPlaying(false);
						offsetStartRef.current = null;
						log.log(
							`SILENCE detected (${(silenceDuration / 1000).toFixed(1)}s, pass=${gatePassRef.current} reject=${gateRejectRef.current})`,
						);
						gatePassRef.current = 0;
						gateRejectRef.current = 0;
					}
				} else {
					offsetStartRef.current = null;
				}
			}
		}

		rafRef.current = requestAnimationFrame(tick);

		return () => {
			cancelAnimationFrame(rafRef.current);
		};
	}, [analyserNodeRef]);

	return { isPlaying, energy };
}
