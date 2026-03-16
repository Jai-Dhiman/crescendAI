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

export interface AudioActivityState {
	isPlaying: boolean;
	energy: number;
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

	useEffect(() => {
		const dataArrayRef: { current: Uint8Array<ArrayBuffer> | null } = { current: null };

		function tick(timestamp: number) {
			rafRef.current = requestAnimationFrame(tick);

			const analyser = analyserNodeRef.current;
			if (!analyser) return;

			// Lazily allocate data array
			if (!dataArrayRef.current || dataArrayRef.current.length !== analyser.frequencyBinCount) {
				dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount) as Uint8Array<ArrayBuffer>;
			}

			// Compute spectral energy
			analyser.getByteFrequencyData(dataArrayRef.current);
			let sum = 0;
			for (let i = 0; i < dataArrayRef.current.length; i++) {
				sum += dataArrayRef.current[i];
			}
			const rawEnergy = sum / (dataArrayRef.current.length * 255);

			setEnergy(rawEnergy);

			const aboveThreshold = rawEnergy > ENERGY_THRESHOLD;

			if (!isPlayingRef.current) {
				// Currently silent -- check for onset
				if (aboveThreshold) {
					onsetCountRef.current++;
					if (onsetCountRef.current >= ONSET_FRAMES) {
						isPlayingRef.current = true;
						setIsPlaying(true);
						onsetCountRef.current = 0;
						offsetStartRef.current = null;
						log.log(`PLAYING detected (sustained ${ONSET_FRAMES} frames)`);
					} else {
						log.log(
							`Energy: ${rawEnergy.toFixed(3)} (above threshold ${ENERGY_THRESHOLD}) -- onset debounce ${onsetCountRef.current}/${ONSET_FRAMES}`,
						);
					}
				} else {
					onsetCountRef.current = 0;
					// Throttle idle energy logs to 1/sec
					if (timestamp - lastLogTimeRef.current > 1000) {
						log.log(
							`Energy: ${rawEnergy.toFixed(3)} (below threshold ${ENERGY_THRESHOLD}) -- idle`,
						);
						lastLogTimeRef.current = timestamp;
					}
				}
			} else {
				// Currently playing -- check for offset
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
							`SILENCE detected (sustained ${(silenceDuration / 1000).toFixed(1)}s)`,
						);
					}
				} else {
					// Reset offset timer -- still playing
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
