import { useCallback, useEffect, useRef, useState } from "react";
import type { PauseState } from "../lib/pause-state";
import { computePauseState } from "../lib/pause-state";

export interface UsePauseTrackerReturn extends PauseState {
	/** Resets the silence clock in place. Does not touch isPlaying or any
	 * session/recording state — the auto-stop banner is UI-only (see spec). */
	resume: () => void;
}

/**
 * Wraps computePauseState with a live clock. silenceStartedAt is a ref, not
 * state: it changes every tick indirectly (via the 1s interval re-deriving
 * `now`), and putting the timestamp itself in state would double the
 * re-render rate for no visible benefit.
 */
export function usePauseTracker(isPlaying: boolean): UsePauseTrackerReturn {
	const silenceStartedAtRef = useRef<number | null>(
		isPlaying ? null : Date.now(),
	);
	const [state, setState] = useState<PauseState>(() =>
		computePauseState({
			isPlaying,
			silenceStartedAt: silenceStartedAtRef.current,
			now: Date.now(),
		}),
	);

	useEffect(() => {
		if (isPlaying) {
			silenceStartedAtRef.current = null;
		} else if (silenceStartedAtRef.current === null) {
			silenceStartedAtRef.current = Date.now();
		}
		setState(
			computePauseState({
				isPlaying,
				silenceStartedAt: silenceStartedAtRef.current,
				now: Date.now(),
			}),
		);
	}, [isPlaying]);

	useEffect(() => {
		const id = setInterval(() => {
			setState(
				computePauseState({
					isPlaying,
					silenceStartedAt: silenceStartedAtRef.current,
					now: Date.now(),
				}),
			);
		}, 1000);
		return () => clearInterval(id);
	}, [isPlaying]);

	const resume = useCallback(() => {
		silenceStartedAtRef.current = isPlaying ? null : Date.now();
		setState(
			computePauseState({
				isPlaying,
				silenceStartedAt: silenceStartedAtRef.current,
				now: Date.now(),
			}),
		);
	}, [isPlaying]);

	return { ...state, resume };
}
