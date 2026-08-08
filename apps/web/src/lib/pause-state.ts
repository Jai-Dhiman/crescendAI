/**
 * How long the student must be silent before a mark may show. Tunable
 * per the epic's open question ("20s is a starting value") — a single
 * named constant, not a scatter of magic numbers, is what makes that tuning
 * a one-line change later.
 */
export const MARK_SILENCE_MS = 20_000;

/** How long the student must be silent before the soft auto-stop banner shows. */
export const AUTO_STOP_SILENCE_MS = 60_000;

export interface PauseStateInput {
	readonly isPlaying: boolean;
	/** Timestamp (ms, same clock as `now`) silence began, or null while playing. */
	readonly silenceStartedAt: number | null;
	/** Current timestamp (ms), supplied by the caller so this stays a pure function. */
	readonly now: number;
}

export interface PauseState {
	readonly silenceMs: number;
	readonly canShowMark: boolean;
	readonly autoStopped: boolean;
}

/**
 * Pure boundary arithmetic over one silence interval. No timers, no DOM —
 * the caller (usePauseTracker) owns the clock and the ref; this only answers
 * "given this much silence, what should the UI show."
 */
export function computePauseState(input: PauseStateInput): PauseState {
	if (input.isPlaying || input.silenceStartedAt === null) {
		return { silenceMs: 0, canShowMark: false, autoStopped: false };
	}
	const silenceMs = Math.max(0, input.now - input.silenceStartedAt);
	return {
		silenceMs,
		canShowMark: silenceMs >= MARK_SILENCE_MS,
		autoStopped: silenceMs >= AUTO_STOP_SILENCE_MS,
	};
}
