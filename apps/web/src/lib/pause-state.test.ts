import { describe, expect, it } from "vitest";
import {
	AUTO_STOP_SILENCE_MS,
	computePauseState,
	MARK_SILENCE_MS,
} from "./pause-state";

describe("computePauseState", () => {
	it("reports no mark and no auto-stop while playing", () => {
		const state = computePauseState({
			isPlaying: true,
			silenceStartedAt: null,
			now: 100_000,
		});
		expect(state).toEqual({
			silenceMs: 0,
			canShowMark: false,
			autoStopped: false,
		});
	});

	it("allows a mark at exactly the 20s boundary, not before", () => {
		const justUnder = computePauseState({
			isPlaying: false,
			silenceStartedAt: 0,
			now: MARK_SILENCE_MS - 1,
		});
		expect(justUnder.canShowMark).toBe(false);

		const atBoundary = computePauseState({
			isPlaying: false,
			silenceStartedAt: 0,
			now: MARK_SILENCE_MS,
		});
		expect(atBoundary.canShowMark).toBe(true);
	});

	it("auto-stops at exactly the 60s boundary, not before", () => {
		const justUnder = computePauseState({
			isPlaying: false,
			silenceStartedAt: 0,
			now: AUTO_STOP_SILENCE_MS - 1,
		});
		expect(justUnder.autoStopped).toBe(false);

		const atBoundary = computePauseState({
			isPlaying: false,
			silenceStartedAt: 0,
			now: AUTO_STOP_SILENCE_MS,
		});
		expect(atBoundary.autoStopped).toBe(true);
	});
});
