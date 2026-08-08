import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AUTO_STOP_SILENCE_MS, MARK_SILENCE_MS } from "../lib/pause-state";
import { usePauseTracker } from "./usePauseTracker";

describe("usePauseTracker", () => {
	beforeEach(() => {
		vi.useFakeTimers();
	});
	afterEach(() => {
		vi.useRealTimers();
	});

	it("shows no mark and does not auto-stop while playing", () => {
		const { result } = renderHook(() => usePauseTracker(true));
		act(() => {
			vi.advanceTimersByTime(AUTO_STOP_SILENCE_MS + 5000);
		});
		expect(result.current.canShowMark).toBe(false);
		expect(result.current.autoStopped).toBe(false);
	});

	it("allows a mark once silence reaches the threshold after playing stops", () => {
		const { result, rerender } = renderHook(
			({ isPlaying }) => usePauseTracker(isPlaying),
			{ initialProps: { isPlaying: true } },
		);
		rerender({ isPlaying: false });
		act(() => {
			vi.advanceTimersByTime(MARK_SILENCE_MS);
		});
		expect(result.current.canShowMark).toBe(true);
		expect(result.current.autoStopped).toBe(false);
	});

	it("resume() resets the silence clock without requiring isPlaying to change", () => {
		const { result, rerender } = renderHook(
			({ isPlaying }) => usePauseTracker(isPlaying),
			{ initialProps: { isPlaying: true } },
		);
		rerender({ isPlaying: false });
		act(() => {
			vi.advanceTimersByTime(AUTO_STOP_SILENCE_MS);
		});
		expect(result.current.autoStopped).toBe(true);

		act(() => {
			result.current.resume();
		});
		expect(result.current.autoStopped).toBe(false);
		expect(result.current.silenceMs).toBe(0);
	});
});
