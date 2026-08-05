import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AudioWaveformRing } from "./AudioWaveformRing";

// jsdom has no ResizeObserver -- stub it so the component's effect can set a
// non-zero canvas size synchronously (the draw loop bails out early on 0x0).
class MockResizeObserver {
	callback: ResizeObserverCallback;
	constructor(callback: ResizeObserverCallback) {
		this.callback = callback;
	}
	observe(target: Element) {
		this.callback(
			[
				{
					target,
					contentRect: { width: 100, height: 100 } as DOMRectReadOnly,
				} as ResizeObserverEntry,
			],
			this as unknown as ResizeObserver,
		);
	}
	unobserve() {}
	disconnect() {}
}

describe("AudioWaveformRing", () => {
	let strokeStyleHistory: string[];
	let rafCallback: FrameRequestCallback | null;

	beforeEach(() => {
		strokeStyleHistory = [];
		rafCallback = null;

		vi.stubGlobal("ResizeObserver", MockResizeObserver);

		// Capture the scheduled draw callback instead of running the animation
		// loop for real -- we invoke it exactly once, synchronously, per test.
		vi.stubGlobal(
			"requestAnimationFrame",
			vi.fn((cb: FrameRequestCallback) => {
				rafCallback = cb;
				return 1;
			}),
		);
		vi.stubGlobal("cancelAnimationFrame", vi.fn());

		const recordingCtx = {
			setTransform: vi.fn(),
			clearRect: vi.fn(),
			beginPath: vi.fn(),
			moveTo: vi.fn(),
			quadraticCurveTo: vi.fn(),
			closePath: vi.fn(),
			stroke: vi.fn(),
			lineWidth: 0,
			get strokeStyle() {
				return strokeStyleHistory.at(-1) ?? "";
			},
			set strokeStyle(value: string) {
				strokeStyleHistory.push(value);
			},
		};

		vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
			// biome-ignore lint/suspicious/noExplicitAny: partial CanvasRenderingContext2D mock for the test boundary
			recordingCtx as any,
		);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
		vi.restoreAllMocks();
		document.documentElement.style.removeProperty("--color-accent");
	});

	it("derives the ring stroke color from --color-accent at draw time", () => {
		document.documentElement.style.setProperty("--color-accent", "#123456");

		// Sanity check on the jsdom custom-property read the component relies on.
		expect(
			getComputedStyle(document.documentElement)
				.getPropertyValue("--color-accent")
				.trim(),
		).toBe("#123456");

		render(
			<AudioWaveformRing analyserNode={null} isPlaying={false} active={true} />,
		);

		expect(rafCallback).not.toBeNull();
		act(() => {
			rafCallback?.(1000);
		});

		expect(strokeStyleHistory.length).toBeGreaterThan(0);
		const lastStroke = strokeStyleHistory.at(-1);
		expect(lastStroke).toMatch(/^rgba\(18, 52, 86, [\d.]+\)$/);
		expect(lastStroke).not.toMatch(/^rgba\(122, 154, 130,/);
	});
});
