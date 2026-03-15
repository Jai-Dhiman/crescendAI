import { describe, expect, it } from "vitest";
import { ObservationThrottle } from "./observation-throttle";
import type { ObservationEvent } from "./practice-api";

function makeObs(text: string): ObservationEvent {
	return { text, dimension: "dynamics", framing: "correction" };
}

describe("ObservationThrottle", () => {
	it("blocks observations before minChunks reached", () => {
		const throttle = new ObservationThrottle({ minChunksBeforeFirst: 4 });
		// Only 2 chunks processed
		throttle.onChunkProcessed();
		throttle.onChunkProcessed();
		const result = throttle.enqueue(makeObs("too early"));
		expect(result).toBeNull();
	});

	it("delivers first observation after minChunks reached", () => {
		const throttle = new ObservationThrottle({ minChunksBeforeFirst: 4 });
		for (let i = 0; i < 4; i++) throttle.onChunkProcessed();
		const result = throttle.enqueue(makeObs("ready"));
		expect(result).not.toBeNull();
		expect(result?.text).toBe("ready");
	});

	it("blocks second observation within throttle window", () => {
		const throttle = new ObservationThrottle({
			minChunksBeforeFirst: 1,
			windowMs: 180_000,
		});
		throttle.onChunkProcessed();
		// First one delivers
		const first = throttle.enqueue(makeObs("first"));
		expect(first).not.toBeNull();
		// Second one is blocked (within window)
		const second = throttle.enqueue(makeObs("second"));
		expect(second).toBeNull();
	});

	it("releases queued observation via tick() after window expires", () => {
		const throttle = new ObservationThrottle({
			minChunksBeforeFirst: 1,
			windowMs: 100, // Short window for testing
		});
		throttle.onChunkProcessed();

		throttle.enqueue(makeObs("first"));
		throttle.enqueue(makeObs("queued"));

		// tick() before window expires returns null
		const tooEarly = throttle.tick();
		expect(tooEarly).toBeNull();

		// Wait for window to expire
		return new Promise<void>((resolve) => {
			setTimeout(() => {
				const released = throttle.tick();
				expect(released).not.toBeNull();
				expect(released?.text).toBe("queued");
				resolve();
			}, 150);
		});
	});

	it("replaces queued observation with newer one", () => {
		const throttle = new ObservationThrottle({
			minChunksBeforeFirst: 1,
			windowMs: 180_000,
		});
		throttle.onChunkProcessed();

		throttle.enqueue(makeObs("first"));
		throttle.enqueue(makeObs("old queued"));
		throttle.enqueue(makeObs("new queued"));

		const drained = throttle.drain();
		expect(drained).toHaveLength(1);
		expect(drained[0].text).toBe("new queued");
	});

	it("onChunkProcessed releases queued observation when minChunks newly met", () => {
		const throttle = new ObservationThrottle({
			minChunksBeforeFirst: 3,
			windowMs: 180_000,
		});
		// Enqueue before minChunks met
		throttle.onChunkProcessed();
		throttle.enqueue(makeObs("waiting"));

		// Still not enough chunks
		const r2 = throttle.onChunkProcessed();
		expect(r2).toBeNull();

		// Third chunk meets minimum
		const r3 = throttle.onChunkProcessed();
		expect(r3).not.toBeNull();
		expect(r3?.text).toBe("waiting");
	});

	it("drain() returns queued observation and empties queue", () => {
		const throttle = new ObservationThrottle({
			minChunksBeforeFirst: 1,
			windowMs: 180_000,
		});
		throttle.onChunkProcessed();
		throttle.enqueue(makeObs("first"));
		throttle.enqueue(makeObs("queued"));

		const drained = throttle.drain();
		expect(drained).toHaveLength(1);
		expect(drained[0].text).toBe("queued");

		// Drain again returns empty
		expect(throttle.drain()).toHaveLength(0);
	});

	it("reset() clears all state", () => {
		const throttle = new ObservationThrottle({ minChunksBeforeFirst: 1 });
		throttle.onChunkProcessed();
		throttle.enqueue(makeObs("obs"));
		throttle.reset();

		// After reset, minChunks not met again
		const result = throttle.enqueue(makeObs("after reset"));
		expect(result).toBeNull();
	});
});
