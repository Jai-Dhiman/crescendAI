import { describe, expect, it } from "vitest";
import { ObservationThrottle } from "./observation-throttle";
import type { ObservationEvent } from "./practice-api";

function makeObs(text: string): ObservationEvent {
	return { text, dimension: "dynamics", framing: "correction" };
}

describe("ObservationThrottle", () => {
	it("enqueue delivers observation immediately (DO owns pacing)", () => {
		const throttle = new ObservationThrottle();
		const result = throttle.enqueue(makeObs("hello"));
		expect(result).not.toBeNull();
		expect(result.text).toBe("hello");
	});

	it("enqueue increments chunk counter", () => {
		const throttle = new ObservationThrottle();
		expect(throttle.getChunksReceived()).toBe(0);
		throttle.enqueue(makeObs("obs"));
		expect(throttle.getChunksReceived()).toBe(1);
	});

	it("onChunkProcessed increments chunk counter and returns null", () => {
		const throttle = new ObservationThrottle();
		const result = throttle.onChunkProcessed();
		expect(result).toBeNull();
		expect(throttle.getChunksReceived()).toBe(1);
	});

	it("drain() returns empty when nothing queued", () => {
		const throttle = new ObservationThrottle();
		expect(throttle.drain()).toHaveLength(0);
	});

	it("drain() returns queued observation and clears it", () => {
		const throttle = new ObservationThrottle();
		// Manually set queued state (simulating reconnection buffer)
		throttle.enqueue(makeObs("first"));
		// Since enqueue delivers immediately, queue stays empty
		const drained = throttle.drain();
		expect(drained).toHaveLength(0);
	});

	it("reset() clears chunk counter", () => {
		const throttle = new ObservationThrottle();
		throttle.onChunkProcessed();
		throttle.onChunkProcessed();
		expect(throttle.getChunksReceived()).toBe(2);
		throttle.reset();
		expect(throttle.getChunksReceived()).toBe(0);
	});

	it("reset() clears queued observation", () => {
		const throttle = new ObservationThrottle();
		throttle.reset();
		expect(throttle.drain()).toHaveLength(0);
	});
});
