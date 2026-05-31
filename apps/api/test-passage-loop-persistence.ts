// Regression test for Fix A (Finding 4): PassageLoopDetector persistence across
// DO hibernation. The detector's debounce state (lastEventTs/lastPassageKey) must
// survive a toJSON()/fromJSON() round-trip so that a student reconnecting mid-session
// cannot re-trigger a loop attempt the detector already suppressed.
//
// Run: bun test apps/api/test-passage-loop-persistence.ts
//
// Hypothesis under test: serializing the detector into SessionState and rehydrating
// it preserves debounce state; the current module-level WeakMap loses it on hibernation.

import { describe, expect, it } from "bun:test";
import { PassageLoopDetector } from "./src/do/passage-loop-detector";
import type { SegmentLoopArtifact } from "./src/harness/artifacts/segment-loop";

// processPosition only reads barsStart/barsEnd off the assignment; the rest of the
// SegmentLoopArtifact shape is irrelevant to debounce behavior, so we cast a minimal stub.
const assignment = {
	barsStart: 4,
	barsEnd: 7,
} as SegmentLoopArtifact;

const inBoundsSpan = { startBar: 4, endBar: 7, durationMs: 15000 };

describe("PassageLoopDetector persistence (Fix A / Finding 4)", () => {
	it("preserves debounce state across toJSON/fromJSON: a chunk at the same passage stays suppressed after rehydration", () => {
		const detector = new PassageLoopDetector();

		// Chunk 1: first in-bounds attempt fires.
		const first = detector.processPosition(inBoundsSpan, assignment);
		expect(first?.inBounds).toBe(true);

		// Chunk 2: same passage within the debounce window is suppressed.
		const second = detector.processPosition(inBoundsSpan, assignment);
		expect(second).toBeNull();

		// Hibernate + reconstruct: serialize, then rehydrate into a fresh instance.
		const serialized = detector.toJSON();
		const rehydrated = PassageLoopDetector.fromJSON(serialized);

		// Chunk 3 at the same bar range AFTER rehydration must STILL be suppressed —
		// the debounce key/timestamp were preserved.
		const third = rehydrated.processPosition(inBoundsSpan, assignment);
		expect(third).toBeNull();
	});

	it("NEGATIVE (documents the bug): a blank detector — what the WeakMap yields after hibernation — does NOT suppress the same passage", () => {
		// Simulate the pre-fix behavior: on hibernation the module-level WeakMap is blank,
		// so the DO constructs `new PassageLoopDetector()` with zeroed debounce state.
		const detector = new PassageLoopDetector();
		const first = detector.processPosition(inBoundsSpan, assignment);
		expect(first?.inBounds).toBe(true);
		const second = detector.processPosition(inBoundsSpan, assignment);
		expect(second).toBeNull();

		// A brand-new detector (lastEventTs=0, lastPassageKey="") re-fires the same
		// passage — the exact re-trigger Finding 4 describes.
		const blank = new PassageLoopDetector();
		const reFired = blank.processPosition(inBoundsSpan, assignment);
		expect(reFired?.inBounds).toBe(true);
	});
});
