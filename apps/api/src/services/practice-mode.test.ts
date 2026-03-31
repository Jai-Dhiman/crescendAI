import { describe, it, expect } from "vitest";
import { ModeDetector, PracticeMode } from "./practice-mode";
import type { ChunkSignal } from "./practice-mode";

function makeSignal(overrides?: Partial<ChunkSignal>): ChunkSignal {
	return {
		chunkIndex: 0,
		timestampMs: Date.now(),
		barRange: null,
		pitchBigrams: new Set(),
		hasPieceMatch: false,
		barsProgressing: false,
		scores: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
		...overrides,
	};
}

function bigrams(...pairs: [string, string][]): Set<string> {
	return new Set(pairs.map(([a, b]) => `${a}-${b}`));
}

const RICH_BIGRAMS = new Set(["60-62", "62-64", "64-65", "65-67", "67-69"]);

describe("ModeDetector", () => {
	it("starts in Warming mode", () => {
		const d = new ModeDetector();
		expect(d.mode).toBe(PracticeMode.Warming);
	});

	it("Warming -> Regular after 4 ambiguous chunks", () => {
		const d = new ModeDetector();
		const now = Date.now();
		for (let i = 0; i < 4; i++) {
			d.update(makeSignal({ chunkIndex: i, timestampMs: now + i * 15000 }));
		}
		expect(d.mode).toBe(PracticeMode.Regular);
	});

	it("Warming -> Running with piece match + progression", () => {
		const d = new ModeDetector();
		const now = Date.now();
		// First signal establishes a bar range in the window.
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now, barRange: [1, 4] }));
		// Second signal has piece match + higher bar range (progress).
		d.update(
			makeSignal({
				chunkIndex: 1,
				timestampMs: now + 15000,
				hasPieceMatch: true,
				barsProgressing: true,
				barRange: [5, 8],
			}),
		);
		expect(d.mode).toBe(PracticeMode.Running);
	});

	it("Warming -> Drilling on repeated passage with substance", () => {
		const d = new ModeDetector();
		const now = Date.now();
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now, pitchBigrams: RICH_BIGRAMS, barRange: [5, 10] }));
		d.update(makeSignal({ chunkIndex: 1, timestampMs: now + 15000, pitchBigrams: RICH_BIGRAMS, barRange: [5, 10] }));
		const transitions = d.update(
			makeSignal({ chunkIndex: 2, timestampMs: now + 30000, pitchBigrams: RICH_BIGRAMS, barRange: [5, 10] }),
		);
		expect(d.mode).toBe(PracticeMode.Drilling);
		expect(transitions.some((t) => t.to === PracticeMode.Drilling)).toBe(true);
	});

	it("detects Winding on 60s+ gap", () => {
		const d = new ModeDetector();
		const now = Date.now();
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now }));
		const transitions = d.update(makeSignal({ chunkIndex: 1, timestampMs: now + 70000 }));
		expect(transitions.some((t) => t.to === PracticeMode.Winding)).toBe(true);
	});

	it("two-step silence emits Winding then resumes (Running if piece match)", () => {
		const d = new ModeDetector();
		const now = Date.now();
		// Get into Running first.
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now, barRange: [1, 4] }));
		d.update(makeSignal({ chunkIndex: 1, timestampMs: now + 15000, hasPieceMatch: true, barsProgressing: true, barRange: [5, 8] }));
		expect(d.mode).toBe(PracticeMode.Running);

		const transitions = d.update(
			makeSignal({
				chunkIndex: 2,
				timestampMs: now + 15000 + 70000,
				hasPieceMatch: true,
				barRange: [9, 12],
			}),
		);
		expect(transitions.length).toBeGreaterThanOrEqual(2);
		expect(transitions[0].to).toBe(PracticeMode.Winding);
		expect(d.mode).toBe(PracticeMode.Running); // resumed via two-step
	});

	it("Winding policy suppresses observations", () => {
		const d = new ModeDetector();
		const now = Date.now();
		// Force into Winding via silence gap.
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now }));
		// Send a signal that triggers silence but no piece match (resume to Regular).
		d.update(makeSignal({ chunkIndex: 1, timestampMs: now + 70000 }));
		// The detector resumed to Regular, but Winding policy should suppress.
		const windingDetector = new ModeDetector();
		// @ts-expect-error -- accessing private for test
		windingDetector.mode = PracticeMode.Winding;
		expect(windingDetector.observationPolicy.suppress).toBe(true);
	});

	it("Warming policy does not suppress", () => {
		const d = new ModeDetector();
		expect(d.observationPolicy.suppress).toBe(false);
	});

	it("Running -> Drilling after dwell + repetition", () => {
		const d = new ModeDetector();
		const now = Date.now();
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now, barRange: [1, 4] }));
		d.update(makeSignal({ chunkIndex: 1, timestampMs: now + 15000, hasPieceMatch: true, barsProgressing: true, barRange: [5, 8] }));
		expect(d.mode).toBe(PracticeMode.Running);

		// Advance past RUNNING_DWELL_MS (30s) with repeated passage.
		d.update(makeSignal({ chunkIndex: 2, timestampMs: now + 50000, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		d.update(makeSignal({ chunkIndex: 3, timestampMs: now + 65000, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		d.update(makeSignal({ chunkIndex: 4, timestampMs: now + 80000, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		expect(d.mode).toBe(PracticeMode.Drilling);
	});

	it("Drilling -> Running on new material after dwell", () => {
		const d = new ModeDetector();
		const now = Date.now();
		// Get into Drilling.
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		d.update(makeSignal({ chunkIndex: 1, timestampMs: now + 15000, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		d.update(makeSignal({ chunkIndex: 2, timestampMs: now + 30000, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		expect(d.mode).toBe(PracticeMode.Drilling);

		// After DRILLING_DWELL_MS, send new material with piece match + progress.
		const NEW_BIGRAMS = new Set(["72-74", "74-76", "76-77", "77-79"]);
		d.update(makeSignal({ chunkIndex: 3, timestampMs: now + 65000, pitchBigrams: NEW_BIGRAMS, barRange: [15, 18], hasPieceMatch: true }));
		d.update(makeSignal({ chunkIndex: 4, timestampMs: now + 80000, pitchBigrams: new Set(["77-79", "79-81", "81-83"]), barRange: [19, 22], hasPieceMatch: true }));
		expect(d.mode).toBe(PracticeMode.Running);
	});

	it("Regular -> Running after dwell + piece match + progress", () => {
		const d = new ModeDetector();
		const now = Date.now();
		// Exhaust warming to reach Regular.
		for (let i = 0; i < 4; i++) {
			d.update(makeSignal({ chunkIndex: i, timestampMs: now + i * 1000 }));
		}
		expect(d.mode).toBe(PracticeMode.Regular);

		// After REGULAR_DWELL_MS (15s), piece match + progress.
		d.update(makeSignal({ chunkIndex: 4, timestampMs: now + 16000, barRange: [1, 4] }));
		d.update(makeSignal({ chunkIndex: 5, timestampMs: now + 31000, hasPieceMatch: true, barRange: [5, 8] }));
		expect(d.mode).toBe(PracticeMode.Running);
	});

	it("min dwell prevents early exit from Drilling", () => {
		const d = new ModeDetector();
		const now = Date.now();
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		d.update(makeSignal({ chunkIndex: 1, timestampMs: now + 15000, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		d.update(makeSignal({ chunkIndex: 2, timestampMs: now + 30000, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		expect(d.mode).toBe(PracticeMode.Drilling);

		// Only 1s later -- should stay Drilling despite new material.
		const NEW_BIGRAMS = new Set(["72-74", "74-76"]);
		d.update(makeSignal({ chunkIndex: 3, timestampMs: now + 31000, pitchBigrams: NEW_BIGRAMS, barRange: [20, 25], hasPieceMatch: true }));
		expect(d.mode).toBe(PracticeMode.Drilling);
	});

	it("drilling increments repetition count without emitting a transition", () => {
		const d = new ModeDetector();
		const now = Date.now();
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		d.update(makeSignal({ chunkIndex: 1, timestampMs: now + 15000, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		d.update(makeSignal({ chunkIndex: 2, timestampMs: now + 30000, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		expect(d.mode).toBe(PracticeMode.Drilling);

		// Another repeat within dwell -- no transition emitted.
		const transitions = d.update(
			makeSignal({ chunkIndex: 3, timestampMs: now + 32000, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }),
		);
		expect(transitions).toHaveLength(0);
		expect(d.mode).toBe(PracticeMode.Drilling);
	});

	it("serializes and deserializes correctly", () => {
		const d = new ModeDetector();
		const now = Date.now();
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now, barRange: [1, 4] }));
		d.update(makeSignal({ chunkIndex: 1, timestampMs: now + 15000, hasPieceMatch: true, barsProgressing: true, barRange: [5, 8] }));
		expect(d.mode).toBe(PracticeMode.Running);

		const json = d.toJSON();
		const d2 = ModeDetector.fromJSON(json);
		expect(d2.mode).toBe(d.mode);
		expect(d2.observationPolicy.minIntervalMs).toBe(d.observationPolicy.minIntervalMs);
	});

	it("serializes Set<string> bigrams round-trip", () => {
		const d = new ModeDetector();
		const now = Date.now();
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now, pitchBigrams: RICH_BIGRAMS }));

		const d2 = ModeDetector.fromJSON(d.toJSON());
		// Further updates should work correctly after deserialization.
		expect(() => {
			d2.update(makeSignal({ chunkIndex: 1, timestampMs: now + 15000, pitchBigrams: RICH_BIGRAMS }));
		}).not.toThrow();
	});

	it("no crash on empty signal", () => {
		const d = new ModeDetector();
		expect(() => {
			d.update(makeSignal({ chunkIndex: 0, timestampMs: Date.now() }));
		}).not.toThrow();
		expect(d.mode).toBe(PracticeMode.Warming);
	});

	it("Warming observation policy: 30s interval, not suppressed, not comparative", () => {
		const d = new ModeDetector();
		const policy = d.observationPolicy;
		expect(policy.suppress).toBe(false);
		expect(policy.minIntervalMs).toBe(30_000);
		expect(policy.comparative).toBe(false);
	});

	it("Drilling observation policy: 90s interval, comparative", () => {
		const d = new ModeDetector();
		const now = Date.now();
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		d.update(makeSignal({ chunkIndex: 1, timestampMs: now + 15000, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		d.update(makeSignal({ chunkIndex: 2, timestampMs: now + 30000, pitchBigrams: RICH_BIGRAMS, barRange: [10, 14] }));
		expect(d.mode).toBe(PracticeMode.Drilling);
		const policy = d.observationPolicy;
		expect(policy.suppress).toBe(false);
		expect(policy.minIntervalMs).toBe(90_000);
		expect(policy.comparative).toBe(true);
	});

	it("Running observation policy: 150s interval, not comparative", () => {
		const d = new ModeDetector();
		const now = Date.now();
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now, barRange: [1, 4] }));
		d.update(makeSignal({ chunkIndex: 1, timestampMs: now + 15000, hasPieceMatch: true, barsProgressing: true, barRange: [5, 8] }));
		expect(d.mode).toBe(PracticeMode.Running);
		const policy = d.observationPolicy;
		expect(policy.minIntervalMs).toBe(150_000);
		expect(policy.comparative).toBe(false);
	});

	it("Regular observation policy: 180s interval", () => {
		const d = new ModeDetector();
		const now = Date.now();
		for (let i = 0; i < 4; i++) {
			d.update(makeSignal({ chunkIndex: i, timestampMs: now + i * 1000 }));
		}
		expect(d.mode).toBe(PracticeMode.Regular);
		expect(d.observationPolicy.minIntervalMs).toBe(180_000);
	});

	it("Winding -> Regular when no piece match on resume", () => {
		const d = new ModeDetector();
		const now = Date.now();
		for (let i = 0; i < 4; i++) {
			d.update(makeSignal({ chunkIndex: i, timestampMs: now + i * 1000 }));
		}
		expect(d.mode).toBe(PracticeMode.Regular);

		// Trigger Winding via silence gap, no piece match -> resumes to Regular.
		const transitions = d.update(makeSignal({ chunkIndex: 4, timestampMs: now + 4000 + 70000 }));
		expect(transitions.some((t) => t.to === PracticeMode.Winding)).toBe(true);
		expect(d.mode).toBe(PracticeMode.Regular);
	});

	it("ModeTransition carries from/to/chunkIndex/timestampMs", () => {
		const d = new ModeDetector();
		const now = Date.now();
		d.update(makeSignal({ chunkIndex: 0, timestampMs: now, barRange: [1, 4] }));
		const transitions = d.update(
			makeSignal({ chunkIndex: 1, timestampMs: now + 15000, hasPieceMatch: true, barsProgressing: true, barRange: [5, 8] }),
		);
		expect(transitions).toHaveLength(1);
		const t = transitions[0];
		expect(t.from).toBe(PracticeMode.Warming);
		expect(t.to).toBe(PracticeMode.Running);
		expect(t.chunkIndex).toBe(1);
		expect(t.timestampMs).toBe(now + 15000);
	});
});
