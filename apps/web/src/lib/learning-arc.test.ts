import { describe, expect, it } from "vitest";
import { describeArc } from "./learning-arc";

// `now` is injected, never read from the system clock, so these assertions
// hold in every timezone and need no fake timers.
const NOW = new Date("2026-08-27T12:00:00.000Z");

describe("describeArc", () => {
	it("describes a piece that has never been practiced", () => {
		expect(
			describeArc({ sessionCount: 0, lastPracticedAt: null, now: NOW }),
		).toBe("Not practiced yet");
	});

	it("uses the singular for exactly one session", () => {
		expect(
			describeArc({
				sessionCount: 1,
				lastPracticedAt: "2026-08-27T09:00:00.000Z",
				now: NOW,
			}),
		).toBe("1 session · last practiced today");
	});

	it("says yesterday for the previous day", () => {
		expect(
			describeArc({
				sessionCount: 4,
				lastPracticedAt: "2026-08-26T09:00:00.000Z",
				now: NOW,
			}),
		).toBe("4 sessions · last practiced yesterday");
	});

	it("names the weekday within the past week", () => {
		// 2026-08-23 is a Sunday, 4 days before 2026-08-27.
		expect(
			describeArc({
				sessionCount: 6,
				lastPracticedAt: "2026-08-23T09:00:00.000Z",
				now: NOW,
			}),
		).toBe("6 sessions · last practiced Sunday");
	});

	it("falls back to an absolute date beyond a week", () => {
		expect(
			describeArc({
				sessionCount: 12,
				lastPracticedAt: "2026-07-04T09:00:00.000Z",
				now: NOW,
			}),
		).toBe("12 sessions · last practiced 4 Jul");
	});
});
