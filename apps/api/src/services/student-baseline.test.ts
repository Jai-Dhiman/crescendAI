import { describe, expect, it } from "vitest";
import {
	type BaselineState,
	BaselineStateSchema,
	type SessionSamples,
	initialBaselineState,
	updateBaseline,
} from "./student-baseline";

function runSequence(sessions: SessionSamples[]): BaselineState[] {
	const trace: BaselineState[] = [];
	let state = initialBaselineState();
	for (const session of sessions) {
		state = updateBaseline(state, session);
		trace.push(state);
	}
	return trace;
}

// Symmetric jitter (+/-0.01 around 0.5, equal counts each side) so the median
// absolute deviation equals every point's own deviation -- no point exceeds
// DEVIANT_SAMPLE_MULTIPLE * MAD, so a clean cluster never self-triggers.
const CLUSTER = [0.49, 0.51, 0.49, 0.51, 0.49, 0.51];

describe("runSequence harness", () => {
	it("folds sessions through updateBaseline and traces lifecycle", () => {
		const trace = runSequence([
			{ timestamp: "2026-01-01T00:00:00Z", scores: { pedaling: CLUSTER } },
		]);
		expect(trace).toHaveLength(1);
		expect(trace[0].dimensions.pedaling.lifecycle).toBe("absent");
	});
});

describe("cold start", () => {
	it("uses the same call shape as any other session, and round-trips through JSON", () => {
		const state = initialBaselineState();
		const result = updateBaseline(state, {
			timestamp: "2026-01-01T00:00:00Z",
			scores: { phrasing: CLUSTER },
		});
		expect(result.dimensions.phrasing.lifecycle).toBe("absent");
		expect(typeof result.dimensions.phrasing.noiseFloor).toBe("number");
		const roundTripped = JSON.parse(JSON.stringify(result));
		expect(() => BaselineStateSchema.parse(roundTripped)).not.toThrow();
		expect(roundTripped).toEqual(result);
	});
});

describe("explicit failures", () => {
	it("throws on an unknown dimension", () => {
		const state = initialBaselineState();
		expect(() =>
			updateBaseline(state, {
				timestamp: "2026-01-01T00:00:00Z",
				scores: { not_a_dimension: CLUSTER } as never,
			}),
		).toThrow(/unknown dimension/);
	});

	it("throws on a non-finite score", () => {
		const state = initialBaselineState();
		expect(() =>
			updateBaseline(state, {
				timestamp: "2026-01-01T00:00:00Z",
				scores: { pedaling: [0.5, Number.NaN, 0.5] },
			}),
		).toThrow(/non-finite/);
	});

	it("throws on an unparseable timestamp", () => {
		const state = initialBaselineState();
		expect(() =>
			updateBaseline(state, {
				timestamp: "not-a-date",
				scores: { pedaling: CLUSTER },
			}),
		).toThrow(/unparseable timestamp/);
	});

	it("throws on an out-of-order session timestamp", () => {
		const state = updateBaseline(initialBaselineState(), {
			timestamp: "2026-01-05T00:00:00Z",
			scores: { pedaling: CLUSTER },
		});
		expect(() =>
			updateBaseline(state, {
				timestamp: "2026-01-01T00:00:00Z",
				scores: { pedaling: CLUSTER },
			}),
		).toThrow(/precedes/);
	});
});

const CLUSTER_WITH_3_OUTLIERS = [
	0.49, 0.51, 0.49, 0.51, 0.49, 0.51, 0.1, 0.1, 0.1,
];

const CLUSTER_WITH_1_OUTLIER = [0.49, 0.51, 0.49, 0.51, 0.49, 0.51, 0.1];

describe("session 1 within-session evidence", () => {
	it("fires immediately from >=3 deviant samples in the first sitting", () => {
		const trace = runSequence([
			{
				timestamp: "2026-01-01T00:00:00Z",
				scores: { pedaling: CLUSTER_WITH_3_OUTLIERS },
			},
		]);
		expect(trace[0].dimensions.pedaling.lifecycle).toBe("active");
	});

	it("stays quiet on a single deviant observation", () => {
		const trace = runSequence([
			{
				timestamp: "2026-01-01T00:00:00Z",
				scores: { pedaling: CLUSTER_WITH_1_OUTLIER },
			},
		]);
		expect(trace[0].dimensions.pedaling.lifecycle).toBe("absent");
	});
});

describe("across-session evidence", () => {
	it("fires on persistent deviation across FIRE_PERSISTENCE sessions", () => {
		const shifted = [0.79, 0.81, 0.79, 0.81, 0.79, 0.81];
		const trace = runSequence([
			{ timestamp: "2026-01-01T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-02T00:00:00Z", scores: { dynamics: shifted } },
			{ timestamp: "2026-01-03T00:00:00Z", scores: { dynamics: shifted } },
			{ timestamp: "2026-01-04T00:00:00Z", scores: { dynamics: shifted } },
		]);
		expect(trace[0].dimensions.dynamics.lifecycle).toBe("absent");
		expect(trace[1].dimensions.dynamics.lifecycle).toBe("absent");
		expect(trace[2].dimensions.dynamics.lifecycle).toBe("absent");
		expect(trace[3].dimensions.dynamics.lifecycle).toBe("active");
	});

	it("stays quiet on a single off session amid consistent sessions", () => {
		const mildlyOff = [0.59, 0.61, 0.59, 0.61, 0.59, 0.61];
		const trace = runSequence([
			{ timestamp: "2026-01-01T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-02T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-03T00:00:00Z", scores: { dynamics: mildlyOff } },
			{ timestamp: "2026-01-04T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-05T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-06T00:00:00Z", scores: { dynamics: CLUSTER } },
		]);
		for (const s of trace) {
			expect(s.dimensions.dynamics.lifecycle).toBe("absent");
		}
	});
});

describe("band width", () => {
	it("narrows the band monotonically under consistent evidence", () => {
		// Same call shape every session -- no session-count branch anywhere in
		// updateBaseline. `halfWidth` is read directly off the returned state
		// (the exact value foldDimension used), not recomputed here -- a test
		// that re-derives the implementation's own arithmetic would only prove
		// two copies of a formula agree, not that the formula is right.
		const sessions: SessionSamples[] = Array.from({ length: 6 }, (_, i) => ({
			timestamp: `2026-01-0${i + 1}T00:00:00Z`,
			scores: { phrasing: CLUSTER },
		}));
		const trace = runSequence(sessions);
		const halfWidths = trace.map((s) => s.dimensions.phrasing.halfWidth);
		// Hand-verified for this exact CLUSTER sequence and DEFAULT_BASELINE_CONFIG:
		// 0.0628 -> 0.0341 -> 0.0247 -> 0.02 -> 0.0173 -> 0.0155 (strictly decreasing).
		for (let i = 1; i < halfWidths.length; i++) {
			expect(halfWidths[i]).toBeLessThan(halfWidths[i - 1]);
		}
		expect(halfWidths[0]).toBeCloseTo(0.0629, 3);
		expect(halfWidths[5]).toBeCloseTo(0.0155, 3);
	});
});
