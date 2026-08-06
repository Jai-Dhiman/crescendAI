import { describe, expect, it } from "vitest";
import { ValidationError } from "../lib/errors";
import {
	type BaselineState,
	BaselineStateSchema,
	DEFAULT_BASELINE_CONFIG,
	initialBaselineState,
	type SessionSamples,
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
		).toThrow(ValidationError);
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
		).toThrow(ValidationError);
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
		).toThrow(ValidationError);
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
		).toThrow(ValidationError);
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

describe("combined within-session and across-session evidence", () => {
	// Shifted centre (0.79/0.81 vs the baseline's 0.49/0.51) AND internally
	// noisy: 3 points at 0.3 are far enough from this session's own centre to
	// trip the within-session MAD threshold too, so a single fold produces
	// both a nonzero `withinSessionDeviants` and `acrossSessionOutOfBand`.
	const shiftedNoisy = [0.79, 0.81, 0.79, 0.81, 0.79, 0.81, 0.3, 0.3, 0.3];
	const shiftedClean = [0.79, 0.81, 0.79, 0.81, 0.79, 0.81];

	it("sums within-session and across-session contribution in the same fold", () => {
		const trace = runSequence([
			{ timestamp: "2026-01-01T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-02T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-03T00:00:00Z", scores: { dynamics: shiftedNoisy } },
		]);
		// Hand-verified for this exact fixture: contribution = 3 within-session
		// deviants (capped at MAX_WITHIN_SESSION_CONTRIBUTION) + 1 across-session
		// = 4, taking consecutiveOutOfBand from 0 straight to 4 in one fold --
		// only possible if both sources fired in the same fold.
		expect(trace[2].dimensions.dynamics.consecutiveOutOfBand).toBe(4);
		expect(trace[2].dimensions.dynamics.lifecycle).toBe("active");
	});

	it("fires in strictly fewer sessions than across-session evidence alone", () => {
		const combined = runSequence([
			{ timestamp: "2026-01-01T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-02T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-03T00:00:00Z", scores: { dynamics: shiftedNoisy } },
			{ timestamp: "2026-01-04T00:00:00Z", scores: { dynamics: shiftedNoisy } },
		]);
		const acrossOnly = runSequence([
			{ timestamp: "2026-01-01T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-02T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-03T00:00:00Z", scores: { dynamics: shiftedClean } },
			{ timestamp: "2026-01-04T00:00:00Z", scores: { dynamics: shiftedClean } },
			{ timestamp: "2026-01-05T00:00:00Z", scores: { dynamics: shiftedClean } },
			{ timestamp: "2026-01-06T00:00:00Z", scores: { dynamics: shiftedClean } },
		]);
		const combinedFireSession = combined.findIndex(
			(s) => s.dimensions.dynamics.lifecycle === "active",
		);
		const acrossOnlyFireSession = acrossOnly.findIndex(
			(s) => s.dimensions.dynamics.lifecycle === "active",
		);
		expect(combinedFireSession).toBeGreaterThanOrEqual(0);
		expect(acrossOnlyFireSession).toBeGreaterThanOrEqual(0);
		// Same shift magnitude, same starting history -- the only difference is
		// whether within-session evidence contributes. Fewer sessions to fire
		// proves the two sources sum rather than merely co-occur.
		expect(combinedFireSession).toBeLessThan(acrossOnlyFireSession);
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

describe("symmetric retirement", () => {
	it("walks active -> improving -> resolved on persistent return-to-band", () => {
		const shifted = [0.79, 0.81, 0.79, 0.81, 0.79, 0.81];
		const sessions: SessionSamples[] = [
			{ timestamp: "2026-01-01T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-01-02T00:00:00Z", scores: { timing: shifted } },
			{ timestamp: "2026-01-03T00:00:00Z", scores: { timing: shifted } },
			{ timestamp: "2026-01-04T00:00:00Z", scores: { timing: shifted } }, // fires -> active
			{ timestamp: "2026-02-01T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-02T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-03T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-04T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-05T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-06T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-07T00:00:00Z", scores: { timing: CLUSTER } }, // -> improving
			{ timestamp: "2026-02-08T00:00:00Z", scores: { timing: CLUSTER } }, // -> resolved
		];
		const trace = runSequence(sessions);
		expect(trace[3].dimensions.timing.lifecycle).toBe("active");
		expect(trace[10].dimensions.timing.lifecycle).toBe("improving");
		expect(trace[11].dimensions.timing.lifecycle).toBe("resolved");
	});
});

describe("recurrence", () => {
	it("returns lifecycle to active after a resolved dimension recurs", () => {
		const shifted = [0.79, 0.81, 0.79, 0.81, 0.79, 0.81];
		const sessions: SessionSamples[] = [
			{ timestamp: "2026-01-01T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-01-02T00:00:00Z", scores: { timing: shifted } },
			{ timestamp: "2026-01-03T00:00:00Z", scores: { timing: shifted } },
			{ timestamp: "2026-01-04T00:00:00Z", scores: { timing: shifted } }, // fires -> active
			{ timestamp: "2026-02-01T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-02T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-03T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-04T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-05T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-06T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-07T00:00:00Z", scores: { timing: CLUSTER } }, // -> improving
			{ timestamp: "2026-02-08T00:00:00Z", scores: { timing: CLUSTER } }, // -> resolved
			{ timestamp: "2026-03-01T00:00:00Z", scores: { timing: shifted } },
			{ timestamp: "2026-03-02T00:00:00Z", scores: { timing: shifted } },
			{ timestamp: "2026-03-03T00:00:00Z", scores: { timing: shifted } }, // recurs -> active
		];
		const trace = runSequence(sessions);
		expect(trace[11].dimensions.timing.lifecycle).toBe("resolved");
		expect(trace[14].dimensions.timing.lifecycle).toBe("active");
	});
});

describe("promotion", () => {
	it("promotes only once out-of-band evidence while active spans >=2 distinct ISO weeks, and stays promoted through resolution", () => {
		const shifted = [0.79, 0.81, 0.79, 0.81, 0.79, 0.81];
		// After promotion, 15 consistent CLUSTER sessions carry the dimension
		// through improving to resolved (generated by date arithmetic, not
		// string interpolation, so the run safely crosses the Feb/Mar boundary).
		const inBandSessions: SessionSamples[] = Array.from(
			{ length: 15 },
			(_, i) => {
				const date = new Date(Date.UTC(2026, 1, 1));
				date.setUTCDate(date.getUTCDate() + i);
				return {
					timestamp: date.toISOString(),
					scores: { articulation: CLUSTER },
				};
			},
		);
		const trace = runSequence([
			{ timestamp: "2026-01-05T00:00:00Z", scores: { articulation: CLUSTER } }, // Mon wk02
			{ timestamp: "2026-01-06T00:00:00Z", scores: { articulation: shifted } }, // wk02
			{ timestamp: "2026-01-07T00:00:00Z", scores: { articulation: shifted } }, // wk02
			{ timestamp: "2026-01-08T00:00:00Z", scores: { articulation: shifted } }, // wk02, fires
			{ timestamp: "2026-01-13T00:00:00Z", scores: { articulation: shifted } }, // wk03, more evidence
			...inBandSessions,
		]);
		expect(trace[3].dimensions.articulation.lifecycle).toBe("active");
		expect(trace[3].dimensions.articulation.promoted).toBe(false);
		expect(trace[4].dimensions.articulation.promoted).toBe(true);
		const resolvedEntry = trace.find(
			(s) => s.dimensions.articulation.lifecycle === "resolved",
		);
		expect(resolvedEntry).toBeDefined();
		expect(resolvedEntry?.dimensions.articulation.promoted).toBe(true);
	});

	it("does not promote when all evidence falls inside a single ISO week", () => {
		const shifted = [0.79, 0.81, 0.79, 0.81, 0.79, 0.81];
		const trace = runSequence([
			{ timestamp: "2026-01-05T00:00:00Z", scores: { articulation: CLUSTER } },
			{ timestamp: "2026-01-06T00:00:00Z", scores: { articulation: shifted } },
			{ timestamp: "2026-01-07T00:00:00Z", scores: { articulation: shifted } },
			{ timestamp: "2026-01-08T00:00:00Z", scores: { articulation: shifted } },
		]);
		expect(trace[3].dimensions.articulation.lifecycle).toBe("active");
		expect(trace[3].dimensions.articulation.promoted).toBe(false);
	});
});

describe("sparse sessions never measure spread", () => {
	it("stays quiet when every session supplies too few samples to measure spread", () => {
		const centres = [
			[0.6, 0.6],
			[0.61, 0.61],
			[0.59, 0.59],
			[0.62, 0.62],
			[0.6, 0.6],
			[0.61, 0.61],
		];
		const trace = runSequence(
			centres.map((scores, i) => ({
				timestamp: `2026-01-0${i + 1}T00:00:00Z`,
				scores: { dynamics: scores },
			})),
		);
		for (const s of trace) {
			expect(s.dimensions.dynamics.lifecycle).toBe("absent");
			expect(s.dimensions.dynamics.consecutiveOutOfBand).toBeLessThan(
				DEFAULT_BASELINE_CONFIG.firePersistence,
			);
		}
	});

	it("measures the noise floor from the first session that supplies enough samples", () => {
		const dense = [0.5, 0.7, 0.5, 0.7, 0.5, 0.7];
		const trace = runSequence([
			{ timestamp: "2026-01-01T00:00:00Z", scores: { dynamics: [0.6, 0.6] } },
			{ timestamp: "2026-01-02T00:00:00Z", scores: { dynamics: [0.61, 0.61] } },
			{ timestamp: "2026-01-03T00:00:00Z", scores: { dynamics: dense } },
		]);
		expect(trace[0].dimensions.dynamics.noiseFloorMeasured).toBe(false);
		expect(trace[1].dimensions.dynamics.noiseFloorMeasured).toBe(false);
		expect(trace[2].dimensions.dynamics.noiseFloorMeasured).toBe(true);
		expect(trace[2].dimensions.dynamics.noiseFloor).toBeGreaterThan(0);
		expect(trace[2].dimensions.dynamics.noiseFloor).toBeCloseTo(0.1, 5);
	});
});

describe("confidence", () => {
	it("fires while confidence is still exploratory", () => {
		const trace = runSequence([
			{
				timestamp: "2026-01-01T00:00:00Z",
				scores: { pedaling: CLUSTER_WITH_3_OUTLIERS },
			},
		]);
		expect(trace[0].dimensions.pedaling.lifecycle).toBe("active");
		expect(trace[0].dimensions.pedaling.confidence).toBe("exploratory");
	});

	it("advances confidence toward established purely from accumulated updates, independent of lifecycle", () => {
		const sessions: SessionSamples[] = Array.from({ length: 8 }, (_, i) => ({
			timestamp: `2026-01-0${i + 1}T00:00:00Z`,
			scores: { phrasing: CLUSTER },
		}));
		const trace = runSequence(sessions);
		expect(trace[1].dimensions.phrasing.confidence).toBe("exploratory"); // updateCount 2
		expect(trace[2].dimensions.phrasing.confidence).toBe("provisional"); // updateCount 3
		expect(trace[7].dimensions.phrasing.confidence).toBe("established"); // updateCount 8
		// CLUSTER never deviates, so lifecycle stays absent throughout -- this
		// dimension becomes fully established while still saying nothing,
		// which is the other half of "confidence never gates."
		expect(trace[7].dimensions.phrasing.lifecycle).toBe("absent");
	});
});
