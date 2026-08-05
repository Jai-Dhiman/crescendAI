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
});
