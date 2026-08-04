import { describe, expect, it } from "vitest";
import type { TeachingMoment } from "../services/wasm-bridge";
import { buildObservationPayload } from "./session-brain";

function moment(overrides: Partial<TeachingMoment> = {}): TeachingMoment {
	return {
		chunk_index: 4,
		dimension: "timing",
		score: 0.31,
		baseline: 0.62,
		deviation: -0.31,
		reasoning: "rushed through the left-hand accompaniment",
		is_positive: false,
		...overrides,
	};
}

const evalDetail = {
	predictions: [0.31, 0.5, 0.44, 0.7, 0.62, 0.55] as [
		number,
		number,
		number,
		number,
		number,
		number,
	],
	baselines: {
		dynamics: 0.55,
		timing: 0.62,
		pedaling: 0.48,
		articulation: 0.6,
		phrasing: 0.51,
		interpretation: 0.58,
	},
	analysisFacts: { tier: 1, selected: { dimension: "timing" } },
	barRange: [9, 12] as [number, number],
	analysisTier: 1,
};

describe("buildObservationPayload", () => {
	it("sends only the student-facing keys for a normal session", () => {
		const payload = buildObservationPayload(moment(), null);

		expect(Object.keys(payload).sort()).toEqual([
			"dimension",
			"framing",
			"text",
			"type",
		]);
		expect(payload.framing).toBe("correction");
		// The student never receives raw scores -- the product deliberately keeps
		// numbers off the wire (#143).
		expect(JSON.stringify(payload)).not.toContain("0.31");
	});

	it("frames a positive moment as recognition", () => {
		const payload = buildObservationPayload(
			moment({ is_positive: true, dimension: "dynamics" }),
			null,
		);

		expect(payload.framing).toBe("recognition");
		expect(payload.text).toContain("dynamics");
	});

	it("attaches the fields the eval reads when the session is an eval session", () => {
		const payload = buildObservationPayload(moment(), evalDetail);
		const ctx = payload.eval_context;

		// These five defaulted silently to 0/0.0/0.0/"" on every observation
		// before #143, because the DO never emitted them.
		expect(ctx?.chunk_index).toBe(4);
		expect(ctx?.score).toBe(0.31);
		expect(ctx?.baseline).toBe(0.62);
		expect(ctx?.reasoning_trace).toBe(
			"rushed through the left-hand accompaniment",
		);
	});

	it("attaches the judge context so observations are not graded against nothing", () => {
		const payload = buildObservationPayload(moment(), evalDetail);
		const ctx = payload.eval_context;

		expect(ctx?.predictions).toEqual(evalDetail.predictions);
		expect(ctx?.baselines).toEqual(evalDetail.baselines);
		expect(ctx?.analysis_facts).toEqual(evalDetail.analysisFacts);
		expect(ctx?.bar_range).toEqual([9, 12]);
		expect(ctx?.analysis_tier).toBe(1);
	});

	it("distinguishes observations from different chunks", () => {
		const a = buildObservationPayload(moment({ chunk_index: 0 }), evalDetail);
		const b = buildObservationPayload(moment({ chunk_index: 7 }), evalDetail);

		// Same-chunk_index for every observation is what collapsed every trace
		// file onto "<recording>_chunk0.json" and overwrote them (#143).
		expect(a.eval_context?.chunk_index).toBe(0);
		expect(b.eval_context?.chunk_index).toBe(7);
	});

	it("carries null analysis context when the chunk had no bar alignment", () => {
		const payload = buildObservationPayload(moment(), {
			...evalDetail,
			analysisFacts: null,
			barRange: null,
			analysisTier: 3,
		});

		expect(payload.eval_context?.analysis_facts).toBeNull();
		expect(payload.eval_context?.bar_range).toBeNull();
		expect(payload.eval_context?.analysis_tier).toBe(3);
	});
});
