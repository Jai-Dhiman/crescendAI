import type { ToolDefinition } from "../../loop/types";
import { DIMENSIONS } from "../../artifacts/diagnosis";

export const fetchReferencePercentile: ToolDefinition = {
	name: "fetch-reference-percentile",
	description:
		"Returns the percentile rank [0,100] of a MuQ dimension score against a cohort table. Uses linear interpolation between adjacent percentile buckets. Returns 50 when the score equals the cohort median.",
	input_schema: {
		type: "object",
		properties: {
			dimension: { type: "string", enum: DIMENSIONS },
			score: { type: "number", minimum: 0, maximum: 1 },
			cohort_table: {
				type: "array",
				items: {
					type: "object",
					properties: {
						p: {
							type: "number",
							minimum: 0,
							maximum: 100,
							description: "Percentile",
						},
						value: {
							type: "number",
							description: "MuQ score at this percentile",
						},
					},
					required: ["p", "value"],
				},
				description: "Percentile lookup table sorted by value ascending.",
			},
		},
		required: ["dimension", "score", "cohort_table"],
	},
	invoke: async (input: unknown): Promise<number> => {
		const { dimension, score, cohort_table } = input as {
			dimension: string;
			score: number;
			cohort_table: { p: number; value: number }[];
		};
		if (!(DIMENSIONS as readonly string[]).includes(dimension)) {
			throw new Error(
				`fetch-reference-percentile: unknown dimension ${dimension}`,
			);
		}
		const sorted = [...cohort_table].sort((a, b) => a.value - b.value);
		if (sorted.length === 0) return 50;
		if (score <= sorted[0].value) return sorted[0].p;
		if (score >= sorted[sorted.length - 1].value)
			return sorted[sorted.length - 1].p;
		for (let i = 0; i < sorted.length - 1; i++) {
			const lo = sorted[i],
				hi = sorted[i + 1];
			if (score >= lo.value && score <= hi.value) {
				const fraction = (score - lo.value) / (hi.value - lo.value);
				return lo.p + fraction * (hi.p - lo.p);
			}
		}
		throw new Error(
			"fetch-reference-percentile: interpolation invariant violated — cohort_table may contain NaN or unsorted values",
		);
	},
};
