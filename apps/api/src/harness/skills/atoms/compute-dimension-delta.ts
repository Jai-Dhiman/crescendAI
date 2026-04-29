import type { ToolDefinition } from "../../loop/types";
import { DIMENSIONS } from "../../artifacts/diagnosis";

export const computeDimensionDelta: ToolDefinition = {
	name: "compute-dimension-delta",
	description:
		"Computes the signed z-score between a MuQ dimension score and the student baseline. Returns (current - baseline.mean) / baseline.stddev. Returns 0 when baseline.stddev is 0. Negative = below baseline; positive = above.",
	input_schema: {
		type: "object",
		properties: {
			dimension: { type: "string", enum: DIMENSIONS },
			current: {
				type: "number",
				minimum: 0,
				maximum: 1,
				description: "MuQ score for this dimension",
			},
			baseline: {
				type: "object",
				properties: {
					mean: { type: "number" },
					stddev: { type: "number", minimum: 0 },
				},
				required: ["mean", "stddev"],
			},
		},
		required: ["dimension", "current", "baseline"],
	},
	invoke: async (input: unknown): Promise<number> => {
		const { dimension, current, baseline } = input as {
			dimension: string;
			current: number;
			baseline: { mean: number; stddev: number };
		};
		if (!(DIMENSIONS as readonly string[]).includes(dimension)) {
			throw new Error(
				`compute-dimension-delta: dimension must be one of ${DIMENSIONS.join(", ")}`,
			);
		}
		if (baseline.stddev === 0) return 0;
		return (current - baseline.mean) / baseline.stddev;
	},
};
