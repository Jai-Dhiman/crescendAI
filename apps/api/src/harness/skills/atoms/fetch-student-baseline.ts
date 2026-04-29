import type { ToolDefinition } from "../../loop/types";
import { DIMENSIONS } from "../../artifacts/diagnosis";

export type Baseline = {
	dimension: string;
	mean: number;
	stddev: number;
	n_sessions: number;
};

export const fetchStudentBaseline: ToolDefinition = {
	name: "fetch-student-baseline",
	description:
		"Computes rolling mean and stddev from pre-materialized per-session MuQ means for one dimension. Returns null when fewer than 3 sessions are available. Input session_means are the last 10 session means from digest.",
	input_schema: {
		type: "object",
		properties: {
			dimension: { type: "string", enum: DIMENSIONS },
			session_means: {
				type: "array",
				items: { type: "number", minimum: 0, maximum: 1 },
				description:
					"Per-session MuQ mean scores for this dimension, up to 10 sessions.",
			},
		},
		required: ["dimension", "session_means"],
	},
	invoke: async (input: unknown): Promise<Baseline | null> => {
		const { dimension, session_means } = input as {
			dimension: string;
			session_means: number[];
		};
		if (!(DIMENSIONS as readonly string[]).includes(dimension)) {
			throw new Error(`fetch-student-baseline: unknown dimension ${dimension}`);
		}
		const n = session_means.length;
		if (n < 3) return null;
		const mean = session_means.reduce((a, b) => a + b, 0) / n;
		const variance = session_means.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
		return { dimension, mean, stddev: Math.sqrt(variance), n_sessions: n };
	},
};
