import type { ToolDefinition } from "../../loop/types";

export type OnsetDrift = {
	note_index: number;
	drift_ms: number;
	signed: number;
};

export const computeOnsetDrift: ToolDefinition = {
	name: "compute-onset-drift",
	description:
		"Computes per-note millisecond drift between performance onset and score-aligned expected onset. drift_ms is always non-negative; signed is negative for early, zero for on-time, positive for late.",
	input_schema: {
		type: "object",
		properties: {
			notes: {
				type: "array",
				items: {
					type: "object",
					properties: {
						onset_ms: { type: "number" },
						expected_onset_ms: { type: "number" },
					},
					required: ["onset_ms", "expected_onset_ms"],
				},
				description:
					"Score-aligned notes with both performance and expected onsets.",
			},
		},
		required: ["notes"],
	},
	invoke: async (input: unknown): Promise<OnsetDrift[]> => {
		const { notes } = input as {
			notes: { onset_ms: number; expected_onset_ms: number }[];
		};
		return notes.map((n, i) => ({
			note_index: i,
			drift_ms: Math.abs(n.onset_ms - n.expected_onset_ms),
			signed: n.onset_ms - n.expected_onset_ms,
		}));
	},
};
