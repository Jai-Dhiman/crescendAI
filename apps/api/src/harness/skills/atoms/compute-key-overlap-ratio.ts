import type { ToolDefinition } from "../../loop/types";

export const computeKeyOverlapRatio: ToolDefinition = {
	name: "compute-key-overlap-ratio",
	description:
		"Computes mean per-pair overlap ratio for a monophonic sequence: positive = legato (notes overlap), near-zero = detache, negative = staccato (gap). Requires consecutive monophonic notes; caller must reduce polyphonic passages to one voice.",
	input_schema: {
		type: "object",
		properties: {
			notes: {
				type: "array",
				items: {
					type: "object",
					properties: {
						onset_ms: { type: "number" },
						duration_ms: { type: "number", minimum: 0 },
					},
					required: ["onset_ms", "duration_ms"],
				},
				minItems: 3,
				description:
					"Consecutive monophonic notes sorted by onset_ms ascending.",
			},
		},
		required: ["notes"],
	},
	invoke: async (input: unknown): Promise<number> => {
		const { notes } = input as {
			notes: { onset_ms: number; duration_ms: number }[];
		};
		if (!Array.isArray(notes) || notes.length < 3) {
			throw new Error(
				"compute-key-overlap-ratio: requires at least 3 consecutive notes",
			);
		}
		const sorted = [...notes].sort((a, b) => a.onset_ms - b.onset_ms);
		let ratioSum = 0;
		for (let i = 0; i < sorted.length - 1; i++) {
			const overlap =
				sorted[i].onset_ms + sorted[i].duration_ms - sorted[i + 1].onset_ms;
			ratioSum += overlap / sorted[i].duration_ms;
		}
		return ratioSum / (sorted.length - 1);
	},
};
