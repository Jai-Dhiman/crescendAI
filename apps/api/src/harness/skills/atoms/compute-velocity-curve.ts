import type { ToolDefinition } from "../../loop/types";

export type VelocityCurve = {
	bar: number;
	mean_velocity: number;
	p90_velocity: number;
};

function p90(values: number[]): number {
	if (values.length === 0) return 0;
	const sorted = [...values].sort((a, b) => a - b);
	const idx = 0.9 * (sorted.length - 1);
	const lo = Math.floor(idx),
		hi = Math.ceil(idx);
	return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
}

export const computeVelocityCurve: ToolDefinition = {
	name: "compute-velocity-curve",
	description:
		"Computes per-bar mean and p90 MIDI velocity for notes in a bar range. Returns exactly bar_range[1] - bar_range[0] + 1 entries, ordered by bar ascending.",
	input_schema: {
		type: "object",
		properties: {
			bar_range: {
				type: "array",
				items: { type: "integer" },
				minItems: 2,
				maxItems: 2,
			},
			notes: {
				type: "array",
				items: {
					type: "object",
					properties: {
						onset_ms: { type: "number" },
						velocity: { type: "number", minimum: 0, maximum: 127 },
						bar: { type: "integer" },
					},
					required: ["onset_ms", "velocity", "bar"],
				},
			},
		},
		required: ["bar_range", "notes"],
	},
	invoke: async (input: unknown): Promise<VelocityCurve[]> => {
		const { bar_range, notes } = input as {
			bar_range: [number, number];
			notes: { onset_ms: number; velocity: number; bar: number }[];
		};
		const [startBar, endBar] = bar_range;
		const result: VelocityCurve[] = [];
		for (let bar = startBar; bar <= endBar; bar++) {
			const barNotes = notes.filter((n) => n.bar === bar);
			if (barNotes.length === 0) {
				result.push({ bar, mean_velocity: 0, p90_velocity: 0 });
				continue;
			}
			const vels = barNotes.map((n) => n.velocity);
			const mean_velocity = vels.reduce((a, b) => a + b, 0) / vels.length;
			result.push({ bar, mean_velocity, p90_velocity: p90(vels) });
		}
		return result;
	},
};
