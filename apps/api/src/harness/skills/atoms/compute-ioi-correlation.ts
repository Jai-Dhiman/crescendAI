import type { ToolDefinition } from "../../loop/types";

function pearsonR(xs: number[], ys: number[]): number {
	const n = xs.length;
	const mx = xs.reduce((a, b) => a + b, 0) / n;
	const my = ys.reduce((a, b) => a + b, 0) / n;
	let num = 0,
		dx2 = 0,
		dy2 = 0;
	for (let i = 0; i < n; i++) {
		const dx = xs[i] - mx,
			dy = ys[i] - my;
		num += dx * dy;
		dx2 += dx * dx;
		dy2 += dy * dy;
	}
	const denom = Math.sqrt(dx2 * dy2);
	return denom === 0 ? 0 : num / denom;
}

export const computeIoiCorrelation: ToolDefinition = {
	name: "compute-ioi-correlation",
	description:
		"Computes Pearson r between performance inter-onset intervals and score inter-onset intervals. Returns null when fewer than 4 aligned notes exist (correlation unreliable). Aligned notes must have expected_onset_ms set (non-null).",
	input_schema: {
		type: "object",
		properties: {
			notes: {
				type: "array",
				items: {
					type: "object",
					properties: {
						onset_ms: { type: "number" },
						expected_onset_ms: { type: ["number", "null"] },
					},
					required: ["onset_ms", "expected_onset_ms"],
				},
				description:
					"Aligned notes sorted by onset_ms ascending. Notes with null expected_onset_ms are excluded.",
			},
		},
		required: ["notes"],
	},
	invoke: async (input: unknown): Promise<number | null> => {
		const { notes } = input as {
			notes: { onset_ms: number; expected_onset_ms: number | null }[];
		};
		const aligned = notes.filter((n) => n.expected_onset_ms !== null);
		if (aligned.length < 4) return null;
		const perfIOIs: number[] = [];
		const scoreIOIs: number[] = [];
		for (let i = 0; i < aligned.length - 1; i++) {
			perfIOIs.push(aligned[i + 1].onset_ms - aligned[i].onset_ms);
			scoreIOIs.push(
				aligned[i + 1].expected_onset_ms! - aligned[i].expected_onset_ms!,
			);
		}
		return pearsonR(perfIOIs, scoreIOIs);
	},
};
