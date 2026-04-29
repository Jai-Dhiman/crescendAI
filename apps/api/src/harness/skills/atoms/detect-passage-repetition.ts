import type { ToolDefinition } from "../../loop/types";

export type RepetitionEntry = {
	bar_range: [number, number];
	attempt_count: number;
	first_attempt_ms: number;
	last_attempt_ms: number;
};

function barOverlapFraction(a: [number, number], b: [number, number]): number {
	const overlapStart = Math.max(a[0], b[0]);
	const overlapEnd = Math.min(a[1], b[1]);
	if (overlapEnd < overlapStart) return 0;
	const overlapBars = overlapEnd - overlapStart + 1;
	const aLength = a[1] - a[0] + 1;
	const bLength = b[1] - b[0] + 1;
	return overlapBars / Math.min(aLength, bLength);
}

export const detectPassageRepetition: ToolDefinition = {
	name: "detect-passage-repetition",
	description:
		"Detects bar ranges practiced 2+ times in a session. Groups attempts by >= 50% bar overlap. Returns only groups with attempt_count >= 2, sorted by attempt_count descending.",
	input_schema: {
		type: "object",
		properties: {
			attempts: {
				type: "array",
				items: {
					type: "object",
					properties: {
						bar_range: {
							type: "array",
							items: { type: "integer" },
							minItems: 2,
							maxItems: 2,
						},
						time_ms: { type: "number" },
					},
					required: ["bar_range", "time_ms"],
				},
			},
		},
		required: ["attempts"],
	},
	invoke: async (input: unknown): Promise<RepetitionEntry[]> => {
		const { attempts } = input as {
			attempts: { bar_range: [number, number]; time_ms: number }[];
		};
		const groups: { bar_range: [number, number]; times: number[] }[] = [];
		for (const attempt of attempts) {
			const match = groups.find(
				(g) => barOverlapFraction(g.bar_range, attempt.bar_range) >= 0.5,
			);
			if (match) {
				match.times.push(attempt.time_ms);
			} else {
				groups.push({ bar_range: attempt.bar_range, times: [attempt.time_ms] });
			}
		}
		return groups
			.filter((g) => g.times.length >= 2)
			.map((g) => ({
				bar_range: g.bar_range,
				attempt_count: g.times.length,
				first_attempt_ms: Math.min(...g.times),
				last_attempt_ms: Math.max(...g.times),
			}))
			.sort((a, b) => b.attempt_count - a.attempt_count);
	},
};
