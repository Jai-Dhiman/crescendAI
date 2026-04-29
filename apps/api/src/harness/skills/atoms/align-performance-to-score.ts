import type { ToolDefinition } from "../../loop/types";

export type Alignment = {
	perf_index: number;
	score_index: number;
	expected_onset_ms: number | null;
	bar: number;
};

const UNALIGNED_COST_THRESHOLD = 500;

function cost(
	perfOnset: number,
	scoreOnset: number,
	perfPitch: number,
	scorePitch: number,
	scoreRange: number,
): number {
	const onsetCost =
		scoreRange > 0 ? Math.abs(perfOnset - scoreOnset) / scoreRange : 0;
	const pitchCost = perfPitch !== scorePitch ? 100 : 0;
	return onsetCost + pitchCost;
}

export const alignPerformanceToScore: ToolDefinition = {
	name: "align-performance-to-score",
	description:
		"Aligns AMT performance notes to a score MIDI via DTW on (onset, pitch). Returns one Alignment entry per performance note. score_index is -1 and expected_onset_ms is null for unaligned notes (cost > 500).",
	input_schema: {
		type: "object",
		properties: {
			perf_notes: {
				type: "array",
				items: {
					type: "object",
					properties: {
						pitch: { type: "integer", minimum: 0, maximum: 127 },
						onset_ms: { type: "number" },
					},
					required: ["pitch", "onset_ms"],
				},
			},
			score_notes: {
				type: "array",
				items: {
					type: "object",
					properties: {
						pitch: { type: "integer", minimum: 0, maximum: 127 },
						expected_onset_ms: { type: "number" },
						bar: { type: "integer" },
					},
					required: ["pitch", "expected_onset_ms", "bar"],
				},
			},
		},
		required: ["perf_notes", "score_notes"],
	},
	invoke: async (input: unknown): Promise<Alignment[]> => {
		const { perf_notes, score_notes } = input as {
			perf_notes: { pitch: number; onset_ms: number }[];
			score_notes: { pitch: number; expected_onset_ms: number; bar: number }[];
		};
		if (score_notes.length === 0) {
			return perf_notes.map((_, i) => ({
				perf_index: i,
				score_index: -1,
				expected_onset_ms: null,
				bar: -1,
			}));
		}
		const onsets = score_notes.map((s) => s.expected_onset_ms);
		const scoreRange = Math.max(...onsets) - Math.min(...onsets);

		// Build DTW cost matrix
		const P = perf_notes.length,
			S = score_notes.length;
		const dp = Array.from({ length: P + 1 }, () =>
			new Array(S + 1).fill(Infinity),
		);
		dp[0][0] = 0;
		for (let i = 1; i <= P; i++) dp[i][0] = Infinity;
		for (let j = 1; j <= S; j++) dp[0][j] = Infinity;

		for (let i = 1; i <= P; i++) {
			for (let j = 1; j <= S; j++) {
				const c = cost(
					perf_notes[i - 1].onset_ms,
					score_notes[j - 1].expected_onset_ms,
					perf_notes[i - 1].pitch,
					score_notes[j - 1].pitch,
					scoreRange,
				);
				dp[i][j] = c + Math.min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]);
			}
		}

		// Traceback (monotonic): only record entry when i decreases
		const path: [number, number][] = [];
		let i = P,
			j = S;
		while (i > 0 && j > 0) {
			const diag = dp[i - 1][j - 1],
				up = dp[i - 1][j],
				left = dp[i][j - 1];
			if (diag <= up && diag <= left) {
				path.unshift([i - 1, j - 1]);
				i--;
				j--;
			} else if (up <= left) {
				path.unshift([i - 1, -1]);
				i--;
			} else {
				j--;
			}
		}
		while (i > 0) {
			path.unshift([i - 1, -1]);
			i--;
		}

		return path.map(([pi, si]) => {
			if (si === -1) {
				return {
					perf_index: pi,
					score_index: -1,
					expected_onset_ms: null,
					bar: -1,
				};
			}
			const c = cost(
				perf_notes[pi].onset_ms,
				score_notes[si].expected_onset_ms,
				perf_notes[pi].pitch,
				score_notes[si].pitch,
				scoreRange,
			);
			if (c > UNALIGNED_COST_THRESHOLD) {
				return {
					perf_index: pi,
					score_index: -1,
					expected_onset_ms: null,
					bar: -1,
				};
			}
			return {
				perf_index: pi,
				score_index: si,
				expected_onset_ms: score_notes[si].expected_onset_ms,
				bar: score_notes[si].bar,
			};
		});
	},
};
