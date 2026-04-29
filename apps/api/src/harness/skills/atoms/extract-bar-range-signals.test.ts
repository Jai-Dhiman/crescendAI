import { test, expect } from "vitest";
import { extractBarRangeSignals } from "./extract-bar-range-signals";

test("extractBarRangeSignals: only chunks overlapping [12,16] contribute; out-of-range chunk excluded", async () => {
	// chunk_a covers bars 10-14 (overlaps [12,16])
	// chunk_b covers bars 14-18 (overlaps [12,16])
	// chunk_c covers bars 18-22 (does NOT overlap [12,16])
	const chunks = [
		{
			chunk_id: "chunk_a",
			bar_coverage: [10, 14] as [number, number],
			muq_scores: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
			midi_notes: [
				{ pitch: 60, onset_ms: 100, duration_ms: 200, velocity: 70, bar: 12 },
			],
			pedal_cc: [{ time_ms: 50, value: 127 }],
			alignment: [
				{ perf_index: 0, score_index: 0, expected_onset_ms: 100, bar: 12 },
			],
		},
		{
			chunk_id: "chunk_b",
			bar_coverage: [14, 18] as [number, number],
			muq_scores: [0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
			midi_notes: [
				{ pitch: 64, onset_ms: 500, duration_ms: 200, velocity: 75, bar: 14 },
			],
			pedal_cc: [],
			alignment: [],
		},
		{
			chunk_id: "chunk_c",
			bar_coverage: [18, 22] as [number, number],
			muq_scores: [0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
			midi_notes: [
				{ pitch: 67, onset_ms: 900, duration_ms: 200, velocity: 80, bar: 20 },
			],
			pedal_cc: [],
			alignment: [],
		},
	];
	const result = (await extractBarRangeSignals.invoke({
		bar_range: [12, 16],
		chunks,
	})) as {
		muq_scores: number[][];
		midi_notes: unknown[];
		pedal_cc: unknown[];
		alignment: unknown[];
	};
	// Only chunk_a and chunk_b overlap [12,16]
	expect(result.muq_scores).toHaveLength(2);
	// chunk_c's note (bar=20) should not appear
	const bars = (result.midi_notes as { bar?: number }[]).map((n) => n.bar);
	expect(bars.includes(20)).toBe(false);
});
