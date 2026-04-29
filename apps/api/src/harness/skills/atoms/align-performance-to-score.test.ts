import { test, expect } from "vitest";
import { alignPerformanceToScore } from "./align-performance-to-score";

test("alignPerformanceToScore: two perf notes align to two score notes with matching pitch and near onset", async () => {
	const perf_notes = [
		{ pitch: 60, onset_ms: 1000 },
		{ pitch: 64, onset_ms: 1500 },
	];
	const score_notes = [
		{ pitch: 60, expected_onset_ms: 1000, bar: 12 },
		{ pitch: 64, expected_onset_ms: 1450, bar: 12 },
	];
	const result = (await alignPerformanceToScore.invoke({
		perf_notes,
		score_notes,
	})) as {
		perf_index: number;
		score_index: number;
		expected_onset_ms: number | null;
		bar: number;
	}[];
	expect(result).toHaveLength(2);
	expect(result[0]).toEqual({
		perf_index: 0,
		score_index: 0,
		expected_onset_ms: 1000,
		bar: 12,
	});
	expect(result[1]).toEqual({
		perf_index: 1,
		score_index: 1,
		expected_onset_ms: 1450,
		bar: 12,
	});
});
