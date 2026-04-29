import { describe, expect, it } from "vitest";
import { toEnrichedChunk } from "./session-brain";

describe("toEnrichedChunk", () => {
	it("converts onset seconds to ms and reshapes NoteAlignment to Alignment", () => {
		const muqScores = [0.6, 0.5, 0.7, 0.55, 0.6, 0.65];
		const perfNotes = [
			{ pitch: 60, onset: 1.0, offset: 1.5, velocity: 80 },
			{ pitch: 64, onset: 1.25, offset: 1.75, velocity: 75 },
		];
		const perfPedal = [
			{ time: 1.0, value: 100 },
			{ time: 1.4, value: 0 },
		];
		const alignments = [
			{
				perf_onset: 1.0,
				perf_pitch: 60,
				perf_velocity: 80,
				score_bar: 3,
				score_beat: 1.0,
				score_pitch: 60,
				onset_deviation_ms: 15,
			},
			{
				perf_onset: 1.25,
				perf_pitch: 64,
				perf_velocity: 75,
				score_bar: 3,
				score_beat: 2.0,
				score_pitch: 64,
				onset_deviation_ms: -10,
			},
		];
		const barCoverage: [number, number] = [3, 4];

		const result = toEnrichedChunk(
			0,
			muqScores,
			perfNotes,
			perfPedal,
			alignments,
			barCoverage,
		);

		expect(result.chunkIndex).toBe(0);
		expect(result.muq_scores).toEqual(muqScores);
		// onset in seconds → ms
		expect(result.midi_notes[0]?.onset_ms).toBe(1000);
		expect(result.midi_notes[1]?.onset_ms).toBe(1250);
		// duration = (offset - onset) * 1000
		expect(result.midi_notes[0]?.duration_ms).toBe(500);
		// pedal time seconds → ms
		expect(result.pedal_cc[0]?.time_ms).toBe(1000);
		expect(result.pedal_cc[1]?.time_ms).toBe(1400);
		// alignment: expected_onset_ms = perf_onset * 1000 - onset_deviation_ms
		expect(result.alignment[0]?.expected_onset_ms).toBe(985);
		expect(result.alignment[1]?.expected_onset_ms).toBe(1260);
		// alignment: bar from score_bar
		expect(result.alignment[0]?.bar).toBe(3);
		// alignment: score_index = array index
		expect(result.alignment[0]?.score_index).toBe(0);
		expect(result.alignment[1]?.score_index).toBe(1);
		expect(result.bar_coverage).toEqual([3, 4]);
	});

	it("returns empty arrays when perfNotes and perfPedal are empty", () => {
		const result = toEnrichedChunk(
			2,
			[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
			[],
			[],
			[],
			null,
		);
		expect(result.midi_notes).toEqual([]);
		expect(result.pedal_cc).toEqual([]);
		expect(result.alignment).toEqual([]);
		expect(result.bar_coverage).toBeNull();
	});
});
