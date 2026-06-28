// apps/api/src/services/wasm-bridge.test.ts
import { describe, expect, it, vi } from "vitest";

// Argument-forwarding contract for the TS wrappers. Math correctness is covered
// by the Rust cargo test `chroma_dtw_roundtrip` in
// apps/api/src/wasm/score-analysis/src/chroma_dtw.rs. This file pins only the
// TS wrapper's forwarding behavior — that the bridge actually loads its WASM
// imports and passes arguments through without rearrangement.

const mockAlignChunkChroma = vi.fn();
const mockAlignChunkNotes = vi.fn();
const mockAnalyzeTier1 = vi.fn();
const mockAnalyzeTier2 = vi.fn();
const mockSelectTeachingMoment = vi.fn();
const mockIdentifyPiece = vi.fn();

vi.mock("../wasm/score-analysis/pkg/score_analysis", () => ({
	align_chunk_chroma: mockAlignChunkChroma,
	align_chunk_notes: mockAlignChunkNotes,
	analyze_tier1: mockAnalyzeTier1,
	analyze_tier2: mockAnalyzeTier2,
	select_teaching_moment: mockSelectTeachingMoment,
}));

vi.mock("../wasm/piece-identify/pkg/piece_identify", () => ({
	identify_piece: mockIdentifyPiece,
}));

describe("alignChunkChroma", () => {
	it("forwards all five arguments to align_chunk_chroma and returns its result", async () => {
		const { alignChunkChroma } = await import("./wasm-bridge");
		const fakeResult = {
			bar_min: 0,
			bar_max: 4,
			cost: 0.1,
			bar_per_frame: [0, 0, 1, 1, 2, 2, 3, 3],
		};
		mockAlignChunkChroma.mockReturnValue(fakeResult);

		const audioBytes = new Uint8Array(12 * 4);
		const bars: never[] = [];
		const result = alignChunkChroma(audioBytes, 1, bars, 50.0, 5.0);

		expect(mockAlignChunkChroma).toHaveBeenCalledWith(
			audioBytes,
			1,
			bars,
			50.0,
			5.0,
		);
		expect(result).toBe(fakeResult);
	});
});

describe("alignChunkNotes", () => {
	it("forwards all eight arguments to align_chunk_notes and returns its result", async () => {
		const { alignChunkNotes } = await import("./wasm-bridge");
		const fakeBarMap = {
			chunk_index: 2,
			bar_start: 4,
			bar_end: 7,
			alignments: [],
			confidence: 0.9,
			is_reanchored: false,
		};
		mockAlignChunkNotes.mockReturnValue(fakeBarMap);

		const audioBytes = new Uint8Array(12 * 4);
		const perfNotes: never[] = [];
		const bars: never[] = [];
		const result = alignChunkNotes(audioBytes, 1, perfNotes, bars, 50.0, 5.0, 2, 0.1);

		expect(mockAlignChunkNotes).toHaveBeenCalledWith(
			audioBytes,
			1,
			perfNotes,
			bars,
			50.0,
			5.0,
			2,
			0.1,
		);
		expect(result).toBe(fakeBarMap);
	});
});

describe("selectTeachingMoment", () => {
	it("forwards chunks, baselines, and recent observations", async () => {
		const { selectTeachingMoment } = await import("./wasm-bridge");
		mockSelectTeachingMoment.mockReturnValue(null);
		const chunks = [
			{ chunk_index: 0, scores: [1, 2, 3, 4, 5, 6] as const },
		];
		const baselines = {
			dynamics: 1,
			timing: 1,
			pedaling: 1,
			articulation: 1,
			phrasing: 1,
			interpretation: 1,
		};
		const recent = [{ dimension: "phrasing" }];

		selectTeachingMoment(chunks as never, baselines, recent);

		expect(mockSelectTeachingMoment).toHaveBeenCalledWith(
			chunks,
			baselines,
			recent,
		);
	});
});

describe("identifyPiece", () => {
	it("forwards notes, artifact JSON, and threshold to identify_piece", async () => {
		const { identifyPiece } = await import("./wasm-bridge");
		mockIdentifyPiece.mockReturnValue(
			JSON.stringify({ piece_id: "p", composer: "c", title: "t", margin: 0.2, locked: true }),
		);
		const notes = [{ pitch: 60, onset: 0, offset: 0.5, velocity: 80 }];
		identifyPiece(notes, '{"version":"v2","onset_tol_ms":50,"pieces":[]}', 0.0935);
		expect(mockIdentifyPiece).toHaveBeenCalledWith(notes, '{"version":"v2","onset_tol_ms":50,"pieces":[]}', 0.0935);
	});
});
