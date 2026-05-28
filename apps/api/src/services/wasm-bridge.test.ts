// apps/api/src/services/wasm-bridge.test.ts
import { describe, expect, it, vi } from "vitest";

// Argument-forwarding contract for the TS wrappers. Math correctness is covered
// by the Rust cargo test `chroma_dtw_roundtrip` in
// apps/api/src/wasm/score-analysis/src/chroma_dtw.rs. This file pins only the
// TS wrapper's forwarding behavior — that the bridge actually loads its WASM
// imports and passes arguments through without rearrangement.

const mockAlignChunkChroma = vi.fn();
const mockAnalyzeTier1 = vi.fn();
const mockAnalyzeTier2 = vi.fn();
const mockSelectTeachingMoment = vi.fn();
const mockNgramRecall = vi.fn();
const mockRerankCandidates = vi.fn();
const mockDtwConfirm = vi.fn();

vi.mock("../wasm/score-analysis/pkg/score_analysis", () => ({
	align_chunk_chroma: mockAlignChunkChroma,
	analyze_tier1: mockAnalyzeTier1,
	analyze_tier2: mockAnalyzeTier2,
	select_teaching_moment: mockSelectTeachingMoment,
}));

vi.mock("../wasm/piece-identify/pkg/piece_identify", () => ({
	ngram_recall: mockNgramRecall,
	rerank_candidates: mockRerankCandidates,
	dtw_confirm: mockDtwConfirm,
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

describe("ngramRecall", () => {
	it("forwards notes and index to ngram_recall", async () => {
		const { ngramRecall } = await import("./wasm-bridge");
		mockNgramRecall.mockReturnValue([]);
		const notes = [{ pitch: 60, onset: 0, offset: 0.5, velocity: 80 }];
		const index = { "60,62,64": [["piece-a", 1] as [string, number]] };

		ngramRecall(notes, index);

		expect(mockNgramRecall).toHaveBeenCalledWith(notes, index);
	});
});
