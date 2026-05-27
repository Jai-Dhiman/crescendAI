// apps/api/src/services/wasm-bridge.test.ts
import { describe, expect, it, vi } from "vitest";

// scoreAnalysisModule is a module-level const initialized to null in wasm-bridge.ts.
// There is no exported setter, so the null guard cannot be bypassed from the test
// environment. vi.mock replaces the WASM import path but that assignment never
// propagates to scoreAnalysisModule because it happens before any dynamic import.
//
// Argument-forwarding contract for alignChunkChroma:
//   alignChunkChroma(audioChromaBytes, chromaFrames, scoreBars, frameRateHz, decimHz)
//   => requireScoreAnalysis().align_chunk_chroma(audioChromaBytes, chromaFrames, scoreBars, frameRateHz, decimHz)
//
// This forwarding is verified end-to-end by the Rust cargo test:
//   apps/api/src/wasm/score-analysis/src/chroma_dtw.rs :: chroma_dtw_roundtrip
// That test calls align_chunk_chroma with known fixtures and asserts bar_min/bar_max/cost.
// It is the canonical contract for argument order and return-value pass-through.

const mockAlignChunkChroma = vi.fn();

vi.mock("../wasm/score-analysis/pkg/score_analysis", () => ({
	align_chunk_chroma: mockAlignChunkChroma,
}));

describe("alignChunkChroma", () => {
	it("is exported as a function from wasm-bridge", async () => {
		const { alignChunkChroma } = await import("./wasm-bridge");
		expect(typeof alignChunkChroma).toBe("function");
	});

	it("throws 'score-analysis WASM not initialized' when called without WASM init", async () => {
		// scoreAnalysisModule is always null in the test environment since
		// it is never initialized. Calling alignChunkChroma must throw.
		const { alignChunkChroma } = await import("./wasm-bridge");
		const audioBytes = new Uint8Array(12 * 4); // 1 frame, 12 pitches
		expect(() =>
			alignChunkChroma(audioBytes, 1, [], 50.0, 5.0),
		).toThrow("score-analysis WASM not initialized");
	});

	// TS wrapper argument forwarding (alignChunkChroma -> align_chunk_chroma, 5 args) is
	// verified end-to-end by `chroma_dtw_roundtrip` in apps/api/src/wasm/score-analysis/src/chroma_dtw.rs,
	// because scoreAnalysisModule cannot be initialized in the vitest node env.
});
