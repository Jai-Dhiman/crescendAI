// apps/api/src/services/wasm-bridge.test.ts
import { describe, expect, it, vi, beforeEach } from "vitest";

// We mock the WASM module path so scoreAnalysisModule can be controlled.
// The mock must use a factory that does NOT use importOriginal, which is
// not supported in the bun/vitest workers environment.

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
});
