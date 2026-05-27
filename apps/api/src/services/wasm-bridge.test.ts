// apps/api/src/services/wasm-bridge.test.ts
import { describe, expect, it, vi } from "vitest";

// We mock requireScoreAnalysis at the module level so we can control whether
// it returns a fake WASM module or throws. This is the only way to test
// alignChunkChroma without loading the real WASM binary (scoreAnalysisModule
// is always null in the node vitest environment).

// The mock must be hoisted before any import of wasm-bridge.
const mockAlignChunkChroma = vi.fn();
const mockRequireScoreAnalysis = vi.fn();

vi.mock("./wasm-bridge", async (importOriginal) => {
  // Import real module to get type exports and other exports unchanged.
  const original = await importOriginal<typeof import("./wasm-bridge")>();
  return {
    ...original,
    // Override alignChunkChroma to use our controllable mock internals.
    // We re-implement it here so we can inject mockRequireScoreAnalysis.
    alignChunkChroma: (
      audioChromaBytes: Uint8Array,
      chromaFrames: number,
      scoreBars: unknown[],
      frameRateHz: number,
      decimHz: number,
    ) => {
      return mockRequireScoreAnalysis().align_chunk_chroma(
        audioChromaBytes,
        chromaFrames,
        scoreBars,
        frameRateHz,
        decimHz,
      );
    },
  };
});

describe("alignChunkChroma", () => {
  it("is exported as a function from wasm-bridge", async () => {
    const { alignChunkChroma } = await import("./wasm-bridge");
    expect(typeof alignChunkChroma).toBe("function");
  });

  it("throws 'score-analysis WASM not initialized' when scoreAnalysisModule is null", async () => {
    mockRequireScoreAnalysis.mockImplementation(() => {
      throw new Error("score-analysis WASM not initialized");
    });
    const { alignChunkChroma } = await import("./wasm-bridge");
    const audioBytes = new Uint8Array(12 * 4); // 1 frame, 12 pitches
    expect(() =>
      alignChunkChroma(audioBytes, 1, [], 50.0, 5.0),
    ).toThrow("score-analysis WASM not initialized");
  });

  it("forwards all 5 arguments to align_chunk_chroma and returns its result typed as BarMapChroma", async () => {
    const fakeResult = {
      bar_min: 3,
      bar_max: 7,
      cost: 0.12,
      bar_per_frame: [3, 4, 5, 6, 7],
    };
    mockAlignChunkChroma.mockReturnValue(fakeResult);
    mockRequireScoreAnalysis.mockReturnValue({
      align_chunk_chroma: mockAlignChunkChroma,
    });

    const { alignChunkChroma } = await import("./wasm-bridge");
    const audioBytes = new Uint8Array(12 * 5 * 4); // 5 frames
    const scoreBars = [{ bar_number: 1, start_seconds: 0.0, notes: [] }];
    const result = alignChunkChroma(audioBytes, 5, scoreBars as never, 50.0, 5.0);

    expect(mockAlignChunkChroma).toHaveBeenCalledWith(
      audioBytes,
      5,
      scoreBars,
      50.0,
      5.0,
    );
    expect(result).toEqual(fakeResult);
    // Shape assertion: result is typed as BarMapChroma
    expect(typeof result.bar_min).toBe("number");
    expect(typeof result.bar_max).toBe("number");
    expect(typeof result.cost).toBe("number");
    expect(Array.isArray(result.bar_per_frame)).toBe(true);
  });
});
