// apps/api/src/services/inference.test.ts
import { describe, expect, it } from "vitest";
import { callMuqEndpoint } from "./inference";
import type { MuqResult } from "./inference";

// Minimal Bindings stub — only MUQ_ENDPOINT is needed for these tests
function makeEnv(url: string): { MUQ_ENDPOINT: string } {
  return { MUQ_ENDPOINT: url } as unknown as Parameters<typeof callMuqEndpoint>[0];
}

const VALID_SCORES = {
  dynamics: 0.6,
  timing: 0.7,
  pedaling: 0.5,
  articulation: 0.8,
  phrasing: 0.65,
  interpretation: 0.72,
};

function makeChromaB64(nFrames: number): string {
  // 12 * nFrames float32 values, all zeros, row-major
  const buf = new ArrayBuffer(12 * nFrames * 4);
  const view = new DataView(buf);
  for (let i = 0; i < 12 * nFrames; i++) {
    view.setFloat32(i * 4, 0.5, true); // LE, value 0.5
  }
  const bytes = new Uint8Array(buf);
  return btoa(String.fromCharCode(...bytes));
}

describe("callMuqEndpoint chroma fields", () => {
  it("returns chromaBytes decoded from chroma_b64 when present in response", async () => {
    const nFrames = 10;
    const b64 = makeChromaB64(nFrames);

    const mockResponse = JSON.stringify({
      predictions: VALID_SCORES,
      chroma_b64: b64,
      chroma_frames: nFrames,
      chroma_frame_rate_hz: 50.0,
    });

    const { parseMuqResponse } = await import("./inference");
    const result = parseMuqResponse(JSON.parse(mockResponse));

    expect(result.chromaBytes).not.toBeNull();
    expect(result.chromaBytes!.byteLength).toBe(12 * nFrames * 4);
    expect(result.chromaFrames).toBe(nFrames);
    expect(result.chromaFrameRateHz).toBe(50.0);
  });

  it("returns chromaBytes=null when chroma_b64 absent in response", async () => {
    const { parseMuqResponse } = await import("./inference");
    const result = parseMuqResponse({
      predictions: VALID_SCORES,
    });
    expect(result.chromaBytes).toBeNull();
    expect(result.chromaFrames).toBe(0);
  });
});
