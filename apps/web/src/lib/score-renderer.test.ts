// apps/web/src/lib/score-renderer.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";

class MockWorker {
  onmessage: ((e: MessageEvent) => void) | null = null;
  postMessage = vi.fn((msg: { requestId: string }) => {
    const handler = this.onmessage;
    Promise.resolve().then(() => {
      handler?.({
        data: { requestId: msg.requestId, svg: "<svg>mock</svg>" },
      } as MessageEvent);
    });
  });
  terminate = vi.fn();
}

const mockGetData = vi.fn().mockResolvedValue(new ArrayBuffer(8));
vi.mock("./api", () => ({
  api: {
    scores: {
      getData: (...args: unknown[]) => mockGetData(...args),
    },
  },
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("Worker", MockWorker);
  vi.resetModules();
});

describe("scoreRenderer.getClip", () => {
  it("resolves with the SVG string returned by the Worker", async () => {
    const { scoreRenderer } = await import("./score-renderer");
    const svg = await scoreRenderer.getClip("chopin.ballades.1", 1, 4);
    expect(svg).toBe("<svg>mock</svg>");
    expect(mockGetData).toHaveBeenCalledWith("chopin.ballades.1");
  });
});

describe("scoreRenderer.getClip error paths", () => {
  it("rejects when the Worker returns an error response", async () => {
    const ErrorWorker = class {
      onmessage: ((e: MessageEvent) => void) | null = null;
      postMessage = vi.fn((msg: { requestId: string }) => {
        const handler = this.onmessage;
        Promise.resolve().then(() => {
          handler?.({
            data: { requestId: msg.requestId, error: "render failed" },
          } as MessageEvent);
        });
      });
      terminate = vi.fn();
    };
    vi.stubGlobal("Worker", ErrorWorker);
    const { scoreRenderer } = await import("./score-renderer");
    await expect(scoreRenderer.getClip("chopin.ballades.1", 1, 4)).rejects.toThrow("render failed");
  });

  it("rejects when api.scores.getData fails", async () => {
    mockGetData.mockRejectedValueOnce(new Error("network error"));
    const { scoreRenderer } = await import("./score-renderer");
    await expect(scoreRenderer.getClip("chopin.ballades.1", 1, 4)).rejects.toThrow("network error");
  });
});
