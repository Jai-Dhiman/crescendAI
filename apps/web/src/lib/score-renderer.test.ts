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

vi.stubGlobal("Worker", MockWorker);

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
