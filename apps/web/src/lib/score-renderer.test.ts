// apps/web/src/lib/score-renderer.test.ts
// Note: vi.stubGlobal, vi.resetModules, vi.doMock are not available in this
// Vitest version (4.0.18). Worker is mocked via globalThis in beforeEach/afterEach.
// api is mocked at module level via vi.mock() to avoid dynamic import issues.
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";

vi.mock("./api", () => ({
  api: {
    scores: {
      getData: vi.fn().mockResolvedValue(new ArrayBuffer(8)),
    },
  },
}));

vi.mock("./sentry", () => ({
  Sentry: {
    captureException: vi.fn(),
  },
}));

// Worker mock: intercepts postMessage and captures onmessage/onerror handlers.
const mockPostMessage = vi.fn();
let workerInstance: MockWorker | null = null;

class MockWorker {
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  constructor() {
    workerInstance = this;
  }
  postMessage(data: unknown) {
    mockPostMessage(data);
  }
}

const FAKE_IR = {
  pieceId: "test-piece",
  verovioVersion: "4.0.0",
  pageWidth: 2400,
  pages: [{ pageN: 1, viewBox: "0 0 2400 800", width: 2400, height: 800, systemBboxes: [] }],
  bars: [],
  notes: {},
};

function simulateWorkerResponse(requestId: string, payload: unknown) {
  if (workerInstance?.onmessage) {
    workerInstance.onmessage(new MessageEvent("message", { data: { requestId, payload } }));
  }
}

let originalWorker: unknown;

describe("ScoreRenderer.load", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    workerInstance = null;
    originalWorker = (globalThis as Record<string, unknown>).Worker;
    (globalThis as Record<string, unknown>).Worker = MockWorker;
  });

  afterEach(() => {
    (globalThis as Record<string, unknown>).Worker = originalWorker;
  });

  it("resolves with {ir, pageSvgs} after a successful load message exchange", async () => {
    const { ScoreRenderer } = await import("./score-renderer");
    const renderer = new ScoreRenderer();

    const payload = { ir: FAKE_IR, pageSvgs: ["<svg>page1</svg>"] };

    const loadPromise = renderer.load("test-piece");

    // Wait for ensureBytes (async fetch) to complete and postMessage to be called.
    await new Promise((r) => setTimeout(r, 10));

    const sentMsg = mockPostMessage.mock.calls[0]?.[0] as { requestId: string };
    simulateWorkerResponse(sentMsg.requestId, payload);

    const result = await loadPromise;
    expect(result).not.toBe("failed");
    if (result === "failed") return;
    expect(result.ir.pieceId).toBe("test-piece");
    expect(result.pageSvgs).toEqual(["<svg>page1</svg>"]);
  });

  it("getIR returns the cached IR synchronously after load resolves", async () => {
    const { ScoreRenderer } = await import("./score-renderer");
    const renderer = new ScoreRenderer();
    const payload = { ir: FAKE_IR, pageSvgs: ["<svg>page1</svg>"] };

    const loadPromise = renderer.load("test-piece");
    await new Promise((r) => setTimeout(r, 10));
    const sentMsg = mockPostMessage.mock.calls[0]?.[0] as { requestId: string };
    simulateWorkerResponse(sentMsg.requestId, payload);
    await loadPromise;

    const ir = renderer.getIR("test-piece");
    expect(ir).not.toBeNull();
    expect(ir?.pieceId).toBe("test-piece");
  });

  it("getIR returns null when load has not been called", async () => {
    const { ScoreRenderer } = await import("./score-renderer");
    const renderer = new ScoreRenderer();
    expect(renderer.getIR("nonexistent-piece")).toBeNull();
  });
});
