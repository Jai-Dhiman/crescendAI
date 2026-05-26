// apps/web/src/lib/score-cursor.test.ts
// Note: vi.stubGlobal is not available in Vitest 4.0.18.
// requestAnimationFrame/cancelAnimationFrame are mocked via globalThis assignment.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ScoreIR } from "./score-ir";

vi.mock("./sentry", () => ({
  Sentry: {
    captureException: vi.fn(),
  },
}));

// jsdom does not implement requestAnimationFrame; provide a manual scheduler.
let rafCallback: FrameRequestCallback | null = null;
const mockCancelAnimationFrame = vi.fn();

// Override globals before any test runs.
(globalThis as Record<string, unknown>).requestAnimationFrame = (cb: FrameRequestCallback) => {
  rafCallback = cb;
  return 1;
};
(globalThis as Record<string, unknown>).cancelAnimationFrame = mockCancelAnimationFrame;

function flushRaf() {
  if (rafCallback) {
    const cb = rafCallback;
    rafCallback = null;
    cb(performance.now());
  }
}

// Minimal IR: 2 bars, 2 notes per bar, one page.
const FAKE_IR: ScoreIR = {
  pieceId: "test",
  verovioVersion: "4.0.0",
  pageWidth: 2400,
  pages: [{ pageN: 1, viewBox: "0 0 2400 800", width: 2400, height: 800, systemBboxes: [] }],
  bars: [
    {
      barNumber: 1,
      measureOn: "m1",
      pageN: 1,
      bbox: { x: 100, y: 0, w: 0, h: 0 },
      noteIds: ["n1", "n2"],
      qstampStart: 0,
      qstampEnd: 4,
    },
    {
      barNumber: 2,
      measureOn: "m2",
      pageN: 1,
      bbox: { x: 500, y: 0, w: 0, h: 0 },
      noteIds: ["n3", "n4"],
      qstampStart: 4,
      qstampEnd: 8,
    },
  ],
  notes: {
    n1: { id: "n1", bbox: { x: 100, y: 200, w: 0, h: 0 }, qstamp: 0, staff: 1 },
    n2: { id: "n2", bbox: { x: 300, y: 200, w: 0, h: 0 }, qstamp: 2, staff: 1 },
    n3: { id: "n3", bbox: { x: 500, y: 200, w: 0, h: 0 }, qstamp: 4, staff: 1 },
    n4: { id: "n4", bbox: { x: 700, y: 200, w: 0, h: 0 }, qstamp: 6, staff: 1 },
  },
};

describe("ScoreCursor", () => {
  let container: HTMLElement;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    rafCallback = null;
    vi.clearAllMocks();
  });

  afterEach(() => {
    document.body.removeChild(container);
  });

  it("mounts an overlay svg on start() and removes it on stop()", async () => {
    const { ScoreCursor } = await import("./score-cursor");
    const cursor = new ScoreCursor({
      pieceId: "test",
      container,
      ir: FAKE_IR,
      qstampSource: () => null,
    });
    cursor.start();
    expect(container.querySelector("svg.score-cursor-overlay")).not.toBeNull();
    cursor.stop();
    expect(container.querySelector("svg.score-cursor-overlay")).toBeNull();
  });

  it("hides the overlay line when qstampSource returns null", async () => {
    const { ScoreCursor } = await import("./score-cursor");
    const cursor = new ScoreCursor({
      pieceId: "test",
      container,
      ir: FAKE_IR,
      qstampSource: () => null,
    });
    cursor.start();
    flushRaf();
    const line = container.querySelector("svg.score-cursor-overlay line") as SVGLineElement | null;
    expect(line?.getAttribute("visibility")).toBe("hidden");
    cursor.stop();
  });

  it("positions the cursor line within 1px of the expected interpolated x for a qstamp inside bar 1", async () => {
    const { ScoreCursor } = await import("./score-cursor");
    // qstamp = 1: halfway between n1 (x=100, qstamp=0) and n2 (x=300, qstamp=2)
    // expected interpolated x = 100 + (1-0)/(2-0) * (300-100) = 200
    const cursor = new ScoreCursor({
      pieceId: "test",
      container,
      ir: FAKE_IR,
      qstampSource: () => 1,
    });
    cursor.start();
    flushRaf();
    const line = container.querySelector("svg.score-cursor-overlay line") as SVGLineElement | null;
    expect(line).not.toBeNull();
    expect(line?.getAttribute("visibility")).not.toBe("hidden");
    const x1 = parseFloat(line?.getAttribute("x1") ?? "NaN");
    expect(Math.abs(x1 - 200)).toBeLessThan(1);
    cursor.stop();
  });

  it("start() is idempotent: calling it twice mounts only one overlay and stop() once fully cancels", async () => {
    const { ScoreCursor } = await import("./score-cursor");
    const cursor = new ScoreCursor({
      pieceId: "test",
      container,
      ir: FAKE_IR,
      qstampSource: () => null,
    });
    cursor.start();
    cursor.start(); // second call must be a no-op
    const overlays = container.querySelectorAll("svg.score-cursor-overlay");
    expect(overlays.length).toBe(1);
    cursor.stop();
    expect(container.querySelector("svg.score-cursor-overlay")).toBeNull();
    expect(mockCancelAnimationFrame).toHaveBeenCalledTimes(1);
  });

  it("keeps the rAF loop alive and captures exception via Sentry when qstampSource throws", async () => {
    const { Sentry } = await import("./sentry");
    const { ScoreCursor } = await import("./score-cursor");
    const cursor = new ScoreCursor({
      pieceId: "test",
      container,
      ir: FAKE_IR,
      qstampSource: () => { throw new Error("source exploded"); },
    });
    cursor.start();
    flushRaf();
    expect(Sentry.captureException).toHaveBeenCalled();
    // rAF loop must have re-scheduled (rafCallback is set again)
    expect(rafCallback).not.toBeNull();
    cursor.stop();
  });
});
