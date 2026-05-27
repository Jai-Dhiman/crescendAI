// apps/web/src/components/ProofCard.autoplay.test.tsx
import { render } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import type { ProofCardManifest } from "../types/landing";

vi.mock("../lib/landing-analytics", () => ({ trackLandingEvent: vi.fn() }));

const MOCK_SCORE_IR = {
  pieceId: "chopin.nocturnes.9-2",
  verovioVersion: "6.1.0",
  pageWidth: 2400,
  pages: [{ pageN: 1, viewBox: "0 0 2400 800", width: 2400, height: 800, systemBboxes: [] }],
  bars: [
    { barNumber: 1, measureOn: "m1", pageN: 1, bbox: { x: 100, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 0, qstampEnd: 4 },
  ],
  notes: {},
};

const FIXTURE_EXERCISE = {
  type: "exercise_set",
  config: { sourcePassage: "bar 4", targetSkill: "Dynamics", exercises: [] },
};

const FIXTURE_MANIFEST: ProofCardManifest = {
  pieceId: "chopin.nocturnes.9-2",
  title: "Nocturne Op. 9 No. 2",
  era: "romantic",
  audioUrl: "/landing/card-1/recording.opus",
  scoreIRUrl: "/landing/card-1/scoreir.json",
  scoreSvgUrl: "/landing/card-1/score.svg",
  focusBar: 4,
  focusBarRange: [3, 5],
  diagnosis: "The diminuendo in bar 4 arrives too early.",
  exerciseUrl: "/landing/card-1/exercise.json",
  barTimeline: [{ bar: 1, tSec: 0.0 }, { bar: 4, tSec: 12.8 }],
  perBarScores: {
    1: { dynamics: 0.72, timing: 0.81, pedaling: 0.68, articulation: 0.75, phrasing: 0.70, interpretation: 0.74 },
    4: { dynamics: 0.52, timing: 0.78, pedaling: 0.60, articulation: 0.71, phrasing: 0.54, interpretation: 0.63 },
  },
};

function mockFetch() {
  globalThis.fetch = vi.fn().mockImplementation((url: string) => {
    if (String(url).includes("scoreir.json")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_SCORE_IR) });
    }
    if (String(url).includes("exercise.json")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(FIXTURE_EXERCISE) });
    }
    if (String(url).includes("score.svg")) {
      return Promise.resolve({ ok: true, text: () => Promise.resolve('<svg xmlns="http://www.w3.org/2000/svg"></svg>') });
    }
    return Promise.resolve({ ok: false, json: () => Promise.resolve(null) });
  });
}

describe("ProofCard scroll autoplay", () => {
  // ProofCard creates two IntersectionObservers: [0] autoplay, [1] prefetch.
  // We capture all callbacks; tests use callbacks[0] for autoplay.
  let observerCallbacks: IntersectionObserverCallback[];
  let mockPlay: ReturnType<typeof vi.fn>;
  let mockPause: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    observerCallbacks = [];
    mockFetch();
    mockPlay = vi.fn().mockResolvedValue(undefined);
    mockPause = vi.fn();

    // Stub IntersectionObserver — capture all callbacks (must be a class/constructor)
    class MockIO {
      observe = vi.fn();
      disconnect = vi.fn();
      constructor(cb: IntersectionObserverCallback) {
        observerCallbacks.push(cb);
      }
    }
    globalThis.IntersectionObserver = MockIO as unknown as typeof IntersectionObserver;

    // Stub HTMLMediaElement.play/pause (jsdom doesn't implement these)
    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      value: mockPlay,
    });
    Object.defineProperty(HTMLMediaElement.prototype, "pause", {
      configurable: true,
      value: mockPause,
    });

    // Ensure not reduced-motion
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: (query: string) => ({
        matches: false,
        media: query,
        addListener: vi.fn(),
        removeListener: vi.fn(),
      }),
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });

  it("calls audio.play() when intersection ratio crosses 60% threshold", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    // Wait for effects to flush so IntersectionObservers are registered
    await new Promise((r) => setTimeout(r, 0));

    // observerCallbacks[0] is the autoplay observer; [1] is the prefetch observer
    const autoplayCallback = observerCallbacks[0];
    expect(autoplayCallback).toBeDefined();

    // Simulate 60% intersection
    autoplayCallback(
      [{ intersectionRatio: 0.65, isIntersecting: true, target: document.createElement("div") } as unknown as IntersectionObserverEntry],
      {} as IntersectionObserver,
    );

    expect(mockPlay).toHaveBeenCalledTimes(1);
  });

  it("does not call audio.play() when intersection ratio is below 60%", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    await new Promise((r) => setTimeout(r, 0));

    const autoplayCallback = observerCallbacks[0];
    expect(autoplayCallback).toBeDefined();

    autoplayCallback(
      [{ intersectionRatio: 0.3, isIntersecting: true, target: document.createElement("div") } as unknown as IntersectionObserverEntry],
      {} as IntersectionObserver,
    );

    expect(mockPlay).not.toHaveBeenCalled();
  });
});
