// apps/web/src/components/ProofCard.reduced-motion.test.tsx
import { render, screen } from "@testing-library/react";
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

describe("ProofCard reduced motion", () => {
  let observerCallback: IntersectionObserverCallback;
  let mockPlay: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockPlay = vi.fn().mockResolvedValue(undefined);

    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      value: mockPlay,
    });

    // Set up IntersectionObserver mock BEFORE resetModules so it is in globalThis
    // when the freshly-imported ProofCard module evaluates its useEffect bodies.
    const MockIntersectionObserver = vi.fn().mockImplementation(function (this: unknown, cb: IntersectionObserverCallback) {
      observerCallback = cb;
      return { observe: vi.fn(), disconnect: vi.fn() };
    });
    globalThis.IntersectionObserver = MockIntersectionObserver as unknown as typeof IntersectionObserver;

    // Simulate prefers-reduced-motion: reduce (must be set before resetModules
    // so the fresh ProofCard module reads matches=true at init time)
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: (query: string) => ({
        matches: query.includes("prefers-reduced-motion"),
        media: query,
        addListener: vi.fn(),
        removeListener: vi.fn(),
      }),
    });

    // resetModules forces ProofCard to re-evaluate and pick up new matchMedia mock.
    // Safe here because this is an isolated file — does not affect other test files.
    vi.resetModules();

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
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });

  it("does not autoplay on intersection when prefers-reduced-motion is reduce", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    // Simulate full intersection
    if (observerCallback) {
      observerCallback(
        [{ intersectionRatio: 0.9, isIntersecting: true, target: document.createElement("div") } as unknown as IntersectionObserverEntry],
        {} as IntersectionObserver,
      );
    }

    expect(mockPlay).not.toHaveBeenCalled();
  });

  it("shows manual play button when prefers-reduced-motion is reduce", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    expect(screen.getByRole("button", { name: /play/i })).not.toBeNull();
  });
});
