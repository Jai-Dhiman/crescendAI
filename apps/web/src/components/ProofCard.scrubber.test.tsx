// apps/web/src/components/ProofCard.scrubber.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
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
  barTimeline: [{ bar: 1, tSec: 0.0 }],
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

describe("ProofCard scrubber state", () => {
  beforeEach(() => {
    mockFetch();

    // Stub play/pause (jsdom doesn't implement HTMLMediaElement playback)
    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      value: vi.fn().mockResolvedValue(undefined),
    });
    Object.defineProperty(HTMLMediaElement.prototype, "pause", {
      configurable: true,
      value: vi.fn(),
    });

    // Ensure not reduced-motion so the play button renders
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: (query: string) => ({
        matches: true, // reduced motion = true so showPlayButton is true
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

  it("scrubber max updates from 30 to audio duration after loadedmetadata fires", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    // Wait for the component to mount and effects to flush
    await new Promise((r) => setTimeout(r, 0));

    const audio = document.querySelector("audio") as HTMLAudioElement;
    expect(audio).not.toBeNull();

    // Set duration on the element and fire loadedmetadata
    Object.defineProperty(audio, "duration", { configurable: true, value: 14 });
    audio.dispatchEvent(new Event("loadedmetadata"));

    const scrubber = screen.getByRole("slider", { name: /playback position/i });
    await waitFor(() => {
      expect(scrubber.getAttribute("max")).toBe("14");
    });
  });

  it("play button aria-label is Play initially, then Pause after play event, then Play after pause event", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    await new Promise((r) => setTimeout(r, 0));

    const audio = document.querySelector("audio") as HTMLAudioElement;
    expect(audio).not.toBeNull();

    // Initially not playing
    const button = screen.getByRole("button", { name: /play/i });
    expect(button.getAttribute("aria-label")).toBe("Play");

    // Fire play event
    audio.dispatchEvent(new Event("play"));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /pause/i }).getAttribute("aria-label")).toBe("Pause");
    });

    // Fire pause event
    audio.dispatchEvent(new Event("pause"));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /play/i }).getAttribute("aria-label")).toBe("Play");
    });
  });
});
