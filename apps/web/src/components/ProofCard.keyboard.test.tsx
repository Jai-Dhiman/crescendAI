// apps/web/src/components/ProofCard.keyboard.test.tsx
import { render, waitFor, fireEvent } from "@testing-library/react";
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
    { barNumber: 4, measureOn: "m4", pageN: 1, bbox: { x: 400, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 12, qstampEnd: 16 },
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

describe("ProofCard keyboard navigation", () => {
  beforeEach(() => {
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
  });

  it("Enter on a bar button opens the BarScoreChip for that bar", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    await waitFor(() => {
      expect(document.querySelector('[data-bar="4"]')).not.toBeNull();
    });

    const bar4 = document.querySelector('[data-bar="4"]') as HTMLElement;
    bar4.focus();
    fireEvent.keyDown(bar4, { key: "Enter" });

    await waitFor(() => {
      expect(document.querySelector('[data-testid="bar-score-chip"]')).not.toBeNull();
    });
  });

  it("Escape on a bar button closes the BarScoreChip", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    await waitFor(() => {
      expect(document.querySelector('[data-bar="4"]')).not.toBeNull();
    });

    const bar4 = document.querySelector('[data-bar="4"]') as HTMLElement;

    // Open chip via Enter
    fireEvent.keyDown(bar4, { key: "Enter" });
    await waitFor(() => {
      expect(document.querySelector('[data-testid="bar-score-chip"]')).not.toBeNull();
    });

    // Close chip via Escape
    fireEvent.keyDown(bar4, { key: "Escape" });
    await waitFor(() => {
      expect(document.querySelector('[data-testid="bar-score-chip"]')).toBeNull();
    });
  });

  it("bar buttons are focusable via Tab (all have non-negative tabIndex)", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    await waitFor(() => {
      const barButtons = document.querySelectorAll("[data-bar]");
      expect(barButtons.length).toBeGreaterThan(0);
      for (const btn of barButtons) {
        expect(btn.getAttribute("tabindex")).not.toBe("-1");
      }
    });
  });
});
