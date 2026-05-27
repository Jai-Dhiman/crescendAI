// apps/web/src/components/ProofCard.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import type { ProofCardManifest } from "../types/landing";

// Mock fetch for scoreir.json
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
  barTimeline: [
    { bar: 1, tSec: 0.0 },
    { bar: 4, tSec: 12.8 },
  ],
  perBarScores: {
    1: { dynamics: 0.72, timing: 0.81, pedaling: 0.68, articulation: 0.75, phrasing: 0.70, interpretation: 0.74 },
    4: { dynamics: 0.52, timing: 0.78, pedaling: 0.60, articulation: 0.71, phrasing: 0.54, interpretation: 0.63 },
  },
};

const FIXTURE_EXERCISE = {
  type: "exercise_set",
  config: {
    sourcePassage: "bar 4",
    targetSkill: "Dynamics — diminuendo control",
    exercises: [{ title: "Delayed release", instruction: "Hold forte through beat 2", focusDimension: "dynamics" }],
  },
};

vi.mock("../lib/landing-analytics", () => ({
  trackLandingEvent: vi.fn(),
}));

function mockFetch(scoreIR: unknown, exercise: unknown) {
  globalThis.fetch = vi.fn().mockImplementation((url: string) => {
    if (String(url).includes("scoreir.json")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(scoreIR) });
    }
    if (String(url).includes("exercise.json")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(exercise) });
    }
    if (String(url).includes("score.svg")) {
      return Promise.resolve({ ok: true, text: () => Promise.resolve('<svg xmlns="http://www.w3.org/2000/svg"><g class="measure" data-bar="4"></g></svg>') });
    }
    return Promise.resolve({ ok: false, json: () => Promise.resolve(null) });
  });
}

describe("ProofCard render contract", () => {
  beforeEach(() => {
    mockFetch(MOCK_SCORE_IR, FIXTURE_EXERCISE);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders score container, audio scrubber, diagnosis text, and exercise after loading", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    // Score container (data-testid set by ProofCard)
    expect(document.querySelector('[data-testid="proof-card-score"]')).not.toBeNull();

    // Audio scrubber
    expect(document.querySelector('[data-testid="proof-card-scrubber"]')).not.toBeNull();

    // Diagnosis text
    await waitFor(() => {
      expect(screen.getByText(/The diminuendo in bar 4 arrives too early/)).toBeTruthy();
    });

    // Exercise landmark
    await waitFor(() => {
      expect(document.querySelector('[data-testid="proof-card-exercise"]')).not.toBeNull();
    });
  });
});
