// apps/web/src/routes/index.test.tsx
import { render, screen } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi, beforeAll } from "vitest";
import { configureAxe } from "vitest-axe";
import { toHaveNoViolations } from "vitest-axe/matchers";
import type { ProofCardManifest } from "../types/landing";

expect.extend({ toHaveNoViolations });

const axe = configureAxe({
  rules: {
    // Disable color-contrast for this test (design tokens require full browser rendering)
    "color-contrast": { enabled: false },
  },
});

vi.mock("../lib/landing-analytics", () => ({
  trackLandingEvent: vi.fn(),
}));

const MANIFESTS: ProofCardManifest[] = [
  {
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
    barTimeline: [{ bar: 1, tSec: 0 }, { bar: 4, tSec: 12.8 }],
    perBarScores: {
      1: { dynamics: 0.72, timing: 0.81, pedaling: 0.68, articulation: 0.75, phrasing: 0.70, interpretation: 0.74 },
      4: { dynamics: 0.52, timing: 0.78, pedaling: 0.60, articulation: 0.71, phrasing: 0.54, interpretation: 0.63 },
    },
  },
  {
    pieceId: "bach.wtc1.prelude-c-major",
    title: "Prelude in C Major, BWV 846",
    era: "baroque",
    audioUrl: "/landing/card-2/recording.opus",
    scoreIRUrl: "/landing/card-2/scoreir.json",
    scoreSvgUrl: "/landing/card-2/score.svg",
    focusBar: 3,
    focusBarRange: [2, 4],
    diagnosis: "Bar 3 has a slight rush on the second arpeggio group.",
    exerciseUrl: "/landing/card-2/exercise.json",
    barTimeline: [{ bar: 1, tSec: 0 }, { bar: 3, tSec: 6.2 }],
    perBarScores: {
      1: { dynamics: 0.80, timing: 0.85, pedaling: 0.78, articulation: 0.82, phrasing: 0.79, interpretation: 0.81 },
      3: { dynamics: 0.79, timing: 0.59, pedaling: 0.77, articulation: 0.80, phrasing: 0.76, interpretation: 0.78 },
    },
  },
  {
    pieceId: "satie.gymnopedies.1",
    title: "Gymnopédie No. 1",
    era: "contemporary",
    audioUrl: "/landing/card-3/recording.opus",
    scoreIRUrl: "/landing/card-3/scoreir.json",
    scoreSvgUrl: "/landing/card-3/score.svg",
    focusBar: 5,
    focusBarRange: [4, 6],
    diagnosis: "The melody note on beat 1 of bar 5 needs more weight.",
    exerciseUrl: "/landing/card-3/exercise.json",
    barTimeline: [{ bar: 1, tSec: 0 }, { bar: 5, tSec: 20.0 }],
    perBarScores: {
      1: { dynamics: 0.70, timing: 0.76, pedaling: 0.73, articulation: 0.68, phrasing: 0.72, interpretation: 0.69 },
      5: { dynamics: 0.58, timing: 0.76, pedaling: 0.71, articulation: 0.66, phrasing: 0.70, interpretation: 0.53 },
    },
  },
];

beforeAll(() => {
  globalThis.fetch = vi.fn().mockImplementation((url: string) => {
    if (String(url).includes("scoreir.json")) {
      return Promise.resolve({ ok: false, json: () => Promise.resolve(null) });
    }
    if (String(url).includes("exercise.json")) {
      return Promise.resolve({ ok: false, json: () => Promise.resolve(null) });
    }
    if (String(url).includes("score.svg")) {
      return Promise.resolve({ ok: true, text: () => Promise.resolve('<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="score"></svg>') });
    }
    return Promise.resolve({ ok: false, json: () => Promise.resolve(null) });
  });
});

describe("LandingPage a11y", () => {
  it("passes axe scan with three proof cards", async () => {
    const { ExerciseProofBlock } = await import("../components/ExerciseProofBlock");
    const { container } = render(
      React.createElement("main", { "aria-label": "CrescendAI landing page" },
        React.createElement(ExerciseProofBlock, { manifests: MANIFESTS as [ProofCardManifest, ProofCardManifest, ProofCardManifest] }),
      ),
    );
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});

describe("LandingPage structure", () => {
  it("renders ExerciseProofBlock, FinalCTA, and footer below the hero", async () => {
    // Mock fetch (same as beforeAll above covers this)
    const { default: IndexRoute } = await import("./index");

    // TanStack Router File Route — access the component directly
    // The component is the default export's options.component
    const LandingPage = (IndexRoute as unknown as { options: { component: React.ComponentType } }).options.component;
    if (!LandingPage) throw new Error("LandingPage component not found on route");

    render(React.createElement(LandingPage));

    // Hero stays: the headline
    expect(screen.getByText(/A teacher for every pianist/)).not.toBeNull();

    // Research callout from ExerciseProofBlock
    expect(screen.getByText(/expression, not just notes/)).not.toBeNull();

    // FinalCTA headline
    expect(screen.getByText(/Your playing. Heard clearly./)).not.toBeNull();

    // FinalCTA button
    expect(screen.getByRole("link", { name: /Start your first session/i })).not.toBeNull();

    // Footer footnote
    expect(screen.getByText(/ISMIR/)).not.toBeNull();
  });
});
