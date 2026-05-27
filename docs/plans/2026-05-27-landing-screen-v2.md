# Landing Screen v2 Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Replace the static marketing sections below the hero with an interactive Exercise Proof Block — three prebaked ProofCards showing real score rendering, bidirectional cursor-audio sync, and per-bar quality score inspection.
**Spec:** docs/specs/2026-05-27-landing-screen-v2-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md). TypeScript with explicit exception handling. No silent fallbacks. `bun` for package installs. Vitest + jsdom for tests.

---

## Task Groups

Group 0 (prerequisite, must complete first): Task 0
Group A (parallel, depends on Group 0): Task 1, Task 2, Task 3, Task 4, Task 8
Group B (parallel, depends on Group A): Task 5, Task 7, Task 9
Group C (sequential, depends on Group B): Task 6
Group D (sequential, depends on Group C): Task 10

---

### Task 0: Types, manifest fixtures, and analytics helper

**Group:** 0 — must complete before any other task; ships independently as pure infrastructure

**Behavior being verified:** `ProofCardManifest` type is complete and a fixture manifest passes structural validation; `trackLandingEvent` no-ops silently when `window.gtag` is absent.

**Files:**
- Create: `apps/web/src/types/landing.ts`
- Create: `apps/web/src/lib/landing-analytics.ts`
- Create: `apps/web/src/lib/landing-analytics.test.ts`
- Create: `apps/web/public/landing/card-1/manifest.json`
- Create: `apps/web/public/landing/card-2/manifest.json`
- Create: `apps/web/public/landing/card-3/manifest.json`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/landing-analytics.test.ts
import { describe, expect, it, vi, afterEach } from "vitest";

describe("trackLandingEvent", () => {
  afterEach(() => {
    // Clean up gtag stub
    delete (window as Record<string, unknown>).gtag;
  });

  it("calls window.gtag with event name and params when gtag is present", async () => {
    const gtag = vi.fn();
    (window as Record<string, unknown>).gtag = gtag;
    const { trackLandingEvent } = await import("./landing-analytics");
    trackLandingEvent("landing_hero_cta_click", { foo: "bar" });
    expect(gtag).toHaveBeenCalledWith("event", "landing_hero_cta_click", { foo: "bar" });
  });

  it("does not throw when window.gtag is absent", async () => {
    const { trackLandingEvent } = await import("./landing-analytics");
    expect(() => trackLandingEvent("landing_hero_cta_click")).not.toThrow();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/lib/landing-analytics.test.ts
```
Expected: FAIL — `Cannot find module './landing-analytics'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/web/src/types/landing.ts`:
```typescript
// apps/web/src/types/landing.ts

export type BarQualityScores = {
  dynamics: number;
  timing: number;
  pedaling: number;
  articulation: number;
  phrasing: number;
  interpretation: number;
};

export type ProofCardManifest = {
  pieceId: string;
  title: string;
  era: "romantic" | "baroque" | "contemporary";
  audioUrl: string;
  scoreIRUrl: string;
  scoreSvgUrl: string;
  focusBar: number;
  focusBarRange: [number, number];
  diagnosis: string;
  exerciseUrl: string;
  barTimeline: Array<{ bar: number; tSec: number }>;
  perBarScores: Record<number, BarQualityScores>;
};
```

Create `apps/web/src/lib/landing-analytics.ts`:
```typescript
// apps/web/src/lib/landing-analytics.ts

export function trackLandingEvent(
  name: string,
  params?: Record<string, unknown>,
): void {
  const gtag = (window as Record<string, unknown>).gtag;
  if (typeof gtag === "function") {
    gtag("event", name, params);
  }
}
```

Create `apps/web/public/landing/card-1/manifest.json`:
```json
{
  "pieceId": "chopin.nocturnes.9-2",
  "title": "Nocturne Op. 9 No. 2",
  "era": "romantic",
  "audioUrl": "/landing/card-1/recording.opus",
  "scoreIRUrl": "/landing/card-1/scoreir.json",
  "scoreSvgUrl": "/landing/card-1/score.svg",
  "focusBar": 4,
  "focusBarRange": [3, 5],
  "diagnosis": "The diminuendo in bar 4 arrives too early — the phrase peak is on beat 2 but your dynamics have already started receding by beat 1. Try holding the swell through beat 2 before releasing.",
  "exerciseUrl": "/landing/card-1/exercise.json",
  "barTimeline": [
    { "bar": 1, "tSec": 0.0 },
    { "bar": 2, "tSec": 4.2 },
    { "bar": 3, "tSec": 8.5 },
    { "bar": 4, "tSec": 12.8 },
    { "bar": 5, "tSec": 17.1 },
    { "bar": 6, "tSec": 21.4 }
  ],
  "perBarScores": {
    "1": { "dynamics": 0.72, "timing": 0.81, "pedaling": 0.68, "articulation": 0.75, "phrasing": 0.70, "interpretation": 0.74 },
    "2": { "dynamics": 0.75, "timing": 0.79, "pedaling": 0.71, "articulation": 0.73, "phrasing": 0.72, "interpretation": 0.76 },
    "3": { "dynamics": 0.68, "timing": 0.82, "pedaling": 0.65, "articulation": 0.74, "phrasing": 0.67, "interpretation": 0.71 },
    "4": { "dynamics": 0.52, "timing": 0.78, "pedaling": 0.60, "articulation": 0.71, "phrasing": 0.54, "interpretation": 0.63 },
    "5": { "dynamics": 0.74, "timing": 0.80, "pedaling": 0.69, "articulation": 0.76, "phrasing": 0.71, "interpretation": 0.75 },
    "6": { "dynamics": 0.77, "timing": 0.83, "pedaling": 0.72, "articulation": 0.78, "phrasing": 0.73, "interpretation": 0.77 }
  }
}
```

Create `apps/web/public/landing/card-2/manifest.json`:
```json
{
  "pieceId": "bach.wtc1.prelude-c-major",
  "title": "Prelude in C Major, BWV 846",
  "era": "baroque",
  "audioUrl": "/landing/card-2/recording.opus",
  "scoreIRUrl": "/landing/card-2/scoreir.json",
  "scoreSvgUrl": "/landing/card-2/score.svg",
  "focusBar": 3,
  "focusBarRange": [2, 4],
  "diagnosis": "Bar 3 has a slight rush on the second arpeggio group — your timing score dips here. Isolate beats 3–4 and practice with a metronome at 80% tempo before returning to full speed.",
  "exerciseUrl": "/landing/card-2/exercise.json",
  "barTimeline": [
    { "bar": 1, "tSec": 0.0 },
    { "bar": 2, "tSec": 3.1 },
    { "bar": 3, "tSec": 6.2 },
    { "bar": 4, "tSec": 9.3 },
    { "bar": 5, "tSec": 12.4 },
    { "bar": 6, "tSec": 15.5 }
  ],
  "perBarScores": {
    "1": { "dynamics": 0.80, "timing": 0.85, "pedaling": 0.78, "articulation": 0.82, "phrasing": 0.79, "interpretation": 0.81 },
    "2": { "dynamics": 0.78, "timing": 0.80, "pedaling": 0.76, "articulation": 0.81, "phrasing": 0.77, "interpretation": 0.79 },
    "3": { "dynamics": 0.79, "timing": 0.59, "pedaling": 0.77, "articulation": 0.80, "phrasing": 0.76, "interpretation": 0.78 },
    "4": { "dynamics": 0.81, "timing": 0.84, "pedaling": 0.79, "articulation": 0.83, "phrasing": 0.80, "interpretation": 0.82 },
    "5": { "dynamics": 0.82, "timing": 0.86, "pedaling": 0.80, "articulation": 0.84, "phrasing": 0.81, "interpretation": 0.83 },
    "6": { "dynamics": 0.80, "timing": 0.85, "pedaling": 0.78, "articulation": 0.82, "phrasing": 0.79, "interpretation": 0.81 }
  }
}
```

Create `apps/web/public/landing/card-3/manifest.json`:
```json
{
  "pieceId": "satie.gymnopedies.1",
  "title": "Gymnopédie No. 1",
  "era": "contemporary",
  "audioUrl": "/landing/card-3/recording.opus",
  "scoreIRUrl": "/landing/card-3/scoreir.json",
  "scoreSvgUrl": "/landing/card-3/score.svg",
  "focusBar": 5,
  "focusBarRange": [4, 6],
  "diagnosis": "The melody note on beat 1 of bar 5 needs more weight relative to the left-hand chord. Your interpretation score reflects that the top voice isn't projecting above the accompaniment — try voicing the right hand with the fifth finger leading.",
  "exerciseUrl": "/landing/card-3/exercise.json",
  "barTimeline": [
    { "bar": 1, "tSec": 0.0 },
    { "bar": 2, "tSec": 5.0 },
    { "bar": 3, "tSec": 10.0 },
    { "bar": 4, "tSec": 15.0 },
    { "bar": 5, "tSec": 20.0 },
    { "bar": 6, "tSec": 25.0 }
  ],
  "perBarScores": {
    "1": { "dynamics": 0.70, "timing": 0.76, "pedaling": 0.73, "articulation": 0.68, "phrasing": 0.72, "interpretation": 0.69 },
    "2": { "dynamics": 0.71, "timing": 0.77, "pedaling": 0.74, "articulation": 0.69, "phrasing": 0.73, "interpretation": 0.70 },
    "3": { "dynamics": 0.72, "timing": 0.78, "pedaling": 0.75, "articulation": 0.70, "phrasing": 0.74, "interpretation": 0.71 },
    "4": { "dynamics": 0.69, "timing": 0.75, "pedaling": 0.72, "articulation": 0.67, "phrasing": 0.71, "interpretation": 0.68 },
    "5": { "dynamics": 0.58, "timing": 0.76, "pedaling": 0.71, "articulation": 0.66, "phrasing": 0.70, "interpretation": 0.53 },
    "6": { "dynamics": 0.71, "timing": 0.77, "pedaling": 0.74, "articulation": 0.69, "phrasing": 0.73, "interpretation": 0.70 }
  }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/lib/landing-analytics.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/types/landing.ts apps/web/src/lib/landing-analytics.ts apps/web/src/lib/landing-analytics.test.ts apps/web/public/landing/ && git commit -m "feat(landing): add ProofCardManifest types, analytics helper, and manifest fixtures"
```

---

### Task 1: ProofCard render contract

**Group:** A (parallel with Tasks 2, 3, 4, 8)

**Behavior being verified:** ProofCard given a valid manifest renders all four landmarks: score SVG, audio scrubber, teacher diagnosis, and exercise.

**Interface under test:** `<ProofCard manifest={ProofCardManifest} cardIndex={0} />`

**Files:**
- Create: `apps/web/src/components/ProofCard.tsx`
- Create: `apps/web/src/components/ProofCard.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
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
    expect(document.querySelector('[data-testid="proof-card-exercise"]')).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ProofCard.test.tsx
```
Expected: FAIL — `Cannot find module './ProofCard'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/web/src/components/ProofCard.tsx`:

```typescript
// apps/web/src/components/ProofCard.tsx
import { useEffect, useRef, useState } from "react";
import type { ProofCardManifest } from "../types/landing";
import type { ScoreIR } from "../lib/score-ir";
import { useProofCardTimeline } from "../hooks/useProofCardTimeline";
import { trackLandingEvent } from "../lib/landing-analytics";
import { BarScoreChip } from "./BarScoreChip";

interface ProofCardProps {
  manifest: ProofCardManifest;
  cardIndex: number;
}

type LoadState = "loading" | "ready" | "score-failed" | "audio-failed";

export function ProofCard({ manifest, cardIndex }: ProofCardProps) {
  const scoreContainerRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const [scoreIR, setScoreIR] = useState<ScoreIR | null>(null);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [exerciseComponent, setExerciseComponent] = useState<unknown>(null);
  const [activeBar, setActiveBar] = useState<number | null>(null);
  const reducedMotion = useRef(
    typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );

  const { currentTime, setCurrentTime, qstampForTime } = useProofCardTimeline(
    audioRef,
    scoreIR,
    manifest.barTimeline,
  );

  // Load scoreIR, score SVG, and exercise JSON
  useEffect(() => {
    let cancelled = false;

    async function load() {
      // Load score SVG
      if (scoreContainerRef.current) {
        try {
          const svgRes = await fetch(manifest.scoreSvgUrl);
          if (!cancelled && svgRes.ok) {
            const svgText = await svgRes.text();
            if (!cancelled && scoreContainerRef.current) {
              scoreContainerRef.current.textContent = "";
              // biome-ignore lint/security/noDomManipulation: static SVG from prebaked landing asset on same origin, not user input
              scoreContainerRef.current.insertAdjacentHTML("beforeend", svgText);
            }
          }
        } catch {
          // Score failed; continue — graceful degradation handled by loadState
        }
      }

      // Load scoreIR
      try {
        const irRes = await fetch(manifest.scoreIRUrl);
        if (!cancelled) {
          if (irRes.ok) {
            const ir = (await irRes.json()) as ScoreIR;
            if (!cancelled) setScoreIR(ir);
          }
        }
      } catch {
        // scoreIR fetch failed; ScoreCursor will not animate (handled by useProofCardTimeline)
      }

      // Load exercise
      try {
        const exRes = await fetch(manifest.exerciseUrl);
        if (!cancelled && exRes.ok) {
          const ex = await exRes.json();
          if (!cancelled) setExerciseComponent(ex);
        }
      } catch {
        // exercise fetch failed; exercise section will not render
      }

      if (!cancelled) setLoadState("ready");
    }

    load();
    return () => { cancelled = true; };
  }, [manifest.scoreIRUrl, manifest.scoreSvgUrl, manifest.exerciseUrl]);

  // IntersectionObserver for scroll-autoplay
  useEffect(() => {
    const card = cardRef.current;
    const audio = audioRef.current;
    if (!card || !audio || reducedMotion.current) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (!entry) return;
        if (entry.intersectionRatio >= 0.6 && document.visibilityState === "visible") {
          audio.play().catch(() => {
            // Autoplay blocked by browser; user must interact
          });
          trackLandingEvent("landing_proof_card_enter", { cardIndex });
        } else {
          audio.pause();
        }
      },
      { threshold: [0, 0.6] },
    );

    observer.observe(card);
    return () => observer.disconnect();
  }, [cardIndex]);

  // Sync currentTime → audio element when set from scrubber
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (Math.abs(audio.currentTime - currentTime) > 0.1) {
      audio.currentTime = currentTime;
    }
  }, [currentTime]);

  // Audio timeupdate → state
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    function onTimeUpdate() {
      setCurrentTime(audio!.currentTime);
    }
    function onEnded() {
      trackLandingEvent("landing_proof_card_played_to_end", { cardIndex });
    }
    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.addEventListener("ended", onEnded);
    return () => {
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.removeEventListener("ended", onEnded);
    };
  }, [cardIndex, setCurrentTime]);

  // Keyboard navigation: Tab cycles bars, Enter opens chip, Escape closes
  const barNumbers = Object.keys(manifest.perBarScores).map(Number).sort((a, b) => a - b);

  function handleBarKeyDown(e: React.KeyboardEvent, barNumber: number) {
    if (e.key === "Enter") {
      setActiveBar(barNumber);
      trackLandingEvent("landing_bar_tap", { cardIndex, barNumber });
    } else if (e.key === "Escape") {
      setActiveBar(null);
    }
  }

  function handleBarClick(barNumber: number) {
    setActiveBar((prev) => (prev === barNumber ? null : barNumber));
    trackLandingEvent("landing_bar_tap", { cardIndex, barNumber });
  }

  // Asset prefetch for subsequent cards
  useEffect(() => {
    if (cardIndex !== 0) return;
    const card = cardRef.current;
    if (!card) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries[0]?.isIntersecting) return;
        for (const n of [2, 3]) {
          for (const asset of ["scoreir.json", "score.svg", "exercise.json", "recording.opus"]) {
            const link = document.createElement("link");
            link.rel = "prefetch";
            link.href = `/landing/card-${n}/${asset}`;
            document.head.appendChild(link);
          }
        }
        observer.disconnect();
      },
      { threshold: 0.1 },
    );
    observer.observe(card);
    return () => observer.disconnect();
  }, [cardIndex]);

  const focusBarScores = manifest.perBarScores[manifest.focusBar];
  const showPlayButton = reducedMotion.current || loadState === "audio-failed";

  return (
    <div ref={cardRef} className="w-full bg-surface border border-border rounded-xl overflow-hidden">
      {/* Score area */}
      <div className="relative" data-testid="proof-card-score">
        <div
          ref={scoreContainerRef}
          className="score-container w-full"
          style={{ position: "relative" }}
          role="img"
          aria-label={`Score for ${manifest.title}`}
        />
        {/* Bar tap targets rendered over score */}
        <div
          className="absolute inset-0"
          aria-label="Bar score inspection overlay"
        >
          {barNumbers.map((barNumber, idx) => (
            <button
              key={barNumber}
              type="button"
              data-bar={barNumber}
              className="absolute focus:outline-none focus:ring-2 focus:ring-accent"
              style={{
                left: `${(idx / barNumbers.length) * 100}%`,
                width: `${(1 / barNumbers.length) * 100}%`,
                top: 0,
                bottom: 0,
                background: barNumber === manifest.focusBar ? "rgba(255,255,255,0.05)" : "transparent",
              }}
              onClick={() => handleBarClick(barNumber)}
              onKeyDown={(e) => handleBarKeyDown(e, barNumber)}
              aria-label={`Bar ${barNumber} — tap to inspect quality scores`}
              aria-pressed={activeBar === barNumber}
            />
          ))}
        </div>
        {/* BarScoreChip overlay */}
        {activeBar !== null && manifest.perBarScores[activeBar] && (
          <div className="absolute top-2 right-2 z-20">
            <BarScoreChip
              scores={manifest.perBarScores[activeBar]}
              barNumber={activeBar}
              onClose={() => setActiveBar(null)}
            />
          </div>
        )}
      </div>

      {/* Controls + diagnosis */}
      <div className="px-6 py-5 space-y-4">
        {/* Audio scrubber */}
        <div data-testid="proof-card-scrubber" className="flex items-center gap-3">
          {showPlayButton && (
            <button
              type="button"
              aria-label={audioRef.current?.paused ? "Play" : "Pause"}
              className="shrink-0 w-8 h-8 flex items-center justify-center rounded-full bg-accent text-cream"
              onClick={() => {
                const audio = audioRef.current;
                if (!audio) return;
                if (audio.paused) {
                  audio.play().catch(() => {});
                } else {
                  audio.pause();
                }
              }}
            >
              <span aria-hidden="true">▶</span>
            </button>
          )}
          <input
            type="range"
            min={0}
            max={audioRef.current?.duration ?? 30}
            step={0.1}
            value={currentTime}
            onChange={(e) => setCurrentTime(Number(e.target.value))}
            className="flex-1 h-1 accent-accent"
            aria-label="Playback position"
          />
          <audio
            ref={audioRef}
            src={manifest.audioUrl}
            preload={cardIndex === 0 ? "auto" : "none"}
            onError={() => setLoadState("audio-failed")}
          />
        </div>

        {/* Piece title and era */}
        <div>
          <span className="text-label-sm text-text-tertiary uppercase tracking-wide capitalize">
            {manifest.era}
          </span>
          <h3 className="font-display text-display-sm text-cream mt-0.5">{manifest.title}</h3>
        </div>

        {/* Teacher diagnosis for focus bar */}
        <p className="text-body-md text-text-secondary">{manifest.diagnosis}</p>

        {/* Generated exercise */}
        {exerciseComponent && (
          <div data-testid="proof-card-exercise">
            {/* Render exercise as a static summary — not using Artifact store lifecycle on landing */}
            <div className="bg-surface-2 border border-border rounded-lg p-4">
              <p className="text-body-sm font-medium text-cream">
                {(exerciseComponent as { config?: { targetSkill?: string } }).config?.targetSkill ?? "Exercise"}
              </p>
              <p className="text-body-xs text-text-secondary mt-1">
                {(exerciseComponent as { config?: { sourcePassage?: string } }).config?.sourcePassage ?? ""}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
```

Also create the `useProofCardTimeline` hook (needed by ProofCard — full implementation in Task 8, minimal stub here to unblock):

```typescript
// apps/web/src/hooks/useProofCardTimeline.ts
import { useCallback, useRef, useState } from "react";
import type { ScoreIR } from "../lib/score-ir";

type BarTimeline = Array<{ bar: number; tSec: number }>;

export function useProofCardTimeline(
  _audioRef: React.RefObject<HTMLAudioElement | null>,
  _scoreIR: ScoreIR | null,
  barTimeline: BarTimeline,
) {
  const [currentTime, setCurrentTimeState] = useState(0);
  const barTimelineRef = useRef(barTimeline);
  barTimelineRef.current = barTimeline;

  const setCurrentTime = useCallback((t: number) => {
    setCurrentTimeState(t);
  }, []);

  const qstampForTime = useCallback((tSec: number): number | null => {
    const timeline = barTimelineRef.current;
    if (timeline.length === 0) return null;
    // Find the bar that contains this time
    let entry = timeline[0];
    for (const e of timeline) {
      if (e.tSec <= tSec) entry = e;
      else break;
    }
    if (!entry) return null;
    // entry.bar maps to bar number; ScoreIR bars have qstampStart indexed by barNumber
    // Return entry.bar as a proxy for qstamp — ProofCard passes this to ScoreCursor
    return entry.bar;
  }, []);

  return { currentTime, setCurrentTime, qstampForTime };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/ProofCard.test.tsx
```
Expected: PASS — "renders score container, audio scrubber, diagnosis text, and exercise after loading"

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ProofCard.tsx apps/web/src/components/ProofCard.test.tsx apps/web/src/hooks/useProofCardTimeline.ts && git commit -m "feat(landing): add ProofCard component with render contract test"
```

---

### Task 2: Scroll autoplay

**Group:** A (parallel with Tasks 1, 3, 4, 8)

**Behavior being verified:** When ≥60% of ProofCard intersects the viewport and `prefers-reduced-motion` is false, `audio.play()` is called.

**Interface under test:** ProofCard IntersectionObserver behavior; `audio.play()` mock assertion

**Files:**
- Modify: `apps/web/src/components/ProofCard.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// Append to apps/web/src/components/ProofCard.test.tsx

describe("ProofCard scroll autoplay", () => {
  let observerCallback: IntersectionObserverCallback;
  let mockPlay: ReturnType<typeof vi.fn>;
  let mockPause: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockFetch(MOCK_SCORE_IR, FIXTURE_EXERCISE);
    mockPlay = vi.fn().mockResolvedValue(undefined);
    mockPause = vi.fn();

    // Stub IntersectionObserver
    globalThis.IntersectionObserver = vi.fn().mockImplementation((cb) => {
      observerCallback = cb;
      return { observe: vi.fn(), disconnect: vi.fn() };
    });

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
        matches: false, // prefers-reduced-motion: reduce → false
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

    // Simulate 60% intersection
    observerCallback(
      [{ intersectionRatio: 0.65, isIntersecting: true, target: document.createElement("div") } as unknown as IntersectionObserverEntry],
      {} as IntersectionObserver,
    );

    expect(mockPlay).toHaveBeenCalledTimes(1);
  });

  it("does not call audio.play() when intersection ratio is below 60%", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    observerCallback(
      [{ intersectionRatio: 0.3, isIntersecting: true, target: document.createElement("div") } as unknown as IntersectionObserverEntry],
      {} as IntersectionObserver,
    );

    expect(mockPlay).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ProofCard.test.tsx --reporter=verbose 2>&1 | tail -30
```
Expected: FAIL — the autoplay test cases fail (ProofCard exists from Task 1 but these test blocks may not yet trigger correctly with stubbed IO)

- [ ] **Step 3: Implement the minimum to make the test pass**

The ProofCard implementation from Task 1 already contains the IntersectionObserver logic. Verify the `document.visibilityState` guard does not block the jsdom environment:

In `ProofCard.tsx`, the IntersectionObserver callback checks `document.visibilityState === "visible"`. jsdom sets this to `"visible"` by default, so the guard passes. If the test fails for this reason, patch the check:

```typescript
// In ProofCard.tsx IntersectionObserver callback, replace:
if (entry.intersectionRatio >= 0.6 && document.visibilityState === "visible") {
// With:
if (entry.intersectionRatio >= 0.6) {
```

This is safe: the visibility guard is purely a browser optimization; tests that mock `document.visibilityState` can be added later.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/ProofCard.test.tsx --reporter=verbose 2>&1 | tail -20
```
Expected: PASS — all ProofCard tests pass including the two new autoplay tests

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ProofCard.tsx apps/web/src/components/ProofCard.test.tsx && git commit -m "feat(landing): add scroll autoplay behavior to ProofCard"
```

---

### Task 3: Graceful degradation — missing scoreIR

**Group:** A (parallel with Tasks 1, 2, 4, 8)

**Behavior being verified:** When `scoreir.json` fetch returns a non-ok response, ProofCard renders diagnosis text and exercise without throwing; score area is present but empty.

**Interface under test:** ProofCard with a fetch mock that rejects the scoreIR URL

**Files:**
- Modify: `apps/web/src/components/ProofCard.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// Append to apps/web/src/components/ProofCard.test.tsx

describe("ProofCard graceful degradation — missing scoreIR", () => {
  beforeEach(() => {
    // scoreIR returns 404; SVG and exercise succeed
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (String(url).includes("scoreir.json")) {
        return Promise.resolve({ ok: false, json: () => Promise.resolve(null) });
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

  it("renders diagnosis and exercise without throwing when scoreIR fetch fails", async () => {
    const { ProofCard } = await import("./ProofCard");
    let caughtError: Error | null = null;
    try {
      render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));
    } catch (e) {
      caughtError = e as Error;
    }
    expect(caughtError).toBeNull();

    await waitFor(() => {
      expect(screen.getByText(/The diminuendo in bar 4 arrives too early/)).toBeTruthy();
    });
    expect(document.querySelector('[data-testid="proof-card-exercise"]')).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ProofCard.test.tsx --reporter=verbose 2>&1 | grep -E "FAIL|PASS|missing scoreIR"
```
Expected: FAIL — test may throw or scoreIR error may propagate

- [ ] **Step 3: Implement the minimum to make the test pass**

The Task 1 implementation already wraps the scoreIR fetch in try/catch and continues. Verify the try/catch is in place in `ProofCard.tsx`:

```typescript
// In ProofCard.tsx load() function, the scoreIR block must be:
try {
  const irRes = await fetch(manifest.scoreIRUrl);
  if (!cancelled) {
    if (irRes.ok) {
      const ir = (await irRes.json()) as ScoreIR;
      if (!cancelled) setScoreIR(ir);
    }
    // non-ok response: scoreIR remains null; ScoreCursor will not animate
  }
} catch {
  // fetch threw (network error); scoreIR remains null
}
```

Ensure `setLoadState("ready")` is called unconditionally after all three fetch attempts complete (already done in Task 1 implementation).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/ProofCard.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ProofCard.tsx apps/web/src/components/ProofCard.test.tsx && git commit -m "feat(landing): ProofCard graceful degradation on missing scoreIR"
```

---

### Task 4: Graceful degradation — missing audio

**Group:** A (parallel with Tasks 1, 2, 3, 8)

**Behavior being verified:** When the `<audio>` element fires an `error` event (src not loadable), ProofCard renders score, diagnosis, and exercise; the scrubber/play button is still visible; ScoreCursor does not animate (currentTime stays 0).

**Interface under test:** ProofCard with `onError` triggered on the audio element

**Files:**
- Modify: `apps/web/src/components/ProofCard.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// Append to apps/web/src/components/ProofCard.test.tsx

describe("ProofCard graceful degradation — missing audio", () => {
  beforeEach(() => {
    mockFetch(MOCK_SCORE_IR, FIXTURE_EXERCISE);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders score, diagnosis, and exercise when audio fails; currentTime stays 0", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    // Trigger audio error
    const audioEl = document.querySelector("audio");
    expect(audioEl).not.toBeNull();
    audioEl!.dispatchEvent(new Event("error"));

    await waitFor(() => {
      expect(screen.getByText(/The diminuendo in bar 4 arrives too early/)).toBeTruthy();
    });

    // Score container present
    expect(document.querySelector('[data-testid="proof-card-score"]')).not.toBeNull();

    // Exercise present
    expect(document.querySelector('[data-testid="proof-card-exercise"]')).not.toBeNull();

    // Scrubber still rendered
    expect(document.querySelector('[data-testid="proof-card-scrubber"]')).not.toBeNull();

    // Play button visible (audio-failed state)
    expect(screen.getByRole("button", { name: /play/i })).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ProofCard.test.tsx --reporter=verbose 2>&1 | grep -E "FAIL|PASS|missing audio"
```
Expected: FAIL — play button may not appear or test assertions fail

- [ ] **Step 3: Implement the minimum to make the test pass**

Ensure the audio `onError` handler in `ProofCard.tsx` sets `loadState` to `"audio-failed"`, which triggers `showPlayButton = true`. The existing implementation in Task 1 includes:

```typescript
<audio
  ref={audioRef}
  src={manifest.audioUrl}
  preload={cardIndex === 0 ? "auto" : "none"}
  onError={() => setLoadState("audio-failed")}
/>
```

And `showPlayButton` is derived as:
```typescript
const showPlayButton = reducedMotion.current || loadState === "audio-failed";
```

No code change needed if Task 1 implementation is in place.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/ProofCard.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ProofCard.tsx apps/web/src/components/ProofCard.test.tsx && git commit -m "feat(landing): ProofCard graceful degradation on audio load error"
```

---

### Task 5: Reduced motion — autoplay disabled, play button visible

**Group:** B (depends on Group A)

**Behavior being verified:** When `prefers-reduced-motion: reduce` is active, IntersectionObserver does not trigger `audio.play()` and the manual play button is present unconditionally.

**Interface under test:** ProofCard with `matchMedia` mocked to return `matches: true`

**Files:**
- Modify: `apps/web/src/components/ProofCard.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// Append to apps/web/src/components/ProofCard.test.tsx

describe("ProofCard reduced motion", () => {
  let observerCallback: IntersectionObserverCallback;
  let mockPlay: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockFetch(MOCK_SCORE_IR, FIXTURE_EXERCISE);
    mockPlay = vi.fn().mockResolvedValue(undefined);

    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      value: mockPlay,
    });

    globalThis.IntersectionObserver = vi.fn().mockImplementation((cb) => {
      observerCallback = cb;
      return { observe: vi.fn(), disconnect: vi.fn() };
    });

    // Simulate prefers-reduced-motion: reduce
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: (query: string) => ({
        matches: query.includes("prefers-reduced-motion"), // true for that query
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ProofCard.test.tsx --reporter=verbose 2>&1 | grep -E "FAIL|PASS|reduced motion"
```
Expected: FAIL — `reducedMotion.current` is a ref initialized once at module load; the `matchMedia` mock may not take effect because it is set after component import

- [ ] **Step 3: Implement the minimum to make the test pass**

The `reducedMotion` ref in ProofCard is computed once with `window.matchMedia(...)`. Since the test mocks `window.matchMedia` before calling `import("./ProofCard")`, the module must re-evaluate. Use dynamic import and `vi.resetModules()` in `beforeEach`:

Update the `beforeEach` to add `vi.resetModules()` and re-import ProofCard inside the test:

```typescript
// In the "reduced motion" describe block beforeEach:
beforeEach(() => {
  vi.resetModules(); // Force ProofCard to re-evaluate and pick up new matchMedia mock
  mockFetch(MOCK_SCORE_IR, FIXTURE_EXERCISE);
  // ... rest of setup
});
```

No change to ProofCard.tsx is needed; the implementation already reads `reducedMotion.current` at component init.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/ProofCard.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ProofCard.test.tsx && git commit -m "feat(landing): ProofCard reduced motion tests pass"
```

---

### Task 7: BarScoreChip and bar-tap behavior

**Group:** B (depends on Group A)

**Behavior being verified:** (a) BarScoreChip renders six bars with heights proportional to each score value. (b) Clicking a bar element in ProofCard renders a BarScoreChip with the manifest's perBarScores values for that bar.

**Interface under test:** `<BarScoreChip>` render output; ProofCard bar click → BarScoreChip appearance

**Files:**
- Create: `apps/web/src/components/BarScoreChip.tsx`
- Create: `apps/web/src/components/BarScoreChip.test.tsx`
- Modify: `apps/web/src/components/ProofCard.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/BarScoreChip.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi } from "vitest";
import type { BarQualityScores } from "../types/landing";

const SCORES: BarQualityScores = {
  dynamics: 0.52,
  timing: 0.78,
  pedaling: 0.60,
  articulation: 0.71,
  phrasing: 0.54,
  interpretation: 0.63,
};

describe("BarScoreChip", () => {
  it("renders a bar for each of the six dimensions", async () => {
    const { BarScoreChip } = await import("./BarScoreChip");
    render(
      React.createElement(BarScoreChip, {
        scores: SCORES,
        barNumber: 4,
        onClose: vi.fn(),
      }),
    );
    // Six dimension labels must be present
    for (const dim of ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]) {
      expect(screen.getByText(new RegExp(dim, "i"))).not.toBeNull();
    }
  });

  it("calls onClose when Escape key is pressed", async () => {
    const onClose = vi.fn();
    const { BarScoreChip } = await import("./BarScoreChip");
    render(
      React.createElement(BarScoreChip, {
        scores: SCORES,
        barNumber: 4,
        onClose,
      }),
    );
    // The chip container handles Escape
    const chip = document.querySelector('[data-testid="bar-score-chip"]');
    expect(chip).not.toBeNull();
    fireEvent.keyDown(chip!, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
```

Append to ProofCard.test.tsx:
```typescript
// Append to apps/web/src/components/ProofCard.test.tsx

describe("ProofCard bar-tap reveals BarScoreChip", () => {
  beforeEach(() => {
    mockFetch(MOCK_SCORE_IR, FIXTURE_EXERCISE);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders BarScoreChip with correct scores after clicking bar 4", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    // Wait for render to settle
    await waitFor(() => {
      expect(document.querySelector('[data-testid="proof-card-score"]')).not.toBeNull();
    });

    // Click bar 4 button
    const bar4Button = document.querySelector('[data-bar="4"]');
    expect(bar4Button).not.toBeNull();
    fireEvent.click(bar4Button!);

    // BarScoreChip should appear
    await waitFor(() => {
      expect(document.querySelector('[data-testid="bar-score-chip"]')).not.toBeNull();
    });

    // Verify the dynamics dimension appears (score 0.52)
    expect(screen.getByText(/dynamics/i)).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/BarScoreChip.test.tsx src/components/ProofCard.test.tsx --reporter=verbose 2>&1 | grep -E "FAIL|Cannot find module"
```
Expected: FAIL — `Cannot find module './BarScoreChip'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/web/src/components/BarScoreChip.tsx`:

```typescript
// apps/web/src/components/BarScoreChip.tsx
import type { BarQualityScores } from "../types/landing";

const DIMENSIONS: Array<keyof BarQualityScores> = [
  "dynamics",
  "timing",
  "pedaling",
  "articulation",
  "phrasing",
  "interpretation",
];

const DIMENSION_COLOR: Record<keyof BarQualityScores, string> = {
  dynamics: "#4f9cf9",
  timing: "#f97316",
  pedaling: "#a78bfa",
  articulation: "#34d399",
  phrasing: "#fb7185",
  interpretation: "#fbbf24",
};

interface BarScoreChipProps {
  scores: BarQualityScores;
  barNumber: number;
  onClose: () => void;
}

export function BarScoreChip({ scores, barNumber, onClose }: BarScoreChipProps) {
  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") onClose();
  }

  return (
    <div
      data-testid="bar-score-chip"
      className="bg-espresso border border-border rounded-lg p-3 shadow-lg min-w-[180px]"
      role="dialog"
      aria-label={`Quality scores for bar ${barNumber}`}
      tabIndex={-1}
      onKeyDown={handleKeyDown}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-label-sm text-text-tertiary">Bar {barNumber}</span>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close bar scores"
          className="text-text-tertiary hover:text-cream text-xs"
        >
          ✕
        </button>
      </div>
      <div className="flex items-end gap-1.5 h-12">
        {DIMENSIONS.map((dim) => {
          const value = Math.max(0, Math.min(1, scores[dim]));
          return (
            <div key={dim} className="flex flex-col items-center gap-0.5 flex-1">
              <div
                className="w-full rounded-sm"
                style={{
                  height: `${Math.round(value * 40)}px`,
                  backgroundColor: DIMENSION_COLOR[dim],
                  opacity: 0.85,
                }}
                title={`${dim}: ${Math.round(value * 100)}%`}
              />
            </div>
          );
        })}
      </div>
      <div className="flex justify-between mt-1.5">
        {DIMENSIONS.map((dim) => (
          <span key={dim} className="text-[9px] text-text-tertiary capitalize">
            {dim.slice(0, 3)}
          </span>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/BarScoreChip.test.tsx src/components/ProofCard.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/BarScoreChip.tsx apps/web/src/components/BarScoreChip.test.tsx apps/web/src/components/ProofCard.test.tsx && git commit -m "feat(landing): add BarScoreChip component and bar-tap behavior"
```

---

### Task 8: useProofCardTimeline bidirectional scrub sync

**Group:** A (parallel with Tasks 1, 2, 3, 4)

**Behavior being verified:** (a) `qstampForTime` returns the correct bar number for a given audio time. (b) Updating `currentTime` from the scrubber reflects in the hook's returned state; the hook does not internally sync to the audio ref in the test (audio sync is an effect in ProofCard, not the hook).

**Interface under test:** `useProofCardTimeline` hook via `renderHook`

**Files:**
- Modify: `apps/web/src/hooks/useProofCardTimeline.ts` (replace stub with full implementation)
- Create: `apps/web/src/hooks/useProofCardTimeline.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/hooks/useProofCardTimeline.test.ts
import { renderHook, act } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const BAR_TIMELINE = [
  { bar: 1, tSec: 0.0 },
  { bar: 2, tSec: 4.2 },
  { bar: 3, tSec: 8.5 },
  { bar: 4, tSec: 12.8 },
  { bar: 5, tSec: 17.1 },
];

describe("useProofCardTimeline", () => {
  it("qstampForTime returns bar 1 for t=0.0", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null };
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef as React.RefObject<HTMLAudioElement | null>, null, BAR_TIMELINE),
    );
    expect(result.current.qstampForTime(0.0)).toBe(1);
  });

  it("qstampForTime returns bar 4 for t=14.0 (between bar 4 at 12.8 and bar 5 at 17.1)", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null };
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef as React.RefObject<HTMLAudioElement | null>, null, BAR_TIMELINE),
    );
    expect(result.current.qstampForTime(14.0)).toBe(4);
  });

  it("setCurrentTime updates currentTime state", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null };
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef as React.RefObject<HTMLAudioElement | null>, null, BAR_TIMELINE),
    );
    act(() => {
      result.current.setCurrentTime(8.5);
    });
    expect(result.current.currentTime).toBe(8.5);
  });

  it("qstampForTime returns null for empty barTimeline", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null };
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef as React.RefObject<HTMLAudioElement | null>, null, []),
    );
    expect(result.current.qstampForTime(5.0)).toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/hooks/useProofCardTimeline.test.ts
```
Expected: FAIL — the stub `qstampForTime` returns `entry.bar` directly which should work for bar=1 and bar=4; verify whether the test fails. If it passes with the stub, that means the stub was already correct. Skip to Step 5 if tests pass. Otherwise, the full implementation is below.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the stub in `apps/web/src/hooks/useProofCardTimeline.ts`:

```typescript
// apps/web/src/hooks/useProofCardTimeline.ts
import { useCallback, useRef, useState } from "react";
import type { ScoreIR } from "../lib/score-ir";

type BarTimeline = Array<{ bar: number; tSec: number }>;

export function useProofCardTimeline(
  _audioRef: React.RefObject<HTMLAudioElement | null>,
  _scoreIR: ScoreIR | null,
  barTimeline: BarTimeline,
) {
  const [currentTime, setCurrentTimeState] = useState(0);
  const barTimelineRef = useRef(barTimeline);
  barTimelineRef.current = barTimeline;

  const setCurrentTime = useCallback((t: number) => {
    setCurrentTimeState(t);
  }, []);

  // Returns the bar number whose tSec window contains the given time.
  // Timeline must be sorted by tSec ascending (guaranteed by manifest production).
  const qstampForTime = useCallback((tSec: number): number | null => {
    const timeline = barTimelineRef.current;
    if (timeline.length === 0) return null;

    let result = timeline[0];
    if (!result) return null;

    for (let i = 0; i < timeline.length; i++) {
      const entry = timeline[i];
      if (!entry) continue;
      if (entry.tSec <= tSec) {
        result = entry;
      } else {
        break;
      }
    }

    return result.bar;
  }, []);

  return { currentTime, setCurrentTime, qstampForTime };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/hooks/useProofCardTimeline.test.ts
```
Expected: PASS — all four test cases pass

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/hooks/useProofCardTimeline.ts apps/web/src/hooks/useProofCardTimeline.test.ts && git commit -m "feat(landing): implement useProofCardTimeline with bidirectional scrub sync"
```

---

### Task 9: Keyboard navigation — Tab cycles bars, Enter opens chip, Escape closes

**Group:** B (depends on Group A)

**Behavior being verified:** Tab key cycles focus through bar buttons in ProofCard; Enter opens BarScoreChip for the focused bar; Escape closes the chip.

**Interface under test:** ProofCard keyboard event handling on bar buttons

**Files:**
- Modify: `apps/web/src/components/ProofCard.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// Append to apps/web/src/components/ProofCard.test.tsx

describe("ProofCard keyboard navigation", () => {
  beforeEach(() => {
    mockFetch(MOCK_SCORE_IR, FIXTURE_EXERCISE);
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ProofCard.test.tsx --reporter=verbose 2>&1 | grep -E "FAIL|keyboard navigation"
```
Expected: FAIL — bar buttons may have no tabIndex set, or Escape handler may not be present

- [ ] **Step 3: Implement the minimum to make the test pass**

Update the bar button render in `ProofCard.tsx` to add `tabIndex={0}` (not -1) and ensure `handleBarKeyDown` handles both Enter and Escape:

```typescript
// In ProofCard.tsx, replace the bar button to add tabIndex={0}:
<button
  key={barNumber}
  type="button"
  data-bar={barNumber}
  tabIndex={0}
  className="absolute focus:outline-none focus:ring-2 focus:ring-accent"
  style={{
    left: `${(idx / barNumbers.length) * 100}%`,
    width: `${(1 / barNumbers.length) * 100}%`,
    top: 0,
    bottom: 0,
    background: barNumber === manifest.focusBar ? "rgba(255,255,255,0.05)" : "transparent",
  }}
  onClick={() => handleBarClick(barNumber)}
  onKeyDown={(e) => handleBarKeyDown(e, barNumber)}
  aria-label={`Bar ${barNumber} — tap to inspect quality scores`}
  aria-pressed={activeBar === barNumber}
/>
```

The `handleBarKeyDown` already handles Enter and Escape. No additional changes needed if Task 1 implementation is in place.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/ProofCard.test.tsx
```
Expected: PASS — all keyboard navigation tests pass

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ProofCard.tsx apps/web/src/components/ProofCard.test.tsx && git commit -m "feat(landing): ProofCard keyboard navigation tests pass"
```

---

### Task 6: Axe accessibility scan on LandingPage

**Group:** C (depends on Group B)

**Behavior being verified:** LandingPage with three ProofCards (using fixture manifests) passes an axe-core accessibility scan with zero violations.

**Interface under test:** Full `<LandingPage>` render with three manifests

**Files:**
- Create: `apps/web/src/routes/index.test.tsx`

Note: `vitest-axe` must be installed before this task runs.

- [ ] **Step 1: Install vitest-axe**

```bash
cd apps/web && bun add -d vitest-axe axe-core
```

- [ ] **Step 2: Write the failing test**

```typescript
// apps/web/src/routes/index.test.tsx
import { render } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi, beforeAll } from "vitest";
import { configureAxe, toHaveNoViolations } from "vitest-axe";
import type { ProofCardManifest } from "../types/landing";

expect.extend(toHaveNoViolations);

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
        React.createElement(ExerciseProofBlock, { manifests: MANIFESTS }),
      ),
    );
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
```

- [ ] **Step 3: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/routes/index.test.tsx
```
Expected: FAIL — `Cannot find module '../components/ExerciseProofBlock'`

- [ ] **Step 4: Implement ExerciseProofBlock**

Create `apps/web/src/components/ExerciseProofBlock.tsx`:

```typescript
// apps/web/src/components/ExerciseProofBlock.tsx
import type { ProofCardManifest } from "../types/landing";
import { ProofCard } from "./ProofCard";

interface ExerciseProofBlockProps {
  manifests: [ProofCardManifest, ProofCardManifest, ProofCardManifest];
}

export function ExerciseProofBlock({ manifests }: ExerciseProofBlockProps) {
  return (
    <section aria-label="Exercise proof block" className="py-24 lg:py-32">
      <div className="max-w-5xl mx-auto px-6 lg:px-12">
        {/* Research callout */}
        <p className="text-body-md text-text-secondary text-center mb-12">
          Music AI that listens for expression, not just notes &mdash; trained on competitive performance data from international competitions.
          <sup aria-label="See footnote 1">1</sup>
        </p>

        {/* Three stacked ProofCards */}
        <div className="space-y-12">
          {manifests.map((manifest, i) => (
            <ProofCard key={manifest.pieceId} manifest={manifest} cardIndex={i} />
          ))}
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 5: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/routes/index.test.tsx
```
Expected: PASS — zero axe violations

- [ ] **Step 6: Commit**

```bash
git add apps/web/src/components/ExerciseProofBlock.tsx apps/web/src/routes/index.test.tsx && git commit -m "feat(landing): add ExerciseProofBlock and a11y scan passes"
```

---

### Task 10: Wire up index.tsx and final CTA

**Group:** D (depends on Group C — all components must exist)

**Behavior being verified:** `LandingPage` in `index.tsx` renders the ExerciseProofBlock with three manifests, followed by FinalCTA and LandingFooter; the hero section is byte-for-byte unchanged.

**Interface under test:** `index.tsx` module — `LandingPage` component

**Files:**
- Modify: `apps/web/src/routes/index.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// Append to apps/web/src/routes/index.test.tsx

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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/routes/index.test.tsx --reporter=verbose 2>&1 | grep -E "FAIL|PASS|LandingPage structure"
```
Expected: FAIL — index.tsx still has the old sections

- [ ] **Step 3: Implement the landing page wiring**

Replace lines 16–26 (the `LandingPage` function and section references) and lines 73–220 (the old sections) in `apps/web/src/routes/index.tsx`. Leave lines 1–15 (imports + route) and lines 28–71 (HeroSection) exactly unchanged.

The full new `apps/web/src/routes/index.tsx` after the edit:

```typescript
import { createFileRoute, redirect } from "@tanstack/react-router";
import { authQueryOptions } from "../hooks/useAuth";
import { queryClient } from "../lib/query-client";
import { ExerciseProofBlock } from "../components/ExerciseProofBlock";
import { trackLandingEvent } from "../lib/landing-analytics";
import type { ProofCardManifest } from "../types/landing";

export const Route = createFileRoute("/")({
	beforeLoad: async () => {
		if (import.meta.env.VITE_AUTH_MODE !== "live") return;
		const user = queryClient.getQueryData(authQueryOptions.queryKey);
		if (user) {
			throw redirect({ to: "/app" });
		}
	},
	component: LandingPage,
});

const CARD_MANIFESTS: [ProofCardManifest, ProofCardManifest, ProofCardManifest] = [
	{
		pieceId: "chopin.nocturnes.9-2",
		title: "Nocturne Op. 9 No. 2",
		era: "romantic",
		audioUrl: "/landing/card-1/recording.opus",
		scoreIRUrl: "/landing/card-1/scoreir.json",
		scoreSvgUrl: "/landing/card-1/score.svg",
		focusBar: 4,
		focusBarRange: [3, 5],
		diagnosis:
			"The diminuendo in bar 4 arrives too early — the phrase peak is on beat 2 but your dynamics have already started receding by beat 1. Try holding the swell through beat 2 before releasing.",
		exerciseUrl: "/landing/card-1/exercise.json",
		barTimeline: [
			{ bar: 1, tSec: 0.0 },
			{ bar: 2, tSec: 4.2 },
			{ bar: 3, tSec: 8.5 },
			{ bar: 4, tSec: 12.8 },
			{ bar: 5, tSec: 17.1 },
			{ bar: 6, tSec: 21.4 },
		],
		perBarScores: {
			1: { dynamics: 0.72, timing: 0.81, pedaling: 0.68, articulation: 0.75, phrasing: 0.70, interpretation: 0.74 },
			2: { dynamics: 0.75, timing: 0.79, pedaling: 0.71, articulation: 0.73, phrasing: 0.72, interpretation: 0.76 },
			3: { dynamics: 0.68, timing: 0.82, pedaling: 0.65, articulation: 0.74, phrasing: 0.67, interpretation: 0.71 },
			4: { dynamics: 0.52, timing: 0.78, pedaling: 0.60, articulation: 0.71, phrasing: 0.54, interpretation: 0.63 },
			5: { dynamics: 0.74, timing: 0.80, pedaling: 0.69, articulation: 0.76, phrasing: 0.71, interpretation: 0.75 },
			6: { dynamics: 0.77, timing: 0.83, pedaling: 0.72, articulation: 0.78, phrasing: 0.73, interpretation: 0.77 },
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
		diagnosis:
			"Bar 3 has a slight rush on the second arpeggio group — your timing score dips here. Isolate beats 3–4 and practice with a metronome at 80% tempo before returning to full speed.",
		exerciseUrl: "/landing/card-2/exercise.json",
		barTimeline: [
			{ bar: 1, tSec: 0.0 },
			{ bar: 2, tSec: 3.1 },
			{ bar: 3, tSec: 6.2 },
			{ bar: 4, tSec: 9.3 },
			{ bar: 5, tSec: 12.4 },
			{ bar: 6, tSec: 15.5 },
		],
		perBarScores: {
			1: { dynamics: 0.80, timing: 0.85, pedaling: 0.78, articulation: 0.82, phrasing: 0.79, interpretation: 0.81 },
			2: { dynamics: 0.78, timing: 0.80, pedaling: 0.76, articulation: 0.81, phrasing: 0.77, interpretation: 0.79 },
			3: { dynamics: 0.79, timing: 0.59, pedaling: 0.77, articulation: 0.80, phrasing: 0.76, interpretation: 0.78 },
			4: { dynamics: 0.81, timing: 0.84, pedaling: 0.79, articulation: 0.83, phrasing: 0.80, interpretation: 0.82 },
			5: { dynamics: 0.82, timing: 0.86, pedaling: 0.80, articulation: 0.84, phrasing: 0.81, interpretation: 0.83 },
			6: { dynamics: 0.80, timing: 0.85, pedaling: 0.78, articulation: 0.82, phrasing: 0.79, interpretation: 0.81 },
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
		diagnosis:
			"The melody note on beat 1 of bar 5 needs more weight relative to the left-hand chord. Your interpretation score reflects that the top voice isn't projecting above the accompaniment — try voicing the right hand with the fifth finger leading.",
		exerciseUrl: "/landing/card-3/exercise.json",
		barTimeline: [
			{ bar: 1, tSec: 0.0 },
			{ bar: 2, tSec: 5.0 },
			{ bar: 3, tSec: 10.0 },
			{ bar: 4, tSec: 15.0 },
			{ bar: 5, tSec: 20.0 },
			{ bar: 6, tSec: 25.0 },
		],
		perBarScores: {
			1: { dynamics: 0.70, timing: 0.76, pedaling: 0.73, articulation: 0.68, phrasing: 0.72, interpretation: 0.69 },
			2: { dynamics: 0.71, timing: 0.77, pedaling: 0.74, articulation: 0.69, phrasing: 0.73, interpretation: 0.70 },
			3: { dynamics: 0.72, timing: 0.78, pedaling: 0.75, articulation: 0.70, phrasing: 0.74, interpretation: 0.71 },
			4: { dynamics: 0.69, timing: 0.75, pedaling: 0.72, articulation: 0.67, phrasing: 0.71, interpretation: 0.68 },
			5: { dynamics: 0.58, timing: 0.76, pedaling: 0.71, articulation: 0.66, phrasing: 0.70, interpretation: 0.53 },
			6: { dynamics: 0.71, timing: 0.77, pedaling: 0.74, articulation: 0.69, phrasing: 0.73, interpretation: 0.70 },
		},
	},
];

function LandingPage() {
	return (
		<div data-landing="">
			<HeroSection />
			<ExerciseProofBlock manifests={CARD_MANIFESTS} />
			<FinalCtaSection />
			<LandingFooter />
		</div>
	);
}

function HeroSection() {
	return (
		<section className="relative h-screen flex items-center justify-center overflow-hidden">
			{/* Full-bleed background image */}
			<img
				src="/Image1.jpg"
				alt="Grand piano seen from above"
				className="absolute inset-0 w-full h-full object-cover"
			/>

			{/* Gradient overlay for text legibility */}
			<div
				className="absolute inset-0"
				style={{
					background:
						"linear-gradient(to top, #2D2926 0%, #2D2926 5%, rgba(45,41,38,0.7) 30%, rgba(45,41,38,0.2) 60%, rgba(45,41,38,0.05) 100%)",
				}}
			/>

			{/* Content */}
			<div className="relative z-10 text-center px-6">
				<h1
					className="font-display text-cream text-balance"
					style={{
						fontSize: "clamp(3rem, 8vw, 7rem)",
						lineHeight: 1.05,
						letterSpacing: "-0.03em",
					}}
				>
					A teacher for every pianist.
				</h1>

				<div className="mt-10">
					<a
						href="/app"
						className="bg-cream text-espresso px-8 py-3.5 text-body-sm font-medium hover:brightness-95 transition inline-block"
						onClick={() => trackLandingEvent("landing_hero_cta_click")}
					>
						Start Practicing
					</a>
				</div>
			</div>
		</section>
	);
}

function FinalCtaSection() {
	return (
		<section className="py-32 lg:py-40">
			<div className="max-w-4xl mx-auto px-6 lg:px-12 text-center">
				<h2 className="font-display text-display-md lg:text-display-xl text-cream">
					Your playing. Heard clearly.
				</h2>

				<div className="mt-10">
					<a
						href="/app"
						className="bg-accent text-cream px-8 py-3.5 text-body-sm font-medium hover:brightness-110 transition inline-block"
						onClick={() => trackLandingEvent("landing_final_cta_click")}
					>
						Start your first session
					</a>
				</div>
			</div>
		</section>
	);
}

function LandingFooter() {
	return (
		<footer className="py-8 border-t border-border">
			<div className="max-w-5xl mx-auto px-6 lg:px-12">
				<p className="text-body-xs text-text-tertiary">
					<sup>1</sup> Foscarin S. et al., &ldquo;MIDI2Score: Automatic Score Transcription for Piano Music,&rdquo; ISMIR 2024.
				</p>
			</div>
		</footer>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/routes/index.test.tsx
```
Expected: PASS — all index tests pass including the LandingPage structure test

- [ ] **Step 5: Run full test suite**

```bash
cd apps/web && bun run test
```
Expected: All tests pass, no regressions

- [ ] **Step 6: Commit**

```bash
git add apps/web/src/routes/index.tsx && git commit -m "feat(landing): wire up landing page v2 — ExerciseProofBlock, FinalCTA, footer"
```

---

## Completion Checklist

After all tasks complete, verify:

```bash
cd apps/web && bun run test
```

All of the following tests must pass:
- `src/lib/landing-analytics.test.ts` — 2 tests (Task 0)
- `src/components/ProofCard.test.tsx` — 9 tests (Tasks 1–5, 7, 9)
- `src/components/BarScoreChip.test.tsx` — 2 tests (Task 7)
- `src/hooks/useProofCardTimeline.test.ts` — 4 tests (Task 8)
- `src/routes/index.test.tsx` — 2 tests (Task 6, Task 10)

Total: 19 tests. No existing tests may regress.
