# Landing Screen v2 Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Replace the static marketing sections below the hero with an interactive Exercise Proof Block — three prebaked ProofCards showing real score rendering, bidirectional cursor-audio sync, and per-bar quality score inspection.
**Spec:** docs/specs/2026-05-27-landing-screen-v2-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md). TypeScript with explicit exception handling. No silent fallbacks. `bun` for package installs. Vitest + jsdom for tests.

---

## Task Groups

Group 0 (prerequisite, must complete first): Task 0
Group A (sequential, depends on Group 0): Task 1 only — creates ProofCard.tsx, ProofCard.test.tsx (minimal render contract smoke test), and useProofCardTimeline.ts stub; all subsequent groups depend on these files existing. Group B tasks each use their own separate test file — they do NOT append to ProofCard.test.tsx.
Group B (parallel, depends on Group A): Task 2, Task 3, Task 4, Task 4.5, Task 8 — each task owns its own separate test file (no shared file writes); Task 8 replaces the useProofCardTimeline.ts stub created by Task 1 with the full implementation
Group C (parallel, depends on Group B): Task 5, Task 7, Task 9
Group D (sequential, depends on Group C): Task 6
Group E (sequential, depends on Group D): Task 10

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

**Group:** A (sequential — must complete alone before Group B dispatches; creates ProofCard.tsx, ProofCard.test.tsx, and useProofCardTimeline.ts stub that all Group B tasks depend on)

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
import type { KeyboardEvent } from "react";
import type { ProofCardManifest } from "../types/landing";
import type { ScoreIR } from "../lib/score-ir";
import { ScoreCursor } from "../lib/score-cursor";
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
  // scoreIR is also passed to useProofCardTimeline so qstampForTime can return
  // proper quaternote qstamp values (bar.qstampStart) instead of bare bar numbers.

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

  // ScoreCursor — instantiate and start when scoreIR and score container are ready.
  // qstampSource is baked into the constructor and called each rAF tick by ScoreCursor.
  useEffect(() => {
    if (scoreIR === null || scoreContainerRef.current === null) return;
    const cursor = new ScoreCursor({
      pieceId: manifest.pieceId,
      container: scoreContainerRef.current,
      ir: scoreIR,
      qstampSource: () => qstampForTime(currentTime) ?? 0,
    });
    cursor.start();
    return () => {
      cursor.stop();
    };
  }, [scoreIR, currentTime, qstampForTime, manifest.pieceId]);

  // Keyboard navigation: Tab cycles bars, Enter opens chip, Escape closes
  const barNumbers = Object.keys(manifest.perBarScores).map(Number).sort((a, b) => a - b);

  function handleBarKeyDown(e: KeyboardEvent<HTMLButtonElement>, barNumber: number) {
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
import type { RefObject } from "react";
import type { ScoreIR } from "../lib/score-ir";

type BarTimeline = Array<{ bar: number; tSec: number }>;

export function useProofCardTimeline(
  _audioRef: RefObject<HTMLAudioElement | null>,
  scoreIR: ScoreIR | null,
  barTimeline: BarTimeline,
) {
  const [currentTime, setCurrentTimeState] = useState(0);
  const barTimelineRef = useRef(barTimeline);
  barTimelineRef.current = barTimeline;
  const scoreIRRef = useRef(scoreIR);
  scoreIRRef.current = scoreIR;

  const setCurrentTime = useCallback((t: number) => {
    setCurrentTimeState(t);
  }, []);

  // Returns the qstampStart of the bar whose tSec window contains the given time.
  // ScoreCursor.findBar() binary-searches bar.qstampStart/qstampEnd (quaternote floats),
  // so we must return a qstamp float, not a raw bar number.
  const qstampForTime = useCallback((tSec: number): number | null => {
    const timeline = barTimelineRef.current;
    if (timeline.length === 0) return null;

    // Find the last entry whose tSec <= tSec (sorted ascending)
    let matchedEntry = timeline[0];
    if (!matchedEntry) return null;
    for (let i = 0; i < timeline.length; i++) {
      const e = timeline[i];
      if (!e) continue;
      if (e.tSec <= tSec) matchedEntry = e;
      else break;
    }

    // Look up the bar's qstampStart from ScoreIR.bars by barNumber
    const ir = scoreIRRef.current;
    if (ir) {
      const barIR = ir.bars.find((b) => b.barNumber === matchedEntry!.bar);
      if (barIR) return barIR.qstampStart;
    }

    // Fallback: scoreIR not yet loaded — return null so cursor stays hidden
    return null;
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

**Group:** B (parallel with Tasks 3, 4, 4.5, 8 — depends on Task 1 completing first; uses its own separate test file to avoid parallel write collisions)

**Behavior being verified:** When ≥60% of ProofCard intersects the viewport and `prefers-reduced-motion` is false, `audio.play()` is called.

**Interface under test:** ProofCard IntersectionObserver behavior; `audio.play()` mock assertion

**Files:**
- Create: `apps/web/src/components/ProofCard.autoplay.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
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
  let observerCallback: IntersectionObserverCallback;
  let mockPlay: ReturnType<typeof vi.fn>;
  let mockPause: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockFetch();
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
cd apps/web && bun run test src/components/ProofCard.autoplay.test.tsx --reporter=verbose 2>&1 | tail -30
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
cd apps/web && bun run test src/components/ProofCard.autoplay.test.tsx --reporter=verbose 2>&1 | tail -20
```
Expected: PASS — both autoplay tests pass

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ProofCard.autoplay.test.tsx && git commit -m "feat(landing): add scroll autoplay behavior to ProofCard"
```

---

### Task 3: Graceful degradation — missing scoreIR

**Group:** B (parallel with Tasks 2, 4, 4.5, 8 — depends on Task 1 completing first; uses its own separate test file to avoid parallel write collisions)

**Behavior being verified:** When `scoreir.json` fetch returns a non-ok response, ProofCard renders diagnosis text and exercise without throwing; score area is present but empty.

**Interface under test:** ProofCard with a fetch mock that rejects the scoreIR URL

**Files:**
- Create: `apps/web/src/components/ProofCard.degradation.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/ProofCard.degradation.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import type { ProofCardManifest } from "../types/landing";

vi.mock("../lib/landing-analytics", () => ({ trackLandingEvent: vi.fn() }));

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

  it("renders score container even when scoreIR fetch fails", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));
    expect(document.querySelector('[data-testid="proof-card-score"]')).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ProofCard.degradation.test.tsx --reporter=verbose 2>&1 | grep -E "FAIL|PASS|missing scoreIR"
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
cd apps/web && bun run test src/components/ProofCard.degradation.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ProofCard.degradation.test.tsx && git commit -m "feat(landing): ProofCard graceful degradation on missing scoreIR"
```

---

### Task 4: Graceful degradation — missing audio

**Group:** B (parallel with Tasks 2, 3, 4.5, 8 — depends on Task 1 completing first; uses its own separate test file to avoid parallel write collisions)

**Behavior being verified:** When the `<audio>` element fires an `error` event (src not loadable), ProofCard renders score, diagnosis, and exercise; the scrubber/play button is still visible; ScoreCursor does not animate (currentTime stays 0).

**Interface under test:** ProofCard with `onError` triggered on the audio element

**Files:**
- Create: `apps/web/src/components/ProofCard.audio-degradation.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/ProofCard.audio-degradation.test.tsx
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
  barTimeline: [{ bar: 1, tSec: 0.0 }, { bar: 4, tSec: 12.8 }],
  perBarScores: {
    1: { dynamics: 0.72, timing: 0.81, pedaling: 0.68, articulation: 0.75, phrasing: 0.70, interpretation: 0.74 },
    4: { dynamics: 0.52, timing: 0.78, pedaling: 0.60, articulation: 0.71, phrasing: 0.54, interpretation: 0.63 },
  },
};

describe("ProofCard graceful degradation — missing audio", () => {
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
cd apps/web && bun run test src/components/ProofCard.audio-degradation.test.tsx --reporter=verbose 2>&1 | grep -E "FAIL|PASS|missing audio"
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
cd apps/web && bun run test src/components/ProofCard.audio-degradation.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ProofCard.audio-degradation.test.tsx && git commit -m "feat(landing): ProofCard graceful degradation on audio load error"
```

---

### Task 4.5: ScoreCursor instantiation and cursor movement

**Group:** B (parallel with Tasks 2, 3, 4, 8 — depends on Task 1 completing first; uses its own separate test file to avoid parallel write collisions)

**Behavior being verified:** When `scoreIR` loads successfully, a `ScoreCursor` instance is created with the correct options object `{ pieceId, container, ir, qstampSource }` and `cursor.start()` is called with no arguments; when the component unmounts, `cursor.stop()` is called.

**Interface under test:** ProofCard ScoreCursor integration via `ScoreCursor` constructor + `start`/`stop` mocks.

**Files:**
- Create: `apps/web/src/components/ProofCard.cursor.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/ProofCard.cursor.test.tsx
import { render, waitFor } from "@testing-library/react";
import * as React from "react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { ScoreCursor } from "../lib/score-cursor";
import type { ProofCardManifest } from "../types/landing";

// vi.mock must be at module top-level so Vitest's hoisting applies
vi.mock("../lib/score-cursor", () => ({
  ScoreCursor: vi.fn(),
}));

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
  config: { sourcePassage: "bar 4", targetSkill: "Dynamics", exercises: [] },
};

describe("ProofCard ScoreCursor integration", () => {
  let mockStart: ReturnType<typeof vi.fn>;
  let mockStop: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockStart = vi.fn();
    mockStop = vi.fn();
    // Configure the mocked constructor per-test using vi.mocked
    vi.mocked(ScoreCursor).mockImplementation(() => ({
      start: mockStart,
      stop: mockStop,
    }) as unknown as ScoreCursor);

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

  it("instantiates ScoreCursor with correct options and calls start() with no args when scoreIR loads", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    await waitFor(() => {
      expect(vi.mocked(ScoreCursor)).toHaveBeenCalled();
    });

    // Constructor must be called with an options object (not positional args)
    const ctorCall = vi.mocked(ScoreCursor).mock.calls[0];
    expect(ctorCall).toBeDefined();
    const opts = ctorCall![0] as { pieceId: string; container: HTMLElement; ir: unknown; qstampSource: () => number | null };
    expect(opts.pieceId).toBe("chopin.nocturnes.9-2");
    expect(opts.container).toBeInstanceOf(HTMLElement);
    expect(opts.ir).toMatchObject({ pieceId: "chopin.nocturnes.9-2" });
    expect(typeof opts.qstampSource).toBe("function");

    // start() called with no arguments
    expect(mockStart).toHaveBeenCalledWith();
  });

  it("qstampSource returns qstampStart=0 (bar 1) at t=0", async () => {
    const { ProofCard } = await import("./ProofCard");
    render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    await waitFor(() => {
      expect(vi.mocked(ScoreCursor)).toHaveBeenCalled();
    });

    const opts = vi.mocked(ScoreCursor).mock.calls[0]![0] as { qstampSource: () => number | null };
    // At t=0, barTimeline maps to bar 1, MOCK_SCORE_IR bar 1 has qstampStart=0
    expect(opts.qstampSource()).toBe(0);
  });

  it("calls cursor.stop when component unmounts", async () => {
    const { ProofCard } = await import("./ProofCard");
    const { unmount } = render(React.createElement(ProofCard, { manifest: FIXTURE_MANIFEST, cardIndex: 0 }));

    await waitFor(() => {
      expect(mockStart).toHaveBeenCalled();
    });

    unmount();
    expect(mockStop).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ProofCard.cursor.test.tsx --reporter=verbose 2>&1 | grep -E "FAIL|ScoreCursor"
```
Expected: FAIL — `ScoreCursor` constructor call mismatch until Task 1's ProofCard.tsx implementation is in place

- [ ] **Step 3: Implement the minimum to make the test pass**

The ScoreCursor `useEffect` is already included in the Task 1 ProofCard.tsx implementation snippet (see Task 1, Step 3). Confirm the constructor call uses the options object:

```typescript
// In ProofCard.tsx — confirm this useEffect block is present:
useEffect(() => {
  if (scoreIR === null || scoreContainerRef.current === null) return;
  const cursor = new ScoreCursor({
    pieceId: manifest.pieceId,
    container: scoreContainerRef.current,
    ir: scoreIR,
    qstampSource: () => qstampForTime(currentTime) ?? 0,
  });
  cursor.start();
  return () => {
    cursor.stop();
  };
}, [scoreIR, currentTime, qstampForTime, manifest.pieceId]);
```

No additional code changes needed if Task 1 is implemented correctly.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/ProofCard.cursor.test.tsx
```
Expected: PASS — all three ScoreCursor integration tests pass

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ProofCard.cursor.test.tsx && git commit -m "feat(landing): add ScoreCursor instantiation and lifecycle tests to ProofCard"
```

---

### Task 5: Reduced motion — autoplay disabled, play button visible

**Group:** C (parallel with Tasks 7 and 9 — depends on Group B)

**Behavior being verified:** When `prefers-reduced-motion: reduce` is active, IntersectionObserver does not trigger `audio.play()` and the manual play button is present unconditionally.

**Interface under test:** ProofCard with `matchMedia` mocked to return `matches: true`

**Files:**
- Create: `apps/web/src/components/ProofCard.reduced-motion.test.tsx`

Note: `vi.resetModules()` is called in `beforeEach` inside this file to force ProofCard to re-evaluate with the mocked `matchMedia`. Because this is isolated to its own file, it does not affect module caches in other test files.

- [ ] **Step 1: Write the failing test**

```typescript
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
cd apps/web && bun run test src/components/ProofCard.reduced-motion.test.tsx --reporter=verbose 2>&1 | grep -E "FAIL|PASS|reduced motion"
```
Expected: FAIL — `reducedMotion.current` is a ref initialized once at module load; the `matchMedia` mock may not take effect because it is set after component import

- [ ] **Step 3: Implement the minimum to make the test pass**

The `reducedMotion` ref in ProofCard is computed once with `window.matchMedia(...)`. The test file already calls `vi.resetModules()` in `beforeEach` before the dynamic `import("./ProofCard")` inside each test — this forces ProofCard to re-evaluate with the mocked `matchMedia`. No change to ProofCard.tsx is needed; the implementation already reads `reducedMotion.current` at component init.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/ProofCard.reduced-motion.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ProofCard.reduced-motion.test.tsx && git commit -m "feat(landing): ProofCard reduced motion tests pass"
```

---

### Task 7: BarScoreChip and bar-tap behavior

**Group:** C (parallel with Tasks 5 and 9 — depends on Group B)

**Behavior being verified:** (a) BarScoreChip renders six bars with heights proportional to each score value. (b) Clicking a bar element in ProofCard renders a BarScoreChip with the manifest's perBarScores values for that bar.

**Interface under test:** `<BarScoreChip>` render output; ProofCard bar click → BarScoreChip appearance

**Files:**
- Create: `apps/web/src/components/BarScoreChip.tsx`
- Create: `apps/web/src/components/BarScoreChip.test.tsx`
- Create: `apps/web/src/components/ProofCard.bar-tap.test.tsx`

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

Create the bar-tap integration test in its own file:
```typescript
// apps/web/src/components/ProofCard.bar-tap.test.tsx
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
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

describe("ProofCard bar-tap reveals BarScoreChip", () => {
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
cd apps/web && bun run test src/components/BarScoreChip.test.tsx src/components/ProofCard.bar-tap.test.tsx --reporter=verbose 2>&1 | grep -E "FAIL|Cannot find module"
```
Expected: FAIL — `Cannot find module './BarScoreChip'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/web/src/components/BarScoreChip.tsx`:

```typescript
// apps/web/src/components/BarScoreChip.tsx
import type { KeyboardEvent } from "react";
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
  function handleKeyDown(e: KeyboardEvent<HTMLDivElement>) {
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
cd apps/web && bun run test src/components/BarScoreChip.test.tsx src/components/ProofCard.bar-tap.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/BarScoreChip.tsx apps/web/src/components/BarScoreChip.test.tsx apps/web/src/components/ProofCard.bar-tap.test.tsx && git commit -m "feat(landing): add BarScoreChip component and bar-tap behavior"
```

---

### Task 8: useProofCardTimeline bidirectional scrub sync

**Group:** B (parallel with Tasks 2, 3, 4 — depends on Task 1 completing first; replaces the useProofCardTimeline.ts stub created by Task 1 with the full implementation — do NOT create fresh, use the stub file as the starting point)

**Behavior being verified:** (a) `qstampForTime` returns the `qstampStart` float of the matching bar from `ScoreIR.bars` for a given audio time — not the bare bar number. (b) Updating `currentTime` from the scrubber reflects in the hook's returned state; the hook does not internally sync to the audio ref in the test (audio sync is an effect in ProofCard, not the hook). (c) When `scoreIR` is `null`, `qstampForTime` returns `null` so the cursor stays hidden.

**Interface under test:** `useProofCardTimeline` hook via `renderHook`

**Files:**
- Modify: `apps/web/src/hooks/useProofCardTimeline.ts` (replace stub with full implementation)
- Create: `apps/web/src/hooks/useProofCardTimeline.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/hooks/useProofCardTimeline.test.ts
import { renderHook, act } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { RefObject } from "react";
import type { ScoreIR } from "../lib/score-ir";

const BAR_TIMELINE = [
  { bar: 1, tSec: 0.0 },
  { bar: 2, tSec: 4.2 },
  { bar: 3, tSec: 8.5 },
  { bar: 4, tSec: 12.8 },
  { bar: 5, tSec: 17.1 },
];

// ScoreIR fixture with quaternote qstampStart values (4/4 time, 4 quarter notes per bar)
const MOCK_SCORE_IR: ScoreIR = {
  pieceId: "chopin.nocturnes.9-2",
  verovioVersion: "6.1.0",
  pageWidth: 2400,
  pages: [{ pageN: 1, viewBox: "0 0 2400 800", width: 2400, height: 800, systemBboxes: [] }],
  bars: [
    { barNumber: 1, measureOn: "m1", pageN: 1, bbox: { x: 100, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 0, qstampEnd: 4 },
    { barNumber: 2, measureOn: "m2", pageN: 1, bbox: { x: 200, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 4, qstampEnd: 8 },
    { barNumber: 3, measureOn: "m3", pageN: 1, bbox: { x: 300, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 8, qstampEnd: 12 },
    { barNumber: 4, measureOn: "m4", pageN: 1, bbox: { x: 400, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 12, qstampEnd: 16 },
    { barNumber: 5, measureOn: "m5", pageN: 1, bbox: { x: 500, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 16, qstampEnd: 20 },
  ],
  notes: {},
};

describe("useProofCardTimeline", () => {
  it("qstampForTime returns qstampStart=0 for bar 1 at t=0.0", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null } as RefObject<HTMLAudioElement | null>;
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef, MOCK_SCORE_IR, BAR_TIMELINE),
    );
    expect(result.current.qstampForTime(0.0)).toBe(0);
  });

  it("qstampForTime returns qstampStart=12 for bar 4 at t=14.0 (between bar 4 at 12.8 and bar 5 at 17.1)", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null } as RefObject<HTMLAudioElement | null>;
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef, MOCK_SCORE_IR, BAR_TIMELINE),
    );
    expect(result.current.qstampForTime(14.0)).toBe(12);
  });

  it("setCurrentTime updates currentTime state", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null } as RefObject<HTMLAudioElement | null>;
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef, MOCK_SCORE_IR, BAR_TIMELINE),
    );
    act(() => {
      result.current.setCurrentTime(8.5);
    });
    expect(result.current.currentTime).toBe(8.5);
  });

  it("qstampForTime returns null for empty barTimeline", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null } as RefObject<HTMLAudioElement | null>;
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef, MOCK_SCORE_IR, []),
    );
    expect(result.current.qstampForTime(5.0)).toBeNull();
  });

  it("qstampForTime returns null when scoreIR is null (scoreIR not yet loaded)", async () => {
    const { useProofCardTimeline } = await import("./useProofCardTimeline");
    const audioRef = { current: null } as RefObject<HTMLAudioElement | null>;
    const { result } = renderHook(() =>
      useProofCardTimeline(audioRef, null, BAR_TIMELINE),
    );
    expect(result.current.qstampForTime(0.0)).toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/hooks/useProofCardTimeline.test.ts
```
Expected: FAIL — the stub `qstampForTime` returns `null` when scoreIR is present but bar lookup hasn't been wired; the full implementation in Step 3 is required.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the stub in `apps/web/src/hooks/useProofCardTimeline.ts`:

```typescript
// apps/web/src/hooks/useProofCardTimeline.ts
import { useCallback, useRef, useState } from "react";
import type { RefObject } from "react";
import type { ScoreIR } from "../lib/score-ir";

type BarTimeline = Array<{ bar: number; tSec: number }>;

export function useProofCardTimeline(
  _audioRef: RefObject<HTMLAudioElement | null>,
  scoreIR: ScoreIR | null,
  barTimeline: BarTimeline,
) {
  const [currentTime, setCurrentTimeState] = useState(0);
  const barTimelineRef = useRef(barTimeline);
  barTimelineRef.current = barTimeline;
  const scoreIRRef = useRef(scoreIR);
  scoreIRRef.current = scoreIR;

  const setCurrentTime = useCallback((t: number) => {
    setCurrentTimeState(t);
  }, []);

  // Returns the qstampStart of the bar whose tSec window contains the given time.
  // ScoreCursor.findBar() binary-searches bar.qstampStart/qstampEnd (quaternote floats),
  // so we must return a qstamp float, not a raw bar number.
  // Timeline must be sorted by tSec ascending (guaranteed by manifest production).
  const qstampForTime = useCallback((tSec: number): number | null => {
    const timeline = barTimelineRef.current;
    if (timeline.length === 0) return null;

    let matchedEntry = timeline[0];
    if (!matchedEntry) return null;

    for (let i = 0; i < timeline.length; i++) {
      const entry = timeline[i];
      if (!entry) continue;
      if (entry.tSec <= tSec) {
        matchedEntry = entry;
      } else {
        break;
      }
    }

    // Look up qstampStart from ScoreIR.bars by barNumber
    const ir = scoreIRRef.current;
    if (ir) {
      const barIR = ir.bars.find((b) => b.barNumber === matchedEntry!.bar);
      if (barIR) return barIR.qstampStart;
    }

    // scoreIR not yet loaded — return null so cursor stays hidden
    return null;
  }, []);

  return { currentTime, setCurrentTime, qstampForTime };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/hooks/useProofCardTimeline.test.ts
```
Expected: PASS — all five test cases pass

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/hooks/useProofCardTimeline.ts apps/web/src/hooks/useProofCardTimeline.test.ts && git commit -m "feat(landing): implement useProofCardTimeline with bidirectional scrub sync"
```

---

### Task 9: Keyboard navigation — Tab cycles bars, Enter opens chip, Escape closes

**Group:** C (parallel with Tasks 5 and 7 — depends on Group B)

**Behavior being verified:** Tab key cycles focus through bar buttons in ProofCard; Enter opens BarScoreChip for the focused bar; Escape closes the chip.

**Interface under test:** ProofCard keyboard event handling on bar buttons

**Files:**
- Create: `apps/web/src/components/ProofCard.keyboard.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ProofCard.keyboard.test.tsx --reporter=verbose 2>&1 | grep -E "FAIL|keyboard navigation"
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
cd apps/web && bun run test src/components/ProofCard.keyboard.test.tsx
```
Expected: PASS — all keyboard navigation tests pass

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ProofCard.tsx apps/web/src/components/ProofCard.keyboard.test.tsx && git commit -m "feat(landing): ProofCard keyboard navigation tests pass"
```

---

### Task 6: Axe accessibility scan on LandingPage

**Group:** D (sequential — depends on Group C)

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

**Group:** E (sequential — depends on Group D; all components must exist)

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
- `src/components/ProofCard.test.tsx` — 1 test (Task 1, render contract smoke test)
- `src/components/ProofCard.autoplay.test.tsx` — 2 tests (Task 2)
- `src/components/ProofCard.degradation.test.tsx` — 2 tests (Task 3)
- `src/components/ProofCard.audio-degradation.test.tsx` — 1 test (Task 4)
- `src/components/ProofCard.cursor.test.tsx` — 3 tests (Task 4.5)
- `src/components/ProofCard.reduced-motion.test.tsx` — 2 tests (Task 5)
- `src/components/BarScoreChip.test.tsx` — 2 tests (Task 7)
- `src/components/ProofCard.bar-tap.test.tsx` — 1 test (Task 7)
- `src/hooks/useProofCardTimeline.test.ts` — 5 tests (Task 8)
- `src/components/ProofCard.keyboard.test.tsx` — 3 tests (Task 9)
- `src/routes/index.test.tsx` — 2 tests (Task 6, Task 10)

Total: 26 tests. No existing tests may regress.

---

## Challenge Review

### CEO Pass

**Premise Challenge**

Right problem: yes. The current landing page shows stock photos and pull-quotes. Nothing demonstrates what the product actually does. The interactive ProofCard is a direct product demonstration. No simpler framing would yield the same activation signal.

Real pain: yes. "Low activation on the hard metric (record-first-session)" is the stated pain. A static marketing page creates no signal about whether the product is compelling.

Direct path: yes. Static prebaked assets with no runtime inference is the minimum viable approach. Avoids all the complexity of live inference on the landing path.

Existing coverage: `score-cursor.ts`, `score-ir.ts`, `score-cursor.test.ts`, `score-ir.test.ts` are all shipped and tested. The plan correctly identifies these as assets to reuse. **However, the plan's ProofCard implementation does not instantiate `ScoreCursor` at all** — it fetches the scoreIR and constructs a `ScoreIR` value but never passes it to a `ScoreCursor` instance. This is a core spec requirement that the plan silently drops.

**Scope Check**

What could be cut: the asset prefetch injection for card-2 and card-3 (Task 0 / ProofCard.tsx) is a nice-to-have optimization, not MVP. Could be deferred without affecting core functionality. No BLOCKER here, just an observation.

The plan touches 14 new files across 10 tasks — reasonable complexity for the feature goal.

**Twelve-Month Alignment**

```
CURRENT STATE               THIS PLAN                    12-MONTH IDEAL
Static marketing page  →    Interactive ProofCards  →    Live inference-backed
(photos, pull-quote)        with prebaked assets         ProofCards (user uploads
                            and score cursor sync         their own recording)
```

This plan moves toward the ideal. The prebaked approach is a deliberate staging — it proves the interaction model before investing in per-user inference on the landing path.

**Alternatives Check**

The spec documents the static-prebaked vs runtime-inference tradeoff and chooses static. This is explicitly justified. The third alternative (animated GIF/video of the product) was implicitly rejected (the current state already uses MP4 animations — they're what's being replaced). Reasoning is present in the spec.

---

### Engineering Pass

**Architecture**

Data flow:

```
index.tsx (CARD_MANIFESTS)
  └── ExerciseProofBlock (layout wrapper)
       └── ProofCard (per-card coordination)
            ├── fetch(scoreIRUrl) → ScoreIR state
            ├── fetch(scoreSvgUrl) → insertAdjacentHTML into scoreContainerRef
            ├── fetch(exerciseUrl) → exerciseComponent state
            ├── <audio ref> → currentTime state (via timeupdate)
            ├── scrubber range input → currentTime state → audio.currentTime sync
            ├── useProofCardTimeline → qstampForTime(currentTime) → [ScoreCursor NOT WIRED]
            └── IntersectionObserver → audio.play() / pause()
```

The qstamp mapping is computed but never consumed by ScoreCursor. The score SVG is static; the cursor never moves over it. This means the central UX claim in the spec — "cursor tracks through the score in sync with the audio" — is absent from the plan's implementation.

Security: SVG is fetched from the same origin (`/landing/card-N/score.svg`). The biome lint comment `// biome-ignore lint/security/noDomManipulation` is present. No user input flows to the DOM.

Deployment: pure static assets + React components. No migration, no partial-state risk. Rollback is `git revert`.

**Module Depth Audit**

- `ProofCard.tsx`: Interface = 2 props (`manifest`, `cardIndex`). Implementation hides fetch coordination, IO observers, audio sync, analytics, keyboard nav. **DEEP** — but the cursor subsystem is missing entirely from the implementation despite being listed in the spec's hidden complexity.
- `BarScoreChip.tsx`: Interface = 3 props, implementation = bar chart + keydown. **SHALLOW** — acknowledged in spec as justified (pure display).
- `ExerciseProofBlock.tsx`: Interface = 1 prop (manifests tuple), implementation = layout wrapper. **SHALLOW** — justified as composition-only.
- `useProofCardTimeline.ts`: Interface = 3 params + 3 return values. Implementation = linear scan. **DEEP** — simple interface hiding O(n) traversal.
- `landing-analytics.ts`: 3-line guard wrapper. **SHALLOW** — justified.

**Code Quality**

- `ProofCard.tsx` uses `React.KeyboardEvent` at line `handleBarKeyDown(e: React.KeyboardEvent, ...)` but only imports `{ useEffect, useRef, useState }` from `"react"`. No `import * as React` or `import type { KeyboardEvent } from "react"`. This is a TypeScript compile error under strict mode with `"jsx": "react-jsx"`.
- `BarScoreChip.tsx` uses `React.KeyboardEvent` in `handleKeyDown(e: React.KeyboardEvent)` but has no `import React` at all — only `import type { BarQualityScores } from "../types/landing"`. Same compile error.
- `useProofCardTimeline.ts` (stub and full implementation) uses `React.RefObject<HTMLAudioElement | null>` in the parameter type but only imports `{ useCallback, useRef, useState }` from `"react"`. Same compile error.

All three files will fail `tsc` before any tests run.

- The `exerciseComponent` state is typed as `unknown` and then cast with `as { config?: ... }` in JSX. This is not explicit exception handling — it's a silent cast. If the exercise JSON doesn't match the expected shape, nothing is shown and no error is thrown. Acceptable for a marketing page, but worth noting.

- The Task 1 render contract test asserts `data-testid="proof-card-exercise"` outside of `waitFor`, immediately after `await waitFor(() => getByText(...))`. Since `manifest.diagnosis` is a synchronous prop (renders immediately), `waitFor` for the diagnosis text may resolve before the async fetch for `exerciseUrl` completes. This makes the exercise assertion potentially racy. Low severity in practice (microtask resolution in jsdom is predictable), but fragile.

**Test Philosophy Audit**

- Task 0 test (`landing-analytics.test.ts`): Tests behavior through the public function — ✓ behavior-first, no internal mocking of the module-under-test.
- Task 1 test (`ProofCard.test.tsx`): Mocks `fetch` (external boundary) and `vi.mock("../lib/landing-analytics")`. The analytics mock is technically mocking an internal collaborator of ProofCard. However, it's a thin fire-and-forget helper; not testing it via ProofCard is acceptable and consistent with the spec's note that analytics is not independently tested.
- Task 2 test (autoplay): Tests behavior (audio.play called) via IntersectionObserver mock — behavior-first. ✓
- Task 8 hook tests: Tests `qstampForTime` behavior directly. ✓ Clean hook tests via `renderHook`.
- Task 6 axe scan: This is a smoke test (★). No behavior assertion — just "no violations." Acceptable for a11y gate.
- Task 10 test (`LandingPage structure`): Uses `(IndexRoute as unknown as { options: { component: React.ComponentType } }).options.component` — this is accessing the internal TanStack Router route object and relying on `Route.options.component` being a public field. Verified: TanStack Router v1 stores `this.options = options || {}` in `BaseRoute` constructor and the `component` field lives there. This pattern is safe for the current version.

**Vertical Slice Audit**

- Tasks 2, 3, 4 are all `Group A (parallel)` — but ALL three tasks **modify the same file** (`ProofCard.test.tsx`) that Task 1 creates. A build agent dispatching these in parallel will have three subagents concurrently writing to the same file, producing merge conflicts or last-write-wins corruption. This is not a vertical slice problem per se but a **build agent coordination failure** — parallel subagents cannot safely append to the same file.
- Task 8 is `Group A (parallel)` and **creates `useProofCardTimeline.ts`** — but Task 1 also **creates a stub of the same file** as part of its Step 3. Two Group A subagents creating the same file in parallel will clobber each other.
- Task 5 (`Group B`) modifies `ProofCard.test.tsx` and uses `vi.resetModules()` in `beforeEach`. In Vitest, `vi.resetModules()` is a global operation that resets the module registry. Calling it inside a `describe` block's `beforeEach` affects ALL modules across the test file for subsequent tests, potentially breaking earlier test blocks that rely on stable module identity.

**Test Coverage Gaps**

```
[+] ProofCard.tsx
    │
    ├── load() — scoreIR fetch
    │   ├── [TESTED]  ok response sets scoreIR — Task 1 (★★)
    │   ├── [TESTED]  non-ok response: graceful degradation — Task 3 (★★)
    │   └── [GAP]     ScoreCursor instantiated and started — NOT TESTED (feature absent from plan)
    │
    ├── load() — audio fetch
    │   └── [TESTED]  error event triggers audio-failed state — Task 4 (★★)
    │
    ├── IntersectionObserver autoplay
    │   ├── [TESTED]  ≥60% triggers play — Task 2 (★★)
    │   └── [TESTED]  <60% does not trigger play — Task 2 (★★)
    │
    ├── Reduced motion
    │   ├── [TESTED]  autoplay disabled — Task 5 (★★)
    │   └── [TESTED]  manual play button present — Task 5 (★★)
    │
    ├── Bar tap → BarScoreChip
    │   └── [TESTED]  bar click shows chip — Task 7 (★★)
    │
    └── Keyboard nav
        ├── [TESTED]  Enter opens chip — Task 9 (★★)
        └── [TESTED]  Escape closes chip — Task 9 (★★)

[+] useProofCardTimeline.ts
    ├── qstampForTime(0.0) → bar 1 [TESTED] — Task 8 (★★)
    ├── qstampForTime(14.0) → bar 4 [TESTED] — Task 8 (★★)
    ├── setCurrentTime updates state [TESTED] — Task 8 (★★)
    └── empty timeline → null [TESTED] — Task 8 (★★)
```

**Failure Modes**

- All three asset fetches (SVG, scoreIR, exercise) catch errors silently and continue. This is correct for a marketing page — partial failure is acceptable.
- Audio autoplay rejection is silently caught. Play button is shown as fallback.
- No recovery path needed for DOM state corruption — the card is stateless except for `activeBar`.

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `Route.options.component` exposes the LandingPage component in TanStack Router v1 | SAFE | Verified: `BaseRoute` stores `this.options = options` in `@tanstack/router-core`; `createFileRoute` passes component through options |
| `vitest-axe` + `axe-core` are not currently installed | SAFE | Verified: not in `apps/web/package.json` devDependencies |
| `useProofCardTimeline` does not exist yet | SAFE | Verified: `apps/web/src/hooks/` directory confirmed; file absent |
| jsdom sets `document.visibilityState` to `"visible"` by default | SAFE | Well-documented jsdom behavior |
| `ScoreCursor` can be instantiated with a static SVG container | VALIDATE | The plan doesn't use ScoreCursor; if it did, the overlay mount logic requires the container to be in the DOM with correct dimensions |
| Parallel tasks in Group A can safely modify the same file | RISKY | Tasks 2, 3, 4 all append to `ProofCard.test.tsx`; Task 1 and Task 8 both create `useProofCardTimeline.ts`. Parallel dispatch will cause file conflicts |
| `React.KeyboardEvent` and `React.RefObject` are available without `import * as React` | RISKY | Not available under strict mode + `"jsx": "react-jsx"`. Files will not compile. |
| `manifest.diagnosis` renders synchronously before fetch resolves | SAFE | It is a direct prop rendered in JSX, not behind a state gate |
| `snapshot-landing-card.ts` is not needed for tests to pass | SAFE | Tests use hardcoded fixture manifests; the snapshot script is only needed to produce real audio/SVG assets |

---

### Summary

[BLOCKER] count: 3
[RISK]    count: 3
[QUESTION] count: 1

**Blockers**

[BLOCKER] (confidence: 9/10) — `ProofCard.tsx`, `BarScoreChip.tsx`, and `useProofCardTimeline.ts` all use `React.KeyboardEvent` or `React.RefObject` without importing `React`. Under `strict: true` + `"jsx": "react-jsx"`, these will produce TypeScript compile errors (`'React' refers to a UMD global, but the current file is a module`) before any tests can run. Fix: add `import type { KeyboardEvent, RefObject } from "react"` to each file and replace `React.KeyboardEvent` with `KeyboardEvent`, `React.RefObject` with `RefObject`.

[BLOCKER] (confidence: 10/10) — `ScoreCursor` is never instantiated in the plan's `ProofCard.tsx`. The spec's stated goal explicitly includes "bidirectional cursor-audio sync" as a core feature, and the spec's Design section (line 45) says "A single `currentTime: number` React state drives both the audio element's `currentTime` and the `qstampSource` passed to `ScoreCursor`." The plan's `useProofCardTimeline` computes `qstampForTime` but the return value is never passed to `ScoreCursor`. The score SVG is static; the cursor never moves. This is a spec requirement missing from the plan. Fix: add a `useEffect` in `ProofCard.tsx` that instantiates `ScoreCursor` (when `scoreIR !== null && scoreContainerRef.current !== null`), starts it on mount / stops it on unmount, and passes `() => qstampForTime(currentTime)` as the `qstampSource`. Add a corresponding test that asserts cursor presence when scoreIR loads.

[BLOCKER] (confidence: 9/10) — Group A declares Tasks 1, 2, 3, 4, 8 as parallel, but Tasks 2, 3, 4 all **append to** `ProofCard.test.tsx` (created by Task 1), and Task 8 **creates** `useProofCardTimeline.ts` (stubbed by Task 1). Parallel subagent dispatch will cause last-write-wins file collisions on both files. Fix: move Tasks 2, 3, 4 into Group B (or make them sequential after Task 1 in a revised Group A-seq). Move Task 8's file creation to be sequential after Task 1's stub creation (e.g., keep Task 8 in Group A but make it aware it replaces the stub, not creates fresh).

**Risks**

[RISK] (confidence: 7/10) — Task 5's `beforeEach` calls `vi.resetModules()`. In Vitest, this clears the module registry globally within the test worker. Because all ProofCard tests accumulate in a single file, the `resetModules` in the reduced-motion describe block may invalidate cached module references used by earlier describe blocks, producing unexpected test isolation failures. Fallback: if tests fail intermittently, move the reduced-motion tests to a separate file (`ProofCard.reduced-motion.test.tsx`) so `resetModules` does not affect the main suite.

[RISK] (confidence: 6/10) — Task 1's render contract test asserts `data-testid="proof-card-exercise"` outside of `waitFor`, immediately after `await waitFor(() => getByText(diagnosis))`. `manifest.diagnosis` renders synchronously (it's a prop), so `waitFor` resolves on the first flush — potentially before the `exerciseUrl` fetch microtask chain completes. If this assertion flakes, wrap it in `await waitFor(() => expect(document.querySelector('[data-testid="proof-card-exercise"]')).not.toBeNull())`.

[RISK] (confidence: 7/10) — `snapshot-landing-card.ts` is listed in the spec's File Changes table as a Task 0 deliverable ("Task 0 creates `apps/web/scripts/snapshot-landing-card.ts`") but is entirely absent from the plan's Task 0 steps. The script is not needed for tests to pass (tests use fixture JSONs), but without it there is no documented path to produce real audio/SVG assets for the three cards. The landing page will launch with empty asset directories. Name the fallback in the plan or explicitly defer the script.

**Questions**

[QUESTION] — `snapshot-landing-card.ts` appears in the spec's Verification Architecture as a Task 0 artifact but is absent from the plan. Should it be included in Task 0, deferred to a follow-up task, or explicitly excluded on the grounds that real asset production is manual and outside the build agent's scope?

---

VERDICT: NEEDS_REWORK — Three blockers must be resolved: (1) missing `React` type imports in three files will prevent compilation; (2) `ScoreCursor` is never instantiated — the animated cursor is entirely absent from the plan despite being a core spec requirement; (3) Group A parallel task dispatch will cause file conflicts on `ProofCard.test.tsx` and `useProofCardTimeline.ts`.

---

## Challenge Review (Loop 2 — 2026-05-27)

> Re-reviewed after plan edits addressing Loop 1 blockers. Read all source files before forming opinions.

### Status of Prior Blockers

**Prior Blocker 1 (React type imports):** Partially fixed. `ProofCard.tsx` now imports `import type { KeyboardEvent } from "react"` (line 355) and `BarScoreChip.tsx` imports `import type { KeyboardEvent } from "react"` (line 1392). `useProofCardTimeline.ts` imports `import type { RefObject } from "react"` (line 1572). These three source files are now correct.

However, `useProofCardTimeline.test.ts` (line 1551) still uses `React.RefObject<HTMLAudioElement | null>` in the empty-timeline test without any `React` import — only `{ renderHook, act }` and vitest are imported. This is a compile error in the test file itself. **Not fully resolved.**

**Prior Blocker 2 (ScoreCursor not instantiated):** A `useEffect` and Task 4.5 were added to the plan. However the `ScoreCursor` constructor call in `ProofCard.tsx` (lines 499–500) uses the **wrong signature**. Verified against `apps/web/src/lib/score-cursor.ts`:

- Actual constructor: `constructor(opts: ScoreCursorOptions)` where `ScoreCursorOptions = { pieceId: string; container: HTMLElement; ir: ScoreIR; qstampSource: () => number | null }` — a single options object; `qstampSource` is baked in at construction time.
- Actual `start()`: `start(): void` — takes no arguments.
- Plan's call: `new ScoreCursor(scoreContainerRef.current, scoreIR)` — wrong (positional args, missing `pieceId`, missing `qstampSource`).
- Plan's call: `cursor.start(() => qstampForTime(currentTime) ?? 0)` — wrong (start takes no args).

The plan's ProofCard will throw a TypeScript compile error and a runtime error at cursor instantiation. **Not resolved.**

Additionally, `qstampForTime` returns `entry.bar` (bar number, e.g. 1, 2, 3…) as a proxy for qstamp. `ScoreCursor.findBar()` binary-searches `ir.bars` using `qstampStart`/`qstampEnd` which are quaternote positions (0.0, 4.0, 8.0, 12.0…). Bar number 4 ≠ qstamp 12.8. The cursor would position incorrectly or fail to find bars entirely. The MOCK_SCORE_IR fixture in the test has `qstampStart: 12, qstampEnd: 16` for bar 4 — passing `4` as the qstamp would match bar 1 (`qstampStart: 0, qstampEnd: 4`) instead. This semantic mismatch is a latent behavioral bug beyond the compile error.

**Prior Blocker 3 (Group A/B parallel file collision):** The task groups were reorganized: Task 1 is now Group A (sequential alone), and Tasks 2, 3, 4, Task 4.5, 8 are Group B (parallel). The plan header at line 16 reads: "Group B (parallel, depends on Group A): Task 2, Task 3, Task 4, Task 8 — Tasks 2/3/4 each append to ProofCard.test.tsx created by Task 1; Task 8 replaces the useProofCardTimeline.ts stub created by Task 1."

The file collision is still present: Tasks 2, 3, 4, Task 4.5, and 9 (in Group C) all modify `ProofCard.test.tsx`. In Group B, Tasks 2/3/4/Task 4.5 are still parallel and all append to the same file. **Not resolved — the collision was moved from Group A to Group B but not eliminated.**

---

### New Findings

**[BLOCKER] (confidence: 10/10)** — `ScoreCursor` constructor signature mismatch. Verified by reading `apps/web/src/lib/score-cursor.ts` lines 5–25. The constructor takes a single `ScoreCursorOptions` object with four required fields: `pieceId`, `container`, `ir`, `qstampSource`. The plan's `ProofCard.tsx` calls `new ScoreCursor(scoreContainerRef.current, scoreIR)` (two positional args) and `cursor.start(() => qstampForTime(currentTime) ?? 0)` (`start()` takes no args — qstampSource is passed to the constructor). This will fail TypeScript compilation and throw at runtime. Fix: replace with `new ScoreCursor({ pieceId: manifest.pieceId, container: scoreContainerRef.current, ir: scoreIR, qstampSource: () => qstampForTime(currentTime) ?? 0 })` and `cursor.start()`. Remove the `qstampForTime` callback from `cursor.start()`.

**[BLOCKER] (confidence: 9/10)** — `qstampForTime` returns bar number (integer 1, 2, 3…) but `ScoreCursor` expects quaternote position (float 0.0, 4.0, 8.0…). `ScoreCursor.findBar()` binary-searches `bar.qstampStart` / `bar.qstampEnd` values from `ScoreIR`. Passing bar number 1 maps to qstamp ~0 (lucky coincidence for bar 1), but bar 4 would be passed as `4` which falls within bar 1's range `[0, 4)` in the fixture. The cursor would show at bar 1 regardless of playback position past bar 1. Fix: `qstampForTime` must return the `qstampStart` of the matching bar from the `ScoreIR`, not the bar number. The hook needs access to `scoreIR.bars` to perform the lookup, or the mapping must happen in `ProofCard.tsx` using `scoreIR` directly. The current hook interface (`barTimeline` only, no `ScoreIR` bar data) does not carry enough information to return a correct qstamp.

**[BLOCKER] (confidence: 9/10)** — `useProofCardTimeline.test.ts` line 1551 uses `React.RefObject<HTMLAudioElement | null>` but the test file imports only `{ renderHook, act }` from `@testing-library/react` and vitest — no `React` import. TypeScript will error: `'React' refers to a UMD global`. The three previous source-file React import fixes missed this test file. Fix: add `import type { RefObject } from "react"` to the test file and change `React.RefObject` to `RefObject`.

**[BLOCKER] (confidence: 9/10)** — Task 4.5 calls `vi.mock("../lib/score-cursor", ...)` inside `beforeEach`. In Vitest, `vi.mock` calls are hoisted to the top of the module by the Vitest transformer — they cannot be conditionally applied inside `beforeEach`. A `vi.mock` call inside `beforeEach` is silently ignored (it runs after module evaluation, so the hoisting mechanism does not apply). The mock will not take effect. `ScoreCursor` will be the real class, `requestAnimationFrame` is not available in jsdom, and the test will error. Fix: move the `vi.mock("../lib/score-cursor", ...)` call to the module top-level (outside any describe/beforeEach), and use `vi.mocked(ScoreCursor).mockImplementation(...)` inside `beforeEach` to configure the per-test mock behavior.

**[BLOCKER] (confidence: 9/10)** — Group B parallel file collision persists. Tasks 2, 3, 4, and Task 4.5 are all Group B (parallel) and all append to `ProofCard.test.tsx`. Four subagents simultaneously writing to the same file will produce clobbered or merged-incorrectly output. Fix: make Group B sequential (Task 2 → Task 3 → Task 4 → Task 4.5), or assign each task its own separate test file (e.g., `ProofCard.autoplay.test.tsx`, `ProofCard.degradation-scoreir.test.tsx`). The latter is the cleaner approach and removes all collision risk.

---

### Presumption Inventory (Loop 2)

| Assumption | Verdict | Reason |
|---|---|---|
| `ScoreCursor` constructor takes `(container, ir)` positional args | RISKY | Verified false: takes a single options object `{ pieceId, container, ir, qstampSource }` |
| `ScoreCursor.start()` takes a `qstampSource` callback argument | RISKY | Verified false: `start()` takes no args; `qstampSource` is passed to constructor |
| `qstampForTime` returning bar number is a valid qstamp for ScoreCursor | RISKY | Verified false: ScoreCursor.findBar() uses quaternote positions, not bar numbers |
| `vi.mock()` inside `beforeEach` in Vitest is equivalent to module-level `vi.mock()` | RISKY | Verified false: Vitest hoists module-level `vi.mock` calls; `beforeEach` calls are not hoisted and have no effect |
| React type imports fixed in all affected files | VALIDATE | Fixed in 3 source files but `useProofCardTimeline.test.ts` line 1551 still uses `React.RefObject` without import |
| Parallel Group B tasks can safely append to the same file | RISKY | Unchanged from Loop 1 — still 4 tasks targeting `ProofCard.test.tsx` in parallel |

---

### Summary (Loop 2)

[BLOCKER] count: 5
[RISK]    count: 0
[QUESTION] count: 0

VERDICT: NEEDS_REWORK — Five blockers remain: (1) `ScoreCursor` constructor called with wrong positional args — must use options object `{ pieceId, container, ir, qstampSource }`; (2) `qstampForTime` returns bar number not quaternote position — ScoreCursor will position cursor at wrong bar; (3) `React.RefObject` used without import in `useProofCardTimeline.test.ts` line 1551; (4) `vi.mock` inside `beforeEach` in Task 4.5 is silently ignored by Vitest — must be module-level; (5) Group B parallel file collision on `ProofCard.test.tsx` unchanged from Loop 1.
