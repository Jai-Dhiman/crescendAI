# Phase 3: Practice Pipeline — Real-Time Analysis + Durable Objects

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the complete real-time practice pipeline from Rust to TypeScript — the Durable Object session brain, WebSocket communication, parallel inference dispatch, score following, piece identification, teaching moment selection, practice mode state machine, and session synthesis.

**Architecture:** A single `SessionBrain` Durable Object manages each practice session's lifecycle. It receives chunk-ready notifications via WebSocket, dispatches parallel MuQ + AMT inference, runs DTW score following, accumulates teaching moments, and triggers synthesis via alarm. Pure algorithm services (STOP classifier, mode detector, score follower, piece ID) are stateless TypeScript modules. The DO orchestrates them via its per-chunk pipeline.

**Tech Stack:** Hono, CF Durable Objects (Hibernation API), CF Workflows (synthesis), Drizzle ORM, Zod, R2 buckets, AI Gateway

**Style Guide:** ALL code must follow `apps/api/TS_STYLE.md`. Critical DO rules: `this.ctx.acceptWebSocket()` (not `server.accept()`), alarms (not setTimeout), clone-before-await + compare-and-swap for state versioning, Zod validation on every storage read.

**Decision: TS-first for algorithms.** The spec mentions WASM extraction for DTW and N-gram. Given pre-beta data sizes (max 600 notes, 242 pieces), we'll implement all algorithms in TypeScript first. WASM extraction can be a follow-up optimization if profiling shows a need. This avoids wasm-pack build tooling, serde-wasm-bindgen complexity, and bundle size overhead during rapid iteration.

---

## File Structure

```
apps/api/src/
  services/
    stop-classifier.ts        -- CREATE: logistic regression STOP model
    teaching-moments.ts        -- CREATE: teaching moment selection algorithm
    practice-mode.ts           -- CREATE: ModeDetector state machine
    accumulator.ts             -- CREATE: SessionAccumulator + types
    score-follower.ts          -- CREATE: subsequence DTW + bar alignment
    piece-identify.ts          -- CREATE: N-gram recall + rerank + DTW confirm
    bar-analysis.ts            -- CREATE: Tier 1/2/3 per-dimension analysis
    inference.ts               -- CREATE: MuQ + AMT endpoint callers with retry
    synthesis.ts               -- CREATE: synthesis prompt + LLM call + persistence
    ask.ts                     -- CREATE: two-stage teaching pipeline (subagent + teacher)
    prompts.ts                 -- MODIFY: add subagent, teacher, synthesis prompts
  do/
    session-brain.ts           -- CREATE: Durable Object class
    session-brain.schema.ts    -- CREATE: Zod schemas for DO state
  routes/
    practice.ts                -- CREATE: start, chunk upload, WS upgrade, deferred synthesis
    practice.test.ts           -- CREATE: route tests
  lib/
    types.ts                   -- MODIFY: add DO namespace, AMT binding to Bindings
    dims.ts                    -- CREATE: dimension constants and mapping
  index.ts                     -- MODIFY: mount practice routes, export DO
wrangler.toml                  -- MODIFY: add DO binding
```

---

### Task 1: STOP Classifier + Teaching Moment Selection

Two tightly coupled pure algorithms. The STOP classifier is a logistic regression model with hardcoded weights. Teaching moment selection runs STOP on scored chunks, finds max negative deviation, and deduplicates against recent observations.

**Files:**
- Create: `apps/api/src/lib/dims.ts`
- Create: `apps/api/src/services/stop-classifier.ts`
- Create: `apps/api/src/services/teaching-moments.ts`

- [ ] **Step 1: Create dimension constants**

```typescript
// apps/api/src/lib/dims.ts
export const DIMS_6 = [
  "dynamics",
  "timing",
  "pedaling",
  "articulation",
  "phrasing",
  "interpretation",
] as const;

export type Dimension = (typeof DIMS_6)[number];

export const DIM_INDEX: Record<Dimension, number> = {
  dynamics: 0,
  timing: 1,
  pedaling: 2,
  articulation: 3,
  phrasing: 4,
  interpretation: 5,
};
```

- [ ] **Step 2: Write STOP classifier tests**

```typescript
// apps/api/src/services/stop-classifier.test.ts
import { describe, it, expect } from "vitest";
import { classify, stopProbability } from "./stop-classifier";

describe("STOP classifier", () => {
  it("returns probability between 0 and 1", () => {
    const prob = stopProbability([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    expect(prob).toBeGreaterThanOrEqual(0);
    expect(prob).toBeLessThanOrEqual(1);
  });

  it("triggers on low dynamics + pedaling (negative weights)", () => {
    // Very low dynamics and pedaling, high timing — should trigger STOP
    const result = classify([0.3, 0.55, 0.2, 0.54, 0.52, 0.35]);
    expect(result.probability).toBeGreaterThan(0.5);
    expect(result.triggered).toBe(true);
  });

  it("does not trigger on average scores", () => {
    // Scores near scaler mean should be close to sigmoid(bias)
    const result = classify([0.545, 0.485, 0.459, 0.537, 0.519, 0.506]);
    expect(result.probability).toBeCloseTo(0.5286, 2); // sigmoid(0.1147)
  });

  it("identifies top contributing dimension", () => {
    const result = classify([0.3, 0.5, 0.2, 0.54, 0.52, 0.5]);
    expect(result.topDimension).toBeDefined();
    expect(typeof result.topDeviation).toBe("number");
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd apps/api && bun run test -- --run src/services/stop-classifier.test.ts`

- [ ] **Step 4: Implement STOP classifier**

```typescript
// apps/api/src/services/stop-classifier.ts
import type { Dimension } from "../lib/dims";
import { DIMS_6 } from "../lib/dims";

// Trained on 1,699 labeled masterclass segments, balanced logistic regression, LOVO CV AUC = 0.649
// Dimension order: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
export const SCALER_MEAN = [0.545, 0.4848, 0.4594, 0.5369, 0.5188, 0.5064];
const SCALER_STD = [0.0689, 0.0388, 0.0791, 0.0154, 0.0186, 0.0555];
const WEIGHTS = [-0.5266, 0.3681, -0.5483, 0.4884, 0.2427, -0.1541];
const BIAS = 0.1147;
const DEFAULT_THRESHOLD = 0.5;

export interface StopResult {
  probability: number;
  triggered: boolean;
  topDimension: Dimension;
  topDeviation: number;
}

export function stopProbability(scores: number[]): number {
  let logit = BIAS;
  for (let i = 0; i < 6; i++) {
    const scaled = (scores[i] - SCALER_MEAN[i]) / SCALER_STD[i];
    logit += scaled * WEIGHTS[i];
  }
  return 1 / (1 + Math.exp(-logit));
}

export function classify(scores: number[], threshold = DEFAULT_THRESHOLD): StopResult {
  const probability = stopProbability(scores);

  // Find top contributing dimension by |scaled * weight|
  let maxContribution = 0;
  let topIdx = 0;
  for (let i = 0; i < 6; i++) {
    const scaled = (scores[i] - SCALER_MEAN[i]) / SCALER_STD[i];
    const contribution = Math.abs(scaled * WEIGHTS[i]);
    if (contribution > maxContribution) {
      maxContribution = contribution;
      topIdx = i;
    }
  }

  return {
    probability,
    triggered: probability >= threshold,
    topDimension: DIMS_6[topIdx],
    topDeviation: (scores[topIdx] - SCALER_MEAN[topIdx]) / SCALER_STD[topIdx],
  };
}
```

- [ ] **Step 5: Run tests, verify pass**

- [ ] **Step 6: Write teaching moment selection tests**

```typescript
// apps/api/src/services/teaching-moments.test.ts
import { describe, it, expect } from "vitest";
import { selectTeachingMoment } from "./teaching-moments";
import type { ScoredChunk, StudentBaselines } from "./teaching-moments";

const baselines: StudentBaselines = {
  dynamics: 0.55, timing: 0.48, pedaling: 0.46,
  articulation: 0.54, phrasing: 0.52, interpretation: 0.51,
};

describe("selectTeachingMoment", () => {
  it("returns null with fewer than 2 chunks", () => {
    const result = selectTeachingMoment(
      [{ chunkIndex: 0, scores: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] }],
      baselines,
      [],
    );
    expect(result).toBeNull();
  });

  it("selects moment with negative deviation when STOP triggers", () => {
    const chunks: ScoredChunk[] = [
      { chunkIndex: 0, scores: [0.3, 0.55, 0.2, 0.54, 0.52, 0.35] },
      { chunkIndex: 1, scores: [0.3, 0.55, 0.2, 0.54, 0.52, 0.35] },
    ];
    const result = selectTeachingMoment(chunks, baselines, []);
    expect(result).not.toBeNull();
    expect(result!.dimension).toBeDefined();
  });

  it("deduplicates against recent observations", () => {
    const chunks: ScoredChunk[] = [
      { chunkIndex: 0, scores: [0.3, 0.55, 0.2, 0.54, 0.52, 0.35] },
      { chunkIndex: 1, scores: [0.3, 0.55, 0.2, 0.54, 0.52, 0.35] },
    ];
    // If the top dimension was already observed, it should try to find another
    const result = selectTeachingMoment(chunks, baselines, [
      { dimension: "pedaling" },
      { dimension: "dynamics" },
      { dimension: "interpretation" },
    ]);
    expect(result).not.toBeNull();
  });
});
```

- [ ] **Step 7: Implement teaching moment selection**

```typescript
// apps/api/src/services/teaching-moments.ts
import { DIMS_6, type Dimension } from "../lib/dims";
import { classify } from "./stop-classifier";
import { SCALER_MEAN } from "./stop-classifier";

export interface ScoredChunk {
  chunkIndex: number;
  scores: number[]; // [dynamics, timing, pedaling, articulation, phrasing, interpretation]
}

export type StudentBaselines = Record<Dimension, number>;

export interface RecentObservation {
  dimension: string;
}

export interface TeachingMoment {
  chunkIndex: number;
  dimension: Dimension;
  score: number;
  baseline: number;
  deviation: number;
  isPositive: boolean;
  reasoning: string;
}

const MIN_CHUNKS = 2;
const DEDUP_WINDOW = 3;

function maxNegativeDeviation(
  scores: number[],
  baselines: StudentBaselines,
): { dimIdx: number; deviation: number } {
  let minDev = Infinity;
  let minIdx = 0;
  for (let i = 0; i < 6; i++) {
    const dev = scores[i] - baselines[DIMS_6[i]];
    if (dev < minDev) {
      minDev = dev;
      minIdx = i;
    }
  }
  return { dimIdx: minIdx, deviation: minDev };
}

function selectPositiveMoment(
  chunks: ScoredChunk[],
  baselines: StudentBaselines,
): TeachingMoment | null {
  let maxDev = -Infinity;
  let bestChunk: ScoredChunk | null = null;
  let bestDimIdx = 0;

  for (const chunk of chunks) {
    for (let i = 0; i < 6; i++) {
      const dev = chunk.scores[i] - baselines[DIMS_6[i]];
      if (dev > maxDev) {
        maxDev = dev;
        bestChunk = chunk;
        bestDimIdx = i;
      }
    }
  }

  if (!bestChunk) return null;

  const dim = DIMS_6[bestDimIdx];
  return {
    chunkIndex: bestChunk.chunkIndex,
    dimension: dim,
    score: bestChunk.scores[bestDimIdx],
    baseline: baselines[dim],
    deviation: maxDev,
    isPositive: true,
    reasoning: `Positive recognition: ${dim} above baseline`,
  };
}

export function selectTeachingMoment(
  chunks: ScoredChunk[],
  baselines: StudentBaselines,
  recentObservations: RecentObservation[],
): TeachingMoment | null {
  if (chunks.length < MIN_CHUNKS) return null;

  // Find STOP-triggered candidates with max negative deviation
  const candidates: TeachingMoment[] = [];

  for (const chunk of chunks) {
    const stopResult = classify(chunk.scores);
    if (!stopResult.triggered) continue;

    const { dimIdx, deviation } = maxNegativeDeviation(chunk.scores, baselines);
    const dim = DIMS_6[dimIdx];

    candidates.push({
      chunkIndex: chunk.chunkIndex,
      dimension: dim,
      score: chunk.scores[dimIdx],
      baseline: baselines[dim],
      deviation,
      isPositive: false,
      reasoning: `STOP triggered (p=${stopResult.probability.toFixed(3)}), ${dim} deviation: ${deviation.toFixed(3)}`,
    });
  }

  if (candidates.length === 0) {
    return selectPositiveMoment(chunks, baselines);
  }

  // Sort by deviation ascending (most negative first)
  candidates.sort((a, b) => a.deviation - b.deviation);

  // Dedup against recent observations
  const recentDims = new Set(
    recentObservations.slice(0, DEDUP_WINDOW).map((o) => o.dimension),
  );

  for (const candidate of candidates) {
    if (!recentDims.has(candidate.dimension)) {
      return candidate;
    }
  }

  // All deduped — return top candidate anyway
  return candidates[0];
}

/** Default baselines from STOP model scaler means */
export function defaultBaselines(): StudentBaselines {
  const b: Record<string, number> = {};
  for (let i = 0; i < 6; i++) {
    b[DIMS_6[i]] = SCALER_MEAN[i];
  }
  return b as StudentBaselines;
}
```

- [ ] **Step 8: Run tests, verify pass**

- [ ] **Step 9: Commit**

```bash
git add apps/api/src/lib/dims.ts apps/api/src/services/stop-classifier.ts apps/api/src/services/stop-classifier.test.ts apps/api/src/services/teaching-moments.ts apps/api/src/services/teaching-moments.test.ts
git commit -m "feat(api): add STOP classifier and teaching moment selection"
```

---

### Task 2: Practice Mode State Machine

The ModeDetector tracks 5 practice modes (Warming, Drilling, Running, Winding, Regular) with dwell-based transitions, repetition detection, and per-mode observation policies.

**Files:**
- Create: `apps/api/src/services/practice-mode.ts`
- Create: `apps/api/src/services/practice-mode.test.ts`

- [ ] **Step 1: Write tests for mode transitions**

```typescript
// apps/api/src/services/practice-mode.test.ts
import { describe, it, expect } from "vitest";
import { ModeDetector, PracticeMode } from "./practice-mode";
import type { ChunkSignal } from "./practice-mode";

function makeSignal(overrides?: Partial<ChunkSignal>): ChunkSignal {
  return {
    chunkIndex: 0,
    timestampMs: Date.now(),
    barRange: null,
    pitchBigrams: new Set(),
    hasPieceMatch: false,
    barsProgressing: false,
    scores: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ...overrides,
  };
}

describe("ModeDetector", () => {
  it("starts in Warming mode", () => {
    const detector = new ModeDetector();
    expect(detector.mode).toBe(PracticeMode.Warming);
  });

  it("transitions from Warming to Regular after 4 ambiguous chunks", () => {
    const detector = new ModeDetector();
    const now = Date.now();
    for (let i = 0; i < 4; i++) {
      detector.update(makeSignal({ chunkIndex: i, timestampMs: now + i * 15000 }));
    }
    expect(detector.mode).toBe(PracticeMode.Regular);
  });

  it("transitions to Winding on silence gap > 60s", () => {
    const detector = new ModeDetector();
    const now = Date.now();
    detector.update(makeSignal({ chunkIndex: 0, timestampMs: now }));
    // 70s gap
    const transitions = detector.update(
      makeSignal({ chunkIndex: 1, timestampMs: now + 70000 }),
    );
    const windingTransition = transitions.find((t) => t.to === PracticeMode.Winding);
    expect(windingTransition).toBeDefined();
  });

  it("observation policy: Winding suppresses observations", () => {
    const detector = new ModeDetector();
    // Force into winding
    const now = Date.now();
    detector.update(makeSignal({ chunkIndex: 0, timestampMs: now }));
    detector.update(makeSignal({ chunkIndex: 1, timestampMs: now + 70000 }));
    expect(detector.observationPolicy.suppress).toBe(false);
    // After Winding resumes to Regular (no piece match)
    // The suppression only applies while IN Winding, which is brief
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement ModeDetector**

Port the full state machine from `practice/session/practice_mode.rs`. Key elements:
- `PracticeMode` enum: `Warming | Drilling | Running | Winding | Regular`
- `ChunkSignal`: `{ chunkIndex, timestampMs, barRange, pitchBigrams, hasPieceMatch, barsProgressing, scores }`
- `ModeTransition`: `{ from, to, chunkIndex, timestampMs, dwellMs }`
- `ObservationPolicy`: `{ minIntervalMs, suppress, comparative }`
- `ModeDetector` class with `update(signal) -> ModeTransition[]`
- Silence detection: gap > 60s → Winding, then immediate resume evaluation
- Repetition detection: last 3 signals, bar overlap ≥ 0.5 OR pitch bigram Dice ≥ 0.6
- Dwell times: Warming=0 (chunk-count based), Running=30s, Drilling=30s, Regular=15s
- `DrillingPassage` tracking for drilling records

Constants:
```typescript
const SILENCE_GAP_MS = 60_000;
const WARMING_CHUNK_LIMIT = 4;
const RUNNING_DWELL_MS = 30_000;
const DRILLING_DWELL_MS = 30_000;
const REGULAR_DWELL_MS = 15_000;
const RECENT_WINDOW = 4;
const BAR_OVERLAP_THRESHOLD = 0.5;
const DICE_THRESHOLD = 0.6;
```

Observation policies:
```typescript
const MODE_POLICIES: Record<PracticeMode, ObservationPolicy> = {
  Warming: { minIntervalMs: 30_000, suppress: false, comparative: false },
  Drilling: { minIntervalMs: 90_000, suppress: false, comparative: true },
  Running: { minIntervalMs: 150_000, suppress: false, comparative: false },
  Regular: { minIntervalMs: 180_000, suppress: false, comparative: false },
  Winding: { minIntervalMs: 0, suppress: true, comparative: false },
};
```

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/practice-mode.ts apps/api/src/services/practice-mode.test.ts
git commit -m "feat(api): add practice mode state machine with 5-mode transitions"
```

---

### Task 3: Session Accumulator + DO State Schema

Data structures for accumulating teaching moments, mode transitions, drilling records, and timeline events during a session. Also Zod schemas for DO state validation.

**Files:**
- Create: `apps/api/src/services/accumulator.ts`
- Create: `apps/api/src/do/session-brain.schema.ts`

- [ ] **Step 1: Create the accumulator**

```typescript
// apps/api/src/services/accumulator.ts
import type { Dimension } from "../lib/dims";
import { DIMS_6 } from "../lib/dims";
import type { PracticeMode } from "./practice-mode";

export interface AccumulatedMoment {
  chunkIndex: number;
  dimension: Dimension;
  score: number;
  baseline: number;
  deviation: number;
  isPositive: boolean;
  reasoning: string;
  barRange: [number, number] | null;
  analysisTier: number; // 1=full, 2=absolute, 3=scores only
  timestampMs: number;
  llmAnalysis: string | null;
}

export interface ModeTransitionRecord {
  from: PracticeMode;
  to: PracticeMode;
  chunkIndex: number;
  timestampMs: number;
  dwellMs: number;
}

export interface DrillingRecord {
  barRange: [number, number] | null;
  repetitionCount: number;
  firstScores: number[];
  finalScores: number[];
  startedAtChunk: number;
  endedAtChunk: number;
}

export interface TimelineEvent {
  chunkIndex: number;
  timestampMs: number;
  hasAudio: boolean;
}

export class SessionAccumulator {
  teachingMoments: AccumulatedMoment[] = [];
  modeTransitions: ModeTransitionRecord[] = [];
  drillingRecords: DrillingRecord[] = [];
  timeline: TimelineEvent[] = [];

  accumulateMoment(moment: AccumulatedMoment) {
    this.teachingMoments.push(moment);
  }

  accumulateModeTransition(record: ModeTransitionRecord) {
    this.modeTransitions.push(record);
  }

  accumulateDrillingRecord(record: DrillingRecord) {
    this.drillingRecords.push(record);
  }

  accumulateTimelineEvent(event: TimelineEvent) {
    this.timeline.push(event);
  }

  hasTeachingContent(): boolean {
    return this.teachingMoments.length > 0 || this.timeline.length > 0;
  }

  /**
   * Select top teaching moments for synthesis.
   * Per dimension: pick highest |deviation|. Also pick top positive if different.
   * Cap at 8, sort by chunk index (chronological).
   */
  topMoments(dimensionWeights?: Record<string, number>): AccumulatedMoment[] {
    const byDim = new Map<string, AccumulatedMoment[]>();
    for (const m of this.teachingMoments) {
      const list = byDim.get(m.dimension) ?? [];
      list.push(m);
      byDim.set(m.dimension, list);
    }

    const candidates: AccumulatedMoment[] = [];
    for (const dim of DIMS_6) {
      const moments = byDim.get(dim);
      if (!moments || moments.length === 0) continue;

      // Top by |deviation|
      moments.sort((a, b) => Math.abs(b.deviation) - Math.abs(a.deviation));
      candidates.push(moments[0]);

      // Top positive if different
      const topPositive = moments.find((m) => m.isPositive);
      if (topPositive && topPositive !== moments[0]) {
        candidates.push(topPositive);
      }
    }

    // Re-sort by weighted deviation if weights provided
    if (dimensionWeights) {
      candidates.sort((a, b) => {
        const wA = Math.abs(a.deviation) * (dimensionWeights[a.dimension] ?? 1);
        const wB = Math.abs(b.deviation) * (dimensionWeights[b.dimension] ?? 1);
        return wB - wA;
      });
    }

    // Cap at 8, sort chronologically
    return candidates.slice(0, 8).sort((a, b) => a.chunkIndex - b.chunkIndex);
  }

  toJSON() {
    return {
      teachingMoments: this.teachingMoments,
      modeTransitions: this.modeTransitions,
      drillingRecords: this.drillingRecords,
      timeline: this.timeline,
    };
  }

  static fromJSON(data: unknown): SessionAccumulator {
    const acc = new SessionAccumulator();
    const d = data as Record<string, unknown[]>;
    acc.teachingMoments = (d.teachingMoments ?? []) as AccumulatedMoment[];
    acc.modeTransitions = (d.modeTransitions ?? []) as ModeTransitionRecord[];
    acc.drillingRecords = (d.drillingRecords ?? []) as DrillingRecord[];
    acc.timeline = (d.timeline ?? []) as TimelineEvent[];
    return acc;
  }
}
```

- [ ] **Step 2: Create Zod schemas for DO state**

```typescript
// apps/api/src/do/session-brain.schema.ts
import { z } from "zod";

export const sessionStateSchema = z.object({
  version: z.number().int(),
  sessionId: z.string(),
  studentId: z.string(),
  conversationId: z.string().nullable(),
  isEval: z.boolean().default(false),
  chunksInFlight: z.number().int().default(0),
  sessionEnding: z.boolean().default(false),
  synthesisCompleted: z.boolean().default(false),
  finalized: z.boolean().default(false),
  inferenceFailures: z.number().int().default(0),
  accumulator: z.unknown().default({}),
  baselines: z.record(z.string(), z.number()).nullable().default(null),
  baselinesLoaded: z.boolean().default(false),
  scoredChunks: z.array(z.object({
    chunkIndex: z.number().int(),
    scores: z.array(z.number()),
  })).default([]),
  pieceQuery: z.string().nullable().default(null),
  pieceLocked: z.boolean().default(false),
  pieceIdentification: z.object({
    pieceId: z.string(),
    confidence: z.number(),
    method: z.string(),
  }).nullable().default(null),
  followerState: z.object({
    lastKnownBar: z.number().int().nullable(),
  }).default({ lastKnownBar: null }),
  modeDetector: z.unknown().default(null), // Serialized ModeDetector
  identificationNoteCount: z.number().int().default(0),
});

export type SessionState = z.infer<typeof sessionStateSchema>;

/** WebSocket incoming message types */
export const wsChunkReadySchema = z.object({
  type: z.literal("chunk_ready"),
  index: z.number().int(),
  r2Key: z.string(),
});

export const wsEndSessionSchema = z.object({
  type: z.literal("end_session"),
});

export const wsSetPieceSchema = z.object({
  type: z.literal("set_piece"),
  query: z.string(),
});

export const wsIncomingMessageSchema = z.discriminatedUnion("type", [
  wsChunkReadySchema,
  wsEndSessionSchema,
  wsSetPieceSchema,
]);
```

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/services/accumulator.ts apps/api/src/do/session-brain.schema.ts
git commit -m "feat(api): add session accumulator and DO state Zod schemas"
```

---

### Task 4: Inference Client (MuQ + AMT)

HTTP callers for MuQ (quality scoring) and AMT (transcription) inference endpoints with retry logic.

**Files:**
- Modify: `apps/api/src/lib/types.ts` (add inference bindings)
- Create: `apps/api/src/services/inference.ts`

- [ ] **Step 1: Add inference bindings to types.ts**

Add to `Bindings`:
```typescript
MUQ_ENDPOINT: string;         // HF MuQ inference endpoint URL
AMT_ENDPOINT: string;         // AMT service URL
SESSION_BRAIN: DurableObjectNamespace;  // Practice session DO
```

Add to `wrangler.toml`:
```toml
MUQ_ENDPOINT = ""
AMT_ENDPOINT = ""

[[durable_objects.bindings]]
name = "SESSION_BRAIN"
class_name = "SessionBrain"

[[migrations]]
tag = "v1"
new_classes = ["SessionBrain"]
```

- [ ] **Step 2: Write the inference client**

```typescript
// apps/api/src/services/inference.ts
import { InferenceError } from "../lib/errors";
import type { Bindings } from "../lib/types";

const RETRY_DELAYS_MS = [10_000, 20_000, 40_000];
const RETRY_DELAYS_ENDING_MS = [3_000, 5_000];

export interface MuqScores {
  dynamics: number;
  timing: number;
  pedaling: number;
  articulation: number;
  phrasing: number;
  interpretation: number;
}

export interface PerfNote {
  pitch: number;
  onset: number;
  offset: number;
  velocity: number;
}

export interface PerfPedalEvent {
  time: number;
  value: number; // >= 64 = on
}

export interface AmtResult {
  notes: PerfNote[];
  pedalEvents: PerfPedalEvent[];
}

async function fetchWithRetry(
  url: string,
  init: RequestInit,
  retryDelays: number[],
): Promise<Response> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retryDelays.length; attempt++) {
    const res = await fetch(url, init);
    if (res.ok) return res;

    if (res.status === 503 || res.status === 429) {
      if (attempt < retryDelays.length) {
        await new Promise((r) => setTimeout(r, retryDelays[attempt]));
        continue;
      }
    }

    const text = await res.text();
    lastError = new InferenceError(`${url} returned ${res.status}: ${text}`);
    break;
  }

  throw lastError ?? new InferenceError(`${url} failed after retries`);
}

export async function callMuqEndpoint(
  env: Bindings,
  audioBytes: ArrayBuffer,
  sessionEnding = false,
): Promise<MuqScores> {
  const delays = sessionEnding ? RETRY_DELAYS_ENDING_MS : RETRY_DELAYS_MS;
  const res = await fetchWithRetry(
    env.MUQ_ENDPOINT,
    {
      method: "POST",
      headers: { "Content-Type": "audio/webm;codecs=opus" },
      body: audioBytes,
    },
    delays,
  );

  const data = (await res.json()) as Record<string, number>;
  return {
    dynamics: data.dynamics,
    timing: data.timing,
    pedaling: data.pedaling,
    articulation: data.articulation,
    phrasing: data.phrasing,
    interpretation: data.interpretation,
  };
}

export async function callAmtEndpoint(
  env: Bindings,
  chunkAudio: ArrayBuffer,
  contextAudio: ArrayBuffer | null,
  sessionEnding = false,
): Promise<AmtResult> {
  const delays = sessionEnding ? RETRY_DELAYS_ENDING_MS : RETRY_DELAYS_MS;

  // Encode audio as base64 for JSON transport
  const chunkB64 = btoa(
    String.fromCharCode(...new Uint8Array(chunkAudio)),
  );
  const payload: Record<string, unknown> = { chunk_audio: chunkB64 };
  if (contextAudio) {
    payload.context_audio = btoa(
      String.fromCharCode(...new Uint8Array(contextAudio)),
    );
  }

  const res = await fetchWithRetry(
    `${env.AMT_ENDPOINT}/transcribe`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    delays,
  );

  const data = (await res.json()) as {
    notes?: Array<{ pitch: number; onset: number; offset: number; velocity: number }>;
    pedal_events?: Array<{ time: number; value: number }>;
  };

  return {
    notes: data.notes ?? [],
    pedalEvents: data.pedal_events ?? [],
  };
}
```

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/lib/types.ts apps/api/src/services/inference.ts apps/api/wrangler.toml
git commit -m "feat(api): add MuQ + AMT inference client with retry"
```

---

### Task 5: Score Follower (DTW Alignment)

Subsequence DTW for aligning performance notes to score. Produces a BarMap with per-note alignments and onset deviations.

**Files:**
- Create: `apps/api/src/services/score-follower.ts`
- Create: `apps/api/src/services/score-follower.test.ts`

- [ ] **Step 1: Write tests**

Test the DTW alignment on a simple synthetic case: 3 matching notes should align with high confidence.

- [ ] **Step 2: Implement score follower**

Port from `practice/analysis/score_follower.rs`. Key elements:
- `subsequenceDtw(perfSeq, scoreSeq)` — free start, backtrace, pitch penalty (same=0, semitone=0.125, octave=0.25, other=0.5)
- `alignChunk(perfNotes, scoreNotes, followerState, scoreContext)` — restrict search to `[lastBar-5, lastBar+30]`, reanchor if cost > 0.3
- `buildBarMap(path, perfNotes, scoreNotes)` — median offset correction, per-note NoteAlignment with `onsetDeviationMs`
- `NoteAlignment`, `BarMap`, `FollowerState` types
- Constants: `MIN_PERF_NOTES = 3`, `REANCHOR_COST_THRESHOLD = 0.3`, `PITCH_MISMATCH_PENALTY = 0.5`

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/services/score-follower.ts apps/api/src/services/score-follower.test.ts
git commit -m "feat(api): add DTW score follower with bar alignment"
```

---

### Task 6: Piece Identification (N-gram + Rerank + DTW)

Three-stage piece identification: N-gram recall, statistical rerank, DTW confirmation.

**Files:**
- Create: `apps/api/src/services/piece-identify.ts`
- Create: `apps/api/src/services/piece-identify.test.ts`

- [ ] **Step 1: Write tests**

Test N-gram recall with a synthetic index, rerank feature computation, and cosine similarity.

- [ ] **Step 2: Implement piece identification**

Port from `practice/analysis/piece_identify.rs` and `session_piece_id.rs`. Key elements:
- `NgramIndex`: `Map<string, Array<{ pieceId: string; bar: number }>>` — loaded from R2
- `RerankFeatures`: `Map<string, number[]>` — 128-dim per piece, loaded from R2
- `ngramRecall(notes, index)` — extract pitch trigrams, count hits per piece, top 10
- `computeRerankFeatures(notes)` — 128-dim vector (pitch class, interval, IOI, velocity histograms + stats)
- `rerankCandidates(notes, candidates, features)` — cosine similarity, top 2
- `tryIdentifyPiece(notes, index, features, scoreLoader, followerState)` — window sizes [60, 120, 200], DTW confirmation threshold 0.3
- Constants: `MIN_NOTES = 10`, `ID_WINDOW_SIZES = [60, 120, 200]`, `ID_MAX_TOTAL_NOTES = 600`, `ID_MIN_NEW_NOTES = 30`, `DTW_CONFIRM_THRESHOLD = 0.3`

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/services/piece-identify.ts apps/api/src/services/piece-identify.test.ts
git commit -m "feat(api): add N-gram + rerank + DTW piece identification"
```

---

### Task 7: Bar Analysis (Tier 1/2/3)

Per-dimension analysis of a chunk's MuQ scores + AMT MIDI data against score context.

**Files:**
- Create: `apps/api/src/services/bar-analysis.ts`
- Create: `apps/api/src/services/bar-analysis.test.ts`

- [ ] **Step 1: Write tests**

Test Tier 2 analysis (no score context) with synthetic MIDI notes — should produce analysis strings for each dimension.

- [ ] **Step 2: Implement bar analysis**

Port from `practice/analysis/bar_analysis.rs`. Key elements:
- `ChunkAnalysis`: `{ tier: number, barRange: [number, number] | null, dimensions: DimensionAnalysis[] }`
- `DimensionAnalysis`: `{ dimension, analysis, scoreMarking?, referenceComparison? }`
- `analyzeTier1(barMap, scoreContext, scores)` — full bar-aligned with reference comparison:
  - dynamics: perf velocity vs score velocity, crescendo/diminuendo detection
  - timing: mean/std onset deviation, rushing/dragging/close classification
  - pedaling: event count, duration, vs score markings
  - articulation: note duration ratio → legato/staccato/normal
  - phrasing: first-third vs last-third onset shift
  - interpretation: mean absolute onset deviation
- `analyzeTier2(perfNotes, pedalEvents, scores)` — absolute MIDI stats, no reference
- Tier selection: no notes → tier 3 (scores only), score+barMap → tier 1, else → tier 2

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/services/bar-analysis.ts apps/api/src/services/bar-analysis.test.ts
git commit -m "feat(api): add Tier 1/2/3 bar-aligned analysis"
```

---

### Task 8: Synthesis Service

Builds the synthesis prompt from accumulated session data and calls Anthropic for a session summary.

**Files:**
- Create: `apps/api/src/services/synthesis.ts`
- Modify: `apps/api/src/services/prompts.ts` (add SESSION_SYNTHESIS_SYSTEM, SUBAGENT_SYSTEM, TEACHER_SYSTEM)

- [ ] **Step 1: Add synthesis and pipeline prompts to prompts.ts**

Add `SESSION_SYNTHESIS_SYSTEM`, `SUBAGENT_SYSTEM`, `TEACHER_SYSTEM`, `buildSubagentUserPrompt()`, `buildTeacherUserPrompt()`, `buildSynthesisPrompt()`, `exerciseToolDefinition()`.

Port from `services/prompts.rs` — these are the exact prompts used by the Rust API.

- [ ] **Step 2: Write synthesis service**

```typescript
// apps/api/src/services/synthesis.ts
import type { Db, Bindings, ServiceContext } from "../lib/types";
import type { SessionAccumulator } from "./accumulator";
import type { StudentBaselines } from "./teaching-moments";
import { callAnthropic } from "./llm";
import { buildMemoryContext } from "./memory";
import { conversations, messages } from "../db/schema/conversations";
import { observations } from "../db/schema/observations";
import { sessions } from "../db/schema/sessions";
import { eq } from "drizzle-orm";

export interface SynthesisContext {
  sessionId: string;
  studentId: string;
  conversationId: string | null;
  baselines: StudentBaselines | null;
  pieceContext: { composer: string; title: string; pieceId: string } | null;
  studentMemory: string | null;
  totalChunks: number;
  sessionDurationMs: number;
}

export interface SynthesisResult {
  text: string;
  isFallback: boolean;
}
```

Key functions:
- `buildSynthesisPrompt(accumulator, context)` — structured JSON with session_duration, chunks_processed, practice_pattern, top_moments, baselines, piece, drilling_progress
- `callSynthesisLlm(env, promptContext)` — Anthropic call with SESSION_SYNTHESIS_SYSTEM, fallback on error
- `persistSynthesisMessage(db, conversationId, text)` — insert into messages table
- `persistAccumulatedMoments(db, studentId, sessionId, conversationId, moments)` — insert into observations table
- `clearNeedsSynthesis(db, sessionId)` — update sessions.needsSynthesis = false

- [ ] **Step 3: Commit**

```bash
git add apps/api/src/services/synthesis.ts apps/api/src/services/prompts.ts
git commit -m "feat(api): add session synthesis service with prompts"
```

---

### Task 9: Teaching Pipeline (Ask)

Two-stage LLM pipeline: Groq subagent analyzes → Anthropic teacher generates observation with optional exercise tool_use.

**Files:**
- Create: `apps/api/src/services/ask.ts`

- [ ] **Step 1: Implement the two-stage pipeline**

Port from `services/ask.rs`. Key functions:
- `handleAskInner(env, ctx, request)` — the core pipeline (reused by DO and HTTP handler):
  1. Build memory context
  2. Stage 1: call Groq (subagent) with SUBAGENT_SYSTEM + structured user prompt
  3. Parse subagent JSON output (selected_moment, framing, learning_arc)
  4. Look up catalog exercises from DB
  5. Stage 2: call Anthropic (teacher) with TEACHER_SYSTEM + tool definition + teacher user prompt
  6. Post-process: strip markdown, truncate at 500 chars
  7. Process exercise tool calls if any (insert generated exercises to DB)
  8. Return `{ observationText, dimension, framing, reasoningTrace, isFallback, componentsJson }`

- `processExerciseToolCall(db, toolInput, dimension)` — validate dimensions, insert exercise + exercise_dimensions
- `splitSubagentOutput(text)` — extract JSON block + narrative
- `postProcessObservation(text)` — strip markdown bold/quotes, truncate at 500 chars on last period

- [ ] **Step 2: Commit**

```bash
git add apps/api/src/services/ask.ts
git commit -m "feat(api): add two-stage teaching pipeline (subagent + teacher)"
```

---

### Task 10: Practice Session Durable Object

The core orchestrator. Handles WebSocket lifecycle, dispatches parallel inference, runs the 10-step per-chunk pipeline, and triggers synthesis via alarm.

**Files:**
- Create: `apps/api/src/do/session-brain.ts`

- [ ] **Step 1: Implement the SessionBrain DO**

```typescript
// apps/api/src/do/session-brain.ts
import { DurableObject } from "cloudflare:workers";
import type { Bindings } from "../lib/types";
import { sessionStateSchema, type SessionState } from "./session-brain.schema";
// ... imports for all services
```

Key methods:
- `fetch(request)` — WebSocket upgrade:
  1. Parse session_id, student_id, conversation_id from URL
  2. Persist identity to `this.ctx.storage`
  3. `this.ctx.acceptWebSocket(server)` (Hibernation API)
  4. Set 30-minute alarm
  5. Send `{"type":"connected","sessionId":...}`

- `webSocketMessage(ws, message)` — dispatch by type:
  - `chunk_ready`: increment chunksInFlight, call `handleChunkReady()`, decrement, check if last chunk while ending
  - `end_session`: set sessionEnding, schedule 1ms alarm if no in-flight chunks
  - `set_piece`: set pieceQuery, reset score context, lock piece

- `webSocketClose(ws)` — set sessionEnding, schedule 1ms alarm

- `alarm()` — convergence point for all exit paths:
  1. Reload state from storage (Zod validate)
  2. `runSynthesisAndPersist()` (idempotent via synthesisCompleted flag)
  3. `finalizeSession()` (idempotent via finalized flag)

- `handleChunkReady(ws, index, r2Key)` — the 10-step pipeline:
  1. Fetch audio from R2
  2. Parallel MuQ + AMT dispatch (`Promise.all`)
  3. Process MuQ scores → send `chunk_processed` WS event
  4. Process AMT → align_chunk → bar analysis
  5. Try piece identification
  6. Load baselines (one-time)
  7. Mode detector update → send `mode_change` events
  8. Every 2 chunks: select teaching moment → accumulate → send `observation`
  9. Persist state to storage
  10. Reset alarm

**CRITICAL: Clone-before-await pattern** for all async operations:
```typescript
const stateVersion = state.version;
const [muqResult, amtResult] = await Promise.all([...]);
// Re-read state, check version hasn't changed
const currentState = sessionStateSchema.parse(await this.ctx.storage.get("state"));
if (currentState.version !== stateVersion) {
  // Conflict — another event mutated state during our await
  return;
}
currentState.version++;
await this.ctx.storage.put("state", currentState);
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/src/do/session-brain.ts
git commit -m "feat(api): add SessionBrain Durable Object with WebSocket hibernation"
```

---

### Task 11: Practice HTTP Handlers + Route Mounting

HTTP routes for session start, chunk upload, WebSocket upgrade to DO, and deferred synthesis.

**Files:**
- Create: `apps/api/src/routes/practice.ts`
- Create: `apps/api/src/routes/practice.test.ts`
- Modify: `apps/api/src/index.ts` (mount routes, export DO)
- Modify: `apps/api/wrangler.toml` (DO binding)

- [ ] **Step 1: Write practice routes**

```typescript
// apps/api/src/routes/practice.ts
import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
```

Four routes:
- `POST /start` — create conversation + session in DB, return `{ sessionId, conversationId }`
- `POST /chunk?sessionId=X&chunkIndex=N` — upload raw bytes to R2, return `{ r2Key, sessionId, chunkIndex }`
- `GET /ws/:sessionId` — WebSocket upgrade, forward to DO stub
- `GET /needs-synthesis?conversationId=X` — find sessions needing deferred synthesis
- `POST /synthesize` — run deferred synthesis for a session

WebSocket upgrade pattern (from TS_STYLE.md):
```typescript
.get("/ws/:sessionId", async (c) => {
  if (c.req.header("Upgrade") !== "websocket") {
    return c.text("Expected WebSocket upgrade", 426);
  }
  requireAuth(c.var.studentId);
  const sessionId = c.req.param("sessionId");
  const id = c.env.SESSION_BRAIN.idFromName(sessionId);
  const stub = c.env.SESSION_BRAIN.get(id);
  // Forward with student_id and conversation_id as query params
  const url = new URL(c.req.url);
  url.searchParams.set("student_id", c.var.studentId);
  const doReq = new Request(url.toString(), c.req.raw);
  return stub.fetch(doReq);
})
```

- [ ] **Step 2: Write tests**

```typescript
// apps/api/src/routes/practice.test.ts
describe("practice routes", () => {
  it("POST /api/practice/start returns 401 without auth", ...);
  it("POST /api/practice/chunk returns 401 without auth", ...);
  it("GET /api/practice/ws/:id returns 426 without upgrade header", ...);
});
```

- [ ] **Step 3: Mount routes and export DO in index.ts**

```typescript
import { practiceRoutes } from "./routes/practice";
export { SessionBrain } from "./do/session-brain";

const routes = app
  .route("/health", healthRoutes)
  // ... existing routes ...
  .route("/api/practice", practiceRoutes);
```

- [ ] **Step 4: Run all tests and type check**

```bash
cd apps/api && bun run test -- --run && bun run typecheck
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/routes/practice.ts apps/api/src/routes/practice.test.ts apps/api/src/index.ts apps/api/wrangler.toml
git commit -m "feat(api): add practice routes with WS upgrade, chunk upload, deferred synthesis"
```

---

## Validation Gate

After all tasks complete:

1. **All tests pass:** `cd apps/api && bun run test -- --run` (including STOP, mode detector, score follower, piece ID, bar analysis, practice routes)
2. **Type check passes:** `cd apps/api && bun run typecheck`
3. **wrangler dev smoke test:**
   - Health + all Phase 2 endpoints still work
   - `POST /api/practice/start` returns sessionId + conversationId
   - `POST /api/practice/chunk` uploads to R2
   - WebSocket upgrade reaches DO
4. **State machine transitions match Rust behavior:** Unit tests cover all 5 modes and key transitions
5. **STOP classifier matches Rust output:** Unit tests verify probability calculations against known inputs
6. **DTW score follower aligns correctly:** Unit tests with synthetic note sequences
