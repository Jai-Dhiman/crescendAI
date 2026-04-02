# Phase 3: Practice Pipeline — Real-Time Analysis + Durable Objects

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the real-time practice pipeline on the new Hono stack — the Durable Object session brain, WebSocket communication, parallel inference dispatch, and session synthesis. Compute-heavy algorithms (DTW score following, piece identification, STOP classifier, bar analysis) are extracted from the existing Rust codebase as standalone WASM modules.

**Architecture:** A single `SessionBrain` Durable Object manages each practice session's lifecycle. It receives chunk-ready notifications via WebSocket, dispatches parallel MuQ + AMT inference, calls into WASM modules for score following and piece identification, accumulates teaching moments, and triggers synthesis via alarm. The TypeScript layer handles orchestration, I/O, state management, and LLM calls. The Rust WASM layer handles all numerical computation.

**Tech Stack:** Hono, CF Durable Objects (Hibernation API), Drizzle ORM, Zod, R2 buckets, AI Gateway, Rust WASM (`wasm-pack --target bundler`, `serde-wasm-bindgen`)

**Style Guide:** ALL TypeScript code must follow `apps/api/TS_STYLE.md`. Critical DO rules: `this.ctx.acceptWebSocket()` (not `server.accept()`), alarms (not setTimeout), clone-before-await + compare-and-swap for state versioning, Zod validation on every storage read.

**Decision: WASM for compute, TS for orchestration.** The existing Rust algorithms are proven, tested, and performant. Rewriting them in TypeScript would introduce bugs for zero gain. We extract them as standalone WASM crates compiled via `wasm-pack --target bundler` to `wasm32-unknown-unknown`. Data crosses the boundary via `serde-wasm-bindgen` (Rust structs <-> JS objects). The DO and all I/O/LLM orchestration stays in TypeScript. WASM modules count toward the 10MB Worker bundle limit but these algorithms are small (~100-200KB compiled).

**WASM crate structure:** Two crates under `apps/api/src/wasm/`:
- `score-analysis/` — STOP classifier, bar analysis, score follower (DTW), teaching moment selection
- `piece-identify/` — N-gram recall, rerank features, DTW confirmation

Each crate exposes a clean JS API via `#[wasm_bindgen]` with `serde-wasm-bindgen` for complex types.

---

## File Structure

```
apps/api/src/
  wasm/
    score-analysis/            -- RUST CRATE: STOP, DTW, bar analysis, teaching moments
      Cargo.toml
      src/
        lib.rs                 -- wasm_bindgen entry points
        stop.rs                -- extracted from api-rust/src/services/stop.rs
        score_follower.rs      -- extracted from api-rust/src/practice/analysis/score_follower.rs
        bar_analysis.rs        -- extracted from api-rust/src/practice/analysis/bar_analysis.rs
        teaching_moments.rs    -- extracted from api-rust/src/services/teaching_moments.rs
        dims.rs                -- extracted from api-rust/src/practice/dims.rs
        types.rs               -- shared types (ScoredChunk, StudentBaselines, etc.)
      pkg/                     -- wasm-pack output (git-tracked)
    piece-identify/            -- RUST CRATE: N-gram recall, rerank, DTW confirm
      Cargo.toml
      src/
        lib.rs                 -- wasm_bindgen entry points
        ngram.rs               -- extracted from api-rust/src/practice/analysis/piece_identify.rs
        rerank.rs              -- feature computation + cosine similarity
        dtw_confirm.rs         -- DTW confirmation stage
        types.rs               -- NgramIndex, RerankFeatures, PieceIdentification
      pkg/                     -- wasm-pack output (git-tracked)
  services/
    practice-mode.ts           -- CREATE: ModeDetector state machine (TS — pure state logic, no math)
    accumulator.ts             -- CREATE: SessionAccumulator + types (TS — data structures)
    inference.ts               -- CREATE: MuQ + AMT endpoint callers with retry
    synthesis.ts               -- CREATE: synthesis prompt + LLM call + persistence
    ask.ts                     -- CREATE: two-stage teaching pipeline (subagent + teacher)
    prompts.ts                 -- MODIFY: add subagent, teacher, synthesis prompts
    wasm-bridge.ts             -- CREATE: TS wrappers that import + call WASM modules
  do/
    session-brain.ts           -- CREATE: Durable Object class
    session-brain.schema.ts    -- CREATE: Zod schemas for DO state
  routes/
    practice.ts                -- CREATE: start, chunk upload, WS upgrade, deferred synthesis
    practice.test.ts           -- CREATE: route tests
  lib/
    types.ts                   -- MODIFY: add DO namespace, AMT binding to Bindings
    dims.ts                    -- CREATE: dimension constants and mapping (mirrors Rust dims)
  index.ts                     -- MODIFY: mount practice routes, export DO
wrangler.toml                  -- MODIFY: add DO binding
```

---

### Task 1: WASM Crate — score-analysis

Extract STOP classifier, score follower (DTW), bar analysis, and teaching moment selection from the existing Rust codebase into a standalone WASM crate. These algorithms are proven and tested — we extract, not rewrite.

**Source files (copy from `apps/api-rust/src/`):**
- `services/stop.rs` → `wasm/score-analysis/src/stop.rs`
- `practice/analysis/score_follower.rs` → `wasm/score-analysis/src/score_follower.rs`
- `practice/analysis/bar_analysis.rs` → `wasm/score-analysis/src/bar_analysis.rs`
- `services/teaching_moments.rs` → `wasm/score-analysis/src/teaching_moments.rs`
- `practice/dims.rs` → `wasm/score-analysis/src/dims.rs`

**Files:**
- Create: `apps/api/src/wasm/score-analysis/Cargo.toml`
- Create: `apps/api/src/wasm/score-analysis/src/lib.rs` (wasm_bindgen entry points)
- Create: `apps/api/src/wasm/score-analysis/src/*.rs` (extracted algorithm files)
- Create: `apps/api/src/lib/dims.ts` (TS mirror of dimension constants)

- [ ] **Step 1: Create Cargo.toml for the WASM crate**

```toml
# apps/api/src/wasm/score-analysis/Cargo.toml
[package]
name = "score-analysis"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"
serde_json = "1"

[profile.release]
opt-level = "s"     # optimize for size (10MB Worker bundle limit)
lto = true
```

- [ ] **Step 2: Extract Rust source files**

Copy the algorithm files from `apps/api-rust/src/` into the new crate. Strip all D1/CF Worker dependencies — these modules must be pure computation with no I/O. Remove any `worker::*` imports, `Env` parameters, or database calls. The functions should take typed inputs and return typed outputs.

Key adaptations:
- Remove `use worker::*` and any CF-specific imports
- Replace `JsValue` returns with `serde-wasm-bindgen` serialization
- Remove any `console_log!` / `console_error!` (use return values for errors)
- Keep all constants, algorithms, and type definitions intact

- [ ] **Step 3: Create wasm_bindgen entry points in lib.rs**

```rust
// apps/api/src/wasm/score-analysis/src/lib.rs
use wasm_bindgen::prelude::*;
use serde_wasm_bindgen::{from_value, to_value};

mod stop;
mod score_follower;
mod bar_analysis;
mod teaching_moments;
mod dims;
mod types;

#[wasm_bindgen]
pub fn classify_stop(scores: Vec<f64>, threshold: f64) -> JsValue {
    let result = stop::classify(&scores, threshold);
    to_value(&result).unwrap()
}

#[wasm_bindgen]
pub fn select_teaching_moment(
    chunks_js: JsValue,
    baselines_js: JsValue,
    recent_observations_js: JsValue,
) -> JsValue {
    let chunks = from_value(chunks_js).unwrap();
    let baselines = from_value(baselines_js).unwrap();
    let recent = from_value(recent_observations_js).unwrap();
    let result = teaching_moments::select_teaching_moment(&chunks, &baselines, &recent);
    to_value(&result).unwrap()
}

#[wasm_bindgen]
pub fn align_chunk(
    perf_notes_js: JsValue,
    score_notes_js: JsValue,
    follower_state_js: JsValue,
) -> JsValue {
    let perf_notes = from_value(perf_notes_js).unwrap();
    let score_notes = from_value(score_notes_js).unwrap();
    let state = from_value(follower_state_js).unwrap();
    let result = score_follower::align_chunk(&perf_notes, &score_notes, &state);
    to_value(&result).unwrap()
}

#[wasm_bindgen]
pub fn analyze_chunk(
    bar_map_js: JsValue,
    score_context_js: JsValue,
    perf_notes_js: JsValue,
    pedal_events_js: JsValue,
    scores: Vec<f64>,
) -> JsValue {
    // Dispatches to tier1 or tier2 based on available data
    let bar_map = from_value(bar_map_js).ok();
    let score_ctx = from_value(score_context_js).ok();
    let perf_notes = from_value(perf_notes_js).unwrap();
    let pedal = from_value(pedal_events_js).unwrap();
    let result = bar_analysis::analyze(&bar_map, &score_ctx, &perf_notes, &pedal, &scores);
    to_value(&result).unwrap()
}
```

- [ ] **Step 4: Build WASM with wasm-pack**

```bash
cd apps/api/src/wasm/score-analysis && wasm-pack build --target bundler --out-dir pkg
```

Verify `pkg/` contains: `score_analysis_bg.wasm`, `score_analysis.js`, `score_analysis.d.ts`

- [ ] **Step 5: Create TS dimension constants (mirrors Rust dims)**

```typescript
// apps/api/src/lib/dims.ts
export const DIMS_6 = [
  "dynamics", "timing", "pedaling",
  "articulation", "phrasing", "interpretation",
] as const;

export type Dimension = (typeof DIMS_6)[number];

export const DIM_INDEX: Record<Dimension, number> = {
  dynamics: 0, timing: 1, pedaling: 2,
  articulation: 3, phrasing: 4, interpretation: 5,
};
```

- [ ] **Step 6: Write WASM integration tests**

```typescript
// apps/api/src/services/score-analysis.test.ts
import { describe, it, expect } from "vitest";
import { classify_stop } from "../wasm/score-analysis/pkg/score_analysis";

describe("WASM score-analysis", () => {
  it("STOP classifier returns probability between 0 and 1", () => {
    const result = classify_stop([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 0.5);
    expect(result.probability).toBeGreaterThanOrEqual(0);
    expect(result.probability).toBeLessThanOrEqual(1);
  });

  it("STOP triggers on low dynamics + pedaling", () => {
    const result = classify_stop([0.3, 0.55, 0.2, 0.54, 0.52, 0.35], 0.5);
    expect(result.triggered).toBe(true);
  });
});
```

- [ ] **Step 7: Commit**

```bash
git add apps/api/src/wasm/score-analysis/ apps/api/src/lib/dims.ts apps/api/src/services/score-analysis.test.ts
git commit -m "feat(api): extract score-analysis WASM crate (STOP, DTW, bar analysis)"
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

### Task 5: WASM Crate — piece-identify

Extract piece identification algorithms (N-gram recall, rerank features, DTW confirmation) from the existing Rust codebase into a standalone WASM crate.

**Source files (copy from `apps/api-rust/src/practice/analysis/`):**
- `piece_identify.rs` → `wasm/piece-identify/src/ngram.rs` + `rerank.rs`
- `session_piece_id.rs` → `wasm/piece-identify/src/dtw_confirm.rs`
- `piece_match.rs` → `wasm/piece-identify/src/text_match.rs`

**Files:**
- Create: `apps/api/src/wasm/piece-identify/Cargo.toml`
- Create: `apps/api/src/wasm/piece-identify/src/lib.rs` (wasm_bindgen entry points)
- Create: `apps/api/src/wasm/piece-identify/src/*.rs` (extracted algorithm files)

- [ ] **Step 1: Create Cargo.toml**

Same structure as score-analysis crate. Also depends on the score-analysis crate for DTW functions used in DTW confirmation stage.

```toml
[package]
name = "piece-identify"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"
serde_json = "1"

[profile.release]
opt-level = "s"
lto = true
```

- [ ] **Step 2: Extract and adapt Rust source files**

Strip all D1/CF/R2 dependencies. The N-gram index and rerank features are passed in as arguments (loaded by the TS layer from R2). Key adaptations:
- `ngram_recall(notes, index)` — takes pre-loaded NgramIndex, returns top-10 candidates
- `compute_rerank_features(notes)` — returns 128-dim feature vector
- `rerank_candidates(notes, candidates, features)` — cosine similarity, top 2
- `dtw_confirm(perf_notes, score_notes, threshold)` — runs DTW, returns confidence + cost
- `match_piece_text(query, catalog)` — Dice similarity text matching

- [ ] **Step 3: Create wasm_bindgen entry points**

```rust
#[wasm_bindgen]
pub fn ngram_recall(notes_js: JsValue, index_js: JsValue) -> JsValue { ... }

#[wasm_bindgen]
pub fn compute_rerank_features(notes_js: JsValue) -> Vec<f64> { ... }

#[wasm_bindgen]
pub fn rerank_candidates(notes_js: JsValue, candidates_js: JsValue, features_js: JsValue) -> JsValue { ... }

#[wasm_bindgen]
pub fn dtw_confirm(perf_notes_js: JsValue, score_notes_js: JsValue, threshold: f64) -> JsValue { ... }
```

- [ ] **Step 4: Build WASM**

```bash
cd apps/api/src/wasm/piece-identify && wasm-pack build --target bundler --out-dir pkg
```

- [ ] **Step 5: Write WASM integration tests**

Test N-gram recall with a synthetic index, rerank feature vector dimensions, and DTW confirmation threshold.

- [ ] **Step 6: Commit**

```bash
git add apps/api/src/wasm/piece-identify/
git commit -m "feat(api): extract piece-identify WASM crate (N-gram, rerank, DTW confirm)"
```

---

### Task 6: WASM Bridge — TypeScript Wrappers

TypeScript module that imports both WASM packages and provides typed async wrappers for the DO to call. Handles WASM module initialization and provides clean interfaces that hide the `JsValue` serialization boundary.

**Files:**
- Create: `apps/api/src/services/wasm-bridge.ts`

- [ ] **Step 1: Create the WASM bridge**

```typescript
// apps/api/src/services/wasm-bridge.ts
import * as scoreAnalysis from "../wasm/score-analysis/pkg/score_analysis";
import * as pieceIdentify from "../wasm/piece-identify/pkg/piece_identify";
import type { Dimension } from "../lib/dims";

// Re-export clean typed interfaces for the DO to call

export interface StopResult {
  probability: number;
  triggered: boolean;
  topDimension: Dimension;
  topDeviation: number;
}

export function classifyStop(scores: number[], threshold = 0.5): StopResult {
  return scoreAnalysis.classify_stop(scores, threshold);
}

export function selectTeachingMoment(chunks, baselines, recentObs) {
  return scoreAnalysis.select_teaching_moment(chunks, baselines, recentObs);
}

export function alignChunk(perfNotes, scoreNotes, followerState) {
  return scoreAnalysis.align_chunk(perfNotes, scoreNotes, followerState);
}

export function analyzeChunk(barMap, scoreContext, perfNotes, pedalEvents, scores) {
  return scoreAnalysis.analyze_chunk(barMap, scoreContext, perfNotes, pedalEvents, scores);
}

export function ngramRecall(notes, index) {
  return pieceIdentify.ngram_recall(notes, index);
}

export function computeRerankFeatures(notes): number[] {
  return pieceIdentify.compute_rerank_features(notes);
}

export function rerankCandidates(notes, candidates, features) {
  return pieceIdentify.rerank_candidates(notes, candidates, features);
}

export function dtwConfirm(perfNotes, scoreNotes, threshold = 0.3) {
  return pieceIdentify.dtw_confirm(perfNotes, scoreNotes, threshold);
}
```

Full type signatures will match the Rust `serde` output types. The bridge is the ONLY file that imports from `../wasm/*/pkg/` — all other TS code imports from the bridge.

- [ ] **Step 2: Commit**

```bash
git add apps/api/src/services/wasm-bridge.ts
git commit -m "feat(api): add WASM bridge with typed wrappers for score analysis + piece ID"
```

---

### Task 7: Synthesis Service

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

### Task 8: Teaching Pipeline (Ask)

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

### Task 9: Practice Session Durable Object

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

### Task 10: Practice HTTP Handlers + Route Mounting

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

1. **WASM crates build cleanly:** `wasm-pack build --target bundler` succeeds for both crates
2. **WASM integration tests pass:** STOP classifier, DTW alignment, and piece ID produce correct outputs from TS test harness
3. **All TS tests pass:** `cd apps/api && bun run test -- --run` (mode detector, accumulator, practice routes)
4. **Type check passes:** `cd apps/api && bun run typecheck`
5. **wrangler dev smoke test:**
   - Health + all Phase 2 endpoints still work
   - `POST /api/practice/start` returns sessionId + conversationId
   - `POST /api/practice/chunk` uploads to R2
   - WebSocket upgrade reaches DO
6. **State machine transitions match Rust behavior:** Unit tests cover all 5 modes and key transitions
7. **WASM regression:** Run known inputs through extracted WASM modules and compare outputs against the original Rust monolith (same constants, same results)
8. **Bundle size check:** Combined WASM modules < 1MB (leaving headroom in 10MB Worker limit)
