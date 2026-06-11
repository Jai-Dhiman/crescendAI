# Issue #45 — Own-Passage Loop Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** `own_passage_loop` exercise cards render score as hero with looped playback (cursor + metronome + synthesized piano) at teacher-prescribed tempo.
**Spec:** docs/specs/2026-06-11-issue-45-own-passage-loop-design.md
**Style:** Follow apps/api/TS_STYLE.md for any API-side edits. Explicit exceptions, no silent fallbacks. No emojis. `bun run test` to run tests (NOT `bun test`).

---

## Task Groups

Group A (parallel): Task 1, Task 2, Task 3
Group B (parallel, depends on A): Task 4, Task 5
Group C (sequential, depends on B): Task 6
Group D (sequential, depends on C): Task 7, Task 8

---

### Task 1: Add `tempoFactor` to `scoreClip` type + API construction sites

**Group:** A (parallel with Task 2, Task 3)

**Behavior being verified:** `ExerciseSetConfig.scoreClip` carries `tempoFactor` when the API returns an `own_passage_loop` prescription.
**Interface under test:** `ExerciseSetConfig` type shape + `exercises.ts` / `tool-processor.ts` construction logic (verified via existing API service tests that already import these modules).

**Files:**
- Modify: `apps/web/src/lib/types.ts`
- Modify: `apps/api/src/services/exercises.ts`
- Modify: `apps/api/src/services/tool-processor.ts`

- [ ] **Step 1: Write the failing test**

Read `apps/api/src/services/tool-processor.test.ts` to understand the existing mock context pattern:

```bash
cat apps/api/src/services/tool-processor.test.ts
```

The existing file tests `prescribe_exercise` with `own_passage_loop`. Add one new case to its existing `describe("prescribe_exercise")` block that asserts `scoreClip.tempoFactor` is forwarded. Use the same `mockCtx` and `processToolCall` pattern already present in that file:

```typescript
// Add inside the existing describe("prescribe_exercise") block
// in apps/api/src/services/tool-processor.test.ts:
it("own_passage_loop scoreClip carries tempoFactor from input", async () => {
  const result = await processToolCall(mockCtx, "student-1", "prescribe_exercise", {
    kind: "own_passage_loop",
    target_dimension: "dynamics",
    bar_range: [5, 8],
    tempo_factor: 0.75,
    piece_id: "chopin.ballades.1",
  });
  expect(result).toHaveLength(1);
  const config = (result[0] as {
    type: string;
    config: { scoreClip?: { tempoFactor?: number } };
  }).config;
  expect(config.scoreClip?.tempoFactor).toBe(0.75);
});
```

**Important:** Do not copy-paste this blindly. Read the existing test file first and use its exact mock context variable name, import path for `processToolCall`, and describe block structure.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test src/services/tool-processor.test.ts
```
Expected: FAIL — `tempoFactor` is `undefined` because `scoreClip` does not yet carry it.

- [ ] **Step 3: Implement**

In `apps/web/src/lib/types.ts`, change line 41:
```typescript
// Before:
scoreClip?: { pieceId: string; bars: [number, number] };
// After:
scoreClip?: { pieceId: string; bars: [number, number]; tempoFactor?: number };
```

In `apps/api/src/services/exercises.ts`, at the `own_passage_loop` scoreClip construction (~line 222–227), change:
```typescript
// Before:
const scoreClip =
  pieceId !== null
    ? { pieceId, bars: routing.bar_range as [number, number] }
    : undefined;
// After:
const scoreClip =
  pieceId !== null
    ? { pieceId, bars: routing.bar_range as [number, number], tempoFactor: routing.tempo_factor }
    : undefined;
```

In `apps/api/src/services/tool-processor.ts`, at the `own_passage_loop` scoreClip construction (~line 92–95), change:
```typescript
// Before:
const scoreClip =
  input.piece_id !== null
    ? { pieceId: input.piece_id, bars: input.bar_range as [number, number] }
    : undefined;
// After:
const scoreClip =
  input.piece_id !== null
    ? { pieceId: input.piece_id, bars: input.bar_range as [number, number], tempoFactor: input.tempo_factor }
    : undefined;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test src/services/tool-processor.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/types.ts apps/api/src/services/exercises.ts apps/api/src/services/tool-processor.ts apps/api/src/services/tool-processor.test.ts && git commit -m "feat(#45): add tempoFactor to scoreClip type and API construction sites"
```

---

### Task 2: `get_clip_playback` — worker message + renderer method

**Group:** A (parallel with Task 1, Task 3)

**Behavior being verified:** `ScoreRenderer.getClipPlayback(pieceId, startBar, endBar)` resolves to `{ svg, ir, notes }` where `ir.bars` contains only bars in the requested range and `notes` has entries with numeric `midi`, `startQ`, `endQ`.
**Interface under test:** `processGetClipPlaybackRequest(tk, measures, startBar, endBar)` exported from `score-worker.ts` + `ScoreRenderer.getClipPlayback`.

**Files:**
- Modify: `apps/web/src/lib/score-worker.ts`
- Modify: `apps/web/src/lib/score-renderer.ts`
- Modify: `apps/web/src/lib/score-worker.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `apps/web/src/lib/score-worker.test.ts` inside a new `describe("processGetClipPlaybackRequest")` block:

```typescript
describe("processGetClipPlaybackRequest", () => {
  it("returns svg, clip-scoped ir, and notes array from timemap + MIDI values", async () => {
    const mockSelect = vi.fn();
    const mockRedoLayout = vi.fn();
    const mockGetVersion = vi.fn().mockReturnValue("4.0.0");
    const mockGetPageCount = vi.fn().mockReturnValue(1);
    // Timemap: two notes at qstamps 0 and 2, plus a measureOn
    const mockTimemap = [
      { qstamp: 0, measureOn: "measure-id-1", on: ["note-a"] },
      { qstamp: 2, on: ["note-b"] },
      { qstamp: 4, measureOn: "measure-id-2", on: [] },
    ];
    // Each note has a MIDI value
    const mockGetMIDIValuesForElement = vi.fn((id: string) =>
      id === "note-a" ? [{ pitch: 60 }] : [{ pitch: 64 }]
    );
    const tk = {
      ...fakeTk,
      select: mockSelect,
      redoLayout: mockRedoLayout,
      getVersion: mockGetVersion,
      getPageCount: mockGetPageCount,
      renderToTimemap: vi.fn().mockReturnValue(mockTimemap),
      getMIDIValuesForElement: mockGetMIDIValuesForElement,
    };

    const { processGetClipPlaybackRequest } = await import("./score-worker");
    // biome-ignore lint/suspicious/noExplicitAny: test mock
    const result = processGetClipPlaybackRequest(tk as any, fakeMeasures, 1, 2);

    expect(result).not.toBe("failed");
    if (result === "failed") return;

    expect(result.svg).toBeTypeOf("string");
    expect(result.ir.bars.length).toBeGreaterThan(0);
    expect(result.notes).toBeInstanceOf(Array);
    if (result.notes.length > 0) {
      const n = result.notes[0];
      expect(n).toHaveProperty("midi");
      expect(n).toHaveProperty("startQ");
      expect(n).toHaveProperty("endQ");
      expect(typeof n.midi).toBe("number");
      expect(typeof n.startQ).toBe("number");
      expect(typeof n.endQ).toBe("number");
    }
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/lib/score-worker.test.ts
```
Expected: FAIL — `processGetClipPlaybackRequest is not a function`.

- [ ] **Step 3: Implement**

In `apps/web/src/lib/score-worker.ts`, add the following after `processRenderClipRequest`:

```typescript
export interface ClipNote {
  midi: number;
  startQ: number;
  endQ: number;
}

export interface ClipPlaybackResult {
  svg: string;
  ir: import("./score-ir").ScoreIR;
  notes: ClipNote[];
}

export function processGetClipPlaybackRequest(
  tk: VerovioTk,
  measures: MeasureEntry[],
  startBar: number,
  endBar: number,
): ClipPlaybackResult | "failed" {
  const startEntry = measures[startBar - 1];
  if (!startEntry) return "failed";

  // Set clip render options and crop to the requested bar range.
  tk.setOptions(CLIP_RENDER_OPTS);
  tk.select({ measureRange: `${startBar}-${endBar}` });
  tk.redoLayout({});
  const svg = tk.renderToSVG(1) as string;

  // Build noteQstampMap from the clip-scoped timemap.
  const timemap: Array<{ qstamp: number; on?: string[]; off?: string[]; measureOn?: string }> =
    tk.renderToTimemap({ includeMeasures: true });

  const noteQstampMap = new Map<string, number>();
  const noteOffMap = new Map<string, number>();
  for (const entry of timemap) {
    if (Array.isArray(entry.on)) {
      for (const id of entry.on) {
        noteQstampMap.set(id, entry.qstamp);
      }
    }
    if (Array.isArray(entry.off)) {
      for (const id of entry.off) {
        noteOffMap.set(id, entry.qstamp);
      }
    }
  }

  // Build a clip-scoped measure index from the clip timemap.
  const clipMeasures: MeasureEntry[] = [];
  const seenQstamps = new Set<number>();
  for (const entry of timemap) {
    if (entry.measureOn !== undefined && !seenQstamps.has(entry.qstamp)) {
      seenQstamps.add(entry.qstamp);
      clipMeasures.push({ qstamp: entry.qstamp, measureOn: entry.measureOn });
    }
  }
  clipMeasures.sort((a, b) => a.qstamp - b.qstamp);

  const { parseScoreIR } = await import("./score-ir") as unknown as typeof import("./score-ir");
  // NOTE: score-worker runs in a Worker where top-level await is supported.
  // However, processGetClipPlaybackRequest must be synchronous for the dispatch
  // table. Use a synchronous require-style dynamic import pattern:
  // Since we're in a Worker module context, we can use the already-imported
  // parseScoreIR. To keep this synchronous, import parseScoreIR at the top of
  // the file via a lazy module-level variable.

  // Restore full-piece layout before returning.
  tk.select({});
  tk.setOptions(VEROVIO_OPTS);
  tk.redoLayout({});

  // parseScoreIR is called synchronously using the module-level lazy import.
  const ir = _parseScoreIR(
    "",
    [svg],
    clipMeasures,
    noteQstampMap,
    tk.getVersion() as string,
    CLIP_RENDER_OPTS.pageWidth,
  );

  // Extract notes with MIDI pitch and qstamp range.
  const notes: ClipNote[] = [];
  for (const [id, startQ] of noteQstampMap) {
    const midiValues = tk.getMIDIValuesForElement(id) as Array<{ pitch: number }> | null;
    if (!midiValues || midiValues.length === 0) continue;
    const midi = midiValues[0].pitch;
    const endQ = noteOffMap.get(id) ?? startQ + 1;
    notes.push({ midi, startQ, endQ });
  }
  notes.sort((a, b) => a.startQ - b.startQ);

  return { svg, ir, notes };
}
```

**Note:** The `async import("./score-ir")` inside a sync function is not valid. Instead, add a module-level variable for `parseScoreIR` using the pattern already used in `loadPiece`:

At the top of `score-worker.ts`, add a module-level lazy variable:

```typescript
// Module-level lazy reference to parseScoreIR, set on first use.
// This avoids a circular async import inside synchronous processGetClipPlaybackRequest.
let _parseScoreIR: typeof import("./score-ir").parseScoreIR | null = null;

async function ensureParseScoreIR(): Promise<typeof import("./score-ir").parseScoreIR> {
  if (!_parseScoreIR) {
    const mod = await import("./score-ir");
    _parseScoreIR = mod.parseScoreIR;
  }
  return _parseScoreIR;
}
```

Then make `processGetClipPlaybackRequest` async and call `await ensureParseScoreIR()`:

```typescript
export async function processGetClipPlaybackRequest(
  tk: VerovioTk,
  measures: MeasureEntry[],
  startBar: number,
  endBar: number,
): Promise<ClipPlaybackResult | "failed"> {
  const startEntry = measures[startBar - 1];
  if (!startEntry) return "failed";

  tk.setOptions(CLIP_RENDER_OPTS);
  tk.select({ measureRange: `${startBar}-${endBar}` });
  tk.redoLayout({});
  const svg = tk.renderToSVG(1) as string;

  const timemap: Array<{ qstamp: number; on?: string[]; off?: string[]; measureOn?: string }> =
    tk.renderToTimemap({ includeMeasures: true });

  const noteQstampMap = new Map<string, number>();
  const noteOffMap = new Map<string, number>();
  for (const entry of timemap) {
    if (Array.isArray(entry.on)) {
      for (const id of entry.on) {
        noteQstampMap.set(id, entry.qstamp);
      }
    }
    if (Array.isArray(entry.off)) {
      for (const id of entry.off) {
        noteOffMap.set(id, entry.qstamp);
      }
    }
  }

  const clipMeasures: MeasureEntry[] = [];
  const seenQstamps = new Set<number>();
  for (const entry of timemap) {
    if (entry.measureOn !== undefined && !seenQstamps.has(entry.qstamp)) {
      seenQstamps.add(entry.qstamp);
      clipMeasures.push({ qstamp: entry.qstamp, measureOn: entry.measureOn });
    }
  }
  clipMeasures.sort((a, b) => a.qstamp - b.qstamp);

  const parseScoreIR = await ensureParseScoreIR();

  const ir = parseScoreIR(
    "",
    [svg],
    clipMeasures,
    noteQstampMap,
    tk.getVersion() as string,
    CLIP_RENDER_OPTS.pageWidth,
  );

  // Restore full-piece layout.
  tk.select({});
  tk.setOptions(VEROVIO_OPTS);
  tk.redoLayout({});

  const notes: ClipNote[] = [];
  for (const [id, startQ] of noteQstampMap) {
    const midiValues = tk.getMIDIValuesForElement(id) as Array<{ pitch: number }> | null;
    if (!midiValues || midiValues.length === 0) continue;
    const midi = midiValues[0].pitch;
    const endQ = noteOffMap.get(id) ?? startQ + 1;
    notes.push({ midi, startQ, endQ });
  }
  notes.sort((a, b) => a.startQ - b.startQ);

  return { svg, ir, notes };
}
```

Add the worker message handler for `get_clip_playback` in the `onmessage` dispatch. In the `WorkerInMsg` type union, add:
```typescript
| { type: "get_clip_playback"; requestId: string; pieceId: string; startBar: number; endBar: number }
```

In the `onmessage` handler, after the `get_ir` branch, add:
```typescript
} else if (msg.type === "get_clip_playback") {
  const playback = await processGetClipPlaybackRequest(tk, measures, msg.startBar, msg.endBar);
  if (playback === "failed") {
    (self as unknown as Worker).postMessage({
      requestId: msg.requestId,
      error: `get_clip_playback failed for ${msg.pieceId} bars ${msg.startBar}-${msg.endBar}`,
    });
  } else {
    (self as unknown as Worker).postMessage({ requestId: msg.requestId, payload: playback });
  }
}
```

In `apps/web/src/lib/score-renderer.ts`, add `getClipPlayback` method after `getClip`:

```typescript
async getClipPlayback(
  pieceId: string,
  startBar: number,
  endBar: number,
): Promise<{ svg: string; ir: ScoreIR; notes: import("./score-worker").ClipNote[] }> {
  const worker = this.ensureWorker();
  return new Promise((resolve, reject) => {
    const requestId = `req-${++this.requestCounter}`;
    this.pendingRequests.set(requestId, {
      resolve: resolve as (v: unknown) => void,
      reject,
      pieceId,
    });
    worker.postMessage({ type: "get_clip_playback", requestId, pieceId, startBar, endBar });
  });
}
```

Also add `import type { ScoreIR } from "./score-ir";` at the top of `score-renderer.ts` if not already present (it is already imported via the existing `irCache` field type — confirm by reading the top).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/lib/score-worker.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/score-worker.ts apps/web/src/lib/score-renderer.ts apps/web/src/lib/score-worker.test.ts && git commit -m "feat(#45): add get_clip_playback worker message and ScoreRenderer.getClipPlayback"
```

---

### Task 3: `LoopClock` — pure synthetic loop timing module

**Group:** A (parallel with Task 1, Task 2)

**Behavior being verified:** `LoopClock.qstampNow(nowMs)` correctly wraps at clip end, delays by count-in, and rescales when `setTempoFactor` is called.
**Interface under test:** `LoopClock` public API.

**Files:**
- Create: `apps/web/src/lib/loop-clock.ts`
- Create: `apps/web/src/lib/loop-clock.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/loop-clock.test.ts
import { describe, expect, it } from "vitest";
import { LoopClock } from "./loop-clock";

const OPTS = {
  clipStartQ: 16,   // bar 5 in a 4/4 piece at 4Q/bar
  clipEndQ: 32,     // bar 9 (exclusive)
  beatsPerBar: 4,
  bpmAtUnity: 120,  // 120 BPM at 1.0x
  tempoFactor: 1.0,
};

describe("LoopClock", () => {
  it("returns null before start() is called", () => {
    const clock = new LoopClock(OPTS);
    expect(clock.qstampNow(1000)).toBeNull();
  });

  it("returns null during count-in (first bar of metronome)", () => {
    const clock = new LoopClock(OPTS);
    clock.start(0); // startMs = 0
    // Count-in = 1 bar = beatsPerBar / (bpmAtUnity * tempoFactor) * 60 s
    // = 4 / (120 * 1.0) * 60 = 2 seconds = 2000ms
    // At 1000ms we are still in count-in.
    expect(clock.qstampNow(1000)).toBeNull();
  });

  it("returns clipStartQ immediately after count-in ends", () => {
    const clock = new LoopClock(OPTS);
    clock.start(0);
    // Count-in = 2000ms at 1.0x tempo.
    // At exactly 2000ms: elapsed = 0Q past clipStart.
    const q = clock.qstampNow(2000);
    expect(q).not.toBeNull();
    expect(q!).toBeCloseTo(OPTS.clipStartQ, 2);
  });

  it("advances qstamp proportionally to elapsed time", () => {
    const clock = new LoopClock(OPTS);
    clock.start(0);
    // After count-in (2000ms), one full clip = (clipEndQ - clipStartQ) / (bpmAtUnity * tempoFactor / 60)
    // = 16Q / (120/60 Q/s) = 16 / 2 = 8 seconds = 8000ms.
    // At 2000ms (count-in end) + 4000ms (half-clip) = 6000ms:
    const q = clock.qstampNow(6000);
    expect(q).not.toBeNull();
    // Half of clip range: clipStartQ + 8 = 24
    expect(q!).toBeCloseTo(24, 1);
  });

  it("wraps back to clipStartQ when clip duration elapses", () => {
    const clock = new LoopClock(OPTS);
    clock.start(0);
    // count-in (2000ms) + full clip (8000ms) = 10000ms → start of second pass
    const q = clock.qstampNow(10000);
    expect(q).not.toBeNull();
    expect(q!).toBeCloseTo(OPTS.clipStartQ, 2);
  });

  it("scales elapsed time by tempoFactor (0.5x takes twice as long)", () => {
    const clock = new LoopClock({ ...OPTS, tempoFactor: 0.5 });
    clock.start(0);
    // count-in at 0.5x = 4 seconds = 4000ms.
    // Still in count-in at 2000ms.
    expect(clock.qstampNow(2000)).toBeNull();
    // At 4000ms: just past count-in, q ≈ clipStartQ.
    const q = clock.qstampNow(4000);
    expect(q).not.toBeNull();
    expect(q!).toBeCloseTo(OPTS.clipStartQ, 2);
  });

  it("setTempoFactor rescales future qstamps without resetting position", () => {
    const clock = new LoopClock(OPTS);
    clock.start(0);
    // count-in = 2000ms. At t=6000ms: playback elapsed = 4s × 2 Q/s = 8Q → q = 16+8 = 24.
    const q1 = clock.qstampNow(6000);
    expect(q1).not.toBeNull();
    expect(q1!).toBeCloseTo(24, 1);

    // Halve tempo at t=6000ms — recalibrate so current position (q=24) is preserved.
    clock.setTempoFactor(0.5, 6000);

    // At t=6000ms + 1000ms = 7000ms: with 0.5x, qPerSec = 120*0.5/60 = 1 Q/s.
    // 1 second elapsed since calibration at q=24 → q ≈ 25.
    const q2 = clock.qstampNow(7000);
    expect(q2).not.toBeNull();
    expect(q2!).toBeCloseTo(25, 1);
  });

  it("stop() causes qstampNow to return null", () => {
    const clock = new LoopClock(OPTS);
    clock.start(0);
    clock.stop();
    expect(clock.qstampNow(3000)).toBeNull();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/lib/loop-clock.test.ts
```
Expected: FAIL — `Cannot find module './loop-clock'`.

- [ ] **Step 3: Implement**

```typescript
// apps/web/src/lib/loop-clock.ts

export interface LoopClockOptions {
  clipStartQ: number;
  clipEndQ: number;
  beatsPerBar: number;
  bpmAtUnity: number;
  tempoFactor: number;
}

/**
 * LoopClock maps wall-time (milliseconds) to a qstamp within [clipStartQ, clipEndQ),
 * wrapping on each pass and prefixing a one-bar count-in during which qstampNow() returns null.
 *
 * All state is pure: no DOM, no AudioContext, no requestAnimationFrame.
 */
export class LoopClock {
  private readonly clipStartQ: number;
  private readonly clipEndQ: number;
  private readonly clipRangeQ: number;
  private readonly beatsPerBar: number;
  private readonly bpmAtUnity: number;

  private startMs: number | null = null;
  private stopped = false;

  // Tempo factor and the recalibration anchor.
  // When setTempoFactor is called, we record the current elapsed position
  // (in Q) and reset the anchor so future mapping uses the new factor.
  private tempoFactor: number;
  // Effective "phase origin" in Q — offset added to the linear position.
  // After setTempoFactor(f, nowMs), this captures the Q position at nowMs
  // minus what the new factor would compute from nowMs.
  private phaseOriginQ = 0;
  // The wall-time at which the last calibration happened (relative to startMs).
  private calibrationMs = 0;

  constructor(opts: LoopClockOptions) {
    this.clipStartQ = opts.clipStartQ;
    this.clipEndQ = opts.clipEndQ;
    this.clipRangeQ = opts.clipEndQ - opts.clipStartQ;
    this.beatsPerBar = opts.beatsPerBar;
    this.bpmAtUnity = opts.bpmAtUnity;
    this.tempoFactor = opts.tempoFactor;
  }

  /** Begin the clock. nowMs is the current wall-time (e.g. Date.now() or performance.now()). */
  start(nowMs: number): void {
    this.startMs = nowMs;
    this.stopped = false;
    this.phaseOriginQ = 0;
    this.calibrationMs = 0;
  }

  /** Recalibrate tempo factor at the given wall-time, preserving current position. */
  setTempoFactor(factor: number, nowMs: number): void {
    if (this.startMs === null || this.stopped) {
      this.tempoFactor = factor;
      return;
    }
    // Capture current Q position before changing factor.
    // Use qstampNow to get the wrapped position (which is already in [clipStartQ, clipEndQ)).
    // Store the absolute unwrapped value so rawQAtMs produces the correct continuation.
    const wrappedQ = this.qstampNow(nowMs);
    if (wrappedQ === null) {
      // Still in count-in; only update the factor (count-in will be recalculated with new factor).
      this.tempoFactor = factor;
      return;
    }
    // Record the calibration point.
    this.calibrationMs = nowMs - this.startMs;
    // phaseOriginQ is the Q offset from clipStartQ at the calibration point.
    this.phaseOriginQ = wrappedQ - this.clipStartQ;
    this.tempoFactor = factor;
  }

  /** Returns the current qstamp, or null if in count-in or stopped/not started. */
  qstampNow(nowMs: number): number | null {
    if (this.startMs === null || this.stopped) return null;

    const countInMs = this.countInMs();
    const elapsedMs = nowMs - this.startMs;
    if (elapsedMs < countInMs) return null;

    const rawQ = this.rawQAtMs(nowMs);
    // Wrap into [clipStartQ, clipEndQ).
    const range = this.clipRangeQ;
    const offset = ((rawQ - this.clipStartQ) % range + range) % range;
    return this.clipStartQ + offset;
  }

  /** Stop the clock; qstampNow returns null after this. */
  stop(): void {
    this.stopped = true;
  }

  // Q per second at current tempo factor.
  private qPerSecond(): number {
    return (this.bpmAtUnity * this.tempoFactor) / 60;
  }

  // Count-in duration in milliseconds (one bar at current tempo).
  private countInMs(): number {
    const secondsPerBeat = 60 / (this.bpmAtUnity * this.tempoFactor);
    return this.beatsPerBar * secondsPerBeat * 1000;
  }

  // Raw Q position (not wrapped) at the given wall-time.
  // Piecewise linear: uses phaseOriginQ + (elapsed-since-calibration - count-in) * qPerSecond.
  // The count-in offset is subtracted so that Q = clipStartQ at the moment count-in ends.
  private rawQAtMs(nowMs: number): number {
    const countInSec = this.countInMs() / 1000;
    const elapsedSec = (nowMs - this.startMs!) / 1000 - this.calibrationMs / 1000;
    // After calibration, the count-in has already elapsed (calibrationMs >= countInMs on
    // any setTempoFactor call that happens post-count-in). On the initial pass
    // (calibrationMs = 0), subtract countInSec to anchor Q=clipStartQ at count-in end.
    const playbackSec = this.calibrationMs === 0
      ? Math.max(0, elapsedSec - countInSec)
      : elapsedSec;
    return this.clipStartQ + this.phaseOriginQ + playbackSec * this.qPerSecond();
  }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/lib/loop-clock.test.ts
```
Expected: PASS (all 8 assertions)

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/loop-clock.ts apps/web/src/lib/loop-clock.test.ts && git commit -m "feat(#45): add LoopClock pure timing module"
```

---

### Task 4: Install smplr + fetch-soundfont script

**Group:** B (depends on Group A — needs `ClipNote` type from Task 2, and smplr must be installed before Task 5)

**Behavior being verified:** `smplr` package is importable; `apps/web/public/soundfonts/acoustic_grand_piano/` directory is populated after running the script.
**Interface under test:** Script execution + filesystem output.

**Files:**
- Modify: `apps/web/package.json` (via `bun add smplr`)
- Create: `apps/web/scripts/fetch-soundfont.ts`

- [ ] **Step 1: Write the failing test**

The "test" here is running the fetch script and confirming the soundfont directory is created. Write the script first so that it is testable:

```typescript
// apps/web/scripts/fetch-soundfont.ts
// Fetches the acoustic grand piano SoundFont sample pack from the smplr CDN
// and writes it into apps/web/public/soundfonts/ for self-hosted playback.
//
// Usage: bun apps/web/scripts/fetch-soundfont.ts
//
// The smplr package serves samples from:
//   https://gleitz.github.io/midi-js-soundfonts/MusyngKite/{instrument}-mp3.js
// Each file is a JS object mapping note names to base64 data URIs.
// We download the acoustic_grand_piano instrument and write:
//   apps/web/public/soundfonts/acoustic_grand_piano-mp3.js

import { writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const INSTRUMENT = "acoustic_grand_piano";
const CDN_URL = `https://gleitz.github.io/midi-js-soundfonts/MusyngKite/${INSTRUMENT}-mp3.js`;
const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(__dirname, "../public/soundfonts");
const OUT_FILE = resolve(OUT_DIR, `${INSTRUMENT}-mp3.js`);

async function main() {
  console.log(`Fetching ${CDN_URL} ...`);
  const res = await fetch(CDN_URL);
  if (!res.ok) {
    throw new Error(`Fetch failed: ${res.status} ${res.statusText}`);
  }
  const text = await res.text();
  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(OUT_FILE, text, "utf-8");
  console.log(`Written to ${OUT_FILE} (${text.length} bytes)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
```

- [ ] **Step 2: Run test — verify it FAILS (before install)**

Check smplr is not yet installed:
```bash
cd apps/web && node -e "require('smplr')" 2>&1 | grep -q "Cannot find" && echo "FAIL: smplr not installed yet"
```
Expected: prints "FAIL: smplr not installed yet"

- [ ] **Step 3: Implement**

```bash
cd apps/web && bun add smplr
```

Then create the scripts directory and the fetch script:
```bash
mkdir -p apps/web/scripts
```
Write `apps/web/scripts/fetch-soundfont.ts` as shown in Step 1.

Run the fetch script to populate the soundfont:
```bash
bun apps/web/scripts/fetch-soundfont.ts
```
Expected: `Written to apps/web/public/soundfonts/acoustic_grand_piano-mp3.js (... bytes)`.

Add the soundfont file to `.gitignore` if it is large (> 5MB) — add this line to `apps/web/public/.gitignore` (create it if absent):
```
soundfonts/
```

Actually: the soundfont JS file is ~3MB. Commit it to the repo so `just dev-light` and tests work without running the script. Do NOT add to gitignore. Check size first:
```bash
wc -c apps/web/public/soundfonts/acoustic_grand_piano-mp3.js
```
If < 10MB, commit it. If > 10MB, add to gitignore and document that the script must be run.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
ls apps/web/public/soundfonts/acoustic_grand_piano-mp3.js && echo "PASS"
```
Expected: file exists.

```bash
cd apps/web && node -e "import('smplr').then(m => console.log('smplr version:', m.Soundfont ? 'ok' : 'missing Soundfont'))"
```
Expected: `smplr version: ok`

- [ ] **Step 5: Commit**

```bash
git add apps/web/package.json apps/web/bun.lock apps/web/scripts/fetch-soundfont.ts apps/web/public/soundfonts/ && git commit -m "feat(#45): add smplr dependency and self-hosted acoustic grand piano soundfont"
```

---

### Task 5: `LoopPlayer` — audio orchestrator

**Group:** B (depends on Group A — needs `LoopClock` from Task 3, `ClipNote` from Task 2, smplr from Task 4)

**Behavior being verified:** `LoopPlayer.play()` schedules notes with start times proportional to `tempoFactor`; transport state transitions are correct; `setTempoFactor` changes timing of future notes.
**Interface under test:** `LoopPlayer` public API.

**Files:**
- Create: `apps/web/src/lib/loop-player.ts`
- Create: `apps/web/src/lib/loop-player.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/loop-player.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ClipNote } from "./score-worker";
import type { ScoreIR } from "./score-ir";

// Minimal stub ScoreIR — LoopPlayer only reads ir.bars[].qstampStart and qstampEnd
// to compute bpmAtUnity and clip range. We supply a 4-bar clip at 4Q/bar.
const STUB_IR: ScoreIR = {
  pieceId: "test",
  verovioVersion: "4.0.0",
  pageWidth: 1600,
  pages: [{ pageN: 1, viewBox: "0 0 1600 600", width: 1600, height: 600, systemBboxes: [] }],
  bars: [
    { barNumber: 5, measureOn: "m5", pageN: 1, bbox: { x: 0, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 16, qstampEnd: 20 },
    { barNumber: 6, measureOn: "m6", pageN: 1, bbox: { x: 100, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 20, qstampEnd: 24 },
    { barNumber: 7, measureOn: "m7", pageN: 1, bbox: { x: 200, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 24, qstampEnd: 28 },
    { barNumber: 8, measureOn: "m8", pageN: 1, bbox: { x: 300, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 28, qstampEnd: 32 },
  ],
  notes: {},
};

const STUB_NOTES: ClipNote[] = [
  { midi: 60, startQ: 16, endQ: 18 },
  { midi: 64, startQ: 20, endQ: 22 },
];

// Fake AudioContext — exposes currentTime as mutable number.
function makeFakeAudioContext() {
  const scheduled: Array<{ note: number; time: number; duration: number }> = [];
  const oscillators: Array<{ frequency: { value: number }; start: (t: number) => void; stop: (t: number) => void; connect: () => void }> = [];
  const ctx = {
    currentTime: 0,
    state: "running" as "running" | "suspended",
    destination: {},
    resume: vi.fn().mockResolvedValue(undefined),
    close: vi.fn().mockResolvedValue(undefined),
    createOscillator: vi.fn(() => {
      const osc = { frequency: { value: 0 }, type: "sine" as OscillatorType, start: vi.fn(), stop: vi.fn(), connect: vi.fn() };
      oscillators.push(osc);
      return osc;
    }),
    createGain: vi.fn(() => ({
      gain: { setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() },
      connect: vi.fn(),
    })),
  } as unknown as AudioContext;
  return { ctx, scheduled, oscillators };
}

// Fake smplr Soundfont stub.
function makeFakePiano(scheduled: Array<{ note: number; time: number; duration: number }>) {
  return {
    loaded: true,
    start: vi.fn(({ note, time, duration }: { note: number; time: number; duration: number }) => {
      scheduled.push({ note, time, duration });
    }),
    stop: vi.fn(),
  };
}

vi.mock("smplr", () => ({
  Soundfont: vi.fn().mockImplementation((_ctx: unknown, _opts: unknown) => ({
    load: vi.fn().mockResolvedValue(undefined),
    start: vi.fn(),
    stop: vi.fn(),
    loaded: true,
  })),
}));

describe("LoopPlayer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  it("starts in idle state", async () => {
    const { LoopPlayer } = await import("./loop-player");
    const { ctx } = makeFakeAudioContext();
    const player = new LoopPlayer({
      ctx,
      instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
      clipIR: STUB_IR,
      clipNotes: STUB_NOTES,
      beatsPerBar: 4,
      bpmAtUnity: 120,
      tempoFactor: 1.0,
    });
    expect(player.state).toBe("idle");
    player.destroy();
  });

  it("transitions to counting-in then playing after play()", async () => {
    const { LoopPlayer } = await import("./loop-player");
    const { ctx } = makeFakeAudioContext();
    const player = new LoopPlayer({
      ctx,
      instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
      clipIR: STUB_IR,
      clipNotes: STUB_NOTES,
      beatsPerBar: 4,
      bpmAtUnity: 120,
      tempoFactor: 1.0,
    });
    await player.play();
    expect(player.state === "counting-in" || player.state === "playing").toBe(true);
    player.destroy();
  });

  it("pause() transitions from playing to paused", async () => {
    const { LoopPlayer } = await import("./loop-player");
    const { ctx } = makeFakeAudioContext();
    const player = new LoopPlayer({
      ctx,
      instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
      clipIR: STUB_IR,
      clipNotes: STUB_NOTES,
      beatsPerBar: 4,
      bpmAtUnity: 120,
      tempoFactor: 1.0,
    });
    await player.play();
    player.pause();
    expect(player.state).toBe("paused");
    player.destroy();
  });

  it("stop() transitions to idle and qstampSource returns null", async () => {
    const { LoopPlayer } = await import("./loop-player");
    const { ctx } = makeFakeAudioContext();
    const player = new LoopPlayer({
      ctx,
      instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
      clipIR: STUB_IR,
      clipNotes: STUB_NOTES,
      beatsPerBar: 4,
      bpmAtUnity: 120,
      tempoFactor: 1.0,
    });
    await player.play();
    player.stop();
    expect(player.state).toBe("idle");
    expect(player.qstampSource()).toBeNull();
    player.destroy();
  });

  it("setTempoFactor updates tempoFactor on the clock", async () => {
    const { LoopPlayer } = await import("./loop-player");
    const { ctx } = makeFakeAudioContext();
    const player = new LoopPlayer({
      ctx,
      instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
      clipIR: STUB_IR,
      clipNotes: STUB_NOTES,
      beatsPerBar: 4,
      bpmAtUnity: 120,
      tempoFactor: 1.0,
    });
    await player.play();
    player.setTempoFactor(0.5);
    // After setTempoFactor, the exposed tempoFactor should reflect the new value.
    expect(player.tempoFactor).toBe(0.5);
    player.destroy();
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/lib/loop-player.test.ts
```
Expected: FAIL — `Cannot find module './loop-player'`.

- [ ] **Step 3: Implement**

```typescript
// apps/web/src/lib/loop-player.ts
import { Soundfont } from "smplr";
import { LoopClock } from "./loop-clock";
import { Sentry } from "./sentry";
import type { ClipNote } from "./score-worker";
import type { ScoreIR } from "./score-ir";

export type LoopPlayerState = "idle" | "counting-in" | "playing" | "paused";

export interface LoopPlayerOptions {
  ctx: AudioContext;
  instrumentUrl: string;
  clipIR: ScoreIR;
  clipNotes: ClipNote[];
  beatsPerBar: number;
  bpmAtUnity: number;
  tempoFactor: number;
}

const LOOKAHEAD_SEC = 0.1;   // schedule 100ms ahead
const SCHEDULER_INTERVAL_MS = 25;

export class LoopPlayer {
  state: LoopPlayerState = "idle";
  tempoFactor: number;
  audioUnavailable = false;

  private readonly ctx: AudioContext;
  private readonly clipNotes: ClipNote[];
  private readonly beatsPerBar: number;
  private readonly bpmAtUnity: number;
  private readonly clipStartQ: number;
  private readonly clipEndQ: number;

  private piano: Soundfont | null = null;
  private pianoLoadPromise: Promise<void> | null = null;
  private clock: LoopClock | null = null;
  private schedulerTimer: ReturnType<typeof setInterval> | null = null;
  // Track which notes have been scheduled in the current pass to avoid re-scheduling.
  // Key: `${passIndex}-${noteIndex}`.
  private scheduledNoteKeys = new Set<string>();
  private currentPassIndex = 0;
  private lastScheduledQstamp = -Infinity;

  constructor(opts: LoopPlayerOptions) {
    this.ctx = opts.ctx;
    this.tempoFactor = opts.tempoFactor;
    this.clipNotes = opts.clipNotes;
    this.beatsPerBar = opts.beatsPerBar;
    this.bpmAtUnity = opts.bpmAtUnity;

    const bars = opts.clipIR.bars;
    this.clipStartQ = bars.length > 0 ? bars[0].qstampStart : 0;
    this.clipEndQ = bars.length > 0 ? bars[bars.length - 1].qstampEnd : 0;

    // Begin loading instrument in the background.
    this.pianoLoadPromise = this.loadInstrument(opts.instrumentUrl);
  }

  private async loadInstrument(instrumentUrl: string): Promise<void> {
    try {
      this.piano = new Soundfont(this.ctx, { instrument: instrumentUrl });
      await this.piano.load;
    } catch (err) {
      this.audioUnavailable = true;
      Sentry.captureException(err);
      console.error("[LoopPlayer] smplr Soundfont load failed:", err);
    }
  }

  async play(): Promise<void> {
    if (this.state === "playing" || this.state === "counting-in") return;

    if (this.ctx.state === "suspended") {
      await this.ctx.resume();
    }

    // Wait for instrument (non-blocking if already loaded or failed).
    if (this.pianoLoadPromise) {
      await this.pianoLoadPromise;
    }

    const nowMs = performance.now();
    this.clock = new LoopClock({
      clipStartQ: this.clipStartQ,
      clipEndQ: this.clipEndQ,
      beatsPerBar: this.beatsPerBar,
      bpmAtUnity: this.bpmAtUnity,
      tempoFactor: this.tempoFactor,
    });
    this.clock.start(nowMs);
    this.scheduledNoteKeys.clear();
    this.currentPassIndex = 0;
    this.lastScheduledQstamp = -Infinity;

    this.state = "counting-in";
    this.startScheduler();
  }

  pause(): void {
    if (this.state !== "playing" && this.state !== "counting-in") return;
    this.stopScheduler();
    this.clock?.stop();
    this.state = "paused";
  }

  stop(): void {
    this.stopScheduler();
    this.clock?.stop();
    this.clock = null;
    this.scheduledNoteKeys.clear();
    this.state = "idle";
  }

  setTempoFactor(factor: number): void {
    this.tempoFactor = factor;
    this.clock?.setTempoFactor(factor, performance.now());
  }

  qstampSource(): number | null {
    if (!this.clock) return null;
    return this.clock.qstampNow(performance.now());
  }

  destroy(): void {
    this.stop();
  }

  private startScheduler(): void {
    if (this.schedulerTimer !== null) return;
    this.schedulerTimer = setInterval(() => this.scheduleTick(), SCHEDULER_INTERVAL_MS);
  }

  private stopScheduler(): void {
    if (this.schedulerTimer !== null) {
      clearInterval(this.schedulerTimer);
      this.schedulerTimer = null;
    }
  }

  private scheduleTick(): void {
    if (!this.clock) return;

    const nowMs = performance.now();
    const nowQ = this.clock.qstampNow(nowMs);

    // Still in count-in.
    if (nowQ === null) {
      // Schedule metronome count-in beats.
      this.scheduleMetronome(nowMs);
      return;
    }

    if (this.state === "counting-in") {
      this.state = "playing";
    }

    // Compute the lookahead horizon in Q.
    const qPerSec = (this.bpmAtUnity * this.tempoFactor) / 60;
    const lookaheadQ = LOOKAHEAD_SEC * qPerSec;
    const horizonQ = nowQ + lookaheadQ;

    // Detect pass wrap: if horizonQ >= clipEndQ, we may be scheduling into the next pass.
    const passAdvance = Math.floor((horizonQ - this.clipStartQ) / (this.clipEndQ - this.clipStartQ));
    if (passAdvance > this.currentPassIndex) {
      this.currentPassIndex = passAdvance;
      this.scheduledNoteKeys.clear();
    }

    for (let i = 0; i < this.clipNotes.length; i++) {
      const note = this.clipNotes[i];
      // Check if this note falls within [nowQ, horizonQ).
      const noteQInPass = note.startQ;
      if (noteQInPass < nowQ || noteQInPass >= horizonQ) continue;

      const key = `${this.currentPassIndex}-${i}`;
      if (this.scheduledNoteKeys.has(key)) continue;
      this.scheduledNoteKeys.add(key);

      if (this.piano && !this.audioUnavailable) {
        // Convert qstamp to audio-context absolute time.
        const qOffset = noteQInPass - nowQ;
        const secOffset = qOffset / qPerSec;
        const audioTime = this.ctx.currentTime + secOffset;
        const durationQ = note.endQ - note.startQ;
        const durationSec = Math.max(0.05, durationQ / qPerSec);

        this.piano.start({
          note: note.midi,
          time: audioTime,
          duration: durationSec,
          velocity: 80,
        });
      }
    }

    // Schedule metronome beats in the lookahead window.
    this.scheduleMetronome(nowMs);
  }

  private scheduleMetronome(nowMs: number): void {
    // Simple oscillator click for each beat in the lookahead window.
    // Uses AudioContext time for tight scheduling.
    const qPerSec = (this.bpmAtUnity * this.tempoFactor) / 60;
    const secPerBeat = 60 / (this.bpmAtUnity * this.tempoFactor);
    const lookaheadAudioTime = this.ctx.currentTime + LOOKAHEAD_SEC;

    // Compute the next beat audio time from the clock's current position.
    const nowQ = this.clock?.qstampNow(nowMs);
    if (nowQ === null && this.clock) {
      // In count-in: schedule beats relative to clock start.
      // Count-in started at clock.startMs; beats are evenly spaced by secPerBeat.
      // This is handled implicitly by the metronome audio scheduling below.
    }

    // Metronome scheduling uses a persistent nextMetronomeBeatTime ref,
    // initialized when play() is called.
    if (this.nextMetronomeBeatTime === null) return;

    while (this.nextMetronomeBeatTime <= lookaheadAudioTime) {
      this.playMetronomeClick(this.nextMetronomeBeatTime, this.metronomeBeatIndex % this.beatsPerBar === 0);
      this.nextMetronomeBeatTime += secPerBeat;
      this.metronomeBeatIndex++;
    }
  }

  // Initialized by play(); reset on stop().
  private nextMetronomeBeatTime: number | null = null;
  private metronomeBeatIndex = 0;

  // Override startScheduler to also initialize metronome timing.
  // (Add this after the play() call to this.clock.start() completes.)

  private playMetronomeClick(time: number, accent: boolean): void {
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.frequency.value = accent ? 1000 : 800;
    osc.type = "sine";
    gain.gain.setValueAtTime(accent ? 0.5 : 0.25, time);
    gain.gain.exponentialRampToValueAtTime(0.001, time + 0.04);
    osc.start(time);
    osc.stop(time + 0.04);
  }
}
```

**Note:** The `nextMetronomeBeatTime` field and metronome initialization need to be wired into `play()`. Update `play()` to initialize them:

```typescript
// In play(), after this.clock.start(nowMs):
this.nextMetronomeBeatTime = this.ctx.currentTime;
this.metronomeBeatIndex = 0;
```

And in `stop()`:
```typescript
// Add after stopScheduler():
this.nextMetronomeBeatTime = null;
this.metronomeBeatIndex = 0;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/lib/loop-player.test.ts
```
Expected: PASS (all 5 assertions)

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/loop-player.ts apps/web/src/lib/loop-player.test.ts && git commit -m "feat(#45): add LoopPlayer audio orchestrator (smplr + metronome + LoopClock)"
```

---

### Task 6: `useLoopPlayer` hook + `LoopTransport` UI component

**Group:** C (depends on Group B — needs LoopPlayer from Task 5)

**Behavior being verified:** `ExerciseSetCard` with `scoreClip.tempoFactor` renders a transport bar (play/pause/stop, tempo slider); corpus_drill stub renders no transport.
**Interface under test:** `ExerciseSetCard` render (via `@testing-library/react`).

**Files:**
- Create: `apps/web/src/hooks/useLoopPlayer.ts`
- Create: `apps/web/src/components/LoopTransport.tsx`
- Modify: `apps/web/src/components/cards/ExerciseSetCard.tsx`
- Modify: `apps/web/src/components/cards/ExerciseSetCard.test.tsx`

- [ ] **Step 1: Write the failing test**

Add to `apps/web/src/components/cards/ExerciseSetCard.test.tsx`:

```typescript
// Add at the top of the file, alongside existing mocks:
const mockGetClipPlayback = vi.fn();

vi.mock("../../hooks/useLoopPlayer", () => ({
  useLoopPlayer: vi.fn().mockReturnValue({
    isPlaying: false,
    isCounting: false,
    audioUnavailable: false,
    tempoFactor: 0.75,
    play: vi.fn(),
    pause: vi.fn(),
    stop: vi.fn(),
    setTempoFactor: vi.fn(),
    qstampSource: vi.fn().mockReturnValue(null),
  }),
}));

// Also mock score-renderer to expose getClipPlayback:
vi.mock("../../lib/score-renderer", () => ({
  scoreRenderer: {
    getClip: (...args: unknown[]) => mockGetClip(...args),
    getClipPlayback: (...args: unknown[]) => mockGetClipPlayback(...args),
  },
}));
```

Add test cases:

```typescript
it("renders score as hero and shows transport when scoreClip has tempoFactor", async () => {
  mockGetClipPlayback.mockResolvedValue({
    svg: "<svg data-test='loop-clip'></svg>",
    ir: {
      pieceId: "test",
      verovioVersion: "4.0.0",
      pageWidth: 1600,
      pages: [{ pageN: 1, viewBox: "0 0 1600 600", width: 1600, height: 600, systemBboxes: [] }],
      bars: [
        { barNumber: 5, measureOn: "m5", pageN: 1, bbox: { x: 0, y: 0, w: 0, h: 0 }, noteIds: [], qstampStart: 16, qstampEnd: 20 },
      ],
      notes: {},
    },
    notes: [{ midi: 60, startQ: 16, endQ: 18 }],
  });
  const config: ExerciseSetConfig = {
    sourcePassage: "bars 5-8",
    targetSkill: "dynamics focus",
    scoreClip: { pieceId: "chopin.ballades.1", bars: [5, 8], tempoFactor: 0.75 },
    exercises: [
      { title: "Loop passage", instruction: "Loop at 75%.", focusDimension: "dynamics" },
    ],
  };
  const { ExerciseSetCard } = await import("./ExerciseSetCard");
  render(React.createElement(ExerciseSetCard, { config }));

  await waitFor(() => {
    expect(mockGetClipPlayback).toHaveBeenCalledWith("chopin.ballades.1", 5, 8);
    expect(document.body.innerHTML).toContain('data-test="loop-clip"');
  });
  // Transport rendered — play button present.
  expect(document.body.querySelector('[data-testid="loop-transport"]')).not.toBeNull();
});

it("corpus_drill (no scoreClip, no tempoFactor) renders no transport", async () => {
  const config: ExerciseSetConfig = {
    sourcePassage: "bars 1-8",
    targetSkill: "timing focus",
    exercises: [
      {
        title: "Timing corpus drill",
        instruction: "Timing drill coming soon.",
        focusDimension: "timing",
      },
    ],
  };
  const { ExerciseSetCard } = await import("./ExerciseSetCard");
  render(React.createElement(ExerciseSetCard, { config }));
  await waitFor(() => {
    expect(document.body.textContent).toContain("timing focus");
  });
  expect(document.body.querySelector('[data-testid="loop-transport"]')).toBeNull();
  expect(mockGetClipPlayback).not.toHaveBeenCalled();
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/cards/ExerciseSetCard.test.tsx
```
Expected: FAIL — `useLoopPlayer` module not found, `[data-testid="loop-transport"]` not present.

- [ ] **Step 3: Implement**

**`apps/web/src/hooks/useLoopPlayer.ts`:**

```typescript
// apps/web/src/hooks/useLoopPlayer.ts
import { useCallback, useEffect, useRef, useState } from "react";
import { LoopPlayer } from "../lib/loop-player";
import type { ClipNote } from "../lib/score-worker";
import type { ScoreIR } from "../lib/score-ir";

export interface UseLoopPlayerConfig {
  clipIR: ScoreIR | null;
  clipNotes: ClipNote[];
  beatsPerBar: number;
  bpmAtUnity: number;
  tempoFactor: number;
}

export interface UseLoopPlayerReturn {
  isPlaying: boolean;
  isCounting: boolean;
  audioUnavailable: boolean;
  tempoFactor: number;
  play: () => void;
  pause: () => void;
  stop: () => void;
  setTempoFactor: (f: number) => void;
  qstampSource: () => number | null;
}

export function useLoopPlayer(config: UseLoopPlayerConfig): UseLoopPlayerReturn {
  const playerRef = useRef<LoopPlayer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isCounting, setIsCounting] = useState(false);
  const [audioUnavailable, setAudioUnavailable] = useState(false);
  const [tempoFactor, setTempoFactorState] = useState(config.tempoFactor);
  const ctxRef = useRef<AudioContext | null>(null);

  // Destroy the player when clipIR changes or component unmounts.
  useEffect(() => {
    return () => {
      playerRef.current?.destroy();
      playerRef.current = null;
      ctxRef.current?.close().catch(() => {});
      ctxRef.current = null;
    };
  }, []);

  // Rebuild player when clipIR becomes available.
  useEffect(() => {
    if (!config.clipIR) return;
    if (playerRef.current) {
      playerRef.current.destroy();
    }
    if (!ctxRef.current || ctxRef.current.state === "closed") {
      ctxRef.current = new AudioContext();
    }
    playerRef.current = new LoopPlayer({
      ctx: ctxRef.current,
      instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
      clipIR: config.clipIR,
      clipNotes: config.clipNotes,
      beatsPerBar: config.beatsPerBar,
      bpmAtUnity: config.bpmAtUnity,
      tempoFactor: config.tempoFactor,
    });
  }, [config.clipIR, config.clipNotes, config.beatsPerBar, config.bpmAtUnity, config.tempoFactor]);

  const play = useCallback(() => {
    const player = playerRef.current;
    if (!player) return;
    player.play().then(() => {
      setIsPlaying(true);
      setIsCounting(player.state === "counting-in");
      setAudioUnavailable(player.audioUnavailable);
    }).catch(() => {});
  }, []);

  const pause = useCallback(() => {
    playerRef.current?.pause();
    setIsPlaying(false);
    setIsCounting(false);
  }, []);

  const stop = useCallback(() => {
    playerRef.current?.stop();
    setIsPlaying(false);
    setIsCounting(false);
  }, []);

  const setTempoFactor = useCallback((f: number) => {
    playerRef.current?.setTempoFactor(f);
    setTempoFactorState(f);
  }, []);

  const qstampSource = useCallback((): number | null => {
    return playerRef.current?.qstampSource() ?? null;
  }, []);

  return { isPlaying, isCounting, audioUnavailable, tempoFactor, play, pause, stop, setTempoFactor, qstampSource };
}
```

**`apps/web/src/components/LoopTransport.tsx`:**

```typescript
// apps/web/src/components/LoopTransport.tsx

interface LoopTransportProps {
  isPlaying: boolean;
  isCounting: boolean;
  audioUnavailable: boolean;
  tempoFactor: number;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onTempoChange: (f: number) => void;
}

export function LoopTransport({
  isPlaying,
  isCounting,
  audioUnavailable,
  tempoFactor,
  onPlay,
  onPause,
  onStop,
  onTempoChange,
}: LoopTransportProps) {
  return (
    <div
      data-testid="loop-transport"
      className="flex items-center gap-3 px-4 py-2 bg-surface-card border-t border-border/60"
    >
      {/* Play / Pause */}
      <button
        type="button"
        onClick={isPlaying || isCounting ? onPause : onPlay}
        className="shrink-0 w-8 h-8 flex items-center justify-center rounded-full border border-border hover:border-accent text-text-secondary hover:text-cream transition-colors"
        aria-label={isPlaying || isCounting ? "Pause" : "Play"}
      >
        {isPlaying || isCounting ? (
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
            <rect x="1" y="1" width="4" height="10" rx="1" />
            <rect x="7" y="1" width="4" height="10" rx="1" />
          </svg>
        ) : (
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
            <polygon points="2,1 11,6 2,11" />
          </svg>
        )}
      </button>

      {/* Stop */}
      <button
        type="button"
        onClick={onStop}
        disabled={!isPlaying && !isCounting}
        className="shrink-0 w-8 h-8 flex items-center justify-center rounded-full border border-border hover:border-accent text-text-secondary hover:text-cream transition-colors disabled:opacity-40"
        aria-label="Stop"
      >
        <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor" aria-hidden="true">
          <rect x="0" y="0" width="10" height="10" rx="1" />
        </svg>
      </button>

      {/* Tempo slider */}
      <div className="flex items-center gap-2 flex-1 min-w-0">
        <span className="text-label-sm text-text-tertiary shrink-0">Speed</span>
        <input
          type="range"
          min={0.25}
          max={1.0}
          step={0.05}
          value={tempoFactor}
          onChange={(e) => onTempoChange(Number.parseFloat(e.target.value))}
          className="flex-1 accent-accent"
          aria-label="Tempo factor"
        />
        <span className="text-label-sm text-text-tertiary shrink-0 w-10 text-right">
          {Math.round(tempoFactor * 100)}%
        </span>
      </div>

      {/* Audio unavailable badge */}
      {audioUnavailable && (
        <span className="text-label-xs text-amber-400 shrink-0">Audio unavailable</span>
      )}

      {/* Count-in indicator */}
      {isCounting && (
        <span className="text-label-xs text-text-tertiary shrink-0 animate-pulse">Count in...</span>
      )}
    </div>
  );
}
```

**Update `ExerciseSetCard.tsx`** to:
1. Call `scoreRenderer.getClipPlayback` (instead of `getClip`) when `scoreClip` has `tempoFactor`.
2. Fall back to `getClip` for scoreClip without tempoFactor (backward compat).
3. Wire `useLoopPlayer` and `ScoreCursor` over the clip IR.
4. Render score as hero, transport below score, text below transport.

Key changes to `ExerciseSetCard.tsx`:

```typescript
// Replace the current useEffect + state section with:
import { useEffect, useRef, useCallback, useState } from "react";
import { scoreRenderer } from "../../lib/score-renderer";
import { ScoreCursor } from "../../lib/score-cursor";
import { useLoopPlayer } from "../../hooks/useLoopPlayer";
import { LoopTransport } from "../LoopTransport";
import type { ScoreIR } from "../../lib/score-ir";
import type { ClipNote } from "../../lib/score-worker";

// Inside ExerciseSetCard:
const [scoreClipSvg, setScoreClipSvg] = useState<string | null>(null);
const [clipIR, setClipIR] = useState<ScoreIR | null>(null);
const [clipNotes, setClipNotes] = useState<ClipNote[]>([]);
const [clipLoadError, setClipLoadError] = useState(false);
const scoreContainerRef = useRef<HTMLDivElement>(null);

const hasTempoFactor = !!config.scoreClip?.tempoFactor;

useEffect(() => {
  if (!config.scoreClip) return;
  let cancelled = false;
  if (hasTempoFactor) {
    scoreRenderer
      .getClipPlayback(config.scoreClip.pieceId, config.scoreClip.bars[0], config.scoreClip.bars[1])
      .then((r) => {
        if (!cancelled) {
          setScoreClipSvg(r.svg);
          setClipIR(r.ir);
          setClipNotes(r.notes);
        }
      })
      .catch((err) => {
        console.error("ExerciseSetCard: failed to load clip playback", err);
        if (!cancelled) setClipLoadError(true);
      });
  } else {
    scoreRenderer
      .getClip(config.scoreClip.pieceId, config.scoreClip.bars[0], config.scoreClip.bars[1])
      .then((r) => { if (!cancelled) setScoreClipSvg(r); })
      .catch((err) => {
        console.error("ExerciseSetCard: failed to load score clip", err);
        if (!cancelled) setClipLoadError(true);
      });
  }
  return () => { cancelled = true; };
}, [config.scoreClip, hasTempoFactor]);

const loopPlayer = useLoopPlayer({
  clipIR: hasTempoFactor ? clipIR : null,
  clipNotes,
  beatsPerBar: 4,   // default 4/4; could be made configurable in S3
  bpmAtUnity: 120,  // default; will be tunable in S3 when tempo BPM is known
  tempoFactor: config.scoreClip?.tempoFactor ?? 1.0,
});

// ScoreCursor wired to loop clock qstamp source.
useEffect(() => {
  if (!hasTempoFactor || clipIR === null || scoreContainerRef.current === null) return;
  const cursor = new ScoreCursor({
    pieceId: config.scoreClip!.pieceId,
    container: scoreContainerRef.current,
    ir: clipIR,
    qstampSource: loopPlayer.qstampSource,
  });
  cursor.start();
  return () => cursor.stop();
}, [clipIR, hasTempoFactor, config.scoreClip, loopPlayer.qstampSource]);
```

The JSX becomes:

```tsx
return (
  <div className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3">
    {/* HERO: Score clip */}
    {config.scoreClip && scoreClipSvg && !clipLoadError && (
      <div
        ref={hasTempoFactor ? scoreContainerRef : undefined}
        className="bg-white border-b border-border/60 relative"
        style={{ minHeight: "120px" }}
      >
        {/* For tempoFactor path, scoreContainerRef is used by ScoreCursor;
            we still render the SVG via ClipSvg for initial paint. */}
        {!hasTempoFactor && <ClipSvg svg={scoreClipSvg} />}
        {hasTempoFactor && <ClipSvg svg={scoreClipSvg} />}
      </div>
    )}

    {/* Transport (own_passage_loop with tempoFactor only) */}
    {hasTempoFactor && scoreClipSvg && !clipLoadError && (
      <LoopTransport
        isPlaying={loopPlayer.isPlaying}
        isCounting={loopPlayer.isCounting}
        audioUnavailable={loopPlayer.audioUnavailable}
        tempoFactor={loopPlayer.tempoFactor}
        onPlay={loopPlayer.play}
        onPause={loopPlayer.pause}
        onStop={loopPlayer.stop}
        onTempoChange={loopPlayer.setTempoFactor}
      />
    )}

    {/* Header (text — secondary) */}
    <div className="px-4 pt-4 pb-3 flex items-start justify-between gap-3">
      <div className="min-w-0">
        <h4 className="font-display text-body-md text-text-primary leading-snug">
          {config.targetSkill}
        </h4>
        <p className="text-body-xs text-text-tertiary mt-0.5 truncate">
          {config.sourcePassage}
        </p>
      </div>
      {onExpand && (
        <button
          type="button"
          onClick={onExpand}
          className="shrink-0 text-text-tertiary hover:text-cream transition-colors pt-0.5"
          aria-label="Expand exercise set"
        >
          <ArrowsOut size={14} />
        </button>
      )}
    </div>

    <div className="border-t border-border/60" />

    {/* Exercise rows */}
    <div>
      {config.exercises.map((exercise, i) => (
        <ExerciseItem
          key={exercise.title}
          exercise={exercise}
          isExpanded={expandedIndex === i}
          onToggle={() => setExpandedIndex(expandedIndex === i ? null : i)}
          artifactId={artifactId}
          isFirst={i === 0}
        />
      ))}
    </div>
  </div>
);
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/cards/ExerciseSetCard.test.tsx
```
Expected: PASS (all tests including existing corpus_drill stub test)

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/hooks/useLoopPlayer.ts apps/web/src/components/LoopTransport.tsx apps/web/src/components/cards/ExerciseSetCard.tsx apps/web/src/components/cards/ExerciseSetCard.test.tsx && git commit -m "feat(#45): score-first card redesign with LoopTransport, useLoopPlayer, ScoreCursor"
```

---

### Task 7: Full test run + green confirmation

**Group:** D (depends on Group C — final gate)

**Behavior being verified:** All web tests pass; no regressions in existing tests.

**Files:** None created/modified — this is a verification-only task.

- [ ] **Step 1: Run all web tests**

```bash
cd apps/web && bun run test
```
Expected: All tests PASS. Pay attention to:
- `ExerciseSetCard.test.tsx` — all 6+ cases green
- `loop-clock.test.ts` — all 8 cases green
- `loop-player.test.ts` — all 5 cases green
- `score-worker.test.ts` — existing cases + new `processGetClipPlaybackRequest` case green
- All pre-existing tests unchanged

If any test fails, diagnose and fix in this task before committing.

- [ ] **Step 2: Run API tests**

```bash
cd apps/api && bun run test src/services/tool-processor.test.ts
```
Expected: PASS including new `tempoFactor` assertion.

- [ ] **Step 3: Commit (only if fixes were needed)**

```bash
git add -p && git commit -m "fix(#45): address test failures found in full test run"
```
If no fixes needed, no commit required.

---

### Task 8: Manual browser verification (`just dev-light`)

**Group:** D (parallel with Task 7 — can be done while tests run)

This task is the only verification the automated test suite cannot provide: confirming that the loop actually plays in the browser with cursor + metronome + piano at the correct tempo.

**Steps:**

1. Ensure the soundfont is present:
   ```bash
   ls apps/web/public/soundfonts/acoustic_grand_piano-mp3.js || bun apps/web/scripts/fetch-soundfont.ts
   ```

2. Start dev:
   ```bash
   just dev-light
   ```

3. Sign in at `http://localhost:3000`, open a chat conversation.

4. Type a message such as: "I need help with bars 5-8 of my piece."

5. Confirm the teacher prescribes an `own_passage_loop` exercise card (the teacher uses `prescribe_exercise` tool with `own_passage_loop`).

6. Verify:
   - Score clip occupies the top of the card (large, white background)
   - Transport bar is below the score: play button, stop button, tempo slider at teacher's tempo_factor %
   - Pressing play → count-in clicks heard → loop begins → cursor sweeps through clip
   - Piano notes are audible and in sync with cursor
   - Adjusting tempo slider live rescales playback speed
   - `corpus_drill` cards (if any) show no transport, only text

7. If any element is missing or broken, diagnose and fix. This step is considered DONE when all 6 verification points are confirmed visually in the browser.

**Note:** If a real session is not convenient to set up, use the eval WebSocket override (`?eval=true&evalStudentId=...`) with a known piece that has a pending exercise record.

---

## Success Criteria

1. `bun run test` in `apps/web/` — all tests PASS.
2. `bun run test` in `apps/api/` — all existing tests PASS, new `tempoFactor` test PASS.
3. Manual `just dev-light` confirms: score as hero, transport plays, cursor animates, piano audible, metronome audible, tempo slider works.
4. `corpus_drill` stub test remains green (no regression).
5. No edits to `apps/evals/`, `model/`, or `apps/api/src/lib/types.ts`.
