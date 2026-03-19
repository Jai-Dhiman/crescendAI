# Practice Mode State Machine Design

**Date:** 2026-03-19
**Status:** Approved (brainstorm)
**Scope:** DO-side practice mode detection + mode-aware observation pacing + client throttle simplification

---

## Problem

The Durable Object practice session (`apps/api/src/practice/session.rs`) uses a flat 3-minute throttle (`OBSERVATION_THROTTLE_MS = 180_000`) between observations. This ignores what the student is actually doing -- warming up, drilling a passage, running through a piece, or winding down. The product vision calls for a session brain that detects practice mode and adapts observation pacing accordingly.

## Decisions

| Decision | Chosen | Rationale |
|----------|--------|-----------|
| Drilling detection | C: Bar range overlap + pitch bigram fallback | Bar overlap is richer when score context exists; pitch similarity provides universal coverage for unknown pieces |
| Ambiguous transitions | A: Explicit signal thresholds with Regular fallback | Conservative and legible. Regular mode preserves current 3-min behavior when signals are unclear |
| Throttle authority | A: DO is sole authority, simplify client to passthrough | Single source of truth. Client throttle was a stopgap for a dumb timer |
| Mode broadcast | A: Send mode_change over WebSocket | Cheap (3-5 messages per session), enables future UI, useful for debugging |
| Architecture | Approach 2: Separate practice_mode.rs module | Testable in isolation, clean separation from session orchestration |

## Data Model

### PracticeMode Enum

```rust
enum PracticeMode {
    Warming,   // Student settling in -- suppress observations
    Drilling,  // Repeating a passage -- 90s interval, comparative feedback
    Running,   // Playing through a piece -- 150s interval
    Winding,   // Cooling down -- suppress observations
    Regular,   // Ambiguous / fallback -- 180s interval (current behavior)
}
```

Sessions always start in `Warming`.

### ChunkSignal

Bridge between inference results and mode detection. Built from data already available in `process_inference_result`:

```rust
struct ChunkSignal {
    chunk_index: usize,
    timestamp_ms: u64,            // js_sys::Date::now()
    pitch_bigrams: Vec<(u8, u8)>, // consecutive pitch pairs from perf_notes
    bar_range: Option<(u32, u32)>, // from BarMap (score_follower::align_chunk), NOT ChunkAnalysis
    has_piece_match: bool,         // score_context is Some
    scores: [f64; 6],             // raw 6-dim scores for drilling comparison
}
```

`pitch_bigrams` are extracted from the `PerfNote` MIDI data that the HF endpoint already returns. No new inference cost. A 15s chunk with ~50 notes produces ~49 bigrams.

**`bar_range` source:** Derived from the `BarMap` returned by `score_follower::align_chunk`, not from `ChunkAnalysis.bar_range` (which is a formatted `Option<String>`). `process_inference_result` must extract `(min_bar, max_bar)` from the `BarMap` before it is consumed by analysis. The `BarMap` contains `u32` bar numbers; the min and max of its keys give the chunk's bar range.

**Transposition limitation:** Pitch bigrams use raw MIDI pitch values `(u8, u8)`. A passage practiced an octave higher would not match the original. This is acceptable for v1 -- pitch-class bigrams `(pitch % 12, pitch % 12)` can be added later if false negatives in drilling detection are observed.

### ModeDetector

```rust
struct ModeDetector {
    mode: PracticeMode,
    entered_at_ms: u64,                    // when current mode started
    chunk_count: usize,                    // total chunks seen
    recent_signals: VecDeque<ChunkSignal>, // sliding window (last 4 chunks = ~60s)
    last_chunk_at_ms: u64,                 // for silence gap detection
    drilling_passage: Option<DrillingPassage>,
}

struct DrillingPassage {
    bar_range: Option<(u32, u32)>,
    repetition_count: usize,
    first_scores: [f64; 6],  // scores from first repetition (for comparison)
}
// When entering Drilling from any mode, drilling_passage is initialized fresh
// from the current chunk's bar_range and scores. Previous drilling data is discarded.
```

The `recent_signals` window of 4 chunks (~60 seconds) provides enough history for mode detection without unbounded growth.

### ObservationPolicy

```rust
struct ObservationPolicy {
    suppress: bool,           // true = no observations at all
    min_interval_ms: u64,     // minimum time between observations
    comparative: bool,        // true = drilling mode, teacher should compare repetitions
}
```

## Mode Detection Logic

### Transition Rules

Each chunk, `ModeDetector::update()` evaluates transitions based on explicit signal thresholds. The current mode determines which transitions are checked.

**From Warming (initial state):**
- -> `Running`: piece match detected AND bar coverage progressing forward
- -> `Drilling`: repetition detected in last 3 chunks
- -> `Winding`: silence gap > 60s (via two-step transition, see Silence Gap Detection)
- -> `Regular`: 4+ chunks processed without triggering Running or Drilling
- Stays `Warming`: < 4 chunks AND no piece match AND no repetition

**From Running:**
- -> `Drilling`: repetition detected in last 3 chunks
- -> `Winding`: silence gap > 60s
- Stays `Running`: forward progress continues

**From Drilling:**
- -> `Running`: the two most recent chunks (positions `[n]` and `[n-1]` only, not the full window) show no repetition AND forward bar progress (or new pitch material). This prevents stale drilling signals in the window from keeping the mode locked.
- -> `Winding`: silence gap > 60s (via two-step transition)
- Stays `Drilling`: repetition continues, increment `repetition_count`

**From Regular (fallback):**
- -> `Running`: piece match + forward progress detected
- -> `Drilling`: repetition detected
- -> `Winding`: silence gap > 60s
- Stays `Regular`: signals still ambiguous

**From Winding:**
- -> `Running`: new chunk with piece match after long gap
- -> `Regular`: new chunk without piece match
- Stays `Winding`: silence continues

### Repetition Detection (Drilling Signal)

Two-layer approach:

**Layer 1 -- Bar range overlap (when score context available):**
```
overlap = intersection(chunk_n.bar_range, chunk_n-1.bar_range)
drilling = overlap >= 50% of either chunk's bar span
```
If 2 of the last 3 chunks have >= 50% bar overlap, classify as drilling.

**Layer 2 -- Pitch bigram similarity (fallback, no score context):**
```
dice = 2 * |bigrams_a intersect bigrams_b| / (|bigrams_a| + |bigrams_b|)
drilling = dice >= 0.6
```
Dice coefficient on pitch bigrams. Threshold 0.6 catches passages played at different tempos or with minor corrections.

### Silence Gap Detection (Two-Step Transition)

Silence means no `chunk_ready` arrives (client's silence gate skips upload). Detected by comparing `last_chunk_at_ms` to current timestamp on each new chunk arrival.

**The problem with naive detection:** If we simply transition to Winding on gap detection, the first chunk after a break would be evaluated under Winding's suppress policy -- exactly backwards, since that chunk represents the student resuming play.

**Solution: two-step transition within a single `update()` call.**

```
fn update(&mut self, signal: &ChunkSignal) -> Vec<ModeTransition> {
    let gap_ms = signal.timestamp_ms - self.last_chunk_at_ms;
    let mut transitions = Vec::new();

    // Step 1: If long silence detected, transition to Winding first
    if gap_ms > 60_000 && self.mode != PracticeMode::Winding {
        self.mode = PracticeMode::Winding;
        transitions.push(ModeTransition::new(PracticeMode::Winding));
    }

    // Step 2: Evaluate the current chunk's signals against the (possibly new) mode
    // If we just entered Winding, this immediately evaluates the resume transition
    // (Winding -> Running or Winding -> Regular)
    if let Some(t) = self.evaluate_transitions(signal) {
        transitions.push(t);
    }

    self.last_chunk_at_ms = signal.timestamp_ms;
    transitions
}
```

This ensures: (1) the gap is recorded as a Winding period, (2) the new chunk is evaluated under the correct resume logic, and (3) two `mode_change` WebSocket messages fire in sequence (Winding, then Running/Regular) so the client sees the full transition history.

**Note:** `update()` returns `Vec<ModeTransition>` (not `Option<PracticeMode>`) to support the two-step case. Most calls return 0 or 1 transitions; the silence gap case returns 2.

### Minimum Dwell Time

Prevents rapid oscillation between modes:

| Mode | Min dwell | Rationale |
|------|-----------|-----------|
| Warming | 0s | Can exit immediately if piece detected on first chunk |
| Drilling | 30s (2 chunks) | Need at least 2 repetitions to confirm |
| Running | 30s (2 chunks) | Brief forward progress might be a fluke |
| Regular | 15s (1 chunk) | Escape quickly if better signal arrives |
| Winding | 0s | Resume immediately when student plays again |

## Observation Pacing

### Per-Mode Parameters

| Mode | suppress | min_interval_ms | comparative | Behavior |
|------|----------|----------------|-------------|----------|
| Warming | true | N/A | false | Let student settle in |
| Drilling | false | 90,000 | true | Compare repetitions, more frequent |
| Running | false | 150,000 | false | Observe at natural moments |
| Regular | false | 180,000 | false | Current behavior, safe fallback |
| Winding | true | N/A | false | No new observations |

### Integration with session.rs

The current throttle check:

```rust
let should_generate = {
    let s = self.inner.borrow();
    stop_result.triggered
        && s.baselines.is_some()
        && self.throttle_allows(&s)
};
```

Becomes:

```rust
let policy = self.inner.borrow().mode_detector.observation_policy();
let should_generate = {
    let s = self.inner.borrow();
    !policy.suppress
        && stop_result.triggered
        && s.baselines.is_some()
        && self.mode_throttle_allows(&s, &policy)
};
```

The old `OBSERVATION_THROTTLE_MS` constant and `throttle_allows()` are replaced by `mode_throttle_allows()` which uses `policy.min_interval_ms`.

### Comparative Context for Drilling

When `policy.comparative` is true, drilling context is added to the teaching moment JSON:

```json
{
    "drilling_context": {
        "repetition_count": 4,
        "first_attempt_scores": { "dynamics": 0.45, "timing": 0.52, ... },
        "current_scores": { "dynamics": 0.62, "timing": 0.58, ... },
        "bar_range": [12, 16]
    }
}
```

Enables the teacher to say "your fourth pass at bars 12-16 has noticeably better dynamics."

### WebSocket Mode Broadcast

On mode transition, the DO sends:

```json
{
    "type": "mode_change",
    "mode": "drilling",
    "chunkIndex": 7,
    "context": {
        "bars": [12, 16],
        "repetition": 3
    }
}
```

Context varies by mode:
- **Drilling:** `bars` (optional), `repetition` count
- **Running:** `piece` (if matched)
- **Warming/Winding/Regular:** empty object

### Client-Side Throttle Simplification

`observation-throttle.ts` changes:
- **Remove:** `windowMs` logic, `canDeliver()` time check, `tick()` interval, `minChunksBeforeFirst` (the DO's Warming mode now handles early-session suppression -- keeping both would create dual authority)
- **Keep:** `queued`/`drain()` for reconnection resilience and session-end flush, `chunksReceived` counter (useful for UI display)
- **Add:** Handle `mode_change` messages in `usePracticeSession.ts` (store in state for future UI use)

### TypeScript Types for Mode Events

Added to `practice-api.ts`:

```typescript
type PracticeMode = "warming" | "drilling" | "running" | "winding" | "regular";

// ModeChangeContext is discriminated by the parent ModeChangeEvent.mode field
type ModeChangeContext =
    | { bars?: [number, number]; repetition: number } // drilling
    | { piece?: string }                               // running
    | Record<string, never>;                           // warming/winding/regular

interface ModeChangeEvent {
    type: "mode_change";
    mode: PracticeMode;
    chunkIndex: number;  // chunk that triggered the transition
    context: ModeChangeContext;
}
```

The `mode_change` event is added to the `PracticeWsEvent` discriminated union.

## File Changes

| File | Change |
|------|--------|
| `apps/api/src/practice/practice_mode.rs` | **NEW** -- PracticeMode enum, ChunkSignal, ModeDetector, ObservationPolicy, transition logic, repetition detection |
| `apps/api/src/practice/mod.rs` | Add `pub mod practice_mode;` |
| `apps/api/src/practice/session.rs` | Add `ModeDetector` to `SessionState`, build `ChunkSignal` in `process_inference_result`, call `mode_detector.update()`, replace `throttle_allows()` with mode-aware pacing, send `mode_change` WS messages, pass drilling context to `generate_observation` |
| `apps/web/src/lib/observation-throttle.ts` | Remove window-based throttle logic and minChunksBeforeFirst, keep queue/drain only |
| `apps/web/src/hooks/usePracticeSession.ts` | Handle `mode_change` WS message, store practice mode in hook state |
| `apps/web/src/lib/practice-api.ts` | Add `ModeChangeEvent` type, add `mode_change` to message union |

## Testing Strategy

### Unit Tests (`practice_mode.rs`)

Pure state machine with no IO dependencies. Test with synthetic `ChunkSignal` sequences:

**Transition tests (one per edge):**
1. Warming -> Running: piece match + ascending bar ranges
2. Warming -> Drilling: 3 chunks with >0.6 pitch bigram Dice similarity
3. Warming -> Regular: 4 chunks, no piece match, no repetition
4. Running -> Drilling: 3 chunks with overlapping bar ranges
5. Running -> Winding: chunk with 65s gap
6. Drilling -> Running: 2 chunks with new pitch material + forward progress
7. Winding -> Running/Regular: new chunk after long gap

**Edge cases:**
8. Min dwell enforcement: transition to Drilling, immediately send non-repeating chunk, assert stays Drilling
9. Empty MIDI: chunk with 0 perf_notes, no crash, no false repetition
10. Bar range fallback: chunk with bar_range: None falls back to pitch bigram similarity
11. Rapid mode cycling: Warming -> Running -> Drilling -> Running, verify clean transitions
12. Two-step silence transition: send chunk after 65s gap from Running. Assert `update()` returns two transitions: [Winding, Running/Regular]. Assert the chunk is evaluated under the resumed mode's policy, NOT Winding's suppress.

**Pacing tests:**
13. Suppression in Warming/Winding: `observation_policy().suppress == true`
14. Interval per mode: correct `min_interval_ms` for each mode
15. Comparative flag: `comparative == true` only in Drilling

### Integration Test

Use existing `eval_chunk` WebSocket message (dev-only path). Send a chunk sequence simulating warming -> running -> drilling and verify:
- `mode_change` WebSocket messages fire at correct points
- Observations respect mode-specific intervals
- Drilling observations include `drilling_context`

### Not Tested (Tuning Parameters)

Exact threshold values (0.6 Dice, 50% bar overlap, 60s silence) need calibration with real session data. Tests verify the mechanism, not the calibration.

## Known Limitations

**DO eviction:** Cloudflare DOs lose in-memory state on eviction (deploys, 30s inactivity without alarm). The `ModeDetector` would reset to `Warming`. This is acceptable -- sessions are short-lived, the alarm keeps the DO alive during active recording, and re-detection happens within 4 chunks (~60s). If eviction during active sessions becomes a real problem, persist `ModeDetector` state to DO storage (Approach 3 from the brainstorm).
