# Cold-Start (No-Baseline) Synthesis Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** A student with zero prior observations gets a synthesis grounded in what actually happened this session (real teaching moments + a real duration), instead of the teacher saying "I don't see any practice data."
**Spec:** docs/specs/2026-06-03-cold-start-synthesis-design.md
**Issue:** #24 (epic:agentic-teacher). Branch: `issue-24-cold-start-synthesis`. Blocks #22 Tasks 7/8.
**Style:** Follow `apps/api/TS_STYLE.md` for all TypeScript. Use Sonnet 4.6 for any subagents. No emojis. Explicit exception handling, no silent fallbacks.

---

## Task Groups

```
Group A (parallel): Task 1 (Rust), Task 4 (duration), Task 6 (framing guardrail)
Group B (sequential, depends on Task 1): Task 2 (bridge wrapper)
Group C (sequential, depends on Task 2): Task 3 (buildColdStartMoments)
Group D (sequential, depends on Task 3 + Task 6): Task 5 (DO wiring + thread referenceMode)
```

- **Task 1** touches only `teaching_moments.rs`. **Task 4** and **Task 5** both touch `session-brain.ts` -> Task 4 is in Group A but Task 5 (also `session-brain.ts`) is deferred to Group D; they must not run concurrently. Task 4 modifies/adds an exported function; Task 5 modifies `runSynthesisAndPersist` and `SynthesisInput`. To keep Group A parallel-safe, **Task 4 and Task 5 are in different groups** (A vs D), so no two concurrent tasks touch `session-brain.ts`.
- **Task 6** touches only `prompts.ts` / `prompts.test.ts`. **Task 5** also touches `teacher.ts` and `session-brain.ts` but not `prompts.ts`, so Task 6 (Group A) and Task 5 (Group D) never collide.

`[SHIPS INDEPENDENTLY]`: **Task 4** (duration fix) ships standalone value -- correct session length in every synthesis, baseline or not. Group A as a whole does not ship the cold-start fix (that needs Group D), but Task 4 alone fixes the 0-minute bug.

---

### Task 1: Rust `select_session_moments` — within-session-relative moment selection
**Group:** A (parallel with Task 4, Task 6)

**Behavior being verified:** Given a session's scored chunks and a within-session reference (per-dim mean), the function ranks chunks by how far below the reference their worst dimension is, returns up to `max` moments spanning distinct dimensions, falls back to a positive moment when all chunks sit at the reference, and returns empty when fewer than 2 chunks.
**Interface under test:** `select_session_moments(chunks: &[ScoredChunk], reference: &[f64; 6], max: usize) -> Vec<TeachingMoment>` (pure Rust).

**Files:**
- Modify: `apps/api/src/wasm/score-analysis/src/teaching_moments.rs`
- Test: `apps/api/src/wasm/score-analysis/src/teaching_moments.rs` (cargo `#[cfg(test)]` module, same file)

- [ ] **Step 1: Write the failing test**

Append these four tests inside the existing `mod tests` block in `teaching_moments.rs` (after `positive_moment_picks_highest_improvement`, before the closing `}` of the module):

```rust
    fn reference_mean(chunks: &[ScoredChunk]) -> [f64; 6] {
        let mut sums = [0.0f64; 6];
        for c in chunks {
            for i in 0..6 {
                sums[i] += c.scores[i];
            }
        }
        let n = chunks.len() as f64;
        let mut mean = [0.0f64; 6];
        for i in 0..6 {
            mean[i] = sums[i] / n;
        }
        mean
    }

    #[test]
    fn session_moments_rank_by_magnitude_vs_session_mean() {
        // Chunk 1 is far below the session mean on pedaling -> should rank first.
        let chunks = vec![
            ScoredChunk { chunk_index: 0, scores: [0.55, 0.50, 0.50, 0.54, 0.52, 0.50] },
            ScoredChunk { chunk_index: 1, scores: [0.55, 0.50, 0.10, 0.54, 0.52, 0.50] },
            ScoredChunk { chunk_index: 2, scores: [0.55, 0.50, 0.48, 0.54, 0.52, 0.50] },
        ];
        let reference = reference_mean(&chunks);
        let moments = select_session_moments(&chunks, &reference, 6);
        assert!(!moments.is_empty(), "should return at least one moment");
        assert_eq!(moments[0].dimension, "pedaling");
        assert_eq!(moments[0].chunk_index, 1);
        assert!(moments[0].deviation < 0.0, "weakest moment is below the session mean");
        assert!(!moments[0].is_positive);
        assert!(
            moments[0].reasoning.contains("this session"),
            "reasoning must be within-session phrased, got: {}",
            moments[0].reasoning
        );
    }

    #[test]
    fn session_moments_return_distinct_dimensions_up_to_max() {
        let chunks = vec![
            ScoredChunk { chunk_index: 0, scores: [0.10, 0.80, 0.80, 0.80, 0.80, 0.80] },
            ScoredChunk { chunk_index: 1, scores: [0.80, 0.10, 0.80, 0.80, 0.80, 0.80] },
            ScoredChunk { chunk_index: 2, scores: [0.80, 0.80, 0.10, 0.80, 0.80, 0.80] },
        ];
        let reference = reference_mean(&chunks);
        let moments = select_session_moments(&chunks, &reference, 2);
        assert_eq!(moments.len(), 2, "must respect max");
        assert_ne!(
            moments[0].dimension, moments[1].dimension,
            "moments must span distinct dimensions"
        );
    }

    #[test]
    fn session_moments_positive_fallback_when_all_at_mean() {
        // Every chunk identical -> deviation from mean is 0 everywhere -> positive fallback.
        let chunks = vec![
            ScoredChunk { chunk_index: 0, scores: [0.60, 0.60, 0.60, 0.60, 0.60, 0.60] },
            ScoredChunk { chunk_index: 1, scores: [0.60, 0.60, 0.60, 0.60, 0.60, 0.60] },
        ];
        let reference = reference_mean(&chunks);
        let moments = select_session_moments(&chunks, &reference, 6);
        assert_eq!(moments.len(), 1, "positive fallback returns a single moment");
        assert!(moments[0].is_positive);
    }

    #[test]
    fn session_moments_empty_when_too_few_chunks() {
        let chunks = vec![ScoredChunk { chunk_index: 0, scores: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3] }];
        let reference = reference_mean(&chunks);
        let moments = select_session_moments(&chunks, &reference, 6);
        assert!(moments.is_empty(), "fewer than 2 chunks -> empty");
    }
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test select_session_moments 2>&1 | tail -20
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test session_moments 2>&1 | tail -20
```
Expected: FAIL — `cannot find function 'select_session_moments' in this scope` (compile error).

- [ ] **Step 3: Implement the minimum to make the test pass**

Add this function to `teaching_moments.rs`, immediately after `select_teaching_moment` (before `select_positive_moment`). Do NOT modify `select_teaching_moment`.

```rust
/// Select up to `max` within-session teaching moments, ranked against a
/// caller-supplied within-session `reference` (typically the per-dimension
/// session mean) rather than a stored longitudinal baseline.
///
/// Algorithm:
/// 1. For each chunk, find its worst dimension vs `reference` (largest negative deviation).
/// 2. Keep chunks whose worst dimension is below the reference (deviation < 0).
/// 3. Rank by deviation magnitude (most-below-mean first).
/// 4. Walk ranked candidates, emitting at most one moment per distinct dimension, up to `max`.
/// 5. If no chunk is below the reference anywhere, return a single positive moment.
/// 6. If fewer than `MIN_CHUNKS` chunks, return an empty vec.
///
/// Reasoning strings are within-session phrased (reference-neutral), never longitudinal.
/// This function is additive; `select_teaching_moment` (the live path) is unaffected.
pub fn select_session_moments(
    chunks: &[ScoredChunk],
    reference: &[f64; 6],
    max: usize,
) -> Vec<TeachingMoment> {
    if chunks.len() < MIN_CHUNKS {
        return Vec::new();
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    for chunk in chunks {
        let (dim_idx, deviation) = max_negative_deviation(&chunk.scores, reference);
        if deviation >= 0.0 {
            continue;
        }
        candidates.push(Candidate {
            chunk_index: chunk.chunk_index,
            dim_idx,
            score: chunk.scores[dim_idx],
            baseline: reference[dim_idx],
            deviation,
        });
    }

    candidates.sort_by(|a, b| {
        a.deviation
            .partial_cmp(&b.deviation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut moments: Vec<TeachingMoment> = Vec::new();
    let mut used_dims: Vec<usize> = Vec::new();
    for candidate in &candidates {
        if moments.len() >= max {
            break;
        }
        if used_dims.contains(&candidate.dim_idx) {
            continue;
        }
        used_dims.push(candidate.dim_idx);
        moments.push(candidate.to_session_moment());
    }

    if moments.is_empty() {
        moments.push(select_positive_moment(chunks, reference));
    }

    moments
}
```

Add this method to the existing `impl Candidate` block (after `to_teaching_moment`):

```rust
    fn to_session_moment(&self) -> TeachingMoment {
        let dim_name = DIMS_6[self.dim_idx];
        let reasoning = format!(
            "{} was the weakest relative to the rest of this session: {:.2} vs session average {:.2} (deviation {:.2}).",
            dim_name, self.score, self.baseline, self.deviation,
        );
        TeachingMoment {
            chunk_index: self.chunk_index,
            dimension: dim_name.to_string(),
            score: self.score,
            baseline: self.baseline,
            deviation: self.deviation,
            reasoning,
            is_positive: false,
        }
    }
```

Note: `select_positive_moment` already produces reasoning beginning "No issues detected." which is reference-neutral; reuse it as-is for the all-at-mean fallback.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test session_moments 2>&1 | tail -20
```
Expected: PASS — all four `session_moments_*` tests green, and the pre-existing `select_teaching_moment` tests still green.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/wasm/score-analysis/src/teaching_moments.rs && git commit -m "feat(wasm): add select_session_moments for within-session moment selection"
```

---

### Task 2: WASM-bindgen export + TS bridge wrapper for `selectSessionMoments`
**Group:** B (depends on Task 1)

**Behavior being verified:** The TS bridge `selectSessionMoments` loads the real WASM module and returns real computed moments for a multi-chunk session.
**Interface under test:** `selectSessionMoments(chunks, reference, max)` in `wasm-bridge.ts`, exercising the real `pkg/` build under workerd.

**Files:**
- Modify: `apps/api/src/wasm/score-analysis/src/lib.rs`
- Modify: `apps/api/src/services/wasm-bridge.ts`
- Modify: `apps/api/src/services/wasm-bridge.workerd.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `wasm-bridge.workerd.test.ts` (inside the file, as a new top-level `describe`; the file already imports `describe, expect, it` from vitest and types from `./wasm-bridge`):

```ts
describe("selectSessionMoments (real WASM)", () => {
	it("returns real within-session moments for a multi-chunk session", async () => {
		const { selectSessionMoments } = await import("./wasm-bridge");
		const chunks = [
			{ chunk_index: 0, scores: [0.55, 0.5, 0.5, 0.54, 0.52, 0.5] as [number, number, number, number, number, number] },
			{ chunk_index: 1, scores: [0.55, 0.5, 0.1, 0.54, 0.52, 0.5] as [number, number, number, number, number, number] },
			{ chunk_index: 2, scores: [0.55, 0.5, 0.48, 0.54, 0.52, 0.5] as [number, number, number, number, number, number] },
		];
		const reference = {
			dynamics: 0.55,
			timing: 0.5,
			pedaling: 0.36,
			articulation: 0.54,
			phrasing: 0.52,
			interpretation: 0.5,
		};
		const moments = selectSessionMoments(chunks, reference, 6);
		expect(Array.isArray(moments)).toBe(true);
		expect(moments.length).toBeGreaterThan(0);
		expect(moments[0]?.dimension).toBe("pedaling");
		expect(moments[0]?.reasoning).toContain("this session");
	});

	it("returns an empty array when fewer than 2 chunks", async () => {
		const { selectSessionMoments } = await import("./wasm-bridge");
		const chunks = [
			{ chunk_index: 0, scores: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3] as [number, number, number, number, number, number] },
		];
		const reference = {
			dynamics: 0.3,
			timing: 0.3,
			pedaling: 0.3,
			articulation: 0.3,
			phrasing: 0.3,
			interpretation: 0.3,
		};
		expect(selectSessionMoments(chunks, reference, 6)).toEqual([]);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- src/services/wasm-bridge.workerd.test.ts 2>&1 | tail -25
```
Expected: FAIL — `selectSessionMoments is not a function` (the bridge does not yet export it; the `pkg/` is not yet rebuilt).

- [ ] **Step 3: Implement the minimum to make the test pass**

(a) Add the `#[wasm_bindgen]` export to `lib.rs`, immediately after the existing `select_teaching_moment` wasm export (after its closing `}`):

```rust
/// Select up to `max` within-session teaching moments from a session's scored chunks.
///
/// `chunks_js`: array of `{ chunk_index: number, scores: number[] }`
/// `reference_js`: `{ dynamics, timing, pedaling, articulation, phrasing, interpretation }`
///   (typically the per-dimension session mean)
/// `max`: maximum number of moments to return
///
/// Returns a serialized `Vec<TeachingMoment>` (empty array if fewer than 2 chunks).
#[wasm_bindgen]
pub fn select_session_moments(
    chunks_js: JsValue,
    reference_js: JsValue,
    max: usize,
) -> Result<JsValue, JsValue> {
    let chunks: Vec<types::ScoredChunk> =
        serde_wasm_bindgen::from_value(chunks_js).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let reference: types::StudentBaselines = serde_wasm_bindgen::from_value(reference_js)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result =
        teaching_moments::select_session_moments(&chunks, &reference.as_array(), max);
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}
```

(b) Add the bridge wrapper to `wasm-bridge.ts`, immediately after the existing `selectTeachingMoment` function (after its closing `}`):

```ts
/**
 * Select up to `max` within-session teaching moments, ranked against a
 * within-session reference (typically the per-dimension session mean).
 * Returns an empty array if fewer than 2 chunks are provided.
 */
export function selectSessionMoments(
	chunks: ScoredChunk[],
	reference: StudentBaselines,
	max: number,
): TeachingMoment[] {
	return scoreAnalysisModule.select_session_moments(
		chunks,
		reference,
		max,
	) as TeachingMoment[];
}
```

(c) Rebuild the WASM pkg so the workerd test sees the new export:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run build:wasm
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- src/services/wasm-bridge.workerd.test.ts 2>&1 | tail -25
```
Expected: PASS — both `selectSessionMoments` tests green; pre-existing workerd tests still green.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/wasm/score-analysis/src/lib.rs apps/api/src/services/wasm-bridge.ts apps/api/src/services/wasm-bridge.workerd.test.ts apps/api/src/wasm/score-analysis/pkg && git commit -m "feat(wasm): export select_session_moments via bridge"
```

---

### Task 3: `buildColdStartMoments` — session-mean + WASM + AccumulatedMoment mapping
**Group:** C (depends on Task 2)

**Behavior being verified:** Given a session's scored chunks, the function computes the per-dimension session mean, selects within-session moments via real WASM, and returns them as `AccumulatedMoment[]` spanning distinct dimensions; returns `[]` for a single-chunk session.
**Interface under test:** `buildColdStartMoments(scoredChunks, max)` exported from `session-brain.ts`, with real WASM under workerd.

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Modify: `apps/api/src/do/session-brain.unit.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `session-brain.unit.test.ts` (the file already imports `describe, expect, it` from vitest and `SessionAccumulator, type AccumulatedMoment` from `../services/accumulator`; add `buildColdStartMoments` to the existing `import { buildV6WsPayload } from "./session-brain"` line -> `import { buildV6WsPayload, buildColdStartMoments } from "./session-brain"`):

```ts
describe("buildColdStartMoments", () => {
	it("returns distinct-dimension moments for a multi-chunk first session", () => {
		const scoredChunks = [
			{ chunkIndex: 0, scores: [0.1, 0.8, 0.8, 0.8, 0.8, 0.8] },
			{ chunkIndex: 1, scores: [0.8, 0.1, 0.8, 0.8, 0.8, 0.8] },
			{ chunkIndex: 2, scores: [0.8, 0.8, 0.1, 0.8, 0.8, 0.8] },
		];
		const moments = buildColdStartMoments(scoredChunks, 2);
		expect(moments.length).toBe(2);
		expect(moments[0]?.dimension).not.toBe(moments[1]?.dimension);
		// AccumulatedMoment shape: baseline carries the session-mean reference.
		expect(typeof moments[0]?.baseline).toBe("number");
		expect(moments[0]?.analysisTier).toBe(3);
		expect(moments[0]?.barRange).toBeNull();
		expect(moments[0]?.llmAnalysis).toBeNull();
	});

	it("returns an empty array for a single-chunk session", () => {
		const scoredChunks = [{ chunkIndex: 0, scores: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3] }];
		expect(buildColdStartMoments(scoredChunks, 6)).toEqual([]);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- src/do/session-brain.unit.test.ts 2>&1 | tail -25
```
Expected: FAIL — `buildColdStartMoments is not exported` / `is not a function`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add this exported function to `session-brain.ts`, near the other exported helpers (e.g. immediately after `nextSynthesisAlarmDelayMs`, before the `export class SessionBrain` declaration).

Import status (verified against the current file): `wasm`, `Dimension`, `DIMS_6`, `ScoredChunk`, and `StudentBaselines` are ALREADY imported at the top of `session-brain.ts` -- do not re-import them. `AccumulatedMoment` is NOT imported; change the existing line `import { SessionAccumulator } from "../services/accumulator";` to `import { SessionAccumulator, type AccumulatedMoment } from "../services/accumulator";`. (`DIMS_6` is imported but not used by `buildColdStartMoments`; the function indexes the fixed 6-element arrays directly, so no `DIMS_6` reference is added.)

```ts
/**
 * Build within-session ("cold-start") teaching moments for a student with no
 * stored baselines. Computes the per-dimension session mean as the reference,
 * selects moments via the WASM bridge, and maps them to AccumulatedMoment[].
 *
 * Returns [] when there are too few chunks (the bridge requires >= 2).
 * Throws if the WASM module is unavailable; callers must catch.
 */
export function buildColdStartMoments(
	scoredChunks: { chunkIndex: number; scores: number[] }[],
	max: number,
): AccumulatedMoment[] {
	if (scoredChunks.length < 2) {
		return [];
	}

	// Per-dimension session mean (the within-session reference).
	const sums = [0, 0, 0, 0, 0, 0];
	for (const c of scoredChunks) {
		for (let i = 0; i < 6; i++) {
			sums[i] = (sums[i] ?? 0) + (c.scores[i] ?? 0);
		}
	}
	const n = scoredChunks.length;
	const meanArr = sums.map((s) => s / n);
	const reference: StudentBaselines = {
		dynamics: meanArr[0] ?? 0,
		timing: meanArr[1] ?? 0,
		pedaling: meanArr[2] ?? 0,
		articulation: meanArr[3] ?? 0,
		phrasing: meanArr[4] ?? 0,
		interpretation: meanArr[5] ?? 0,
	};

	const wasmChunks: ScoredChunk[] = scoredChunks.map((c) => ({
		chunk_index: c.chunkIndex,
		scores: c.scores as [number, number, number, number, number, number],
	}));

	const moments = wasm.selectSessionMoments(wasmChunks, reference, max);

	return moments.map((m) => ({
		chunkIndex: m.chunk_index,
		dimension: m.dimension as Dimension,
		score: m.score,
		baseline: m.baseline,
		deviation: m.deviation,
		isPositive: m.is_positive,
		reasoning: m.reasoning,
		barRange: null,
		analysisTier: 3,
		timestampMs: 0,
		llmAnalysis: null,
	}));
}
```

Note: `StudentBaselines`, `ScoredChunk` types are already imported in `session-brain.ts` from `../services/wasm-bridge` (used by the live path). Verify they are in the existing import list; if not, add them.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- src/do/session-brain.unit.test.ts 2>&1 | tail -25
```
Expected: PASS — both `buildColdStartMoments` tests green; pre-existing unit tests still green.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.ts apps/api/src/do/session-brain.unit.test.ts && git commit -m "feat(synthesis): add buildColdStartMoments helper"
```

---

### Task 4: `computeSessionDurationMs` — chunk-count-based duration
**Group:** A (parallel with Task 1, Task 6)

**Behavior being verified:** Session duration is computed from the scored-chunk count (15s per MuQ chunk), yielding a non-zero value for any session with chunks, independent of wall-clock timeline stamps.
**Interface under test:** `computeSessionDurationMs(scoredChunkCount)` exported from `session-brain.ts`.

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Modify: `apps/api/src/do/session-brain.unit.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `session-brain.unit.test.ts` (add `computeSessionDurationMs` to the `./session-brain` import line):

```ts
describe("computeSessionDurationMs", () => {
	it("derives duration from scored-chunk count at 15s per chunk", () => {
		expect(computeSessionDurationMs(10)).toBe(150000);
	});

	it("returns a positive duration for a short multi-chunk replay (not 0)", () => {
		expect(computeSessionDurationMs(8)).toBeGreaterThan(0);
	});

	it("returns 0 for a session with no scored chunks", () => {
		expect(computeSessionDurationMs(0)).toBe(0);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- src/do/session-brain.unit.test.ts 2>&1 | tail -25
```
Expected: FAIL — `computeSessionDurationMs is not exported` / `is not a function`.

- [ ] **Step 3: Implement the minimum to make the test pass**

(a) Add the exported function to `session-brain.ts`, near `nextSynthesisAlarmDelayMs`:

```ts
/** MuQ audio chunk length in milliseconds (15s per scored chunk). */
const MUQ_CHUNK_MS = 15000;

/**
 * Compute session duration from the number of scored chunks rather than
 * wall-clock timeline stamps. Eval replay stamps all timeline events within
 * the same millisecond, so the stamp-based calc rounded to 0 minutes.
 */
export function computeSessionDurationMs(scoredChunkCount: number): number {
	return scoredChunkCount * MUQ_CHUNK_MS;
}
```

(b) Replace the timeline-stamp calc in `runSynthesisAndPersist`. Find:

```ts
		const acc = SessionAccumulator.fromJSON(state.accumulator);
		const sessionDurationMs =
			acc.timeline.length > 0
				? (acc.timeline[acc.timeline.length - 1]?.timestampMs ?? 0) -
					(acc.timeline[0]?.timestampMs ?? 0)
				: 0;
```

Replace the `sessionDurationMs` assignment with:

```ts
		const acc = SessionAccumulator.fromJSON(state.accumulator);
		const sessionDurationMs = computeSessionDurationMs(state.scoredChunks.length);
```

> **Guidance (eval-mode scoredChunks emptiness):** `state.scoredChunks` can be empty at synthesis time in eval/dev mode, because the eval path persists scored chunks under per-chunk `eval_score:` storage keys "as fallback (robust against wrangler dev serialization gaps)" rather than relying on `state.scoredChunks` (see the existing `evalContext` sourcing in `runSynthesisAndPersist`, ~lines 1379-1390). If empty, `computeSessionDurationMs(0) === 0` reintroduces the 0-minute bug this task fixes. The implementation MUST mirror that existing `eval_score:` per-chunk-key fallback to source the chunk count when `state.scoredChunks` is empty in eval mode, so the duration computation does not silently no-op. This is guidance, not new scope: reuse the exact fallback already used for `evalContext`; do not invent a new mechanism.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- src/do/session-brain.unit.test.ts 2>&1 | tail -25
```
Expected: PASS — three `computeSessionDurationMs` tests green; pre-existing unit tests still green.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.ts apps/api/src/do/session-brain.unit.test.ts && git commit -m "fix(synthesis): derive session duration from scored-chunk count"
```

---

### Task 5: DO wiring — cold-start branch + thread `referenceMode`
**Group:** D (depends on Task 3 and Task 6)

**Behavior being verified:** When `synthesize` is called with `referenceMode: "within_session"`, the system blocks it sends to Anthropic carry the within-session `reference_mode` field and the no-history guardrail; with `referenceMode: null` they carry neither. (The DO null-baseline branch wiring -- populating `referenceMode` and `acc.teachingMoments` -- is verified by `buildColdStartMoments` in Task 3 plus diff review of the call site; see spec Verification Architecture.)
**Interface under test:** `synthesize(ctx, input)` in `teacher.ts`, observed through the `system` argument captured at the mocked `callAnthropic` HTTP boundary.

**Files:**
- Modify: `apps/api/src/services/teacher.ts`
- Modify: `apps/api/src/do/session-brain.ts`
- Create: `apps/api/src/services/teacher-synthesize-reference-mode.test.ts`

(A dedicated new test file avoids touching `teacher.test.ts`, whose top-level `vi.mock("./llm")` only stubs `callAnthropicStream`, not the non-stream `callAnthropic` that `synthesize` uses. A separate file gets its own clean mock of `callAnthropic`.)

- [ ] **Step 1: Write the failing test**

Create `apps/api/src/services/teacher-synthesize-reference-mode.test.ts`. `synthesize` calls `callAnthropic(env, request)` (from `./llm`) and `buildMemoryContext` (from `./memory`); mock both. A text-only Anthropic response (no `tool_use` blocks) avoids `processToolUse`. Capture the `system` array from the mock's call args.

```ts
import { describe, expect, it, vi } from "vitest";

const mockCallAnthropic = vi.fn();
vi.mock("./llm", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./llm")>();
	return { ...actual, callAnthropic: mockCallAnthropic };
});
vi.mock("./memory", () => ({
	buildMemoryContext: vi.fn().mockResolvedValue(""),
}));

import { synthesize, type SynthesisInput } from "./teacher";
import type { ServiceContext } from "../lib/types";

function baseInput(referenceMode: "within_session" | null): SynthesisInput {
	return {
		studentId: "stu_1",
		conversationId: null,
		sessionDurationMs: 120_000,
		practicePattern: "continuous_play",
		topMoments: [{ dimension: "timing", score: 0.3 }],
		drillingRecords: [],
		pieceMetadata: { composer: "Chopin", title: "Etude" },
		enrichedChunks: [],
		baselines: null,
		sessionHistory: [],
		pastDiagnoses: [],
		pieceId: null,
		referenceMode,
	};
}

async function captureSystemBlocks(
	referenceMode: "within_session" | null,
): Promise<Array<{ type: string; text?: string }>> {
	mockCallAnthropic.mockReset();
	mockCallAnthropic.mockResolvedValue({
		content: [{ type: "text", text: "Your session." }],
	});
	const ctx = { db: {}, env: {} } as unknown as ServiceContext;
	await synthesize(ctx, baseInput(referenceMode));
	const callArgs = mockCallAnthropic.mock.calls[0];
	const request = callArgs?.[1] as { system: Array<{ type: string; text?: string }> };
	return request.system;
}

describe("synthesize referenceMode threading", () => {
	it("forwards within_session reference mode into the teacher framing", async () => {
		const system = await captureSystemBlocks("within_session");
		// Match the angle-bracketed `<session_data>` tag, which is unique to the
		// framing block. UNIFIED_TEACHER_SYSTEM contains only the bare
		// `show_session_data` substring (prompts.ts:99), so a plain
		// `includes("session_data")` would resolve to systemBlocks[0] instead.
		const framing = system.find((b) => b.text?.includes("<session_data>"));
		expect(framing?.text).toContain("This is the student's first session");
		expect(framing?.text).toContain('"reference_mode"');
	});

	it("omits the first-session guardrail when referenceMode is null", async () => {
		const system = await captureSystemBlocks(null);
		// See note above: match `<session_data>` to select the framing block, not
		// UNIFIED_TEACHER_SYSTEM (which contains the bare `show_session_data`).
		const framing = system.find((b) => b.text?.includes("<session_data>"));
		expect(framing?.text).not.toContain("This is the student's first session");
		expect(framing?.text).not.toContain("reference_mode");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- src/services/teacher-synthesize-reference-mode.test.ts 2>&1 | tail -25
```
Expected: FAIL — the within_session framing does not contain `"reference_mode"` / the first-session instruction, because `synthesize` does not yet pass `referenceMode` to `buildSynthesisFraming` and `SynthesisInput` has no `referenceMode` field (TS compile error on `referenceMode` in `baseInput` until Step 3a lands).

- [ ] **Step 3: Implement the minimum to make the test pass**

(a) In `teacher.ts`, add `referenceMode` to `SynthesisInput`:

```ts
export interface SynthesisInput {
	studentId: string;
	conversationId: string | null;
	sessionDurationMs: number;
	practicePattern: string;
	topMoments: unknown[];
	drillingRecords: unknown[];
	pieceMetadata: { composer?: string; title?: string } | null;
	enrichedChunks: EnrichedChunk[];
	baselines: Record<string, number> | null;
	sessionHistory: SessionHistoryRecord[];
	pastDiagnoses: PastDiagnosisRecord[];
	pieceId?: string | null;
	referenceMode?: "within_session" | null;
}
```

(b) In `teacher.ts` `synthesize`, pass `referenceMode` as the new 8th argument to `buildSynthesisFraming`. Find the existing call:

```ts
		const synthesisFraming = buildSynthesisFraming(
			input.sessionDurationMs,
			input.practicePattern,
			input.topMoments,
			input.drillingRecords,
			input.pieceMetadata,
			memoryContext,
			composer,
		);
```

Replace with:

```ts
		const synthesisFraming = buildSynthesisFraming(
			input.sessionDurationMs,
			input.practicePattern,
			input.topMoments,
			input.drillingRecords,
			input.pieceMetadata,
			memoryContext,
			composer,
			input.referenceMode ?? null,
		);
```

(c) In `session-brain.ts` `runSynthesisAndPersist`, wire the cold-start branch. Immediately after the line `const acc = SessionAccumulator.fromJSON(state.accumulator);` and the (Task 4) `sessionDurationMs` line, add:

> **Guidance (eval-mode scoredChunks emptiness):** Same caveat as Task 4 — in eval/dev mode `state.scoredChunks` may be empty at synthesis time because scored chunks are persisted under per-chunk `eval_score:` storage keys (the existing `evalContext` fallback in `runSynthesisAndPersist`, ~lines 1379-1390). If empty, `buildColdStartMoments([], 6) === []` and the cold-start fix silently no-ops — the exact failure this plan targets. The implementation MUST source the scored chunks from the same `eval_score:` per-chunk-key fallback the eval branch already uses when `state.scoredChunks` is empty in eval mode, then map those into `buildColdStartMoments`. This is guidance, not new scope: reuse the existing fallback mechanism, do not add a new one. During `/build` verification, confirm the cold-start moments are non-empty in a real eval run (unit tests inject chunks directly and cannot catch an empty `state.scoredChunks` at the call site).

```ts
		// Cold-start: a student with no stored baselines never accumulated live
		// teaching moments (the live gate requires baselines !== null). Build
		// within-session moments now so acc.topMoments() has data to surface.
		let referenceMode: "within_session" | null = null;
		if (state.baselines === null) {
			referenceMode = "within_session";
			try {
				const coldStartMoments = buildColdStartMoments(
					state.scoredChunks.map((c) => ({
						chunkIndex: c.chunkIndex,
						scores: c.scores,
					})),
					6,
				);
				for (const m of coldStartMoments) {
					acc.accumulateMoment(m);
				}
			} catch (err) {
				const error = err as Error;
				console.log(
					JSON.stringify({
						level: "warn",
						message: "cold-start moment selection failed",
						sessionId: state.sessionId,
						error: error.message,
					}),
				);
			}
		}
```

(d) In `session-brain.ts`, add `referenceMode` to the `synthInput: SynthesisInput` object literal. Find the existing literal and add the field (alongside `pieceId`):

```ts
			pieceId: state.pieceIdentification?.pieceId ?? null,
			referenceMode,
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- src/services/teacher-synthesize-reference-mode.test.ts 2>&1 | tail -25
```
Expected: PASS — both `referenceMode threading` tests green. Also run the broader suite to confirm no regression: `bun run test -- src/services/teacher.test.ts src/do/session-brain.unit.test.ts 2>&1 | tail -15` (still green).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/teacher.ts apps/api/src/do/session-brain.ts apps/api/src/services/teacher-synthesize-reference-mode.test.ts && git commit -m "feat(synthesis): wire cold-start moments and referenceMode through synthesis"
```

---

### Task 6: `buildSynthesisFraming` reference-mode guardrail
**Group:** A (parallel with Task 1, Task 4)

**Behavior being verified:** When `buildSynthesisFraming` is called with `referenceMode === "within_session"`, the output contains a `reference_mode` field in `session_data` and the first-session no-history instruction; when called with `null` or omitted, it contains neither.
**Interface under test:** `buildSynthesisFraming(...)` with the new trailing `referenceMode` argument.

**Files:**
- Modify: `apps/api/src/services/prompts.ts`
- Modify: `apps/api/src/services/prompts.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `prompts.test.ts` (inside the existing `describe("buildSynthesisFraming", ...)` block, or as a new `describe`):

```ts
describe("buildSynthesisFraming referenceMode", () => {
	const piece = { title: "Etude", composer: "Chopin", skill_level: 3 };
	const moments = [{ dimension: "timing", score: 0.3 }];

	it("emits reference_mode and the first-session guardrail when within_session", () => {
		const out = buildSynthesisFraming(
			120_000,
			"continuous_play",
			moments,
			[],
			piece,
			"",
			"Chopin",
			"within_session",
		);
		expect(out).toContain('"reference_mode"');
		expect(out).toContain('"within_session"');
		expect(out).toContain("This is the student's first session");
		expect(out).toContain("do not reference past sessions");
	});

	it("omits reference_mode and the guardrail when referenceMode is null", () => {
		const out = buildSynthesisFraming(
			120_000,
			"continuous_play",
			moments,
			[],
			piece,
			"",
			"Chopin",
			null,
		);
		expect(out).not.toContain("reference_mode");
		expect(out).not.toContain("This is the student's first session");
	});

	it("omits reference_mode when the argument is not supplied (existing callers unchanged)", () => {
		const out = buildSynthesisFraming(
			120_000,
			"continuous_play",
			moments,
			[],
			piece,
			"",
			"Chopin",
		);
		expect(out).not.toContain("reference_mode");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- src/services/prompts.test.ts 2>&1 | tail -25
```
Expected: FAIL — output does not contain `"reference_mode"` or the first-session instruction (the param and conditional output do not exist yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `prompts.ts`, modify `buildSynthesisFraming`. Change the signature to add a trailing optional param and conditionally include the field + instruction. Find:

```ts
export function buildSynthesisFraming(
	sessionDurationMs: number,
	practicePattern: unknown,
	topMoments: unknown,
	drillingRecords: unknown,
	pieceMetadata: unknown,
	memoryContext: string,
	composer: string,
): string {
	const parts: string[] = [];

	const sessionData = {
		duration_minutes: Math.round(sessionDurationMs / 60000),
		practice_pattern: practicePattern,
		top_moments: topMoments,
		drilling_records: drillingRecords,
		piece: pieceMetadata,
	};
```

Replace with:

```ts
export function buildSynthesisFraming(
	sessionDurationMs: number,
	practicePattern: unknown,
	topMoments: unknown,
	drillingRecords: unknown,
	pieceMetadata: unknown,
	memoryContext: string,
	composer: string,
	referenceMode: "within_session" | null = null,
): string {
	const parts: string[] = [];

	const sessionData: Record<string, unknown> = {
		duration_minutes: Math.round(sessionDurationMs / 60000),
		practice_pattern: practicePattern,
		top_moments: topMoments,
		drilling_records: drillingRecords,
		piece: pieceMetadata,
	};
	if (referenceMode === "within_session") {
		sessionData["reference_mode"] = "within_session";
	}
```

Then, immediately before the final `<task>` push (find the existing `parts.push("");` that precedes the `<task>...` push), add the conditional instruction. Find:

```ts
	parts.push("");
	parts.push(
		"<task>Write <analysis>...</analysis> first as a reasoning scratchpad (this will be stripped before delivery). Then write your teacher response: 3-6 sentences, conversational, warm, specific. Use tools if they would add value. Do not mention scores or numbers. Do not list all dimensions -- focus on what matters most for this session.</task>",
	);

	return parts.join("\n");
```

Replace with:

```ts
	if (referenceMode === "within_session") {
		parts.push("");
		parts.push(
			"This is the student's first session -- describe only what happened within this session; do not reference past sessions or claim improvement over time.",
		);
	}

	parts.push("");
	parts.push(
		"<task>Write <analysis>...</analysis> first as a reasoning scratchpad (this will be stripped before delivery). Then write your teacher response: 3-6 sentences, conversational, warm, specific. Use tools if they would add value. Do not mention scores or numbers. Do not list all dimensions -- focus on what matters most for this session.</task>",
	);

	return parts.join("\n");
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test -- src/services/prompts.test.ts 2>&1 | tail -25
```
Expected: PASS — three `referenceMode` tests green; all pre-existing `buildSynthesisFraming` tests (which call with 7 args) still green because the 8th param defaults to `null`.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/prompts.ts apps/api/src/services/prompts.test.ts && git commit -m "feat(synthesis): add within_session reference-mode guardrail to framing"
```

---

## Plan Self-Review

- **Spec coverage:** Slice 1 -> Task 1 + Task 2; Slice 2 (duration) -> Task 4; Slice 3 (DO wiring) -> Task 3 (helper) + Task 5 (wiring); Slice 4 (guardrail) -> Task 6 + Task 5 (threading). All four spec slices covered.
- **Type consistency:** `select_session_moments(chunks, reference, max)` consistent across Rust (Task 1), lib.rs export (Task 2), bridge (Task 2). `selectSessionMoments` bridge name consistent in Task 2 and Task 3. `buildColdStartMoments(scoredChunks, max)` consistent Task 3/5. `computeSessionDurationMs(count)` consistent Task 4. `referenceMode: "within_session" | null` consistent across `SynthesisInput` (Task 5), `buildSynthesisFraming` 8th param (Task 6), and the DO literal (Task 5). `AccumulatedMoment` field names (chunkIndex, dimension, score, baseline, deviation, isPositive, reasoning, barRange, analysisTier, timestampMs, llmAnalysis) match `accumulator.ts`.
- **Group correctness:** Group A = Task 1 (teaching_moments.rs), Task 4 (session-brain.ts + unit.test.ts), Task 6 (prompts.ts + prompts.test.ts) -- no shared files. Task 3 and Task 5 both touch session-brain.ts and are in separate later groups (C, D), never concurrent with Task 4 or each other. Task 5 touches teacher.ts + teacher.test.ts + session-brain.ts -- no overlap with any concurrent task (Group D runs alone).
- **Vertical slice check:** Each task = one focused test block + one implementation + one commit. Task 1's test block contains four asserts of the same new function's behavior in one cycle (acceptable: single function, single implementation step, written then implemented together as one tracer bullet -- not horizontal scaffolding for other tasks).
- **Behavior test check:** All tests exercise public interfaces (`select_session_moments`, `selectSessionMoments`, `buildColdStartMoments`, `computeSessionDurationMs`, `buildSynthesisFraming`, `synthesize`). No internal-state assertions, no mocking of internal collaborators. The only mocks are `callAnthropic` (external HTTP boundary) and `buildMemoryContext` (external DB read), both in Task 5's dedicated test file -- the same boundaries `teacher.test.ts` already mocks for the v6 path. The real-WASM workerd tests exercise the actual runtime.
- **Task 5 test is self-contained:** Task 5 creates a new test file `teacher-synthesize-reference-mode.test.ts` with verbatim code, rather than appending to `teacher.test.ts` (whose `vi.mock("./llm")` only stubs `callAnthropicStream`, not the non-stream `callAnthropic` that the legacy `synthesize` uses). No placeholder helper; the build agent copies the code as given.

---

## Challenge Review

Reviewed against actual source: `teaching_moments.rs`, `types.rs`, `lib.rs`, `wasm-bridge.ts`, `session-brain.ts` (`runSynthesisAndPersist`, live gate, scoredChunks state), `session-brain.schema.ts`, `accumulator.ts` (`AccumulatedMoment`, `accumulateMoment`, `topMoments`), `teacher.ts` (`synthesize`, `SynthesisInput`), `prompts.ts` (`buildSynthesisFraming`, `UNIFIED_TEACHER_SYSTEM`), `vitest.config.ts`, `package.json`.

### CEO Pass

**Premise — VALID.** The problem statement is verified against code. `session-brain.ts:1267` gates live teaching-moment accumulation on `shouldAttemptMoment && state.baselines !== null`, so a cold-start (null-baseline) student genuinely accumulates zero moments, and `acc.topMoments()` is empty → empty `top_moments` in framing → "I don't see any practice data." The duration bug is also real: `runSynthesisAndPersist` derives `sessionDurationMs` from `acc.timeline` first/last `timestampMs`, which collapses to 0 under same-millisecond eval replay. Both are real production/eval defects, not proxy problems.

**Scope — TIGHT and correct.** 6 tasks, ~10 files, additive Rust function + 3 small TS seams + one threaded param. No new services or classes. The decision to add `select_session_moments` rather than overload `select_teaching_moment` is the surgical choice — the live path's contract (top-1, dedup, baseline-relative) genuinely differs from cold-start (up-to-max, distinct dims, mean-relative), and overloading would risk regressing the live path. Existing-pattern reuse is good: `computeSessionDurationMs` mirrors the `nextSynthesisAlarmDelayMs` seam; the bridge wrapper mirrors `selectTeachingMoment`.

[OBS] — Task 4 (duration) ships independent standalone value and is correctly flagged `[SHIPS INDEPENDENTLY]`. It is the lowest-risk, highest-certainty win; if the rest slips, Task 4 alone is worth landing.

[OBS] — v6 path does not receive the `reference_mode` guardrail (only `synthesize`/legacy calls `buildSynthesisFraming`; `synthesizeV6` does not). Spec explicitly scopes v6 out and notes it is gated off, so this is acceptable, but when v6 is eventually enabled the cold-start guardrail will silently not apply to it. Worth a follow-up note on #22.

**12-Month alignment — moves toward ideal.** Within-session-relative reference is the right primitive: it is the only honest reference a first-session student can have, and it generalizes cleanly to a future cohort/population reference (swap the `reference` arg) without touching the selection algorithm. No tech debt introduced; the `reference: &[f64; 6]` parameter is the correct abstraction boundary.

[QUESTION] — The spec's "Alternatives" reasoning (new function vs. parameter overload) is documented inline in the Design section, which satisfies the alternatives requirement. No separate concern.

### Engineering Pass

**Architecture — sound, matches the code.** Data flow traced end to end: `runSynthesisAndPersist` → (null-baseline) `buildColdStartMoments(state.scoredChunks)` → session mean → `wasm.selectSessionMoments` → `AccumulatedMoment[]` → `acc.accumulateMoment` → `acc.topMoments()` (called later in the same method) → `synthInput.topMoments` + `synthInput.referenceMode` → `synthesize` → `buildSynthesisFraming`. Ordering is correct: the cold-start branch is inserted immediately after `const acc = ...`, well before `acc.topMoments()` is read. All type contracts verified: `scoredChunks` element is `{ chunkIndex: number; scores: number[] }` (schema line 16-23), `AccumulatedMoment` fields match the Task 3 mapping exactly, `TeachingMoment` serializes snake_case (no serde rename) so `m.chunk_index`/`m.is_positive` are correct, `StudentBaselines::as_array()` exists, `select_positive_moment(chunks, &[f64;6])` signature matches the Task 1 call. No SQL/shell/prompt injection surface (reference is computed numerics; reasoning strings are server-generated). No N+1 or unbounded fan-out.

**Module depth — acceptable.** `select_session_moments` (Rust) and `buildColdStartMoments` (TS) are DEEP (one call hides ranking/marshalling/mapping). `selectSessionMoments` bridge and `computeSessionDurationMs` are SHALLOW but justified by existing convention (bridge = sole `pkg/` importer; duration seam = testable constant), consistent with the spec's own depth audit.

[BLOCKER — RESOLVED loop 1] The Task 5 test helper now matches on the angle-bracketed `<session_data>` tag (unique to the framing block; `UNIFIED_TEACHER_SYSTEM` uses only the bare `show_session_data`, verified at prompts.ts:99 vs prompts.ts:128). The original finding is retained below for context.

[BLOCKER] (confidence: 9/10) — **Task 5's test selects the wrong system block and therefore does not test what it claims.** The helper does `const framing = system.find((b) => b.text?.includes("session_data"))`. `Array.find` returns the FIRST match. `systemBlocks[0]` is `UNIFIED_TEACHER_SYSTEM`, which contains the substring `session_data` via `show_session_data` at `prompts.ts:99`. So `framing` resolves to `UNIFIED_TEACHER_SYSTEM`, NOT the framing block (`systemBlocks[1]`) that actually carries `reference_mode` and the first-session instruction. Consequence: the `within_session` test (`expect(framing?.text).toContain("This is the student's first session")` / `toContain('"reference_mode"')`) will FAIL even when Tasks 5/6 are implemented correctly (the assertions live on the wrong block), and the `null` test (`not.toContain(...)`) is a FALSE GREEN that would pass regardless of implementation. Fix before executing: select the framing block unambiguously — e.g. `system[1]` directly, or match on a string unique to the framing block such as `b.text?.includes("<session_data>")` (the framing uses the angle-bracketed tag at `prompts.ts:128`; `UNIFIED_TEACHER_SYSTEM` uses the bare `show_session_data`). Verify the chosen discriminator does NOT also appear in `UNIFIED_TEACHER_SYSTEM`.

[RISK] (confidence: 7/10) — **Task 4 and Task 5 both read `state.scoredChunks` in `runSynthesisAndPersist`, the very field the eval path treats as unreliable.** Line 1379-1390 reads scored chunks from per-chunk `eval_score:` storage keys "as fallback (robust against wrangler dev serialization gaps)," implying `state.scoredChunks` can be empty at synthesis time in eval/dev. If it is empty, `computeSessionDurationMs(0) === 0` (duration bug returns) AND `buildColdStartMoments([...], 6) === []` (no cold-start moments — the exact failure this plan fixes). The spec discovered this bug in eval replay, so the eval path is in scope. In production, the live handler writes `state.scoredChunks` to DO state on each chunk, so it should be populated — but this is unverified for the synthesis-time read. Fallback to monitor: if cold-start moments come back empty during `/build` verification, source the chunk count/scores from the same `eval_score:` keys the eval branch already uses, mirroring lines 1383-1390. At minimum, the build agent should confirm `state.scoredChunks` is non-empty at the synthesis call site in a real run, not just in unit tests (which inject chunks directly and cannot catch this).

**Code quality — clean.** Error handling is explicit and non-silent: the Task 5 cold-start branch wraps `buildColdStartMoments` in try/catch logging structured JSON and degrading to empty moments (matches the existing live-path convention around `wasm.selectTeachingMoment`). No catch-all `catch (e)` swallowing — every catch logs. The `?? 0` guards in the mean computation handle malformed score arrays. The 8th param on `buildSynthesisFraming` defaults to `null`, so the single existing caller (`teacher.ts:652`) is byte-compatible.

**Test philosophy — behavior-through-interface, no internal mocking.** All tests exercise public functions. The only mocks (Task 5: `callAnthropic`, `buildMemoryContext`) are external boundaries (HTTP, DB), the same ones `teacher.test.ts` already mocks. Workerd tests exercise real WASM. No shape-only tests, no private-method assertions. (The Task 5 block-selection defect above is a correctness bug in the test, not a philosophy violation.)

**Vertical slice — compliant.** Each task = one test block + one implementation + one commit. Task 1's four asserts target one new function in one tracer-bullet cycle (acceptable per the plan's own justification — not horizontal scaffolding for later tasks). Watch-it-fail steps are present and the predicted failure modes are accurate (compile error for missing Rust fn; "not a function" for missing exports; missing-substring for the framing param).

**Test coverage — adequate, one gap.** `select_session_moments`: ranking [TESTED], distinct-dims-up-to-max [TESTED], positive fallback [TESTED], empty-on-<2 [TESTED]. `computeSessionDurationMs`: positive/zero [TESTED]. `buildColdStartMoments`: multi-chunk distinct dims + single-chunk empty [TESTED]. `buildSynthesisFraming`: within_session/null/omitted [TESTED]. Gap: the DO null-baseline wiring (the cold-start branch in `runSynthesisAndPersist`) is verified only by `/review` diff inspection, not by an automated test — the spec acknowledges this (DO boot requires workers bindings + storage fixtures). Acceptable given the helper is fully unit-tested through its public interface; the residual risk is the [RISK] above (scoredChunks emptiness at the call site), which a unit test structurally cannot catch.

**Failure modes — covered.** WASM-unavailable → catch + log + degrade (synthesis still runs on duration + practice_pattern). `<2` chunks → empty, honest. `baselines !== null` → branch never entered (regression-lock verified against line 1267). No corrupt-state path: moments are accumulated in-memory before synthesis; a failure leaves `acc.teachingMoments` empty, not partial. Zero silent failures.

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| Live gate (line 1267) blocks cold-start moment accumulation | SAFE | Verified: `shouldAttemptMoment && state.baselines !== null` |
| `acc.topMoments()` empty for first session | SAFE | Verified: filters `teachingMoments`, which is empty when gate never fires |
| `AccumulatedMoment` field names match Task 3 mapping | SAFE | Verified field-by-field in `accumulator.ts:4-16` |
| `TeachingMoment` serializes snake_case (`chunk_index`, `is_positive`) | SAFE | Verified: no serde rename in `types.rs:161-170`; TS bridge type matches |
| `StudentBaselines::as_array()` exists | SAFE | Verified `types.rs:144` |
| `wasm`, `ScoredChunk`, `StudentBaselines`, `Dimension`, `DIMS_6` already imported in session-brain.ts | SAFE | Verified imports (lines 9-10, 42-46) |
| `AccumulatedMoment` NOT yet imported in session-brain.ts | SAFE | Verified: line 12 imports only `SessionAccumulator` |
| `accumulateMoment` pushes to `teachingMoments` | SAFE | Verified `accumulator.ts:48-50` |
| `scoredChunks` element shape `{ chunkIndex, scores: number[] }` | SAFE | Verified schema `session-brain.schema.ts:16-23` |
| Only one caller of `buildSynthesisFraming` (legacy path) | SAFE | Verified: sole call at `teacher.ts:652`; v6 does not call it |
| `synthesize` calls `callAnthropic(env, request)` with `system` as arg[1] | SAFE | Verified `teacher.ts`; mock captures `calls[0][1]` correctly |
| Task 5 test `find(includes("<session_data>"))` selects the framing block | SAFE (fixed loop 1) | `<session_data>` tag is unique to the framing block (prompts.ts:128); `UNIFIED_TEACHER_SYSTEM` has only bare `show_session_data` (prompts.ts:99). |
| `state.scoredChunks` populated at synthesis time | VALIDATE | Eval path treats it as unreliable (per-chunk-key fallback at line 1379-1390); prod likely OK but unverified at the synthesis read. See RISK. |
| `select_positive_moment(chunks, &reference)` reused for all-at-mean fallback | SAFE | Verified signature `types::&[f64;6]`; returns single `is_positive` moment |
| `bun run test -- <file>` filters by path under single workers pool | SAFE | `vitest.config.ts` is a single `defineWorkersConfig`; all tests run under workerd with real WASM |
| `build:wasm` script exists to rebuild `pkg/` | SAFE | Verified `package.json:15` |

### Summary

[BLOCKER] count: 1
[RISK]    count: 1
[QUESTION] count: 0 (alternatives reasoning is documented inline in the spec)

VERDICT: NEEDS_REWORK — Task 5's `captureSystemBlocks` helper selects `UNIFIED_TEACHER_SYSTEM` instead of the framing block because `UNIFIED_TEACHER_SYSTEM` contains the substring `session_data` (`show_session_data`, prompts.ts:99) and `Array.find` returns the first match. As written, the within_session assertions fail-for-the-wrong-reason and the null assertions are false-greens, so Task 5's test does not verify referenceMode threading at all. Fix the block discriminator (use `system[1]` or match on the angle-bracketed `<session_data>` tag, confirmed absent from `UNIFIED_TEACHER_SYSTEM`) before execution. Also monitor the `state.scoredChunks`-emptiness RISK (Tasks 4 and 5) during `/build` — if it is empty at synthesis time in eval, both fixes silently no-op; mirror the existing `eval_score:` per-chunk-key fallback if so.
