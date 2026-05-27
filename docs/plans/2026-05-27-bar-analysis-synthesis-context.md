# Bar-Analysis-in-Synthesis Context Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task) where tasks are marked parallel. Do NOT start execution until `/challenge` returns `VERDICT: PROCEED`. Task Group 0 (baseline reproduction) is a gate — every other task is blocked until it passes.

**Goal:** The teacher LLM receives structured per-bar facts (velocity, onset deviation, articulation ratio, pedal events, reference comparisons) for the chunk that triggered each top teaching moment, so its feedback can be bar-specific and actionable.

**Spec:** `docs/specs/2026-05-27-bar-analysis-synthesis-context-design.md`

**Style:** Follow `apps/api/TS_STYLE.md` for all TypeScript. Python style: type-annotated, `from __future__ import annotations`, `uv` for dependency operations. No emojis. Explicit exception handling (no silent fallbacks). One sentence per file responsibility.

---

## Task Groups

- **Group 0 (gate, sequential, must pass before anything else):** Task 0
- **Group A (parallel — disjoint files):** Task 1 (bar-analysis-facts module + test), Task 5 (piece_score_map module + test), Task 7 (bar_analysis_local — Tier-2 stats), Task 9 (bar_analysis_local — Tier-1 deltas), Task 10 (bar_analysis_local — facts filter)
- **Group B (sequential, depends on A):** Task 2 (accumulator type change), then Task 3 (session-brain wiring), then Task 4 (prompts.ts emit bar_analysis), then Task 11 (run_eval.py inject bar_analysis), then Task 6 (synthesis.ts persist), then Task 8 (worst-chunk selector wired)
- **Group C (validation, depends on B):** Task 12 (full-cache eval run + lift verification)

Group A tasks are parallel because each touches a unique new file. Task 7, 9, and 10 all touch `bar_analysis_local.py` so they must be **sequential within Group A** even though their group label is A — the build agent must dispatch 1 and 5 in parallel, then dispatch 7 → 9 → 10 sequentially.

Decouple flags:

- Group 0 is verification-only; no shipping value.
- Tasks 1–6 together ship the production TS surface and could in principle deploy without the eval mirror, but the success criterion is measured in the eval. **No `[SHIPS INDEPENDENTLY]` until Group C passes.**
- Tasks 7–11 are the eval mirror; they cannot ship without 1–6 because the prompt shape change is shared.

---

## Task 0: Reproduce the locked ASCF baseline on this branch (Group 0 gate)

**Group:** 0 (sequential, blocks everything)

**Behavior being verified:** The teaching-knowledge eval, run from the current branch with no feature code, reproduces the locked baseline ASCF outcome 1.387 ± 0.15 on a 10-recording smoke sample.

**Interface under test:** `apps/evals/teaching_knowledge/run_eval.py` CLI.

**Files:**
- Create: `apps/evals/teaching_knowledge/results/baseline_smoke.jsonl` (artifact, not committed; add to .gitignore if not already)
- Modify: none

- [ ] **Step 1: Run the eval smoke on the current branch**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m teaching_knowledge.run_eval --limit 10 --out results/baseline_smoke.jsonl
```

Expected: command completes without error and writes 10 JSONL rows.

- [ ] **Step 2: Compute ASCF outcome mean from the smoke result**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -c "
import json
rows = [json.loads(l) for l in open('results/baseline_smoke.jsonl')]
ascf = [r['judge_dimensions']['audible_specific_corrective_feedback']['outcome']
        for r in rows if 'judge_dimensions' in r and r.get('error') in (None, '')]
print(f'n={len(ascf)} mean={sum(ascf)/len(ascf):.3f}')
"
```

Expected: prints `n=<at least 8> mean=<value between 1.237 and 1.537>`. The locked baseline is 1.387; the ±0.15 band tolerates the n=10 vs n=513 sampling noise. The exact JSON path may differ — if the `judge_dimensions` shape is not `{dim: {outcome: float}}` in the smoke output, the build agent must inspect one row with `cat results/baseline_smoke.jsonl | head -1 | python -m json.tool` and adjust the extraction path. Reproducibility is the gate; the exact extraction expression is not.

- [ ] **Step 3: If mean is outside [1.237, 1.537], STOP and report**

Do not proceed with any other task. The plan's success criterion (≥ 0.3 lift over 1.387) is undefined if the baseline cannot be reproduced. Report the observed value back to the user.

- [ ] **Step 4: Commit nothing**

This task produces no committed artifact. It is a gate. Record the observed mean in the PR description when the feature ships.

---

## Task 1: `buildBarAnalysisFacts` returns null when `analysis.dimensions` is empty

**Group:** A (parallel with Tasks 5)

**Behavior being verified:** When the WASM analyzer produces no dimension records (e.g., Tier-3 fallback path), the public function returns `null` rather than a partial object.

**Interface under test:** `buildBarAnalysisFacts(analysis, scoresArray, baselines, selectedDimension)` from `apps/api/src/services/bar-analysis-facts.ts`.

**Files:**
- Create: `apps/api/src/services/bar-analysis-facts.ts`
- Create: `apps/api/src/services/bar-analysis-facts.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/bar-analysis-facts.test.ts
import { describe, expect, it } from "vitest";
import { buildBarAnalysisFacts } from "./bar-analysis-facts";
import type { ChunkAnalysis } from "./wasm-bridge";

const baselines = {
	dynamics: 0.5,
	timing: 0.5,
	pedaling: 0.5,
	articulation: 0.5,
	phrasing: 0.5,
	interpretation: 0.5,
} as const;

describe("buildBarAnalysisFacts", () => {
	it("returns null when analysis.dimensions is empty", () => {
		const analysis: ChunkAnalysis = {
			tier: 3,
			bar_range: null,
			dimensions: [],
		};
		const scores: [number, number, number, number, number, number] = [
			0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		];
		const result = buildBarAnalysisFacts(
			analysis,
			scores,
			baselines,
			"timing",
		);
		expect(result).toBeNull();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test bar-analysis-facts
```

Expected: FAIL — `Cannot find module './bar-analysis-facts'`.

- [ ] **Step 3: Implement the minimum**

```typescript
// apps/api/src/services/bar-analysis-facts.ts
import type { Dimension } from "../lib/dims";
import type { ChunkAnalysis, DimensionAnalysis } from "./wasm-bridge";

export interface StudentBaselines {
	dynamics: number;
	timing: number;
	pedaling: number;
	articulation: number;
	phrasing: number;
	interpretation: number;
}

export interface BarAnalysisFacts {
	tier: number;
	bar_range: string | null;
	selected: DimensionAnalysis;
	correlated: DimensionAnalysis[];
}

export function buildBarAnalysisFacts(
	analysis: ChunkAnalysis,
	scoresArray: [number, number, number, number, number, number],
	baselines: StudentBaselines,
	selectedDimension: Dimension,
): BarAnalysisFacts | null {
	if (analysis.dimensions.length === 0) {
		return null;
	}
	// Unreachable in this task; future tasks extend.
	throw new Error("buildBarAnalysisFacts: non-empty case not yet implemented");
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test bar-analysis-facts
```

Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/bar-analysis-facts.ts apps/api/src/services/bar-analysis-facts.test.ts && git commit -m "feat(api): scaffold buildBarAnalysisFacts with null-on-empty case"
```

---

## Task 1b: `buildBarAnalysisFacts` returns the selected dimension and excludes it from correlated

**Group:** A (sequential after Task 1; same file)

**Behavior being verified:** Given a populated `ChunkAnalysis` and a `selectedDimension`, the result's `selected` is the `DimensionAnalysis` matching that dimension name, and `correlated` does not contain it.

**Interface under test:** Same.

**Files:**
- Modify: `apps/api/src/services/bar-analysis-facts.ts`
- Modify: `apps/api/src/services/bar-analysis-facts.test.ts`

- [ ] **Step 1: Append the failing test**

```typescript
it("selected is the matching dimension; correlated excludes it", () => {
	const analysis: ChunkAnalysis = {
		tier: 1,
		bar_range: "4-7",
		dimensions: [
			{ dimension: "dynamics", analysis: "dyn-text" },
			{ dimension: "timing", analysis: "tim-text" },
			{ dimension: "pedaling", analysis: "ped-text" },
			{ dimension: "articulation", analysis: "art-text" },
			{ dimension: "phrasing", analysis: "phr-text" },
			{ dimension: "interpretation", analysis: "int-text" },
		],
	};
	// All scores equal baseline → no correlated entries clear the 0.15 threshold.
	const scores: [number, number, number, number, number, number] = [
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
	];
	const result = buildBarAnalysisFacts(analysis, scores, baselines, "timing");
	expect(result).not.toBeNull();
	expect(result?.selected.dimension).toBe("timing");
	expect(result?.selected.analysis).toBe("tim-text");
	expect(result?.correlated.map((d) => d.dimension)).not.toContain("timing");
	expect(result?.correlated).toHaveLength(0);
	expect(result?.tier).toBe(1);
	expect(result?.bar_range).toBe("4-7");
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test bar-analysis-facts
```

Expected: FAIL — `buildBarAnalysisFacts: non-empty case not yet implemented`.

- [ ] **Step 3: Implement**

Replace the `throw` body with:

```typescript
// apps/api/src/services/bar-analysis-facts.ts (replace the throw in buildBarAnalysisFacts)
const DIM_ORDER: Dimension[] = [
	"dynamics",
	"timing",
	"pedaling",
	"articulation",
	"phrasing",
	"interpretation",
];

// ... inside buildBarAnalysisFacts, after the empty-check:
const selected = analysis.dimensions.find(
	(d) => d.dimension === selectedDimension,
);
if (selected === undefined) {
	return null;
}

const deviations = DIM_ORDER.map((dim, i) => ({
	dim,
	dev: Math.abs((scoresArray[i] ?? 0) - baselines[dim]),
}));

const correlated = analysis.dimensions
	.filter((d) => d.dimension !== selectedDimension)
	.map((d) => {
		const entry = deviations.find((e) => e.dim === d.dimension);
		return { d, dev: entry?.dev ?? 0 };
	})
	.filter((x) => x.dev >= 0.15)
	.sort((a, b) => b.dev - a.dev)
	.slice(0, 2)
	.map((x) => x.d);

return {
	tier: analysis.tier,
	bar_range: analysis.bar_range,
	selected,
	correlated,
};
```

Note: `DIM_ORDER` is hoisted to module scope (above the function). The build agent must place it there, not inside the function.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test bar-analysis-facts
```

Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/bar-analysis-facts.ts apps/api/src/services/bar-analysis-facts.test.ts && git commit -m "feat(api): buildBarAnalysisFacts returns selected and excludes from correlated"
```

---

## Task 1c: `buildBarAnalysisFacts` includes correlated dimensions ranked by absolute deviation, capped at 2

**Group:** A (sequential after Task 1b; same file)

**Behavior being verified:** When three or more non-selected dimensions clear the 0.15 deviation threshold, only the top 2 by absolute deviation are returned in `correlated`, in descending order.

**Interface under test:** Same.

**Files:**
- Modify: `apps/api/src/services/bar-analysis-facts.test.ts` (append test only)

- [ ] **Step 1: Append the failing test**

```typescript
it("correlated is capped at 2 and sorted by absolute deviation descending", () => {
	const analysis: ChunkAnalysis = {
		tier: 1,
		bar_range: "12-14",
		dimensions: [
			{ dimension: "dynamics", analysis: "dyn" },
			{ dimension: "timing", analysis: "tim" },
			{ dimension: "pedaling", analysis: "ped" },
			{ dimension: "articulation", analysis: "art" },
			{ dimension: "phrasing", analysis: "phr" },
			{ dimension: "interpretation", analysis: "int" },
		],
	};
	// deviations from baseline 0.5: dyn=+0.30, tim=selected, ped=-0.20, art=+0.40, phr=+0.05, int=-0.25
	// Non-selected ≥0.15: dyn(0.30), ped(0.20), art(0.40), int(0.25). Top 2: art(0.40), dyn(0.30).
	const scores: [number, number, number, number, number, number] = [
		0.80, 0.50, 0.30, 0.90, 0.55, 0.25,
	];
	const result = buildBarAnalysisFacts(analysis, scores, baselines, "timing");
	expect(result?.correlated.map((d) => d.dimension)).toEqual([
		"articulation",
		"dynamics",
	]);
});

it("dimensions below the 0.15 threshold are excluded from correlated", () => {
	const analysis: ChunkAnalysis = {
		tier: 1,
		bar_range: "1-3",
		dimensions: [
			{ dimension: "dynamics", analysis: "dyn" },
			{ dimension: "timing", analysis: "tim" },
		],
	};
	// dyn deviation 0.14 < 0.15 → excluded.
	const scores: [number, number, number, number, number, number] = [
		0.64, 0.50, 0.50, 0.50, 0.50, 0.50,
	];
	const result = buildBarAnalysisFacts(analysis, scores, baselines, "timing");
	expect(result?.correlated).toHaveLength(0);
});
```

- [ ] **Step 2: Run test — verify both FAIL or PASS deterministically**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test bar-analysis-facts
```

Expected: With the Task 1b implementation, these two cases may already PASS. If both pass without changes, that means the implementation in 1b already covers them — that is acceptable here because the implementation was written with the spec's full filter logic, not just the 1b test. **However** the build agent MUST first comment out the filter+sort+slice block, re-run to confirm FAIL, then restore — proving the tests bind to behavior, not shape. If the agent skips this watch-it-fail step, the cap and sort logic is untested.

- [ ] **Step 3: Confirm implementation unchanged (no code edit needed) and tests pass**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test bar-analysis-facts
```

Expected: PASS (4 passed total).

- [ ] **Step 4: Commit**

```bash
git add apps/api/src/services/bar-analysis-facts.test.ts && git commit -m "test(api): cover correlated cap and threshold for buildBarAnalysisFacts"
```

---

## Task 2: `AccumulatedMoment.llmAnalysis` accepts `BarAnalysisFacts | null`

**Group:** B (sequential, depends on Task 1c)

**Behavior being verified:** Building an `AccumulatedMoment` with a `BarAnalysisFacts` object as `llmAnalysis` and round-tripping through `SessionAccumulator.toJSON()` / `fromJSON()` preserves the structured value.

**Interface under test:** `SessionAccumulator` from `apps/api/src/services/accumulator.ts`.

**Files:**
- Modify: `apps/api/src/services/accumulator.ts`
- Create: `apps/api/src/services/accumulator.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/accumulator.test.ts
import { describe, expect, it } from "vitest";
import { SessionAccumulator, type AccumulatedMoment } from "./accumulator";
import type { BarAnalysisFacts } from "./bar-analysis-facts";

describe("AccumulatedMoment.llmAnalysis", () => {
	it("round-trips a BarAnalysisFacts object through toJSON/fromJSON", () => {
		const facts: BarAnalysisFacts = {
			tier: 1,
			bar_range: "4-7",
			selected: { dimension: "timing", analysis: "rushing 45ms" },
			correlated: [{ dimension: "articulation", analysis: "clipped" }],
		};
		const moment: AccumulatedMoment = {
			chunkIndex: 0,
			dimension: "timing",
			score: 0.3,
			baseline: 0.5,
			deviation: -0.2,
			isPositive: false,
			reasoning: "below baseline",
			barRange: [4, 7],
			analysisTier: 1,
			timestampMs: 1,
			llmAnalysis: facts,
		};
		const acc = new SessionAccumulator();
		acc.accumulateMoment(moment);
		const restored = SessionAccumulator.fromJSON(JSON.parse(JSON.stringify(acc.toJSON())));
		expect(restored.teachingMoments[0]?.llmAnalysis).toEqual(facts);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test accumulator
```

Expected: FAIL — TypeScript compile error: `Type 'BarAnalysisFacts' is not assignable to type 'string | null'`.

- [ ] **Step 3: Implement — change the type**

In `apps/api/src/services/accumulator.ts`, replace:
```typescript
llmAnalysis: string | null;
```
with:
```typescript
import type { BarAnalysisFacts } from "./bar-analysis-facts";
// ...
llmAnalysis: BarAnalysisFacts | null;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test accumulator
```

Expected: PASS.

- [ ] **Step 5: Verify the rest of the API still typechecks**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run typecheck 2>&1 | head -40
```

Expected: Any new type error MUST be in `session-brain.ts` or `synthesis.ts` (callers we will fix in Tasks 3 and 6). Specifically, expect errors like `Type 'null' is fine but the old code passed a string somewhere — find each and convert it`. If errors appear in unrelated files, the type change broke something the plan did not anticipate; stop and surface.

Then make the **minimum** compile-only fix in `synthesis.ts` and `session-brain.ts`:
- In `synthesis.ts::persistAccumulatedMoments`, change `const reasoningTrace = moment.llmAnalysis ?? moment.reasoning;` to `const reasoningTrace = moment.llmAnalysis !== null ? JSON.stringify(moment.llmAnalysis) : moment.reasoning;`. **This is the only line modified in synthesis.ts in this task.** Behavior is covered by Task 6.
- In `session-brain.ts`, no change needed — both `llmAnalysis: null` assignments still typecheck.

- [ ] **Step 6: Re-run typecheck and tests**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run typecheck && bun run test
```

Expected: PASS, no type errors.

- [ ] **Step 7: Commit**

```bash
git add apps/api/src/services/accumulator.ts apps/api/src/services/accumulator.test.ts apps/api/src/services/synthesis.ts && git commit -m "feat(api): AccumulatedMoment.llmAnalysis accepts BarAnalysisFacts"
```

---

## Task 3: `session-brain.ts` wires `buildBarAnalysisFacts` at both accumulation sites

**Group:** B (sequential, depends on Task 2)

**Behavior being verified:** When `handleProcessChunk` (and separately `handleEvalChunk`) produces an `accMoment`, that moment's `llmAnalysis` is populated from the `ChunkAnalysis` for the chunk that triggered the moment.

**Interface under test:** The `SessionBrain` Durable Object handlers. The test verifies behavior via the accumulator's `teachingMoments` array after driving the DO with synthetic input.

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Modify: `apps/api/src/do/session-brain.unit.test.ts` (or add a new sibling test if the existing file doesn't have a suitable harness)

**Implementation strategy for the build agent:**

`session-brain.ts` lines 580–680 compute `analysis` in scope local to the WASM branches but only extract `chunkAnalysisTier` and `chunkBarRange`. The build agent must hoist a new `let chunkAnalysis: ChunkAnalysis | null = null;` alongside the existing `let chunkAnalysisTier = 3;` and `let chunkBarRange: [number, number] | null = null;`, then assign `chunkAnalysis = analysis;` immediately after each `wasm.analyzeTier1(...)` and `wasm.analyzeTier2(...)` call (4 sites in `handleProcessChunk`, 4 sites in `handleEvalChunk`).

Then around line 963 (`accMoment` build site in `handleProcessChunk`), replace `llmAnalysis: null` with:

```typescript
llmAnalysis:
	chunkAnalysis !== null
		? buildBarAnalysisFacts(chunkAnalysis, scoresArray, baselines, momentDim)
		: null,
```

And around line 1240 (`handleEvalChunk` site), do the same — but note that `baselines` in `handleEvalChunk` is the locally-constructed `baselines: StudentBaselines` object from line 1210. The variable names match — confirm before the edit.

Add the import at the top of `session-brain.ts`:
```typescript
import { buildBarAnalysisFacts } from "../services/bar-analysis-facts";
```

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/do/session-brain.unit.test.ts (append)
import { describe, expect, it } from "vitest";
import { SessionAccumulator, type AccumulatedMoment } from "../services/accumulator";
import { buildBarAnalysisFacts } from "../services/bar-analysis-facts";
import type { ChunkAnalysis } from "../services/wasm-bridge";

describe("session-brain accumulation contract", () => {
	it("buildBarAnalysisFacts result is the shape session-brain must attach to accMoment.llmAnalysis", () => {
		const analysis: ChunkAnalysis = {
			tier: 1,
			bar_range: "4-7",
			dimensions: [
				{ dimension: "timing", analysis: "rushing" },
				{ dimension: "dynamics", analysis: "soft" },
			],
		};
		const baselines = {
			dynamics: 0.5, timing: 0.5, pedaling: 0.5,
			articulation: 0.5, phrasing: 0.5, interpretation: 0.5,
		};
		const scores: [number, number, number, number, number, number] = [
			0.20, 0.30, 0.50, 0.50, 0.50, 0.50,
		];
		const facts = buildBarAnalysisFacts(analysis, scores, baselines, "timing");

		const moment: AccumulatedMoment = {
			chunkIndex: 0,
			dimension: "timing",
			score: 0.30,
			baseline: 0.50,
			deviation: -0.20,
			isPositive: false,
			reasoning: "rushing",
			barRange: [4, 7],
			analysisTier: 1,
			timestampMs: 0,
			llmAnalysis: facts,
		};
		const acc = new SessionAccumulator();
		acc.accumulateMoment(moment);
		const top = acc.topMoments();
		expect(top[0]?.llmAnalysis).not.toBeNull();
		expect(top[0]?.llmAnalysis?.selected.dimension).toBe("timing");
		expect(top[0]?.llmAnalysis?.correlated.map((d) => d.dimension)).toContain("dynamics");
	});
});
```

This test pins the *contract* the session-brain edit must satisfy. The session-brain code itself is hard to unit-test in isolation (DO context, WASM dependency); the integration is verified via Task 12's full eval run. **Justify in the PR description:** session-brain.ts is exercised end-to-end by the eval harness; isolated DO unit tests would require mocking the WASM bridge, which violates the "no mocking internal collaborators" rule.

- [ ] **Step 2: Run test — verify it FAILS (or PASSES if Task 2 already covered the round-trip)**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test session-brain.unit
```

If this test passes immediately, that is acceptable — it is a contract test that the session-brain edit must continue to satisfy. The failure case the *session-brain edit itself* must address is **typecheck**: confirm step 3 below produces a real edit.

- [ ] **Step 3: Edit `session-brain.ts`**

Make the four edits described in the implementation strategy above:
1. Add the import.
2. Hoist `let chunkAnalysis: ChunkAnalysis | null = null;` in `handleProcessChunk` (near line 585).
3. Assign `chunkAnalysis = analysis;` after each of the 4 `analyzeTier1`/`analyzeTier2` calls in `handleProcessChunk` (around lines 620, 635, 655 — and any I missed; the build agent must read the function and locate every `wasm.analyzeTier(1|2)(...)` call inside it).
4. Replace `llmAnalysis: null,` at line ~974 with the conditional `buildBarAnalysisFacts(...)` call.
5. Repeat 2–4 in `handleEvalChunk` (hoist near line 1062, assignments around lines 1081/1091/1102, accMoment edit at line ~1240).

Also add `import type { ChunkAnalysis } from "../services/wasm-bridge";` if not already present.

- [ ] **Step 4: Run typecheck and tests**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run typecheck && bun run test
```

Expected: PASS. No type errors. All existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.ts apps/api/src/do/session-brain.unit.test.ts && git commit -m "feat(api): wire buildBarAnalysisFacts at both DO accumulation sites"
```

---

## Task 4: `buildSynthesisFraming` emits `bar_analysis` per top moment when present

**Group:** B (sequential, depends on Task 3)

**Behavior being verified:** A top-moment object whose serialized form includes a `bar_analysis` field is preserved verbatim in the JSON inside `<session_data>` of the synthesis user message. A top moment without that field omits it (no `"bar_analysis": null` noise).

**Interface under test:** `buildSynthesisFraming` from `apps/api/src/services/prompts.ts`.

**Files:**
- Modify: `apps/api/src/services/prompts.test.ts`
- Modify: `apps/api/src/services/prompts.ts` (only if the test reveals current behavior is wrong; otherwise no change because `topMoments: unknown` already passes through whatever shape the caller hands in)

**Important context for the build agent:** `buildSynthesisFraming`'s `topMoments` parameter type is `unknown` and it is fed straight into `JSON.stringify(sessionData)`. The function may already pass `bar_analysis` through unchanged. The test verifies the public contract; if it passes without code changes, that is the correct outcome and the task is "lock the contract via a test."

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/prompts.test.ts (append in the buildSynthesisFraming describe block)
it("includes bar_analysis field in session_data JSON when provided on a top moment", () => {
	const facts = {
		tier: 1,
		bar_range: "4-7",
		selected: { dimension: "timing", analysis: "rushing 45ms" },
		correlated: [{ dimension: "articulation", analysis: "clipped" }],
	};
	const topMoments = [
		{ dimension: "timing", score: 0.3, bar_analysis: facts },
	];
	const out = buildSynthesisFraming(
		300_000,
		"continuous_play",
		topMoments,
		[],
		{ title: "Etude", composer: "Chopin", skill_level: 3 },
		"",
		"Chopin",
	);
	expect(out).toContain('"bar_analysis"');
	expect(out).toContain('"rushing 45ms"');
	expect(out).toContain('"bar_range": "4-7"');
});

it("omits bar_analysis from a top moment that did not include it", () => {
	const topMoments = [{ dimension: "timing", score: 0.3 }];
	const out = buildSynthesisFraming(
		300_000,
		"continuous_play",
		topMoments,
		[],
		{ title: "Etude", composer: "Chopin", skill_level: 3 },
		"",
		"Chopin",
	);
	expect(out).not.toContain("bar_analysis");
});
```

- [ ] **Step 2: Run test — verify behavior**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test prompts
```

If both tests pass without any change to `prompts.ts`, the contract is already correct and the test now locks it. If the first fails (e.g., because `bar_analysis` is stripped somewhere), the build agent must read the current `buildSynthesisFraming` body and locate the strip — there is no expected stripping in the current code as of 2026-05-27, so this should pass on first try.

- [ ] **Step 3: If both tests pass with no code change, that is the implementation**

If a code change is needed, it is: make `sessionData.top_moments` propagate `bar_analysis`. The current `top_moments: topMoments` assignment already does this because `topMoments` is `unknown`.

- [ ] **Step 4: Run all prompts tests**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test prompts
```

Expected: PASS (all original tests + 2 new).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/prompts.test.ts apps/api/src/services/prompts.ts && git commit -m "test(api): lock bar_analysis pass-through in buildSynthesisFraming"
```

---

## Task 5: `piece_score_map.get_score_path_for_piece` returns the curated score JSON path

**Group:** A (parallel with Task 1; disjoint file)

**Behavior being verified:** For each `piece_slug` listed in `PIECE_SCORE_MAP`, calling `get_score_path_for_piece(slug)` returns an existing `Path` under `model/data/scores/`. For unmapped slugs (e.g. `clair_de_lune`, `schumann_traumerei`), it returns `None`.

**Interface under test:** `get_score_path_for_piece` from `apps/evals/teaching_knowledge/piece_score_map.py`.

**Files:**
- Create: `apps/evals/teaching_knowledge/piece_score_map.py`
- Create: `apps/evals/teaching_knowledge/tests/__init__.py` (empty, if not already present — check first; Bash `ls`)
- Create: `apps/evals/teaching_knowledge/tests/test_piece_score_map.py`

**Mapping procedure (the build agent populates the table from `ls` output — the plan deliberately does NOT pre-fill any specific score file):**

The piece_slugs to consider are every directory under `model/data/evals/skill_eval/`:

```bash
ls /Users/jdhiman/Documents/crescendai/model/data/evals/skill_eval/
```

For each piece_slug, the build agent must run a targeted `ls | grep` against `model/data/scores/` to determine whether a matching score JSON exists. Musical knowledge is required to pair slugs with filenames (e.g., "Moonlight" = Beethoven Op.27 No.2 = Sonata No. 14; "Pathétique" = Op.13 = Sonata No. 8). **Verify presence with `ls` before accepting any pairing.**

Candidate grep patterns for each known slug (the build agent runs these and ONLY maps the slug if the grep returns the expected file):

| piece_slug | candidate grep pattern | mapping decision |
|------------|------------------------|------------------|
| `bach_prelude_c_wtc1` | `bach\.prelude\.bwv_846` | needs verify |
| `chopin_ballade_1` | `chopin\.ballades\.1\.json` | needs verify |
| `moonlight_sonata_mvt1` | `beethoven\.piano_sonatas\.14-` | needs verify (only map if a `14-1` mvt-1 file is present; if only `14-3` exists, leave unmapped — mvt 1 file is not in the scores dir) |
| `pathetique_mvt2` | `beethoven\.piano_sonatas\.8-2` | needs verify |
| `fantaisie_impromptu` | `chopin.*fantaisie\|chopin\.fantasie` | needs verify |
| `fur_elise` | `bagatelle\|fur_elise\|woo_59` | needs verify |
| `rachmaninoff_prelude_csm` | `rachmaninoff\.preludes` | needs verify (slug is ambiguous — only map if you can confirm WHICH prelude "csm" refers to; otherwise leave unmapped) |
| `chopin_etude_op10no4` | `chopin\.etudes_op_10\.4` | needs verify |
| `chopin_waltz_csm` | `chopin.*waltz` | needs verify |
| `liszt_liebestraum_3` | `liszt.*liebestraum` | needs verify |
| `nocturne_op9no2` | `chopin.*nocturne.*9.*2` | needs verify |
| `debussy_arabesque_1` | `debussy.*arabesque` | needs verify |
| `bach_invention_1` | `bach.*invention` | needs verify |
| `clair_de_lune` | `debussy.*clair\|suite_bergamasque` | needs verify |
| `mozart_k545_mvt1` | `mozart.*k.*545\|mozart\.piano_sonatas\.16` | needs verify |
| `schumann_traumerei` | `schumann.*traumerei\|schumann.*kinderszenen` | needs verify |
| `ensemble_4fold` | (not a piece — likely meta) | unmapped |

**Rules for the build agent:**
1. Run `ls model/data/scores/ | grep -iE "<pattern>"` for each row.
2. If the grep returns exactly one file that musically matches the slug, add `slug -> filename` to `PIECE_SCORE_MAP`.
3. If the grep returns no files, or returns files that do not musically match (e.g., grep for Moonlight mvt 1 returns only `14-3.json` which is mvt 3), leave the slug OUT of `PIECE_SCORE_MAP` — it will return `None` from `get_score_path_for_piece`.
4. If the grep returns multiple plausible files, leave the slug OUT and surface to the user — do NOT guess.
5. The final committed `PIECE_SCORE_MAP` dict must contain ONLY rows the build agent personally verified by running `ls`. Empty mappings are acceptable; wrong mappings are not.

- [ ] **Step 1: Build the mapping table from `ls` output**

For each piece_slug in the table above, run the candidate grep against `model/data/scores/`. Example:

```bash
ls /Users/jdhiman/Documents/crescendai/model/data/scores/ | grep -iE "beethoven\.piano_sonatas\.14-"
ls /Users/jdhiman/Documents/crescendai/model/data/scores/ | grep -iE "beethoven\.piano_sonatas\.8-2"
ls /Users/jdhiman/Documents/crescendai/model/data/scores/ | grep -iE "chopin\.ballades\.1\.json"
# ...repeat for every piece_slug
```

Apply the five rules above. Record which slugs map to which files in scratch (e.g., a comment block in the implementation file); leave unmapped slugs absent from the dict. The committed `PIECE_SCORE_MAP` is your `ls`-verified result, not a copy of the plan's candidate table.

- [ ] **Step 2: Write the failing test**

```python
# apps/evals/teaching_knowledge/tests/test_piece_score_map.py
from __future__ import annotations

from pathlib import Path

import pytest

from teaching_knowledge.piece_score_map import get_score_path_for_piece


from teaching_knowledge.piece_score_map import PIECE_SCORE_MAP


@pytest.mark.parametrize("slug", sorted(PIECE_SCORE_MAP.keys()))
def test_mapped_pieces_return_existing_path(slug: str) -> None:
    result = get_score_path_for_piece(slug)
    assert result is not None
    assert isinstance(result, Path)
    assert result.exists()
    assert result.suffix == ".json"


def test_unmapped_piece_returns_none() -> None:
    assert get_score_path_for_piece("clair_de_lune") is None
    assert get_score_path_for_piece("schumann_traumerei") is None
    assert get_score_path_for_piece("nonexistent_piece") is None
```

- [ ] **Step 3: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/tests/test_piece_score_map.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'teaching_knowledge.piece_score_map'`.

- [ ] **Step 4: Implement**

```python
# apps/evals/teaching_knowledge/piece_score_map.py
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCORES_DIR = REPO_ROOT / "model" / "data" / "scores"

# Hand-curated mapping. Each entry MUST be verified by `ls model/data/scores/`
# before being added. Pieces not in this table return None.
# Build agent: populate this dict from the Step 1 ls-verification results.
# Do NOT add any row without confirming the file exists.
PIECE_SCORE_MAP: dict[str, str] = {
    # Example shape (only commit entries you personally verified):
    # "bach_prelude_c_wtc1": "bach.prelude.bwv_846.json",
}


def get_score_path_for_piece(piece_slug: str) -> Path | None:
    filename = PIECE_SCORE_MAP.get(piece_slug)
    if filename is None:
        return None
    path = SCORES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"piece_score_map: {piece_slug} -> {filename} does not exist under {SCORES_DIR}"
        )
    return path
```

The `raise` on missing file is intentional (the user prefers explicit exception handling over silent fallbacks). The mapping is data the developer curated; if a listed file is missing, the curation is wrong and the build agent must surface it, not paper over it.

- [ ] **Step 5: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/tests/test_piece_score_map.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/evals/teaching_knowledge/piece_score_map.py apps/evals/teaching_knowledge/tests/test_piece_score_map.py && git commit -m "feat(evals): add hand-curated piece_slug -> score JSON map"
```

---

## Task 6: `persistAccumulatedMoments` writes JSON-stringified `llmAnalysis` to `reasoning_trace`

**Group:** B (sequential, depends on Task 5 — though file-disjoint, the Group B chain serializes for clarity)

**Behavior being verified:** When a moment's `llmAnalysis` is a non-null `BarAnalysisFacts`, the persisted `observations.reasoning_trace` is the JSON-stringified facts. When `llmAnalysis` is null, `reasoning_trace` falls back to `moment.reasoning`.

**Interface under test:** `persistAccumulatedMoments` from `apps/api/src/services/synthesis.ts`. Verified through the DB insert by spying on the `db.insert(...).values(...)` call chain via a fake `Db` that records its `.values(...)` argument. This is borderline — but inserting into an in-memory Postgres is overkill for this assertion, so a fake DB that records calls is the simplest public-interface test. The fake is NOT a mock of an internal collaborator; it is a stand-in for the external Postgres dependency, which is the documented test pattern in the existing `apps/api/src/services/synthesis.test.ts`.

**Files:**
- Modify: `apps/api/src/services/synthesis.ts`
- Modify: `apps/api/src/services/synthesis.test.ts`

- [ ] **Step 1: Read `synthesis.test.ts` to match its existing fake-DB pattern**

```bash
cat /Users/jdhiman/Documents/crescendai/apps/api/src/services/synthesis.test.ts
```

Adopt whatever fake-DB pattern is already in place; do not invent a new one.

- [ ] **Step 2: Write the failing test**

```typescript
// apps/api/src/services/synthesis.test.ts (append)
import { describe, expect, it } from "vitest";
import { persistAccumulatedMoments } from "./synthesis";
import type { AccumulatedMoment } from "./accumulator";
import type { BarAnalysisFacts } from "./bar-analysis-facts";

describe("persistAccumulatedMoments reasoning_trace", () => {
	function makeFakeDb() {
		const inserts: unknown[] = [];
		const db = {
			insert() {
				return {
					values(v: unknown) {
						inserts.push(v);
						return { onConflictDoNothing: async () => {} };
					},
				};
			},
		};
		return { db, inserts };
	}

	const baseMoment: AccumulatedMoment = {
		chunkIndex: 0,
		dimension: "timing",
		score: 0.3,
		baseline: 0.5,
		deviation: -0.2,
		isPositive: false,
		reasoning: "rushing ahead of beat",
		barRange: [4, 7],
		analysisTier: 1,
		timestampMs: 1,
		llmAnalysis: null,
	};

	it("falls back to moment.reasoning when llmAnalysis is null", async () => {
		const { db, inserts } = makeFakeDb();
		await persistAccumulatedMoments(db as never, "s1", "sess1", null, [baseMoment]);
		expect((inserts[0] as { reasoningTrace: string }).reasoningTrace).toBe(
			"rushing ahead of beat",
		);
	});

	it("writes JSON-stringified facts when llmAnalysis is non-null", async () => {
		const facts: BarAnalysisFacts = {
			tier: 1,
			bar_range: "4-7",
			selected: { dimension: "timing", analysis: "rushing 45ms" },
			correlated: [],
		};
		const { db, inserts } = makeFakeDb();
		await persistAccumulatedMoments(db as never, "s1", "sess1", null, [
			{ ...baseMoment, llmAnalysis: facts },
		]);
		const trace = (inserts[0] as { reasoningTrace: string }).reasoningTrace;
		expect(JSON.parse(trace)).toEqual(facts);
	});
});
```

- [ ] **Step 3: Run test — verify it FAILS (or already PASSES from the Task 2 minimum fix)**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test synthesis
```

If the test passes immediately, Task 2's minimum compile-only edit was actually correct behaviorally — that is fine, the test locks it. To prove the test binds to behavior, the build agent must temporarily revert the line to `const reasoningTrace = moment.llmAnalysis ?? moment.reasoning;`, run the test, observe the failure, then restore the fix.

- [ ] **Step 4: Verify implementation in `synthesis.ts::persistAccumulatedMoments`**

The line must be:
```typescript
const reasoningTrace =
	moment.llmAnalysis !== null ? JSON.stringify(moment.llmAnalysis) : moment.reasoning;
```

- [ ] **Step 5: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api && bun run test synthesis
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/api/src/services/synthesis.ts apps/api/src/services/synthesis.test.ts && git commit -m "feat(api): persist bar-analysis facts as JSON in reasoning_trace"
```

---

## Task 7: `bar_analysis_local.compute_tier2_dimensions` returns six DimensionAnalysis-shaped dicts from a chunk

**Group:** A (parallel with Tasks 1, 5 — but must be the FIRST of the 7/9/10 trio because 9 and 10 share the file)

**Behavior being verified:** Given a single chunk's `midi_notes` and `pedal_events`, the function returns six dicts (one per dimension) with non-empty `analysis` strings that summarize the chunk's statistics.

**Interface under test:** `compute_tier2_dimensions(midi_notes, pedal_events)` from `apps/evals/teaching_knowledge/bar_analysis_local.py`.

**Files:**
- Create: `apps/evals/teaching_knowledge/bar_analysis_local.py`
- Create: `apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py`

- [ ] **Step 1: Read the Rust Tier-2 implementation to mirror its statistics**

```bash
grep -n "fn analyze_tier2\|fn velocity_stats\|fn ioi_stats\|fn articulation\|fn pedal_count" /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis/src/bar_analysis.rs | head -20
```

Use the actual formulas from `bar_analysis.rs` — do not invent new ones. The Python port must reproduce the Rust numbers within rounding error.

- [ ] **Step 2: Write the failing test**

```python
# apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py
from __future__ import annotations

from teaching_knowledge.bar_analysis_local import compute_tier2_dimensions


def test_returns_six_dimensions_with_analysis_strings() -> None:
    midi_notes = [
        {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
        {"pitch": 62, "onset": 0.5, "offset": 1.0, "velocity": 70},
        {"pitch": 64, "onset": 1.0, "offset": 1.5, "velocity": 90},
    ]
    pedal_events = [{"time": 0.2, "value": 127}, {"time": 0.8, "value": 0}]
    result = compute_tier2_dimensions(midi_notes, pedal_events)
    assert len(result) == 6
    dims = [r["dimension"] for r in result]
    assert dims == [
        "dynamics",
        "timing",
        "pedaling",
        "articulation",
        "phrasing",
        "interpretation",
    ]
    for r in result:
        assert isinstance(r["analysis"], str)
        assert len(r["analysis"]) > 0


def test_dynamics_string_mentions_velocity() -> None:
    midi_notes = [
        {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
        {"pitch": 62, "onset": 0.5, "offset": 1.0, "velocity": 100},
    ]
    result = compute_tier2_dimensions(midi_notes, [])
    dynamics = next(r for r in result if r["dimension"] == "dynamics")
    assert "velocity" in dynamics["analysis"].lower()
```

- [ ] **Step 3: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/tests/test_bar_analysis_local.py -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

```python
# apps/evals/teaching_knowledge/bar_analysis_local.py
"""Python port of bar_analysis.rs Tier-1/2 statistics for the eval harness.

Mirrors apps/api/src/wasm/score-analysis/src/bar_analysis.rs. The Rust file is
the source of truth; if its formulas change, this file must change to match.
"""

from __future__ import annotations

from statistics import mean, stdev
from typing import Any

DIMS_ORDER: list[str] = [
    "dynamics",
    "timing",
    "pedaling",
    "articulation",
    "phrasing",
    "interpretation",
]


def compute_tier2_dimensions(
    midi_notes: list[dict[str, Any]],
    pedal_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return six DimensionAnalysis-shaped dicts from raw chunk MIDI + pedal."""
    if not midi_notes:
        return [{"dimension": d, "analysis": "No notes in chunk."} for d in DIMS_ORDER]

    velocities = [n["velocity"] for n in midi_notes]
    onsets = sorted(n["onset"] for n in midi_notes)
    iois = [b - a for a, b in zip(onsets, onsets[1:])] if len(onsets) >= 2 else []
    durations = [n["offset"] - n["onset"] for n in midi_notes]

    vel_mean = mean(velocities)
    vel_range = max(velocities) - min(velocities)
    ioi_mean = mean(iois) if iois else 0.0
    ioi_std = stdev(iois) if len(iois) >= 2 else 0.0
    dur_mean = mean(durations)
    pedal_count = len(pedal_events)

    return [
        {"dimension": "dynamics",
         "analysis": f"Mean velocity {vel_mean:.1f} (range {vel_range})."},
        {"dimension": "timing",
         "analysis": f"Mean inter-onset interval {ioi_mean:.3f}s (std {ioi_std:.3f}s)."},
        {"dimension": "pedaling",
         "analysis": f"{pedal_count} pedal events across chunk."},
        {"dimension": "articulation",
         "analysis": f"Mean note duration {dur_mean:.2f}s."},
        {"dimension": "phrasing",
         "analysis": f"{len(midi_notes)} notes; mean IOI {ioi_mean:.3f}s."},
        {"dimension": "interpretation",
         "analysis": f"Velocity range {vel_range}, IOI std {ioi_std:.3f}s."},
    ]
```

- [ ] **Step 5: Run test — verify PASS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/tests/test_bar_analysis_local.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/evals/teaching_knowledge/bar_analysis_local.py apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py && git commit -m "feat(evals): port bar_analysis Tier-2 stats to Python"
```

---

## Task 8: `bar_analysis_local.select_worst_chunk` picks the chunk with maximum absolute deviation

**Group:** B (sequential after Task 7; same file)

**Behavior being verified:** Given a list of chunks each containing `predictions` (a dict of dim → score) and a baselines dict, return the chunk whose maximum `|score - baseline|` across dimensions is largest, paired with the triggering dimension name.

**Interface under test:** `select_worst_chunk(chunks, baselines)` from `bar_analysis_local.py`.

**Files:**
- Modify: `apps/evals/teaching_knowledge/bar_analysis_local.py`
- Modify: `apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_bar_analysis_local.py
from teaching_knowledge.bar_analysis_local import select_worst_chunk


def test_selects_chunk_with_largest_absolute_deviation() -> None:
    baselines = {d: 0.5 for d in ["dynamics", "timing", "pedaling",
                                  "articulation", "phrasing", "interpretation"]}
    chunks = [
        {"chunk_index": 0, "predictions": {"dynamics": 0.55, "timing": 0.45}},
        {"chunk_index": 1, "predictions": {"dynamics": 0.50, "timing": 0.20}},  # 0.30
        {"chunk_index": 2, "predictions": {"dynamics": 0.60, "timing": 0.50}},
    ]
    result = select_worst_chunk(chunks, baselines)
    assert result is not None
    assert result["chunk_index"] == 1
    assert result["dimension"] == "timing"


def test_returns_none_on_empty_chunks() -> None:
    assert select_worst_chunk([], {"timing": 0.5}) is None
```

- [ ] **Step 2: Run test — FAIL**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/tests/test_bar_analysis_local.py::test_selects_chunk_with_largest_absolute_deviation -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement**

Append to `bar_analysis_local.py`:

```python
def select_worst_chunk(
    chunks: list[dict[str, Any]],
    baselines: dict[str, float],
) -> dict[str, Any] | None:
    """Return {chunk_index, dimension, chunk} for the chunk + dim with max |score-baseline|."""
    if not chunks:
        return None
    best: dict[str, Any] | None = None
    best_dev = -1.0
    for chunk in chunks:
        preds = chunk.get("predictions", {})
        for dim, score in preds.items():
            if dim not in baselines:
                continue
            dev = abs(float(score) - float(baselines[dim]))
            if dev > best_dev:
                best_dev = dev
                best = {
                    "chunk_index": chunk.get("chunk_index"),
                    "dimension": dim,
                    "chunk": chunk,
                }
    return best
```

- [ ] **Step 4: PASS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/tests/test_bar_analysis_local.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/bar_analysis_local.py apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py && git commit -m "feat(evals): worst-chunk selection in bar_analysis_local"
```

---

## Task 9: `bar_analysis_local.compute_tier1_dimensions` adds notated-score comparison when score JSON is provided

**Group:** B (sequential after Task 8; same file)

**Behavior being verified:** When given a score JSON dict (the `model/data/scores/*.json` shape with a `bars` array of notated notes), the Tier-1 function returns six dicts whose `analysis` strings reference both performance and notated statistics for the **articulation** dimension (performance-to-score duration ratio). Dynamics is deliberately NOT enriched in Tier-1 — see Code Quality note below.

**Code Quality note — why dynamics is not enriched in Tier-1:** Inspecting `model/data/scores/bach.prelude.bwv_846.json` (and other score JSONs) shows every note has `velocity: 80` — the default MIDI export velocity, not a notated dynamic. Comparing performance velocity against a constant 80 carries zero pedagogical signal. The Rust Tier-1 in production has the same input limitation; this Python port deliberately drops the dynamics-vs-notated line to keep the prompt clean. Articulation duration ratio IS real signal because durations vary across score notes. A comment in `bar_analysis_local.py` documents this deliberate divergence from any "compare every dim" intuition.

**Interface under test:** `compute_tier1_dimensions(midi_notes, pedal_events, score_json)` from `bar_analysis_local.py`.

**Files:**
- Modify: `apps/evals/teaching_knowledge/bar_analysis_local.py`
- Modify: `apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py`

- [ ] **Step 1: Write the failing test**

```python
# append
from teaching_knowledge.bar_analysis_local import compute_tier1_dimensions


def test_tier1_articulation_mentions_duration_ratio() -> None:
    midi_notes = [
        {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 90},
        {"pitch": 62, "onset": 0.5, "offset": 1.0, "velocity": 95},
    ]
    score_json = {
        "bars": [
            {"bar_number": 1, "notes": [
                {"pitch": 60, "velocity": 80, "duration_seconds": 0.5},
                {"pitch": 62, "velocity": 80, "duration_seconds": 0.5},
            ]}
        ]
    }
    result = compute_tier1_dimensions(midi_notes, [], score_json)
    articulation = next(r for r in result if r["dimension"] == "articulation")
    text = articulation["analysis"].lower()
    assert "ratio" in text or "score" in text


def test_tier1_dynamics_does_not_mention_notated_score() -> None:
    # Score JSONs use default MIDI velocity 80 for every note; comparing
    # performance velocity to that constant is signal-free. The Tier-1 port
    # deliberately omits a dynamics-vs-notated line. See bar_analysis_local.py.
    midi_notes = [{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 90}]
    score_json = {"bars": [{"bar_number": 1, "notes": [
        {"pitch": 60, "velocity": 80, "duration_seconds": 0.5}
    ]}]}
    result = compute_tier1_dimensions(midi_notes, [], score_json)
    dynamics = next(r for r in result if r["dimension"] == "dynamics")
    text = dynamics["analysis"].lower()
    assert "notated" not in text
    assert "(score)" not in text
```

- [ ] **Step 2: FAIL**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/tests/test_bar_analysis_local.py::test_tier1_dynamics_mentions_notated_score -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement**

Append to `bar_analysis_local.py`:

```python
def compute_tier1_dimensions(
    midi_notes: list[dict[str, Any]],
    pedal_events: list[dict[str, Any]],
    score_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """Tier-1: compute_tier2_dimensions + notated articulation comparison.

    NOTE: Dynamics is deliberately NOT enriched with a score comparison. Score
    JSONs under model/data/scores/ use the default MIDI export velocity 80 for
    every note (not a real notated dynamic), so a performance-vs-score velocity
    line carries zero signal and would dilute the prompt. The Rust Tier-1 in
    production has the same input limitation. Articulation IS enriched because
    notated note durations vary and the perf/score duration ratio is real signal.
    """
    base = compute_tier2_dimensions(midi_notes, pedal_events)
    if not midi_notes:
        return base

    score_notes: list[dict[str, Any]] = []
    for bar in score_json.get("bars", []):
        score_notes.extend(bar.get("notes", []))
    if not score_notes:
        return base

    perf_dur_mean = mean(n["offset"] - n["onset"] for n in midi_notes)
    score_dur_mean = mean(n.get("duration_seconds", 0.0) for n in score_notes) or 1e-6

    enriched = []
    for d in base:
        if d["dimension"] == "articulation":
            ratio = perf_dur_mean / score_dur_mean
            d = {
                **d,
                "analysis": (
                    f"{d['analysis']} Performance/score duration ratio {ratio:.2f}."
                ),
            }
        enriched.append(d)
    return enriched
```

- [ ] **Step 4: PASS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/tests/test_bar_analysis_local.py -v
```

Expected: PASS (all bar_analysis_local tests).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/bar_analysis_local.py apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py && git commit -m "feat(evals): Tier-1 notated-score deltas in bar_analysis_local"
```

---

## Task 10: `bar_analysis_local.build_bar_analysis` filters to selected + ≤2 correlated, returning the same shape as TS `BarAnalysisFacts`

**Group:** B (sequential after Task 9; same file)

**Behavior being verified:** End-to-end public entry point: given a chunk list, baselines, and an optional score_json, return either `None` (no chunks) or a dict matching the TS `BarAnalysisFacts` shape — `{tier, bar_range, selected, correlated}` — applying the 0.15 threshold and cap-2 rule identically.

**Interface under test:** `build_bar_analysis(chunks, baselines, score_json)` from `bar_analysis_local.py`.

**Files:**
- Modify: `apps/evals/teaching_knowledge/bar_analysis_local.py`
- Modify: `apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py`

- [ ] **Step 1: Failing test**

```python
# append
from teaching_knowledge.bar_analysis_local import build_bar_analysis


def test_returns_none_on_no_chunks() -> None:
    assert build_bar_analysis([], {"timing": 0.5}, None) is None


def test_returns_tier2_facts_when_score_json_is_none() -> None:
    baselines = {d: 0.5 for d in ["dynamics", "timing", "pedaling",
                                  "articulation", "phrasing", "interpretation"]}
    chunks = [{
        "chunk_index": 0,
        "predictions": {"dynamics": 0.55, "timing": 0.20, "pedaling": 0.50,
                        "articulation": 0.50, "phrasing": 0.50, "interpretation": 0.50},
        "midi_notes": [
            {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
            {"pitch": 62, "onset": 0.5, "offset": 1.0, "velocity": 70},
        ],
        "pedal_events": [],
    }]
    result = build_bar_analysis(chunks, baselines, None)
    assert result is not None
    assert result["tier"] == 2
    assert result["selected"]["dimension"] == "timing"
    # dynamics dev = 0.05 < 0.15 → correlated should be empty
    assert result["correlated"] == []


def test_correlated_includes_dimensions_above_threshold_cap_2() -> None:
    baselines = {d: 0.5 for d in ["dynamics", "timing", "pedaling",
                                  "articulation", "phrasing", "interpretation"]}
    chunks = [{
        "chunk_index": 0,
        # devs: dyn 0.30, tim selected(-0.30), ped 0.25, art 0.20, phr 0.10, int 0.05
        "predictions": {"dynamics": 0.80, "timing": 0.20, "pedaling": 0.75,
                        "articulation": 0.70, "phrasing": 0.60, "interpretation": 0.55},
        "midi_notes": [{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80}],
        "pedal_events": [],
    }]
    result = build_bar_analysis(chunks, baselines, None)
    dims = [c["dimension"] for c in result["correlated"]]
    assert len(dims) == 2
    assert dims == ["dynamics", "pedaling"]  # top 2 by |dev|
```

- [ ] **Step 2: FAIL**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/tests/test_bar_analysis_local.py::test_returns_tier2_facts_when_score_json_is_none -v
```

Expected: FAIL — `ImportError` (or missing attribute).

- [ ] **Step 3: Implement**

Append to `bar_analysis_local.py`:

```python
DEVIATION_THRESHOLD = 0.15
CORRELATED_CAP = 2


def build_bar_analysis(
    chunks: list[dict[str, Any]],
    baselines: dict[str, float],
    score_json: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Public entry: pick worst chunk, compute Tier-1 or Tier-2, filter to facts."""
    worst = select_worst_chunk(chunks, baselines)
    if worst is None:
        return None
    chunk = worst["chunk"]
    selected_dim = worst["dimension"]
    midi = chunk.get("midi_notes", [])
    pedal = chunk.get("pedal_events", [])

    if score_json is not None:
        dims = compute_tier1_dimensions(midi, pedal, score_json)
        tier = 1
    else:
        dims = compute_tier2_dimensions(midi, pedal)
        tier = 2

    selected = next((d for d in dims if d["dimension"] == selected_dim), None)
    if selected is None:
        return None

    preds = chunk.get("predictions", {})
    candidates = []
    for d in dims:
        if d["dimension"] == selected_dim:
            continue
        if d["dimension"] not in baselines or d["dimension"] not in preds:
            continue
        dev = abs(float(preds[d["dimension"]]) - float(baselines[d["dimension"]]))
        if dev >= DEVIATION_THRESHOLD:
            candidates.append((dev, d))
    candidates.sort(key=lambda x: x[0], reverse=True)
    correlated = [d for _, d in candidates[:CORRELATED_CAP]]

    return {
        "tier": tier,
        "bar_range": None,  # cache lacks bar-aligned ranges; production has them via WASM
        "selected": selected,
        "correlated": correlated,
    }
```

- [ ] **Step 4: PASS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/tests/test_bar_analysis_local.py -v
```

Expected: PASS (all bar_analysis_local tests).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/bar_analysis_local.py apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py && git commit -m "feat(evals): build_bar_analysis entry point with threshold/cap filter"
```

---

## Task 11: `run_eval.build_synthesis_user_msg` injects `bar_analysis` into the top moment

**Group:** B (sequential after Task 10)

**Behavior being verified:** Calling `build_synthesis_user_msg(...)` with a recording whose cached chunks include `midi_notes` and `predictions`, and whose piece slug maps to a known score JSON, produces a user message whose `<session_data>` JSON contains a `bar_analysis` field on the highest-deviation top moment.

**Interface under test:** `build_synthesis_user_msg` from `apps/evals/teaching_knowledge/run_eval.py`.

**Files:**
- Modify: `apps/evals/teaching_knowledge/run_eval.py`
- Create: `apps/evals/teaching_knowledge/tests/test_run_eval_bar_analysis.py`

**Important:** `build_synthesis_user_msg`'s current signature is `(muq_means, duration_seconds, meta)`. To inject bar_analysis, the build agent must either:
- (a) extend the signature to accept the chunks list and call `build_bar_analysis` internally, OR
- (b) compute the facts at the caller (in `run`) and pass them in.

Choose **(a)** — keep the chunks-to-prompt translation in one place. Add a parameter `chunks: list[dict] | None = None` (defaulting to None preserves existing test_run_eval_blocks behavior).

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teaching_knowledge/tests/test_run_eval_bar_analysis.py
from __future__ import annotations

import json

from teaching_knowledge.run_eval import build_synthesis_user_msg


def test_bar_analysis_appears_on_top_moment_when_chunks_provided() -> None:
    muq_means = {
        "dynamics": 0.50, "timing": 0.20, "pedaling": 0.50,
        "articulation": 0.50, "phrasing": 0.50, "interpretation": 0.50,
    }
    meta = {"piece_slug": "chopin_ballade_1", "title": "Ballade", "composer": "Chopin", "skill_bucket": 3}
    chunks = [{
        "chunk_index": 0,
        "predictions": muq_means,
        "midi_notes": [
            {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
            {"pitch": 62, "onset": 0.5, "offset": 1.0, "velocity": 70},
        ],
        "pedal_events": [],
    }]
    out = build_synthesis_user_msg(muq_means, 60.0, meta, chunks=chunks)
    assert "bar_analysis" in out
    # The JSON inside <session_data> must parse and contain the field
    start = out.index("<session_data>") + len("<session_data>")
    end = out.index("</session_data>")
    payload = json.loads(out[start:end].strip())
    top = payload["top_moments"]
    has_bar = [m for m in top if "bar_analysis" in m]
    assert len(has_bar) >= 1


def test_no_bar_analysis_when_chunks_none() -> None:
    muq_means = {
        "dynamics": 0.50, "timing": 0.20, "pedaling": 0.50,
        "articulation": 0.50, "phrasing": 0.50, "interpretation": 0.50,
    }
    meta = {"piece_slug": "clair_de_lune", "title": "Clair", "composer": "Debussy", "skill_bucket": 3}
    out = build_synthesis_user_msg(muq_means, 60.0, meta, chunks=None)
    assert "bar_analysis" not in out
```

- [ ] **Step 2: FAIL**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/tests/test_run_eval_bar_analysis.py -v
```

Expected: FAIL — `TypeError: build_synthesis_user_msg() got an unexpected keyword argument 'chunks'`.

- [ ] **Step 3: Implement**

Edit `run_eval.py::build_synthesis_user_msg`:
1. Add `chunks: list[dict[str, Any]] | None = None` to the signature.
2. After top_moments is computed (current line ~146), if `chunks` is provided, call:
   ```python
   from teaching_knowledge.bar_analysis_local import build_bar_analysis
   from teaching_knowledge.piece_score_map import get_score_path_for_piece
   import json as _json

   if chunks is not None and top_moments:
       score_path = get_score_path_for_piece(meta["piece_slug"])
       score_json = _json.loads(score_path.read_text()) if score_path is not None else None
       baselines_for_facts = {dim: SCALER_MEAN[dim] for dim in DIMS}
       facts = build_bar_analysis(chunks, baselines_for_facts, score_json)
       if facts is not None:
           selected_dim_name = facts["selected"]["dimension"]
           for m in top_moments:
               if m["dimension"] == selected_dim_name:
                   m["bar_analysis"] = facts
                   break
   ```
3. Also update the caller in `run()` (around line 361) to pass `chunks=cached["chunks"]` — the build agent must read the existing call site and identify the right local variable name for the chunks list. Do NOT guess.

- [ ] **Step 4: PASS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/tests/test_run_eval_bar_analysis.py -v
```

Expected: PASS.

Also run the existing eval tests to make sure nothing regressed:

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run pytest teaching_knowledge/ -v
```

Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teaching_knowledge/run_eval.py apps/evals/teaching_knowledge/tests/test_run_eval_bar_analysis.py && git commit -m "feat(evals): inject bar_analysis into synthesis user message"
```

---

## Task 12: Full eval run + lift verification

**Group:** C (sequential, depends on all of Group B)

**Behavior being verified:** Running `run_eval` on the full 513-recording cache after the feature is wired produces an ASCF outcome mean ≥ 1.687 (= 1.387 + 0.30) with a 95% bootstrap CI whose lower bound exceeds 1.537 (= 1.387 + 0.15, the noise floor). Secondary: the Tier-1 subset (recordings whose piece is in `PIECE_SCORE_MAP`) shows a larger ASCF lift than the Tier-2-only subset.

**Interface under test:** `apps/evals/teaching_knowledge/run_eval.py` CLI + downstream analysis.

**Files:**
- Create: `apps/evals/teaching_knowledge/results/bar_analysis_run.jsonl` (artifact, not committed)
- Modify: none

- [ ] **Step 1: Run the full eval**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -m teaching_knowledge.run_eval --limit 513 --out results/bar_analysis_run.jsonl
```

Expected: completes without error. Wall clock is significant (multiple hours possible depending on Sonnet RPS); the build agent should run this with `run_in_background=true` and poll only when notified.

- [ ] **Step 2: Compute lift and CI**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/evals && uv run python -c "
import json, random, statistics
from teaching_knowledge.piece_score_map import get_score_path_for_piece

rows = [json.loads(l) for l in open('results/bar_analysis_run.jsonl') if l.strip()]
valid = [r for r in rows if r.get('error') in (None, '') and 'judge_dimensions' in r]
ascf = [r['judge_dimensions']['audible_specific_corrective_feedback']['outcome'] for r in valid]
tier1 = [r['judge_dimensions']['audible_specific_corrective_feedback']['outcome']
         for r in valid if get_score_path_for_piece(r['piece_slug']) is not None]
tier2 = [r['judge_dimensions']['audible_specific_corrective_feedback']['outcome']
         for r in valid if get_score_path_for_piece(r['piece_slug']) is None]

def boot_ci(xs, n=1000):
    means = [statistics.mean(random.choices(xs, k=len(xs))) for _ in range(n)]
    means.sort()
    return means[int(n*0.025)], means[int(n*0.975)]

random.seed(42)
print(f'n={len(ascf)} mean={statistics.mean(ascf):.3f}')
print(f'  95% CI: {boot_ci(ascf)}')
print(f'tier1 n={len(tier1)} mean={statistics.mean(tier1):.3f}' if tier1 else 'tier1: empty')
print(f'tier2 n={len(tier2)} mean={statistics.mean(tier2):.3f}' if tier2 else 'tier2: empty')
"
```

The output reveals whether the success criterion is met.

- [ ] **Step 3: Decision**

- If `mean ≥ 1.687` AND `CI lower bound > 1.537`: success. Proceed to `/review` → `/ship`.
- If `1.537 < mean < 1.687`: partial signal. Surface to user; do NOT auto-ship. Per spec Open Question 3, investigate threshold/cap tuning before declaring done.
- If `mean ≤ 1.537`: hypothesis falsified. Surface to user with the numbers and the per-tier breakdown.

- [ ] **Step 4: Commit nothing**

Like Task 0, this is verification-only. Record the lift numbers and per-tier means in the PR description.

---

## Coverage Map (spec → tasks)

| Spec requirement | Task(s) |
|------------------|---------|
| `buildBarAnalysisFacts` filters to selected + correlated with 0.15 threshold, cap 3 (1 + 2) | 1, 1b, 1c |
| `AccumulatedMoment.llmAnalysis` is `BarAnalysisFacts \| null` | 2 |
| Session-brain wires facts at both accumulation sites | 3 |
| Synthesis prompt emits `bar_analysis` JSON | 4 |
| Synthesis persists JSON to `reasoning_trace` | 6 |
| `piece_score_map.py` exposes `get_score_path_for_piece` | 5 |
| `bar_analysis_local.py` mirrors Tier-1/2 + facts filter | 7, 8, 9, 10 |
| `run_eval.build_synthesis_user_msg` injects `bar_analysis` | 11 |
| Verification: baseline reproduces | Task 0 (gate) |
| Verification: ≥0.3 lift, Tier-1 > Tier-2 | Task 12 |

No spec requirement is unmapped.

---

## Plan Self-Review Notes

- **Placeholder scan:** No "TBD" / "implement later" remain. Where the plan delegates to the build agent (e.g. Task 5 mapping verification, Task 11 caller variable name), the delegation is explicit with the exact `ls`/`grep` command to run.
- **Type consistency:** `BarAnalysisFacts` shape is identical in TS (Task 1) and Python (Task 10). `selected: DimensionAnalysis`, `correlated: DimensionAnalysis[]`, `tier: number`, `bar_range: string | null`.
- **Group correctness:** Tasks 7/8/9/10 all touch `bar_analysis_local.py` → they are NOT parallel; the plan calls this out explicitly. Tasks 1/5 are parallel (disjoint files). Group B tasks are explicitly serial.
- **Vertical slice check:** Every task has one test + one implementation + one commit. The two tasks 1b/1c append a test to the same file as 1; each remains a single vertical slice.
- **Behavior-test check:** No internal-collaborator mocks. The `makeFakeDb` in Task 6 is a stand-in for the external Postgres dependency, matching the existing `synthesis.test.ts` pattern. The contract test in Task 3 is justified in-line (DO/WASM cannot be unit-tested in isolation without violating the mock rule).
- **Watch-it-fail discipline:** Tasks 1c, 4, and 6 explicitly call out that the build agent must temporarily revert / comment out the implementation to prove the test fails for the right reason before proceeding. Skipping this step makes the test test shape, not behavior.

---

**Feynman Summary:** Right now the Rust analyzer hands the API six per-dimension sentences like "your onset deviation is 45 ms early, score reference is 5 ms early" — and the API throws them away. The teacher then guesses at specifics. This plan stops throwing them away. A small filter function picks the one dimension that triggered the teaching moment plus up to two other dimensions that are also noticeably off, and ships that JSON to the teacher inside the synthesis prompt. The Python eval harness gets the same treatment so we can measure whether the teacher actually starts being more specific (the ASCF score on 513 recordings). If the lift is ≥ 0.3, we ship; if it's marginal, we tune the threshold; if it's zero, we know "any measured fact" is not enough and we need to enrich the features themselves before training.

---

## Challenge Review

### CEO Pass

**Premise:** The right problem. ASCF outcome 1.387/3.0 is the weakest dimension in the locked baseline and `session-brain.ts:974` literally throws away the structured `DimensionAnalysis` records the Rust analyzer already produces. The causal chain (analyzer produces facts → API discards facts → teacher has no facts → ASCF judge sees no specificity) is verified in code (`session-brain.ts` lines 614, 974, 1081, 1240 all confirmed). Direct path.

**Scope:** Tight. 6 production files, 5 eval files, no new services, no schema change (reuses `observations.reasoning_trace`). Could the MVP be smaller? Yes — Tasks 7/9/10 (the Python Tier-1/2 port) are a meaningful re-implementation of `bar_analysis.rs` and could be deferred if Task 5+11 alone produced lift on the production wire. But the spec's measurement instrument (the eval harness) requires the Python mirror; cutting it would leave the success criterion unverifiable. The eval mirror is load-bearing, not gold-plating.

**12-Month Alignment:** Moves toward the north star. Bar-specific facts are exactly the substrate the Stage 2/3 teacher LoRA needs to learn from. If this lifts ASCF, the same `bar_analysis` JSON is high-signal training data for the Qwen finetune. If it does not lift ASCF, that is a falsifying signal that "any measured fact" is insufficient and the team can stop investing in feature enrichment and pivot to training-time methods. Either outcome is information.

**Alternatives:** Spec is light on alternatives. The brainstorm presumably explored "enrich the Rust analyzer first" vs "use what we have"; only the chosen path is documented.

[QUESTION] — Spec does not document the rejected alternatives. If "add tempo/rubato split in Rust first" or "concatenated prose instead of structured JSON" were considered and rejected, the reasoning belongs in the spec for future reference. (See spec Open Question 2 — it touches this but does not document the rejection.)

### Engineering Pass

#### Architecture

Data flow verified by reading the actual code:

```
WASM analyzer (analyze_tier1/2) → ChunkAnalysis{tier, bar_range, dimensions[]}
                                          ↓ [TODAY: dimensions[] dropped at session-brain.ts:621]
                                          ↓ [PLAN: hoist chunkAnalysis, pass to buildBarAnalysisFacts]
                                  AccumulatedMoment.llmAnalysis: BarAnalysisFacts | null
                                          ↓
                                  topMoments() → buildSynthesisFraming → JSON.stringify → <session_data>
                                          ↓
                                  persistAccumulatedMoments → observations.reasoning_trace (JSON-stringified)
```

The chain holds in real code. The four WASM call sites in `session-brain.ts` (lines 614, 635, 655, 1081, 1091, 1102 — actually 6 sites total when both handlers are counted, plan undercounts) all assign to a local `analysis` then promptly drop the `dimensions` array. The plan's hoist-and-assign edit is the minimal correct intervention.

[RISK] (confidence: 8/10) — The plan's Task 3 implementation strategy says "4 sites in `handleProcessChunk`, 4 sites in `handleEvalChunk`" but `grep` finds **3 `analyzeTier1`/`analyzeTier2` calls in each handler** (6 total), not 4+4. The plan also hand-waves at line 586 ("and any I missed; the build agent must read the function and locate every wasm.analyzeTier(1|2)(...) call"). Fine instruction, but the count in the task header is wrong. Build agent must trust the grep, not the count. Watch during execution.

#### Module Depth Audit

- **`bar-analysis-facts.ts` (TS):** Interface = 1 function + 2 types. Implementation hides dim→index mapping, threshold (0.15), cap (2), abs-deviation sort, selected-exclusion, null-on-empty. **DEEP.**
- **`bar_analysis_local.py`:** Interface = 1 public function `build_bar_analysis` + 3 helpers (`compute_tier1_dimensions`, `compute_tier2_dimensions`, `select_worst_chunk`) the tests reach into directly. The "public" surface is 4 functions, not 1. Three of those are tested as if they were public. **UNCLEAR / borderline SHALLOW.**

[RISK] (confidence: 7/10) — `bar_analysis_local.py` tests reach in at three layers (`compute_tier2_dimensions`, `select_worst_chunk`, `compute_tier1_dimensions`, `build_bar_analysis`). The spec says "Tested through: The public function. Inputs are fixture chunk lists..." but the plan tests four functions individually. Either the spec is wrong (it is — the helpers must be testable in isolation to verify formula parity with Rust) or the plan deviates. Defensible because the Rust port needs formula-level tests, but note the deviation from spec text. Mitigation: keep the helper tests; relax the spec text in Task 12 PR description.

- **`piece_score_map.py`:** Spec acknowledges shallow-but-unavoidable. Pure lookup table. Acceptable.

#### Code Quality

[BLOCKER] (confidence: 9/10) — **Task 5 mapping table contains a verified-wrong row.** Plan line 712 proposes `moonlight_sonata_mvt1 → beethoven.piano_sonatas.14-1.json`. `ls model/data/scores/` shows `beethoven.piano_sonatas.14-3.json` exists but **`14-1.json` does NOT exist** (the Adagio sostenuto, mvt 1 of Op.27 No.2, is absent from the scores dir). The plan instructs the build agent to verify before committing — good — but if the build agent runs `ls | grep "beethoven\.piano_sonatas\.14"`, they will see only `14-3` and leave `moonlight_sonata_mvt1` unmapped. This is salvageable but the plan's high-confidence claim that 14-1 maps Moonlight is empirically wrong and must be removed from the table before execution to avoid the build agent committing a wrong mapping if they trust the table over the verification step. Fix: rewrite the table to show all rows as "needs verify" and let the `ls` step populate it.

[BLOCKER] (confidence: 9/10) — **Tier-1 notated-velocity comparison is signal-free.** Reading `bach.prelude.bwv_846.json`, every note has `velocity: 80` — this is the default MIDI export velocity, not a notated dynamic. Therefore `score_vel_mean` in Task 9's `compute_tier1_dimensions` will be 80 for every score and the "Notated mean velocity 80.0 (score); performance X" line carries zero pedagogical information. The teacher LLM will be shown a constant. The articulation duration ratio is still real signal (durations vary), but dynamics is not. Either drop the dynamics-vs-notated line entirely (it actively dilutes the prompt), or document this and accept that Tier-1 dynamics adds no signal over Tier-2. Decision must happen before Task 9 commits; otherwise the eval lift on Tier-1 will be diluted by a known-noise feature and the secondary success signal ("Tier-1 lift > Tier-2 lift") may fail for reasons unrelated to the core hypothesis.

[RISK] (confidence: 7/10) — Task 11's `build_synthesis_user_msg` uses `SCALER_MEAN` (the MuQ global mean) as the baselines passed to `build_bar_analysis`. But the production path uses *student-specific* baselines (`baselines: StudentBaselines` constructed per session in `session-brain.ts` ~line 1210). The eval and production therefore compute "correlated dimensions" against different baselines. For the eval this may be acceptable (no student history exists in the cache), but the plan does not flag this divergence. The Tier-1-vs-Tier-2 success signal could mislead if student-baseline vs global-mean is a stronger lever than score-presence. Mitigation: document in Task 11 PR that eval uses global mean and prod uses per-student baselines; spec should acknowledge this is an unmodeled difference.

[OBS] — `session-brain.ts:974` currently sets `llmAnalysis: null`. The existing `synthesis.ts:48` reads `moment.llmAnalysis ?? moment.reasoning` — i.e. today `llmAnalysis` always falls through to `reasoning`. The plan's Task 2 step 5 line `moment.llmAnalysis !== null ? JSON.stringify(moment.llmAnalysis) : moment.reasoning` is behaviorally identical to the old code *when* `llmAnalysis: null` (which is everywhere pre-Task-3). Tests in Task 6 must run *after* Task 3 wires real facts, OR the test fixture must explicitly construct a non-null `llmAnalysis` (the plan does the latter — OK).

#### Test Philosophy — The Three Author-Flagged Concerns

The author flagged three areas. Verdicts after reading the code:

**1. Task 3 contract test rather than DO integration test.** Defensible (confidence 8/10). The DO requires a Durable Object environment, WASM runtime, and a constructed `wasm` bridge — none can be set up without either (a) mocking the WASM bridge (forbidden by project test philosophy) or (b) the Miniflare DO harness, which `session-brain.unit.test.ts` already deliberately avoids in favor of pulling pure functions out of the DO and testing them. The plan's contract test pins the shape that the session-brain edit must produce, and Task 12's full eval run is the integration check. **However:** the contract test in Task 3 is verifying that `buildBarAnalysisFacts` returns the right shape — *which is already tested by Task 1c*. It is not actually testing `session-brain.ts` wiring. The session-brain change is verified only by typecheck (Task 3 Step 4) and by the full eval in Task 12.

[RISK] (confidence: 7/10) — Task 3 wires production code with no test that fails before the wire is in place. If the build agent makes a typo (e.g. swaps `scoresArray` for `predictions`, or uses `momentDim` from the wrong scope), nothing in Tasks 1–11 catches it; only Task 12 catches it, hours into a multi-hour run. Mitigation: have the build agent run `git diff session-brain.ts` and self-review the edit before Task 4 commit. The plan should explicitly add a self-review step here.

**2. Task 5 mapping table verify.** The instruction to `ls`-verify is good practice, but as flagged in the BLOCKER above, the table contains a row that the build agent cannot verify (because the file does not exist) and the plan does not pre-flag that row as unmapped. Defensible only after the mapping is corrected.

**3. Tasks 4 and 6 watch-it-fail revert-then-restore.** Defensible (confidence 8/10) but fragile. The plan's instruction at lines 364, 671, 911 is correct in principle — to bind a test to behavior rather than shape, you must observe the failure. But the discipline depends entirely on the build agent following the revert step; there is no automated enforcement. For Task 4 specifically, the test asserts `out).toContain('"bar_analysis"')` — this could pass for any reason that the string "bar_analysis" appears in the output (it appears because `topMoments` is passed `unknown`-typed). The revert-then-restore step is the only thing that converts this from a shape test to a behavior test.

[RISK] (confidence: 6/10) — The watch-it-fail discipline in Tasks 1c/4/6 is the only thing distinguishing these tests from shape tests. Build-agent compliance must be enforced (e.g., subagent must paste failing test output into commit message). If skipped, these three tasks add no behavior coverage.

#### Vertical Slice Audit

- Task 1, 1b, 1c: each is one test + one impl + one commit. Compliant.
- Task 1c step 3 says "no code edit needed" because 1b implementation already covers the cap-2 case. The watch-it-fail revert restores behavior coverage. Borderline compliant.
- Task 2: one type change + one round-trip test + minimum caller fixup in synthesis.ts. Bundles a synthesis.ts edit with an accumulator test. **Mild bundling.**

[RISK] (confidence: 6/10) — Task 2 Step 5 quietly modifies `synthesis.ts` (the `JSON.stringify` line) as a "minimum compile-only fix" but commits it alongside the accumulator changes (Step 7's `git add` includes synthesis.ts). The synthesis.ts edit is behavioral, not compile-only — it changes what gets written to `reasoning_trace`. The plan acknowledges this and defers behavior coverage to Task 6, which then has to revert-restore Task 2's edit to verify failure. Cleaner alternative: keep Task 2 type-only (cast to string with `JSON.stringify` already, but assert behavior is unchanged via a Task 2 test), then Task 6 changes behavior. Not blocking — the revert-restore discipline papers over this — but it is the source of the "may pass without code change" awkwardness the author flagged.

- Tasks 3–11: one test + one impl + one commit each. Compliant.
- Task 0, Task 12: verification gates, no commit. Compliant by exception.

#### Test Coverage Gaps

```
[+] bar-analysis-facts.ts
    └── buildBarAnalysisFacts()
        ├── [TESTED ★★] empty dimensions → null (Task 1)
        ├── [TESTED ★★] selected dimension present, correlated excludes it (Task 1b)
        ├── [TESTED ★★] correlated cap at 2, sorted (Task 1c)
        ├── [TESTED ★★] threshold 0.15 (Task 1c)
        └── [GAP]      selectedDimension not in analysis.dimensions → null
                       (Task 1b implementation handles this via `find(...) === undefined`
                        but no test covers it)

[+] session-brain.ts (Task 3 wire)
    └── handleProcessChunk + handleEvalChunk accMoment build
        ├── [GAP] chunkAnalysis assignment after Tier1 — typecheck only
        ├── [GAP] chunkAnalysis assignment after Tier2 — typecheck only
        ├── [GAP] Tier-3 path (no analyzeTier call) → chunkAnalysis stays null → llmAnalysis null
                  (the "correct" behavior is asserted nowhere; only Task 12 sees it)

[+] bar_analysis_local.py
    └── all helpers tested individually — OK
    └── build_bar_analysis() tested for None-on-empty, Tier-2 path, cap-2.
        └── [GAP] Tier-1 path end-to-end (score_json provided) → no test asserting
                  the returned dict's `tier == 1` and the dynamics analysis string
                  contains "notated"

[+] run_eval.build_synthesis_user_msg
    └── Task 11 tests chunks-present and chunks-none paths.
        └── [GAP] piece_slug maps to a real score but file load fails → behavior?
                  (Task 5 raises FileNotFoundError; Task 11 does not catch it →
                   whole eval row errors out. Is that intended? Probably yes per
                   "explicit exception handling" preference, but not tested.)
```

[RISK] (confidence: 7/10) — selectedDimension-not-found branch in `buildBarAnalysisFacts` (returns `null`) is exercised by no test. The branch exists in Task 1b's code. Add a one-line test.

[RISK] (confidence: 7/10) — Tier-3 path in session-brain (no analyzeTier1/2 call → `chunkAnalysis` stays null) has zero test coverage. If the Tier-3 branch is the most common in production (per the project memory "AMT not deployed (all sessions Tier 3)"), this is the **dominant production path** and the only verification is the eval, which doesn't exercise the production session-brain.

#### Failure Modes

[RISK] (confidence: 7/10) — Task 12 full eval at 513 recordings × Sonnet RPS may take many hours. If it fails halfway (Sonnet 5xx, rate limit, network), there is no resume mechanism documented. `run_eval.py` may support resume — verify before kicking off — otherwise budget for a full re-run. Not blocking; just calendar risk.

[OBS] — Task 5 raises `FileNotFoundError` when a mapped file doesn't exist. Good — matches user preference for explicit exceptions over silent fallbacks.

[OBS] — Task 11's `_json.loads(score_path.read_text())` per row is wasteful (re-reads + re-parses ~1 MB JSON for every recording of the same piece). For 513 recordings across ~17 pieces this is OK (~50 MB throwaway). For a production-style harness it would warrant a cache. Not blocking.

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| ASCF baseline 1.387 reproduces on 10-recording smoke | VALIDATE | Task 0 explicitly validates; gate present |
| `topMoments` parameter in `buildSynthesisFraming` passes `bar_analysis` through untouched | SAFE | Verified: line 121–124 passes `topMoments` (typed `unknown`) directly into `JSON.stringify` with no filtering |
| `analysis.dimensions` array populated for all Tier-1/2 chunks | SAFE | Verified in `bar_analysis.rs` — analyze_tier1/2 always populate 6 entries |
| Score JSON files contain real notated velocities | RESOLVED | Verified empirically wrong; Task 9 now omits the dynamics-vs-notated line and documents why. Articulation duration ratio retained as the real-signal Tier-1 feature. |
| `model/data/scores/beethoven.piano_sonatas.14-1.json` exists | RESOLVED | Verified wrong (only 14-3 exists). Task 5 mapping table rewritten as a procedure — build agent populates from `ls` output, no row pre-blessed. |
| Task 3 contract test is sufficient because Task 12 eval covers integration | VALIDATE | Plausible but unverified until Task 12 succeeds; until then the production wire is typecheck-only |
| Tier-3 (no AMT) is the rare path; Tier-1/2 dominates | RISKY | Project memory says the opposite: "AMT not deployed (all sessions Tier 3)". If true, the entire production wire produces `llmAnalysis: null` and the only path being measured is the eval. The production lift cannot be observed until AMT deploys. |
| Build agent will follow the watch-it-fail revert-restore discipline | VALIDATE | Plan instructs it; subagent discipline depends on `/build` enforcement |
| Existing `synthesis.test.ts` uses `vi.fn()` mocks; Task 6 plan uses hand-rolled `makeFakeDb` | VALIDATE | Plan says "adopt whatever pattern is in place"; build agent should choose `vi.fn()` to match. Plan's example code shows the wrong pattern but the instruction to match existing is correct. |
| `SCALER_MEAN` in `run_eval.py` is the right baseline for eval correlated-dim filter | RISKY | Production uses per-student baselines; eval uses global means. Different baselines → different correlated dimensions. Documented nowhere. |
| `chopin.etudes_op_10.4.json` not `chopin.etudes.10.4.json` is the correct filename | SAFE | Verified: file is named `chopin.etudes_op_10.4.json`. Plan's "unmapped" row for chopin_etude_op10no4 is wrong — the file exists. Build agent's `ls` will catch this. |

### Summary

[BLOCKER] count: 2
[RISK]    count: 8
[QUESTION] count: 1
[OBS]      count: 3

The two blockers are both empirical: the Moonlight mapping row in Task 5 is verifiably wrong, and the notated-velocity comparison in Task 9 is signal-free because score JSONs lack real notated dynamics. Both must be addressed in the plan text before execution, not deferred to build-agent discovery. The three author-flagged concerns (contract test, mapping verify, watch-it-fail) are defensible in principle but depend on build-agent discipline that the plan should make harder to skip.

Additionally, the project memory note that AMT is not deployed (all production sessions are Tier 3) means the production wire of this change produces `llmAnalysis: null` for every real user today; only the eval harness exercises the feature meaningfully. That is not a blocker for shipping the plumbing — but the team should know that "ship" means "ship the plumbing and the eval lift; production lift gated on AMT deploy." Worth a note in the plan's success-criteria section so reviewers don't expect production-side ASCF movement until AMT ships.

VERDICT: NEEDS_REWORK — Fix two blockers before execution: (1) correct the Task 5 mapping table (mark all rows "needs verify" or remove the empirically wrong Moonlight 14-1 row); (2) decide what to do about signal-free notated-velocity comparison in Task 9 (either drop the dynamics-vs-notated line or document and accept). After these two are addressed, the plan can proceed with the noted risks tracked during build.

---

## Challenge Review — Loop 2

Re-review after the two loop-1 blockers were addressed. Scope of this pass: verify the two fixes are sound and complete, check for new blockers introduced by the edits, assess whether previous non-blocker risks are now plan-breaking.

### Blocker 1 — Task 5 mapping table rewritten as procedure

**Verified resolution.** Empirical check: `ls model/data/scores/ | grep -iE "beethoven.*14"` returns only `beethoven.piano_sonatas.14-3.json`; the previously wrong `moonlight_sonata_mvt1 → 14-1.json` row is no longer in the plan. The candidate table at plan lines 718–736 now uses neutral grep patterns (e.g. `beethoven\.piano_sonatas\.14-`) with explicit per-row "needs verify" markers, and the Moonlight row carries an inline carve-out: "only map if a `14-1` mvt-1 file is present; if only `14-3` exists, leave unmapped." The `PIECE_SCORE_MAP` skeleton at plan line 812 ships empty with a build-agent instruction "populate this dict from the Step 1 ls-verification results. Do NOT add any row without confirming the file exists." Five rules at lines 738–743 enforce the verification gate. **The fix is sound and complete; this blocker is resolved.**

### Blocker 2 — Task 9 dropped notated-velocity comparison

**Verified resolution.** Empirical check: sampling `bach.fugue.bwv_846.json` confirms every note has `velocity: 80` (default MIDI export, not a notated dynamic). The plan now removes the dynamics-vs-notated line from `compute_tier1_dimensions`:

- Plan line 1206 says "Dynamics is deliberately NOT enriched in Tier-1 — see Code Quality note below."
- The Code Quality note at lines 1208 documents the rationale.
- The implementation at lines 1297–1308 only enriches articulation (perf/score duration ratio); dynamics is left at its Tier-2 string.
- The docstring at lines 1277–1283 captures the divergence so future readers do not "fix" it back.
- The new test `test_tier1_dynamics_does_not_mention_notated_score` (lines 1242–1255) locks the absence as intentional behavior, asserting neither "notated" nor "(score)" appears in the dynamics string.
- The articulation duration ratio (real signal) is retained and tested by `test_tier1_articulation_mentions_duration_ratio`.

**The fix is sound and complete; this blocker is resolved.** The Presumption Inventory rows for these two assumptions were updated to `RESOLVED` with reasons.

### New Blockers Introduced By The Edits

None. The edits are subtractive (remove a wrong table row, remove a signal-free line) or additive in a way that locks intentional absence (a test that asserts a string is NOT present). No new file dependencies, no new types, no new control-flow branches were introduced.

One small consistency check worth noting (not a blocker): Task 9 Step 2's stale shell command at line 1260 still references `test_tier1_dynamics_mentions_notated_score` as the function to run — but the test was renamed to `test_tier1_dynamics_does_not_mention_notated_score` (line 1242). The build agent runs the whole file (`pytest teaching_knowledge/tests/test_bar_analysis_local.py -v`), so the wrong filter just selects nothing and the file's other tests still execute. Cosmetic; flag as `[OBS]`, not `[BLOCKER]`.

### Loop-1 Non-Blocker Risks — Status Check

Walking the prior risks to see if any are now plan-breaking:

| Prior risk | Loop-2 status |
|---|---|
| Task 3 call-site count off-by-one ("4+4" vs actual 3+3) | Unchanged; plan still hand-waves "and any I missed; the build agent must locate every call." Acceptable per `/build` discipline. **Ride through.** |
| `bar_analysis_local.py` tests reach into helpers (UNCLEAR depth) | Unchanged; defensible because formula parity with Rust requires helper-level tests. **Ride through.** |
| Task 3 production wire has no failing test before edit (typecheck-only) | Unchanged; Task 12 eval is the only behavior check. Risk acknowledged. **Ride through with self-review at commit time.** |
| Watch-it-fail discipline in 1c/4/6 depends on subagent compliance | Unchanged. **Ride through; `/build` two-stage review catches drift.** |
| Task 2 bundles synthesis.ts edit with accumulator commit | Unchanged. **Ride through.** |
| selectedDimension-not-found branch untested | Unchanged. One-line test gap. **Ride through.** |
| Tier-3 production path (dominant per project memory) has no test | Unchanged; structural — production lift gated on AMT deploy, called out in loop-1 summary. **Ride through; not a build blocker.** |
| Task 11 uses `SCALER_MEAN` (global) where prod uses per-student baselines | Unchanged. **Ride through; document in PR.** |
| Task 12 full eval has no documented resume mechanism | Unchanged; calendar risk only. **Ride through.** |
| Task 11 re-parses score JSON per row (~50 MB throwaway) | Unchanged; negligible at 17 pieces. **Ride through.** |

None of the prior risks were elevated by the edits. None became plan-breaking.

### Updated Summary

[BLOCKER] count: 0 (both loop-1 blockers resolved; no new blockers)
[RISK]    count: 8 (all carried over from loop-1, all ride-through)
[QUESTION] count: 1 (carried over: spec does not document rejected alternatives)
[OBS]      count: 4 (one new: stale test-name reference in Task 9 Step 2 shell command)

VERDICT: PROCEED_WITH_CAUTION — Both loop-1 blockers are empirically and structurally resolved. Carry-over risks are tracked and have named fallbacks; the dominant one (Tier-3 production path means production-side ASCF will not move until AMT deploys) is structural, not a plan defect — surface in the PR description so reviewers calibrate expectations. Build agent should self-review the `session-brain.ts` edit diff before committing Task 3 and observe the watch-it-fail revert step on Tasks 1c, 4, and 6.
