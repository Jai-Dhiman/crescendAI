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

**Hand-curated mapping (the build agent MUST verify each line by `ls /Users/jdhiman/Documents/crescendai/model/data/scores/ | grep <pattern>` before committing):**

| piece_slug | score JSON file (under `model/data/scores/`) | confidence |
|------------|------------------------------------------------|------------|
| `bach_prelude_c_wtc1` | `bach.prelude.bwv_846.json` | high |
| `chopin_ballade_1` | `chopin.ballades.1.json` | high |
| `moonlight_sonata_mvt1` | `beethoven.piano_sonatas.14-1.json` | needs verify (Sonata 14 = Op.27 No.2 "Moonlight") |
| `pathetique_mvt2` | `beethoven.piano_sonatas.8-2.json` | needs verify (Sonata 8 = Op.13 "Pathétique") |
| `fur_elise` | unmapped (no Bagatelle WoO.59 in scores dir as of 2026-05-27) | low |
| `rachmaninoff_prelude_csm` | unmapped (need to determine which prelude) | unknown |
| `chopin_etude_op10no4` | unmapped (no `chopin.etudes.10.4` in scores dir as of 2026-05-27) | unknown |
| All others | None | — |

The build agent's first action in this task is `ls /Users/jdhiman/Documents/crescendai/model/data/scores/ | grep -iE "beethoven\.piano_sonatas\.(8|14)"` etc. — confirm or refute each "needs verify" row. Pieces that cannot be confidently matched stay `None`. The plan does NOT bless any specific score file blindly; the table above is a starting point.

- [ ] **Step 1: Verify the mapping table by inspecting `model/data/scores/`**

```bash
ls /Users/jdhiman/Documents/crescendai/model/data/scores/ | grep -iE "bach\.prelude\.bwv_846|chopin\.ballades\.1|beethoven\.piano_sonatas\.(8|14)"
```

For each line in the proposed table marked "needs verify", confirm the file exists. For each "low" or "unknown" line, attempt to find the right file; if none, leave the slug unmapped.

- [ ] **Step 2: Write the failing test**

```python
# apps/evals/teaching_knowledge/tests/test_piece_score_map.py
from __future__ import annotations

from pathlib import Path

import pytest

from teaching_knowledge.piece_score_map import get_score_path_for_piece


@pytest.mark.parametrize(
    "slug",
    [
        "bach_prelude_c_wtc1",
        "chopin_ballade_1",
    ],  # Build agent: add every confirmed mapping from Step 1.
)
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

# Hand-curated mapping. Each entry was verified by inspecting model/data/scores/
# on 2026-05-27. Pieces not in this table return None.
PIECE_SCORE_MAP: dict[str, str] = {
    "bach_prelude_c_wtc1": "bach.prelude.bwv_846.json",
    "chopin_ballade_1": "chopin.ballades.1.json",
    # Build agent: extend with any rows confirmed in Step 1.
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

**Behavior being verified:** When given a score JSON dict (the `model/data/scores/*.json` shape with a `bars` array of notated notes), the Tier-1 function returns six dicts whose `analysis` strings reference both performance and notated statistics (e.g. mention "score" or "notated" for at least the dynamics dimension).

**Interface under test:** `compute_tier1_dimensions(midi_notes, pedal_events, score_json)` from `bar_analysis_local.py`.

**Files:**
- Modify: `apps/evals/teaching_knowledge/bar_analysis_local.py`
- Modify: `apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py`

- [ ] **Step 1: Write the failing test**

```python
# append
from teaching_knowledge.bar_analysis_local import compute_tier1_dimensions


def test_tier1_dynamics_mentions_notated_score() -> None:
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
    dynamics = next(r for r in result if r["dimension"] == "dynamics")
    text = dynamics["analysis"].lower()
    assert "score" in text or "notated" in text
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
    """Tier-1: compute_tier2_dimensions + notated comparison from score_json."""
    base = compute_tier2_dimensions(midi_notes, pedal_events)
    if not midi_notes:
        return base

    score_notes: list[dict[str, Any]] = []
    for bar in score_json.get("bars", []):
        score_notes.extend(bar.get("notes", []))
    if not score_notes:
        return base

    perf_vel_mean = mean(n["velocity"] for n in midi_notes)
    score_vel_mean = mean(n["velocity"] for n in score_notes)
    perf_dur_mean = mean(n["offset"] - n["onset"] for n in midi_notes)
    score_dur_mean = mean(n.get("duration_seconds", 0.0) for n in score_notes) or 1e-6

    enriched = []
    for d in base:
        if d["dimension"] == "dynamics":
            d = {
                **d,
                "analysis": (
                    f"{d['analysis']} Notated mean velocity {score_vel_mean:.1f} "
                    f"(score); performance {perf_vel_mean:.1f}."
                ),
            }
        elif d["dimension"] == "articulation":
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
