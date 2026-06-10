# DO Wire Piece-ID (v2 gate) Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** During a live (or eval-replay) session, the `SessionBrain` DO identifies the piece using the certified v2 chroma-recall + elastic-DTW margin gate (locking only on a certified-confident match), replacing the uncertified legacy 3-stage pipeline, which is then deleted.
**Spec:** docs/specs/2026-06-09-do-wire-piece-id-design.md
**Style:** Follow `apps/api/TS_STYLE.md` and the root `CLAUDE.md`. No `any`; never destructure `c.env`/service bindings; Zod-validate DO state on read; structured `console.log(JSON.stringify(...))`; DO state versioning across awaits. Closes #26 (PR body must contain `Closes #26`).

---

## Sequencing rationale (read first)

The tree MUST compile and tests MUST pass at every commit. Therefore:

1. **Wire the v2 gate IN first** (Tasks 1–3), making the DO call `identify_piece` via the new buffer + shared helper. The legacy Rust exports and bridge wrappers still exist and still compile during this phase — they are simply no longer reached at runtime.
2. **Delete the legacy surface only AFTER nothing calls it** (Tasks 4–6). Task 4 confirms (by grep) that `session-brain.ts` no longer references any legacy wrapper before deletion.

`identificationNoteCount` → `identificationNoteBuffer` is a breaking state-shape change, so Task 1 (schema) and Task 2 (DO rewire) are sequential and land together before any other group.

## Task Groups

- **Group A (sequential):** Task 1 → Task 2 → Task 3. All touch `session-brain.schema.ts` / `session-brain.ts` / its tests; the DO must compile and the new integration test must pass before legacy deletion. `[SHIPS INDEPENDENTLY]` after Task 3 — at that point the certified gate is live at runtime and the legacy path is dead but harmless; the user can already run a real session and get v2 identification.
- **Group B (sequential, depends on A):** Task 4 → Task 5 → Task 6. Legacy deletion (TS bridge, Rust crate, justfile). Cannot start until Task 3 proves the runtime no longer needs the legacy path.

Tasks within a group touch overlapping files (`session-brain.ts`, `wasm-bridge.ts`, `lib.rs`) and so are strictly sequential.

---

### Task 1: Replace identificationNoteCount with a bounded note buffer in DO state

**Group:** A (first; Task 2 depends on it)

**Behavior being verified:** A freshly created session state carries an empty `identificationNoteBuffer` and validates; the retired `identificationNoteCount` field is gone.
**Interface under test:** `createInitialState` + `sessionStateSchema` (the DO's persisted-state contract, exercised by `session-brain.schema.test.ts`).

**Files:**
- Modify: `apps/api/src/do/session-brain.schema.ts`
- Test: `apps/api/src/do/session-brain.schema.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `apps/api/src/do/session-brain.schema.test.ts` (inside the existing top-level `describe`, or as a new `describe` block — match the file's existing structure):

```typescript
describe("identificationNoteBuffer state field", () => {
	it("createInitialState starts with an empty note buffer and no legacy count", () => {
		const s = createInitialState("sess", "stud", null);
		expect(s.identificationNoteBuffer).toEqual([]);
		expect("identificationNoteCount" in s).toBe(false);
	});

	it("sessionStateSchema accepts a populated note buffer and round-trips it", () => {
		const s = createInitialState("sess", "stud", null);
		s.identificationNoteBuffer = [
			{ pitch: 60, onset: 0, offset: 0.4, velocity: 100 },
			{ pitch: 64, onset: 0.5, offset: 0.9, velocity: 90 },
		];
		const parsed = sessionStateSchema.parse(s);
		expect(parsed.identificationNoteBuffer).toHaveLength(2);
		expect(parsed.identificationNoteBuffer[0]?.pitch).toBe(60);
	});

	it("sessionStateSchema defaults the buffer to [] when absent", () => {
		const raw = { ...createInitialState("sess", "stud", null) } as Record<
			string,
			unknown
		>;
		delete raw.identificationNoteBuffer;
		const parsed = sessionStateSchema.parse(raw);
		expect(parsed.identificationNoteBuffer).toEqual([]);
	});
});
```

Confirm the test file already imports `createInitialState` and `sessionStateSchema` from `./session-brain.schema`; if not, add the import.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test src/do/session-brain.schema.test.ts
```
Expected: FAIL — `expect(s.identificationNoteBuffer).toEqual([])` fails because the field does not exist (`undefined` !== `[]`), and `"identificationNoteCount" in s` is still `true`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/do/session-brain.schema.ts`, in `sessionStateSchema`, **replace** the line:

```typescript
	identificationNoteCount: z.number().int().default(0),
```

with:

```typescript
	identificationNoteBuffer: z
		.array(
			z.object({
				pitch: z.number().int(),
				onset: z.number(),
				offset: z.number(),
				velocity: z.number(),
			}),
		)
		.default([]),
```

In `createInitialState`, **replace** the line:

```typescript
		identificationNoteCount: 0,
```

with:

```typescript
		identificationNoteBuffer: [],
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test src/do/session-brain.schema.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.schema.ts apps/api/src/do/session-brain.schema.test.ts && git commit -m "feat(#26): add identificationNoteBuffer to DO state, retire note count"
```

---

### Task 2: Wire the certified v2 gate into the DO and identify across accumulated chunks

**Group:** A (depends on Task 1; Task 3 depends on this)

**Behavior being verified:** Driving the DO through its `eval_chunk` WebSocket interface with notes that match an in-catalog piece (accumulated across chunks past the MIN-notes threshold) locks the piece via the v2 gate and emits a `piece_identified` event; an ambiguous performance does not lock.
**Interface under test:** the `eval_chunk` WebSocket message path of `SessionBrain` (public, via `webSocketMessage`), with the v2 artifact seeded into the miniflare `SCORES` R2 binding.

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Test: `apps/api/src/do/session-brain.piece-id.test.ts` (new)

This task creates the integration test (one focused locking scenario) AND the implementation that makes it pass: the new state constants, the rewritten `tryIdentifyPiece`, the shared `accumulateAndIdentify` helper, and its call from `handleEvalChunk`. (`finalizeChunk` is rewired in this same task because it shares the helper; the chunk_ready path is then covered by the existing canary, which seeds `pieceLocked = true` and so is unaffected.)

- [ ] **Step 1: Write the failing test**

Create `apps/api/src/do/session-brain.piece-id.test.ts`:

```typescript
/// <reference types="@cloudflare/vitest-pool-workers" />
import { describe, expect, it } from "vitest";
import { env, runInDurableObject } from "cloudflare:test";
import { SessionBrain } from "./session-brain";
import { createInitialState, type SessionState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
		SCORES: R2Bucket;
	}
}

// A minimal v2 artifact: "exact" shares the query's chord-events + chroma; "decoy" is disjoint.
// This is the certified locking fixture proven in wasm-bridge.workerd.test.ts.
const V2_ARTIFACT = JSON.stringify({
	version: "v2",
	onset_tol_ms: 50,
	pieces: [
		{
			piece_id: "decoy",
			composer: "X",
			title: "Decoy",
			chroma: new Array(12).fill(0),
			events: [16, 32, 64, 128],
		},
		{
			piece_id: "exact",
			composer: "Y",
			title: "Exact",
			chroma: (() => {
				const a = new Array(12).fill(0);
				a[0] = 0.5;
				a[4] = 0.5;
				a[7] = 0.5;
				return a;
			})(),
			events: [1, 16, 128, 1],
		},
	],
});

// C-E-G-C across 4 onsets -> 4 chord-events {0},{4},{7},{0}; matches "exact".
const MATCH_NOTES = [
	{ pitch: 60, onset: 0.0, offset: 0.4, velocity: 100 },
	{ pitch: 64, onset: 0.5, offset: 0.9, velocity: 100 },
	{ pitch: 67, onset: 1.0, offset: 1.4, velocity: 100 },
	{ pitch: 72, onset: 1.5, offset: 1.9, velocity: 100 },
];

const PREDICTIONS = {
	dynamics: 0.5,
	timing: 0.5,
	pedaling: 0.5,
	articulation: 0.5,
	phrasing: 0.5,
	interpretation: 0.5,
};

// Capture every WS frame the DO sends.
function recordingWs(): { ws: WebSocket; sent: unknown[] } {
	const sent: unknown[] = [];
	const ws = {
		send(data: string) {
			sent.push(JSON.parse(data));
		},
		close() {},
	} as unknown as WebSocket;
	return { ws, sent };
}

const evalChunk = (i: number, notes: typeof MATCH_NOTES) =>
	JSON.stringify({
		type: "eval_chunk",
		chunk_index: i,
		predictions: PREDICTIONS,
		midi_notes: notes,
		pedal_events: [],
	});

const readState = (storage: DurableObjectStorage) =>
	storage.get("state") as Promise<SessionState>;

describe("SessionBrain piece-ID v2 gate (eval_chunk path)", () => {
	it("locks to the in-catalog piece and emits piece_identified once the buffer crosses the threshold", async () => {
		await env.SCORES.put("fingerprint/v2/piece_index.json", V2_ARTIFACT);
		const stub = env.SESSION_BRAIN.get(
			env.SESSION_BRAIN.idFromName("pid-lock"),
		);
		await runInDurableObject(stub, async (inst: SessionBrain, state) => {
			const seeded = createInitialState("sess", "stud", null);
			seeded.baselinesLoaded = true; // skip Hyperdrive baseline query
			seeded.baselines = null;
			await state.storage.put("state", seeded);

			const { ws, sent } = recordingWs();
			// 11 four-note chunks => 44 notes accumulated > MIN_NOTES_FOR_IDENTIFICATION (30).
			for (let i = 0; i < 11; i++) {
				await inst.webSocketMessage(ws, evalChunk(i, MATCH_NOTES));
			}

			const st = await readState(state.storage);
			expect(st.pieceLocked).toBe(true);
			expect(st.pieceIdentification?.pieceId).toBe("exact");
			expect(st.pieceIdentification?.method).toBe("identify_v2");

			const identified = sent.find(
				(m): m is { type: string; pieceId: string } =>
					typeof m === "object" &&
					m !== null &&
					(m as { type?: string }).type === "piece_identified",
			);
			expect(identified?.pieceId).toBe("exact");
		});
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test src/do/session-brain.piece-id.test.ts
```
Expected: FAIL — the DO never locks because `handleEvalChunk` does no identification today: `st.pieceLocked` is `false` and no `piece_identified` frame is sent. (It must NOT pass before implementation; if it does, the test is wrong.)

- [ ] **Step 3: Implement the minimum to make the test pass**

**3a. Add named constants** near the existing `MIN_NOTES_FOR_IDENTIFICATION = 30;` (line ~99) in `apps/api/src/do/session-brain.ts`:

```typescript
const PIECE_ID_MARGIN_THRESHOLD = 0.0935;
const MAX_IDENTIFICATION_BUFFER = 1200;
```

**3b. Rewrite `tryIdentifyPiece`** — replace the entire method body (currently ~lines 2181–2274, the N-gram/rerank/DTW version) with the v2 version. New signature takes the accumulated buffer and returns the lock payload only when the gate locks:

```typescript
	/**
	 * Run the certified v2 gate over the accumulated identification buffer.
	 * Loads the v2 artifact text from SCORES R2 and calls the WASM margin gate.
	 * Returns the lock payload only when the gate clears the certified margin
	 * threshold; returns null on R2 miss, parse failure, or a non-locking result.
	 */
	private async tryIdentifyPiece(buffer: PerfNote[]): Promise<{
		pieceId: string;
		composer: string;
		title: string;
		confidence: number;
		method: string;
	} | null> {
		let artifactJson: string;
		try {
			const obj = await this.env.SCORES.get(
				"fingerprint/v2/piece_index.json",
			);
			if (!obj) {
				console.log(
					JSON.stringify({
						level: "warn",
						message: "v2 fingerprint artifact not found in SCORES R2",
					}),
				);
				return null;
			}
			artifactJson = await obj.text();
		} catch (err) {
			const error = err as Error;
			console.log(
				JSON.stringify({
					level: "warn",
					message: "v2 fingerprint load failed",
					error: error.message,
				}),
			);
			return null;
		}

		const result = wasm.identifyPiece(
			buffer,
			artifactJson,
			PIECE_ID_MARGIN_THRESHOLD,
		);
		if (result === null || !result.locked) return null;

		return {
			pieceId: result.piece_id,
			composer: result.composer,
			title: result.title,
			confidence: result.margin,
			method: "identify_v2",
		};
	}
```

**3c. Add the shared `accumulateAndIdentify` helper** immediately after `tryIdentifyPiece`:

```typescript
	/**
	 * Append this chunk's perf notes to the bounded identification buffer and,
	 * once it crosses MIN_NOTES_FOR_IDENTIFICATION, run the certified v2 gate on
	 * the accumulated buffer. On a confident lock, mutate the passed-in state and
	 * emit a piece_identified frame. Mutates `state` in place; the caller persists
	 * it under its own concurrency discipline. No-op once the piece is locked.
	 */
	private async accumulateAndIdentify(
		state: SessionState,
		perfNotes: PerfNote[],
		ws: WebSocket,
	): Promise<void> {
		if (state.pieceLocked || perfNotes.length === 0) return;

		state.identificationNoteBuffer.push(...perfNotes);
		if (state.identificationNoteBuffer.length > MAX_IDENTIFICATION_BUFFER) {
			state.identificationNoteBuffer = state.identificationNoteBuffer.slice(
				-MAX_IDENTIFICATION_BUFFER,
			);
		}

		if (state.identificationNoteBuffer.length < MIN_NOTES_FOR_IDENTIFICATION) {
			return;
		}

		try {
			const identified = await this.tryIdentifyPiece(
				state.identificationNoteBuffer,
			);
			if (identified !== null) {
				state.pieceLocked = true;
				state.pieceIdentification = {
					pieceId: identified.pieceId,
					confidence: identified.confidence,
					method: identified.method,
				};
				this.sendWs(ws, {
					type: "piece_identified",
					pieceId: identified.pieceId,
					composer: identified.composer,
					title: identified.title,
					confidence: identified.confidence,
					method: identified.method,
				});
			}
		} catch (err) {
			const error = err as Error;
			console.log(
				JSON.stringify({
					level: "warn",
					message: "piece identification skipped",
					error: error.message,
				}),
			);
		}
	}
```

**3d. Rewire `finalizeChunk`** — replace the block at ~lines 937–978 (`// 7. Try piece identification ...` through its closing brace) with:

```typescript
		// 7. Accumulate notes + try certified v2 piece identification
		await this.accumulateAndIdentify(currentState, perfNotes, ws);
```

**3e. Call the helper from `handleEvalChunk`** — in `handleEvalChunk`, immediately before the `// Persist updated state` block (the `state.accumulator = acc.toJSON();` line, ~1483), add:

```typescript
		// Accumulate notes + try certified v2 piece identification (eval-replay path)
		await this.accumulateAndIdentify(state, perfNotes, ws);
```

**3f. Add the `SessionState` import** if not already present. `session-brain.ts` imports from `./session-brain.schema` (e.g. `createInitialState`); add the `SessionState` type to that import. Confirm `PerfNote` is already imported from `../services/wasm-bridge` (it is, line ~52) and `wasm.identifyPiece` is available via `import * as wasm` (line ~60).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test src/do/session-brain.piece-id.test.ts
```
Expected: PASS — buffer accumulates to 44 notes across 11 chunks, the v2 gate locks `exact`, state and the `piece_identified` frame both reflect it.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.ts apps/api/src/do/session-brain.piece-id.test.ts && git commit -m "feat(#26): wire certified v2 piece-ID gate into SessionBrain DO"
```

---

### Task 3: Stay unknown on an ambiguous performance; lock only after crossing the note threshold

**Group:** A (depends on Task 2; last in Group A) `[SHIPS INDEPENDENTLY]` — after this task the certified gate is live at runtime; a real session yields v2 identification and no false locks.

**Behavior being verified:** (a) An ambiguous artifact (two equally-good candidates) never locks. (b) A single sub-threshold chunk does not lock; the lock fires only after enough chunks accumulate.
**Interface under test:** the `eval_chunk` WebSocket path of `SessionBrain` (public), with R2-seeded artifacts.

**Files:**
- Modify: `apps/api/src/do/session-brain.piece-id.test.ts`

This task adds the negative + cross-chunk-threshold behaviors. No new implementation is expected — the Task 2 code should already satisfy them; this task proves it (and would catch a too-low threshold or an off-by-one in the MIN-notes gate).

- [ ] **Step 1: Write the failing test**

Append two `it` blocks inside the existing `describe("SessionBrain piece-ID v2 gate (eval_chunk path)", ...)` in `apps/api/src/do/session-brain.piece-id.test.ts`. Add this ambiguous artifact constant near `V2_ARTIFACT` at the top of the file:

```typescript
// Two identical candidates => margin 0 < threshold => never locks (OOD/ambiguous proxy).
const V2_AMBIGUOUS = JSON.stringify({
	version: "v2",
	onset_tol_ms: 50,
	pieces: [
		{
			piece_id: "a",
			composer: "X",
			title: "A",
			chroma: (() => {
				const c = new Array(12).fill(0);
				c[0] = 0.5;
				c[4] = 0.5;
				c[7] = 0.5;
				return c;
			})(),
			events: [1, 16, 128, 1],
		},
		{
			piece_id: "b",
			composer: "X",
			title: "B",
			chroma: (() => {
				const c = new Array(12).fill(0);
				c[0] = 0.5;
				c[4] = 0.5;
				c[7] = 0.5;
				return c;
			})(),
			events: [1, 16, 128, 1],
		},
	],
});
```

Then the two tests:

```typescript
	it("stays unknown on an ambiguous performance (no false lock)", async () => {
		await env.SCORES.put("fingerprint/v2/piece_index.json", V2_AMBIGUOUS);
		const stub = env.SESSION_BRAIN.get(
			env.SESSION_BRAIN.idFromName("pid-ambiguous"),
		);
		await runInDurableObject(stub, async (inst: SessionBrain, state) => {
			const seeded = createInitialState("sess", "stud", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			await state.storage.put("state", seeded);

			const { ws, sent } = recordingWs();
			for (let i = 0; i < 11; i++) {
				await inst.webSocketMessage(ws, evalChunk(i, MATCH_NOTES));
			}

			const st = await readState(state.storage);
			expect(st.pieceLocked).toBe(false);
			expect(st.pieceIdentification).toBeNull();
			expect(sent.some((m) => (m as { type?: string }).type === "piece_identified")).toBe(false);
		});
	});

	it("does not lock after a single sub-threshold chunk", async () => {
		await env.SCORES.put("fingerprint/v2/piece_index.json", V2_ARTIFACT);
		const stub = env.SESSION_BRAIN.get(
			env.SESSION_BRAIN.idFromName("pid-subthreshold"),
		);
		await runInDurableObject(stub, async (inst: SessionBrain, state) => {
			const seeded = createInitialState("sess", "stud", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			await state.storage.put("state", seeded);

			const { ws } = recordingWs();
			// One 4-note chunk: 4 < MIN_NOTES_FOR_IDENTIFICATION (30) -> no identification yet.
			await inst.webSocketMessage(ws, evalChunk(0, MATCH_NOTES));

			const st = await readState(state.storage);
			expect(st.pieceLocked).toBe(false);
			expect(st.identificationNoteBuffer).toHaveLength(4);
		});
	});
```

- [ ] **Step 2: Run test — verify it FAILS (then confirm it is a real assertion, not a no-op)**

```bash
cd apps/api && bun run test src/do/session-brain.piece-id.test.ts
```
Expected: the two new tests should PASS immediately against the Task 2 implementation (ambiguous margin 0 < 0.0935 → no lock; 4 < 30 → no identify). Because no new production behavior is needed, this is the one task where the test legitimately passes on first run — it pins behavior the spec requires that Task 2's single happy-path test did not cover (the negative + threshold cases). To prove the tests are real (not shape-only), temporarily set `PIECE_ID_MARGIN_THRESHOLD = -1` in `session-brain.ts`, re-run, and confirm the "ambiguous" test now FAILS (it would lock). Restore `0.0935` before committing. Record the observed failure in the commit body.

- [ ] **Step 3: Implement the minimum to make the test pass**

No production change required (Task 2 already satisfies these). If the falsification probe in Step 2 revealed the threshold is not actually gating, fix the gate wiring in `accumulateAndIdentify`/`tryIdentifyPiece` until both new tests pass with `PIECE_ID_MARGIN_THRESHOLD = 0.0935`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test src/do/session-brain.piece-id.test.ts
```
Expected: PASS — all three scenarios (lock / ambiguous-no-lock / sub-threshold-no-lock) green, with the threshold restored to `0.0935`.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.piece-id.test.ts && git commit -m "test(#26): assert v2 gate stays unknown on ambiguous + sub-threshold input"
```

---

### Task 4: Remove the legacy bridge wrappers and interfaces

**Group:** B (first; depends on all of Group A — the DO must no longer reference legacy wrappers)

**Behavior being verified:** The `identifyPiece` wrapper still forwards to the WASM gate; the removed legacy wrappers (`ngramRecall`/`rerankCandidates`/`dtwConfirm`) are gone and nothing references them.
**Interface under test:** `wasm-bridge.ts:identifyPiece` (kept) via its existing mock test; absence of the legacy wrappers via typecheck + grep.

**Files:**
- Modify: `apps/api/src/services/wasm-bridge.ts`
- Modify: `apps/api/src/services/wasm-bridge.test.ts`
- Modify: `apps/api/src/services/wasm-bridge.workerd.test.ts`

- [ ] **Step 1: Write the failing test**

First confirm the runtime no longer needs the legacy path (must print nothing):

```bash
cd apps/api && grep -n "ngramRecall\|rerankCandidates\|dtwConfirm" src/do/session-brain.ts
```

Then convert the kept bridge test into the guard for this task. In `apps/api/src/services/wasm-bridge.test.ts`, **remove** the `describe("ngramRecall", ...)` block (lines ~86–97) and the three legacy mocks it depends on (`mockNgramRecall`, `mockRerankCandidates`, `mockDtwConfirm`) plus their entries in the `vi.mock("../wasm/piece-identify/pkg/piece_identify", ...)` factory — keeping `identify_piece: mockIdentifyPiece`. The remaining `describe("identifyPiece", ...)` block stays as the behavior pin. In `apps/api/src/services/wasm-bridge.workerd.test.ts`, **remove** the `ngramRecall` import (line 19) and the `it("ngramRecall executes without 'is not a function' error", ...)` block (lines ~34–44); keep the `identifyPiece (real WASM)` describe.

The "failing test" for this deletion task is the typecheck/grep guard: after editing the bridge in Step 3, the kept tests must still pass and no reference may remain.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run typecheck
```
Expected: BEFORE the Step-3 edit, typecheck PASSES (wrappers still exist). To see the guard fail, the deletion in Step 3 must come first; this task's true failing signal is: if you delete the bridge wrappers (3) but leave a test importing `ngramRecall`, `bun run test src/services/wasm-bridge.workerd.test.ts` FAILS with `ngramRecall is not exported`. Run it after Step 3 to confirm green.

- [ ] **Step 3: Implement — delete the legacy wrappers and interfaces**

In `apps/api/src/services/wasm-bridge.ts`:
- Delete the interfaces `NgramCandidate`, `RerankResult`, `DtwConfirmResult` (lines ~167–181) and the type aliases `NgramIndex`, `RerankFeatures` (lines ~194–198). Keep `TextMatchResult` and `CatalogEntry`.
- Delete the three wrapper functions `ngramRecall` (lines ~329–339), `rerankCandidates` (lines ~341–355), and `dtwConfirm` (lines ~357–375). Keep `identifyPiece` and its `IdentifyResult` interface.

In `apps/api/src/do/session-brain.ts`, remove the now-unused imports `NgramIndex` and `RerankFeatures` from the `import type { ... } from "../services/wasm-bridge"` block (lines ~50, ~54). (After Task 2 they are already unreferenced in the method bodies; this removes the dangling type imports so typecheck stays clean.)

Apply the test edits described in Step 1.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run typecheck && bun run test src/services/wasm-bridge.test.ts src/services/wasm-bridge.workerd.test.ts && grep -rn "ngramRecall\|rerankCandidates\|dtwConfirm\|NgramIndex\|RerankFeatures\|NgramCandidate\|RerankResult\|DtwConfirmResult" src/ --include="*.ts" || true
```
Expected: typecheck clean; bridge tests PASS; the final grep prints NOTHING (no remaining references in TS).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/wasm-bridge.ts apps/api/src/services/wasm-bridge.test.ts apps/api/src/services/wasm-bridge.workerd.test.ts apps/api/src/do/session-brain.ts && git commit -m "refactor(#26): delete legacy piece-ID bridge wrappers + interfaces"
```

---

### Task 5: Delete the legacy Rust modules and exports; rebuild WASM

**Group:** B (depends on Task 4 — the bridge must already not import the legacy exports)

**Behavior being verified:** The piece-identify crate compiles and its tests pass with only the certified gate + text-match surface; the legacy exports are gone; the rebuilt WASM pkg still exposes `identify_piece` (the bridge + DO tests still pass).
**Interface under test:** `cargo test` for the crate; the rebuilt WASM via the kept `identifyPiece (real WASM)` workerd test.

**Files:**
- Delete: `apps/api/src/wasm/piece-identify/src/ngram.rs`
- Delete: `apps/api/src/wasm/piece-identify/src/rerank.rs`
- Delete: `apps/api/src/wasm/piece-identify/src/dtw_confirm.rs`
- Delete: `apps/api/src/wasm/piece-identify/src/real_recording_test.rs`
- Modify: `apps/api/src/wasm/piece-identify/src/lib.rs`
- Modify: `apps/api/src/wasm/piece-identify/src/types.rs`
- Modify (rebuild output): `apps/api/src/wasm/piece-identify/pkg/*`

- [ ] **Step 1: Write the failing test**

The guard is `cargo test`. First, delete the four `.rs` files:

```bash
cd apps/api/src/wasm/piece-identify/src && rm ngram.rs rerank.rs dtw_confirm.rs real_recording_test.rs
```

This alone breaks compilation (`lib.rs` still has `mod ngram;` etc.), which is the failing signal.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api/src/wasm/piece-identify && cargo test
```
Expected: FAIL — `error[E0583]: file not found for module 'ngram'` (and `rerank`, `dtw_confirm`, `real_recording_test`), because `lib.rs` still declares the deleted modules.

- [ ] **Step 3: Implement — drop the legacy module decls, exports, and types**

In `apps/api/src/wasm/piece-identify/src/lib.rs`:
- Remove the module declarations `mod dtw_confirm;`, `mod ngram;`, `mod rerank;` (lines ~13–15) and the `#[cfg(test)] mod real_recording_test;` block (lines ~19–20). Keep `mod chroma; mod gate; mod identify; mod text_match; mod types;` and `#[cfg(test)] mod parity_test;`.
- Remove the four `#[wasm_bindgen]` exports and their doc comments: `ngram_recall` (Stage 1 block), `compute_rerank_features` (Stage 2a), `rerank_candidates` (Stage 2b), and `dtw_confirm` (Stage 3) — the entire span from the `// Stage 1: N-gram recall` banner (~line 25) through the end of the `dtw_confirm` fn (~line 127). Keep `match_piece_text`, `identify_piece`, and the `#[cfg(test)] mod lib_tests` smoke test.

In `apps/api/src/wasm/piece-identify/src/types.rs`:
- Remove the legacy type aliases `NgramIndex`, `RerankFeatures` (lines ~7–13) and the structs `NgramCandidate`, `RerankResult`, `DtwConfirmResult` (lines ~24–47). Keep `PerfNote`, `TextMatchResult`, `CatalogEntry`, `PieceArtifact`, `PieceIndex`, `IdentifyResult`.

- [ ] **Step 4: Run test — verify it PASSES (cargo, then rebuild WASM, then bridge/DO tests)**

```bash
cd apps/api/src/wasm/piece-identify && cargo test
```
Expected: PASS — crate compiles with only the certified + text-match surface; `parity_test` and `lib_tests`/`gate`/`identify` tests green.

Then rebuild the WASM pkg and re-run the real-WASM consumers (proves the rebuilt pkg still exports `identify_piece`):

```bash
cd apps/api && bun run build:wasm && bun run typecheck && bun run test src/services/wasm-bridge.workerd.test.ts src/do/session-brain.piece-id.test.ts
```
Expected: build succeeds; typecheck clean; both real-WASM/DO tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/wasm/piece-identify/src apps/api/src/wasm/piece-identify/pkg && git commit -m "refactor(#26): delete legacy piece-ID Rust modules/exports; rebuild WASM"
```

---

### Task 6: Drop the vacuous real_recording skip filter from the justfile

**Group:** B (depends on Task 5 — the `real_recording_test` it referenced no longer exists)

**Behavior being verified:** `just test-piece-id` runs the crate tests without referencing the deleted `real_recording` test.
**Interface under test:** the `test-piece-id` just recipe.

**Files:**
- Modify: `justfile`

- [ ] **Step 1: Write the failing test**

The guard is running the recipe. Inspect the current recipe:

```bash
grep -n -A1 "^test-piece-id:" justfile
```
It reads `cd apps/api/src/wasm/piece-identify && cargo test -- --skip real_recording`. The `--skip real_recording` filter is now vacuous (no such test exists). The "failing" signal is that the recipe still carries a dead filter referencing a deleted test.

- [ ] **Step 2: Run test — verify it FAILS (carries the stale filter)**

```bash
grep -q -- "--skip real_recording" justfile && echo "STALE FILTER PRESENT" || echo "clean"
```
Expected: prints `STALE FILTER PRESENT`.

- [ ] **Step 3: Implement — remove the stale filter**

In `justfile`, change the `test-piece-id` recipe body from:

```
    cd apps/api/src/wasm/piece-identify && cargo test -- --skip real_recording
```

to:

```
    cd apps/api/src/wasm/piece-identify && cargo test
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
grep -q -- "--skip real_recording" justfile && echo "STALE FILTER PRESENT" || echo "clean" && just test-piece-id
```
Expected: prints `clean`; `just test-piece-id` runs the crate tests green.

- [ ] **Step 5: Commit**

```bash
git add justfile && git commit -m "chore(#26): drop vacuous real_recording skip from test-piece-id recipe"
```

---

## Final verification (run after all tasks; not a task itself)

```bash
cd apps/api && bun run typecheck && bun run test && bun run test:scripts
cd apps/api/src/wasm/piece-identify && cargo test
```
Expected: typecheck clean; workerd `bun run test` green (including the new `session-brain.piece-id.test.ts`); `bun run test:scripts` (node) green; `cargo test` green. No new failures beyond the documented pre-existing catalog/harness ones (`src/harness/skills/__catalog__/**`, `validator.test.ts`, already excluded by `vitest.config.ts`). `just test-model` is untouched by this work and need not be re-run.

PR body must include `Closes #26`.
