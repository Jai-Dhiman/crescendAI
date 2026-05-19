# Play Passage Artifact Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task). Within a group, tasks are sequential (each task modifies the same files as its predecessor).
> Do NOT start running tasks until `/challenge` returns `VERDICT: PROCEED`.

**Goal:** Let the teacher LLM say "listen here" — emitting a chat artifact that plays a bar-bounded slice of the student's own recording, score-aligned, with a tinted focus sub-range.
**Spec:** `docs/specs/2026-05-18-play-passage-artifact-design.md`
**Style:** Follow `apps/api/TS_STYLE.md` for code in `apps/api/`. Web app follows existing conventions in `apps/web/src/`.

---

## Task Groups

Five parallel tracks, three phases:

- **Phase 1 (parallel tracks A, B, C, D):**
  - **Track A** (Tasks 1 → 2 → 3): `buildPassageManifest` service. Sequential within track.
  - **Track B** (Tasks 4 → 5 → 6): `PassagePlayer` client library. Sequential within track.
  - **Track C** (Tasks 7 → 8 → 9): `play_passage` tool, types, `reference_browser` removal. Sequential within track.
  - **Track D** (Task 10): Chunk byte read route.
- **Phase 2 (depends on Tracks A, C, D):**
  - **Track E** (Tasks 11 → 12 → 13): Sessions route + DO `/passage` handler. Sequential within track.
- **Phase 3 (depends on Tracks B, C, E):**
  - **Track F** (Tasks 14 → 15 → 16): `PlayPassageCard`. Sequential within track.
  - **Track G** (Task 17): Artifact collapsed-props wiring. Parallel with Track F.

A build agent can dispatch Tracks A/B/C/D in parallel from the start. After all complete, dispatch Track E. After Track E and Track B/C complete, dispatch Track F. Track G can run alongside Track F after Task 14.

---

## Track A: `buildPassageManifest` service

### Task 1: buildPassageManifest happy path
**Group:** A (parallel with B, C, D)

**Behavior being verified:** Given enriched chunks whose `bar_coverage` covers bars `[N, M]`, the manifest returns the chunks spanning that range, a `startOffsetSec` at bar N's earliest alignment, an `endOffsetSec` at bar M's latest alignment, and a `barTimeline` with one entry per bar.
**Interface under test:** `buildPassageManifest(args)`

**Files:**
- Create: `apps/api/src/services/passage-manifest.ts`
- Test: `apps/api/src/services/passage-manifest.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/passage-manifest.test.ts
import { describe, expect, it } from "vitest";
import { buildPassageManifest } from "./passage-manifest";
import type { EnrichedChunk } from "../do/session-brain";

const baseUrl = "https://api.example.com";

function chunk(
	chunkIndex: number,
	barCoverage: [number, number] | null,
	alignments: Array<{ bar: number; expected_onset_ms: number }>,
): EnrichedChunk {
	return {
		chunkIndex,
		muq_scores: [0, 0, 0, 0, 0, 0],
		midi_notes: [],
		pedal_cc: [],
		alignment: alignments.map((a, i) => ({
			perf_index: i,
			score_index: i,
			expected_onset_ms: a.expected_onset_ms,
			bar: a.bar,
		})),
		bar_coverage: barCoverage,
	};
}

describe("buildPassageManifest", () => {
	it("builds a manifest covering the requested bar range", () => {
		const enrichedChunks: EnrichedChunk[] = [
			chunk(0, [1, 4], [
				{ bar: 1, expected_onset_ms: 0 },
				{ bar: 2, expected_onset_ms: 4000 },
				{ bar: 3, expected_onset_ms: 8000 },
				{ bar: 4, expected_onset_ms: 12000 },
			]),
			chunk(1, [5, 8], [
				{ bar: 5, expected_onset_ms: 1000 },
				{ bar: 6, expected_onset_ms: 5000 },
				{ bar: 7, expected_onset_ms: 9000 },
				{ bar: 8, expected_onset_ms: 13000 },
			]),
			chunk(2, [9, 12], [
				{ bar: 9, expected_onset_ms: 2000 },
			]),
		];

		const result = buildPassageManifest({
			enrichedChunks,
			bars: [5, 8],
			pieceId: "chopin.ballades.1",
			sessionId: "00000000-0000-0000-0000-0000000000aa",
			baseUrl,
		});

		expect("error" in result).toBe(false);
		if ("error" in result) return;
		expect(result.pieceId).toBe("chopin.ballades.1");
		expect(result.bars).toEqual([5, 8]);
		expect(result.source).toEqual({
			kind: "session",
			sessionId: "00000000-0000-0000-0000-0000000000aa",
		});
		expect(result.chunks).toHaveLength(1);
		expect(result.chunks[0].chunkIndex).toBe(1);
		expect(result.chunks[0].url).toBe(
			`${baseUrl}/api/practice/chunk?sessionId=00000000-0000-0000-0000-0000000000aa&chunkIndex=1`,
		);
		expect(result.chunks[0].durationSec).toBe(15);
		expect(result.startOffsetSec).toBe(1.0);
		expect(result.endOffsetSec).toBe(13.0);
		expect(result.barTimeline).toEqual([
			{ bar: 5, tSec: 0 },
			{ bar: 6, tSec: 4 },
			{ bar: 7, tSec: 8 },
			{ bar: 8, tSec: 12 },
		]);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Run command:

```
cd apps/api && bun test src/services/passage-manifest.test.ts
```

Expected: FAIL — `Cannot find module './passage-manifest'` (the implementation file does not exist yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/services/passage-manifest.ts
import type { EnrichedChunk } from "../do/session-brain";

const CHUNK_DURATION_SEC = 15;

export interface PassageManifest {
	source: { kind: "session"; sessionId: string };
	pieceId: string;
	bars: [number, number];
	chunks: Array<{ url: string; chunkIndex: number; durationSec: number }>;
	startOffsetSec: number;
	endOffsetSec: number;
	barTimeline: Array<{ bar: number; tSec: number }>;
}

export type PassageManifestError = { error: "no_alignment" | "out_of_range" };

export interface BuildPassageManifestArgs {
	enrichedChunks: EnrichedChunk[];
	bars: [number, number];
	pieceId: string;
	sessionId: string;
	baseUrl: string;
}

export function buildPassageManifest(
	args: BuildPassageManifestArgs,
): PassageManifest | PassageManifestError {
	const [startBar, endBar] = args.bars;

	const covering = args.enrichedChunks.filter(
		(c) =>
			c.bar_coverage !== null &&
			c.bar_coverage[1] >= startBar &&
			c.bar_coverage[0] <= endBar,
	);

	if (covering.length === 0) {
		return { error: "out_of_range" };
	}

	covering.sort((a, b) => a.chunkIndex - b.chunkIndex);
	const firstChunkIndex = covering[0].chunkIndex;

	const barTimeline: Array<{ bar: number; tSec: number }> = [];

	for (let bar = startBar; bar <= endBar; bar++) {
		for (const c of covering) {
			const hit = c.alignment.find((a) => a.bar === bar);
			if (hit !== undefined) {
				const tSec =
					(c.chunkIndex - firstChunkIndex) * CHUNK_DURATION_SEC +
					hit.expected_onset_ms / 1000;
				barTimeline.push({ bar, tSec });
				break;
			}
		}
	}

	if (barTimeline.length === 0) {
		return { error: "no_alignment" };
	}

	const startOffsetSec = barTimeline[0].tSec;
	const lastBarHit = barTimeline[barTimeline.length - 1];
	const endOffsetSec = lastBarHit.tSec;
	for (const entry of barTimeline) {
		entry.tSec -= startOffsetSec;
	}

	return {
		source: { kind: "session", sessionId: args.sessionId },
		pieceId: args.pieceId,
		bars: args.bars,
		chunks: covering.map((c) => ({
			url: `${args.baseUrl}/api/practice/chunk?sessionId=${args.sessionId}&chunkIndex=${c.chunkIndex}`,
			chunkIndex: c.chunkIndex,
			durationSec: CHUNK_DURATION_SEC,
		})),
		startOffsetSec,
		endOffsetSec,
		barTimeline,
	};
}
```

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/api && bun test src/services/passage-manifest.test.ts
```

Expected: PASS

- [ ] **Step 5: Commit**

Run command:

```
git add apps/api/src/services/passage-manifest.ts apps/api/src/services/passage-manifest.test.ts && git commit -m "feat(api): add buildPassageManifest service (happy path)"
```

---

### Task 2: buildPassageManifest distinguishes no_alignment from out_of_range
**Group:** A (sequential after Task 1)

**Behavior being verified:** When no chunks in the session have any alignment data (`bar_coverage === null` for all), the function returns `{ error: "no_alignment" }` rather than `out_of_range`. The semantic difference matters: "we have no alignment data" versus "you asked about bars that aren't in our alignment range."
**Interface under test:** `buildPassageManifest(args)`

**Files:**
- Modify: `apps/api/src/services/passage-manifest.ts`
- Test: `apps/api/src/services/passage-manifest.test.ts`

- [ ] **Step 1: Write the failing test**

Append inside the existing `describe` block:

```typescript
it("returns no_alignment when no chunks have bar_coverage at all", () => {
	const enrichedChunks: EnrichedChunk[] = [
		chunk(0, null, []),
		chunk(1, null, []),
	];

	const result = buildPassageManifest({
		enrichedChunks,
		bars: [5, 8],
		pieceId: "chopin.ballades.1",
		sessionId: "00000000-0000-0000-0000-0000000000aa",
		baseUrl,
	});

	expect("error" in result && result.error).toBe("no_alignment");
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Run command:

```
cd apps/api && bun test src/services/passage-manifest.test.ts
```

Expected: FAIL — `expected 'out_of_range' to be 'no_alignment'` (current implementation returns `out_of_range` because the filter sees no covering chunks).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/passage-manifest.ts`, replace the early-return block (the `if (covering.length === 0) return { error: "out_of_range" }`) with:

```typescript
const anyAlignedChunk = args.enrichedChunks.some(
	(c) => c.bar_coverage !== null,
);
if (!anyAlignedChunk) {
	return { error: "no_alignment" };
}
if (covering.length === 0) {
	return { error: "out_of_range" };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/api && bun test src/services/passage-manifest.test.ts
```

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

Run command:

```
git add apps/api/src/services/passage-manifest.ts apps/api/src/services/passage-manifest.test.ts && git commit -m "feat(api): distinguish no_alignment from out_of_range in passage manifest"
```

---

### Task 3: buildPassageManifest out_of_range regression guard
**Group:** A (sequential after Task 2)

**Behavior being verified:** When alignment data exists for the session but no chunk covers the requested bar range, returns `{ error: "out_of_range" }`. Locks in the contract differentiating this from `no_alignment`.
**Interface under test:** `buildPassageManifest(args)`

**Files:**
- Test: `apps/api/src/services/passage-manifest.test.ts`

- [ ] **Step 1: Write the failing test**

Append inside the existing `describe` block:

```typescript
it("returns out_of_range when alignment exists but does not cover requested bars", () => {
	const enrichedChunks: EnrichedChunk[] = [
		chunk(0, [1, 3], [
			{ bar: 1, expected_onset_ms: 0 },
			{ bar: 2, expected_onset_ms: 5000 },
			{ bar: 3, expected_onset_ms: 10000 },
		]),
	];

	const result = buildPassageManifest({
		enrichedChunks,
		bars: [5, 8],
		pieceId: "chopin.ballades.1",
		sessionId: "00000000-0000-0000-0000-0000000000aa",
		baseUrl,
	});

	expect("error" in result && result.error).toBe("out_of_range");
});
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

Run command:

```
cd apps/api && bun test src/services/passage-manifest.test.ts
```

Expected behavior: this test PASSES on Task 2's implementation (which already differentiates these cases). This task is a regression guard, not driving new code.

- [ ] **Step 3: No new implementation required** — behavior is already correct from Task 2.

- [ ] **Step 4: Run all passage-manifest tests — verify PASS**

Run command:

```
cd apps/api && bun test src/services/passage-manifest.test.ts
```

Expected: PASS (three tests total).

- [ ] **Step 5: Commit**

Run command:

```
git add apps/api/src/services/passage-manifest.test.ts && git commit -m "test(api): lock in out_of_range behavior for passage manifest"
```

---

## Track B: `PassagePlayer` client library

### Task 4: PassagePlayer.load() transitions to ready state
**Group:** B (parallel with A, C, D)

**Behavior being verified:** After construction with a manifest and a successful `load()`, state is `"ready"` and `duration` equals the last bar's tSec plus 1.0s padding past bar M.
**Interface under test:** `new PassagePlayer(manifest, audioContext).load()` and `.state` and `.duration`.

**Files:**
- Create: `apps/web/src/lib/passage-player.ts`
- Test: `apps/web/src/lib/passage-player.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/passage-player.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PassagePlayer } from "./passage-player";

type PassageManifest = {
	source: { kind: "session"; sessionId: string };
	pieceId: string;
	bars: [number, number];
	chunks: Array<{ url: string; chunkIndex: number; durationSec: number }>;
	startOffsetSec: number;
	endOffsetSec: number;
	barTimeline: Array<{ bar: number; tSec: number }>;
};

const manifest: PassageManifest = {
	source: { kind: "session", sessionId: "s1" },
	pieceId: "chopin.ballades.1",
	bars: [5, 8],
	chunks: [{ url: "https://api/c1.webm", chunkIndex: 1, durationSec: 15 }],
	startOffsetSec: 1.0,
	endOffsetSec: 13.0,
	barTimeline: [
		{ bar: 5, tSec: 0 },
		{ bar: 6, tSec: 4 },
		{ bar: 7, tSec: 8 },
		{ bar: 8, tSec: 12 },
	],
};

function makeStubAudioContext() {
	const decodedBuffer = { duration: 15, length: 15 * 44100 } as AudioBuffer;
	const startCalls: Array<{ when: number; offset: number; duration?: number }> = [];
	const ctx = {
		currentTime: 0,
		state: "running",
		decodeAudioData: vi.fn().mockResolvedValue(decodedBuffer),
		createBufferSource: vi.fn(() => ({
			buffer: null as AudioBuffer | null,
			connect: vi.fn(),
			start: vi.fn((when?: number, offset?: number, duration?: number) => {
				startCalls.push({ when: when ?? 0, offset: offset ?? 0, duration });
			}),
			stop: vi.fn(),
			onended: null as (() => void) | null,
		})),
		destination: {},
		resume: vi.fn().mockResolvedValue(undefined),
		close: vi.fn().mockResolvedValue(undefined),
	} as unknown as AudioContext;
	return { ctx, startCalls };
}

beforeEach(() => {
	globalThis.fetch = vi
		.fn()
		.mockResolvedValue(
			new Response(new ArrayBuffer(1024), { status: 200 }),
		) as typeof fetch;
});

describe("PassagePlayer", () => {
	it("transitions to ready and exposes duration after load", async () => {
		const { ctx } = makeStubAudioContext();
		const player = new PassagePlayer(manifest, ctx);
		expect(player.state).toBe("idle");
		await player.load();
		expect(player.state).toBe("ready");
		expect(player.duration).toBe(13);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Run command:

```
cd apps/web && bun test src/lib/passage-player.test.ts
```

Expected: FAIL — `Cannot find module './passage-player'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/web/src/lib/passage-player.ts
type PassageManifest = {
	source: { kind: "session"; sessionId: string };
	pieceId: string;
	bars: [number, number];
	chunks: Array<{ url: string; chunkIndex: number; durationSec: number }>;
	startOffsetSec: number;
	endOffsetSec: number;
	barTimeline: Array<{ bar: number; tSec: number }>;
};

type PlayerState = "idle" | "loading" | "ready" | "playing" | "paused" | "ended" | "error";

export class PassagePlayer {
	state: PlayerState = "idle";
	duration = 0;
	private buffers: AudioBuffer[] = [];
	private manifest: PassageManifest;
	private ctx: AudioContext;

	constructor(manifest: PassageManifest, ctx: AudioContext) {
		this.manifest = manifest;
		this.ctx = ctx;
	}

	async load(): Promise<void> {
		this.state = "loading";
		try {
			const responses = await Promise.all(
				this.manifest.chunks.map((c) => fetch(c.url, { credentials: "include" })),
			);
			const arrayBuffers = await Promise.all(responses.map((r) => r.arrayBuffer()));
			this.buffers = await Promise.all(
				arrayBuffers.map((ab) => this.ctx.decodeAudioData(ab)),
			);
			const last = this.manifest.barTimeline[this.manifest.barTimeline.length - 1];
			this.duration = last !== undefined ? last.tSec + 1.0 : 0;
			this.state = "ready";
		} catch (err) {
			this.state = "error";
			throw err;
		}
	}
}
```

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/web && bun test src/lib/passage-player.test.ts
```

Expected: PASS

- [ ] **Step 5: Commit**

Run command:

```
git add apps/web/src/lib/passage-player.ts apps/web/src/lib/passage-player.test.ts && git commit -m "feat(web): add PassagePlayer with load-to-ready state"
```

---

### Task 5: PassagePlayer.play() schedules sources and emits monotonic ticks
**Group:** B (sequential after Task 4)

**Behavior being verified:** After `play()`, the player schedules `AudioBufferSourceNode.start()` with the correct offset for the first chunk (`startOffsetSec`), state becomes `"playing"`, and `onTick(cb)` receives monotonically increasing tSec values driven by `AudioContext.currentTime`.
**Interface under test:** `player.play()`, `player.onTick(cb)`, `player.state`.

**Files:**
- Modify: `apps/web/src/lib/passage-player.ts`
- Test: `apps/web/src/lib/passage-player.test.ts`

- [ ] **Step 1: Write the failing test**

Append to the existing `describe` block:

```typescript
it("play() schedules source with startOffsetSec and emits monotonic ticks", async () => {
	const { ctx, startCalls } = makeStubAudioContext();
	const player = new PassagePlayer(manifest, ctx);
	await player.load();

	const ticks: number[] = [];
	player.onTick((t) => ticks.push(t));

	player.play();
	expect(player.state).toBe("playing");
	expect(startCalls).toHaveLength(1);
	expect(startCalls[0].offset).toBe(1.0);

	(ctx as unknown as { currentTime: number }).currentTime = 0.05;
	await player.__testTick();
	(ctx as unknown as { currentTime: number }).currentTime = 0.1;
	await player.__testTick();
	(ctx as unknown as { currentTime: number }).currentTime = 0.2;
	await player.__testTick();

	expect(ticks.length).toBeGreaterThanOrEqual(3);
	for (let i = 1; i < ticks.length; i++) {
		expect(ticks[i]).toBeGreaterThan(ticks[i - 1]);
	}
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Run command:

```
cd apps/web && bun test src/lib/passage-player.test.ts
```

Expected: FAIL — `player.play is not a function` (or similar).

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the body of `PassagePlayer` in `apps/web/src/lib/passage-player.ts` with:

```typescript
type TickCb = (tSec: number) => void;

export class PassagePlayer {
	state: PlayerState = "idle";
	duration = 0;
	private buffers: AudioBuffer[] = [];
	private manifest: PassageManifest;
	private ctx: AudioContext;
	private sources: AudioBufferSourceNode[] = [];
	private tickCallbacks = new Set<TickCb>();
	private playStartedAtCtxTime = 0;

	constructor(manifest: PassageManifest, ctx: AudioContext) {
		this.manifest = manifest;
		this.ctx = ctx;
	}

	async load(): Promise<void> {
		this.state = "loading";
		try {
			const responses = await Promise.all(
				this.manifest.chunks.map((c) => fetch(c.url, { credentials: "include" })),
			);
			const arrayBuffers = await Promise.all(responses.map((r) => r.arrayBuffer()));
			this.buffers = await Promise.all(
				arrayBuffers.map((ab) => this.ctx.decodeAudioData(ab)),
			);
			const last = this.manifest.barTimeline[this.manifest.barTimeline.length - 1];
			this.duration = last !== undefined ? last.tSec + 1.0 : 0;
			this.state = "ready";
		} catch (err) {
			this.state = "error";
			throw err;
		}
	}

	onTick(cb: TickCb): () => void {
		this.tickCallbacks.add(cb);
		return () => this.tickCallbacks.delete(cb);
	}

	play(): void {
		if (this.state !== "ready" && this.state !== "paused") return;
		this.playStartedAtCtxTime = this.ctx.currentTime;
		const baseWhen = this.ctx.currentTime;
		let cursor = 0;
		this.sources = [];
		for (let i = 0; i < this.buffers.length; i++) {
			const buf = this.buffers[i];
			const source = this.ctx.createBufferSource();
			source.buffer = buf;
			source.connect(this.ctx.destination);
			const offset = i === 0 ? this.manifest.startOffsetSec : 0;
			const remaining =
				i === this.buffers.length - 1
					? this.manifest.endOffsetSec - offset
					: buf.duration - offset;
			source.start(baseWhen + cursor, offset, remaining);
			this.sources.push(source);
			cursor += remaining;
		}
		this.state = "playing";
		this.startRafLoop();
	}

	private startRafLoop(): void {
		const tick = () => {
			if (this.state !== "playing") return;
			const t = this.ctx.currentTime - this.playStartedAtCtxTime;
			for (const cb of this.tickCallbacks) cb(t);
			if (typeof requestAnimationFrame !== "undefined") {
				requestAnimationFrame(tick);
			}
		};
		if (typeof requestAnimationFrame !== "undefined") {
			requestAnimationFrame(tick);
		}
	}

	async __testTick(): Promise<void> {
		const t = this.ctx.currentTime - this.playStartedAtCtxTime;
		for (const cb of this.tickCallbacks) cb(t);
	}
}
```

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/web && bun test src/lib/passage-player.test.ts
```

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

Run command:

```
git add apps/web/src/lib/passage-player.ts apps/web/src/lib/passage-player.test.ts && git commit -m "feat(web): PassagePlayer schedules sequential playback and emits ticks"
```

---

### Task 6: PassagePlayer.pause() halts further ticks
**Group:** B (sequential after Task 5)

**Behavior being verified:** After `pause()`, state becomes `"paused"`, scheduled sources are stopped, and the source's `stop()` method is invoked.
**Interface under test:** `player.pause()`, `player.state`.

**Files:**
- Modify: `apps/web/src/lib/passage-player.ts`
- Test: `apps/web/src/lib/passage-player.test.ts`

- [ ] **Step 1: Write the failing test**

Append to the existing `describe` block:

```typescript
it("pause() stops scheduled sources and transitions to paused", async () => {
	const { ctx } = makeStubAudioContext();
	const player = new PassagePlayer(manifest, ctx);
	await player.load();
	player.play();
	expect(player.state).toBe("playing");

	player.pause();
	expect(player.state).toBe("paused");

	type SourceMock = { stop: { mock: { calls: unknown[] } } };
	type CtxWithMock = AudioContext & {
		createBufferSource: { mock: { results: Array<{ value: SourceMock }> } };
	};
	const results = (ctx as CtxWithMock).createBufferSource.mock.results;
	const totalStopCalls = results.reduce(
		(acc, r) => acc + r.value.stop.mock.calls.length,
		0,
	);
	expect(totalStopCalls).toBeGreaterThanOrEqual(1);
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Run command:

```
cd apps/web && bun test src/lib/passage-player.test.ts
```

Expected: FAIL — `player.pause is not a function`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add the `pause()` method to the `PassagePlayer` class:

```typescript
pause(): void {
	if (this.state !== "playing") return;
	for (const s of this.sources) {
		try { s.stop(); } catch { /* already stopped */ }
	}
	this.sources = [];
	this.state = "paused";
}

destroy(): void {
	for (const s of this.sources) {
		try { s.stop(); } catch { /* ignore */ }
	}
	this.sources = [];
	this.tickCallbacks.clear();
	this.state = "idle";
}
```

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/web && bun test src/lib/passage-player.test.ts
```

Expected: PASS (three tests total).

- [ ] **Step 5: Commit**

Run command:

```
git add apps/web/src/lib/passage-player.ts apps/web/src/lib/passage-player.test.ts && git commit -m "feat(web): PassagePlayer.pause halts playback and exposes destroy"
```

---

## Track C: `play_passage` tool, types, `reference_browser` removal

### Task 7: `play_passage` tool processor returns valid component
**Group:** C (parallel with A, B, D)

**Behavior being verified:** `processToolUse(ctx, studentId, "play_passage", validInput)` returns a single `InlineComponent` with `type === "play_passage"` and config carrying `sessionId`, `bars`, `dimension`, `annotation`, and optional `focusBars`.
**Interface under test:** `processToolUse` plus `TOOL_REGISTRY["play_passage"]`.

**Files:**
- Modify: `apps/api/src/services/tool-processor.ts`
- Modify: `apps/api/src/services/tool-processor.test.ts`
- Modify: `apps/web/src/lib/types.ts`

- [ ] **Step 1: Write the failing test**

Append to `apps/api/src/services/tool-processor.test.ts` (after existing test blocks):

```typescript
describe("play_passage tool", () => {
	it("processToolUse returns a play_passage InlineComponent for valid input", async () => {
		const ctx = { db: {} as unknown, env: {} as unknown } as Parameters<typeof processToolUse>[0];
		const result = await processToolUse(ctx, "student-1", "play_passage", {
			session_id: "00000000-0000-0000-0000-0000000000aa",
			bars: [5, 8],
			focus_bars: [6, 7],
			dimension: "timing",
			annotation: "you rushed here",
		});
		expect(result.isError).toBe(false);
		expect(result.componentsJson).toHaveLength(1);
		const c = result.componentsJson[0];
		expect(c.type).toBe("play_passage");
		expect(c.config).toMatchObject({
			sessionId: "00000000-0000-0000-0000-0000000000aa",
			bars: [5, 8],
			focusBars: [6, 7],
			dimension: "timing",
			annotation: "you rushed here",
		});
	});
});
```

Also update the assertion at the existing `it("has all 6 tools", ...)` block: change `toHaveLength(6)` to `toHaveLength(7)`, and add `"play_passage"` to the registered-names list near the top of the file (around line 19).

- [ ] **Step 2: Run test — verify it FAILS**

Run command:

```
cd apps/api && bun test src/services/tool-processor.test.ts
```

Expected: FAIL — `Unknown tool 'play_passage'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/tool-processor.ts`, insert after the `reference_browser` section (around line 414):

```typescript
// ---------------------------------------------------------------------------
// Tool: play_passage
// ---------------------------------------------------------------------------

const playPassageSchema = z
	.object({
		session_id: z.string().uuid(),
		bars: z
			.tuple([z.number().int().min(1), z.number().int().min(1)])
			.refine(([s, e]) => s <= e, { message: "bars start must be <= end" }),
		focus_bars: z
			.tuple([z.number().int().min(1), z.number().int().min(1)])
			.refine(([s, e]) => s <= e, { message: "focus_bars start must be <= end" })
			.optional(),
		dimension: dimensionEnum,
		annotation: z.string().min(1).max(280),
	})
	.refine(
		(d) =>
			d.focus_bars === undefined ||
			(d.focus_bars[0] >= d.bars[0] && d.focus_bars[1] <= d.bars[1]),
		{ message: "focus_bars must be within bars", path: ["focus_bars"] },
	);

async function processPlayPassage(
	_ctx: ServiceContext,
	_studentId: string,
	rawInput: unknown,
): Promise<InlineComponent[]> {
	const input = playPassageSchema.parse(rawInput);
	const config: Record<string, unknown> = {
		sessionId: input.session_id,
		bars: input.bars,
		dimension: input.dimension,
		annotation: input.annotation,
	};
	if (input.focus_bars !== undefined) {
		config.focusBars = input.focus_bars;
	}
	return [{ type: "play_passage", config }];
}

const playPassageAnthropicSchema: AnthropicToolSchema = {
	name: "play_passage",
	description:
		"Play back a bar-bounded slice of the student's own recording, with the score visible. Use when you want the student to LISTEN to a specific passage they just played, not just read about it. Only emit when a piece is identified for the current session and score alignment covers the requested bars — otherwise rely on text. The artifact shows the score for `bars` with `focus_bars` tinted in the dimension color.",
	input_schema: {
		type: "object",
		properties: {
			session_id: {
				type: "string",
				format: "uuid",
				description: "UUID of the practice session whose recording to play.",
			},
			bars: {
				type: "array",
				items: { type: "integer", minimum: 1 },
				minItems: 2,
				maxItems: 2,
				description:
					"Outer passage range as [start, end]. The full clip plays from start to end.",
			},
			focus_bars: {
				type: "array",
				items: { type: "integer", minimum: 1 },
				minItems: 2,
				maxItems: 2,
				description:
					"Optional tinted sub-range inside `bars`. Use to draw attention to the specific moment within musical context.",
			},
			dimension: {
				type: "string",
				enum: DIMS_6,
				description: "The single musical dimension this observation is about.",
			},
			annotation: {
				type: "string",
				description:
					"One sentence (<=280 chars) that the student reads next to the playback control.",
			},
		},
		required: ["session_id", "bars", "dimension", "annotation"],
	},
};
```

Then in `TOOL_REGISTRY`, add:

```typescript
play_passage: {
	name: "play_passage",
	description: playPassageAnthropicSchema.description,
	schema: playPassageSchema,
	anthropicSchema: playPassageAnthropicSchema,
	concurrencySafe: true,
	maxResultChars: 2000,
	process: processPlayPassage,
},
```

In `apps/web/src/lib/types.ts`, add to the `InlineComponent` union:

```typescript
	| { type: "play_passage"; config: PlayPassageConfig }
```

And add these interfaces:

```typescript
export interface PlayPassageConfig {
	sessionId: string;
	bars: [number, number];
	focusBars?: [number, number];
	dimension: string;
	annotation: string;
}

export interface PassageManifest {
	source: { kind: "session"; sessionId: string };
	pieceId: string;
	bars: [number, number];
	chunks: Array<{ url: string; chunkIndex: number; durationSec: number }>;
	startOffsetSec: number;
	endOffsetSec: number;
	barTimeline: Array<{ bar: number; tSec: number }>;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/api && bun test src/services/tool-processor.test.ts
```

Expected: PASS (existing tests plus the new `play_passage` test).

- [ ] **Step 5: Commit**

Run command:

```
git add apps/api/src/services/tool-processor.ts apps/api/src/services/tool-processor.test.ts apps/web/src/lib/types.ts && git commit -m "feat(api): add play_passage teacher tool"
```

---

### Task 8: `play_passage` schema rejects focus_bars outside bars
**Group:** C (sequential after Task 7)

**Behavior being verified:** Input with `focus_bars` that extends outside `bars` is rejected by the Zod schema, producing `isError: true` from `processToolUse`.
**Interface under test:** `processToolUse(ctx, sid, "play_passage", invalidInput)`.

**Files:**
- Test: `apps/api/src/services/tool-processor.test.ts`

- [ ] **Step 1: Write the failing test**

Append to the `describe("play_passage tool")` block:

```typescript
it("rejects focus_bars outside bars range", async () => {
	const ctx = { db: {} as unknown, env: {} as unknown } as Parameters<typeof processToolUse>[0];
	const result = await processToolUse(ctx, "student-1", "play_passage", {
		session_id: "00000000-0000-0000-0000-0000000000aa",
		bars: [5, 8],
		focus_bars: [9, 12],
		dimension: "timing",
		annotation: "out of range",
	});
	expect(result.isError).toBe(true);
	expect(result.errorMessage).toBeDefined();
});
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

Run command:

```
cd apps/api && bun test src/services/tool-processor.test.ts
```

Task 7's `refine` already enforces this constraint, so the test PASSES on the existing implementation. This task is a regression guard.

- [ ] **Step 3: No new implementation required.**

- [ ] **Step 4: Run all tool-processor tests — verify PASS**

Run command:

```
cd apps/api && bun test src/services/tool-processor.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run command:

```
git add apps/api/src/services/tool-processor.test.ts && git commit -m "test(api): guard focus_bars subset constraint"
```

---

### Task 9: Remove `reference_browser` tool
**Group:** C (sequential after Task 8)

**Behavior being verified:** `reference_browser` no longer appears in `TOOL_REGISTRY`, `getAnthropicToolSchemas()`, or the web `InlineComponent` union. `processToolUse` with name `"reference_browser"` returns `isError: true` (Unknown tool).
**Interface under test:** `TOOL_REGISTRY` keys, `processToolUse`.

**Files:**
- Modify: `apps/api/src/services/tool-processor.ts`
- Modify: `apps/api/src/services/tool-processor.test.ts`
- Modify: `apps/web/src/lib/types.ts`
- Modify: `apps/web/src/routes/app.sandbox.tsx`

- [ ] **Step 1: Write the failing test**

In `apps/api/src/services/tool-processor.test.ts`:
1. Update the existing `it("has all 6 tools", ...)` count assertion from 7 (set in Task 7) back to 6. Remove `"reference_browser"` from the registered-names list (around line 19).
2. Delete the existing `expect(TOOL_REGISTRY.reference_browser.concurrencySafe).toBe(true)` line (around line 50).
3. Delete the existing `expect(TOOL_REGISTRY.reference_browser.maxResultChars).toBe(2000)` line (around line 58).
4. Delete the entire `describe("reference_browser schema validation", ...)` block (around line 389).
5. Delete the `it("reference_browser returns ToolResult with correct type", ...)` block (around line 532).
6. Add a new guard test:

```typescript
it("reference_browser is no longer registered", async () => {
	expect(TOOL_REGISTRY.reference_browser).toBeUndefined();
	const ctx = { db: {} as unknown, env: {} as unknown } as Parameters<typeof processToolUse>[0];
	const result = await processToolUse(ctx, "student-1", "reference_browser", {
		description: "test",
	});
	expect(result.isError).toBe(true);
	expect(result.errorMessage).toContain("Unknown tool");
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Run command:

```
cd apps/api && bun test src/services/tool-processor.test.ts
```

Expected: FAIL — the deletion guard test fails because `TOOL_REGISTRY.reference_browser` is still defined; the count assertion also fails (still 7).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/tool-processor.ts`:
1. Delete `referenceBrowserSchema` (the const around lines 371–375).
2. Delete `processReferenceBrowser` function (around lines 377–414).
3. Delete `referenceBrowserAnthropicSchema` (around lines 711–734).
4. Delete the `reference_browser` entry in `TOOL_REGISTRY` (around lines 812–820).

In `apps/web/src/lib/types.ts`:
1. Delete `| { type: "reference_browser"; config: ReferenceBrowserConfig }` from the `InlineComponent` union.
2. Delete `export interface ReferenceBrowserConfig { [key: string]: unknown }`.

In `apps/web/src/routes/app.sandbox.tsx`:
1. Remove the `ReferenceBrowserConfig` import from the import block (the names in the import from `../lib/types`).

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/api && bun test src/services/tool-processor.test.ts
```

Then verify web typechecks:

```
cd apps/web && bun run build
```

Expected: PASS for tests; web build succeeds.

- [ ] **Step 5: Commit**

Run command:

```
git add apps/api/src/services/tool-processor.ts apps/api/src/services/tool-processor.test.ts apps/web/src/lib/types.ts apps/web/src/routes/app.sandbox.tsx && git commit -m "feat(api): remove reference_browser stub tool"
```

---

## Track D: Chunk byte read route

### Task 10: `GET /api/practice/chunk` returns 401 without auth
**Group:** D (parallel with A, B, C)

**Behavior being verified:** A GET request to the new chunk-read endpoint returns 401 when the caller is unauthenticated.
**Interface under test:** Hono route via `testApp.request`.

**Files:**
- Modify: `apps/api/src/routes/practice.ts`
- Modify: `apps/api/src/routes/practice.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `apps/api/src/routes/practice.test.ts` inside the existing `describe("practice routes", ...)` block:

```typescript
it("GET /api/practice/chunk returns 401 without auth", async () => {
	const res = await testApp.request(
		"/api/practice/chunk?sessionId=00000000-0000-0000-0000-000000000001&chunkIndex=0",
	);
	expect(res.status).toBe(401);
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Run command:

```
cd apps/api && bun test src/routes/practice.test.ts
```

Expected: FAIL — the GET route does not exist; Hono returns 404 (not 401).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/routes/practice.ts`, insert a new GET handler chained after the existing POST `/chunk` handler (before `.get("/ws/:sessionId", ...)`):

```typescript
.get("/chunk", validate("query", chunkQuerySchema), async (c) => {
	requireAuth(c.var.studentId);
	const { sessionId, chunkIndex } = c.req.valid("query");

	const session = await c.var.db.query.sessions.findFirst({
		where: (s, { eq: e, and: a }) =>
			a(e(s.id, sessionId), e(s.studentId, c.var.studentId!)),
	});
	if (!session) throw new NotFoundError("session", sessionId);

	const r2Key = `sessions/${sessionId}/chunks/${chunkIndex}.webm`;
	const obj = await c.env.CHUNKS.get(r2Key);
	if (!obj) throw new NotFoundError("chunk", `${sessionId}/${chunkIndex}`);

	return new Response(obj.body, {
		status: 200,
		headers: {
			"Content-Type": "audio/webm",
			"Cache-Control": "private, max-age=0, no-store",
		},
	});
})
```

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/api && bun test src/routes/practice.test.ts
```

Expected: PASS — `requireAuth` throws when `studentId` is null, and the error middleware maps it to 401.

- [ ] **Step 5: Commit**

Run command:

```
git add apps/api/src/routes/practice.ts apps/api/src/routes/practice.test.ts && git commit -m "feat(api): add GET /api/practice/chunk auth-gated R2 read"
```

---

## Track E: Sessions route + DO `/passage` handler

### Task 11: `GET /api/sessions/:id/passage` returns 401 without auth
**Group:** E (depends on Tracks A, C, D)

**Behavior being verified:** The new sessions route returns 401 when called without auth.
**Interface under test:** Hono route.

**Files:**
- Create: `apps/api/src/routes/sessions.ts`
- Create: `apps/api/src/routes/sessions.test.ts`
- Modify: `apps/api/src/index.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/routes/sessions.test.ts
import { Hono } from "hono";
import { describe, expect, it } from "vitest";
import { sessionsRoutes } from "./sessions";

const testApp = new Hono().route("/api/sessions", sessionsRoutes);

describe("sessions routes", () => {
	it("GET /api/sessions/:id/passage returns 401 without auth", async () => {
		const res = await testApp.request(
			"/api/sessions/00000000-0000-0000-0000-000000000001/passage?bars=5-8",
		);
		expect(res.status).toBe(401);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Run command:

```
cd apps/api && bun test src/routes/sessions.test.ts
```

Expected: FAIL — `Cannot find module './sessions'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/routes/sessions.ts
import { Hono } from "hono";
import { z } from "zod";
import { NotFoundError } from "../lib/errors";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";

const barsRegex = /^(\d+)-(\d+)$/;

const passageQuerySchema = z.object({
	bars: z
		.string()
		.regex(barsRegex, { message: "bars must be in form 'N-M' with integers" })
		.transform((s) => {
			const m = barsRegex.exec(s);
			if (!m) throw new Error("unreachable after regex validation");
			return [Number(m[1]), Number(m[2])] as [number, number];
		})
		.refine(([s, e]) => s >= 1 && e >= s, {
			message: "bars start must be >= 1 and <= end",
		}),
});

const sessionIdParamSchema = z.object({ id: z.string().uuid() });

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>().get(
	"/:id/passage",
	validate("param", sessionIdParamSchema),
	validate("query", passageQuerySchema),
	async (c) => {
		requireAuth(c.var.studentId);
		const { id: sessionId } = c.req.valid("param");
		const { bars } = c.req.valid("query");

		const session = await c.var.db.query.sessions.findFirst({
			where: (s, { eq: e, and: a }) =>
				a(e(s.id, sessionId), e(s.studentId, c.var.studentId!)),
		});
		if (!session) throw new NotFoundError("session", sessionId);

		const doId = c.env.SESSION_BRAIN.idFromName(sessionId);
		const stub = c.env.SESSION_BRAIN.get(doId);
		const doRes = await stub.fetch(
			new Request(
				`https://do/passage?bars=${bars[0]}-${bars[1]}&sessionId=${sessionId}`,
				{ method: "GET" },
			),
		);
		if (doRes.status === 409) {
			return c.json({ error: "passage_unavailable" }, 409);
		}
		if (!doRes.ok) {
			return c.json({ error: "internal" }, 500);
		}
		const manifest = (await doRes.json()) as unknown;
		return c.json(manifest, 200);
	},
);

export { app as sessionsRoutes };
```

In `apps/api/src/index.ts`, add an import for `sessionsRoutes` from `./routes/sessions` and mount it via the existing `.route()` chain pattern: `.route("/api/sessions", sessionsRoutes)`.

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/api && bun test src/routes/sessions.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run command:

```
git add apps/api/src/routes/sessions.ts apps/api/src/routes/sessions.test.ts apps/api/src/index.ts && git commit -m "feat(api): add /api/sessions route with passage endpoint (auth-gated)"
```

---

### Task 12: DO `/passage` handler returns manifest built from enriched chunks
**Group:** E (sequential after Task 11)

**Behavior being verified:** When the sessions route forwards to the DO at `/passage?bars=N-M&sessionId=...`, the DO loads its `chunk_enriched:*` storage, invokes `buildPassageManifest` with the identified `pieceId`, and returns a JSON manifest when alignment exists; otherwise returns 409.
**Interface under test:** End-to-end through `GET /api/sessions/:id/passage` with a stubbed `SESSION_BRAIN` binding that simulates the DO response.

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Modify: `apps/api/src/routes/sessions.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `apps/api/src/routes/sessions.test.ts`:

```typescript
import type { PassageManifest } from "../services/passage-manifest";

it("GET /:id/passage returns manifest when DO returns 200", async () => {
	const cannedManifest: PassageManifest = {
		source: { kind: "session", sessionId: "00000000-0000-0000-0000-000000000001" },
		pieceId: "chopin.ballades.1",
		bars: [5, 8],
		chunks: [
			{ url: "https://api/c1", chunkIndex: 1, durationSec: 15 },
		],
		startOffsetSec: 1.0,
		endOffsetSec: 13.0,
		barTimeline: [
			{ bar: 5, tSec: 0 },
			{ bar: 6, tSec: 4 },
			{ bar: 7, tSec: 8 },
			{ bar: 8, tSec: 12 },
		],
	};

	const stubFetch = async (_: Request) =>
		new Response(JSON.stringify(cannedManifest), {
			status: 200,
			headers: { "Content-Type": "application/json" },
		});

	const env = {
		SESSION_BRAIN: {
			idFromName: () => ({}),
			get: () => ({ fetch: stubFetch }),
		},
	} as unknown as Record<string, unknown>;

	const dbStub = {
		query: {
			sessions: {
				findFirst: async () => ({ id: "00000000-0000-0000-0000-000000000001", studentId: "student-1" }),
			},
		},
	};

	const testAppAuthed = new Hono()
		.use("*", async (c, next) => {
			(c as unknown as { set: (k: string, v: unknown) => void }).set("studentId", "student-1");
			(c as unknown as { set: (k: string, v: unknown) => void }).set("db", dbStub);
			await next();
		})
		.route("/api/sessions", sessionsRoutes);

	const res = await testAppAuthed.request(
		"/api/sessions/00000000-0000-0000-0000-000000000001/passage?bars=5-8",
		{},
		env,
	);
	expect(res.status).toBe(200);
	const body = (await res.json()) as PassageManifest;
	expect(body.pieceId).toBe("chopin.ballades.1");
	expect(body.barTimeline).toHaveLength(4);
});
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

Run command:

```
cd apps/api && bun test src/routes/sessions.test.ts
```

The route forwards to the stubbed DO. If the route correctly forwards and parses the response, this test PASSES without changes to the DO. The DO handler is still required for real traffic — proceed to Step 3 regardless.

- [ ] **Step 3: Implement the DO `/passage` branch**

In `apps/api/src/do/session-brain.ts`, locate the existing `fetch()` method (around line 150) which already handles `/synthesize`. Add a new branch BEFORE the WebSocket-upgrade path:

```typescript
if (url.pathname === "/passage" && request.method === "GET") {
	const barsParam = url.searchParams.get("bars") ?? "";
	const sessionIdParam = url.searchParams.get("sessionId") ?? "";
	const m = /^(\d+)-(\d+)$/.exec(barsParam);
	if (!m || !sessionIdParam) {
		return new Response(JSON.stringify({ error: "bad_request" }), {
			status: 400,
			headers: { "Content-Type": "application/json" },
		});
	}
	const bars: [number, number] = [Number(m[1]), Number(m[2])];

	const state = await this.ctx.storage.get<SessionState>("state");
	const pieceId = state?.pieceIdentification?.pieceId;
	if (!pieceId) {
		return new Response(JSON.stringify({ error: "no_piece" }), {
			status: 409,
			headers: { "Content-Type": "application/json" },
		});
	}

	const enrichedMap = await this.ctx.storage.list<EnrichedChunk>({ prefix: "chunk_enriched:" });
	const enrichedChunks = Array.from(enrichedMap.values()).sort(
		(a, b) => a.chunkIndex - b.chunkIndex,
	);

	const { buildPassageManifest } = await import("../services/passage-manifest");
	const baseUrl = `https://${request.headers.get("host") ?? "api.crescend.ai"}`;
	const result = buildPassageManifest({
		enrichedChunks,
		bars,
		pieceId,
		sessionId: sessionIdParam,
		baseUrl,
	});

	if ("error" in result) {
		return new Response(JSON.stringify({ error: result.error }), {
			status: 409,
			headers: { "Content-Type": "application/json" },
		});
	}

	return new Response(JSON.stringify(result), {
		status: 200,
		headers: { "Content-Type": "application/json" },
	});
}
```

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/api && bun test src/routes/sessions.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run command:

```
git add apps/api/src/do/session-brain.ts apps/api/src/routes/sessions.test.ts && git commit -m "feat(api): DO /passage handler returns passage manifest"
```

---

### Task 13: Passage route propagates 409 from DO
**Group:** E (sequential after Task 12)

**Behavior being verified:** When the DO returns 409 (no piece identified or alignment missing), the route propagates 409 to the client.
**Interface under test:** End-to-end route call with a stubbed DO returning 409.

**Files:**
- Test: `apps/api/src/routes/sessions.test.ts`

- [ ] **Step 1: Write the failing test**

Append:

```typescript
it("GET /:id/passage propagates 409 from DO", async () => {
	const stubFetch = async () =>
		new Response(JSON.stringify({ error: "no_alignment" }), { status: 409 });
	const env = {
		SESSION_BRAIN: { idFromName: () => ({}), get: () => ({ fetch: stubFetch }) },
	} as unknown as Record<string, unknown>;

	const dbStub = {
		query: {
			sessions: {
				findFirst: async () => ({ id: "00000000-0000-0000-0000-000000000001", studentId: "student-1" }),
			},
		},
	};

	const testAppAuthed = new Hono()
		.use("*", async (c, next) => {
			(c as unknown as { set: (k: string, v: unknown) => void }).set("studentId", "student-1");
			(c as unknown as { set: (k: string, v: unknown) => void }).set("db", dbStub);
			await next();
		})
		.route("/api/sessions", sessionsRoutes);

	const res = await testAppAuthed.request(
		"/api/sessions/00000000-0000-0000-0000-000000000001/passage?bars=5-8",
		{},
		env,
	);
	expect(res.status).toBe(409);
});
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

Run command:

```
cd apps/api && bun test src/routes/sessions.test.ts
```

The route already propagates 409 (added in Task 11), so this test PASSES on the existing implementation. Regression guard only.

- [ ] **Step 3: No new implementation required.**

- [ ] **Step 4: Run all sessions-route tests — verify PASS**

Run command:

```
cd apps/api && bun test src/routes/sessions.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run command:

```
git add apps/api/src/routes/sessions.test.ts && git commit -m "test(api): guard 409 propagation in passage route"
```

---

## Track F: `PlayPassageCard`

### Task 14: PlayPassageCard renders annotation, dimension label, and bar range
**Group:** F (depends on Tracks B, C, E)

**Behavior being verified:** Given a valid `PlayPassageConfig`, the card calls `api.sessions.getPassage(sessionId, bars)`, awaits `scoreRenderer.getClip`, constructs a `PassagePlayer`, and renders the annotation text, dimension label, and bar range.
**Interface under test:** React component rendered via Testing Library.

**Files:**
- Create: `apps/web/src/components/cards/PlayPassageCard.tsx`
- Create: `apps/web/src/components/cards/PlayPassageCard.test.tsx`
- Modify: `apps/web/src/lib/api.ts`
- Modify: `apps/web/src/components/InlineCard.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/cards/PlayPassageCard.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { PassageManifest, PlayPassageConfig } from "../../lib/types";

const mockGetPassage = vi.fn();
const mockGetClip = vi.fn();

vi.mock("../../lib/api", () => ({
	api: {
		sessions: {
			getPassage: (...args: unknown[]) => mockGetPassage(...args),
		},
	},
}));

vi.mock("../../lib/score-renderer", () => ({
	scoreRenderer: {
		getClip: (...args: unknown[]) => mockGetClip(...args),
	},
}));

vi.mock("../../lib/passage-player", () => ({
	PassagePlayer: vi.fn().mockImplementation(() => ({
		state: "ready",
		duration: 13,
		load: vi.fn().mockResolvedValue(undefined),
		play: vi.fn(),
		pause: vi.fn(),
		onTick: () => () => undefined,
		destroy: vi.fn(),
	})),
}));

beforeEach(() => {
	vi.clearAllMocks();
	globalThis.AudioContext = vi.fn() as unknown as typeof AudioContext;
});

describe("PlayPassageCard", () => {
	const config: PlayPassageConfig = {
		sessionId: "00000000-0000-0000-0000-0000000000aa",
		bars: [5, 8],
		focusBars: [6, 7],
		dimension: "timing",
		annotation: "you rushed here",
	};

	const manifest: PassageManifest = {
		source: { kind: "session", sessionId: config.sessionId },
		pieceId: "chopin.ballades.1",
		bars: [5, 8],
		chunks: [{ url: "https://api/c1.webm", chunkIndex: 1, durationSec: 15 }],
		startOffsetSec: 1.0,
		endOffsetSec: 13.0,
		barTimeline: [
			{ bar: 5, tSec: 0 },
			{ bar: 6, tSec: 4 },
			{ bar: 7, tSec: 8 },
			{ bar: 8, tSec: 12 },
		],
	};

	it("renders annotation, dimension label, and bar range after manifest loads", async () => {
		mockGetPassage.mockResolvedValue(manifest);
		mockGetClip.mockResolvedValue({
			svg: "<svg></svg>",
			startMeasureId: null,
			endMeasureId: null,
		});

		const { PlayPassageCard } = await import("./PlayPassageCard");
		render(React.createElement(PlayPassageCard, { config }));

		await waitFor(() => {
			expect(screen.getByText("you rushed here")).toBeInTheDocument();
			expect(screen.getByText("timing")).toBeInTheDocument();
			expect(screen.getByText(/bars 5/)).toBeInTheDocument();
			expect(mockGetPassage).toHaveBeenCalledWith(config.sessionId, [5, 8]);
		});
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Run command:

```
cd apps/web && bun test src/components/cards/PlayPassageCard.test.tsx
```

Expected: FAIL — `Cannot find module './PlayPassageCard'`.

- [ ] **Step 3: Implement**

First add the api client method. Read `apps/web/src/lib/api.ts` to learn the existing namespace structure, then add a `sessions` namespace alongside the others:

```typescript
sessions: {
	async getPassage(
		sessionId: string,
		bars: [number, number],
	): Promise<import("./types").PassageManifest> {
		const res = await fetch(
			`${API_BASE_URL}/api/sessions/${sessionId}/passage?bars=${bars[0]}-${bars[1]}`,
			{ credentials: "include" },
		);
		if (!res.ok) {
			throw new Error(`getPassage failed: ${res.status}`);
		}
		return (await res.json()) as import("./types").PassageManifest;
	},
},
```

(If `API_BASE_URL` is not the local convention, follow the existing pattern in the same file for building the URL.)

Then create the card:

```typescript
// apps/web/src/components/cards/PlayPassageCard.tsx
import { useEffect, useRef, useState } from "react";
import { api } from "../../lib/api";
import { DIMENSION_COLORS } from "../../lib/mock-session";
import { PassagePlayer } from "../../lib/passage-player";
import type { ClipResult } from "../../lib/score-renderer";
import { scoreRenderer } from "../../lib/score-renderer";
import type { PassageManifest, PlayPassageConfig } from "../../lib/types";
import { SvgClip } from "../SvgClip";

interface PlayPassageCardProps {
	config: PlayPassageConfig;
	onExpand?: () => void;
	artifactId?: string;
}

type LoadState = "loading" | "ready" | "error";

export function PlayPassageCard({ config }: PlayPassageCardProps) {
	const [loadState, setLoadState] = useState<LoadState>("loading");
	const [clip, setClip] = useState<ClipResult | null>(null);
	const [manifest, setManifest] = useState<PassageManifest | null>(null);
	const playerRef = useRef<PassagePlayer | null>(null);

	useEffect(() => {
		let cancelled = false;
		(async () => {
			try {
				const m = await api.sessions.getPassage(config.sessionId, config.bars);
				if (cancelled) return;
				const c = await scoreRenderer.getClip(m.pieceId, config.bars[0], config.bars[1]);
				if (cancelled) return;
				const ctx = new AudioContext();
				const player = new PassagePlayer(m, ctx);
				await player.load();
				if (cancelled) {
					player.destroy?.();
					return;
				}
				playerRef.current = player;
				setManifest(m);
				setClip(c);
				setLoadState("ready");
			} catch (err) {
				console.error("PlayPassageCard load failed", err);
				if (!cancelled) setLoadState("error");
			}
		})();
		return () => {
			cancelled = true;
			playerRef.current?.destroy?.();
		};
	}, [config.sessionId, config.bars[0], config.bars[1]]);

	const color =
		DIMENSION_COLORS[config.dimension as keyof typeof DIMENSION_COLORS] ?? "#7a9a82";

	return (
		<div className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3">
			{loadState === "loading" && (
				<div className="h-10 flex items-center justify-center">
					<div className="w-3.5 h-3.5 rounded-full border-2 border-text-tertiary/50 border-t-transparent animate-spin" />
				</div>
			)}
			{loadState === "ready" && clip && manifest && (
				<div className="px-3 pt-3">
					<div
						style={{
							position: "relative",
							borderRadius: "6px",
							border: `1.5px solid ${color}40`,
							backgroundColor: "white",
							overflow: "hidden",
						}}
					>
						<SvgClip
							svgMarkup={clip.svg}
							startMeasureId={clip.startMeasureId}
							endMeasureId={clip.endMeasureId}
						/>
					</div>
				</div>
			)}
			{loadState === "error" && (
				<div className="p-4 text-body-sm text-text-tertiary">couldn't load audio</div>
			)}
			<div className="p-4 flex flex-col gap-3.5">
				<div className="flex items-center gap-1.5 shrink-0">
					<span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
					<span className="text-label-sm text-text-tertiary uppercase tracking-wide">
						{config.dimension}
					</span>
				</div>
				<span className="text-body-xs text-text-tertiary">
					bars {config.bars[0]}–{config.bars[1]}
				</span>
				<p className="text-body-sm text-text-primary mt-0.5 leading-snug">
					{config.annotation}
				</p>
			</div>
		</div>
	);
}
```

Modify `apps/web/src/components/InlineCard.tsx` — import `PlayPassageCard` and add a switch case:

```typescript
case "play_passage":
	return (
		<PlayPassageCard
			config={component.config}
			onExpand={onExpand}
			artifactId={artifactId}
		/>
	);
```

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/web && bun test src/components/cards/PlayPassageCard.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run command:

```
git add apps/web/src/components/cards/PlayPassageCard.tsx apps/web/src/components/cards/PlayPassageCard.test.tsx apps/web/src/lib/api.ts apps/web/src/components/InlineCard.tsx && git commit -m "feat(web): add PlayPassageCard with manifest fetch and score-clip rendering"
```

---

### Task 15: PlayPassageCard play button invokes the player
**Group:** F (sequential after Task 14)

**Behavior being verified:** After the manifest loads, the card renders a play button; activating it invokes `player.play()`.
**Interface under test:** Component DOM via Testing Library; `PassagePlayer` mock instance.

**Files:**
- Modify: `apps/web/src/components/cards/PlayPassageCard.tsx`
- Modify: `apps/web/src/components/cards/PlayPassageCard.test.tsx`

- [ ] **Step 1: Write the failing test**

Append to the existing `describe("PlayPassageCard", ...)` block:

```typescript
it("clicking play invokes PassagePlayer.play()", async () => {
	mockGetPassage.mockResolvedValue(manifest);
	mockGetClip.mockResolvedValue({
		svg: "<svg></svg>",
		startMeasureId: null,
		endMeasureId: null,
	});

	const playFn = vi.fn();
	const playerMod = await import("../../lib/passage-player");
	(playerMod.PassagePlayer as unknown as { mockImplementation: (fn: () => unknown) => void }).mockImplementation(() => ({
		state: "ready",
		duration: 13,
		load: vi.fn().mockResolvedValue(undefined),
		play: playFn,
		pause: vi.fn(),
		onTick: () => () => undefined,
		destroy: vi.fn(),
	}));

	const { PlayPassageCard } = await import("./PlayPassageCard");
	render(React.createElement(PlayPassageCard, { config }));

	const btn = await screen.findByRole("button", { name: /play/i });
	btn.click();
	expect(playFn).toHaveBeenCalled();
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Run command:

```
cd apps/web && bun test src/components/cards/PlayPassageCard.test.tsx
```

Expected: FAIL — no button with name matching `/play/i` exists.

- [ ] **Step 3: Implement**

In `apps/web/src/components/cards/PlayPassageCard.tsx`, inside the `loadState === "ready"` block, after the score clip, add:

```typescript
<button
	type="button"
	aria-label="Play passage"
	onClick={() => playerRef.current?.play()}
	className="mt-3 px-3 py-1.5 rounded-md border border-border text-body-sm text-text-primary hover:bg-surface transition-colors"
>
	Play
</button>
```

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/web && bun test src/components/cards/PlayPassageCard.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run command:

```
git add apps/web/src/components/cards/PlayPassageCard.tsx apps/web/src/components/cards/PlayPassageCard.test.tsx && git commit -m "feat(web): PlayPassageCard play button wires to PassagePlayer"
```

---

### Task 16: PlayPassageCard exposes error state on manifest fetch failure
**Group:** F (sequential after Task 15)

**Behavior being verified:** When `api.sessions.getPassage` rejects (e.g. 409 from server), the card renders the error state ("couldn't load audio") and does not crash.
**Interface under test:** Component DOM after a rejected manifest fetch.

**Files:**
- Test: `apps/web/src/components/cards/PlayPassageCard.test.tsx`

- [ ] **Step 1: Write the failing test**

Append:

```typescript
it("shows error state when manifest fetch rejects", async () => {
	mockGetPassage.mockRejectedValue(new Error("getPassage failed: 409"));
	const { PlayPassageCard } = await import("./PlayPassageCard");
	render(React.createElement(PlayPassageCard, { config }));
	await waitFor(() => {
		expect(screen.getByText("couldn't load audio")).toBeInTheDocument();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

Run command:

```
cd apps/web && bun test src/components/cards/PlayPassageCard.test.tsx
```

The error branch is already implemented in Task 14's card (`setLoadState("error")` inside the `catch` block). Expected: PASS. Regression guard.

- [ ] **Step 3: No new implementation required.**

- [ ] **Step 4: Run all PlayPassageCard tests — verify PASS**

Run command:

```
cd apps/web && bun test src/components/cards/PlayPassageCard.test.tsx
```

Expected: PASS (three tests total).

- [ ] **Step 5: Commit**

Run command:

```
git add apps/web/src/components/cards/PlayPassageCard.test.tsx && git commit -m "test(web): guard PlayPassageCard error state on failed manifest fetch"
```

---

## Track G: Artifact collapsed-props wiring

### Task 17: `getCollapsedProps` handles `play_passage`
**Group:** G (depends on Track C — can run in parallel with Track F)

**Behavior being verified:** `getCollapsedProps` returns a `title`/`subtitle`/`badge` triple for `play_passage` components.
**Interface under test:** Direct call to `getCollapsedProps`.

**Files:**
- Modify: `apps/web/src/components/Artifact.tsx`
- Create: `apps/web/src/components/Artifact.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/Artifact.test.tsx
import { describe, expect, it } from "vitest";
import { getCollapsedProps } from "./Artifact";
import type { InlineComponent } from "../lib/types";

describe("getCollapsedProps", () => {
	it("returns title and subtitle for play_passage", () => {
		const component: InlineComponent = {
			type: "play_passage",
			config: {
				sessionId: "00000000-0000-0000-0000-0000000000aa",
				bars: [5, 8],
				dimension: "timing",
				annotation: "you rushed here",
			},
		};
		const result = getCollapsedProps(component);
		expect(result.title).toBe("Play Passage");
		expect(result.subtitle).toBe("bars 5-8, timing");
		expect(result.badge).toBe("");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Run command:

```
cd apps/web && bun test src/components/Artifact.test.tsx
```

Expected: FAIL — current `getCollapsedProps` falls through to the default branch producing `title: "play passage"` (lowercase).

- [ ] **Step 3: Implement**

In `apps/web/src/components/Artifact.tsx`, add a new branch in `getCollapsedProps` after the `score_highlight` branch (around line 39):

```typescript
if (component.type === "play_passage") {
	return {
		title: "Play Passage",
		subtitle: `bars ${component.config.bars[0]}-${component.config.bars[1]}, ${component.config.dimension}`,
		badge: "",
	};
}
```

- [ ] **Step 4: Run test — verify it PASSES**

Run command:

```
cd apps/web && bun test src/components/Artifact.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run command:

```
git add apps/web/src/components/Artifact.tsx apps/web/src/components/Artifact.test.tsx && git commit -m "feat(web): Artifact getCollapsedProps handles play_passage"
```

---

## Post-implementation verification

After all 17 tasks merge:

Run command:

```
cd apps/api && bun test && cd ../web && bun test
```

All test suites must pass. Then run a typecheck across the monorepo per the project's standard process. Manually verify in `app.sandbox` that the new card renders end-to-end against a real session (this is qualitative and not part of automated CI).
