# SessionBrain DO concurrency probe audit (2026-05-31)

Code-only audit covering production-env probes 1, 2, 3, 6, 7 from the
2026-05-31 pipeline element audit. Findings ranked by severity and grouped
by code-only-fixable vs needs-wrangler-to-verify.

## Findings

### Finding 1 (HIGH) — `chunksInFlight` read-modify-write race

`apps/api/src/do/session-brain.ts:414-417` (handleChunkReady step 1-2) and
`:1825-1834` (decrementChunksInFlight).

Pattern:
```ts
const state = await this.readState();   // yield point
state.chunksInFlight++;                 // sync increment
await this.ctx.storage.put("state", state); // yield point
```

CF Durable Objects do NOT provide atomicity across `await` points within a
single handler — the input-gate model only serializes synchronous spans. Two
concurrent `chunk_ready` messages can both `readState()` (each sees
`chunksInFlight: 0`), each increment to 1, and each write 1. The counter is
undercounted by the number of concurrent chunks minus one.

**Downstream impact:** `handleEndSession` (line 1319, post-Issue-3-fix)
checks `state.chunksInFlight === 0` to decide whether to fire synthesis
immediately. If the counter is undercounted, synthesis fires while chunks are
still processing → accumulator missing those chunks → underweight synthesis.

**Verification:** integration — fire 5 `chunk_ready` in rapid succession from
pipeline_client.py, end_session, assert resulting state.scoredChunks.length === 5.
Will likely fail under wrangler dev.

**Fix sketch:** use `ctx.blockConcurrencyWhile` around the read-modify-write,
or move counter to a dedicated atomic storage key with `transaction` semantics.

### Finding 2 (HIGH) — Bail-on-version-change silently loses chunk output

`apps/api/src/do/session-brain.ts:483-499` (handleChunkReady step 4).

If state.version changed during the ~1-3s inference await (because another
chunk completed, or `set_piece` fired, or `end_session` mutated state), the
current chunk:
- discards its MuQ scores (don't reach `scoredChunks.push`)
- discards its AMT notes (don't reach `chunk_enriched:N` storage)
- discards its ModeDetector update
- discards its piece-ID attempt
- discards its teaching-moment selection
- still emits a `chunk_processed` ws message to the client

The client sees "chunk processed" but the session state has no record of it.
For any concurrent chunk pair, only the first-finishing chunk persists; the
rest are silently dropped.

**Downstream impact:** under any real concurrency this is severe data loss.
A session with 10 parallel chunks could end up with 1-2 chunks of accumulator
state.

**Verification:** integration — same as Finding 1; assert
`scoredChunks.length === sent_count`.

**Fix sketch:** the bail-on-version-change pattern is too aggressive. Per-chunk
mutations are additive (push to scoredChunks, append to timeline, accumulate
moments) — they don't conflict. The version-check should guard semantically
conflicting fields (pieceIdentification, sessionEnding), not all writes.
Restructure to: re-read state right before writing, merge our additive changes
onto the latest state, then put.

### Finding 3 (MEDIUM) — webSocketClose fires synthesis ignoring chunksInFlight

`apps/api/src/do/session-brain.ts:374`.

```ts
await this.ctx.storage.setAlarm(Date.now() + 1);
```

Unconditional 1ms alarm. Unlike `handleEndSession` which guards on
`state.chunksInFlight === 0`, `webSocketClose` fires the alarm regardless.
If the client closes the WS mid-inference (network drop, browser tab close),
synthesis runs against incomplete accumulator.

**Verification:** integration — open WS, send 3 chunks, close WS before
inference completes, observe synthesis fires prematurely.

**Fix sketch:** same guard as handleEndSession — only set 1ms alarm if
`chunksInFlight === 0`, else leave the existing alarm in place (chunks reset
it to 30min on completion).

### Finding 4 (MEDIUM) — PassageLoopDetector + previousChunkAudio in module WeakMap

`apps/api/src/do/session-brain.ts:78, 81`.

```ts
const previousChunkAudio = new WeakMap<SessionBrain, ArrayBuffer | null>();
const detectorMap = new WeakMap<SessionBrain, PassageLoopDetector>();
```

Module-level WeakMaps keyed on the DO instance. When the DO hibernates and is
re-instantiated (which CF does aggressively to free memory), both maps are
blank for the new instance. Impacts:

- `previousChunkAudio` lost: next AMT call has no context audio for the
  windowed-overlap inference. AMT loses ~50ms of context per resume.
- `detectorMap` lost: passage-loop debounce state resets to zero. A student
  reconnecting mid-session could re-trigger the same loop attempt the
  detector previously suppressed.

This is confirmed by audit probe #3 ("PassageLoopDetector recreated blank on
reconnect").

**Verification:** integration — open WS, send 3 chunks that should trigger a
loop attempt, close WS, reopen same sessionId, send 3 more chunks at the
same bar range, check `loop_attempt` ws messages count.

**Fix sketch:** persist `passageLoopDetector` state into `SessionState` (like
ModeDetector already does at line 1272: `state.modeDetector = modeDetector.toJSON()`).
Drop `detectorMap` entirely. `previousChunkAudio` can probably be tolerated
as best-effort (a 50ms AMT context gap on reconnect is minor) — document it.

### Finding 5 (LOW) — Single state key unbounded growth in long sessions

`apps/api/src/do/session-brain.ts` SessionState write at line 1274.

`SessionState.scoredChunks: { chunkIndex, scores[6] }[]` grows unbounded. At
~50 bytes per entry, 1000 chunks = 50KB — well under the 1MB-per-key limit.
But `state.accumulator` (timeline events + mode transitions + teaching
moments) also accumulates in the same key. For a 4-hour session at one chunk
every 15s = 960 chunks plus timeline events plus teaching moments every 4
chunks (240 moments).

Estimate: ~200-400KB for a long session. Under the 1MB key limit but
approaches it. `chunk_enriched:N` keys are already split out (good).

**Verification:** integration — run a long-session simulator (or just inspect
final state size after a real long session).

**Fix sketch:** none needed yet. Add a `state_size_bytes` log line on each
writeState if you want telemetry.

## Probes still untested

| Probe | Status |
|---|---|
| #1 DO concurrent-chunk safety | Audited above (Findings 1, 2). Needs integration test to confirm. |
| #2 Session finalization vs in-flight | Audited above (Finding 3). Needs integration test. |
| #3 WS reconnect mid-session | Audited above (Finding 4). Needs integration test. |
| #4 Empty/silent audio | Not audited. Code spot to inspect: handleEvalChunk `if (perfNotes.length > 0)` branch at line 1084 — empty notes skip bar analysis, fall through; should be safe. |
| #5 Very short sessions | Likely safe — runSynthesisAndPersist handles small accumulator state (Issue 3 verify run showed 2 chunks → 371-char synthesis). |
| #6 Very long sessions | See Finding 5. |
| #7 AMT partial failure | Code path exists at handleChunkReady:535-545 (`amtResult.status === "rejected"` → Tier 3). Looks correct on inspection. |

## Recommended next moves

In order of expected leverage:

1. **Fix Finding 2 (silent chunk discard)** with a merge-on-write pattern. It's
   the highest-severity bug and the fix has clear semantics — chunk additions
   are additive. Smallest reversible code change, biggest correctness win.
2. **Fix Finding 4 (PassageLoopDetector persistence)** — copy the ModeDetector
   serialization pattern. Pre-baked solution.
3. **Fix Finding 3 (webSocketClose alarm guard)** — one-line condition copy
   from handleEndSession.
4. Write an integration test for probes #1/#2/#3 that fires N parallel chunks
   to wrangler dev and asserts `scoredChunks.length === N`. Until 1-3 land,
   that test will fail.

## Out of scope

Finding 5 doesn't need a fix yet. Finding 1 (chunksInFlight race) is gated on
Cloudflare's `blockConcurrencyWhile` API; a real fix here requires reading
the CF DO docs to confirm the right serialization primitive for storage RMW.
