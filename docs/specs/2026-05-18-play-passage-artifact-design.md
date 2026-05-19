# Play Passage Artifact Design

**Goal:** Let the teacher LLM say "listen here" — emitting a chat artifact that plays a bar-bounded slice of the student's own recording, score-aligned, with a tinted focus sub-range.

**Not in scope:**
- Pro reference audio source (post-beta; the tool schema reserves a `source` discriminator slot for it)
- AB-compare between student and pro (post-beta)
- Multi-range overlays inside one passage
- Event markers (e.g. "pedal fired here") and dimension traces
- iOS rendering
- Sub-beat / sample-accurate cursor precision
- Whole-session playback navigation
- Audio-only fallback when score alignment is missing — tool is contractually score-bound

## Problem

The teacher LLM today can *describe* a moment in the student's recording ("you rushed in bars 6–7") but cannot *play it back* with the score visible. The closest existing affordance is `score_highlight` (static notation with annotations) and `reference_browser` (a literal empty stub: `ReferenceBrowserConfig: { [key: string]: unknown }` at `apps/web/src/lib/types.ts:62`, with a placeholder server processor at `apps/api/src/services/tool-processor.ts:377`). Neither lets the student *hear* what the teacher heard.

Concretely, the gap:
- The DO at `apps/api/src/do/session-brain.ts:680` already persists `EnrichedChunk` to DO storage (`chunk_enriched:${index}`) with score-bar alignments per note (`alignment[].bar`, `alignment[].expected_onset_ms`). Bar→time data exists at chunk granularity but no read path exposes it.
- The R2 bucket `CHUNKS` (`apps/api/src/routes/practice.ts:81`) holds the 15s WebM chunks keyed `sessions/{sessionId}/chunks/{i}.webm`, but no read endpoint exists for them post-write.
- Score rendering already supports clip extraction by bar range (`apps/web/src/lib/score-renderer.ts:135` `getClip(pieceId, startBar, endBar)`).

So everything needed exists; nothing is wired up.

## Solution (from the user's perspective)

Sarah is mid-conversation in `crescend.ai`. The teacher says: *"You rushed in bars 6–7 — listen back."* An inline card appears: the score for bars 1–8, with bars 6–7 tinted in the timing color, a one-sentence annotation, and a play button. She taps play. The audio of her own recording for that passage plays. A cursor slides across the score in time with the audio. When it ends, the cursor parks at the end of bar 8 and she can replay.

If the system can't trust the score alignment for that range, the artifact does not appear — the teacher delivers the same observation in plain text instead.

## Design

### Three layers

1. **Tool schema (`play_passage`)** — Anthropic tool definition in `apps/api/src/services/tool-processor.ts`. The teacher LLM emits this tool call; the server validates with Zod, returns a `play_passage` `InlineComponent` whose config carries the call's parameters verbatim (no enrichment on the tool-process path — the manifest is fetched by the client at render time).

2. **Manifest endpoint (`GET /api/sessions/:id/passage?bars=N-M`)** — New route in a new `apps/api/src/routes/sessions.ts` file. Auth + ownership gated. Talks to the session's DO over an internal HTTP route (`/passage?bars=N-M`). The DO loads its `chunk_enriched:*` storage entries, hands them to a pure service function `buildPassageManifest(enrichedChunks, bars, pieceId, sessionId)`, returns the manifest. If alignment is missing or doesn't cover `[N, M]`, the DO returns 409 → the route returns 409.

3. **Card (`PlayPassageCard`)** — Composes existing `scoreRenderer.getClip()` for the SVG with a new `PassagePlayer` for audio + RAF cursor. The card fetches the manifest on mount, builds the player, draws the focus-bar tint identically to `ScoreHighlightCard.tsx:84`.

### Manifest shape

```ts
type PassageManifest = {
  source: { kind: "session"; sessionId: string };
  pieceId: string;
  bars: [number, number];           // outer passage range
  chunks: Array<{
    url: string;                     // /api/sessions/:id/chunks/:i (auth-cookie-gated)
    chunkIndex: number;
    durationSec: number;             // 15.0 except possibly the last
  }>;
  startOffsetSec: number;            // seconds into chunks[0] where bar N's first note begins
  endOffsetSec: number;              // seconds into the LAST chunk where bar M's coverage ends
  barTimeline: Array<{ bar: number; tSec: number }>; // tSec relative to passage start
};
```

`tSec` for bar `k` is computed from the first alignment with `bar == k` in any covered chunk: `tSec = (chunkIndex - chunks[0].chunkIndex) * 15 + (expected_onset_ms / 1000) - startOffsetSec`. `startOffsetSec` is bar N's earliest alignment within its chunk; `endOffsetSec` is bar M's latest alignment within its chunk (or `durationSec` of the last chunk if M is the score's final bar).

### Why this approach over alternatives

- **vs. server-side stitch+trim (per-call ffmpeg):** Workers cannot run ffmpeg; the manifest approach keeps the Worker doing only signed routing + tiny JSON.
- **vs. session-end concat written once to R2:** A session-end concat is fine but adds a new asset, a new R2 key, a synthesis-time job, and another failure mode. The manifest path needs zero new persisted artifacts because everything is already in R2 + DO storage.
- **vs. MediaSource Extensions concat in the browser:** MSE on WebM Opus chunks recorded via separate MediaRecorder starts is fragile (header continuity, codec init). Web Audio decode-and-schedule is robust and three decodes per passage is cheap.

**Trade-off accepted:** The cursor's beat-level position is interpolated linearly between bar anchors from `barTimeline`. This is visually smooth but does not claim sub-beat accuracy that DTW (±~200ms typical) cannot deliver.

### Failure modes

- **No piece identified for session, or alignment never produced for `[N, M]`:** DO returns 409. Route returns 409. The teacher LLM harness layer gates `play_passage` emission on the same piece-known signal that already gates `score_highlight`, so the 409 path is a defensive backstop logged to Sentry.
- **A chunk byte fetch fails after manifest load:** `PassagePlayer.load()` rejects; card surfaces "couldn't load audio" on the play button. Score clip + annotation still render.
- **R2 byte route returns 404 (chunk was never uploaded — possible if recording stopped mid-chunk):** Same as decode failure; card degrades to score+text only.

No silent fallbacks. Every failure either shows in the UI or logs to Sentry.

## Modules

### `PassagePlayer` (NEW, DEEP)
- **Interface:** `class PassagePlayer { constructor(manifest); play(); pause(); seek(tSec); onTick(cb: (tSec: number) => void): () => void; destroy(); readonly duration: number; readonly state: "idle" | "loading" | "ready" | "playing" | "paused" | "ended" | "error" }`
- **Hides:** `AudioContext` lifecycle, parallel `fetch` + `decodeAudioData` for each chunk, sequential `AudioBufferSourceNode.start(when, offset, duration)` scheduling honoring `startOffsetSec` / `endOffsetSec`, RAF clock, drift correction against `AudioContext.currentTime`.
- **Tested through:** public methods only — never spies on `AudioContext` internals. Tests use a stub `AudioContext` (Vitest mocks `window.AudioContext`) that records `start(when, offset, duration)` calls; assertions are on `onTick` outputs and `state` transitions.

### `buildPassageManifest` (NEW, DEEP)
- **Interface:** `function buildPassageManifest(args: { enrichedChunks: EnrichedChunk[]; bars: [number, number]; pieceId: string; sessionId: string; baseUrl: string }): PassageManifest | { error: "no_alignment" | "out_of_range" }`
- **Hides:** Filtering chunks whose `bar_coverage` overlaps `[N, M]`; locating first/last alignments for bars N and M; computing `startOffsetSec`/`endOffsetSec` and `barTimeline`; assembling chunk URLs.
- **Tested through:** pure function tests with synthetic `EnrichedChunk[]` arrays — no DO, no fetch, no I/O.

### `play_passage` tool (NEW; extension of `TOOL_REGISTRY` in `tool-processor.ts`)
- **Interface (Anthropic):** `{ session_id, bars: [N,M], focus_bars: [N,M]?, dimension, annotation }`. Description in the tool string explicitly notes "only emit when piece is identified and bars are covered by score alignment."
- **Hides:** Zod validation (bars range, focus_bars ⊆ bars), normalization to camelCase config keys.
- **Tested through:** `processToolUse(ctx, studentId, "play_passage", input)` against a stub `ServiceContext`; assertions on returned `InlineComponent[]`.

### `PlayPassageCard` (NEW, SHALLOW — justified)
- **Interface:** `(props: { config: PlayPassageConfig; onExpand?: () => void; artifactId?: string }) => JSX`
- **Composes:** `scoreRenderer.getClip()`, `api.sessions.getPassage()`, `new PassagePlayer()`, focus-bar tint div, annotation text, play/pause button.
- **Justification for shallowness:** This is a leaf UI component orchestrating three deep modules. Adding internal abstraction would not hide complexity — it would multiply it. The card's logic is "fetch manifest, hand to player, draw cursor from `onTick`."
- **Tested through:** RTL render with `vi.mock` boundary stubs for `scoreRenderer`, `passage-player`, and `api.sessions.getPassage` — never spies on internal React state.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/services/tool-processor.ts` | Add `play_passage` tool (schema + Anthropic schema + processor + registry entry). Remove `reference_browser` tool entirely (schema, Anthropic schema, processor, registry entry). | Modify |
| `apps/api/src/services/passage-manifest.ts` | Pure `buildPassageManifest()` service. | New |
| `apps/api/src/services/passage-manifest.test.ts` | Unit tests for `buildPassageManifest()`. | New |
| `apps/api/src/routes/sessions.ts` | Hono router with `GET /:id/passage?bars=N-M`; talks to DO via internal `/passage` route. | New |
| `apps/api/src/routes/sessions.test.ts` | Endpoint tests (auth, ownership, 200, 409). | New |
| `apps/api/src/routes/practice.ts` | Add `GET /chunk?sessionId=&chunkIndex=` for auth-gated R2 read-through of `sessions/{id}/chunks/{i}.webm`. | Modify |
| `apps/api/src/routes/practice.test.ts` | Test for the new chunk-read route. | Modify |
| `apps/api/src/do/session-brain.ts` | In `fetch()`, add branch for `/passage?bars=N-M`: load `chunk_enriched:*` from storage, call `buildPassageManifest`, return JSON or 409. | Modify |
| `apps/api/src/index.ts` | Mount `sessionsRoutes` at `/api/sessions`. | Modify |
| `apps/web/src/lib/types.ts` | Add `play_passage` to `InlineComponent`; add `PlayPassageConfig`. Remove `reference_browser` from union; remove `ReferenceBrowserConfig`. | Modify |
| `apps/web/src/lib/api.ts` | Add `api.sessions.getPassage(sessionId, bars)`. | Modify |
| `apps/web/src/lib/passage-player.ts` | `PassagePlayer` class. | New |
| `apps/web/src/lib/passage-player.test.ts` | Behavior tests for `PassagePlayer` against stub `AudioContext`. | New |
| `apps/web/src/components/cards/PlayPassageCard.tsx` | The card. | New |
| `apps/web/src/components/cards/PlayPassageCard.test.tsx` | RTL behavior tests. | New |
| `apps/web/src/components/InlineCard.tsx` | Route `play_passage` → `PlayPassageCard`. | Modify |
| `apps/web/src/components/Artifact.tsx` | Add `play_passage` case in `getCollapsedProps`. | Modify |
| `apps/api/src/services/tool-processor.test.ts` | Add `play_passage` validation test; remove `reference_browser` references. | Modify |
| `apps/web/src/routes/app.sandbox.tsx` | Drop reference to `ReferenceBrowserConfig` import; add a `PlayPassageCard` fixture under the existing artifact-sandbox layout for visual review. | Modify |

## Open Questions

- **Q:** Should the chunk byte route stream from R2 with `Range` request support (for HTTP seek), or return the whole 15s blob each time?  
  **Default:** Whole-blob response. Web Audio decodes the full buffer before scheduling; partial-range support is unnecessary for the beta and adds a code path we don't need.

- **Q:** Should `play_passage` tool calls also persist to the `messages` table as a structured component (like `score_highlight` does today), or live only in the SSE stream?  
  **Default:** Follow existing pattern — `processToolUse` returns the `InlineComponent` which the chat pipeline already persists as part of the assistant message's `components`. No new persistence path.

- **Q:** Last-chunk `durationSec` — the final chunk of a session may be shorter than 15s if the student stopped mid-chunk. How does the manifest learn its real duration?  
  **Default:** The R2 object's `Content-Length` is not enough (encoded WebM size ≠ playback duration). The DO does not currently persist per-chunk duration. For the beta we assume 15.0s for non-final chunks; for the final chunk that may carry bar M, the `PassagePlayer` calls `AudioBuffer.duration` post-decode and clamps scheduling to that value. The manifest's `endOffsetSec` is treated as an upper bound, not a contract.
