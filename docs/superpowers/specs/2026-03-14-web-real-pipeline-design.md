# Web App: Real Pipeline Integration

Wire the CrescendAI web practice companion from mock mode to the real inference and observation pipeline. The API endpoints already exist; this spec covers the client-side (apps/web/) changes only.

## Scope

- Remove `MOCK_MODE` flag from `usePracticeSession.ts`
- Real session lifecycle (start, chunk upload, WebSocket, stop)
- Observation throttle (3-minute window, client-side)
- "How was that?" explicit ask via `/api/ask`
- Session summary in chat
- Error handling (reconnection, offline, auth expiry)
- ScorePanel gated to dev-only (real score alignment not yet available)

## Out of Scope

- `apps/api/` (Rust worker) -- endpoints exist
- `apps/ios/` (Swift)
- `model/` (Python ML pipeline)
- New components for exercises or scores (separate agent)
- Score alignment / bar-range mapping (pipeline status: NOT STARTED)

---

## Architecture

### File Changes

| File | Change |
|------|--------|
| `src/hooks/usePracticeSession.ts` | Remove MOCK_MODE flag and all mock branches. Add chunk upload state tracking (`ChunkState[]`). Add `wsStatus` state. Add `askHowWasThat()` method. Remove `mockSessionData` from return type. Increase max reconnect attempts from 3 to 5. |
| `src/lib/practice-api.ts` | Add `ask()` streaming method for `POST /api/ask`. Add `AskStreamEvent` type. |
| `src/lib/observation-throttle.ts` | **New file.** Pure TypeScript class for 3-minute observation delivery window. No React dependencies. |
| `src/components/AppChat.tsx` | Remove `mockSessionData` references. Wire "How was that?" to `practice.askHowWasThat()`. Build session summary from real observations. Gate ScorePanel auto-open behind `import.meta.env.DEV`. |
| `src/components/RecordingBar.tsx` | Add upload progress indicator (spinner/count). Add WebSocket reconnection indicator. Accept new props: `wsStatus`, `chunkStates`. |
| `src/components/ListeningMode.tsx` | Add reconnection indicator. Accept `wsStatus` prop. |
| `src/stores/score-panel.ts` | Gate `open()` behind `import.meta.env.DEV` -- no-op in production. |

### Data Flow

```
User taps Record
  -> usePracticeSession.start()
    -> POST /api/practice/start -> sessionId
    -> WS /api/practice/ws/{sessionId} -> connected
    -> MediaRecorder.start(15000)

Every 15s chunk:
  -> practiceApi.uploadChunk(sessionId, idx, blob)
    -> { r2Key } response
    -> WS send: { type: "chunk_ready", index, r2Key }
  -> On failure: retry once after 2s, then mark failed, continue

Server pushes via WS:
  -> { type: "chunk_processed", scores }
    -> update latestScores, chunksProcessed
    -> throttle.onChunkProcessed() -> may release queued observation
  -> { type: "observation", text, dim, framing }
    -> throttle.enqueue(obs)
      -> if window open + minChunks met: show toast immediately
      -> if throttled: queue (replacing lower-priority), release on next tick()

User taps "How was that?":
  -> practiceApi.ask({ sessionId, conversationId })
    -> SSE streaming response -> chat messages (reuses existing appendDelta pattern)

User taps Stop:
  -> MediaRecorder.stop() (triggers final ondataavailable)
  -> WS send: { type: "end_session" }
  -> Server pushes: { type: "session_summary", observations, summary }
    -> throttle.drain() for undelivered observations
    -> build summary chat message
    -> cleanup
```

---

## Component Designs

### ObservationThrottle (`src/lib/observation-throttle.ts`)

A plain TypeScript class with no React dependencies and no internal timers.

```typescript
class ObservationThrottle {
  constructor(options?: {
    windowMs?: number;          // default: 180_000 (3 minutes)
    minChunksBeforeFirst?: number; // default: 4
  })

  enqueue(obs: ObservationEvent): ObservationEvent | null;
  // Returns observation immediately if window open + minChunks met.
  // Otherwise queues it (queue size = 1, new replaces old). Returns null.

  onChunkProcessed(): ObservationEvent | null;
  // Increments chunk count. Returns queued observation if now releasable.

  tick(): ObservationEvent | null;
  // Called from the hook's existing 1-second elapsed timer.
  // Returns queued observation if throttle window has expired.

  drain(): ObservationEvent[];
  // Returns all queued observations. Used at session end for summary.

  reset(): void;
  // Clears all state for reuse.
}
```

Design rationale:
- No internal timers. The hook's existing `setInterval` (1-second elapsed timer) calls `tick()`. Avoids zombie timers and simplifies cleanup.
- Queue size = 1. Per pipeline spec, only top-1 candidate kept between windows. Server already ranks candidates.
- `minChunksBeforeFirst = 4` matches pipeline doc (1 minute of playing before first observation).
- Server handles ranking; throttle only gates delivery timing on the client side.

### Chunk Upload Tracking

New types in `usePracticeSession.ts`:

```typescript
type ChunkStatus = "uploading" | "complete" | "failed";
interface ChunkState { index: number; status: ChunkStatus; }
```

Exposed as `chunkStates: ChunkState[]` on the hook return.

Upload flow per chunk:
1. Push `{ index, status: "uploading" }`
2. On success: update to `"complete"`
3. On failure: wait 2s, retry once. If retry fails: update to `"failed"`, `Sentry.captureException`, continue to next chunk

### RecordingBar Upload Indicator

Minimal visual disruption (student is playing piano):
- All complete: nothing shown
- Uploading: small `CircleNotch` spinner next to chunk count
- Any failed: amber dot with failed count, non-blocking

### WebSocket Status

New state: `wsStatus: "connected" | "reconnecting" | "disconnected"`

- **Connected:** normal operation
- **Reconnecting:** subtle text in RecordingBar ("Reconnecting...") with spinner. Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s cap. Max 5 attempts.
- **Disconnected:** after max attempts, show "Connection lost" error, transition to error state, cleanup.

### Network Offline Handling

Listen to `online`/`offline` browser events:
- **Offline:** pause chunk uploads (queue blobs in memory ref), show "Offline" indicator. Recording continues.
- **Online:** flush queued chunks in order, resume normal flow.

Lightweight: `useState<boolean>` + event listeners + blob queue ref.

### Auth Token Expiry

On 401 from chunk upload or `/api/ask`:
- Set error: "Session expired. Please sign in again."
- Transition to error state, cleanup
- Sentry breadcrumb for auth failure
- Do not silently retry

---

## "How Was That?" Integration

### API Method

Added to `practice-api.ts`:

```typescript
async ask(
  sessionId: string,
  conversationId: string | null,
  onEvent: (event: AskStreamEvent) => void,
): Promise<void>
```

Uses SSE streaming from `POST /api/ask`, same `data: {json}\n` wire format as `api.chat.send()`. Event types: `start`, `delta`, `done`.

### Hook Method

`usePracticeSession` exposes:

```typescript
askHowWasThat(
  conversationId: string | null,
  onEvent: (event: AskStreamEvent) => void,
): Promise<void>
```

Delegates to `practiceApi.ask()` with the current `sessionIdRef`.

### UI Integration

A "How was that?" button appears in `ChatInput` only while `practice.state === "recording"` (or within a brief window after stopping, before summary arrives). Tapping it:
1. Calls `practice.askHowWasThat(activeConversationId, onEvent)`
2. `AppChat` handles the streaming response identically to `handleSend` -- reuses `appendDelta`, `streamingIndexRef`, `flushDeltas`
3. Response appears as an assistant message in the chat

Independent of WebSocket observations. Server bypasses throttle for explicit asks.

---

## Session Summary

When the server sends `{ type: "session_summary" }` via WebSocket after recording ends:

1. Collect delivered observations from session state
2. Call `throttle.drain()` for any undelivered queued observations
3. Build assistant chat message:
   ```
   I listened to {chunksProcessed} sections of your playing.

   During the session, I noticed:
   - {observation 1 text}
   - {observation 2 text}
   ...

   Want to hear more about any of these?
   ```
4. Append student's notepad notes if any (same as current behavior)
5. Insert as assistant message via `setMessages`

---

## ScorePanel (Dev-Only)

The `ScorePanelStore` currently depends on `MockSessionData`. Since real score alignment (mapping chunk timestamps to bar numbers) is not yet implemented, the ScorePanel is gated:

- `score-panel.ts`: `open()` checks `import.meta.env.DEV` and is a no-op in production
- `AppChat.tsx`: the `useEffect` that auto-opens ScorePanel from `mockSessionData` is removed
- Mock imports (`mock-session.ts`) are retained but only imported in dev builds (tree-shaken in production)

---

## Mock Mode Removal

### What Gets Removed

- `const MOCK_MODE = true;` flag
- `if (MOCK_MODE) { ... }` block in `start()` (lines 197-206)
- `if (MOCK_MODE) { ... }` block in `stop()` (lines 303-324)
- `mockSessionData` state and return value
- `generateMockSession()` import and call

### What Gets Retained

- `mock-session.ts` file stays (used by dev-only ScorePanel and useful for future testing)
- `MockSessionData` type stays (referenced by ScorePanel store)
- Import gated: only referenced in dev-only code paths, tree-shaken in production

---

## Error Handling Summary

| Scenario | Behavior |
|----------|----------|
| Microphone denied | Error state, toast message (existing) |
| Session start fails (POST /api/practice/start) | Error state, Sentry, cleanup (existing) |
| WebSocket connect fails | Error state, Sentry, cleanup (existing) |
| WebSocket drops during recording | Reconnect with exponential backoff (1s-30s, 5 attempts), show "Reconnecting..." |
| Max reconnect attempts exceeded | Error state, "Connection lost", cleanup |
| Chunk upload fails | Retry once after 2s, then mark failed, Sentry, continue |
| Auth 401 during session | Error state, "Session expired", cleanup |
| Network offline | Pause uploads, queue blobs, show "Offline", resume on reconnect |
| `/api/ask` stream fails | Toast error, do not crash session |
