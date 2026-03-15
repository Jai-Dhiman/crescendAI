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
| `src/hooks/usePracticeSession.ts` | Remove MOCK_MODE flag and all mock branches. Add chunk upload state tracking (`ChunkState[]`). Add `wsStatus` state (`useState`, see WebSocket Status section). Add `isOnline` state for network awareness. Add `askHowWasThat()` method. Remove `mockSessionData` from return type. Remove `elapsedSeconds` from `stop()` dependency array (no longer needed without mock). Increase max reconnect attempts from 3 to 5. Restructure `ws.onclose` handler for exponential backoff with `wsStatus` transitions. |
| `src/lib/practice-api.ts` | Add `ask()` streaming method for `POST /api/ask`. Reuse `ChatStreamEvent` type from `api.ts` (identical wire format, no new type needed). |
| `src/lib/observation-throttle.ts` | **New file.** Pure TypeScript class for 3-minute observation delivery window. No React dependencies. |
| `src/components/AppChat.tsx` | Remove `mockSessionData` references. Add `handleAskHowWasThat()` function that calls `practice.askHowWasThat()` with an `onEvent` callback wired to the existing `appendDelta`/`streamingIndexRef`/`flushDeltas` machinery (same pattern as `handleSend`). Build session summary from real observations in the `practice.summary` useEffect. Gate ScorePanel auto-open behind `import.meta.env.DEV`. |
| `src/components/ChatInput.tsx` | Add optional props: `onAskHowWasThat?: () => void` and `showHowWasThat?: boolean`. When `showHowWasThat` is true, render a "How was that?" button alongside the record button. |
| `src/components/RecordingBar.tsx` | Add upload progress indicator (spinner/count). Add WebSocket reconnection indicator. Accept new props: `wsStatus`, `chunkStates`. |
| `src/components/ListeningMode.tsx` | Add `wsStatus` prop. Show "Reconnecting..." indicator when `wsStatus === "reconnecting"` while `state` is still `"recording"` (these are independent -- `wsStatus` tracks connection health, `PracticeState` tracks the session lifecycle). The existing `state === "error"` close trigger is unchanged. |
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

Upload flow per chunk (encapsulated in `uploadChunkWithRetry`):
1. Set chunk state to `{ index, status: "uploading" }`
2. Call `practiceApi.uploadChunk(sessionId, idx, blob)` -> `{ r2Key }`
3. On success: update state to `"complete"`, send WS `{ type: "chunk_ready", index, r2Key }`
4. On failure: wait 2s, retry the full sequence (upload + WS notify). If retry also fails: update state to `"failed"`, `Sentry.captureException`, continue to next chunk. Do not send `chunk_ready` on failure -- the server never processes an unfulfilled chunk.

### RecordingBar Upload Indicator

Minimal visual disruption (student is playing piano):
- All complete: nothing shown
- Uploading: small `CircleNotch` spinner next to chunk count
- Any failed: amber dot with failed count, non-blocking

### WebSocket Status

New `useState` in `usePracticeSession`: `wsStatus: "connected" | "reconnecting" | "disconnected"`.

State transitions managed in the `ws.onclose` handler (restructured from current flat-delay approach):

```
ws.onopen  -> setWsStatus("connected"), reset attempt counter
ws.onclose -> if state === "recording" && attempts < 5:
                setWsStatus("reconnecting")
                delay = min(1000 * 2^attempt, 30000)  // 1s, 2s, 4s, 8s, 16s, 30s cap
                setTimeout -> reconnect
              else:
                setWsStatus("disconnected")
                setError("Connection lost"), setState("error"), cleanup()
```

The current `RECONNECT_DELAY_MS = 2000` constant and flat retry are replaced by the exponential calculation inline in `onclose`.

- **Connected:** normal operation, indicator hidden
- **Reconnecting:** subtle "Reconnecting..." text with spinner in RecordingBar/ListeningMode. `PracticeState` stays `"recording"` (session is still active, just the WS pipe is down). Chunk uploads continue independently.
- **Disconnected:** after 5 attempts, transition to error state and cleanup.

### Network Offline Handling

New `useState<boolean>` in `usePracticeSession`, initialized from `navigator.onLine`. Updated via `online`/`offline` event listeners (registered in a `useEffect`).

Integration with chunk uploads in `recorder.ondataavailable`:

```typescript
recorder.ondataavailable = async (event) => {
  if (event.data.size === 0) return;
  const idx = chunkIndexRef.current++;

  if (!isOnlineRef.current) {
    // Queue blob for later upload
    offlineQueueRef.current.push({ index: idx, blob: event.data });
    updateChunkState(idx, "uploading"); // visually queued
    return;
  }

  await uploadChunkWithRetry(sessionId, idx, event.data);
};
```

`offlineQueueRef` is a `useRef<Array<{ index: number; blob: Blob }>>([])`. When the `online` event fires, the handler flushes the queue sequentially (in index order) via `uploadChunkWithRetry`, then clears it.

- **Offline:** queue blobs, show "Offline" indicator in RecordingBar. MediaRecorder continues capturing audio.
- **Online:** flush queued chunks in order, resume normal flow. Each flushed chunk follows the same upload-then-notify-WS sequence.

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
  onEvent: (event: ChatStreamEvent) => void,
): Promise<void>
```

Uses SSE streaming from `POST /api/ask`, same `data: {json}\n` wire format as `api.chat.send()`. Reuses the `ChatStreamEvent` type from `api.ts` (identical `start`/`delta`/`done` event structure). No new type needed.

### Hook Method

`usePracticeSession` exposes:

```typescript
askHowWasThat(
  conversationId: string | null,
  onEvent: (event: ChatStreamEvent) => void,
): Promise<void>
```

Delegates to `practiceApi.ask()` with the current `sessionIdRef`. Throws if no active session.

### UI Integration

**ChatInput changes:** `ChatInput.tsx` receives two new optional props:
- `onAskHowWasThat?: () => void` -- callback when "How was that?" is tapped
- `showHowWasThat?: boolean` -- controls visibility (true when `practice.state === "recording"` or `"summarizing"`)

When `showHowWasThat` is true, ChatInput renders a "How was that?" button alongside the existing record button.

**AppChat wiring:** `AppChat.tsx` defines a `handleAskHowWasThat()` function that:
1. Sets `isStreaming = true`
2. Appends a streaming placeholder message (same pattern as `handleSend`)
3. Calls `practice.askHowWasThat(activeConversationId, onEvent)` where `onEvent` is a callback that uses the existing `appendDelta`/`streamingIndexRef`/`flushDeltas` refs -- identical to the `api.chat.send` callback in `handleSend`
4. On `done`: finalize message, set `isStreaming = false`
5. On error: remove placeholder, toast error, do not crash session

This function is passed to `ChatInput` via the `onAskHowWasThat` prop.

Independent of WebSocket observations. Server bypasses throttle for explicit asks.

---

## Session Summary

The summary is built in `usePracticeSession`'s `handleWsMessage` (replacing the current mock summary logic), not in `AppChat`. This keeps the hook as the single source of truth for session state.

When the server sends `{ type: "session_summary", observations, summary }` via WebSocket:

1. In `handleWsMessage` `session_summary` case:
   a. Call `throttle.drain()` to get any undelivered queued observations
   b. Merge server observations with drained observations (deduplicate by text)
   c. Set `observations` state with the merged list
   d. Build summary string:
      ```
      I listened to {chunksProcessed} sections of your playing.

      During the session, I noticed:
      - {observation 1 text}
      - {observation 2 text}
      ...

      Want to hear more about any of these?
      ```
   e. Call `setSummary(builtSummary)`
   f. Set state to `"idle"`, cleanup

2. In `AppChat`, the existing `useEffect` on `practice.summary` appends the summary + any notepad notes as a chat message (same as current behavior, no change needed).

State machine confirmation: `recording -> summarizing -> (wait for session_summary WS) -> idle`. The `"summarizing"` state is set in `stop()`, the `"idle"` transition happens in the WS handler.

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
