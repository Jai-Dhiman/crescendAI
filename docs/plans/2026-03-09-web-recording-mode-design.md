# Web Recording Mode -- Design

**Date:** 2026-03-09
**Status:** Approved

## Problem

The web app has a text-only chat interface. Pianists cannot record and get feedback on their playing through the web -- only the iOS app (once built) will support audio capture. We need a full practice companion experience on web: continuous audio capture, real-time inference, teacher observations, and session summaries.

## Design

### Interaction Model

A floating overlay (like voice mode in ChatGPT/Claude) triggered by a green record button in the chat input area. The overlay sits on top of the chat view and contains:

- Real-time waveform visualizer (Web Audio API AnalyserNode, no server roundtrip)
- Session timer
- Toast area for teacher observations (auto-dismiss after ~8s)
- Stop button to end the session
- Minimizable to a floating pill (waveform + timer)

When the user stops, a synthesized summary of top observations posts to the chat. The user can then discuss the session with the teacher in chat.

### State Machine

```
Idle
  -> [click green button] -> Requesting mic permission
  -> [granted] -> Connecting (WS + session creation)
  -> [connected] -> Recording (continuous, chunking every 15s)
  -> [click stop] -> Summarizing (DO generating session summary)
  -> [summary received] -> Idle (summary posted to chat)
```

### Audio Capture (Client-Side)

- `navigator.mediaDevices.getUserMedia({ audio: true })` for mic access
- `AudioContext` with `AnalyserNode` for real-time waveform visualization
- `MediaRecorder` with 15-second `timeslice` for automatic chunking
- Encoding: WebM/Opus (browser-native, no transcoding)
- Each chunk: ~60-120KB for 15s of mono audio

### Architecture: Durable Object + R2

Each practice session is managed by a Cloudflare Durable Object. Audio chunks are uploaded to R2 via pre-signed URLs.

```
MediaRecorder.ondataavailable (every 15s)
  -> Request pre-signed R2 upload URL from Worker
  -> PUT chunk to R2 (sessions/{sessionId}/chunks/{index}.webm)
  -> Send WS message to DO: { type: "chunk_ready", index, r2Key }
  -> DO fetches chunk from R2
  -> DO sends to HF inference endpoint
  -> DO receives 6-dim scores
  -> DO runs teaching moment selection (outlier detection)
  -> If teaching moment: DO calls /api/ask -> pushes observation toast via WS
  -> DO stores chunk metadata + scores in transactional storage
```

### Why Durable Objects

- Per-session stateful server-side object: accumulates chunks, scores, observations in memory
- Hibernatable WebSockets: client connects once, DO pushes observations back in real-time
- Automatic lifecycle: spins up on session start, hibernates when idle
- No external state management needed -- the DO is the session

### Why R2 for Chunks

- Keeps WebSocket connection lightweight (control messages only, no large binary payloads)
- Audio chunks persist for replay and re-analysis
- Pre-signed URLs let the client upload directly without routing through the Worker
- Native to Cloudflare, no external dependencies

### WebSocket Protocol

```
Client -> DO:
  { type: "chunk_ready", index: number, r2Key: string }
  { type: "end_session" }

DO -> Client:
  { type: "session_started", sessionId: string }
  { type: "chunk_processed", index: number, scores: DimScores }
  { type: "observation", text: string, dimension: string }
  { type: "session_summary", observations: Observation[], summary: string }
  { type: "error", message: string }
```

### Teaching Moment Selection

Interim heuristic (STOP classifier not yet built):
- Maintain running mean + stddev per dimension across chunks
- Flag chunk as teaching moment when any dimension deviates > 1.5 stddev from running mean
- Also flag positive moments (significant improvement from baseline)

### Observation Delivery

- During recording: toast notification in the overlay (brief awareness, auto-dismiss)
- On session end: DO ranks all observations by significance (deviation magnitude), calls teacher LLM for a 2-3 sentence session summary, writes session to D1, sends summary via WS
- Client posts summary + top observations as a structured chat message
- User can continue discussing in chat

### API Surface (New Endpoints)

All under the existing Rust API Worker (`apps/api/`):

- `POST /api/practice/start` -- creates DO, returns `{ sessionId, wsUrl }`
- `GET /api/practice/upload-url?sessionId=X&chunkIndex=N` -- returns pre-signed R2 PUT URL
- `GET /api/practice/ws/:sessionId` -- WebSocket upgrade, forwarded to DO

Reuses existing:
- `POST /api/ask` -- called by DO for observation generation
- `POST /api/sync` -- session data syncs to D1

### UI Design

Uses existing design tokens from `apps/web/src/styles/app.css`:
- Waveform: sage green (#7A9A82) bars on espresso (#2D2926) background
- Toasts: cream (#FDF8F0) text on surface (#3A3633) background
- Typography: Lora serif, consistent with landing page
- Animations: fade-in-up for toasts, smooth waveform via requestAnimationFrame
