# iOS Live Practice Pipeline Design

**Goal:** iOS users can record piano practice and receive real teacher observations and session synthesis from the live backend pipeline, with session history persisted in SwiftData and accessible from the sidebar.

**Not in scope:** Score rendering (ScoreHighlightCard remains a placeholder box), exercise assignment API calls, focus mode integration, conversation search, metronome, deferred synthesis recovery, KeychainService JWT auth.

---

## Problem

Every backend integration point in the iOS app is mocked with `Task.sleep` and hardcoded strings:

- `sendMessage()` in `ChatViewModel.swift:131` — fake echo after 0.5s sleep
- `stopPractice()` in `ChatViewModel.swift:83` — hardcoded observations after 1.5s sleep
- `askForFeedback()` in `ChatViewModel.swift:108` — canned observation string after 1.5s sleep
- `requestElaboration()` in `ChatViewModel.swift:145` — hardcoded elaboration string
- `APIEndpoints.swift` references `/api/upload` and `/api/analyze` — neither endpoint exists on the current API
- **Auth is broken end-to-end:** iOS sends `Authorization: Bearer <jwt>` but the API's `authSessionMiddleware` calls `auth.api.getSession({ headers })` which reads `better-auth` session cookies, not Bearer tokens. Every authenticated request returns 401.

No audio ever reaches the server. The iOS app is a UI prototype with no live data.

---

## Solution

After this change:

1. Tapping record calls `POST /api/practice/start` — a real DO session and conversationId are created on the backend
2. Every 15-second AAC chunk (with silence gating: RMS < 0.01 = skipped) is uploaded to `POST /api/practice/chunk`
3. A WebSocket connection to `/api/practice/ws/:sessionId` delivers real `observation` events as the backend processes chunks
4. Observations appear in the chat as `ObservationCard` messages; when the teacher includes inline components, `ArtifactRenderer` renders the appropriate card (ExerciseSetCard, ScoreHighlightCard, KeyboardGuideCard) below the message
5. Tapping stop ends audio capture and the backend synthesizes a post-session summary via the WebSocket `synthesis` event
6. The synthesis appears in chat as a teacher message; the `conversationId` is persisted to `ConversationRecord` in SwiftData
7. The sidebar lists past sessions from `ConversationRecord`; tapping one loads the full conversation from `GET /api/conversations/:id`
8. Text chat (`sendMessage`) calls `POST /api/chat` SSE; teacher responses with components also go through `ArtifactRenderer` — the same renderer, same cards, source-agnostic

---

## Design

### Auth

The API uses better-auth with Apple as a social provider. The iOS app must call `POST /api/auth/sign-in/social` with `{ provider: "apple", idToken: { token: "<apple_identity_token>" } }`. On success, better-auth returns a 200 with `Set-Cookie: better-auth.session_token=...; HttpOnly; Path=/`. `URLSession.shared`'s `HTTPCookieStorage` stores this cookie automatically and sends it on all subsequent requests to `api.crescend.ai`. No manual header injection is needed.

`AuthService` is updated:
- New `signIn(identityToken:userId:email:)` public method (testable without mocking `ASAuthorizationAppleIDCredential`)
- `handleAuthorization` calls `signIn` after extracting the identity token
- `loadStoredCredentials` checks `HTTPCookieStorage.shared` for a non-expired `better-auth.session_token` cookie
- `signOut` deletes cookies from `HTTPCookieStorage.shared`
- `URLSession` is injectable via init parameter (`URLSession = .shared`) for testing
- `jwt` and Keychain JWT storage are removed; `appleUserId` is kept as a computed property returning non-nil when `isAuthenticated` (for `ProfileView` compatibility)

`APIClient.attachAuth` (Bearer header injection) is removed. `SyncService` manual Bearer header is removed. Both rely on `URLSession.shared`'s cookie jar.

### Practice Pipeline

`PracticeSessionService` is a `@MainActor @Observable final class` that owns the complete session lifecycle:

1. `start()`: calls `POST /api/practice/start`, starts `PracticeSessionManager` audio capture (with `inferenceProvider: nil` — cloud inference replaces local), connects WebSocket, starts timer, starts chunk upload observer
2. Chunks from `PracticeSessionManager.chunkStream` are uploaded as raw AAC bytes. `ChunkProducer.produceChunk()` gates on `rms(samples:)` — if RMS < 0.01, the chunk is not yielded (never uploaded)
3. WebSocket reconnect: exponential backoff (delay = `min(1.0 × 2^attempt, 30.0)` seconds), 5 max attempts. During reconnect, `PracticeEvent.reconnecting(attempt:)` is emitted; audio and uploads continue uninterrupted
4. `stop()`: stops audio via `PracticeSessionManager.endSession()`, waits up to 30s for `synthesis` WebSocket event, then closes the connection and persists `ConversationRecord` to SwiftData
5. Publishes all events via `AsyncStream<PracticeEvent>`

### Chat (SSE)

`ChatService` is a `@MainActor @Observable final class`. `send(message:conversationId:)` opens a URLSession data task to `POST /api/chat`, accumulates SSE bytes line by line, and returns an `AsyncStream<ChatEvent>`. An internal `SSEParser` struct handles partial-chunk accumulation and maps each `data: {...}\n\n` line to a typed `ChatEvent`.

### Artifacts

`ArtifactConfig` is a Codable enum that maps API component JSON (`{ "type": "...", "config": {...} }`) to Swift structs:

```
.exerciseSet(ExerciseSetConfig)    ← type: "exercise_set"
.scoreHighlight(ScoreHighlightConfig) ← type: "score_highlight"
.keyboardGuide(KeyboardGuideConfig)   ← type: "keyboard_guide"
.unknown(type: String)            ← graceful degradation
```

`ArtifactRenderer` is a `@ViewBuilder` struct that switches on `ArtifactConfig` and renders the existing card view. Used in `ChatView.chatRow()` for any message whose `artifacts` array is non-empty.

`ChatMessage` gains `artifacts: [ArtifactConfig]` and a new `.teacher` role (for SSE chat responses). The `.system` role is kept for session lifecycle messages ("Session started", "Connecting...").

### ChatViewModel Coordination

`ChatViewModel` subscribes to `PracticeSessionService.eventStream` in a background `Task` spawned from `configure(modelContext:)`. It also subscribes to each `ChatService.send()` call's returned stream. It maps incoming events to `ChatMessage` values and appends them to `messages`. It has no knowledge of WebSocket, SSE, chunks, or audio.

For testing, `ChatViewModel.configureForTesting(modelContext:practiceService:chatService:)` accepts protocol conformances in place of concrete services.

### Session Persistence

`ConversationRecord` is a SwiftData `@Model` with `conversationId: String` (unique), `sessionId: UUID`, `startedAt: Date`, `title: String?`, `synthesisText: String?`. `PracticeSessionService` inserts a record when synthesis arrives. `SidebarView` uses `@Query` to fetch records, renders them in the session list, and on tap loads the full conversation from `GET /api/conversations/:id` via `APIClient`.

---

## Modules

### PracticeSessionService
**Interface:** `start() async throws`, `stop()`, `askForFeedback()`, `eventStream: AsyncStream<PracticeEvent>`, `state: State`, `currentLevel: Float`, `elapsedSeconds: TimeInterval`, `conversationId: String?`
**Hides:** PracticeSessionManager (audio), URLSessionWebSocketTask + reconnect loop, chunk upload queue, silence gate, WebSocket event parsing
**Depth:** DEEP — 4 public methods, ~300 lines of networking/audio/reconnect logic behind them
**Tested through:** `PracticeSessionServiceProtocol` mock in ChatViewModel tests; `ChunkProducer.rms(samples:)` static method for silence gate unit test

### ChatService
**Interface:** `func send(message: String, conversationId: String?) -> AsyncStream<ChatEvent>`
**Hides:** URLSession data task, SSE line parser (partial accumulation, `[DONE]` termination, error events)
**Depth:** DEEP — one public method, hides full SSE lifecycle
**Tested through:** `SSEParser.parse(line:)` tested with partial byte sequences

### ArtifactConfig (model)
**Interface:** `enum ArtifactConfig: Codable` with 4 cases
**Hides:** Custom Codable init that dispatches on `"type"` key
**Depth:** DEEP enough — hides the dispatch logic behind a typed enum
**Tested through:** `JSONDecoder().decode(ArtifactConfig.self, from:)` with API fixture JSON

### ArtifactRenderer
**Interface:** `ArtifactRenderer(config: ArtifactConfig)` → View
**Hides:** switch over cases, delegation to card views
**Depth:** SHALLOW — justified: pure routing, no logic to hide. Complexity lives in card views.
**Tested through:** `ArtifactConfig` decode tests; rendering verified via Xcode Previews

### ConversationRecord
**Interface:** `@Model final class ConversationRecord` — 5 stored properties
**Hides:** SwiftData persistence
**Depth:** SHALLOW — data container, no logic, no hidden complexity. Correct for a model.
**Tested through:** Insert + fetch using in-memory ModelContainer

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `CrescendAI/Networking/APIEndpoints.swift` | Replace all stale endpoints; add `signInSocial()`, `practiceStart()`, `practiceChunk(sessionId:chunkIndex:)`, `practiceWs(sessionId:conversationId:)`, `chat()`, `conversation(id:)`, `conversations()` | Modify |
| `CrescendAI/Networking/APIClient.swift` | Remove `attachAuth` Bearer header injection and `upload(audioData:title:)` / `analyze(performanceId:)` stale methods | Modify |
| `CrescendAI/Services/Auth/AuthService.swift` | New `signIn(identityToken:userId:email:)` public method; injectable URLSession; cookie-based credential storage/load/signout; remove JWT Keychain usage | Modify |
| `CrescendAI/Services/Auth/SyncService.swift` | Remove manual `Authorization: Bearer` header — cookie sent automatically | Modify |
| `CrescendAI/Features/Profile/ProfileView.swift` | Replace `appleUserId != nil` with `isAuthenticated`; remove `try` from `signOut()` call | Modify |
| `CrescendAI/Services/Practice/PracticeSessionService.swift` | New: full session lifecycle actor | New |
| `CrescendAI/Services/Chat/ChatService.swift` | New: SSE chat service | New |
| `CrescendAI/Models/ArtifactConfig.swift` | New: Codable enum + config structs | New |
| `CrescendAI/Models/ConversationRecord.swift` | New: SwiftData model | New |
| `CrescendAI/Features/Chat/ArtifactRenderer.swift` | New: @ViewBuilder artifact router | New |
| `CrescendAI/Features/Chat/ChatViewModel.swift` | Add `.teacher` role + `artifacts: [ArtifactConfig]` to `ChatMessage`; wire service streams; remove all `Task.sleep` mocks; add `configureForTesting` | Modify |
| `CrescendAI/Features/Chat/ChatView.swift` | Add `ArtifactRenderer` to `chatRow()` for messages with non-empty artifacts | Modify |
| `CrescendAI/Services/AudioEngine/ChunkProducer.swift` | Add `static func rms(samples: [Float]) -> Float`; silence gate in `produceChunk()`; remove inference call | Modify |
| `CrescendAI/Services/AudioEngine/PracticeSessionManager.swift` | Add `var chunkStream: AsyncStream<AudioChunk>?` computed property | Modify |
| `CrescendAI/App/SidebarView.swift` | Add `@Query var sessions: [ConversationRecord]`; wire `onSelectSession` to load conversation | Modify |
| `CrescendAI/App/CrescendAIApp.swift` | Add `ConversationRecord.self` to ModelContainer schema | Modify |

---

## Open Questions

- **Q:** Does better-auth's Apple social provider accept native iOS identity tokens (from `ASAuthorizationAppleIDCredential.identityToken`) at `POST /api/auth/sign-in/social`?  
  **Default:** Proceed with `{ provider: "apple", idToken: { token: "..." } }`. If rejected (401), add a custom lightweight `/api/auth/apple` route to the API that validates the native token and creates a better-auth session.

- **Q:** Does `URLSessionWebSocketTask` send `HTTPCookieStorage` cookies for `wss://` connections?  
  **Default:** Yes — `URLSessionWebSocketTask` uses the same `URLSession` cookie storage as HTTP tasks. Verify empirically on first session attempt; if cookies are absent, send `Cookie` header manually on the WebSocket URL.
