# V8a — Direct-Action Tools (anchor: `assign_segment_loop`) + Chat Tool-Use Parity

**Goal:** Replace observation-only feedback with a durable, behavior-changing artifact (`assign_segment_loop`) that can be produced by the synthesis path *and* the chat path, gated per trigger context, and verified across sessions via a strict-isolation loop-attempt detector — so the system can actually break the 90%-playthrough practice habit instead of describing it.

**Not in scope:**
- The other two action tools named in `docs/harness.md` — `render_annotation` (deferred to V8b) and `schedule_followup_interrupt` (deferred to V8c).
- Quality-gated completion (per-attempt dimension scoring against baseline). V8a counts repetitions only; quality gating is V8a.1+.
- Cross-session expiry policy (timed decay, max-backlog). The single-active-loop invariant collapses the policy surface; expiry is a future iteration.
- Soft-anchor assignments for unidentified pieces (text-only passage descriptors). Pre-condition failure flows back to the LLM as a tool error; the LLM falls back to text.
- New wire-protocol changes for chat tool turns. The streaming chat path (delivered by Spec X) uses `runStreamingHook` → `runPhase1Streaming` and emits `TeacherEvent` SSE events unchanged. `assign_segment_loop` plugs into the existing streaming pipeline; no new discriminators or payload shapes.
- Verovio rendering inside the segment-loop card. The card has a flagged-off score-snippet slot; rendering itself ships in V8b.
- iOS surfaces. V8a is web-only.
- Production rollout gating. Reuses the V6 launch flag — no separate V8a flag.

---

## Problem

The current pipeline ends at *describing* what happened. The teacher LLM produces an observation; the student reads it; the next session looks like every previous session. The wiki page on Score Following and Music Education reports an empirical finding from field research on home practice: over 90% of practice time is start-to-finish playthrough rather than isolated-passage work, and students repeat sections a fixed number of times regardless of whether errors were corrected. A passive feedback system reinforces that structure rather than changing it.

Today's seams confirm the gap:

1. `apps/api/src/services/chat.ts` calls a streaming text path and never invokes a tool. A student who asks "what should I work on next?" gets a paragraph, not an assignment.
2. The synthesis path emits `SynthesisArtifact` (`apps/api/src/harness/artifacts/synthesis.ts`) whose richest pedagogical move is `proposed_exercises` — three free-text strings the student can ignore.
3. The harness vocabulary (`docs/harness.md`) names three "action tools" — `assign_segment_loop`, `render_annotation`, `schedule_followup_interrupt` — and the V5 skill catalog references them in compound markdown, but no implementation exists. There is no D1/Postgres table for assignments, no atom file in `apps/api/src/harness/atoms/` (V6's atom directory is unbuilt), no client card, and `wrap_tool_call` is the V6 spec's named extension point left as a no-op.

V6 (the agent-loop substrate) is a prerequisite. V6's spec — `docs/specs/2026-04-27-v6-agent-loop-orchestration-design.md` — explicitly excludes action tools (line 7) and explicitly excludes chat-path migration (line 6). Both are V8a's load.

---

## Solution (from the user's perspective)

Two surfaces, one durable artifact.

**Live chat.** The student types "what should I focus on?" The teacher LLM, now driving the V6 agent loop on the chat path, decides whether the right move is text or an assignment. When it chooses an assignment, the chat thread renders a `SegmentLoopArtifact` card in *pending* state ("Your teacher suggests: loop bars 12–16, 3 clean times — Accept / Skip"). Accepting transitions the assignment to *active*; skipping dismisses it. The card persists in the conversation thread and in a sidebar list of active assignments.

**Post-session synthesis.** When a session ends, the synthesis compound (V6's `OnSessionEnd` hook) runs. If its analysis surfaces a passage that warrants targeted work, it dispatches `assign_segment_loop` directly — no confirmation, the student is no longer at the keyboard. The synthesis WebSocket message arrives with the assignment as a component alongside any proposed exercises, and the next session opens with the assignment visible.

**During practice.** The DO loads the student's active assignment for the current piece at session start. As the student plays, a passage-loop detector watches the score-following position track for *isolated* attempts at the assigned bars — a span that begins near the assigned start and ends near the assigned end, not a start-to-finish playthrough that happens to traverse those bars. Each isolated attempt advances the counter; the card updates live over the existing observation WebSocket. When the counter reaches `required_correct`, the assignment status flips to `completed`. The counter persists across sessions until completed, dismissed, or superseded by a new assignment for the same piece.

**One assignment per piece.** Creating a new assignment for a piece that already has an active loop archives the prior loop as `superseded`. There is at most one active assignment per (student, piece) at any time. This collapses the backlog problem the design intentionally avoids.

---

## Design

### Approach

V8a is purely additive to V6. Every load-bearing seam is a named V6 extension point.

**The action tool is an atom with `kind: 'action'`.** The V5/V6 catalog vocabulary already distinguishes "read tools" from "action tools" in `docs/harness.md`, but the V6 codegen treats every atom uniformly. V8a adds a single optional frontmatter field — `kind: 'read' | 'action'` (default `read`) — and Phase 1's tool-set builder routes action atoms through the `wrap_tool_call` middleware where read atoms pass through directly. The directory layout is unchanged: `assign-segment-loop.md` sits alongside other atoms in `docs/harness/skills/atoms/`. Codegen change is one new optional field, not a new directory.

**Permission gating lives in `wrap_tool_call`, by trigger context.** V6's middleware ships as a no-op pass-through specifically so V8a fills in the body, not the wiring. The body reads `ctx.trigger`: when chat, the action tool's effective default status is `pending`; when synthesis, `active`. The atom itself owns the trigger-aware status default — the middleware is the policy seam, the atom is the implementation seam. Future action tools (`render_annotation`, `schedule_followup_interrupt`) declare their own gating policy in the same middleware.

**Chat is a one-phase compound.** V6's two-phase pattern (Phase 1 auto-dispatch → Phase 2 forced single artifact write) exists because synthesis must produce exactly one durable `SynthesisArtifact`. Chat has no such constraint — a chat turn may be text-only, text + one assignment, or text + multiple artifacts. V8a generalizes V6's `runHook` by adding a `phases: 1 | 2` field to `CompoundBinding` (default 2 for `OnSessionEnd` back-compat). The `OnChatMessage` binding declares `phases: 1` and a tool allowlist; Phase 1 runs to terminal-text-turn and the loop ends.

**Chat parity is dispatch + tool-allowlist parity.** The streaming chat path (delivered by Spec X) uses `runStreamingHook` → `runPhase1Streaming` and emits `TeacherEvent` SSE events to the client unchanged. V8a registers `assign_segment_loop` in the `OnChatMessage` binding's tool list (so Anthropic sees the schema) and intercepts it in `chatV6`'s `processToolFn` closure (so the correct `trigger:'chat'` default status and DB call go through `services/segment-loops`). A `SegmentLoopArtifact` component arrives as a standard `tool_result` SSE event, rendered by the existing artifact pipeline extended in Task 12.

**The lifecycle is a state machine with one non-monotonic edge.** `pending → active → completed/dismissed` is the monotone path. The non-monotonic edge — `superseded` from any non-terminal state — implements the single-active invariant explicitly. A partial unique index on `(student_id, piece_id) WHERE status = 'active'` enforces it at the storage layer; `services/segment-loops.ts` enforces it at the service layer with a supersede-then-insert transaction.

**The loop-attempt detector lives in the DO, not in synthesis-time atoms.** V6 atoms run inside Phase 1 against accumulated signals; V8a's detector runs continuously alongside the existing score-following machinery (today centralized in `apps/api/src/do/session-brain.ts`). It consumes the position track produced upstream and emits `LoopAttempt { in_bounds, ts, passage }` events to the DO accumulator. The accumulator increments the assignment's counter (with a same-passage debounce window), persists the increment, and broadcasts a `loop_attempt` message over the existing observation WebSocket so the card counter updates live.

**Strict isolation defines a loop attempt.** A `LoopAttempt` is a contiguous span of score-following hits whose start lies within a tolerance window of the assigned start AND whose end (via stop, restart-near-start, or stream end) lies within a tolerance window of the assigned end. A start-to-finish playthrough that traverses the assigned bars produces zero events. This is the pedagogical claim: the counter does not advance on the very behavior the assignment is meant to suppress. False positives and false negatives are not detected automatically — every `LoopAttempt` event is logged with its raw position-track span for offline tuning.

**The `SynthesisArtifact` schema gains `assigned_loops`.** When synthesis Phase 1 dispatches `assign_segment_loop`, the loop is created server-side as a side effect *and* the resulting artifact reference is passed into Phase 2 as context. Phase 2's voice prompt mentions the assignment in `headline`; the schema's new `assigned_loops: SegmentLoopRef[]` array is required (may be empty) and validated to ensure every reference points at a loop Phase 1 actually created — the model cannot hallucinate loop IDs. The DO consumer maps `assigned_loops` → WebSocket `components` alongside `proposed_exercises`.

**One harness flag, one launch sequence.** V8a reuses `HARNESS_V6_ENABLED`. When the flag flips on, both synthesis and chat route through the harness loop. Implication: V6 must ship and soak with the flag off, then flip on, *then* V8a's PR can land safely. Bisection is at PR-revert granularity, not flag granularity. Acknowledged trade-off in exchange for a smaller config surface.

### Trade-offs chosen

- **Repetition-only completion (no quality gate).** Counting clean repetitions is honest about what we can verify cheaply (existing score-following + bar analysis). Quality gating requires per-attempt dimension comparison against a moving baseline — its own subsystem. The design ships a wedge against passive practice; V8a.1 adds the quality gate when the rest of the surface is stable.
- **Strict isolation over lenient overlap.** A wider "any contiguous span containing the bars" detector would re-introduce the failure mode the spec exists to suppress. The cost of strict isolation is potential false negatives on legitimately wide practice (e.g., student isolating bars 8–18 around an assignment of 12–16). False negatives are recoverable — the student dismisses or the assignment supersedes — and they're observable in logs.
- **Single-active invariant per (student, piece).** Versus expiry-based or backlog-capped models, single-active collapses the policy surface to one rule and matches how teachers actually work ("let's see how that went before I add anything new"). Cost: a student cannot have concurrent loops on different passages of the same piece. If usage data shows that's a real need, V8b extends to a backlog-cap-of-N.
- **Confirm in chat, auto in synthesis (gating by trigger).** Versus always-confirm or always-auto, trigger-aware gating matches the actual UX context: a synthesis call has no one to confirm with; a chat turn is conversational and the student is right there. Cost: two policy paths through `wrap_tool_call`. The middleware boundary is exactly the right place for that policy to live.
- **Reuse `componentsJson` on `messages` rather than add a new `tool_uses` column.** The column already exists for inline UI components. An assistant message that emits an assigned-loop artifact stores its reference in `componentsJson`. Raw tool-call traces for observability live in the harness event stream's structured logs, not in the messages table.
- **Reuse `HARNESS_V6_ENABLED` instead of a separate V8a flag.** Smaller config surface; explicit launch ordering. Trade-off documented above.
- **Tool-restricted chat parity.** The chat allowlist is `['search_catalog', 'create_exercise_artifact', 'assign_segment_loop']` for V8a. Future tools join the allowlist explicitly, surface by surface, rather than auto-inheriting chat exposure.
- **Score-snippet slot, not Verovio rendering.** The `SegmentLoopArtifact` card has a Verovio slot that renders only when `flags.scoreSnippet` is true. V8a leaves the flag off; V8b ships the rendering with proper bar-range MEI slicing and list-view performance work. The integration point exists and is testable; the heavy work is sequenced.

### Hook taxonomy and event shape

V8a binds `OnChatMessage` (declared but unbound in V6 per V6 spec line 75). `runHook<H>(hook, ctx)` parameterization is unchanged; the addition is one map entry in `compound-registry.ts` and one new event type in the union for the chat one-phase path:

```typescript
type HookEvent<TArtifact> =
  | { type: 'phase1_tool_call';   id: string; tool: string; input: unknown }
  | { type: 'phase1_tool_result'; id: string; tool: string; ok: true;  output: unknown }
  | { type: 'phase1_tool_result'; id: string; tool: string; ok: false; error: string }
  | { type: 'phase1_done';        toolCallCount: number; turnCount: number }
  | { type: 'phase2_started' }
  | { type: 'artifact';           value: TArtifact }
  | { type: 'validation_error';   raw: unknown; zodError: string }
  | { type: 'phase_error';        phase: 1 | 2; error: string }
```

`OnSessionEnd` continues to emit `artifact` (Phase 2 forced write) via `runHook`. `OnChatMessage` uses `runStreamingHook` and yields `TeacherEvent` directly — no `HookEvent` wrapper on the chat path. V8a adds no new event types.

### Lifecycle state machine

```
                   ┌─────────────┐
        chat ─────►│   pending   │── decline ──►┌────────────┐
                   └──────┬──────┘              │ dismissed  │
                          │ accept              │ (terminal) │
                          ▼                     └────────────┘
   synth ────────────►┌─────────┐── dismiss ───►       ▲
                      │ active  │                      │
                      └────┬────┘                      │
                  attempts │                           │
            == required    │                           │
                           ▼                           │
                    ┌────────────┐                     │
                    │ completed  │                     │
                    │ (terminal) │     supersede ──────┘
                    └────────────┘     (any non-terminal source)
                                          ▼
                                     ┌────────────┐
                                     │ superseded │
                                     │ (terminal) │
                                     └────────────┘
```

Transitions are validated in `services/segment-loops.ts`. Every illegal transition throws `InvalidStateTransition`. Terminal states (`completed`, `dismissed`, `superseded`) are immutable.

### Data flow — chat path

```
POST /api/chat/send { message }
  → services/chat.ts (HARNESS_V6_ENABLED branch)
    → buildHookContext({ trigger:'chat', studentId, conversationId, sessionId?, pieceId? })
    → runHook('OnChatMessage', ctx)                 [V6, phases:1]
      → Phase 1, tool_choice:auto, allowlist [search_catalog,
        create_exercise_artifact, assign_segment_loop]
        → Sonnet may emit tool_use { name:'assign_segment_loop', input:{...} }
          → wrap_tool_call middleware reads ctx.trigger='chat'
          → atom: assign-segment-loop
            → precondition: piece_id present (else ToolPreconditionError)
            → services/segment-loops.createSegmentLoop({ ..., status:'pending' })
              → tx: archive prior active loop for (student,piece) as superseded
              → INSERT segment_loops { status:'pending', ... }
            → returns SegmentLoopArtifact{ card_state:'pending' }
          → tool_result block back to Sonnet
        → Sonnet emits terminal text turn
        → text_response event + phase1_done
    → adapter consumes stream:
      - if any phase1_tool_call observed → buffer everything, return JSON
        { kind:'tool_turn', text, artifacts:[SegmentLoopArtifact] }
      - else → stream text_response content over SSE as today
    → INSERT messages { role:'assistant', content:text, componentsJson:[ref] }
  → web AppChat renders text + SegmentLoopArtifact card (pending)
    → Accept → POST /api/segment-loops/:id/accept   → status='active'
    → Skip   → POST /api/segment-loops/:id/decline  → status='dismissed'
```

### Data flow — synthesis path

```
DO alarm → runSynthesisAndPersist (V6 path)
  → buildHookContext({ trigger:'synthesis', sessionId, studentId, pieceId, signals })
  → runHook('OnSessionEnd', ctx)                    [V6, phases:2]
    → Phase 1: molecule tools + utility atoms + assign_segment_loop
      → Sonnet may dispatch assign_segment_loop
        → wrap_tool_call middleware reads ctx.trigger='synthesis'
        → atom creates segment_loop with status='active' (no Accept gate)
    → Phase 2: write_synthesis_artifact forced
      → voice prompt receives DiagnosisArtifact[] AND SegmentLoopArtifact[]
      → Sonnet calls write_synthesis_artifact with assigned_loops:[{id,bars,...}]
      → SynthesisArtifactSchema validates including refinement: every id in
        assigned_loops must reference a loop Phase 1 actually created
    → artifact event yielded
  → DO consumer
    → persistSynthesisMessage(headline, components: exercises ⊕ loops)
    → broadcast WS { type:'synthesis', text:headline, components }
```

### Data flow — live counting during practice

```
DO session start
  → load active assignment: services.findActiveForPiece(studentId, pieceId)
  → if found: hydrate passage_loop_detector with assignment bars
  → broadcast WS { type:'segment_loop_status', assignment, attempts_completed }

For each chunk during practice:
  HF inference → score_follower → position track segment
    → passage_loop_detector.processPosition(track, activeAssignment)
      → emits LoopAttempt { in_bounds, ts, passage } | null
    → if in_bounds AND passage matches:
      → services.incrementAttempts(assignmentId, +1)
        → UPDATE segment_loops SET attempts_completed += 1
        → if reached required_correct: status='completed'
      → broadcast WS { type:'loop_attempt', assignment_id, attempts_completed,
                       completed_now }
    → web card updates counter live
```

### Cross-session continuity

```
Session N ends → synthesis may create or update assignment
                → segment_loops row persists (status='active' typically)

Time passes... Session N+1 opens for same piece
  → DO session start
    → findActiveForPiece returns the existing assignment
    → detector hydrates with same bars
    → web client receives initial WS status with current counter
  → student practices
    → counter accumulates across sessions until required_correct reached
  → assignment never decays except via 'completed', 'dismissed', or
    'superseded'-by-new-assignment (single-active invariant)
```

### Error handling

- **`assign_segment_loop` invoked without `piece_id`.** Atom precondition throws `ToolPreconditionError{reason:'no_piece_identified'}`. V6 Phase 1 dispatch returns `tool_result {is_error:true, content:'no_piece_identified'}` to the model. Sonnet falls back to a text response. The failed call is observable in conversation traces — the eval signal Q6 (b) optimizes for. Adapter sees no successful tool calls → returns SSE text path.
- **Sonnet calls tool with malformed input** (e.g., `bars_end < bars_start`, unknown dimension). Atom Zod input validation fails. Same path as precondition failure — structured tool error returned to model.
- **Race: two parallel synthesis writers create a loop for the same (student, piece).** Postgres partial unique index on `status='active'` raises constraint violation. Service catches, retries the supersede-then-insert transaction once; if still failing, treats as "another writer won" and returns the existing active loop to the caller. No duplicate active loops ever exist.
- **`incrementAttempts` after assignment was dismissed mid-session.** Service rejects: source state is terminal. DO swallows, removes the assignment from in-memory hydrated state, stops counting. Logged as `{event:'increment_after_terminal', loop_id, prior_state}`.
- **`SynthesisArtifactSchema` validation fails because Phase 2 emitted `assigned_loops` with an unknown ID.** New schema refinement catches it; `validation_error` event emitted; DO does NOT persist; `needsSynthesis` stays true; V6's deferred-recovery path retries.
- **`runHook('OnChatMessage')` raises before any event.** Adapter try/catch in `services/chat.ts` falls back to legacy `streamText` path for that turn; logs `harness_invocation_failed` to Sentry; user gets a normal text reply.
- **Adapter sees tool_use events but Sonnet's terminal text turn errors out.** V6's `withRetries` retries once; if still failing, emits `phase_error`. Adapter persists the assistant message with `componentsJson` populated but empty `content`; returns `{kind:'tool_turn', text:'', artifacts}`. Client renders the artifact card without preceding text.
- **Accept/decline endpoint hit on a non-`pending` loop.** Service throws `InvalidStateTransition`. API returns 409 `{error:'invalid_state', current_state}`. Client refetches and rerenders the correct state.
- **Cross-student access** — student A attempts to mutate student B's loop. Service-level ownership check returns 404 (not 403 — never disclose existence).
- **WebSocket disconnect during practice; client misses `loop_attempt` events.** On reconnect, client emits new WS request `request_segment_loop_status`; DO replies with current `attempts_completed`. Client overrides local card state.
- **DO eviction mid-session with active assignment hydrated.** Existing DO state-reload pattern in `session-brain.ts` re-fetches via `findActiveForPiece` at the next async boundary.
- **Detector false positive / false negative.** Not auto-detected — pedagogical, not system. Every `LoopAttempt` event is logged with its raw position-track span for offline tuning.
- **`HARNESS_V6_ENABLED=false` after a `pending` chat-created loop exists.** Loop persists in DB. Practice path still hydrates it for sessions. Chat surface (now legacy) won't surface it in the conversation thread until flag flips back. Acceptable degradation.

---

## Modules

### Deep modules (new)

#### `apps/api/src/services/segment-loops.ts`
- **Interface:** `createSegmentLoop(ctx, input) → SegmentLoopArtifact`, `acceptSegmentLoop(ctx, id)`, `declineSegmentLoop(ctx, id)`, `dismissSegmentLoop(ctx, id)`, `findActiveForPiece(ctx, studentId, pieceId) → SegmentLoopArtifact | null`, `incrementAttempts(ctx, id, delta) → { attempts_completed, completed_now }`.
- **Hides:** Single-active-invariant enforcement (atomic supersede-then-insert under the partial unique index, with explicit transaction since Postgres partial uniques don't compose with upsert). Status-transition validation. Trigger-context-driven default status (`pending` for chat, `active` for synthesis). Ownership check.
- **Tested through:** Service-level integration tests against a test Postgres database. Behaviors: creating an active loop while one exists archives the prior as `superseded`; trigger-driven default status; legal/illegal transitions; idempotent threshold-hit; concurrent createSegmentLoop produces exactly one active.
- **Depth verdict:** DEEP.

#### `apps/api/src/do/passage-loop-detector.ts`
- **Interface:** `processPosition(positionTrack: PositionSpan, activeAssignment: SegmentLoopRef): LoopAttempt | null`, called inside the DO's per-chunk handler alongside score-following and bar-aligned analysis.
- **Hides:** Strict bounded-entry-and-exit detection. Tolerance-window classification of span start/end against assigned bars. Exit-signal detection (silence, restart-near-start, stream end). Same-passage repetition debouncing. Out-of-bounds suppression (start-to-finish playthroughs that traverse the assigned bars produce no event).
- **Tested through:** Unit tests with synthesized deterministic position tracks. Behaviors: clean isolated loop → one event with `in_bounds:true`; start-to-finish playthrough → zero events; wide entry/exit → zero events; back-to-back loops → multiple events with distinct timestamps; tolerance-boundary span pinned to a documented behavior.
- **Depth verdict:** DEEP. The boundary-detection logic is the load-bearing pedagogical claim of V8a.

#### `apps/api/src/harness/atoms/assign-segment-loop.ts`
- **Interface:** Standard V6 atom signature — `assignSegmentLoop(ctx: AtomContext, input): Promise<SegmentLoopArtifact>`.
- **Hides:** Pre-condition check on `piece_id` (throws `ToolPreconditionError` on null). Delegation to `services/segment-loops.createSegmentLoop`. Trigger-context derivation from `ctx.trigger` and propagation to the service for default-status logic.
- **Tested through:** Atom unit tests. Behaviors: precondition failure on null `piece_id`; trigger-driven default; output validates against `SegmentLoopArtifactSchema`.
- **Depth verdict:** DEEP. Centralizes the trigger-to-status translation; `wrap_tool_call` middleware gates *invocation*, not status.

### Modified V6 modules

#### `apps/api/src/harness/loop/runHook.ts` (V6, modified)
- **Change:** Generalize to support compounds declaring `phases: 1 | 2`. For `phases: 1`, skip the forced Phase 2 entirely; the event stream terminates on the Phase 1 terminal text turn (emitting `text_response`). The compound binding gains `phases: 1 | 2` (default 2 for back-compat with `OnSessionEnd`).
- **Tested through:** New test case: an `OnChatMessage` binding with `phases:1` produces an event stream containing zero or more `phase1_tool_call`/`phase1_tool_result` pairs and exactly one `text_response`, no `phase2_*` events, no `artifact` event.
- **Depth verdict:** DEEP (existing).

#### `apps/api/src/harness/loop/compound-registry.ts` (V6, modified)
- **Change:** Add `OnChatMessage → chat-response` binding. Binding fields: `compoundMarkdownPath: 'docs/harness/skills/compounds/chat-response.md'`, `phases: 1`, `toolAllowlist: ['search_catalog', 'create_exercise_artifact', 'assign_segment_loop']`, no forced artifact schema.
- **Tested through:** `getCompoundBinding('OnChatMessage')` returns a binding with the expected fields.
- **Depth verdict:** DEEP-ish (existing).

#### `apps/api/src/harness/loop/middleware.ts` (V6, modified)
- **Change:** Implement `wrap_tool_call` body. For `assign_segment_loop`: when `ctx.trigger === 'chat'`, the dispatched call's effective default-status target is `pending`; when `ctx.trigger === 'synthesis'`, target is `active`. For other tools today: pass-through. Future action tools (V8b/V8c) declare their own gating policy here.
- **Tested through:** Per-trigger tests: invocation under `chat` produces a `pending` artifact; under `synthesis` produces `active`. Other tools unaffected.
- **Depth verdict:** DEEP. The middleware boundary becomes the central permission-gating policy surface for action tools.

#### `apps/api/src/services/teacher.ts` (V6, modified — synthesis voice + Phase 2)
- **Change:** Phase 2 voice prompt receives `SegmentLoopArtifact[]` from Phase 1 alongside `DiagnosisArtifact[]`. Voice prompt template instructs the model to mention any newly assigned loops in `headline`. Phase 2 input shape is extended; the model's tool input populates the new `assigned_loops` array.
- **Tested through:** `synthesizeV6` integration tests extended: a fixture session where Phase 1 calls `assign_segment_loop` produces a `SynthesisArtifact` with non-empty `assigned_loops`. Voice fixture asserts `headline` mentions the assignment.
- **Depth verdict:** DEEP (existing).

#### `apps/api/src/services/teacher.ts` — `chatV6` (modified)
- **Change:** The `chatV6` function already routes through `runStreamingHook('OnChatMessage', ...)`. V8a's changes: (1) populate `pieceId` and `trigger: 'chat'` in the `HookContext` passed to `runStreamingHook`; (2) in the `processToolFn` closure, intercept calls where `name === 'assign_segment_loop'` and dispatch to `services/segment-loops.processAssignSegmentLoopTool(ctx, studentId, input)` returning a `ToolResult` with `componentsJson: [{ type: 'segment_loop', config }]`. All other tool calls continue through `processToolUse` unchanged.
- **Tested through:** `teacher-chat-v6.test.ts` extended: fixture where Anthropic stub returns `assign_segment_loop` tool_use — verify `tool_result` SSE event is emitted with `type:'segment_loop'` component; precondition failure fixture returns `tool_error` SSE event.
- **Depth verdict:** DEEP (existing).

#### `apps/api/src/do/session-brain.ts` (modified)
- **Change:** On session start, call `findActiveForPiece(studentId, pieceId)`; if a loop exists, hydrate `passage-loop-detector` and broadcast `segment_loop_status` over WebSocket. On each chunk: feed score-following position track to the detector; on `LoopAttempt{in_bounds:true}`, call `incrementAttempts` and broadcast `loop_attempt`. On WebSocket reconnect message `request_segment_loop_status`, reply with current state. On session end, ensure all in-flight increments persist before invoking synthesis.
- **Tested through:** `practice-mode.test.ts` extended. Behaviors: session start with active assignment broadcasts initial status; a fixture position track of three isolated loops produces three `loop_attempt` messages and `attempts_completed=3`; cross-session continuation hydrates with prior counter.
- **Depth verdict:** DEEP (existing).

### Shallow but acceptable

#### `apps/api/src/harness/artifacts/segment-loop.ts`
- **Interface:** `SegmentLoopArtifactSchema` (Zod) + `SegmentLoopArtifact` (inferred type). Discriminator literal `kind: 'segment_loop'`. Includes all DB columns plus a derived `card_state` enum.
- **Hides:** Schema construction. Sibling pattern matches `diagnosis.ts`, `exercise.ts`, `synthesis.ts`.
- **Tested through:** Round-trip test (DB row → artifact → JSON → artifact, idempotent) and Zod refinement tests.
- **Depth verdict:** SHALLOW (schema-only). Justified by sibling consistency.

#### `apps/api/src/db/schema/segment-loops.ts`
- **Interface:** Drizzle table definition.
- **Hides:** Column types, constraints, the partial unique index.
- **Depth verdict:** SHALLOW (schema-only).

#### `apps/api/src/routes/segment-loops.ts`
- **Interface:** Three Hono routes (`POST /accept`, `POST /decline`, `POST /dismiss`), each a one-line delegation to the service.
- **Hides:** Hono routing boilerplate, error mapping (Service `InvalidStateTransition` → HTTP 409; `NotFoundError` → 404).
- **Depth verdict:** SHALLOW. Justified — routes should be thin.

#### `apps/web/src/components/cards/SegmentLoopArtifact.tsx`
- **Interface:** React component consuming a `SegmentLoopArtifact` ref; subscribes to the existing observation WebSocket for `loop_attempt` and `segment_loop_status` messages tagged with the assignment ID.
- **Hides:** State-driven render branches (pending, active, completed, dismissed, superseded). Live counter from WS. Score-snippet slot collapsed when `flags.scoreSnippet=false`.
- **Tested through:** Component tests with a stub WebSocket. Behaviors: per-state affordances; live counter on incoming WS message; Accept calls the right endpoint and re-renders.
- **Depth verdict:** SHALLOW (state-driven render).

### Non-modules

- No new SSE event types beyond `text_response` (in the harness event stream, not on the wire). The wire change is the `kind` discriminator on the chat response shape.
- No new WebSocket message types in V8a beyond `segment_loop_status`, `loop_attempt`, and `request_segment_loop_status` — all carried over the existing observations channel.
- No new Verovio integration. Slot-only.
- No iOS work.
- No new model routing — V6's `routeModel` unchanged.

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/db/migrations/000X_segment_loops.sql` | New migration: `segment_loops` table + partial unique index | New |
| `apps/api/src/db/schema/segment-loops.ts` | Drizzle table definition | New |
| `apps/api/src/db/schema/index.ts` | Re-export segment-loops table | Modify |
| `apps/api/src/harness/artifacts/segment-loop.ts` | Zod schema + type + tests | New |
| `apps/api/src/harness/artifacts/index.ts` | Register `SegmentLoopArtifact` in `artifactSchemas` and `ARTIFACT_NAMES` | Modify |
| `apps/api/src/harness/atoms/assign-segment-loop.ts` | Action atom + tests | New |
| `apps/api/src/harness/atoms/index.ts` | Register atom in catalog barrel | Modify |
| `apps/api/src/harness/skills/atoms/assign-segment-loop.md` | Skill markdown with `kind: action` frontmatter | New |
| `apps/api/scripts/catalog-codegen.ts` (V6 codegen) | Extend frontmatter parser to read optional `kind: 'read' | 'action'` field; default `read` | Modify |
| `apps/api/src/harness/skills/__catalog__/index.gen.ts` (V6 generated) | Build-time-emitted; gains `kind` field per atom | Generated |
| `apps/api/src/harness/loop/types.ts` (V6) | Add `pieceId?: string` and `trigger: 'chat' \| 'synthesis'` to `HookContext`; add `SegmentLoopRef` type; extend `ToolDefinition.invoke` to accept optional `PhaseContext` second arg | Modify |
| `apps/api/src/harness/loop/runHook.ts` (V6) | Branch on `binding.phases`; for `phases:1`, terminate after Phase 1 terminal text turn emitting `text_response` | Modify |
| `apps/api/src/harness/loop/compound-registry.ts` (V6) | Add `OnChatMessage → chat-response` binding | Modify |
| `apps/api/src/harness/loop/phase1.ts` (V6) | Tool-set builder reads atom `kind`; routes `kind:'action'` atoms through `wrap_tool_call`; emit `text_response` event on terminal text turn for one-phase compounds | Modify |
| `apps/api/src/harness/loop/middleware.ts` (V6) | Implement `wrap_tool_call` body for action tools; per-tool gating dispatched by name | Modify |
| `docs/harness/skills/compounds/chat-response.md` | New compound markdown describing the chat compound's procedure and tool list | New |
| `apps/api/src/services/segment-loops.ts` | Lifecycle service + tests | New |
| `apps/api/src/services/teacher.ts` | Phase 2 input shape extended; voice prompt updated | Modify |
| `apps/api/src/harness/artifacts/synthesis.ts` | `SynthesisArtifactSchema` gains `assigned_loops: SegmentLoopRef[]`; refinement validating every ref points at a loop Phase 1 created | Modify |
| `apps/api/src/services/teacher.ts` | `chatV6`: add `pieceId` and `trigger:'chat'` to `HookContext`; intercept `assign_segment_loop` in `processToolFn`; `synthesizeV6`: add `pieceId` and `trigger:'synthesis'` to `HookContext` | Modify |
| `apps/api/src/routes/chat.ts` | Fix flag: `HARNESS_V6_CHAT_ENABLED` → `HARNESS_V6_ENABLED` | Modify |
| `apps/api/src/routes/segment-loops.ts` | Three routes: accept/decline/dismiss | New |
| `apps/api/src/index.ts` | Mount `segmentLoopsRoutes` at `/api/segment-loops` via Hono `.route()` chain | Modify |
| `apps/api/src/do/passage-loop-detector.ts` | Strict-isolation detector + tests | New |
| `apps/api/src/do/session-brain.ts` | Hydrate active assignment on session start; broadcast loop status; feed detector during chunks; respond to status requests | Modify |
| `apps/api/src/lib/types.ts` | Add `SegmentLoopRef` type; ensure `HookContext` carries `pieceId?: string` and `trigger: 'chat' \| 'synthesis'` (V6 introduces these; V8a confirms presence) | Modify |
| `apps/web/src/components/cards/SegmentLoopArtifact.tsx` | React card with state-driven render + WS subscription | New |
| `apps/web/src/components/Artifact.tsx` | Register `segment_loop` discriminator and route to `cards/SegmentLoopArtifact.tsx` | Modify |
| `apps/web/src/components/AppChat.tsx` | No change needed — `segment_loop` artifacts arrive via existing `tool_result` SSE event | — |
| `apps/web/src/lib/api.ts` | Add typed methods for accept/decline/dismiss endpoints | Modify |

---

## Open Questions

- **Q: Where do `LoopAttempt` raw position-track spans persist for offline detector tuning?** Default: structured logs only (Sentry breadcrumbs + Workers logs). No new DB table for traces. If the detector needs offline replay against captured spans, that's a V8a.1 spec for an `attempt_traces` table.
- **Q: What tolerance window does the strict-isolation detector use for entry/exit?** Default: ±1 bar at both endpoints. Justification: less than ±1 risks missing legitimate isolation due to score-following position drift on the first onset; more than ±1 starts catching wide playthroughs. Tunable via env var or DO config; first deploy uses ±1.
- **Q: Is `required_correct` exposed to the LLM as a tool input or hardcoded?** Default: tool input, valid range 1–10. The LLM's pedagogical judgment chooses how many clean passes to require. Out-of-range values fail Zod input validation and route through the existing tool-error path.
- **Q: When a chat-created assignment is `pending`, does it surface in the practice DO's hydration?** Default: no — only `active` assignments hydrate. A pending assignment that the student hasn't accepted is conversational, not practice state. Practice continues unchanged until acceptance.
- **Q: What happens to a `pending` loop if the student starts a new conversation (different `conversation_id`) before accepting?** Default: the loop persists in `pending` until accepted, declined, or superseded by a new assignment. The card still appears in the original conversation thread (where it was created). The student can navigate back. There is no auto-decline on conversation context loss.
- **Q: Does `findActiveForPiece` consider `pending` loops in its uniqueness invariant?** Default: no. The partial unique index is on `status='active'`. A student may have one `pending` (created via chat) and one `active` (created via synthesis) for the same piece simultaneously. Accepting the pending supersedes the active. Trade-off: this makes synthesis-and-chat racing possible, which is a feature — both surfaces remain operational independently. The single-*active*-invariant is preserved; the broader "one pending or active" is not enforced and need not be.
- **Q: How is `dimension` captured on a segment loop?** Default: nullable text column matching the `DIMENSIONS` enum used elsewhere; the LLM can omit it (assignment is dimension-agnostic) or set it to one of the six. The card renders a dimension chip when present.
