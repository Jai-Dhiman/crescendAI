# V6 — Agent Loop & Orchestration Design

**Goal:** Replace the linear Stage 4a/4b synthesis pipeline with a hook-driven, two-phase compound execution loop that dispatches V5 skill catalog molecules as Anthropic tools and writes a single Zod-validated `SynthesisArtifact` per `OnSessionEnd` invocation.

**Not in scope:**
- Migration of any hook other than `OnSessionEnd` (chat path stays on `services/teacher.ts:chat()`; live-practice path stays on Stage 4a/4b).
- Action tools (`assign_segment_loop`, `render_annotation`, `schedule_followup_interrupt`) — V8a.
- Per-molecule deferred trigger evaluation — V7+ (all molecules exposed unconditionally for the V6 pilot).
- Capability router with multiple model entries — `routeModel()` ships with one entry today (Sonnet 4.6 via `AI_GATEWAY_TEACHER`).
- Streaming Phase 2 voice tokens — current `synthesize()` is non-streaming; V6 preserves that.
- Production rollout gating (shadow-runs, percentage ramps) — zero-user reality, no gating needed.
- iOS or web UI changes — DO WebSocket payload shape (`{type:'synthesis', text, components}`) is preserved, so clients see no diff.

---

## Problem

Today's session synthesis lives in `apps/api/src/services/teacher.ts:synthesize()` and is structurally a hardcoded two-step inside one Anthropic call: prompt template (`buildSynthesisFraming`) + tool registry (`getAnthropicToolSchemas`) + free-form text + optional tool calls for exercises. The compound's "behavior" is implicit in the prompt; there is no separation between **what to diagnose** (analytical) and **how to speak it** (voice). Consequences:

1. **No structural enforcement of single-write.** The model can emit zero or many tool calls; whatever text comes back is whatever text comes back. Today's `result.text` is the raw model output minus `<analysis>` strip.
2. **No structural enforcement of artifact shape.** The output is a free-form string. The V5 `SynthesisArtifact` Zod schema (which exists at `apps/api/src/harness/artifacts/synthesis.ts`) is unreferenced by the runtime path.
3. **Skill catalog (V5) has no executable runtime.** `docs/harness/skills/{atoms,molecules,compounds}/*.md` (28 files) and `apps/api/src/harness/artifacts/{diagnosis,exercise,synthesis}.ts` (Zod schemas) are written and tested as data, but no code dispatches them. V5 shipped contracts; V6 ships the loop that honors those contracts.
4. **Chat and synthesis cannot share infrastructure.** Each is a separate code path with separate prompts, separate tool registries, separate streaming behavior. When chat eventually adopts tool_use beyond exercises (V7+), there is no shared loop to host both.

---

## Solution (from the user's perspective)

**Dev-facing only — zero users today.** With `HARNESS_V6_ENABLED=true`, when a practice session ends:

1. The DO claims its synthesis slot (unchanged).
2. The DO calls `synthesizeV6(ctx, input)` instead of `synthesize(ctx, input)`.
3. The harness loop runs in two phases against Anthropic via the existing AI Gateway:
   - **Phase 1 (analysis):** Sonnet 4.6 with all 9 molecule tools + 2 utility atoms (`prioritize-diagnoses`, `fetch-session-history`) registered, `tool_choice: auto`, sees a structured JSON digest of the V3 accumulator. It dispatches molecules across bar ranges; each tool call invokes the molecule's TS implementation, which returns a Zod-validated `DiagnosisArtifact`. Loop terminates when Sonnet emits a turn with no tool calls or hits an 8-turn cap.
   - **Phase 2 (forced write):** Sonnet 4.6 receives the collected `DiagnosisArtifact[]` plus a voice-instruction prompt, with `tool_choice: {type:'tool', name:'write_synthesis_artifact'}` forcing exactly one tool call. The tool input is validated against `SynthesisArtifactSchema`; on success, an `artifact` event is yielded.
4. The DO consumes the event stream: `phase1_*` events go to structured logs; the `artifact` event triggers existing persistence (`persistSynthesisMessage` reading `artifact.headline`, `persistAccumulatedMoments`, `clearNeedsSynthesis`) and an existing-shape WebSocket message (`{type:'synthesis', text:headline, components, isFallback:false}`).
5. `ctx.waitUntil(reviewArtifact(...))` fires the V4 review reviewer at 10% sample rate after persistence.

With `HARNESS_V6_ENABLED=false` (default), legacy `synthesize()` runs unchanged. The bisection switch is one env-var flip.

---

## Design

### Approach

Hybrid skill runtime, two-phase compound execution, explicit middleware composition, single-entry capability router, S2 adapter pattern in `services/teacher.ts`. Settled in the brainstorm session 2026-04-27.

**Hybrid runtime.** Atoms and molecules are TypeScript functions (deterministic, cached, no LLM round-trips). Compounds are LLM tool-use loops (Sonnet 4.6 driving Anthropic tool dispatch). The compound prompt includes the compound's markdown procedure section + a tool registry of available molecules and utility atoms. Atom and molecule markdown is a contract document — never sent to the model, validated against TS implementations by tests.

**Two-phase compound.** Phase 1 = analysis loop with `tool_choice: auto` over molecule + utility-atom tools. Phase 2 = forced-write call with `tool_choice: {type:'tool', name:'write_synthesis_artifact'}` and the analysis-phase artifacts as context. Single-write is structurally guaranteed by Phase 2's forced single tool.

**Trade-offs chosen:**

- **TS molecules over LLM molecules.** Rejects the wiki's "markdown is the program" framing for atoms/molecules because their procedures are deterministic branching (`if z > -1.0 return neutral; elif ratio > 0.85 ...`). Cost: token waste + nondeterminism. TS gives us cheap, fast, testable molecules. The Qwen-finetune voice replacement slots into Phase 2 only.
- **All 9 molecules exposed unconditionally.** Rejects deferred-loading for V6 because tool registry overhead is ~2KB tokens and the catalog has only 9 molecules. Deferred loading triggers structured-predicate frontmatter, which V5 didn't ship — premature for the pilot. Threshold: revisit V7 when total molecule count crosses 25 OR any single hook's registry crosses 12.
- **Two LLM round-trips per hook.** Phase 1 + Phase 2 are separate sessions. Cost: doubled latency vs a one-call loop, but synthesis is async/post-session so latency is non-binding. Benefit: structural single-write, clean Phase 1/Phase 2 separation for future Qwen voice routing, independent A/B of analysis vs voice prompts.
- **Single-entry capability router.** `routeModel()` returns Sonnet-via-`AI_GATEWAY_TEACHER` for every taskKind today. Stub abstraction; future routes (Qwen, Workers AI) slot into the map without changing call sites.
- **Async `after_model` review.** 10% sampler fires via `ctx.waitUntil` after the artifact is persisted. Non-blocking; the student-facing path (when there are students) is never delayed by review. Persistence-first then notify, matching the wiki's async-agents invariant.
- **No-op `wrap_tool_call` stub.** V6 has no action tools, but the wrapper's call site is plumbed through Phase 1's tool dispatch so V8a fills in a function body, not a thread.
- **S2 adapter at `services/teacher.ts`.** `synthesizeV6(ctx, input)` returns `AsyncIterable<HookEvent>`. DO chooses between `synthesize` and `synthesizeV6` by `HARNESS_V6_ENABLED` flag. Existing `synthesize` untouched; existing DO code (slot claiming, WebSocket forwarding, failure recovery) untouched.
- **Build-time catalog index.** A pre-build Bun script (`apps/api/scripts/catalog-codegen.ts`) parses `docs/harness/skills/**/*.md` and emits a typed TS module at `apps/api/src/harness/skills/__catalog__/index.gen.ts` with frontmatter + procedure metadata. Workers have no FS at runtime; this is the only valid mechanism. Wired via `package.json` `prebuild` and `predeploy` hooks (and a `prepare` hook for `bun install`) so `wrangler dev`, `wrangler deploy`, and `vitest` always see fresh output. The generated file is gitignored — never hand-edited.
- **Hybrid scope-by-plan.** Spec covers full V6 (loop + 15 atoms + 9 molecules). Implementation splits into four plans dispatched independently: `2026-04-27-v6-loop.md`, `2026-04-27-v6-atoms.md`, `2026-04-27-v6-molecules.md`, `2026-04-27-v6-integration.md`. Plan 1 ships the loop with a stub molecule registry (functional but empty); subsequent plans fill in atoms, then molecules, then final compound prompt + end-to-end fixtures.

### Hook taxonomy (parameterized for V7+)

```typescript
type HookKind =
  | 'OnStop'
  | 'OnPieceDetected'
  | 'OnBarRegression'
  | 'OnSessionEnd'    // bound in V6
  | 'OnWeeklyReview'
  | 'OnChatMessage'   // declared, unbound in V6
```

Only `OnSessionEnd` has a `CompoundBinding` registered in V6. The map shape (`Map<HookKind, CompoundBinding>`) and the parameterized event-stream type (`AsyncIterable<HookEvent<TArtifact>>`) ensure V7 chat is additive — add a map entry, write a chat compound, no refactor.

### Event stream shape

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

No `phase2_token` events. Current `synthesize()` is non-streaming; V6 preserves that. V7+ chat compound may extend this union with token events.

### Data flow

```
DO alarm fires → runSynthesisAndPersist (existing)
  → claim slot atomically (existing)
  → build topMoments, drillingRecords, baselines (existing)
  → if HARNESS_V6_ENABLED:
       synthesizeV6(ctx, input) → AsyncIterable<HookEvent<SynthesisArtifact>>
         → runHook('OnSessionEnd', ctx)
             → resolve compound binding (session-synthesis)
             → Phase 1: build tools registry from molecule + atom catalog,
                 build structured JSON digest (topMoments + drilling + transitions
                 + signal-ref index), call Sonnet with tool_choice:auto,
                 dispatch each tool through molecule TS impl, accumulate
                 DiagnosisArtifact[], terminate on no-tool turn or cap
             → Phase 2: build voice prompt with diagnoses, call Sonnet with
                 tool_choice forced to write_synthesis_artifact, validate
                 against SynthesisArtifactSchema, yield artifact event
       Adapter consumer in DO:
         - phase1_* events → structured log (Sentry breadcrumb)
         - artifact event → persistSynthesisMessage(headline), broadcast
                           WebSocket {type:'synthesis', text:headline, components},
                           persistAccumulatedMoments, clearNeedsSynthesis
         - validation_error / phase_error → Sentry capture, leave needsSynthesis=true
       ctx.waitUntil(reviewArtifact(artifact)) at 10% sample
     else:
       synthesize(ctx, input) (existing legacy path) unchanged
```

### Error handling

- **Anthropic 5xx / network failure inside Phase 1 or Phase 2** → `phase_error` event emitted, throws upstream wrapped as `InferenceError`. Existing DO try/catch handles it; `synthesisCompleted` stays true (claim retained), `needsSynthesis` stays true in DB so deferred `/synthesize` recovery path retries.
- **Phase 2 tool input fails Zod validation** → `validation_error` event emitted with raw payload + Zod error message. Adapter consumer logs to Sentry, does NOT persist, leaves `needsSynthesis=true` for retry.
- **Molecule TS function throws (signal cache miss, malformed bar range, etc.)** → caught at the Phase 1 dispatch boundary; emitted as `phase1_tool_result {ok:false, error}`. Returned to Sonnet as Anthropic `tool_result {is_error:true, content:errorMessage}`. Sonnet decides whether to retry, pick a different molecule, or stop. No process crash, no fatal escalation.
- **Phase 1 hits 8-turn cap without Sonnet stopping** → emit `phase1_done` with `turnCount=8`, fall through to Phase 2 with whatever artifacts were collected. Phase 2 prompt notes the cap.
- **Phase 2 returns no tool call** (model defies forced `tool_choice` — should be impossible per Anthropic API contract, defensive only) → `phase_error {phase:2, error:'no tool_use returned despite forced tool_choice'}`. Adapter logs, leaves `needsSynthesis=true`.
- **Build-time catalog index fails to generate** (malformed frontmatter, missing required field) → Vite plugin throws at build time. `vite dev` and `wrangler deploy` fail loudly. No silent runtime fallback.
- **`HARNESS_V6_ENABLED=false`** → `synthesizeV6` is not called; legacy `synthesize` path runs untouched. Bisection switch.

---

## Modules

The deep modules introduced by V6.

### `apps/api/src/harness/loop/runHook.ts`

- **Interface:** `runHook<H extends HookKind>(hook: H, ctx: HookContext): AsyncIterable<HookEvent<ArtifactFor<H>>>`
- **Hides:** Compound-binding resolution, two-phase orchestration, middleware composition, error recovery, event-stream construction. Single async generator.
- **Tested through:** Public `runHook` invocation with a fixture `HookContext` and a stub Anthropic gateway. Asserts on the emitted event sequence + final artifact.
- **Depth verdict:** DEEP. One function call hides Phase 1 + Phase 2 + middleware + dispatch + validation + errors.

### `apps/api/src/harness/loop/compound-registry.ts`

- **Interface:** `getCompoundBinding(hook: HookKind): CompoundBinding | undefined`
- **Hides:** Map literal, the `CompoundBinding` shape (compound markdown excerpt, tool list, artifact schema reference), future plug-in registration mechanism.
- **Tested through:** `getCompoundBinding('OnSessionEnd')` returns a binding pointing at `session-synthesis`; `getCompoundBinding('OnChatMessage')` returns undefined in V6.
- **Depth verdict:** DEEP-ish (small but encapsulates a future extension surface). Acceptable.

### `apps/api/src/harness/loop/phase1.ts`

- **Interface:** `runPhase1(ctx: PhaseContext, binding: CompoundBinding): AsyncIterable<Phase1Event>`
- **Hides:** Anthropic tool-call streaming parsing, multi-turn loop, tool dispatch through molecule registry, per-tool middleware (`wrapToolCall`), tool-result message construction back to the model, turn cap, structured logging.
- **Tested through:** Fixture Anthropic transcript (recorded JSON tool-call sequence), assert that tools are dispatched in order, errors propagate as tool_results, loop terminates correctly.
- **Depth verdict:** DEEP. Hides the entire tool-use loop behind one call.

### `apps/api/src/harness/loop/phase2.ts`

- **Interface:** `runPhase2(ctx: PhaseContext, binding: CompoundBinding, diagnoses: unknown[]): Promise<{ artifact: unknown } | { error: string; raw?: unknown }>`
- **Hides:** Forced-tool-choice request, Anthropic call via gateway, single tool-input extraction, Zod validation against `binding.artifactSchema`.
- **Tested through:** Mocked Anthropic returning a tool_use block; assert artifact is validated and returned. Then a malformed payload; assert `validation_error` shape.
- **Depth verdict:** DEEP. Wraps a forced single round-trip + validation behind one call.

### `apps/api/src/harness/loop/middleware.ts`

- **Interface:** `redactPii(req)`, `withRetries(call)`, `wrapToolCall(call)`, `reviewArtifact(artifact, ctx)`
- **Hides:** Each middleware's policy. Today: `redactPii` is a no-op pass-through (no PII in synthesis signals; future: redact student name from voice prompt). `withRetries` is a 1-retry exponential-backoff wrapper for 5xx/network. `wrapToolCall` is a no-op pass-through. `reviewArtifact` is a 10%-sample stub that emits a structured-log breadcrumb (V8+ replaces with real reviewer agent).
- **Tested through:** Each middleware called as a unit with stub input/output; assert pass-through or transformation.
- **Depth verdict:** SHALLOW individually (each is small) but collectively form a DEEP "middleware boundary" module. Justified: explicit composition (M2 in brainstorm), not chained. Each middleware's depth grows independently in future versions.

### `apps/api/src/harness/loop/route-model.ts`

- **Interface:** `routeModel(taskKind: TaskKind): GatewayClient`
- **Hides:** Map from task kind to model+gateway+API-key tuple. Today: every kind returns the Sonnet-via-`AI_GATEWAY_TEACHER` client. Tomorrow: Qwen voice client, Workers-AI analysis client.
- **Tested through:** `routeModel('phase1_analysis')` and `routeModel('phase2_voice')` both return a Sonnet client whose `model` field is `claude-sonnet-4-20250514`.
- **Depth verdict:** SHALLOW. Justified: stub abstraction with one entry today, but the call sites use it consistently so adding routes later is a one-line change.

### `apps/api/src/harness/loop/types.ts`

- **Interface:** Type-only module exporting `HookKind`, `HookEvent<T>`, `HookContext`, `PhaseContext`, `CompoundBinding`, `ArtifactFor<H>`, `TaskKind`, `GatewayClient`.
- **Hides:** Nothing — pure type declarations.
- **Tested through:** Compilation only. No runtime.
- **Depth verdict:** N/A (types). Centralizes contract surface so other modules don't co-define types.

### `apps/api/src/harness/skills/__catalog__/index.gen.ts` (build-time-emitted)

- **Interface:** `export const catalog: SkillCatalog`. Read-only structured data.
- **Hides:** Markdown parsing, frontmatter validation, procedure-section extraction. Emitted by Vite plugin at `apps/api/scripts/catalog-codegen.ts` from `docs/harness/skills/**/*.md`.
- **Tested through:** A unit test asserts `catalog.atoms.length === 15`, `catalog.molecules.length === 9`, `catalog.compounds.length === 4`, and per-skill frontmatter fields are present.
- **Depth verdict:** DEEP. The codegen + the runtime read are one logical module; the runtime side is just a typed import.

### `apps/api/src/harness/atoms/index.ts` and `apps/api/src/harness/atoms/{name}.ts` (15 files)

- **Interface per atom:** `compute{Name}(ctx: AtomContext, input: ZodInferredInput): Promise<ZodInferredOutput>`. Each atom has a Zod input schema and a Zod output schema co-located.
- **Hides:** Pure compute or pure read against signal cache, AMT MIDI, MuQ scores, score-alignment, DB.
- **Tested through:** Per-atom unit tests with fixture signal-cache reads + assertion on output shape.
- **Depth verdict:** DEEP individually (each atom hides a domain-specific computation behind one call). The barrel is shallow but it's a barrel.

### `apps/api/src/harness/molecules/index.ts` and `apps/api/src/harness/molecules/{name}.ts` (9 files)

- **Interface per molecule:** `run{Name}(ctx: MoleculeContext, input: ZodInferredInput): Promise<DiagnosisArtifact | ExerciseArtifact>`. Each molecule has a Zod input schema (typically `{ session_id, bar_range }`).
- **Hides:** Atom orchestration in the order specified by the molecule's markdown procedure, branching logic, Zod validation of the output artifact.
- **Tested through:** Per-molecule integration tests calling the molecule with fixture signal data through the public function; assert artifact shape + Zod validation. Test the *contract*, not the call sequence to atoms.
- **Depth verdict:** DEEP. Each molecule hides a 4-8-step procedure behind one call.

### `apps/api/src/services/teacher.ts` (modified — adds `synthesizeV6`)

- **Interface:** `synthesizeV6(ctx: ServiceContext, input: SynthesisInput): AsyncIterable<HookEvent<SynthesisArtifact>>`. Existing `synthesize` untouched.
- **Hides:** Construction of `HookContext` from `SynthesisInput`, invocation of `runHook('OnSessionEnd', hookCtx)`, pass-through of events.
- **Tested through:** Adapter integration test calling `synthesizeV6` with fixture `SynthesisInput` and stub Anthropic; assert the event stream matches expected shape.
- **Depth verdict:** SHALLOW. Justified: thin translation between the existing service-layer signature and the harness loop signature. Keeps DO untouched from harness concerns.

### `apps/api/src/do/session-brain.ts` (modified — flag-gated dispatch)

- **Interface unchanged.** Internal: `runSynthesisAndPersist` chooses adapter by env flag.
- **Hides:** All existing DO concerns (slot claim, WebSocket lifecycle, persistence, recovery) plus the flag dispatch.
- **Tested through:** Existing `practice-mode.test.ts` extended with a `HARNESS_V6_ENABLED='true'` case that asserts the V6 path produces the expected `{type:'synthesis', text, components}` WebSocket message.
- **Depth verdict:** DEEP (DO already is). Modification is surgical.

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/scripts/catalog-codegen.ts` | New build-time codegen for skill catalog index (Bun script) | New |
| `apps/api/package.json` | Add `prebuild`, `predeploy`, `prepare` hooks invoking codegen | Modify |
| `apps/api/.gitignore` | Add `src/harness/skills/__catalog__/index.gen.ts` | Modify |
| `apps/api/src/harness/skills/__catalog__/index.gen.ts` | Emitted by codegen; not hand-edited; not committed | Generated |
| `apps/api/src/harness/skills/__catalog__/codegen.test.ts` | Asserts catalog content matches docs/harness/skills | New |
| `apps/api/src/harness/loop/types.ts` | Type-only contract surface | New |
| `apps/api/src/harness/loop/runHook.ts` | Public entry: hook → event stream | New |
| `apps/api/src/harness/loop/runHook.test.ts` | Behavior test through public interface | New |
| `apps/api/src/harness/loop/compound-registry.ts` | `Map<HookKind, CompoundBinding>` | New |
| `apps/api/src/harness/loop/compound-registry.test.ts` | Binding lookup tests | New |
| `apps/api/src/harness/loop/phase1.ts` | Analysis loop with tool dispatch | New |
| `apps/api/src/harness/loop/phase1.test.ts` | Tool-dispatch + multi-turn tests | New |
| `apps/api/src/harness/loop/phase2.ts` | Forced-write + Zod validation | New |
| `apps/api/src/harness/loop/phase2.test.ts` | Validation + forced-tool tests | New |
| `apps/api/src/harness/loop/middleware.ts` | redactPii, withRetries, wrapToolCall, reviewArtifact | New |
| `apps/api/src/harness/loop/middleware.test.ts` | Per-middleware behavior tests | New |
| `apps/api/src/harness/loop/route-model.ts` | Single-entry capability router | New |
| `apps/api/src/harness/loop/route-model.test.ts` | Router stub tests | New |
| `apps/api/src/harness/atoms/{15 names}.ts` | TS implementations of 15 atoms | New (15) |
| `apps/api/src/harness/atoms/{15 names}.test.ts` | Per-atom behavior tests | New (15) |
| `apps/api/src/harness/atoms/index.ts` | Barrel exporting all atoms + atomRegistry | New |
| `apps/api/src/harness/molecules/{9 names}.ts` | TS implementations of 9 molecules | New (9) |
| `apps/api/src/harness/molecules/{9 names}.test.ts` | Per-molecule behavior tests | New (9) |
| `apps/api/src/harness/molecules/index.ts` | Barrel exporting molecules + moleculeRegistry | New |
| `apps/api/src/services/teacher.ts` | Add `synthesizeV6` adapter; keep `synthesize` | Modify |
| `apps/api/src/services/teacher.test.ts` | Add `synthesizeV6` integration tests | Modify |
| `apps/api/src/do/session-brain.ts` | Flag-gated dispatch in `runSynthesisAndPersist` | Modify |
| `apps/api/src/lib/types.ts` | Add `HARNESS_V6_ENABLED: string` to `Bindings` | Modify |
| `apps/api/wrangler.toml` | `[vars] HARNESS_V6_ENABLED = "false"` | Modify |
| `apps/api/src/routes/practice.test.ts` (or equivalent) | E2E test exercising flag=true path | Modify |

---

## Open Questions

- **Q: Does `redactPii` middleware do anything in V6?** Default: stub (pass-through). Voice prompt builder in Phase 2 already takes structured `DiagnosisArtifact[]` not raw student PII; `headline` field is generated by the model. Add real redaction in V8+ when chat path lands and student names enter the prompt.
- **Q: What is the 8-turn cap for Phase 1?** Default: 8 turns hard cap. Compound markdown specifies dispatching 7 diagnosis molecules + exercise-proposal + 2 atoms = 10 tools registered; 8 turns is enough headroom for a sweep + a re-dispatch on tool errors. Can be tuned by env var if observed insufficient.
- **Q: How does the codegen script discover `docs/harness/skills/**/*.md` from inside `apps/api/`?** Default: relative path resolved from `apps/api/scripts/catalog-codegen.ts` to `../../../docs/harness/skills`. Fallback: env var `CRESCENDAI_SKILLS_DIR` for non-monorepo invocations. Plan 1 (loop) implements this.
- **Q: What signal-cache reader interface do atoms use?** Default: a typed reader injected via `AtomContext` reading from R2 (bucket binding `CHUNKS`) for AMT/MuQ blobs and from D1/Hyperdrive for piece score MIDI. Plan 2 (atoms) defines the reader interface; first atom that needs it provides the implementation.
- **Q: Does `synthesizeV6` build the conversation/components shape that the WebSocket message expects, or does the DO consumer do that translation?** Default: DO does the translation (consumes `artifact` event, maps `headline` → text, `proposed_exercises` → components). Adapter stays pure (event stream only).
