# Visible Full Local Eval Design

**Goal:** A single `just e2e-full-session` command drives one real session end-to-end, opens a headed Chromium window so a human can watch everything happen, seeds canary memory facts and visually confirms the teacher recalled them in chat, then prints a per-criterion PASS/FAIL report.

**Not in scope:**
- Deploying anything to production (local merge only).
- Replacing `just e2e-ui-session` (that recipe stays intact, headless by default).
- Asserting the exact text of LLM chat replies (LLM output is non-deterministic; only token presence is checked).
- Multi-session or multi-recording scenarios.
- Modifying the API's write path for synthesized_facts (DB seeded directly for test isolation).

## Problem

After issue #68 and #69 shipped, there is no single operator command that:
1. Seeds memory context and confirms the teacher reads it.
2. Drives a full audio session through the real pipeline (MuQ, AMT, V6 synthesis on glm@WorkersAI).
3. Opens a visible browser so a human can confirm UI rendering without inspecting logs.
4. Exercises the tool-call path (corpus drill) live and reports whether glm actually streamed tool calls.
5. Prints one tidy verdict per criterion rather than requiring the operator to grep logs.

The #68 e2e harness is headless by default and has no chat-turn scripting; #69 verified chat in isolation. There is no unified "show me everything working at once" command.

## Solution (from the user's perspective)

`just e2e-full-session` (or `just e2e-full-session --no-session` to skip the recording phase):

1. Seeds 2 distinctive canary facts into `synthesized_facts` for the debug user (`debug@crescend.ai`).
2. Drives the Nocturne recording through MuQ + AMT + V6 synthesis (reusing `drive_persisted()`).
3. Opens a headed Chrome window with `slow_mo=700ms`. A human can watch the synthesis appear.
4. Asserts the V6 synthesis message renders in the DOM (existing #68 criteria a/b).
5. Types "what do you know about me?" into the chat input, waits (bounded 45s) for the streamed reply, asserts the reply text contains the two canary tokens (case-insensitive).
6. Types "give me a left-hand drill for bars 1-4" into chat, waits (bounded 45s) for a reply, checks whether `[data-testid=exercise-set-card]` appeared (non-fatal; reports TOOL_RENDERED vs TEXT_ONLY).
7. If the session had a prescription, confirms the exercise card from the synthesis (existing #68 criterion c).
8. Prints a table:

```
(a) V6 artifact rendered           PASS
(b) Headline match                 PASS
(c) Exercise confirm flow          SKIP (no prescription)
(e) Memory recall — canary tokens  PASS
(f) Tool action — ExerciseSetCard  TEXT_ONLY (non-fatal)
OVERALL: PASS
```

## Design

### Key decisions

**Headed by default, headless via flag.** The feature's purpose is visual human confirmation. `--no-headless` is not an add-on; it is the default for the new orchestrator. CI can pass `--headless`.

**Bounded waits everywhere.** The prior #68 build wedged for ~14 hours on an unbounded SSE/wait. Every `page.wait_for_selector`, `expect`, and `wait_for_function` call in this build carries an explicit `timeout_ms` argument. The chat-reply wait is max 45 s; page nav is max 15 s; synthesis-message is max 30 s.

**Additive `data-testid="assistant-message"` on the regular assistant bubble.** The existing test IDs (`synthesis-message`, `synthesis-headline`) are in place. Regular chat replies (`role=assistant`, `messageType != "synthesis"`) have no testid today. Adding one is the minimal surgical change enabling Playwright to reliably detect when a new chat reply has finished streaming. The change is purely additive — no behavior is changed.

**canary tokens, not exact reply match.** LLM output is non-deterministic. The test seeds facts with unique token strings (e.g., `"CANARY_RACHMANINOFF_ETUDE"`, `"CANARY_LEFT_HAND_WEAKNESS"`) and asserts those tokens appear somewhere in the reply. This avoids brittle exact-match assertions while still proving the memory context was injected.

**DB seed via psycopg2 + direct INSERT.** `synthesized_facts` has no REST write path. The seeder connects to `crescendai_dev` with the local DSN, INSERTs rows, and returns the `student_id` that was seeded. The seeder also deletes previous canary rows by a prefix check on `fact_text` to keep the DB clean across repeated runs.

**Tool-action outcome is NON-FATAL.** Whether glm emits a `tool_result` (ExerciseSetCard in DOM) or plain text is the open empirical question from #69. The harness reports which happened but does not fail if it is TEXT_ONLY.

**Reuse, do not rebuild.** `drive_persisted()`, `verify_ui()`, `_auth_browser()`, `check_services()`, `_lowest_dim()` are all called unchanged. The new orchestrator is a thin layer on top. `ui_verifier.py` gains one new exported function (`run_chat_turns`) alongside `verify_ui`. `e2e_full_session.py` is a new file parallel to `e2e_ui_session.py`.

**`just e2e-full-session` is a separate recipe.** The existing `just e2e-ui-session` recipe is not touched.

### Module decomposition

Three new or modified modules:

1. `apps/evals/memory_seeder.py` — seeds and cleans canary facts (deep module; hides psycopg2 connection management, SQL, and cleanup).
2. `apps/evals/ui_verifier.py` — gains `run_chat_turns(page, turns, timeout_ms)` (additive; hides Playwright fill/keypress/wait logic behind one call per turn).
3. `apps/evals/e2e_full_session.py` — top-level orchestrator; calls seeder, drive_persisted, verify_ui, run_chat_turns, prints report.

Web change: `apps/web/src/components/ChatMessages.tsx` — add `data-testid="assistant-message"` to the assistant bubble `<div>` (non-synthesis, non-observation path).

## Modules

### `memory_seeder.py`

- **Interface:** `seed_canary_facts(student_id, db_dsn, wrangler_url) -> CanarySeed`; `cleanup_canary_facts(student_id, db_dsn)`. `CanarySeed` is a dataclass with `student_id: str`, `tokens: list[str]`.
- **Hides:** psycopg2 connection lifecycle; the INSERT SQL; the `CANARY_` prefix convention; DELETE for previous canary rows; looking up `student_id` from `/api/auth/debug` if not provided.
- **Tested through:** `seed_canary_facts` returns a `CanarySeed` with `tokens` list non-empty, and `build_insert_rows` helper produces correctly-shaped dicts (unit-testable without DB). Full seed path is live-gated.

### `ui_verifier.run_chat_turns` (addition to `ui_verifier.py`)

- **Interface:** `run_chat_turns(page, turns, reply_timeout_ms) -> list[ChatTurnResult]`. `turns: list[str]` — the message texts to send. `ChatTurnResult`: `turn_text: str`, `reply_text: str`, `elapsed_ms: int`.
- **Hides:** Playwright `fill` on `textarea`, `press("Enter")`, counting assistant-message elements to detect a new reply finished streaming (checks that `[data-testid=assistant-message]:not(.streaming)` count incremented and text is stable for 500ms), screenshot on timeout.
- **Tested through:** The function contract (argument types, return shape) is verified by a unit test using a minimal mock page; live behavior is verified in the full e2e run.

### `e2e_full_session.py`

- **Interface:** `run(...)` returns `int` (0 = PASS, 1 = FAIL); `_cli()` parses args and calls `run()`.
- **Hides:** orchestration ordering (seed → drive → verify_ui → run_chat_turns → report); criterion mapping; screenshot paths.
- **Tested through:** The per-criterion report struct `FullSessionReport` is a pure dataclass testable offline; the `format_report` function is unit-testable.

### `ChatMessages.tsx` (additive change)

- **Interface:** The assistant bubble `<div>` gains `data-testid="assistant-message"` on the non-synthesis, non-session-lifecycle branch (line 228 in current file, the final `return` in `MessageBubble`).
- **Hides:** Nothing new — this is a rendering attribute addition.
- **Tested through:** The existing `ChatMessages.test.tsx` is extended with one render test asserting an assistant-role non-synthesis message produces a `[data-testid=assistant-message]` element.

## Verification Architecture

- **Canonical success state:** `just e2e-full-session` exits 0; a headed Chromium window opened; the printed report shows PASS for (a), (b), (e); (c) shows either PASS or SKIP; (f) shows TOOL_RENDERED or TEXT_ONLY (non-fatal). The two canary tokens appear in the rendered chat reply visible on screen.
- **Automated offline check:** `bun run test` in `apps/web` covers the `assistant-message` testid; `uv run pytest apps/evals/tests/test_memory_seeder.py` covers the SQL builder and canary token logic offline.
- **Live gate:** `just e2e-full-session` with the full stack running. No live DB or browser automation is exercised by the offline tests.
- **Harness:** No Task Group 0 harness is buildable before the feature — the verifier IS the feature. Manual live gate after Task Group C (orchestration) is complete.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/evals/memory_seeder.py` | New module: seed/cleanup canary facts in synthesized_facts | New |
| `apps/evals/tests/test_memory_seeder.py` | Offline unit tests for SQL builder + canary token logic | New |
| `apps/evals/ui_verifier.py` | Add `run_chat_turns()`, `ChatTurnResult` dataclass | Modify |
| `apps/evals/e2e_full_session.py` | New orchestrator: full session + chat turns + report | New |
| `apps/web/src/components/ChatMessages.tsx` | Add `data-testid="assistant-message"` to assistant bubble | Modify |
| `apps/web/src/components/ChatMessages.test.tsx` | Add render test for `assistant-message` testid | Modify |
| `justfile` | Add `e2e-full-session` recipe | Modify |

## Open Questions

- Q: Does the debug user's `student_id` change across `wrangler dev` restarts?
  Default: No — `debug@crescend.ai` is a stable seed user; the UUID is stable in `crescendai_dev`. The seeder fetches it fresh via `POST /api/auth/debug` at runtime rather than hardcoding.
- Q: Should `e2e-full-session` accept `--no-session` to skip the recording phase and use an existing `conversation_id`?
  Default: Yes — this cuts the iteration loop for UI/chat-only re-runs. The `--conversation-id` flag short-circuits `drive_persisted()`.
