# Persisted-Session E2E UI Test Design

**Goal:** Drive one real audio recording through the full local pipeline (MuQ + AMT -> V6 synthesis on glm-4.7-flash@WorkersAI) as a non-eval session so the resulting conversation persists to Postgres and can be asserted against in the live web UI via Playwright.

**Not in scope:**
- Production deployment of any kind
- Changing the existing eval-mode `drive()` function or any code it consumes
- CI integration (the e2e requires live services; it is a manual/just gate only)
- Multiple recordings or parametric variation
- Modifying the DO, API routes, or web app business logic
- Playwright cloud/BrowserStack execution

## Problem

The current eval harness always runs in eval mode (`?eval=true&evalStudentId=...` + `x-eval-secret` header). Eval mode bypasses the persistence gate in session-brain.ts (`state.conversationId !== null`) because the WS URL never forwards `conversationId` to the DO. The V6 synthesis WS payload in non-eval mode also omits `eval_context`, so the existing `drive()` function — which reads `eval_context` — cannot be reused as a non-eval driver.

The consequence: there is no automated verification that the full pipeline path (non-eval session -> V6 synthesis -> DB persist -> web render) works end-to-end. Bugs in the persistence path or web render path are caught only by manual testing.

## Solution (from the user's perspective)

Running `just e2e-ui-session` starts the full local stack (prerequisite: `just dev` already running + `just seed-fingerprint` done), drives the nocturne recording through the real MuQ + AMT + V6 pipeline as a regular authenticated session, waits for synthesis, then opens a Playwright browser, navigates to the conversation that was just persisted, and asserts:

1. The synthesis headline text visible in the DOM matches the text received over WS.
2. Each WS component (score_highlight, play_passage, segment_loop, exercise_set) has a corresponding rendered card in the DOM.
3. If a `pending_exercise` component was present, the Confirm button is clicked, the ExerciseSetCard appears (assign-pending -> reveal flow verified).
4. A screenshot is saved to `apps/evals/results/e2e-ui-session-<timestamp>.png`.

Exit code 0 = PASS. Exit code 1 = FAIL with structured error message.

## Design

### Phase 1 — `drive_persisted()`

Three surgical changes to the driver WS URL and session start call, isolated in a new function that does not touch `drive()`:

**(a)** Capture `conversationId` from the `/api/practice/start` response (already present in the 201 JSON; `drive()` discards it).

**(b)** Append `?conversationId=<cid>` to the WS connect URL so the practice route (`apps/api/src/routes/practice.ts:178-181`) forwards it to the DO via `url.searchParams.set("conversationId", conversationId)`.

**(c)** Omit `?eval=true&evalStudentId=...` and the `x-eval-secret` header so the session is non-eval. The DO's persistence gate (`state.conversationId !== null`, session-brain.ts:1842 and :1927) then fires, writing the synthesis message row to Postgres.

The WS synthesis event in non-eval mode contains only `{type, text, components, isFallback}` — no `eval_context`. `drive_persisted` reads only these fields and raises `RuntimeError` if `isFallback !== False` or `conversationId` is missing.

**Captured fields:**
- `conversation_id` — from `/start` response
- `session_id` — from `/start` response
- `synthesis_text` — `event["text"]` from synthesis WS message
- `components` — `event["components"]` (list of InlineComponent dicts)
- `chunk_scores` — per-chunk `{dynamics, timing, pedaling, articulation, phrasing, interpretation}` accumulated from `chunk_processed` WS messages
- `piece_identification` — from `piece_identified` WS message if present

**Data class:** `PersistedSessionCapture` — separate from `SessionCapture` (which is the existing eval-mode type locked to `score.py`). Defined in `apps/evals/shared/local_session.py`.

### Phase 2 — UI Verifier

A Python Playwright script that:
1. Launches Chromium (headless by default; `--headed` flag for debug)
2. POSTs to `http://localhost:8787/api/auth/debug` within the browser context (not transplanting cookies from the Python requests session — the two contexts are independent, both authenticating as the same `debug@crescend.ai` user)
3. Navigates to `http://localhost:3000/app/c/<conversation_id>`
4. Waits for the synthesis message bubble to appear (polls for the headline text)
5. Asserts headline text equality, component card presence, and conditionally the confirm->reveal flow
6. Saves screenshot

**CORS risk (must be validated in Task 1):** The API CORS config allows `credentials: true` + origin `http://localhost:3000` (hardcoded in `apps/api/src/index.ts:28`). The better-auth cookie is set with no explicit `SameSite` override — on `http://localhost` Chromium treats the context as secure and allows `SameSite=Lax` cookies to be sent cross-origin for `credentials: include` fetches. If this fails (conversation 404 at navigation), the fallback is a dev-only login affordance in the web app or direct API navigation with explicit cookie injection. The plan calls this out explicitly as a build-time risk to validate in Task 4.

### Phase 3 — Orchestrator

Thin CLI at `apps/evals/e2e/ui_session.py`. Runs Phase 1 then Phase 2. Prints structured output. Exits nonzero on failure. Exposed via `just e2e-ui-session`.

### Unit test (offline, no live services)

A pytest test at `apps/evals/e2e/tests/test_drive_persisted_capture.py` that mocks the WS connection and asserts `drive_persisted` parses synthesis events and accumulates chunk scores correctly. Uses `unittest.mock` to replace `websockets.connect`. No service calls.

## Modules

### `drive_persisted()` in `apps/evals/shared/local_session.py`

**Interface:**
```python
def drive_persisted(
    recording: Path,
    piece_slug: str,
    wrangler_url: str = "http://localhost:8787",
    api_dir: Path = DEFAULT_API_DIR,
    timeout_per_event: float = 180.0,
    max_chunks: int = 6,
) -> PersistedSessionCapture: ...
```

**Hides:** debug auth, `/start` POST + conversationId capture, ffmpeg chunking, local R2 upload via wrangler, WS connect (non-eval URL with `?conversationId=...`), chunk_processed score accumulation, piece_identified capture, synthesis event parsing, `isFallback` guard, explicit raises on missing conversationId.

**Depth verdict:** DEEP — the interface is 6 params in, one typed dataclass out. The implementation hides ~150 lines of plumbing across 6 sub-operations.

**Tested through:** `PersistedSessionCapture` fields after mocked WS exchange (offline unit test).

---

### `PersistedSessionCapture` dataclass in `apps/evals/shared/local_session.py`

**Interface:** `@dataclass` with fields: `conversation_id: str`, `session_id: str`, `synthesis_text: str`, `components: list[dict]`, `chunk_scores: list[dict]`, `piece_identification: dict | None`.

**Hides:** nothing (pure data).

**Depth verdict:** SHALLOW (intentional — pure data carrier, no logic).

---

### `UIVerifier` in `apps/evals/e2e/ui_verifier.py`

**Interface:**
```python
@dataclass
class UIAssertionResult:
    passed: bool
    headline_matched: bool
    component_count_matched: bool
    confirm_flow_ran: bool
    confirm_flow_passed: bool | None
    screenshot_path: str
    error: str | None

def verify_session_ui(
    capture: PersistedSessionCapture,
    web_url: str = "http://localhost:3000",
    api_url: str = "http://localhost:8787",
    headed: bool = False,
    screenshot_dir: Path = ...,
) -> UIAssertionResult: ...
```

**Hides:** Playwright browser launch, context creation, in-browser auth via `api.post("/api/auth/debug")`, navigation to `/app/c/<conversation_id>`, DOM wait strategy, text extraction, component card counting, button click sequence, screenshot saving.

**Depth verdict:** DEEP — simple function signature hides Playwright lifecycle, DOM selectors, wait loops, screenshot IO.

**Tested through:** full e2e run (live stack required); no offline unit test (Playwright requires a browser).

---

### `apps/evals/e2e/ui_session.py` (orchestrator)

**Interface:** CLI entry point — `python -m e2e.ui_session [--headed] [--recording PATH] [--piece-slug SLUG]`.

**Hides:** argument parsing, service health check, Phase 1 -> Phase 2 sequencing, exit code.

**Depth verdict:** SHALLOW (intentional — thin glue; all depth is in the two deep modules it calls).

## Verification Architecture

**Canonical success state:** `UIAssertionResult.passed == True` with `headline_matched=True`, `component_count_matched=True`, and `confirm_flow_passed=True` (when prescription present).

**Automated check:** `just e2e-ui-session` exits 0.

**Harness:** No Task Group 0 harness is buildable before the feature — the feature IS the harness. Instead, Task 1 is a service-reachability smoke test baked into the orchestrator that fast-fails with a human-readable error if prerequisites are not met (`just dev`, `just seed-fingerprint`).

**Offline unit test:** Task 2 — mocked WS exchange verifies `drive_persisted` capture-parsing without live services.

**Live stack tests (require `just dev`):** Tasks 3 (drive_persisted live run), 4 (CORS validation), 5 (UI verifier), 6 (confirm flow).

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/evals/shared/local_session.py` | Add `PersistedSessionCapture` dataclass + `drive_persisted()` + `_drive_persisted_async()` + export in `__all__` | Modify |
| `apps/evals/e2e/__init__.py` | New package init | New |
| `apps/evals/e2e/ui_verifier.py` | `UIAssertionResult` + `verify_session_ui()` | New |
| `apps/evals/e2e/ui_session.py` | CLI orchestrator | New |
| `apps/evals/e2e/tests/__init__.py` | New package init | New |
| `apps/evals/e2e/tests/test_drive_persisted_capture.py` | Offline unit test with mocked WS | New |
| `justfile` | Add `e2e-ui-session` recipe | Modify |

**No TS changes** unless build discovers stable Playwright selectors require `data-testid` additions to ReflectionMessage.tsx or InlineCard cards (surgical, additive only, noted as conditional in Task 5).

## Open Questions

- Q: Does the nocturne recording reliably produce a `pending_exercise` component on the first run with glm-4.7-flash@WorkersAI?
  Default: Build Task 3 empirically confirms this. If no prescription, confirm flow is skipped gracefully and `confirm_flow_ran=False` is not a failure.

- Q: Do ReflectionMessage and InlineCard cards have stable DOM selectors (text content, roles, or existing classes) sufficient for Playwright without adding `data-testid`?
  Default: Use text-based selectors where stable. Add `data-testid` hooks surgically if needed (additive, no behavioral change).
