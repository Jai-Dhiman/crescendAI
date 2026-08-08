# Practice Mode: The Digital Music Stand — Design

**Goal:** Replace the chat-first recording overlay (`ListeningMode` +
`AudioWaveformRing`) with a full-screen practice surface — a static,
manually-paged score (or a minimal pieceless timeline) that receives pause
marks silently and soft-auto-stops after a minute of silence.

**Not in scope:**
- The epic-level "Home" surface (repertoire cards, add-piece flow). AppChat's
  existing landing screen is touched only to remove `GREETINGS`; it is not
  redesigned.
- Session review (verdict, consolidated marks, carry-forward) — #159/#162.
- Removing the chat shell, sidebar, or conversation history — #164.
- Passage-scoped ask threads — a later sub-issue.
- The backend mark-generation pipeline (offline alignment, MPM features,
  MoonBeam scoring, the bandwidth/persistence gate). #163 shipped the gate's
  *logic* (`apps/api/src/services/student-baseline.ts`); nothing in the
  running system currently calls it during a live session, and no WebSocket
  event carries a mark from server to client. This plan adds the client-side
  event contract and UI; wiring the server to actually emit `mark` events is
  out of scope (tracked implicitly by the epic's remaining sub-issues).
- Drills, "work on this," passage playback.

## Problem

`apps/web/src/components/ListeningMode.tsx` is the current full-screen
recording surface: it centers `AudioWaveformRing` (a decorative frequency
visualizer), a "Now practicing" piece-name editor, a notepad drawer, and a
metronome popover. It has no score, no pause-driven feedback, and no
auto-stop. `AppChat.tsx`'s empty landing state shows a random line from a
`GREETINGS` array above the chat input. Both are artifacts of the chat-first
design the epic (#154) replaces.

Separately, `#157` shipped the mark system's two rendering canvases
(`ScoreMarkLayer`, `SessionTimelineStrip`) and their pure placement helpers,
but the only place they are mounted is `src/routes/marks-preview.tsx` — a
throwaway route that imports `src/test-utils/mark-fixtures.ts` directly into
the production route tree, shipping test fixtures in the app bundle. It was
accepted only because this issue was scheduled to replace it.

There is currently no client-side concept of "the student stopped playing
for N seconds" beyond the existing 5-second onset/offset debounce in
`useAudioActivity` (`OFFSET_MS = 5000`), which exists to gate audio chunk
uploads, not to drive UI. There is no piece-resolution ladder UI, no score
pagination UI (the existing `ScorePanel.tsx` / `ScoreCursor` combination is
a live-following side panel built for the old chat flow and is not reused
here — `ScoreCursor` in particular drives a moving cursor, which this design
explicitly forbids), and no `mark` WebSocket event type.

## Solution (from the user's perspective)

Tapping record now opens a full-screen music stand instead of a waveform
ring:

- **Score known** (user picked it, or `piece_identified` arrived above the
  ladder's confidence bar and hasn't been dismissed): the score fills the
  screen, paged manually with Prev/Next controls. A slim margin carries a
  recording indicator (dot + elapsed time) and a metronome toggle. Nothing
  moves on its own.
- **Score not known** (no pick, no confident guess, or the guess was
  dismissed): a calm, near-empty screen shows elapsed time, the metronome,
  and a thin timeline strip along the bottom.
- If a confident-but-unpicked guess exists, a dismissible chip names it over
  whichever of the two screens is showing ("Looks like *Nocturne Op. 9 No.
  2* — is that right?" / dismiss).
- The screen stays silent while the student plays. After >=20s of silence,
  at most one mark can land (on the score page if its bar is visible, always
  on the timeline strip). After 60s of silence, the screen softens to "Session
  ended — keep playing?" with a one-tap resume that dismisses the banner and
  changes nothing else — the session, the WebSocket, and the recording never
  stopped.
- `GREETINGS` and the random headline on the chat landing screen are gone;
  the empty state shows the icon and the input, nothing else.

## Design

### Why the auto-stop is UI-only

The chunk-gating state machine in `usePracticeSession` already stops
uploading audio during silence (`ChunkGateState: "waiting"`) and only starts
uploading again on the next onset. "Auto-stop" therefore does not need to
touch the recording session, the `MediaRecorder`, or the WebSocket at all —
tearing any of those down and back up on a soft, resumable pause would add a
reconnect race for no benefit. The 60s state is a banner layered on top of an
otherwise-unaffected live session; "resume" only dismisses the banner and
resets the pause clock. This is also why the failure-mode principle ("auto-stop
is soft, one-tap resume") is easy to honor: there is no teardown to reverse.

### Pause tracking is a pure state machine over one boolean

`useAudioActivity` already exposes `isPlaying: boolean`. A new pure function,
`computePauseState`, takes `{ isPlaying, silenceStartedAt, now, markThresholdMs,
autoStopThresholdMs }` and returns `{ silenceMs, canShowMark, autoStopped }`.
A thin hook (`usePauseTracker`) owns the `silenceStartedAt` ref, a 1s
interval to re-derive `now`, and calls the pure function every tick. Keeping
the arithmetic in a pure function (no DOM, no timers) makes the 20s/60s
boundary conditions unit-testable without fake browser timers driving a whole
hook.

### The piece ladder is a pure decision, not a component

`resolvePieceLadderState({ userPicked, confidentGuess, dismissed })` returns
one of `"user-picked" | "confirm-chip" | "pieceless"`. `PracticeMode` calls it
on every render; it owns no state itself beyond `dismissed`, which flips once
per session and is never un-set (matching "dismissible" — there is no way to
re-summon a dismissed chip mid-session, mirroring how #157 forbids inventing
a fallback state).

### Score stand reuses `scoreRenderer`, not `ScorePanel`/`ScoreCursor`

`ScorePanel.tsx` and `ScoreCursor` exist to drive a moving highlight in a
side panel during chat-era assistant messages; `ScoreCursor` in particular
computes cursor overlays from a live `qstampSource`, i.e., a following
cursor — exactly what "no live following" forbids. The new `ScoreStand`
component calls `scoreRenderer.load(pieceId)` and `scoreRenderer.getPage(pieceId,
pageN)` directly (the same two calls `marks-preview.tsx` already made) and
manages `currentPage` as local `useState`, clamped to `ir.pages.length`. It
mounts `ScoreMarkLayer` with only the bars whose `pageN` equals
`currentPage`, matching `ScoreMarkLayer`'s existing lossy-by-design contract.

### Marks arrive over a new WS event; the client renders whatever arrives, including nothing

`PracticeWsEvent` gains one more variant, `{ type: "mark"; mark: Mark }` (the
exact `Mark` shape from `mark.ts`, since the server is the sole producer of
lifecycle/taxonomy/dimension and the client must not derive any of it).
`usePracticeSession` appends to a new `marks: Mark[]` state array on receipt
and exposes it. No server code in this repo currently sends this event —
`grep` across `apps/api/src` for a mark-shaped WS message found none — so in
the running system `marks` stays empty for the lifetime of this plan, and the
screen correctly shows nothing, which is the specified failure mode
("the system may fall silent, never guess") rather than a bug.

### Verifying "a mark appears" without a server that sends one

Two independent things need verifying, and neither needs a live mark
pipeline:
1. **Wiring**: does `usePracticeSession` correctly append a `Mark` to state
   when a `mark` WS event arrives, and does `PracticeMode` render it? Answered
   by a unit test against `usePracticeSession` with a fake `WebSocket` (the
   existing pattern for this hook already exists for other event types) and a
   component test mounting `PracticeMode` with a `marks` prop.
2. **Geometry**: does a mark that resolves to a bar land on that bar, stay
   inside the score container, and stay clickable — the two properties #157
   proved can only be checked in a real browser? Answered by porting
   `tests/marks.spec.ts` from `/marks-preview` to a new dev-only harness (see
   below) that mounts the real `ScoreStand` and pieceless-mode components
   (not bespoke test markup) with the same fixture marks #157 already wrote.

`PracticeMode`, `ScoreStand`, and the pieceless-mode component all take
`marks` as a prop rather than reading `usePracticeSession` directly — the
same dependency-inversion `ScoreMarkLayer`/`SessionTimelineStrip` already
use. `AppChat` is the only place that wires the live hook's `marks` array to
the prop.

### The replacement harness does not repeat #157's bundle leak

`/marks-preview` was a TanStack Router file route, so it was compiled into
the same bundle as every production route — that is precisely the defect
this issue is charged with fixing, not relocating. The replacement harness,
`src/routes/practice-preview.tsx`, renders `null` unless `import.meta.env.DEV`
is true. Vite statically replaces `import.meta.env.DEV` with `false` in a
production build, and Rollup's dead-code elimination drops the unreachable
branch — including the fixture import, which has no other consumer — from
the production bundle. This only verifies anything if the Playwright run that
exercises it serves a dev build: `playwright.marks.config.ts`'s `webServer`
command changes from `bun run build && vite preview` to `vite dev`, keeping
`import.meta.env.DEV` true for the run while a real production build never
registers the route. The page is still Chromium-rendered by Playwright either
way, so the real-layout guarantee the harness exists for is unaffected.

### `ListeningMode` and `AudioWaveformRing` are deleted outright, not folded in

Their remaining unique behavior — the piece-name inline editor, the notepad
drawer — belongs to the chat-era flow this epic is retiring (notepad is not
named anywhere in #158's goal or the epic's UI-system doc) and is dropped
rather than ported. The metronome control (`useMetronome` + its popover
markup) is the one piece of `ListeningMode` genuinely reused; it is lifted
into `ScoreStand` and the pieceless-mode component directly, both calling
`useMetronome()` independently (it is a self-contained hook with its own
`AudioContext`, so two independent instances across the two branches of the
ladder is correct — only one branch is ever mounted at a time).

## Modules

### `src/lib/pause-state.ts`
- **Interface:** `computePauseState(input: PauseStateInput): PauseState`;
  `MARK_SILENCE_MS = 20_000` (tunable per the epic's open question, exported
  as a named constant so a later config surface has one place to override
  it); `AUTO_STOP_SILENCE_MS = 60_000`.
- **Hides:** the boundary arithmetic (`>=` vs `>`, what happens at exactly
  the threshold, what happens if `isPlaying` flips back to `true` mid-tick).
- **Tested through:** direct calls with synthetic `{isPlaying, silenceStartedAt,
  now}` triples — no timers, no DOM.

### `src/hooks/usePauseTracker.ts`
- **Interface:** `usePauseTracker(isPlaying: boolean): { silenceMs: number;
  canShowMark: boolean; autoStopped: boolean; resume: () => void }`.
- **Hides:** the `setInterval` driving `now`, the `silenceStartedAt` ref, and
  calling `computePauseState` every tick.
- **Tested through:** a component that renders the hook's return value as
  text, driven with `vi.useFakeTimers()` and prop changes to `isPlaying`.

### `src/lib/piece-ladder.ts`
- **Interface:** `resolvePieceLadderState(input: {userPicked: string | null;
  confidentGuess: ConfidentGuess | null; dismissed: boolean}):
  "user-picked" | "confirm-chip" | "pieceless"`; `type ConfidentGuess =
  {pieceId: string; composer: string; title: string; confidence: number}`.
- **Hides:** the precedence order (user pick beats a guess; a dismissed guess
  never resurfaces).
- **Tested through:** direct calls with the four input combinations that
  matter (pick present; guess present and not dismissed; guess present and
  dismissed; neither).

### `src/components/ScoreStand.tsx`
- **Interface:** `<ScoreStand pieceId={string} marks={readonly Mark[]}
  elapsedSeconds={number} isRecording={boolean} />`.
- **Hides:** `scoreRenderer` calls, page-load/error state, `currentPage`
  clamping, which bars belong to the current page, mounting
  `ScoreMarkLayer`.
- **Tested through:** an integration test (jsdom, no layout assertions) that
  Prev/Next changes rendered page content and clamps at the ends; the real
  positional behavior is covered by the geometry harness below, not here.

### `src/components/PieceLessMode.tsx`
- **Interface:** `<PieceLessMode marks={readonly Mark[]} durationSeconds={number}
  elapsedSeconds={number} isRecording={boolean} />`.
- **Hides:** nothing beyond composing `SessionTimelineStrip` with the elapsed
  timer and metronome — intentionally the shallowest module in the set,
  because it has no logic of its own to hide.
- **Tested through:** renders `SessionTimelineStrip` with the marks it was
  given (a snapshot-shaped assertion, not new logic).

### `src/components/ConfirmPieceChip.tsx`
- **Interface:** `<ConfirmPieceChip guess={ConfidentGuess} onDismiss={() =>
  void} />`.
- **Hides:** label formatting ("Looks like ... — is that right?").
- **Tested through:** render + click dismiss calls `onDismiss` once.

### `src/components/SessionEndedBanner.tsx`
- **Interface:** `<SessionEndedBanner onResume={() => void} />`.
- **Hides:** the copy and layout for the soft-stop state.
- **Tested through:** render + click resume calls `onResume` once.

### `src/components/PracticeMode.tsx`
- **Interface:** `<PracticeMode userPickedPieceId={string | null}
  confidentGuess={ConfidentGuess | null} marks={readonly Mark[]}
  elapsedSeconds={number} isPlaying={boolean} isRecording={boolean} />`.
- **Hides:** calling `resolvePieceLadderState` and `usePauseTracker`, and
  switching between `ScoreStand`, `PieceLessMode`, `ConfirmPieceChip`, and
  `SessionEndedBanner` based on their outputs. This is the orchestrator —
  intentionally the one place that knows all four sub-components exist.
- **Tested through:** an integration test that drives `userPickedPieceId` /
  `confidentGuess` / fake-timer silence and asserts which sub-surface is
  showing at each combination.

### `src/routes/practice-preview.tsx` (dev-only harness, replaces `marks-preview.tsx`)
- **Interface:** none (a route component, mounted only by the router).
- **Hides:** the `import.meta.env.DEV` gate and fixture wiring for the real
  `ScoreStand` + `PieceLessMode` components.
- **Tested through:** `tests/marks.spec.ts` (real-browser Playwright), the
  successor to the current file of the same name.

## Verification Architecture

- **Canonical success state:** (1) unit/integration suite green — pause
  threshold math, piece-ladder precedence, WS `mark` event plumbing, ladder
  branching; (2) `tests/marks.spec.ts` green against a real Chromium page
  showing the actual `ScoreStand`/`PieceLessMode` production components with
  real Verovio SVG, proving containment (`documentElement.scrollWidth ===
  clientWidth`) and no-overlap for both canvases; (3) a manual click-through
  per the issue's literal success criterion: record with a picked piece
  (score stand renders, page-turns work, injected marks land on the right
  bar and on the timeline), record pieceless (timeline strip accrues
  injected marks), and auto-stop-then-resume (banner appears at a
  shortened test threshold, resume dismisses it, elapsed time and WS
  connection are unaffected).
- **Automated check:** `bun run test`, `bunx tsc --noEmit`, `bun run lint`,
  `bun run test:a11y`, `bun run test:marks` — all run from
  `/Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web`
  with an explicit `cd`.
- **Harness:** buildable before the feature tasks — Task Group 0 in the plan
  ports `tests/marks.spec.ts` to point at `practice-preview.tsx` instead of
  `marks-preview.tsx` (still red until `ScoreStand`/`PieceLessMode` exist),
  and switches `playwright.marks.config.ts`'s `webServer` to `vite dev`.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/web/src/lib/pause-state.ts` | Pure pause/auto-stop threshold math | New |
| `apps/web/src/lib/pause-state.test.ts` | Unit tests for the above | New |
| `apps/web/src/hooks/usePauseTracker.ts` | Hook wrapping `pause-state.ts` with a live clock | New |
| `apps/web/src/hooks/usePauseTracker.test.ts` | Fake-timer tests | New |
| `apps/web/src/lib/piece-ladder.ts` | Pure ladder-precedence decision | New |
| `apps/web/src/lib/piece-ladder.test.ts` | Unit tests | New |
| `apps/web/src/components/ScoreStand.tsx` | Full-screen static paginated score + margin | New |
| `apps/web/src/components/ScoreStand.test.tsx` | Integration test (no layout assertions) | New |
| `apps/web/src/components/PieceLessMode.tsx` | Minimal elapsed/metronome/timeline screen | New |
| `apps/web/src/components/PieceLessMode.test.tsx` | Integration test | New |
| `apps/web/src/components/ConfirmPieceChip.tsx` | Dismissible piece-guess chip | New |
| `apps/web/src/components/ConfirmPieceChip.test.tsx` | Behavior test | New |
| `apps/web/src/components/SessionEndedBanner.tsx` | Soft auto-stop banner | New |
| `apps/web/src/components/SessionEndedBanner.test.tsx` | Behavior test | New |
| `apps/web/src/components/PracticeMode.tsx` | Orchestrator: ladder + pause tracker + sub-surfaces | New |
| `apps/web/src/components/PracticeMode.test.tsx` | Integration test | New |
| `apps/web/src/routes/practice-preview.tsx` | Dev-only real-browser harness | New |
| `apps/web/tests/marks.spec.ts` | Point at `/practice-preview`, mount real components | Modify |
| `apps/web/playwright.marks.config.ts` | `webServer` command -> `vite dev` | Modify |
| `apps/web/src/lib/practice-api.ts` | Add `{ type: "mark"; mark: Mark }` to `PracticeWsEvent` | Modify |
| `apps/web/src/hooks/usePracticeSession.ts` | Accumulate `marks: Mark[]`, handle `mark` event, expose in return type | Modify |
| `apps/web/src/hooks/usePracticeSession.test.ts` | Test for the new event (existing file — extend) | Modify |
| `apps/web/src/components/AppChat.tsx` | Remove `GREETINGS`/`greeting`; mount `PracticeMode` instead of `ListeningMode` on record | Modify |
| `apps/web/src/components/ListeningMode.tsx` | Deleted | Delete |
| `apps/web/src/components/AudioWaveformRing.tsx` | Deleted | Delete |
| `apps/web/src/routes/marks-preview.tsx` | Deleted (replaced by `practice-preview.tsx`) | Delete |
| `apps/web/src/routes/marks-preview.test.tsx` | Deleted | Delete |
| `apps/web/src/test-utils/mark-fixtures.ts` | Deleted; fixture data inlined into `practice-preview.tsx` | Delete |

## Open Questions

- Q: Should the 20s mark / 60s auto-stop thresholds be user-configurable in
  this issue, or hardcoded constants?
  Default: hardcoded named constants in `pause-state.ts`
  (`MARK_SILENCE_MS`, `AUTO_STOP_SILENCE_MS`), matching the epic doc's "20s is
  a starting value... config-tunable, not a hard commit" — tunable in code,
  not exposed as a user setting yet.
- Q: When the server eventually emits `mark` events, will the WS message
  carry the full `Mark` shape verbatim, or a server-specific DTO the client
  translates?
  Default: verbatim `Mark` (server already owns `BaselineLifecycle` per
  `mark.ts`'s comment referencing `apps/api/src/services/student-baseline.ts`),
  since no server code exists yet to constrain the choice — the client-side
  event type can be tightened without UI change if the server's eventual
  shape differs.
- Q: Does the recording indicator need a `prefers-reduced-motion` variant
  (e.g. a pulsing dot)?
  Default: static dot + text, no animation, consistent with the "serious,
  adult, restrained" visual direction — avoids the question entirely.
