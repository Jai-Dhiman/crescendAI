# Session Review Design

**Goal:** On stop, the practice canvas (#158's `PracticeMode`) transitions in
place into a review state: one qualitative verdict sentence, every session
mark consolidated on the same score/timeline canvas, per-mark expansion that
plays the student's own recorded audio for that mark's bars with a tracking
cursor, evidence in words, a "work on this" drill entry, and exactly one
carry-forward — with every failure state (synthesis failure, WS gaps,
no-comment zones) disclosed loudly instead of silently degrading.

**Not in scope:**
- Backend review synthesis (verdict/carry-forward generation) — that is
  #162. This plan only builds the client contract and UI that consumes it.
- Backend "work on this" drill generation — "work on this" fires a callback
  with the mark; the server-side segment-loop/exercise-generation wiring
  behind that callback is out of scope, the same way #158 deferred
  `user-picked`/`confirm-chip` wiring to #160.
- "Ask about this" passage-scoped conversation threads — that is #161.
- Wiring `AppChat.tsx` to feed a live `sessionId`/verdict/carry-forward into
  review end-to-end. `AppChat` continues to hide `PracticeMode` immediately
  on stop (current behavior); this plan adds the `phase`/review props to
  `PracticeMode` and proves them through the `/practice-preview` harness,
  exactly as #158 did for the ladder rungs `AppChat` doesn't yet reach.
  Wiring `AppChat` for real is blocked on #162 shipping a verdict/marks/
  carry-forward payload to wire in.
- Real end-to-end own-audio playback of a genuinely recorded passage. That
  needs a real `sessionId` with uploaded R2 chunks, which only a live
  Record session produces. The manual click-through in this plan's
  Verification Architecture uses `/practice-preview` with mocked manifest
  and clip data (mirroring `PlayPassageCard.test.tsx`'s existing mocking
  pattern) — real-audio E2E is gated on #162 the same way.

## Problem

`docs/apps/05-ui-system.md` "Session review" and `01-product-vision.md`'s
"Ideal Practice Session" step 4 both specify session review as a first-class
surface, and `01-product-vision.md`'s status table marks it "NOT STARTED."
Today, stopping a practice session (`AppChat.tsx:263` `handleStopPracticeMode`)
calls `practice.stop()` then `handleExitPracticeMode()` immediately —
`PracticeMode` unmounts, and whatever the WS `synthesis`/`session_summary`
event returns is dumped as prose into the (soon-to-be-removed, #164) chat
message stream. There is no review surface, no consolidated marks display, no
own-audio playback outside a chat card, no carry-forward, and no disclosed
failure states.

Two pieces of machinery already exist but are stranded in the chat surface
that is being removed:
- `PlayPassageCard.tsx` fetches a `PassageManifest` via
  `api.sessions.getPassage(sessionId, bars)` and plays it with
  `PassagePlayer` (`apps/web/src/lib/passage-player.ts`) — but renders no
  cursor, and only exists inside `InlineCard`/`Artifact`'s chat-card system.
- `ExerciseSetCard.tsx` demonstrates cursor-tracked clip playback
  (`ScoreCursor` driven by a `qstampSource()`), but its `qstampSource` comes
  from `useLoopPlayer`'s synthesized-MIDI player, not from real recorded
  audio — there is no existing bridge from `PassagePlayer`'s real-time
  playback clock to a `ScoreCursor`-compatible qstamp.

## Solution (from the user's perspective)

The student taps Stop. The score (or timeline, if pieceless) they were just
reading does not disappear — it becomes the review. Above it: one sentence
("the bass blur from Tuesday is gone" or, if the review couldn't be produced,
an explicit "Couldn't put together a review this time — try again" with a
retry button, never a blank space). The marks from the whole session are all
visible on the canvas at once (not one-at-a-time like during play). Tapping
any mark expands it: the evidence sentence, a small clip of the score for
those bars, a Play button that plays back the student's own recording of
exactly that passage with a moving cursor, and a "Work on this" button. Below
the canvas: one carry-forward line for next time. If the connection dropped
for stretches during the session, or if some passage doesn't have grounded
commentary, that is stated plainly, not hidden.

## Design

### Key decisions

1. **Reuse `ScoreStand`/`PieceLessMode`/`ScoreMarkLayer`/`SessionTimelineStrip`
   unchanged in shape, add one optional prop each.** The review canvas is
   the same score/timeline machinery #157/#158 already built and tested —
   marks-consolidated is just "pass every mark instead of the live subset,"
   which those components already do (they take `marks: readonly Mark[]`
   with no live/consolidated distinction baked in). The only new need is a
   different *expansion* — evidence + playback + drill entry, instead of
   evidence alone. Rather than fork the canvases, `ScoreMarkLayer` and
   `SessionTimelineStrip` each gain one optional `renderDetail` prop
   (default: today's `MarkDetail`), and `ScoreStand`/`PieceLessMode` gain a
   pass-through of the same name. Every existing caller and test that omits
   the prop is unaffected — this is additive, not a rewrite, honoring
   "touch only what you must."
   - **Rejected alternative:** duplicate the canvases into
     `ReviewScoreStand`/`ReviewPieceLessMode`. Rejected because it forks
     paging/lane-packing/measurement logic that #157/#158 spent real effort
     proving correct in a real browser (lane packing, clamping, Verovio
     reflow) — a fork drifts.
   - **Rejected alternative:** always render playback+drill in `MarkDetail`
     regardless of context (live pause taps too). Rejected: it's scope creep
     past #159's success criterion (which is about the *review* flow), and
     it re-opens #157/#158's tested live-pause mark-tap UX to regression risk
     for no requirement that asked for it.

2. **A new `tSec -> qstamp` bridge (`passage-cursor.ts`) is the one new
   piece of real algorithmic work.** `PassagePlayer.onTick` reports elapsed
   *playback* seconds against a `PassageManifest.barTimeline: {bar, tSec}[]`
   (coarse, one entry per bar boundary). `ScoreCursor` wants a *qstamp*
   (musical position) sourced from `ScoreIR.bars[].qstampStart/qstampEnd`.
   Nothing in the codebase already converts one to the other — `useLoopPlayer`
   sidesteps this entirely because it drives a synthesized clip from
   `ClipNote` data with an exact qstamp clock, not real audio. The bridge:
   given `tSec`, find the bracketing `barTimeline` pair, compute the
   fraction of the way through that bar, and apply the same fraction to the
   corresponding `ScoreIR` bar's `[qstampStart, qstampEnd]` range. This is a
   pure function, unit-testable with no DOM.

3. **Verdict, carry-forward, WS-gap, and no-comment-zone data are client-side
   types with no backend producer yet.** #162 owns emitting them; #159 owns
   only the shape and the rendering, matching the "no backend emits marks or
   review synthesis yet" constraint. `VerdictState` is a closed union
   (`loading | ready | failed`) so "loud, never blank" is a type-level
   property: there is no state that renders nothing. `CarryForward` is a
   single optional object, not an array — "exactly one carry-forward" is
   enforced the same way `mark.ts`'s branded `MarkAnchor` enforces "no wrong
   bar numbers": by the type shape, not a runtime check a later caller could
   route around.

4. **"Work on this" is a callback, not a fetch.** `SessionReview` takes
   `onWorkOnThis: (mark: Mark) => void` and the review's new
   `MarkPassagePlayback`/drill-entry component fires it. What happens next
   (segment-loop assignment, exercise generation) is server-owned and out of
   scope — proven only as "the callback fires with the tapped mark," which is
   an honest, verifiable claim instead of a fabricated one.

### Alternative considered and rejected: extending the `synthesis` WS event now

Could have added `verdict`/`carryForward` fields to `practice-api.ts`'s
`PracticeWsEvent` `synthesis` variant now, ahead of #162, so `usePracticeSession`
could be wired live. Rejected: the actual backend contract is #162's decision
to make (dimension gating, carry-forward selection logic, etc. live there);
guessing the shape now risks a type this plan owns going stale the moment
#162 lands something different. `SessionReview`'s props are the real contract
this plan commits to; how a future WS event maps onto those props is #162's
problem.

## Modules

### `apps/web/src/lib/session-review.ts`
- **Interface:** `VerdictState`, `CarryForward`, `WsGap`, `NoCommentZone`
  types; `formatWsGap(gap: WsGap): string` (e.g. "4 minutes didn't reach me").
- **Hides:** the wording of the WS-gap disclosure sentence (single source, so
  `SessionReview` and any future surface phrase it identically).
- **Tested through:** `formatWsGap` unit tests (public function, pure).

### `apps/web/src/lib/passage-cursor.ts`
- **Interface:**
  `createPassageQstampSource(barTimeline: PassageManifest["barTimeline"], bars: readonly BarIR[], getTSec: () => number | null): () => number | null`
- **Hides:** bracketing-pair lookup in `barTimeline`, within-bar fraction
  computation, and the fraction's application to the matching `BarIR`'s
  `[qstampStart, qstampEnd]`. Returns `null` when playback hasn't started, is
  past the last known bar, or `bars` has no entry for the resolved bar
  number (silent degradation over guessed drawing, matching mark.ts's
  precedent).
- **Tested through:** the returned function's output for known `tSec`
  inputs — never internals.

### `apps/web/src/components/MarkPassagePlayback.tsx`
- **Interface:**
  `{ mark: Mark; sessionId: string; _mockManifest?: PassageManifest; _mockClip?: string; _playable?: boolean }`
  — same test-escape-hatch shape as `PlayPassageCard`, for the same reason
  (jsdom cannot decode real audio; `/practice-preview` needs deterministic
  data).
- **Hides:** manifest fetch, clip SVG fetch, `PassagePlayer` lifecycle,
  `ScoreCursor` mount/unmount wired through `passage-cursor.ts`, and the
  loading/ready/audio_error/error/no_bars states (a mark whose
  `anchor.type === "timestamp"` has no bars to fetch a passage for — that is
  a fifth, distinct state, not folded into `error`, because it's an honest
  "unavailable," not a failure).
- **Tested through:** rendered states given props/mocked collaborators
  (`api`, `scoreRenderer`, `PassagePlayer`) — the exact mocking shape
  `PlayPassageCard.test.tsx` already established.

### `apps/web/src/components/MarkReviewDetail.tsx`
- **Interface:** `{ mark: Mark; sessionId: string; onClose: () => void; onWorkOnThis: (mark: Mark) => void }`
- **Hides:** composition of evidence text (reusing `mark.ts`'s
  `anchorLabel`/confidence framing, matching `MarkDetail`'s existing wording)
  + `MarkPassagePlayback` + the "Work on this" button.
- **Tested through:** rendered content and the `onWorkOnThis`/`onClose`
  callbacks firing on click.

### `apps/web/src/components/VerdictBanner.tsx`
- **Interface:** `{ verdict: VerdictState; onRetry: () => void }`
- **Hides:** the three renderings (spinner/text copy for `loading`, the
  sentence for `ready`, the explicit retry affordance for `failed`) behind
  one exhaustive switch — TypeScript's exhaustiveness check on the closed
  union is what makes "never blank" a compile-time property: a `VerdictState`
  variant with no case is a type error, not a silently-missing render.
- **Tested through:** rendered output per `VerdictState` variant, and
  `onRetry` firing on click in the `failed` case only.

### `apps/web/src/components/SessionDisclosures.tsx`
- **Interface:** `{ wsGaps: readonly WsGap[]; noCommentZones: readonly NoCommentZone[] }`
- **Hides:** nothing renders when both arrays are empty (a clean session says
  nothing about gaps it didn't have — disclosure is additive, never implies
  a problem that isn't there).
- **Tested through:** rendered list items for given props, and absence of
  the disclosure region when both arrays are empty.

### `apps/web/src/components/SessionReview.tsx`
- **Interface:**
  ```ts
  interface SessionReviewProps {
    pieceId: string | null;
    sessionId: string;
    marks: readonly Mark[];
    durationSeconds: number;
    verdict: VerdictState;
    onRetryVerdict: () => void;
    carryForward: CarryForward | null;
    wsGaps: readonly WsGap[];
    noCommentZones: readonly NoCommentZone[];
    onWorkOnThis: (mark: Mark) => void;
  }
  ```
- **Hides:** which canvas to render (`ScoreStand` when `pieceId` is set, else
  `PieceLessMode`, mirroring `PracticeMode`'s existing ladder logic), wiring
  `renderDetail` on that canvas to `MarkReviewDetail`, and the
  verdict/disclosures/carry-forward chrome around it.
- **Tested through:** rendered composition given props (verdict text present,
  marks passed through to the canvas, carry-forward text present or absent,
  disclosures present or absent) — not the internals of the canvas it
  delegates to (that's `ScoreStand`/`PieceLessMode`'s own test surface).

## Verification Architecture

- **Canonical success state:** `SessionReview`, given a fixed set of props
  (marks, verdict, carry-forward, disclosures), renders the verdict text,
  every mark's glyph, and the carry-forward text; tapping a mark glyph
  renders its evidence and a Play control; tapping Play (with a mocked
  `PassagePlayer`) calls `play()`; tapping "Work on this" calls
  `onWorkOnThis` with that mark. Given `verdict = { status: "failed" }`, the
  retry affordance renders and calling `onRetry` is observable.
- **Automated check:** `bun run test` (vitest) in `apps/web` — component
  tests per module above, plus `bun run typecheck` and `bun run lint`
  (biome) for the new/modified files.
- **Harness:** Task Group 0 extends the existing `/practice-preview` dev-only
  route (`apps/web/src/routes/practice-preview.tsx`) and
  `practice-preview-fixtures.ts` with a review-ready fixture instance and a
  review-failed (synthesis-failure) fixture instance, exactly matching
  #158's precedent for the ladder rungs. The build agent must add both
  fixture sections and confirm `bun run dev` renders both before dispatching
  the component-implementation task groups.
- **Manual click-through (human-lit, not dark-eligible):** with `bun run dev`
  running in `apps/web`, visit `/practice-preview`, confirm the review-ready
  section shows a verdict sentence, tap a mark, confirm evidence + Play
  button render, tap Play (mocked clip data, so playback is silent/instant
  but the state transition and cursor overlay are visible), tap "Work on
  this" and confirm the harness's `data-testid` feedback fires, then check
  the review-failed section renders the explicit retry state (never blank).
  This is the success-criterion flow with two honest caveats recorded up
  front: (a) it exercises fixture data, not a live-recorded session, because
  no backend produces review data yet (#162); (b) "hear passage" here means
  "the playback state machine and cursor visibly work," not literally
  audible sound, since jsdom-safe fixture clips carry no real audio buffer —
  real-audio E2E is gated on #162 same as #158's user-picked ladder was
  gated on #160.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/web/src/lib/session-review.ts` | `VerdictState`, `CarryForward`, `WsGap`, `NoCommentZone`, `formatWsGap` | New |
| `apps/web/src/lib/session-review.test.ts` | `formatWsGap` behavior | New |
| `apps/web/src/lib/passage-cursor.ts` | `createPassageQstampSource` | New |
| `apps/web/src/lib/passage-cursor.test.ts` | bridge behavior for known inputs | New |
| `apps/web/src/components/MarkPassagePlayback.tsx` | promoted play_passage + cursor, session-scoped | New |
| `apps/web/src/components/MarkPassagePlayback.test.tsx` | loading/ready/audio_error/error/no_bars states | New |
| `apps/web/src/components/MarkReviewDetail.tsx` | evidence + playback + "work on this" | New |
| `apps/web/src/components/MarkReviewDetail.test.tsx` | rendered content + callbacks | New |
| `apps/web/src/components/VerdictBanner.tsx` | loading/ready/failed+retry | New |
| `apps/web/src/components/VerdictBanner.test.tsx` | per-variant rendering + retry | New |
| `apps/web/src/components/SessionDisclosures.tsx` | ws gaps + no-comment zones | New |
| `apps/web/src/components/SessionDisclosures.test.tsx` | list rendering + empty case | New |
| `apps/web/src/components/SessionReview.tsx` | top-level review composer | New |
| `apps/web/src/components/SessionReview.test.tsx` | composed rendering + callbacks | New |
| `apps/web/src/components/ScoreMarkLayer.tsx` | optional `renderDetail` prop | Modify |
| `apps/web/src/components/ScoreMarkLayer.test.tsx` | new case: custom `renderDetail` used | Modify |
| `apps/web/src/components/SessionTimelineStrip.tsx` | optional `renderDetail` prop | Modify |
| `apps/web/src/components/ScoreStand.tsx` | optional `renderDetail` pass-through | Modify |
| `apps/web/src/components/PieceLessMode.tsx` | optional `renderDetail` pass-through | Modify |
| `apps/web/src/components/PracticeMode.tsx` | `phase`, review props, `onExitReview`, renders `SessionReview` in review phase | Modify |
| `apps/web/src/components/PracticeMode.test.tsx` | new case: review phase renders `SessionReview` | Modify |
| `apps/web/src/hooks/usePracticeSession.ts` | expose `sessionId` in return | Modify |
| `apps/web/src/routes/practice-preview.tsx` | review-ready + review-failed demo sections | Modify |
| `apps/web/src/routes/practice-preview.test.tsx` | new sections render | Modify |
| `apps/web/src/lib/practice-preview-fixtures.ts` | review fixtures (verdict, carry-forward, ws gaps, no-comment zones) | Modify |

## Open Questions

- Q: Should `SessionReview`'s canvas selection duplicate `PracticeMode`'s
  ladder logic (`resolvePieceLadderState`) or take a plain `pieceId: string | null`?
  Default: plain `pieceId` — by the time review renders, the ladder has
  already resolved (or the session was pieceless from the start); review
  doesn't need confirm-chip/dismiss semantics, only "is there a piece."
- Q: Does `MarkReviewDetail`'s "Work on this" need a loading/assigned state
  like `ExerciseSetCard`'s `handleAssign`? Default: no — since the drill
  generation endpoint behind it doesn't exist yet (out of scope), the button
  fires `onWorkOnThis` and the harness proves it fired; a real loading state
  is #162-adjacent follow-up work once a real endpoint exists.
