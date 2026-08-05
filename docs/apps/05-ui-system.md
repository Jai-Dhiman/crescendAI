# UI System: Score-First Surfaces and the Mark System

How feedback is presented to the user. Covers the four surfaces, the mark
system that unifies them, the residual role of conversation, and the visual
direction.

> **Status (2026-08-04):** DESIGN APPROVED (epic #154, brainstorm 2026-08-04).
> Supersedes the chat-first interface and the unified artifact container
> system entirely. The chat shell (`AppChat`), conversation sidebar, and
> artifact-in-chat choreography are scheduled for removal (#164). Build state
> lives in #154's sub-issues, not here.

---

## Design Philosophy

### Feedback attaches to the artifact, never narrates about it

Feedback renders as **marks on the score** (or on the session timeline when no
score is available) — the way a teacher pencils on your music. Prose exists at
exactly three points: the session verdict, the carry-forward, and on-demand
passage explanations. There is no message stream.

This is grounded in three independent research findings (session 2026-08-04,
recorded in #154):

- **Motor learning:** concurrent feedback during play degrades retention and
  splits attention (guidance hypothesis; Sherwood-line bandwidth feedback is
  the validated alternative). Expert teachers correct infrequently and
  selectively.
- **Interface research:** direct manipulation beats chat for iterate/compare
  loops (DirectGPT, CHI 2024); conversation is the right modality only for
  genuinely ambiguous questions.
- **Landscape:** the strongest structural precedent for post-practice review
  is chess.com's Game Review — summary-first, error taxonomy, coach
  narrative — adapted to respect defensible interpretive range (music has no
  engine evaluation).

### Silent while playing, present at boundaries

The app says nothing while the student plays. Marks appear only during pauses
(>= 20s of silence; tunable) and never demand a response. After 60s of
silence the session ends softly (one-tap resume continues the same session).

### The system may fall silent, never guess

A missing mark costs little; a wrong mark is a trust-killer. Every failure
degrades toward saying less, and every degradation is disclosed in the review
(no-comment zones, connection gaps, unanchored passages).

### Serious, adult, restrained

Unchanged from the original vision: no gamification, no streaks, no confetti.
The interface is the student's music, marked up by someone who listened.

---

## The Mark

The mark is the unit of feedback and the system's central contract. Model
generations change behind it; the UI never knows which model produced a mark.

```json
{
  "anchor": { "type": "bars", "bars": [5, 6] },
  "taxonomy": "needs_work",
  "dimension": "pedaling",
  "evidence": "pedal held through the bass change at 5.3; RH-LH blur 3x your usual",
  "lifecycle": "active"
}
```

- **anchor:** `bars` when score alignment quality permits, else `timestamp`
  (seconds into session). Wrong bar numbers are never shown — anchors degrade
  to timestamps, which are always true.
- **taxonomy:** `needs_work` (◉) | `missed_opportunity` (○) | `strong` (★).
  The three-way split distinguishes doing something wrong from failing to
  take an expressive opportunity from playing something well — and leaves
  room for defensible interpretive choices to go unmarked.
- **dimension:** internal routing signal (dynamics, timing, pedaling,
  articulation, phrasing, interpretation). Never shown as a score or number.
- **evidence:** MPM-grounded "why", available on tap — a drill-down
  affordance, not the default state.
- **lifecycle:** `active -> improving -> resolved`, driven by the student
  baseline's symmetric persistence gate (see `03-memory-system.md`). Marks
  fade on the piece page as passages improve.

### Two canvases, one vocabulary

| Canvas | When | Anchor |
|---|---|---|
| Score overlay (Verovio SVG annotation layer) | Piece known and score in library | bars |
| Session timeline strip | Pieceless sessions, or shaky alignment | timestamp |

Every surface renders marks through the same components on either canvas.

---

## The Four Surfaces

### 1. Home — "Your music"

Repertoire cards (piece, learning arc, open mark count), an add-piece flow
(score library search), and a secondary "just play" record button. Tapping a
piece opens practice mode with its score. No sidebar, no conversation list.

### 2. Practice mode — the digital music stand

The score fills the screen: static pages, manual turns, **no live
following**. The follower is an offline orientation tool for the teacher
(where to write feedback, where the review cursor goes) — never a live
performer-facing cursor. In the margin: a recording indicator and metronome.

Piece resolution ladder:

1. **User picked the piece** — score on the stand, full anchoring.
2. **No pick, confident piece_id** — score appears with a dismissible confirm
   chip naming the guess.
3. **No confident match — pieceless mode** (a permanent first-class state:
   the score library is copyright-cleared only, so many sessions live here
   forever): a calm near-empty screen — elapsed time, metronome, and a thin
   session-timeline strip accruing pause-marks.

During pauses (>= 20s), at most one new mark lands on the canvas. It never
requires a response and recedes when playing resumes.

### 3. Session review

On stop, the practice canvas transitions in place to review state:

1. **Verdict** — one teacher sentence. Qualitative, no numbers, no dimension
   scores, ever. When history supports it, the verdict carries continuity
   ("that bass-blur from Tuesday is gone").
2. **Marks, consolidated** on the score or timeline canvas.
3. **Per-mark expansion** — score clip + playback of the student's own audio
   for those bars (cursor-tracked), the evidence in words, and a "work on
   this" entry into a drill.
4. **One carry-forward** — a single next-session focus. The review must feed
   the next session, not just describe the last one.

Trend statements ("pedaling: improving over three weeks") live one tap down,
as directions, never as plotted model output.

Failure states are loud: synthesis failure renders an explicit retry state
(never blank, never fabricated); connection gaps are disclosed ("4 minutes
didn't reach me"); low-confidence transcription spans are disclosed as
no-comment zones.

### 4. Piece page — the score across time

The score with marks accumulated across sessions, faded by lifecycle state;
session history for the piece; open drills. This is where "a teacher pencils
on your score over the semester" lives.

---

## Drills — adaptive bar-shaping

From any mark, "work on this" generates a bar-bounded drill of exactly the
flagged passage (the segment-loop and exercise-generation machinery, promoted
from chat cards to a first-class flow). Variant selection is habit-informed
via the student model: prescriptions adapt to what the student actually does
with them.

---

## What Remains of Conversation

**Passage-scoped ask threads only.** Tapping "ask about this" on any mark
opens a thread whose context is automatic: piece, bars, the student's audio,
the mark's evidence, and the student model. Threads live and die with the
passage. There is no global chat, no conversation list, no chat history
surface.

Rationale: conversation is reserved for genuine ambiguity ("why does this
passage feel harder than it should?"), where back-and-forth reasoning earns
its cost. Status delivery ("what happened in that take") is a render, not a
conversation.

---

## Visual Direction

**Paper-first light, derived warm dark, time-aware.**

- The score is the protagonist of every surface, and notation is black ink on
  paper. The interface is built around warm ivory paper surfaces so the score
  sits natively — no white score card floating in a dark UI.
- Warm near-black ink for text; **Lora** (display) + **Figtree** (body)
  survive; **sage** accent survives; six muted dimension colors survive for
  mark tinting. Espresso/cream is retired.
- Warm dark is **derived from the same token table** (one table, two value
  columns), auto-selected by time of day with manual override. It is not a
  second bespoke design; it gets polish only after light is proven in use.
- Token source of truth: `apps/web/src/styles/app.css` `@theme`. The iOS
  token set (`apps/ios/.../DesignSystem/Tokens/`) mirrors it by hand — drift
  risk noted in #156.

**Shipped in #156 (2026-08-05).** Durable decisions, recorded here because the
design spec was deleted at merge:

- **Light is the base column, dark is the override.** Inverting the previous
  arrangement is a correctness property, not a preference: a surface that
  forgets a dark value now degrades to a visibly wrong light value instead of
  silently inheriting a plausible-looking dark one.
- Token names describe **role**, not pigment (`surface-page/raised/sunken`,
  `ink-primary/secondary/tertiary`, `border-subtle/strong`, `on-accent`,
  `score-canvas`, `danger`, `warn`). The old names had become lies — light
  mode redefined `--color-espresso` to `#ffffff`.
- **`--color-score-canvas` carries the one light/dark asymmetry as a value,
  not a rule.** `.score-container` has a single unconditional
  `background: var(--color-score-canvas)`. On light it equals `surface-page`
  so the engraving sits on the page with no card; on dark it resolves to a
  warm off-white because Verovio emits black notation. Encoding the exception
  as a token keeps the rule count at one and keeps it visible in the table.
- **`danger`/`warn` are tokens because the raw Tailwind reds failed AA on
  paper** (`red-400` 2.66:1, `amber-400` 1.60:1 on ivory). They passed before
  only because every background was dark.
- **Dimension colours have one source**: `apps/web/src/lib/dimension-colors.ts`
  maps each key to a `var(--dim-*)` reference consumed by inline styles, which
  resolve through the cascade and follow the theme. Two divergent JS colour
  maps were deleted. Two of the six needed a light-column variant; the other
  four are shared.
- **`border-subtle` is exempt from the 3:1 gate.** WCAG 1.4.11 scopes 3:1 to
  boundaries needed to *identify* a component or its state; card edges and
  dividers are decorative. A divider cannot be both subtle and 3:1.
- **Theme precedence** lives in `apps/web/src/lib/theme-resolve.ts`: manual
  override, then device-local time of day (dark 19:00-06:59), then light.
  `prefers-color-scheme` was dropped as an input so two signals cannot
  disagree. The pre-hydration flash script interpolates the same exported
  `DUSK_HOUR`/`DAWN_HOUR` constants, so the two paths cannot drift.
- **Contrast is verified two ways**, because axe's `color-contrast` rule
  cannot run in jsdom — it needs real layout and silently skips rather than
  failing. `src/styles/tokens.contrast.test.ts` computes every token pair in
  both columns; `bun run test:a11y` runs axe in a real browser against a
  preview build.

---

## Open Questions

1. **Pause threshold tuning.** 20s is a starting value; real sessions will
   say whether marks feel timely or laggy. Config-tunable, not a hard commit.
2. **Mark density ceiling.** How many marks per session before the review
   feels like a report card? The bandwidth gate controls this implicitly;
   validate with real use.
3. **Open learner model.** Whether to ever show the student their own
   baseline/trend directly is an unresolved design bet (metacognitive aid vs.
   anxiety; literature is thin both ways).
4. **iOS parity.** All surfaces are designed web-first; native SwiftUI
   equivalents follow after web validation.
