# Mark System Design

**Goal:** One mark vocabulary renders identically on two canvases — a Verovio
score overlay and a session timeline strip — so every score-first surface
(practice, review, piece page, drills) shows feedback the same way.

**Not in scope:**

- Backend. No route handlers, no API calls, no persistence. Marks arrive as
  fixture data.
- Producing marks from model output or from the student baseline. #163 already
  produces `lifecycle`; turning a lifecycle into a mark is server work in a
  later issue.
- Wiring marks into practice mode, session review, the piece page, or drills.
  Those are #158/#159/#162/#160/#161.
- Deleting the chat shell (`AppChat`, `ScorePanel`, `ScoreAnnotation`). That is
  #164. This issue adds the replacement alongside; it removes nothing.
- iOS. `apps/ios/.../DesignSystem/Tokens/` has known drift from the web token
  table (#156). It is unowned and untouched here.

## Problem

Epic #154 replaced the chat interface with score-first, mark-based delivery.
Every remaining surface issue consumes marks, but no mark exists as a shared
concept. What exists today is `ScorePanel.tsx` + `ScoreAnnotation.tsx`: an
observation badge layer welded into the chat shell that #164 deletes.

That prior art also carries the exact defect the epic forbids. `ScorePanel.tsx:362`
resolves a bar by array index:

```ts
const measureIdx = obs.barRange[0] - 1;
const el = measureEls[measureIdx];
```

This assumes the Nth `.measure` element in the SVG is bar N. Pickup bars,
repeats, and multi-page rendering all break that assumption. Worse, when the
lookup fails, line 371 invents a position:

```ts
positions.push({ top: 60 + positions.length * 80, left: 20 });
```

The result is a badge drawn at a confidently wrong location on the score. Per
`docs/apps/05-ui-system.md`, "a missing mark costs little; a wrong mark is a
trust-killer." The replacement must be structurally incapable of this.

Second problem: there is no timeline canvas at all. Pieceless sessions are a
permanent first-class state (the score library is copyright-cleared only, so
many sessions never have a score), and today they have nowhere to show
feedback.

## Solution (from the user's perspective)

A student sees a mark as a small chip: a taxonomy glyph (◉ needs work, ○ missed
opportunity, ★ strong), the dimension named in words, and its location. Tapping
it expands the evidence — the MPM-grounded "why" — with a close affordance.

When the piece is known and the score is on screen, marks sit above their bars
on the engraving. When the piece is unknown, or score alignment is too shaky to
trust, the same marks appear on a thin horizontal timeline strip positioned by
elapsed time. Bar numbers are shown only when they are true; otherwise the mark
reads `2:31`, which is always true.

Marks fade as they resolve: an `active` mark is full strength, `improving` is
dimmer, `resolved` is faint. The student never sees a number or a score.

## Design

### The anchor is a branded discriminated union

The hard constraint — wrong bar numbers are never shown — is enforced by types,
not by a guard a future call site can skip:

```ts
declare const anchorBrand: unique symbol;

export type MarkAnchor = { readonly [anchorBrand]: true } & (
  | { readonly type: "bars"; readonly bars: readonly [number, number]; readonly atSeconds: number }
  | { readonly type: "timestamp"; readonly atSeconds: number }
);
```

`resolveAnchor()` is the only function in the codebase that can mint a
`MarkAnchor` — the brand is not exported, so no object literal elsewhere is
assignable to the type. A call site that wants to display bars must have
obtained them from `resolveAnchor`, which returns a `bars` anchor only when
alignment quality clears `ALIGNMENT_MIN`. There is no path that renders bars
without passing the quality check.

**Every anchor carries `atSeconds`, including the `bars` variant.** This is
deliberate: the timeline strip must be able to place any mark, and a
bar-anchored mark that cannot be drawn on the current score page must still
reach the student somewhere true. Elapsed time is the one coordinate every mark
always has.

### Alignment quality is an input, not a client computation

`resolveAnchor` takes `alignmentQuality: number` (0–1) supplied by the caller
from server-side alignment output. The client never estimates it. The threshold
constant lives with the resolver so there is exactly one place it can change.

### Placement is pure; only the adapter touches the DOM

`placeMarks()` takes already-measured, container-relative measure rectangles
and returns `{ placed, unplaced }`. It reads no DOM. This is forced by reality —
jsdom has no layout engine, so `getBoundingClientRect()` returns zeros there and
placement math would be untestable inside a component — but it is also the right
decomposition: the arithmetic and the fallback policy are the substance, and DOM
reading is three lines of adapter in `ScoreMarkLayer`.

Bar identity comes from the score IR, which already exists and is authoritative.
`score-ir.ts` builds `BarIR = { barNumber, measureOn, pageN, ... }` from
Verovio's **timemap**, and `measureOn` is the `id` attribute of the measure
`<g>` in the rendered SVG (`score-ir.ts:242`, keyed through the
`measureByMeasureOn` lookup at `:185`). So the resolution chain is:

```
mark.anchor.bars[0]  ->  BarIR.barNumber  ->  BarIR.measureOn  ->  getElementById  ->  rect
```

This is the precise correction to the prior art. `ScorePanel`'s defect is not
"used an index" in the abstract — Verovio's timemap ordering *is* the bar
ordering. The defect is that it indexed into **DOM order on the currently
rendered page**, and the DOM holds only one page's measures while bar numbers
span the whole piece. Every bar past page 1 therefore resolved to the wrong
element or to the invented fallback.

Two rules make the wrong-bar defect unrepresentable:

1. Bars are matched through `BarIR.measureOn`, never by position in any array
   or in the DOM. `placeMarks` never sees an index.
2. A bar-anchored mark whose measure has no rect returns in `unplaced`. There is
   no invented-position branch; `placeMarks` cannot return a coordinate it did
   not derive from a real measure rect. Page filtering falls out of this for
   free: a bar on another page has no rect, so it degrades to the timeline
   rather than being drawn somewhere false.

### Unplaced marks fall back to the timeline, and the gap is disclosed

Canvas B renders **every** mark, so it is always the complete view. Canvas A
renders only what it can truly place and discloses the remainder as a count
("2 marks not on this page"). This follows the epic's rule that every
degradation is disclosed rather than silently absorbed.

### Lifecycle is server data, rendered, never derived

`Mark.lifecycle` is `active | improving | resolved` — the three mark-worthy
values of #163's four-value `Lifecycle`. `absent` is excluded from the mark type
by construction: an absent dimension produces no mark. `isMarkWorthy()` is the
single derivation, `lifecycle !== "absent"`, and it is the only place that fact
is expressed. Per #157's brief there is deliberately no `markWorthy` field; two
copies of one fact drift.

No component contains a lifecycle transition. Rendering maps lifecycle to
opacity through a frozen lookup table and nothing else.

### `confidence` frames, never gates

`confidence` (`exploratory | provisional | established`) is optional on a mark
and affects only the wording of the expanded evidence — an `exploratory` mark is
prefixed "Early read —". It gates no rendering, hides no mark, and changes no
placement, matching #163's invariant that confidence never gates firing.

### The dimension tint is decorative; the dimension is text

Dimension colour comes from `DIMENSION_COLOR_VAR` (#156's single source) as a
small accent dot, not as a chip background. Two reasons:

1. `--dim-*` values are muted mid-tones (`--dim-timing` is `#9a8a7a`); against
   `--color-on-accent` they land near 3:1 and would fail a 4.5:1 text gate.
   Using them as backgrounds for text would be shipping a contrast failure.
2. Colour must not be the sole carrier of meaning (WCAG 1.4.1). The dimension
   name appears as visible text and in the accessible name, so the dot is
   decorative and exempt from the 3:1 gate on the same grounds #156 exempted
   `border-subtle`.

The chip itself is `ink-primary` on `surface-raised`, a pair the existing
`tokens.contrast.test.ts` already asserts in both columns.

Note for the record, not fixed here: `readTokenTable()` in
`src/test-utils/read-tokens.ts` parses only the `@theme` block and the
`html[data-theme="dark"]` block. The light `--dim-*` values live in a separate
`:root` block at `app.css:284`, so they are invisible to the token contrast
harness today. This spec avoids depending on them being visible rather than
widening #156's harness as a side effect.

### Alternatives rejected

- **Runtime guard instead of a branded type.** A `if (quality < MIN) return null`
  inside the renderer is bypassable by the next surface that renders marks
  directly. The brand makes the bypass a compile error.
- **Reusing `ScoreAnnotation`/`ScorePanel`.** Both die in #164, and
  `ScorePanel` owns the index-based bar lookup this spec exists to eliminate.
  The measurement *technique* is reused; the components are not.
- **`ScoreMarkLayer` reading the DOM itself.** Would make all placement
  behaviour browser-only and untestable in the vitest suite.
- **Mounting the preview in `/app/sandbox`.** Rejected in favour of a dedicated
  `/marks-preview` route: sandbox sits under `/app`, which redirects to
  `/signin` when `VITE_AUTH_MODE=live`, and carries 49KB of unrelated chat-era
  fixtures that #164 will churn.

## Modules

### `src/lib/mark.ts` — the vocabulary

- **Interface:** `Mark`, `MarkAnchor`, `MarkTaxonomy`, `MarkLifecycle`,
  `MarkConfidence`, `AnchorCandidate`, `resolveAnchor()`, `isMarkWorthy()`,
  `anchorLabel()`, `TAXONOMY_GLYPH`, `TAXONOMY_LABEL`, `LIFECYCLE_OPACITY`,
  `ALIGNMENT_MIN`.
- **Hides:** the alignment threshold, the brand that makes bar anchors
  unforgeable, seconds→`m:ss` formatting, the bars-vs-bar singular/plural rule,
  and the fact that `atSeconds` is universal.
- **Depth:** DEEP — a handful of exported names over the entire correctness
  property of the feature.
- **Tested through:** `resolveAnchor`, `anchorLabel`, `isMarkWorthy` return
  values. No internal state is inspected.

### `src/lib/mark-placement.ts` — bar→pixel mapping

- **Interface:** `MeasureRect`, `PlacedMark`, `Placement`, `placeMarks(bars,
  rectsByMeasureOn, marks)`.
- **Hides:** the `barNumber -> measureOn -> rect` resolution chain, the glyph
  vertical offset, the policy that unresolvable and timestamp-anchored marks go
  to `unplaced`, page filtering as an emergent property, and the absence of any
  fallback coordinate.
- **Depth:** DEEP — one function; behind it sits the entire wrong-bar defence.
- **Tested through:** `placeMarks()` return value against synthetic rects.

### `src/components/MarkGlyph.tsx` — the shared atom

- **Interface:** `<MarkGlyph mark expanded onToggle />`.
- **Hides:** taxonomy glyph selection, dimension tint lookup, lifecycle opacity,
  accessible-name composition, tap/keyboard affordances.
- **Depth:** MEDIUM. Justified rather than collapsed into the canvases: it is
  the literal mechanism by which "one vocabulary, two canvases" is true. If each
  canvas drew its own chip, the two could diverge silently, and the success
  criterion "the same mark renders correctly on both canvases" would be
  unenforceable.
- **Tested through:** rendered DOM — accessible name, visible text, expansion.

### `src/components/MarkDetail.tsx` — the expanded state

- **Interface:** `<MarkDetail mark onClose />`.
- **Hides:** confidence-based prose framing, anchor label rendering, close
  affordance.
- **Depth:** MEDIUM.
- **Tested through:** rendered DOM.

### `src/components/ScoreMarkLayer.tsx` — Canvas A

- **Interface:** `<ScoreMarkLayer containerRef bars marks />`.
- **Hides:** DOM lookup of measure elements by `measureOn` id,
  container-relative coordinate translation, re-measurement on resize, the
  unplaced-count disclosure.
- **Depth:** SHALLOW **by design** — it is the DOM adapter. Its shallowness is
  the point: all substance was pushed into `mark-placement.ts` so it could be
  tested. Justified.
- **Tested through:** rendered DOM with stubbed rects.

### `src/components/SessionTimelineStrip.tsx` — Canvas B

- **Interface:** `<SessionTimelineStrip durationSeconds marks />`.
- **Hides:** time→percentage positioning, ordering, the guarantee that every
  mark appears.
- **Depth:** MEDIUM.
- **Tested through:** rendered DOM.

## Verification Architecture

- **Canonical success state:** given one fixture mark array, both canvases
  render the same set of marks with identical accessible names, identical
  taxonomy glyphs, and identical expansion behaviour; a mark whose
  `alignmentQuality` is below threshold renders a `m:ss` label and no bar number
  on either canvas.
- **Automated check (from the worktree):**
  `cd apps/web && bun run test && bunx tsc --noEmit && bun run lint && bun run test:a11y`
- **Harness (Task Group 0):** a cross-canvas contract test,
  `src/components/mark-canvases.contract.test.tsx`, that renders one fixture
  array through both canvases and asserts (a) Canvas B renders **every** mark,
  and (b) every mark Canvas A does render carries an accessible name
  **identical** to that mark's name on Canvas B. The two name sets are
  deliberately *not* equal — Canvas A is the lossy view by design, so asserting
  set equality would encode the opposite of the intended behaviour. Containment
  plus per-mark identity is the correct contract.
  It fails before either canvas exists (module-not-found), and it is the single
  test that encodes the issue's headline success criterion. It is written first
  and re-run after every subsequent group.
- **Deciding check (manual):** a real-browser click-through of `/marks-preview`
  in both themes — the same mark on both canvases, tap-to-expand on both, and a
  low-alignment mark showing a timestamp and never a bar number. For UI work
  this outranks the test count.
- **Explicitly not used:** `vitest-axe` for colour contrast. axe's
  `color-contrast` rule requires layout and silently skips in jsdom, so a
  passing jsdom axe run proves nothing about contrast. Contrast is covered by
  `bun run test:a11y` against a real preview build.

## File Changes

| File | Change | Type |
|---|---|---|
| `src/lib/mark.ts` | Mark vocabulary, branded anchor, `resolveAnchor`, labels | New |
| `src/lib/mark.test.ts` | Unit tests for the above | New |
| `src/lib/mark-placement.ts` | Pure bar→pixel placement | New |
| `src/lib/mark-placement.test.ts` | Unit tests for placement + unplaced policy | New |
| `src/test-utils/mark-fixtures.ts` | Fixture marks across taxonomy × lifecycle × anchor | New |
| `src/components/MarkGlyph.tsx` | Shared mark atom | New |
| `src/components/MarkGlyph.test.tsx` | Accessible name, glyph, colour-not-sole-means | New |
| `src/components/MarkDetail.tsx` | Expanded evidence panel | New |
| `src/components/MarkDetail.test.tsx` | Evidence, confidence framing, close | New |
| `src/components/ScoreMarkLayer.tsx` | Canvas A adapter + disclosure | New |
| `src/components/ScoreMarkLayer.test.tsx` | Placement, disclosure, expansion | New |
| `src/components/SessionTimelineStrip.tsx` | Canvas B | New |
| `src/components/SessionTimelineStrip.test.tsx` | Positioning, completeness, expansion | New |
| `src/components/mark-canvases.contract.test.tsx` | Task Group 0 harness | New |
| `src/routes/marks-preview.tsx` | Dev preview route mounting both canvases | New |
| `src/routeTree.gen.ts` | Regenerated by the TanStack Router plugin | Modify |
| `playwright.a11y.config.ts` | No change; `tests/a11y.spec.ts` gains the route | — |
| `tests/a11y.spec.ts` | Add `/marks-preview` to the theme cases | Modify |

## Open Questions

- **Q:** Should `resolveAnchor`'s `ALIGNMENT_MIN` be 0.8?
  **Default:** 0.8, exported as a named constant with a comment stating it is
  uncalibrated — the same honesty #165 records for `DEVIANT_SAMPLE_MULTIPLE`.
  Tuning needs real alignment-quality distributions, which do not exist yet.
- **Q:** Should Canvas A re-measure on window resize, or only on score render?
  **Default:** re-measure on resize via `ResizeObserver` on the container.
  Verovio reflows on width change, so stale rects would place marks on the wrong
  bars — the exact defect this spec exists to prevent.
- **Q:** Does the timeline strip need its own zoom/scroll for long sessions?
  **Default:** no. A 60s-soft-stop session is short; revisit when a real session
  produces enough marks to collide.
