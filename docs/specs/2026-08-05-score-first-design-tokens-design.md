# Score-First Design Tokens Design

**Goal:** Every web surface renders from one semantic token table with a light
(paper) and a dark value column, so the engraved score sits natively on the
page instead of inside a hard-coded white card.

**Not in scope:**

- iOS `apps/ios/.../DesignSystem/Tokens/` sync. The iOS token set mirrors this
  table by hand; drift is recorded as an open question here and owned by a
  later issue, not fixed in this one.
- Component layout, spacing, or typography changes. Type scale, `Lora`,
  `Figtree`, and all animation tokens are untouched.
- The four new score-first surfaces (#158-#160). This issue only re-bases the
  tokens the existing components already consume.
- Removing chat-era components (#164 owns that). Chat components get their
  tokens renamed like everything else, then get deleted later.

## Problem

`apps/web/src/styles/app.css` defines a dark-first palette and fakes light
mode by inverting the meaning of colour-named tokens:

```css
@theme            { --color-espresso: #3a3633; }   /* dark brown  */
html[data-theme="light"] {
  --color-espresso: #ffffff;                        /* now white   */
  --color-cream:    #2d2926;                        /* now near-black */
}
```

Three concrete failures follow from this:

1. **The names lie.** `bg-espresso` paints white in light mode and
   `text-cream` paints near-black. Every reader of the 32 consuming files has
   to hold the inversion in their head. `--color-cream` alone has 114
   references.
2. **Light is an afterthought, not a design.** Dark is the base column and
   light is an override, which is backwards for a product whose protagonist is
   black ink on paper (`docs/apps/05-ui-system.md`, Visual Direction).
3. **The score fights the theme.** `.score-container { background: white }`
   (app.css:310) is a hard-coded white card. On the ivory light theme it is a
   visible seam; on dark it is a glaring white rectangle. It is consumed by
   `ScorePanel.tsx:390` and `ProofCard.tsx:262`.

A full inventory of the 32 consuming files turned up three more problems that
a token rename alone would not fix, and that the success criterion cannot be
met without addressing:

4. **The dimension colours are not tokens at all.** The `--dim-*` block
   (app.css:294-301) has **zero consumers**. The real colours live in two
   unrelated JavaScript objects: `lib/mock-session.ts` `DIMENSION_COLORS`
   (the muted family, applied as inline `style={{ backgroundColor }}` by five
   components) and `components/BarScoreChip.tsx` `DIMENSION_COLOR` — a
   completely different bright palette (`#4f9cf9`, `#f97316`, `#a78bfa`, …)
   for the same six dimensions. Three sources of truth, none reading the CSS.
   Because they are JS literals in inline styles, none of them respond to a
   theme change at all.
5. **Token values are duplicated as raw RGB.** The sage accent is written out
   as `122, 154, 130` in `AudioWaveformRing.tsx` and `ListeningMode.tsx`, and
   espresso as `45, 41, 38` in `routes/index.tsx` and `routes/signin.tsx`
   (twice). Five sites that will silently keep the old palette after the swap.
6. **Error and warning states have no tokens.** ~20 sites across 12 files use
   raw `text-red-400`, `bg-red-500/10`, `text-amber-400`. Measured against
   ivory paper every one of them fails AA: `red-400` is 2.66:1, `red-300`
   1.82:1, `amber-400` 1.60:1. They pass today only because the background is
   dark. This is not optional polish — the Playwright axe run in the success
   criterion will fail on these until they are tokenised.

There is also no verification. `axe-core` and `vitest-axe` are already
dependencies but zero of the 40 test files use them, so nothing today would
catch a token whose contrast fails.

## Solution (from the user's perspective)

The app opens on warm ivory paper with near-black ink. Engraved notation sits
directly on that paper with no card, seam, or shadow around it — the score
looks like it is printed on the page. After dusk the app is warm dark instead,
with the same layout, the same accents, and the score tinted to match rather
than glowing white. A manual toggle overrides the time-of-day choice and is
remembered.

## Design

### One table, two value columns

Light is the **base** column, declared in Tailwind's `@theme` block. Dark is
the **second** column, declared once in `html[data-theme="dark"]`. There is no
third place a colour may be defined.

This inverts today's arrangement (dark base, light override). The consequence
that matters: a surface that forgets to declare a dark value degrades to the
*light* value, which is visible and obviously wrong, rather than silently
inheriting a dark value that looks plausible.

Token names describe **role**, not pigment, so they cannot become lies:

| Token | Light (base) | Dark |
|---|---|---|
| `--color-surface-page` | `#fdfaf4` | `#23201d` |
| `--color-surface-raised` | `#f6efe3` | `#2d2926` |
| `--color-surface-sunken` | `#efe6d7` | `#1b1917` |
| `--color-ink-primary` | `#2a2622` | `#f5efe4` |
| `--color-ink-secondary` | `#5c554d` | `#b8b0a6` |
| `--color-ink-tertiary` | `#6f665c` | `#979086` |
| `--color-accent` | `#4a6650` | `#7a9a82` |
| `--color-on-accent` | `#fdfaf4` | `#1b1917` |
| `--color-border-subtle` | `#e6dcc9` | `#3a3633` |
| `--color-border-strong` | `#cdbfa6` | `#504b48` |
| `--color-score-canvas` | `#fdfaf4` | `#f2ece1` |
| `--color-danger` | `#a33a32` | `#f0938c` |
| `--color-warn` | `#8a5a1f` | `#e8b563` |

`espresso`, `cream`, `surface`, `surface-2`, `surface-card`, `border`,
`accent-lighter`, `accent-darker`, `text-primary`, `text-secondary`, and
`text-tertiary` are deleted outright. No deprecated aliases: an alias layer
would let call sites keep the old names indefinitely, and #164 already carries
enough cleanup.

Every ratio above is computed, not eyeballed. All text/surface pairs clear
WCAG AA 4.5:1 in both columns; the tightest is `ink-tertiary` on
`surface-raised` at 4.56:1 (dark) and 4.92:1 (light).

### Error and warning become tokens

`danger` and `warn` join the table because the raw Tailwind reds and ambers in
use today fail AA on paper by a wide margin. The two-column values above clear
AA in both themes (danger 6.28:1 light / 7.14:1 dark; warn 5.66:1 / 8.66:1).
Every `text-red-*`, `bg-red-*`, and `text-amber-*` in the ~20 affected sites
maps to `text-danger` / `bg-danger/10` / `text-warn`.

Two categories stay hard-coded and are explicitly exempted from the sweep and
from the axe run: third-party brand colours (the Google logo SVG fills and the
white/black OAuth buttons in `routes/signin.tsx`) and the dev-only mock SVG
fixture in `routes/app.sandbox.tsx`.

### Dimension colours: one source of truth, then a light column

The `--dim-*` CSS variables have no consumers, so giving them a light column
would change nothing on screen. The fix is to make them real first.

`lib/mock-session.ts` `DIMENSION_COLORS` and `BarScoreChip.tsx`
`DIMENSION_COLOR` are both deleted. In their place, one exported map from
dimension key to CSS variable *reference*:

```ts
// lib/dimension-colors.ts
export const DIMENSION_COLOR_VAR = {
  dynamics: "var(--dim-dynamics)",
  timing: "var(--dim-timing)",
  // ...
} as const
```

Inline styles accept `var()` references, so `style={{ backgroundColor:
DIMENSION_COLOR_VAR[dim] }}` resolves through the cascade and follows the
theme for free. This is what makes the CSS variables load-bearing instead of
decorative, and it is why the light column below is worth defining at all.

`BarScoreChip`'s bright palette (`#4f9cf9`, `#f97316`, …) is drift, not a
deliberate second meaning — the Visual Direction commits to "six muted
dimension colours" for mark tinting. It collapses onto the muted family, which
is a visible change to that component and is called out as such.

With the variables actually consumed, the light column matters. The six were
tuned against a dark background; on ivory paper four still clear the 3:1
non-text UI threshold and two do not:

| Dimension | Light | Dark (unchanged) |
|---|---|---|
| `--dim-dynamics` | `#b0816a` | `#b0816a` |
| `--dim-timing` | `#9a8a7a` | `#9a8a7a` |
| `--dim-pedaling` | `#7a8a9a` | `#7a8a9a` |
| `--dim-articulation` | `#9a7a8a` | `#9a7a8a` |
| `--dim-phrasing` | `#819270` | `#8a9a7a` |
| `--dim-interpretation` | `#948b73` | `#9a917a` |

`dim-phrasing` at its dark value measures 2.89:1 on paper — the single reason
this split exists. Adjusted values hold hue and saturation and lower lightness
only, so the family still reads as one palette.

### Shadow copies of token values are removed

Five sites re-encode a token value as raw channel numbers and would silently
keep the old palette through the swap:

| Site | Literal | Becomes |
|---|---|---|
| `AudioWaveformRing.tsx:11-13` | `SAGE_R/G/B = 122,154,130` | read `--color-accent` at use |
| `ListeningMode.tsx:145,147` | `rgba(122,154,130, …)` | `color-mix()` on `--color-accent` |
| `routes/index.tsx:64` | `#2D2926`, `rgba(45,41,38, …)` | `--color-surface-page` |
| `routes/signin.tsx:66,334` | `rgba(45,41,38, …)` ×2 | `--color-surface-page` |

Two inline `backgroundColor: "white"` props
(`cards/PlayPassageCard.tsx:132-137`, `cards/ScoreHighlightCard.tsx:81-87`)
are the same defect as `.score-container` and take
`var(--color-score-canvas)` for the same reason.

`--color-accent-darker` has zero consumers and is deleted rather than
renamed.

### The score sits on the page

`.score-container`'s `background: white` is replaced by a single rule,
`background: var(--color-score-canvas)`, with no theme-conditional CSS at all.

The asymmetry lives entirely in the value column. On light, `score-canvas`
equals `surface-page`, so the engraving is black ink on the same paper as the
rest of the UI and the container is invisible — no card, no seam. On dark it
resolves to a warm off-white, because Verovio emits black notation that would
be invisible on a dark surface; there the container reads as a sheet of paper
laid on a dark desk.

Encoding the exception as a token value rather than a `html[data-theme="dark"]
.score-container` override is the point: it keeps the rule count at one, and
it means the exception is visible in the same table as everything else instead
of hiding in a selector further down the file.

### Theme resolution becomes time-aware

Today `stores/theme.ts` defaults to `prefers-color-scheme` and `__root.tsx`
treats *absence* of `data-theme` as dark. Both flip.

Precedence, highest first:

1. **Manual override** — `localStorage["crescend-theme"]` is `light` or `dark`.
2. **Time of day** — dark from 19:00 to 06:59 *device-local*, light otherwise.
   Device-local means the boundary follows the user across timezones with no
   stored offset.
3. **Fallback** — light, when no clock is available (SSR).

`prefers-color-scheme` is deliberately dropped as an input. Keeping both it and
time-of-day would produce a two-signal precedence puzzle with no clear answer
when they disagree, and the Visual Direction calls for time-aware, not
system-aware.

Marketing routes (`/` and `/signin`) stay always-dark; that is existing
behaviour and out of scope. They will be driven by an explicit
`data-theme="dark"` instead of by attribute absence.

## Modules

**`resolveTheme` — `apps/web/src/lib/theme-resolve.ts`** *(new)*

- **Interface:** `resolveTheme(input: { stored: string | null; now: Date | null }): "light" | "dark"`
- **Hides:** the precedence order, the dusk/dawn boundary hours, validation of
  untrusted `localStorage` values, and the SSR fallback.
- **Depth:** DEEP. One pure function, three inputs, one of two outputs; the
  entire theme policy lives behind it and callers (the store, the flash
  script, tests) never restate a rule.
- **Tested through:** the exported function only. No store internals, no DOM.

**`readTokenTable` — `apps/web/src/test-utils/read-tokens.ts`** *(new)*

- **Interface:** `readTokenTable(theme: "light" | "dark"): Record<string, string>`
- **Hides:** locating `app.css`, parsing the `@theme` block and the
  `html[data-theme="dark"]` block, and overlaying the dark column on the light
  base so callers get one resolved map.
- **Depth:** DEEP. Callers ask for a theme and get resolved hex values; all
  CSS-text handling is hidden.
- **Tested through:** its own return value, and used by the contrast test.

**`contrastRatio` — `apps/web/src/test-utils/contrast.ts`** *(new)*

- **Interface:** `contrastRatio(fg: string, bg: string): number`
- **Hides:** sRGB linearisation and the WCAG relative-luminance formula.
- **Depth:** DEEP by ratio of interface to implementation, though small.
- **Tested through:** known reference pairs (black/white = 21:1).

## Verification Architecture

- **Canonical success state:** every foreground/surface token pair in both
  columns meets WCAG AA (4.5:1 text, 3:1 non-text UI), no component renders an
  off-table colour, and `.score-container` declares no hard-coded background.
- **Automated checks:**
  1. `bun run test` in `apps/web` — the token-pair contrast test asserts every
     pair in both columns. Fails loudly on a bad hex at the source.
  2. `bun run test:a11y` — Playwright + `@axe-core/playwright` runs the
     `color-contrast` rule against real rendered surfaces in both themes.
     Catches what the token test structurally cannot: a component that
     hard-codes a colour off the table.
  3. `grep -r "espresso\|cream" apps/web/src` returns nothing.
- **Why both:** axe's `color-contrast` rule **cannot run in jsdom** — it needs
  real layout and computed colour resolution, and silently skips instead of
  failing. A vitest+`vitest-axe` contrast check would report success while
  verifying nothing. The token test is therefore the fast deterministic gate,
  and Playwright is the only honest way to run axe contrast at all.
- **Harness:** buildable before the feature. **Task Group 0** delivers
  `contrastRatio`, `readTokenTable`, and the failing token-pair test against
  the *current* palette. It must fail on the current tokens before any token
  is edited — that is the proof the harness measures something.

## File Changes

| File | Change | Type |
|---|---|---|
| `apps/web/src/test-utils/contrast.ts` | WCAG ratio calculation | New |
| `apps/web/src/test-utils/read-tokens.ts` | Parse the two token columns from app.css | New |
| `apps/web/src/styles/tokens.contrast.test.ts` | Assert AA for every pair, both columns | New |
| `apps/web/src/lib/theme-resolve.ts` | `resolveTheme` precedence function | New |
| `apps/web/src/lib/theme-resolve.test.ts` | Precedence and boundary behaviour | New |
| `apps/web/tests/a11y.spec.ts` | Playwright axe contrast, both themes | New |
| `apps/web/playwright.a11y.config.ts` | Playwright config for the a11y run | New |
| `apps/web/src/styles/app.css` | Replace palette; light base + dark column; drop `.score-container` white | Modify |
| `apps/web/src/stores/theme.ts` | Delegate to `resolveTheme`; drop inline system-preference logic | Modify |
| `apps/web/src/routes/__root.tsx` | Flash script and `applyTheme` set `data-theme="dark"` explicitly | Modify |
| `apps/web/package.json` | Add `@axe-core/playwright`; add `test:a11y` script | Modify |
| `apps/web/src/lib/dimension-colors.ts` | Single dimension-key -> CSS-var map | New |
| `apps/web/src/lib/mock-session.ts` | Delete `DIMENSION_COLORS` | Modify |
| `apps/web/src/components/BarScoreChip.tsx` | Delete divergent bright palette; use the shared map | Modify |
| `apps/web/src/components/AudioWaveformRing.tsx` | Drop `SAGE_R/G/B`; read the accent token | Modify |
| `apps/web/src/components/ListeningMode.tsx` | Replace hard-coded sage `rgba()` | Modify |
| `apps/web/src/routes/index.tsx`, `signin.tsx` | Replace hard-coded espresso `rgba()` gradients | Modify |
| `apps/web/src/components/cards/PlayPassageCard.tsx` | Inline `"white"` -> score-canvas token | Modify |
| `apps/web/src/components/cards/ScoreHighlightCard.tsx` | Inline `"white"` -> score-canvas token | Modify |
| ~12 files with red/amber literals | `text-red-*`/`text-amber-*` -> `danger`/`warn` tokens | Modify |
| ~32 component/route files | Rename token utility classes to the new semantic names | Modify |

Measured consumer counts driving the rename sweep: `text-tertiary` 97,
`cream` 112, `surface` 80, `border` 70, `text-secondary` 66, `accent` 64,
`surface-card` 18, `surface-2` 17, `text-primary` 16, `espresso` 13.
Two cross-uses need care rather than a blind find-and-replace:
`border-text-tertiary/50` (a text token used as a border colour, 4 sites) and
`bg-border` (a border token used as a fill, 2 sites). The full per-file list
with exact utility forms is enumerated in the implementation plan.

## Open Questions

- **Q: Do the dusk/dawn boundaries (19:00/07:00) match how people actually
  practise?** Default: ship 19:00/07:00 as named constants in
  `theme-resolve.ts`, revisit when real sessions exist. The manual override
  makes a wrong boundary annoying, not blocking.
- **Q: Does the dark-only score tint (`#f2ece1`) read as "paper on a desk" or
  as "a white card we failed to remove"?** Default: ship the tint with a
  `--color-border-subtle` edge and judge it in the click-through; it is one
  token to change.
- **Q: Is `BarScoreChip`'s bright palette actually deliberate?** It is the one
  visible design change in an otherwise mechanical issue. Default: collapse it
  onto the muted family per the Visual Direction, and show the before/after in
  the click-through. If the bright chips turn out to be intentional
  legibility-at-small-size work, reverting is one file.
- **Q: iOS `Colors.swift` now drifts from this table.** Default: record the
  drift in #156's closing comment and let the iOS surface work pick it up.
  Nothing in this issue reads the iOS tokens, so the drift is inert until
  someone touches iOS.
