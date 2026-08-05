# Score-First Design Tokens Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until `/challenge` returns VERDICT: PROCEED.

**Goal:** Every web surface renders from one semantic token table with a light
(paper) and a dark value column, so the engraved score sits natively on the
page instead of inside a hard-coded white card.
**Spec:** `docs/specs/2026-08-05-score-first-design-tokens-design.md`
**Style:** Follow `CLAUDE.md` and `apps/web/TS_STYLE.md`. Package manager is
`bun`, never `npm`. Lint via `bun run lint` (biome), typecheck via
`bun run typecheck`, unit/integration tests via `bun run test` (vitest,
jsdom), a11y tests via `bun run test:a11y` (new, Playwright).

All commands below assume cwd `apps/web` inside the worktree
`/Users/jdhiman/Documents/crescendai/.worktrees/issue-156-design-tokens`.

---

## Rename Mapping (definitive — apply exactly, do not improvise)

The spec deletes `espresso, cream, surface, surface-2, surface-card, border,
accent-lighter, accent-darker, text-primary, text-secondary, text-tertiary`
and gives the *new* token table, but not the old→new call-site mapping. That
mapping is fixed here after reading every consuming file's actual usage form
(`bg-`, `text-`, `border-`, …). Two consolidations are judgment calls, called
out explicitly:

| Old token | New token | Note |
|---|---|---|
| `surface-card` | `surface-raised` | consolidation: card and mid-level surface become one role |
| `surface-2` | `surface-sunken` | recessed/secondary surface |
| `surface` | `surface-raised` | same consolidation as `surface-card` |
| `espresso` (as `bg-espresso`) | `surface-page` | page-level background |
| `cream` (as `text-cream`) | `ink-primary` | primary text/ink role |
| `text-primary` | `ink-primary` | consolidates onto the same token as `cream` — both tracked "primary text," redundantly |
| `text-secondary` | `ink-secondary` | — |
| `text-tertiary` | `ink-tertiary` | includes `border-text-tertiary/50` sites — the whole substring `text-tertiary` is renamed regardless of utility prefix, so `border-text-tertiary/50` becomes `border-ink-tertiary/50` automatically |
| `border` (as `X-border`) | `border-subtle` | default: no site in this codebase asks for the stronger of the two new border tokens; `border-strong` ships with zero consumers, same status `--dim-*` had before this issue |
| `accent-lighter` | **no direct token** — see per-site table below | table has no "lighter" variant; each site gets an opacity modifier on `accent` instead |
| `accent-darker` | **deleted, zero consumers** | confirmed by repo grep — only ever defined in `app.css`, never read |
| `red-300`/`red-400`/`red-500`/`red-600`/`red-200` (as `text-`/`bg-`/`border-`/`hover:*-`) | `danger` | AA-safe on paper per spec |
| `amber-400` (as `text-`) | `warn` | AA-safe on paper per spec |

**`accent-lighter` per-site resolution** (no mechanical rule — each is a hover/emphasis state, resolved by hand in the file's own task):

| File:line | Old | New |
|---|---|---|
| `MessageContent.tsx:22` | `text-accent-lighter` | `text-accent/70` |
| `MessageContent.tsx:57` | `hover:text-accent-lighter` | `hover:text-accent/70` |
| `ListeningMode.tsx:256` | `hover:text-accent-lighter` | `hover:text-accent/70` |
| `ListeningMode.tsx:389` | `hover:text-accent-lighter` | `hover:text-accent/70` |
| `ErrorBoundary.tsx:54` | `hover:bg-accent-lighter` | `hover:bg-accent/80` |

**Two `text-espresso` sites are NOT `surface-page`** (that token is a
background role; these use espresso as *text*, which is a different role in
the old dark-first scheme):

| File:line | Old | New | Why |
|---|---|---|---|
| `ErrorBoundary.tsx:54` | `text-espresso` | `text-on-accent` | it's the label text on a `bg-accent` button — exactly the role `--color-on-accent` exists for |
| `routes/index.tsx:81` | `bg-cream text-espresso` | `bg-ink-primary text-surface-page` | this hero CTA is a light chip with dark text, on an always-dark landing page. `routes/index.tsx` will resolve as `data-theme="dark"` after Task Group D, at which point `ink-primary`'s **dark column** (`#f5efe4`, light) and `surface-page`'s **dark column** (`#23201d`, dark) reproduce the original light-chip/dark-text pairing exactly by swapping which role supplies the background vs. the text. Flag this pairing in `/review` — if the button reads wrong in the click-through, the one-line fix is `bg-surface-page text-ink-primary` reverted to `bg-ink-primary text-surface-page` (already correct) vs. some other pair; do not silently change it without visual confirmation. |

All other `bg-espresso` sites (11 of them) are the mechanical `→
bg-surface-page` rename with no role ambiguity.

---

## File Structure

| File | Responsibility | Interface | Depth | New/Modify |
|---|---|---|---|---|
| `apps/web/src/test-utils/contrast.ts` | WCAG contrast ratio math | `contrastRatio(fg, bg): number` | DEEP | New |
| `apps/web/src/test-utils/read-tokens.ts` | Parse both token columns out of `app.css` | `readTokenTable(theme): Record<string,string>` | DEEP | New |
| `apps/web/src/styles/tokens.contrast.test.ts` | Assert AA for every foreground/surface pair, both columns | test file | — | New |
| `apps/web/src/lib/theme-resolve.ts` | Theme precedence (manual → time-of-day → fallback) | `resolveTheme(input): "light"\|"dark"` | DEEP | New |
| `apps/web/src/lib/theme-resolve.test.ts` | Precedence + boundary behavior | test file | — | New |
| `apps/web/src/lib/dimension-colors.ts` | Single dimension→CSS-var map | `DIMENSION_COLOR_VAR: Record<Dimension,string>` | DEEP | New |
| `apps/web/src/lib/dimension-colors.test.ts` | Map shape + var-reference format | test file | — | New |
| `apps/web/tests/a11y.spec.ts` | Playwright axe `color-contrast`, both themes | Playwright test | — | New |
| `apps/web/playwright.a11y.config.ts` | Playwright config for the a11y run | config | — | New |
| `apps/web/src/styles/app.css` | Palette table, score-container rule | — | — | Modify |
| `apps/web/src/stores/theme.ts` | Delegates to `resolveTheme` | `useThemeStore` | — | Modify |
| `apps/web/src/routes/__root.tsx` | Flash script + `applyTheme`, explicit dark for landing | — | — | Modify |
| `apps/web/package.json` | `@axe-core/playwright` dep, `test:a11y` script | — | — | Modify |
| `apps/web/src/lib/mock-session.ts` | Delete `DIMENSION_COLORS` | — | — | Modify |
| ~28 component/route files | Rename/rebase to new tokens | — | — | Modify |

---

## Verification Architecture (from spec, restated for the build agent)

- `bun run test` — `tokens.contrast.test.ts` asserts every declared
  foreground/surface pair clears WCAG AA (4.5:1 text, 3:1 non-text) in both
  the light and dark columns. **This test is Task Group 0 and must FAIL
  against the current `app.css` before any token is touched** — that failure
  is the proof the harness measures something real.
- `bun run test:a11y` — Playwright + `@axe-core/playwright` runs the
  `color-contrast` rule against real rendered surfaces, both themes. Catches
  what the token test structurally cannot: a component rendering an
  off-table color. Runs only after every consumer file is migrated (Task
  Group F, last).
- `grep -rE "\b(espresso|cream)\b" src` (excluding `.test.` files) returns
  nothing once Task Groups B–E are done.
- `bun run typecheck` and `bun run lint` must stay green after every task —
  run them as part of each task's verification, not just at the end.

---

## Task Groups

```
Group 0 (parallel):      Task 0.1, Task 0.2
Group 0b (seq, needs 0): Task 0.3   <- MUST FAIL as written
Group A (parallel, needs 0b done): Task A.1, Task A.2
Group B (seq, needs 0b):  Task B.1  <- flips Task 0.3 to PASS
Group C (parallel, needs A.2 + B.1): Tasks C.1–C.13 (hand-treated files)
Group D (seq, needs A.1 + B.1):      Task D.1, Task D.2 (theme resolution)
Group E (parallel, needs B.1; independent of C/D — no shared files):
                                      Tasks E.1–E.4 (mechanical rename, batch 1)
Group E2 (parallel, needs B.1; independent of C/D/E):
                                      Tasks E.5–E.8 (mechanical rename, batch 2)
Group F (seq, needs C, D, E, E2 all done): Task F.1, Task F.2, Task F.3 (a11y harness + final grep gate)
```

`[SHIPS INDEPENDENTLY]`: none. This plan is one atomic visual re-base — a
partial merge would leave the app half-renamed (some surfaces reading old
dark-first tokens, some reading the new light-first table), which is worse
than not shipping. The whole plan lands in one PR.

---

# Task Group 0 — Verification Harness

### Task 0.1: `contrastRatio` WCAG math
**Group:** 0 (parallel with 0.2)

**Behavior being verified:** `contrastRatio` returns the WCAG relative-luminance
contrast ratio for two hex colors, matching known reference values.
**Interface under test:** `contrastRatio(fg: string, bg: string): number`

**Files:**
- Create: `apps/web/src/test-utils/contrast.ts`
- Test: `apps/web/src/test-utils/contrast.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/test-utils/contrast.test.ts
import { describe, expect, it } from "vitest";
import { contrastRatio } from "./contrast";

describe("contrastRatio", () => {
	it("returns 21:1 for black on white", () => {
		expect(contrastRatio("#000000", "#ffffff")).toBeCloseTo(21, 1);
	});

	it("returns 1:1 for identical colors", () => {
		expect(contrastRatio("#7a9a82", "#7a9a82")).toBeCloseTo(1, 5);
	});

	it("is symmetric", () => {
		const a = contrastRatio("#2a2622", "#fdfaf4");
		const b = contrastRatio("#fdfaf4", "#2a2622");
		expect(a).toBeCloseTo(b, 5);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/test-utils/contrast.test.ts
```
Expected: FAIL — `Failed to resolve import "./contrast"` (module does not exist).

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/web/src/test-utils/contrast.ts
// WCAG 2.x relative luminance + contrast ratio.
// https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html

function srgbToLinear(channel: number): number {
	const c = channel / 255;
	return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function hexToRgb(hex: string): [number, number, number] {
	const clean = hex.replace("#", "");
	const r = Number.parseInt(clean.slice(0, 2), 16);
	const g = Number.parseInt(clean.slice(2, 4), 16);
	const b = Number.parseInt(clean.slice(4, 6), 16);
	return [r, g, b];
}

function relativeLuminance(hex: string): number {
	const [r, g, b] = hexToRgb(hex);
	const [rl, gl, bl] = [r, g, b].map(srgbToLinear);
	return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl;
}

/** WCAG contrast ratio between two hex colors, from 1 (no contrast) to 21 (max). */
export function contrastRatio(fg: string, bg: string): number {
	const l1 = relativeLuminance(fg);
	const l2 = relativeLuminance(bg);
	const lighter = Math.max(l1, l2);
	const darker = Math.min(l1, l2);
	return (lighter + 0.05) / (darker + 0.05);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/test-utils/contrast.test.ts
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/test-utils/contrast.ts apps/web/src/test-utils/contrast.test.ts
git commit -m "test(#156): add contrastRatio WCAG helper"
```

---

### Task 0.2: `readTokenTable` parses both columns from `app.css`
**Group:** 0 (parallel with 0.1)

**Behavior being verified:** `readTokenTable("light")` returns the `@theme`
base values; `readTokenTable("dark")` overlays the `html[data-theme="dark"]`
block on top of the base. Both are read from the **current** `app.css` — this
test intentionally exercises today's file (`--color-espresso`, etc.) so it is
a correctness check on the parser, independent of the palette swap.

**Interface under test:** `readTokenTable(theme: "light" | "dark"): Record<string, string>`

**Files:**
- Create: `apps/web/src/test-utils/read-tokens.ts`
- Test: `apps/web/src/test-utils/read-tokens.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/test-utils/read-tokens.test.ts
import { describe, expect, it } from "vitest";
import { readTokenTable } from "./read-tokens";

describe("readTokenTable", () => {
	it("reads the @theme base block as the light table", () => {
		const light = readTokenTable("light");
		expect(light["color-accent"]).toBe("#7a9a82");
	});

	it("overlays html[data-theme=\"dark\"] on the base for the dark table", () => {
		const dark = readTokenTable("dark");
		// today's file only overrides text-primary/etc under [data-theme="light"],
		// so a token with no dark-block entry should still resolve from base.
		expect(dark["font-display"]).toBe('"Lora", Georgia, serif');
	});

	it("strips the leading -- and trailing semicolon from every key/value", () => {
		const light = readTokenTable("light");
		for (const key of Object.keys(light)) {
			expect(key.startsWith("--")).toBe(false);
		}
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/test-utils/read-tokens.test.ts
```
Expected: FAIL — `Failed to resolve import "./read-tokens"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/web/src/test-utils/read-tokens.ts
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const APP_CSS_PATH = fileURLToPath(
	new URL("../styles/app.css", import.meta.url),
);

/** Pulls `--name: value;` declarations out of one `{ ... }` block of raw CSS text. */
function parseDeclarationBlock(blockBody: string): Record<string, string> {
	const table: Record<string, string> = {};
	const re = /--([a-zA-Z0-9-]+):\s*([^;]+);/g;
	let match: RegExpExecArray | null;
	// biome-ignore lint/suspicious/noAssignInExpressions: standard regex exec loop
	while ((match = re.exec(blockBody)) !== null) {
		table[match[1]] = match[2].trim();
	}
	return table;
}

/** Extracts the body of the first top-level `selector { ... }` block matching `selectorRe`. */
function extractBlock(css: string, selectorRe: RegExp): string | null {
	const match = selectorRe.exec(css);
	if (!match) return null;
	const start = match.index + match[0].length;
	let depth = 1;
	let i = start;
	while (i < css.length && depth > 0) {
		if (css[i] === "{") depth++;
		if (css[i] === "}") depth--;
		i++;
	}
	return css.slice(start, i - 1);
}

/**
 * Reads the two-column token table from app.css. Light is the `@theme` base
 * block; dark overlays `html[data-theme="dark"]` on top of it. Returns a flat
 * map of bare variable name (no `--`) to its declared value.
 */
export function readTokenTable(theme: "light" | "dark"): Record<string, string> {
	const css = readFileSync(APP_CSS_PATH, "utf-8");

	const themeBlock = extractBlock(css, /@theme\s*\{/);
	const base = themeBlock ? parseDeclarationBlock(themeBlock) : {};

	if (theme === "light") return base;

	const darkBlock = extractBlock(css, /html\[data-theme=["']dark["']\]\s*\{/);
	const darkOverrides = darkBlock ? parseDeclarationBlock(darkBlock) : {};

	return { ...base, ...darkOverrides };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/test-utils/read-tokens.test.ts
```
Expected: PASS (3 tests) — against the **current** app.css (no `html[data-theme="dark"]`
block exists yet, so the dark-overlay test exercises the "no block found, fall
back to base" path).

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/test-utils/read-tokens.ts apps/web/src/test-utils/read-tokens.test.ts
git commit -m "test(#156): add readTokenTable app.css parser"
```

---

### Task 0.3: Token-pair contrast test — MUST FAIL against the current palette
**Group:** 0b (sequential, depends on Task 0.1 and Task 0.2)

**Behavior being verified:** Every declared foreground/surface pair in both
theme columns clears WCAG AA. This is written against the **new** token names
(`ink-primary`, `surface-page`, `danger`, `warn`, …), which do not exist in
`app.css` yet — so this test fails now for two independent reasons: the keys
are `undefined`, and even where old keys coincidentally exist, today's
dark-first values don't satisfy the new light-base pairing. That double
failure is the harness's proof it is checking something real.

**Interface under test:** `readTokenTable` + `contrastRatio`, composed.

**Files:**
- Create: `apps/web/src/styles/tokens.contrast.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/styles/tokens.contrast.test.ts
import { describe, expect, it } from "vitest";
import { contrastRatio } from "../test-utils/contrast";
import { readTokenTable } from "../test-utils/read-tokens";

// [fg, bg, minimum ratio] — 4.5 for text, 3.0 for non-text UI (borders on surfaces).
const TEXT_PAIRS: Array<[string, string]> = [
	["color-ink-primary", "color-surface-page"],
	["color-ink-primary", "color-surface-raised"],
	["color-ink-secondary", "color-surface-page"],
	["color-ink-secondary", "color-surface-raised"],
	["color-ink-tertiary", "color-surface-page"],
	["color-ink-tertiary", "color-surface-raised"],
	["color-on-accent", "color-accent"],
	["color-danger", "color-surface-page"],
	["color-warn", "color-surface-page"],
];

const UI_PAIRS: Array<[string, string]> = [
	["color-border-subtle", "color-surface-page"],
	["color-border-strong", "color-surface-page"],
];

describe.each(["light", "dark"] as const)("token contrast (%s)", (theme) => {
	const table = readTokenTable(theme);

	it.each(TEXT_PAIRS)("%s on %s clears 4.5:1", (fgKey, bgKey) => {
		const fg = table[fgKey];
		const bg = table[bgKey];
		expect(fg, `${fgKey} is not declared for ${theme}`).toBeDefined();
		expect(bg, `${bgKey} is not declared for ${theme}`).toBeDefined();
		expect(contrastRatio(fg, bg)).toBeGreaterThanOrEqual(4.5);
	});

	it.each(UI_PAIRS)("%s on %s clears 3:1", (fgKey, bgKey) => {
		const fg = table[fgKey];
		const bg = table[bgKey];
		expect(fg, `${fgKey} is not declared for ${theme}`).toBeDefined();
		expect(bg, `${bgKey} is not declared for ${theme}`).toBeDefined();
		expect(contrastRatio(fg, bg)).toBeGreaterThanOrEqual(3);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/styles/tokens.contrast.test.ts
```
Expected: FAIL — every case fails with `color-ink-primary is not declared for light`
(or equivalent) because none of `ink-primary`, `surface-page`, `surface-raised`,
`on-accent`-with-new-value, `danger`, `warn`, `border-subtle`, `border-strong`
exist in `app.css` yet. This is the required pre-edit failure.

- [ ] **Step 3: No implementation in this task.** Task B.1 (below) edits
`app.css` and is what flips this test green. Do not touch `app.css` here.

- [ ] **Step 4: N/A — this task's "pass" state is deferred to Task B.1.**

- [ ] **Step 5: Commit the failing test as its own commit** (this is the
harness-must-fail-first proof; committing it red, separately from the fix,
is intentional and matches the spec's verification architecture)

```bash
git add apps/web/src/styles/tokens.contrast.test.ts
git commit -m "test(#156): add token-pair contrast gate (fails against current palette)"
```

---

# Task Group A — New Deep Modules

### Task A.1: `resolveTheme` precedence function
**Group:** A (parallel with A.2)

**Behavior being verified:** manual override wins; absent that, time-of-day
(dark 19:00–06:59, light otherwise) wins; absent both, light.
**Interface under test:** `resolveTheme(input: { stored: string | null; now: Date | null }): "light" | "dark"`

**Files:**
- Create: `apps/web/src/lib/theme-resolve.ts`
- Test: `apps/web/src/lib/theme-resolve.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/theme-resolve.test.ts
import { describe, expect, it } from "vitest";
import { resolveTheme } from "./theme-resolve";

describe("resolveTheme", () => {
	it("honors a stored manual override over the clock", () => {
		const noon = new Date(2026, 0, 1, 12, 0);
		expect(resolveTheme({ stored: "dark", now: noon })).toBe("dark");
		const midnight = new Date(2026, 0, 1, 23, 0);
		expect(resolveTheme({ stored: "light", now: midnight })).toBe("light");
	});

	it("ignores an invalid stored value and falls through to the clock", () => {
		const noon = new Date(2026, 0, 1, 12, 0);
		expect(resolveTheme({ stored: "sepia", now: noon })).toBe("light");
	});

	it("is dark from 19:00 up to (not including) 07:00, device-local", () => {
		expect(resolveTheme({ stored: null, now: new Date(2026, 0, 1, 19, 0) })).toBe("dark");
		expect(resolveTheme({ stored: null, now: new Date(2026, 0, 1, 23, 59) })).toBe("dark");
		expect(resolveTheme({ stored: null, now: new Date(2026, 0, 1, 6, 59) })).toBe("dark");
		expect(resolveTheme({ stored: null, now: new Date(2026, 0, 1, 18, 59) })).toBe("light");
		expect(resolveTheme({ stored: null, now: new Date(2026, 0, 1, 7, 0) })).toBe("light");
	});

	it("falls back to light when no clock is available", () => {
		expect(resolveTheme({ stored: null, now: null })).toBe("light");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/lib/theme-resolve.test.ts
```
Expected: FAIL — `Failed to resolve import "./theme-resolve"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/web/src/lib/theme-resolve.ts
export type Theme = "light" | "dark";

const DUSK_HOUR = 19; // 19:00 device-local, dark begins
const DAWN_HOUR = 7; // 07:00 device-local, light begins

function isValidTheme(value: string | null): value is Theme {
	return value === "light" || value === "dark";
}

/**
 * Theme precedence, highest first:
 *   1. manual override (`stored`, validated — junk values are ignored)
 *   2. time of day (dark 19:00–06:59 device-local, light 07:00–18:59)
 *   3. light, when no clock is available (SSR)
 */
export function resolveTheme(input: { stored: string | null; now: Date | null }): Theme {
	if (isValidTheme(input.stored)) return input.stored;

	if (input.now === null) return "light";

	const hour = input.now.getHours();
	return hour >= DUSK_HOUR || hour < DAWN_HOUR ? "dark" : "light";
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/lib/theme-resolve.test.ts
```
Expected: PASS (4 tests, 8 assertions)

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/theme-resolve.ts apps/web/src/lib/theme-resolve.test.ts
git commit -m "feat(#156): add resolveTheme time-aware precedence"
```

---

### Task A.2: `dimension-colors.ts` — single source of truth
**Group:** A (parallel with A.1)

**Behavior being verified:** the exported map has exactly the six dimension
keys, each resolving to a `var(--dim-*)` CSS reference (not a raw hex), so
inline `style={{ backgroundColor }}` consumers follow the theme for free.
**Interface under test:** `DIMENSION_COLOR_VAR: Record<Dimension, string>`

**Files:**
- Create: `apps/web/src/lib/dimension-colors.ts`
- Test: `apps/web/src/lib/dimension-colors.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/dimension-colors.test.ts
import { describe, expect, it } from "vitest";
import { DIMENSION_COLOR_VAR } from "./dimension-colors";

describe("DIMENSION_COLOR_VAR", () => {
	it("has exactly the six score dimensions", () => {
		expect(Object.keys(DIMENSION_COLOR_VAR).sort()).toEqual(
			[
				"articulation",
				"dynamics",
				"interpretation",
				"pedaling",
				"phrasing",
				"timing",
			].sort(),
		);
	});

	it("resolves each dimension to its own CSS variable reference", () => {
		expect(DIMENSION_COLOR_VAR.dynamics).toBe("var(--dim-dynamics)");
		expect(DIMENSION_COLOR_VAR.timing).toBe("var(--dim-timing)");
		expect(DIMENSION_COLOR_VAR.pedaling).toBe("var(--dim-pedaling)");
		expect(DIMENSION_COLOR_VAR.articulation).toBe("var(--dim-articulation)");
		expect(DIMENSION_COLOR_VAR.phrasing).toBe("var(--dim-phrasing)");
		expect(DIMENSION_COLOR_VAR.interpretation).toBe(
			"var(--dim-interpretation)",
		);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/lib/dimension-colors.test.ts
```
Expected: FAIL — `Failed to resolve import "./dimension-colors"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/web/src/lib/dimension-colors.ts
import type { Dimension } from "./mock-session";

/**
 * One dimension→color mapping, used everywhere a score dimension needs a
 * swatch. Each entry is a `var()` reference into app.css's `--dim-*` custom
 * properties, not a literal hex — inline `style={{ backgroundColor }}`
 * consumers therefore repaint automatically on a theme change, the same as
 * any Tailwind utility class would.
 */
export const DIMENSION_COLOR_VAR: Record<Dimension, string> = {
	dynamics: "var(--dim-dynamics)",
	timing: "var(--dim-timing)",
	pedaling: "var(--dim-pedaling)",
	articulation: "var(--dim-articulation)",
	phrasing: "var(--dim-phrasing)",
	interpretation: "var(--dim-interpretation)",
};
```

If `Dimension` is not already exported from `src/lib/mock-session.ts`, check
its actual export name with `grep -n "type Dimension\|Dimension =" src/lib/mock-session.ts`
before writing the import — do not guess.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/lib/dimension-colors.test.ts
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/dimension-colors.ts apps/web/src/lib/dimension-colors.test.ts
git commit -m "feat(#156): add single-source dimension color map"
```

---

# Task Group B — The Palette Itself

### Task B.1: Rewrite `app.css`'s token table (flips Task 0.3 green)
**Group:** B (sequential, depends on Task 0.3 existing red)

**Behavior being verified:** `tokens.contrast.test.ts` (Task 0.3) passes for
both theme columns; `.score-container` no longer hard-codes `background: white`.

**Files:**
- Modify: `apps/web/src/styles/app.css`

- [ ] **Step 1: Confirm the gate test still fails, for the right reason**

```bash
cd apps/web && bun run test src/styles/tokens.contrast.test.ts
```
Expected: FAIL (from Task 0.3 — unchanged since then).

- [ ] **Step 2: Implement — replace the `@theme` palette block**

Replace lines 3–29 of `app.css` (the `/* Espresso/Cream palette */` comment
through `--color-text-tertiary: #78716c;`) with:

```css
	/* Score-first palette: light is the base column, dark is declared once
	   below in html[data-theme="dark"]. No third place a color may be defined. */
	--color-surface-page: #fdfaf4;
	--color-surface-raised: #f6efe3;
	--color-surface-sunken: #efe6d7;
	--color-ink-primary: #2a2622;
	--color-ink-secondary: #5c554d;
	--color-ink-tertiary: #6f665c;
	--color-accent: #4a6650;
	--color-on-accent: #fdfaf4;
	--color-border-subtle: #e6dcc9;
	--color-border-strong: #cdbfa6;
	--color-score-canvas: #fdfaf4;
	--color-danger: #a33a32;
	--color-warn: #8a5a1f;
	--shadow-card: 0 2px 8px rgba(0, 0, 0, 0.08);
```

Delete the `[data-landing]` block (lines 129–136) entirely — it exists only
to re-flavor `--color-espresso`/`--color-surface`/etc for the marketing
routes, and those routes now get their dark look from the standard
`html[data-theme="dark"]` block via the explicit `data-theme="dark"` Task D.1
adds, per the spec's Design section ("driven by an explicit
`data-theme="dark"` instead of by attribute absence").

Replace the `html[data-theme="light"] { ... }` block (lines 138–152) with a
`html[data-theme="dark"]` block carrying the dark column:

```css
html[data-theme="dark"] {
	--color-surface-page: #23201d;
	--color-surface-raised: #2d2926;
	--color-surface-sunken: #1b1917;
	--color-ink-primary: #f5efe4;
	--color-ink-secondary: #b8b0a6;
	--color-ink-tertiary: #979086;
	--color-accent: #7a9a82;
	--color-on-accent: #1b1917;
	--color-border-subtle: #3a3633;
	--color-border-strong: #504b48;
	--color-score-canvas: #f2ece1;
	--color-danger: #f0938c;
	--color-warn: #e8b563;
	--shadow-card: 0 2px 8px rgba(0, 0, 0, 0.2);
}
```

Update the `@layer base` block (`body { background-color: var(--color-espresso); color: var(--color-text-primary); }`)
to:

```css
	body {
		background-color: var(--color-surface-page);
		color: var(--color-ink-primary);
		font-family: var(--font-sans);
		line-height: 1.75;
	}
```

Update the `score-pulse` keyframe (currently references the deleted
`--color-accent-lighter`):

```css
@keyframes score-pulse {
	0% {
		color: color-mix(in srgb, var(--color-accent) 60%, var(--color-surface-page));
	}
	100% {
		color: var(--color-accent);
	}
}
```

Update the six `--dim-*` variables under `:root` to the spec's light-column
values (dark stays the current hex, now declared explicitly so it doesn't
silently inherit):

```css
:root {
	--dim-dynamics: #b0816a;
	--dim-timing: #9a8a7a;
	--dim-pedaling: #7a8a9a;
	--dim-articulation: #9a7a8a;
	--dim-phrasing: #819270;
	--dim-interpretation: #948b73;
}

html[data-theme="dark"] {
	--dim-phrasing: #8a9a7a;
	--dim-interpretation: #9a917a;
	/* dynamics, timing, pedaling, articulation are unchanged across themes,
	   so they are declared once above and not repeated here. */
}
```
(Add these two `--dim-phrasing`/`--dim-interpretation` lines into the
existing `html[data-theme="dark"] { ... }` block from above, not as a second
separate block.)

Replace `.score-container`'s hard-coded background:

```css
.score-container {
	min-height: 200px;
	background: var(--color-score-canvas);
	border-radius: 0.5rem;
	padding: 1rem;
}
```

- [ ] **Step 3: Run the gate test — verify it PASSES**

```bash
cd apps/web && bun run test src/styles/tokens.contrast.test.ts
```
Expected: PASS — all cases in both `light` and `dark` describe blocks.

- [ ] **Step 4: Confirm nothing else in app.css regressed**

```bash
cd apps/web && bun run test src/test-utils/read-tokens.test.ts src/test-utils/contrast.test.ts
bun run typecheck
```
Expected: all PASS. (Component tests will fail at this point — that's
expected and fixed by Task Groups C/D/E; do not chase those failures here.)

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/styles/app.css
git commit -m "feat(#156): rebase app.css to the light-base score-first token table"
```

---

# Task Group C — Hand-Treated Files (dimension colors, shadow copies, role-ambiguous renames)

Each task in this group touches exactly one file and folds in every kind of
edit that file needs (mechanical rename + any special-case from the mapping
table above), so no two tasks in this parallel group touch the same file.

### Task C.1: `mock-session.ts` — delete `DIMENSION_COLORS`
**Group:** C (parallel)

**Behavior being verified:** `DIMENSION_COLORS` is no longer exported;
`DIMENSION_LABELS` is untouched.
**Interface under test:** module exports of `src/lib/mock-session.ts`

**Files:**
- Modify: `apps/web/src/lib/mock-session.ts`
- Test: `apps/web/src/lib/mock-session.test.ts` (new — this file had no
  existing test; add a minimal one asserting the export surface)

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/mock-session.test.ts
import { describe, expect, it } from "vitest";
import * as mockSession from "./mock-session";

describe("mock-session exports", () => {
	it("no longer exports DIMENSION_COLORS", () => {
		expect("DIMENSION_COLORS" in mockSession).toBe(false);
	});

	it("still exports DIMENSION_LABELS", () => {
		expect(mockSession.DIMENSION_LABELS.dynamics).toBe("Dynamics");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/lib/mock-session.test.ts
```
Expected: FAIL — `"DIMENSION_COLORS" in mockSession` is `true`.

- [ ] **Step 3: Implement**

Delete lines 18–25 of `src/lib/mock-session.ts` (the `DIMENSION_COLORS`
export block) in full:

```typescript
export const DIMENSION_COLORS: Record<Dimension, string> = {
	dynamics: "#b0816a",
	timing: "#9a8a7a",
	pedaling: "#7a8a9a",
	articulation: "#9a7a8a",
	phrasing: "#8a9a7a",
	interpretation: "#9a917a",
};
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/lib/mock-session.test.ts
```
Expected: PASS (2 tests). This will also make every remaining consumer of
`DIMENSION_COLORS` fail to typecheck — that's expected; Tasks C.2–C.7 fix
each consumer in this same task group.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/mock-session.ts apps/web/src/lib/mock-session.test.ts
git commit -m "refactor(#156): delete dead DIMENSION_COLORS from mock-session"
```

---

### Task C.2: `BarScoreChip.tsx` — shared dimension colors + rename
**Group:** C (parallel)

**Behavior being verified:** the chip's per-dimension bar color reads from
the shared `DIMENSION_COLOR_VAR` map (muted family), not the divergent bright
palette that used to live in this file.
**Interface under test:** rendered `style.backgroundColor` of each bar div

**Files:**
- Modify: `apps/web/src/components/BarScoreChip.tsx`
- Test: `apps/web/src/components/BarScoreChip.test.tsx` (extend if it exists,
  otherwise check `grep -n "BarScoreChip" apps/web/src/components/*.test.tsx`
  first — do not create a duplicate test file)

- [ ] **Step 1: Write the failing test** (append to the existing test file if
found; otherwise create `BarScoreChip.test.tsx` with just this case)

```typescript
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { BarScoreChip } from "./BarScoreChip";

describe("BarScoreChip dimension colors", () => {
	it("colors each bar from the shared DIMENSION_COLOR_VAR map, not a local bright palette", () => {
		const scores = {
			dynamics: 0.5,
			timing: 0.5,
			pedaling: 0.5,
			articulation: 0.5,
			phrasing: 0.5,
			interpretation: 0.5,
		};
		const { container } = render(
			<BarScoreChip scores={scores} barNumber={1} onClose={vi.fn()} />,
		);
		const bars = container.querySelectorAll("[title^='dynamics']");
		expect(bars[0]).toHaveStyle({ backgroundColor: "var(--dim-dynamics)" });
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/BarScoreChip.test.tsx
```
Expected: FAIL — style resolves to `#4f9cf9` (the old bright palette), not
`var(--dim-dynamics)`.

- [ ] **Step 3: Implement**

Delete the local `DIMENSION_COLOR` constant (lines 14–21):

```typescript
const DIMENSION_COLOR: Record<keyof BarQualityScores, string> = {
	dynamics: "#4f9cf9",
	timing: "#f97316",
	pedaling: "#a78bfa",
	articulation: "#34d399",
	phrasing: "#fb7185",
	interpretation: "#fbbf24",
};
```

Add the import and change the one call site (line 73):

```typescript
import { DIMENSION_COLOR_VAR } from "../lib/dimension-colors";
// ...
backgroundColor: DIMENSION_COLOR_VAR[dim],
```

Rename any Tailwind utility classes in this file per the Rename Mapping table
(`bg-espresso` → `bg-surface-page`, `border-border` → `border-border-subtle`,
`text-text-tertiary` → `text-ink-tertiary`, etc — run
`grep -noE "[a-z:./-]*-(espresso|border|text-tertiary)[a-z0-9/]*" apps/web/src/components/BarScoreChip.tsx`
first to confirm the exact sites before editing).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/BarScoreChip.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/BarScoreChip.tsx apps/web/src/components/BarScoreChip.test.tsx
git commit -m "fix(#156): BarScoreChip uses shared dimension color map, not a divergent bright palette"
```

---

### Task C.3: `ScoreAnnotation.tsx` — shared dimension colors + rename
**Group:** C (parallel)

**Behavior being verified:** the annotation's swatch reads `DIMENSION_COLOR_VAR`
instead of the deleted `DIMENSION_COLORS`.

**Files:**
- Modify: `apps/web/src/components/ScoreAnnotation.tsx`
- Test: `apps/web/src/components/ScoreAnnotation.test.tsx` (extend if present)

- [ ] **Step 1: Write the failing test**

```typescript
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ScoreAnnotation } from "./ScoreAnnotation";

describe("ScoreAnnotation dimension color", () => {
	it("reads its swatch color from DIMENSION_COLOR_VAR", () => {
		const { container } = render(
			<ScoreAnnotation dimension="dynamics" text="x" onClick={vi.fn()} />,
		);
		const swatch = container.querySelector("[style]");
		expect(swatch).toHaveStyle({ backgroundColor: "var(--dim-dynamics)" });
	});
});
```
(If `ScoreAnnotation` doesn't render a colored element directly under
`[style]`, adjust the selector to match the actual DOM — read the component
first with `grep -n "color" apps/web/src/components/ScoreAnnotation.tsx`.)

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ScoreAnnotation.test.tsx
```
Expected: FAIL — either a typecheck error (`DIMENSION_COLORS` no longer
exported) surfaces as a test-run failure, or the resolved color is a raw hex.

- [ ] **Step 3: Implement**

```typescript
// was: import { DIMENSION_COLORS, DIMENSION_LABELS } from "../lib/mock-session";
import { DIMENSION_LABELS } from "../lib/mock-session";
import { DIMENSION_COLOR_VAR } from "../lib/dimension-colors";
// ...
// was: DIMENSION_COLORS[dimension as keyof typeof DIMENSION_COLORS] ?? "#7a9a82";
const color =
	DIMENSION_COLOR_VAR[dimension as keyof typeof DIMENSION_COLOR_VAR] ??
	"var(--color-accent)";
```

Rename any Tailwind utility classes in this file per the Rename Mapping table.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/ScoreAnnotation.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ScoreAnnotation.tsx apps/web/src/components/ScoreAnnotation.test.tsx
git commit -m "fix(#156): ScoreAnnotation reads the shared dimension color map"
```

---

### Task C.4: `ScorePanel.tsx` — shared dimension colors + rename
**Group:** C (parallel)

**Behavior being verified:** the observation-chip color reads
`DIMENSION_COLOR_VAR`; `bg-espresso` (×2, lines 224/237) and `bg-border`
(line 250) are rebased to the new tokens.

**Files:**
- Modify: `apps/web/src/components/ScorePanel.tsx`
- Test: `apps/web/src/components/ScorePanel.test.tsx` (extend if present)

- [ ] **Step 1: Write the failing test**

```typescript
// append to existing ScorePanel test suite, or create minimally
it("colors observation chips from the shared dimension map", () => {
	// render with a fixture that has at least one observation, then:
	// expect the chip's inline backgroundColor to be "var(--dim-<x>)"
});
```
Write this against the component's actual observation-rendering fixture —
read `ScorePanel.tsx` lines 170–190 first (`grep -n "observations.map" -A 15 apps/web/src/components/ScorePanel.tsx`)
to match the real prop shape before writing the assertion.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ScorePanel.test.tsx
```
Expected: FAIL — resolves to a raw hex or a typecheck error.

- [ ] **Step 3: Implement**

```typescript
// was: import { DIMENSION_COLORS } from "../lib/mock-session";
import { DIMENSION_COLOR_VAR } from "../lib/dimension-colors";
// ...
// was: DIMENSION_COLORS[obs.dimension as keyof typeof DIMENSION_COLORS] ?? "#7a9a82";
const color =
	DIMENSION_COLOR_VAR[obs.dimension as keyof typeof DIMENSION_COLOR_VAR] ??
	"var(--color-accent)";
```

Rename in this file: `bg-espresso` (×2) → `bg-surface-page`;
`border-border` (line 177/237/415) → `border-border-subtle`; `bg-border`
(line 250) → `bg-border-subtle`. Confirm exact sites first:

```bash
grep -noE "[a-z:./-]*-(espresso|border)[a-z0-9/]*" apps/web/src/components/ScorePanel.tsx
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/ScorePanel.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ScorePanel.tsx apps/web/src/components/ScorePanel.test.tsx
git commit -m "fix(#156): ScorePanel reads shared dimension colors, rebase surface/border tokens"
```

---

### Task C.5: `PlayPassageCard.tsx` — dimension colors + score-canvas + rename
**Group:** C (parallel)

**Behavior being verified:** the clip container uses
`var(--color-score-canvas)` instead of hard-coded `"white"`; dimension color
reads the shared map.

**Files:**
- Modify: `apps/web/src/components/cards/PlayPassageCard.tsx`
- Test: `apps/web/src/components/cards/PlayPassageCard.test.tsx` (extend if
  present)

- [ ] **Step 1: Write the failing test**

```typescript
it("uses the score-canvas token instead of a hard-coded white background", () => {
	// render with a ready loadState + manifest fixture, then:
	const container = /* the div wrapping ClipSvg, per PlayPassageCard.tsx:130-141 */;
	expect(container).toHaveStyle({ backgroundColor: "var(--color-score-canvas)" });
});
```
Match this to the component's real fixture props (`config`, `manifest`,
`clipSvg`, `loadState`) — read the existing test file's setup first.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/cards/PlayPassageCard.test.tsx
```
Expected: FAIL — resolves to `"white"`.

- [ ] **Step 3: Implement**

Line 136: `backgroundColor: "white",` → `backgroundColor: "var(--color-score-canvas)",`

```typescript
// was: import { DIMENSION_COLORS } from "../../lib/mock-session";
import { DIMENSION_COLOR_VAR } from "../../lib/dimension-colors";
// was: DIMENSION_COLORS[config.dimension as keyof typeof DIMENSION_COLORS] ?? "#7a9a82";
const color =
	DIMENSION_COLOR_VAR[config.dimension as keyof typeof DIMENSION_COLOR_VAR] ??
	"var(--color-accent)";
```

Rename: `border-text-tertiary/50` (line 124) → `border-ink-tertiary/50`;
sweep any remaining `border-border`/`text-text-tertiary` in this file per the
mapping table (confirm sites with
`grep -noE "[a-z:./-]*-(border|text-tertiary|text-primary|text-secondary)[a-z0-9/]*" apps/web/src/components/cards/PlayPassageCard.tsx`
first).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/cards/PlayPassageCard.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/cards/PlayPassageCard.tsx apps/web/src/components/cards/PlayPassageCard.test.tsx
git commit -m "fix(#156): PlayPassageCard uses score-canvas token, shared dimension colors"
```

---

### Task C.6: `SessionDataCard.tsx` — shared dimension colors + rename
**Group:** C (parallel)

**Behavior being verified:** dimension color reads `DIMENSION_COLOR_VAR`.

**Files:**
- Modify: `apps/web/src/components/cards/SessionDataCard.tsx`
- Test: `apps/web/src/components/cards/SessionDataCard.test.tsx` (extend if
  present)

- [ ] **Step 1: Write the failing test**

Read the component's actual `DIMENSION_COLORS` call site first
(`grep -n "DIMENSION_COLORS" -B3 -A3 apps/web/src/components/cards/SessionDataCard.tsx`)
and write an assertion in the same shape as Task C.3/C.4 against the real
rendered element.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/cards/SessionDataCard.test.tsx
```
Expected: FAIL

- [ ] **Step 3: Implement**

```typescript
// was: import { DIMENSION_COLORS } from "../../lib/mock-session";
import { DIMENSION_COLOR_VAR } from "../../lib/dimension-colors";
```
and swap the lookup the same way as Task C.3–C.5 (`DIMENSION_COLOR_VAR[key] ?? "var(--color-accent)"`).

Rename `text-text-primary`/`border-border`/etc in this file per the mapping
table — confirm sites first with
`grep -noE "[a-z:./-]*-(text-primary|text-secondary|border|surface)[a-z0-9/]*" apps/web/src/components/cards/SessionDataCard.tsx`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/cards/SessionDataCard.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/cards/SessionDataCard.tsx apps/web/src/components/cards/SessionDataCard.test.tsx
git commit -m "fix(#156): SessionDataCard reads shared dimension colors"
```

---

### Task C.7: `ScoreHighlightCard.tsx` — dimension colors + score-canvas + rename
**Group:** C (parallel)

**Behavior being verified:** both highlight containers use
`var(--color-score-canvas)` instead of `"white"`; dimension color reads the
shared map at all three call sites (lines 75, 135, and the earlier one at 74).

**Files:**
- Modify: `apps/web/src/components/cards/ScoreHighlightCard.tsx`
- Test: `apps/web/src/components/cards/ScoreHighlightCard.test.tsx` (extend
  if present)

- [ ] **Step 1: Write the failing test**

```typescript
it("uses the score-canvas token instead of a hard-coded white background", () => {
	// render with a clips fixture, then inspect the container div at
	// ScoreHighlightCard.tsx:79-90
	const container = /* ... */;
	expect(container).toHaveStyle({ backgroundColor: "var(--color-score-canvas)" });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/cards/ScoreHighlightCard.test.tsx
```
Expected: FAIL — resolves to `"white"`.

- [ ] **Step 3: Implement**

Both `backgroundColor: "white",` sites (line 85 and its `config.highlights`
counterpart further down) → `backgroundColor: "var(--color-score-canvas)",`.

```typescript
// was: import { DIMENSION_COLORS } from "../../lib/mock-session";
import { DIMENSION_COLOR_VAR } from "../../lib/dimension-colors";
```
Swap all three `DIMENSION_COLORS[...] ?? "#7a9a82"` lookups to
`DIMENSION_COLOR_VAR[...] ?? "var(--color-accent)"`.

Rename `border-border` (lines 64, 108) → `border-border-subtle`; `text-text-primary`
etc per the mapping table — confirm with
`grep -noE "[a-z:./-]*-(border|text-primary|text-secondary|text-tertiary)[a-z0-9/]*" apps/web/src/components/cards/ScoreHighlightCard.tsx`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/cards/ScoreHighlightCard.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/cards/ScoreHighlightCard.tsx apps/web/src/components/cards/ScoreHighlightCard.test.tsx
git commit -m "fix(#156): ScoreHighlightCard uses score-canvas token, shared dimension colors"
```

---

### Task C.8: `AudioWaveformRing.tsx` — drop `SAGE_R/G/B`, read the accent token
**Group:** C (parallel)

**Behavior being verified:** the ring's stroke color is derived from
`getComputedStyle(...).getPropertyValue("--color-accent")` at draw time
(so it repaints correctly across a theme change), not from module-level
constants frozen to the old dark-theme sage value.

**Files:**
- Modify: `apps/web/src/components/AudioWaveformRing.tsx`
- Test: `apps/web/src/components/AudioWaveformRing.test.tsx` (extend if
  present; check first)

- [ ] **Step 1: Write the failing test**

```typescript
it("reads its stroke color from the --color-accent CSS variable, not a hard-coded sage constant", () => {
	document.documentElement.style.setProperty("--color-accent", "#123456");
	// render/trigger the draw path per the component's existing test setup
	// (this file animates via requestAnimationFrame — follow the existing
	// test file's pattern for flushing a frame, e.g. vi.useFakeTimers +
	// advancing, or a direct call to the exported draw function if one exists)
	// assert the resulting ctx.strokeStyle call used rgb(18, 52, 86, ...) —
	// i.e. derived from #123456 — not rgba(122, 154, 130, ...).
});
```
Match this to the component's actual test harness (canvas mocking pattern) —
read the existing `AudioWaveformRing.test.tsx` setup before writing new
assertions; do not invent a canvas mock that conflicts with it.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/AudioWaveformRing.test.tsx
```
Expected: FAIL — strokeStyle still resolves to `rgba(122, 154, 130, ...)`.

- [ ] **Step 3: Implement**

Delete the constants (lines 10–13):

```typescript
// Sage green
const SAGE_R = 122;
const SAGE_G = 154;
const SAGE_B = 130;
```

Add a small helper near the top of the file and use it at the draw call
site (line 206):

```typescript
function readAccentRgb(): [number, number, number] {
	const hex = getComputedStyle(document.documentElement)
		.getPropertyValue("--color-accent")
		.trim();
	const clean = hex.replace("#", "");
	return [
		Number.parseInt(clean.slice(0, 2), 16),
		Number.parseInt(clean.slice(2, 4), 16),
		Number.parseInt(clean.slice(4, 6), 16),
	];
}
```

```typescript
// was: ctx.strokeStyle = `rgba(${SAGE_R}, ${SAGE_G}, ${SAGE_B}, ${opacity})`;
const [accentR, accentG, accentB] = readAccentRgb();
ctx.strokeStyle = `rgba(${accentR}, ${accentG}, ${accentB}, ${opacity})`;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/AudioWaveformRing.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/AudioWaveformRing.tsx apps/web/src/components/AudioWaveformRing.test.tsx
git commit -m "fix(#156): AudioWaveformRing reads --color-accent at draw time instead of a frozen sage constant"
```

---

### Task C.9: `ListeningMode.tsx` — shadow copy, red/amber, accent-lighter, rename
**Group:** C (parallel)

**Behavior being verified:** the edge-ring border/glow use `color-mix()` on
`var(--color-accent)` instead of hard-coded sage `rgba()`; the three
red/amber utility sites map to `danger`/`warn`; the two `accent-lighter`
hover sites map to `accent/70`.

**Files:**
- Modify: `apps/web/src/components/ListeningMode.tsx`
- Test: `apps/web/src/components/ListeningMode.test.tsx` (extend if present)

- [ ] **Step 1: Write the failing test**

```typescript
it("derives the edge-ring border/glow from the accent token via color-mix, not a hard-coded sage rgba", () => {
	// render into listening mode per the existing test's activation pattern
	const ring = /* the [key="edge-ring"] motion.div, per ListeningMode.tsx:137-149 */;
	expect(ring).toHaveStyle({
		border: "2px solid color-mix(in srgb, var(--color-accent) 70%, transparent)",
	});
});
```
Match the selector/activation to the existing test file's pattern for
entering listening mode.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ListeningMode.test.tsx
```
Expected: FAIL — resolves to `rgba(122, 154, 130, 0.7)`.

- [ ] **Step 3: Implement**

Lines 145/147:

```typescript
// was: border: "2px solid rgba(122, 154, 130, 0.7)",
border: "2px solid color-mix(in srgb, var(--color-accent) 70%, transparent)",
// was: boxShadow: "0 0 20px rgba(122, 154, 130, 0.3), inset 0 0 20px rgba(122, 154, 130, 0.1)",
boxShadow:
	"0 0 20px color-mix(in srgb, var(--color-accent) 30%, transparent), inset 0 0 20px color-mix(in srgb, var(--color-accent) 10%, transparent)",
```

Red/amber sites (per the Rename Mapping table):
- Line 286: `text-amber-400` → `text-warn`
- Line 319: `bg-red-600 hover:bg-red-500 text-on-accent` → `bg-danger hover:bg-danger/85 text-on-accent`
- Line 486: `bg-red-600 text-on-accent hover:bg-red-500` → `bg-danger text-on-accent hover:bg-danger/85`

`accent-lighter` sites (per the per-site table above):
- Line 256: `hover:text-accent-lighter` → `hover:text-accent/70`
- Line 389: `hover:text-accent-lighter` → `hover:text-accent/70`

`bg-espresso` sites (lines 176, 376) → `bg-surface-page`.

Sweep any remaining `border-border` in this file → `border-border-subtle`
(confirm with `grep -noE "[a-z:./-]*-(border|espresso)[a-z0-9/]*" apps/web/src/components/ListeningMode.tsx`).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/ListeningMode.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ListeningMode.tsx apps/web/src/components/ListeningMode.test.tsx
git commit -m "fix(#156): ListeningMode uses accent color-mix, danger/warn tokens, rebased surfaces"
```

---

### Task C.10: `routes/index.tsx` — shadow copy, danger, hero CTA inversion, rename
**Group:** C (parallel)

**Behavior being verified:** the hero image gradient reads
`var(--color-surface-page)` instead of the hard-coded `#2D2926`; the error
text uses `text-danger`; the hero CTA button's `bg-cream text-espresso` pair
becomes `bg-ink-primary text-surface-page` (see the mapping table's rationale
— this preserves the light-chip/dark-text look once landing forces
`data-theme="dark"` in Task D.1).

**Files:**
- Modify: `apps/web/src/routes/index.tsx`
- Test: `apps/web/src/routes/index.test.tsx` (create if none exists — check
  `ls apps/web/src/routes/*.test.tsx` first)

- [ ] **Step 1: Write the failing test**

```typescript
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
// follow this file's existing route-test rendering pattern if one exists
// elsewhere in src/routes/*.test.tsx (e.g. wrapping in the router's test
// harness); otherwise render the exported route component directly.

describe("landing hero", () => {
	it("uses the surface-page token in the image gradient overlay, not a hard-coded hex", () => {
		render(/* the hero component */);
		const overlay = screen.getByTestId("hero-gradient-overlay"); // add this
		// testid to the div at index.tsx:60-65 if it doesn't have one — a
		// pure-CSS-value assertion needs a stable selector
		expect(overlay).toHaveStyle({
			background: expect.stringContaining("var(--color-surface-page)"),
		});
	});
});
```
If adding a `data-testid` to a production element to make this assertable is
undesirable, use `container.querySelector` scoped to the gradient div's
existing distinguishing className instead — check the file's real structure
before finalizing the selector.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/routes/index.test.tsx
```
Expected: FAIL — style contains `#2D2926`, not `var(--color-surface-page)`.

- [ ] **Step 3: Implement**

Line 64:

```typescript
// was:
// "linear-gradient(to top, #2D2926 0%, #2D2926 5%, rgba(45,41,38,0.7) 30%, rgba(45,41,38,0.2) 60%, rgba(45,41,38,0.05) 100%)"
"linear-gradient(to top, var(--color-surface-page) 0%, var(--color-surface-page) 5%, color-mix(in srgb, var(--color-surface-page) 70%, transparent) 30%, color-mix(in srgb, var(--color-surface-page) 20%, transparent) 60%, color-mix(in srgb, var(--color-surface-page) 5%, transparent) 100%)"
```

Line 314: `text-red-400` → `text-danger`.

Line 81 (the hero CTA, per the mapping table's special case):

```typescript
// was: className="bg-cream text-espresso px-8 py-3.5 text-body-sm font-medium hover:brightness-110 transition inline-block"
className="bg-ink-primary text-surface-page px-8 py-3.5 text-body-sm font-medium hover:brightness-110 transition inline-block"
```

Sweep remaining tokens in this file per the mapping table (`text-cream` in
the header/footer live in `__root.tsx`, not here — confirm this file's own
remaining sites with
`grep -noE "[a-z:./-]*-(espresso|cream|border|surface|text-primary|text-secondary|text-tertiary)[a-z0-9/]*" apps/web/src/routes/index.tsx`).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/routes/index.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/routes/index.tsx apps/web/src/routes/index.test.tsx
git commit -m "fix(#156): landing hero uses surface-page/danger tokens, inverts CTA pairing for dark landing"
```

---

### Task C.11: `routes/signin.tsx` — shadow copy + danger, rename
**Group:** C (parallel)

**Behavior being verified:** both radial-gradient overlays read
`var(--color-surface-page)` instead of `rgba(45,41,38,...)`; both error-text
sites use `text-danger`.

**Files:**
- Modify: `apps/web/src/routes/signin.tsx`
- Test: `apps/web/src/routes/signin.test.tsx` (extend if present; check first)

- [ ] **Step 1: Write the failing test**

```typescript
it("uses the surface-page token in the radial-gradient overlay, not a hard-coded hex", () => {
	// render the signin route per its existing test setup
	const overlays = /* both divs at signin.tsx:60-68 and :328-336 */;
	for (const overlay of overlays) {
		expect(overlay).toHaveStyle({
			background: expect.stringContaining("var(--color-surface-page)"),
		});
	}
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/routes/signin.test.tsx
```
Expected: FAIL — style contains `rgba(45,41,38,...)`.

- [ ] **Step 3: Implement**

Both sites (lines 66 and 334), identical replacement:

```typescript
// was:
// "radial-gradient(ellipse at center, rgba(45,41,38,0.4) 0%, rgba(45,41,38,0.85) 100%)"
"radial-gradient(ellipse at center, color-mix(in srgb, var(--color-surface-page) 40%, transparent) 0%, color-mix(in srgb, var(--color-surface-page) 85%, transparent) 100%)"
```

Lines 143, 351: `text-red-400` → `text-danger`.

Sweep remaining tokens per the mapping table — confirm with
`grep -noE "[a-z:./-]*-(espresso|cream|border|surface|text-primary|text-secondary|text-tertiary)[a-z0-9/]*" apps/web/src/routes/signin.tsx`.
Leave the Google-logo SVG fills and white/black OAuth button colors
untouched — they are explicitly exempt per the spec.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/routes/signin.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/routes/signin.tsx apps/web/src/routes/signin.test.tsx
git commit -m "fix(#156): signin overlays use surface-page token, error text uses danger"
```

---

### Task C.12: `ErrorBoundary.tsx` — on-accent + accent-lighter + rename
**Group:** C (parallel)

**Behavior being verified:** the retry button's text color uses
`text-on-accent` (not `text-espresso`); its hover state uses `hover:bg-accent/80`.

**Files:**
- Modify: `apps/web/src/components/ErrorBoundary.tsx`
- Test: `apps/web/src/components/ErrorBoundary.test.tsx` (extend if present)

- [ ] **Step 1: Write the failing test**

```typescript
it("renders the retry button with on-accent text, not the deleted espresso token", () => {
	// trigger the error state per the existing test's pattern
	const button = screen.getByRole("button", { name: /retry|try again/i });
	expect(button.className).toContain("text-on-accent");
	expect(button.className).not.toContain("text-espresso");
});
```
Match the button's actual accessible name — read the component first.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/components/ErrorBoundary.test.tsx
```
Expected: FAIL — className still contains `text-espresso`.

- [ ] **Step 3: Implement**

Line 54:

```typescript
// was: className="px-6 py-2.5 bg-accent hover:bg-accent-lighter text-espresso font-medium rounded-lg transition-colors"
className="px-6 py-2.5 bg-accent hover:bg-accent/80 text-on-accent font-medium rounded-lg transition-colors"
```

Sweep any other tokens in this file per the mapping table.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/components/ErrorBoundary.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ErrorBoundary.tsx apps/web/src/components/ErrorBoundary.test.tsx
git commit -m "fix(#156): ErrorBoundary retry button uses on-accent text token"
```

---

### Task C.13: `MessageContent.tsx` and `app.chats.tsx` — accent-lighter, danger, rename
**Group:** C (parallel)

**Behavior being verified:** `MessageContent.tsx`'s two `accent-lighter`
sites become `accent/70`; `app.chats.tsx`'s red site becomes `text-danger`.

**Files:**
- Modify: `apps/web/src/components/MessageContent.tsx`
- Modify: `apps/web/src/routes/app.chats.tsx`
- Test: extend `MessageContent.test.tsx` and `app.chats.test.tsx` if present,
  otherwise add the minimal assertion inline in each

- [ ] **Step 1: Write the failing tests**

```typescript
// MessageContent.test.tsx
it("uses an opacity-modified accent for inline-code and link emphasis, not accent-lighter", () => {
	const { container } = render(/* inline code + link fixture */);
	expect(container.innerHTML).not.toContain("text-accent-lighter");
	expect(container.innerHTML).toContain("text-accent/70");
});
```
```typescript
// app.chats.test.tsx
it("uses the danger token for the delete-selection action", () => {
	// render with selection.size > 0 per the existing test's fixture
	const action = screen.getByRole("button", { name: /delete/i });
	expect(action.className).toContain("text-danger");
});
```

- [ ] **Step 2: Run tests — verify both FAIL**

```bash
cd apps/web && bun run test src/components/MessageContent.test.tsx src/routes/app.chats.test.tsx
```
Expected: FAIL on both.

- [ ] **Step 3: Implement**

`MessageContent.tsx` lines 22, 57:

```typescript
// was: className="bg-accent/15 px-1.5 py-0.5 rounded text-body-sm text-accent-lighter"
className="bg-accent/15 px-1.5 py-0.5 rounded text-body-sm text-accent/70"
// was: className="text-accent hover:text-accent-lighter underline underline-offset-2 transition-colors"
className="text-accent hover:text-accent/70 underline underline-offset-2 transition-colors"
```

`app.chats.tsx` line 159:

```typescript
// was: `flex items-center gap-1.5 px-3 py-1.5 rounded-md text-body-xs transition ${selected.size > 0 ? "text-red-400 hover:bg-red-400/10" : "text-text-tertiary/30 cursor-default"}`
`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-body-xs transition ${selected.size > 0 ? "text-danger hover:bg-danger/10" : "text-ink-tertiary/30 cursor-default"}`
```
(the `text-text-tertiary/30` → `text-ink-tertiary/30` rename is folded into
the same edit since it's on the same line.)

Also rename `bg-espresso` (line 89 of `app.chats.tsx`) → `bg-surface-page`,
and sweep any other remaining tokens in both files per the mapping table.

- [ ] **Step 4: Run tests — verify both PASS**

```bash
cd apps/web && bun run test src/components/MessageContent.test.tsx src/routes/app.chats.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/MessageContent.tsx apps/web/src/routes/app.chats.tsx
git add apps/web/src/components/MessageContent.test.tsx apps/web/src/routes/app.chats.test.tsx
git commit -m "fix(#156): MessageContent/app.chats use accent opacity and danger tokens"
```

---

# Task Group D — Theme Resolution

### Task D.1: `stores/theme.ts` delegates to `resolveTheme`
**Group:** D (sequential, depends on Task A.1)

**Behavior being verified:** `useThemeStore`'s initial theme comes from
`resolveTheme`, not from `prefers-color-scheme`.

**Files:**
- Modify: `apps/web/src/stores/theme.ts`
- Test: `apps/web/src/stores/theme.test.ts` (new — no test file exists today)

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/stores/theme.test.ts
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("useThemeStore initial theme", () => {
	beforeEach(() => {
		vi.resetModules();
		localStorage.clear();
	});

	afterEach(() => {
		vi.useRealTimers();
	});

	it("initializes from resolveTheme's time-of-day rule, not prefers-color-scheme", async () => {
		vi.useFakeTimers();
		vi.setSystemTime(new Date(2026, 0, 1, 12, 0)); // noon -> light
		window.matchMedia = vi.fn().mockImplementation(() => ({
			matches: true, // system says dark; should be ignored
		})) as unknown as typeof window.matchMedia;

		const { useThemeStore } = await import("./theme");
		expect(useThemeStore.getState().theme).toBe("light");
	});

	it("honors a stored manual override", async () => {
		localStorage.setItem("crescend-theme", "dark");
		const { useThemeStore } = await import("./theme");
		expect(useThemeStore.getState().theme).toBe("dark");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/stores/theme.test.ts
```
Expected: FAIL — the first case gets `"dark"` (from the mocked
`matchMedia`), because `getSystemTheme`/`prefers-color-scheme` is still the
source of truth.

- [ ] **Step 3: Implement**

```typescript
// apps/web/src/stores/theme.ts
import { create } from "zustand";
import { resolveTheme } from "../lib/theme-resolve";

interface ThemeState {
	theme: "light" | "dark";
	toggleTheme: () => void;
}

function readStorage(): string | null {
	if (typeof window === "undefined") return null;
	return localStorage.getItem("crescend-theme");
}

function now(): Date | null {
	return typeof window === "undefined" ? null : new Date();
}

const initial = resolveTheme({ stored: readStorage(), now: now() });

export const useThemeStore = create<ThemeState>((set) => ({
	theme: initial,
	toggleTheme: () =>
		set((s) => {
			const next = s.theme === "dark" ? "light" : "dark";
			localStorage.setItem("crescend-theme", next);
			return { theme: next };
		}),
}));
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/stores/theme.test.ts
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/stores/theme.ts apps/web/src/stores/theme.test.ts
git commit -m "feat(#156): theme store delegates to resolveTheme, drops prefers-color-scheme"
```

---

### Task D.2: `__root.tsx` — flash script, explicit dark for landing, rename
**Group:** D (sequential, depends on Task D.1 for the store's new shape and
Task A.1 for `resolveTheme`'s boundary hours)

**Behavior being verified:** the inline flash script no longer special-cases
`/` and `/signin` as "always absent `data-theme`" — it sets
`data-theme="dark"` explicitly for those paths and otherwise uses
`resolveTheme`'s precedence; `applyTheme` does the same at runtime.

**Files:**
- Modify: `apps/web/src/routes/__root.tsx`
- Test: `apps/web/src/routes/__root.test.tsx` (new — check
  `ls apps/web/src/routes/__root.test.tsx` first; if a root test already
  exists for other reasons, extend it instead of creating a duplicate)

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/routes/__root.test.tsx (or extend the existing one)
import { describe, expect, it } from "vitest";

// applyTheme's logic is inlined in the component today. Extract the pure
// decision into a small testable function as part of Step 3 (see
// `resolveDocumentTheme` below) and test that function directly — do not
// assert on document.documentElement.dataset from inside a full component
// render, which would couple the test to React's effect timing.
import { resolveDocumentTheme } from "./__root";

describe("resolveDocumentTheme", () => {
	it("is always dark on the always-dark marketing routes", () => {
		expect(resolveDocumentTheme({ pathname: "/", storeTheme: "light" })).toBe("dark");
		expect(resolveDocumentTheme({ pathname: "/signin", storeTheme: "light" })).toBe("dark");
	});

	it("follows the store's theme on app routes", () => {
		expect(resolveDocumentTheme({ pathname: "/app", storeTheme: "light" })).toBe("light");
		expect(resolveDocumentTheme({ pathname: "/app", storeTheme: "dark" })).toBe("dark");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test src/routes/__root.test.tsx
```
Expected: FAIL — `resolveDocumentTheme` is not exported (doesn't exist yet).

- [ ] **Step 3: Implement**

Extract the decision `applyTheme`'s body already makes into an exported pure
function, and call it from both the flash script's logic (reimplemented
inline, since the flash script is a string literal that must run before
React) and the runtime `applyTheme`:

```typescript
// exported near the top of __root.tsx, above RootDocument
export function resolveDocumentTheme(input: {
	pathname: string;
	storeTheme: "light" | "dark";
}): "light" | "dark" {
	const isAlwaysDark = input.pathname === "/" || input.pathname === "/signin";
	return isAlwaysDark ? "dark" : input.storeTheme;
}
```

Replace `THEME_FLASH_SCRIPT` (line 50) — it must stay inline (runs before
React, before `resolveDocumentTheme` exists client-side) but its logic now
mirrors the new precedence and the always-dark override, setting the
attribute explicitly both ways instead of only ever deleting it:

```typescript
const THEME_FLASH_SCRIPT = `(function(){var path=location.pathname;if(path==="/"||path==="/signin"){document.documentElement.dataset.theme="dark";return}var p=localStorage.getItem("crescend-theme");var t;if(p==="light"||p==="dark"){t=p}else{var h=new Date().getHours();t=(h>=19||h<7)?"dark":"light"}document.documentElement.dataset.theme=t})();`;
```

Replace the `applyTheme` function inside the `useEffect` (lines 60–71):

```typescript
function applyTheme() {
	const p = pathnameRef.current;
	const t = useThemeStore.getState().theme;
	document.documentElement.dataset.theme = resolveDocumentTheme({
		pathname: p,
		storeTheme: t,
	});
}
```

Rename remaining tokens in this file: `bg-espresso` → `bg-surface-page`;
`text-text-primary` → `text-ink-primary`; `text-cream` (×3: lines 138, 159,
344 per earlier grep — Header logo, Footer logo, Footer nav wraps around) →
`text-ink-primary`; `text-text-secondary` → `text-ink-secondary`;
`text-text-tertiary` → `text-ink-tertiary`. Confirm the exact set with
`grep -noE "[a-z:./-]*-(espresso|cream|text-primary|text-secondary|text-tertiary|border)[a-z0-9/]*" apps/web/src/routes/__root.tsx`
before editing.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test src/routes/__root.test.tsx
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/routes/__root.tsx apps/web/src/routes/__root.test.tsx
git commit -m "feat(#156): root shell sets data-theme explicitly, marketing routes force dark"
```

---

# Task Group E / E2 — Mechanical Rename Sweep

Every task in this group applies **only** the Rename Mapping table above,
mechanically, to one file. None of these files has a dimension-color,
shadow-copy, or role-ambiguous site — that's what makes them eligible for
this group (all such files were pulled into Group C). Each task's "test" is
a grep-based assertion (deterministic, scriptable, matches the mechanical
nature of the change) rather than a rendered-behavior test, since there is no
behavior change — only which CSS custom property a utility class resolves
through.

**Shared verification pattern for every task in this group:**

```bash
# Step 1 (red): assert the OLD token substrings are present
grep -nE -- '-(espresso|cream|surface-card|surface-2|surface|border|accent-lighter|accent-darker|text-primary|text-secondary|text-tertiary)\b|(text|bg|border)-(red|amber)-[0-9]+' <file>
# Expected: at least one match (the file still has old tokens — test "fails" in the sense
# that the invariant "file has zero old-token references" does not hold yet)

# Step 3 (implementation): rename, longest/most-specific token first to avoid
# partial-match corruption, using word-boundary-safe patterns:
perl -pi -e '
  s/\bsurface-card\b/surface-raised/g;
  s/\bsurface-2\b/surface-sunken/g;
  s/\btext-primary\b/ink-primary/g;
  s/\btext-secondary\b/ink-secondary/g;
  s/\btext-tertiary\b/ink-tertiary/g;
  s/\b(bg|text|border|ring|divide|from|to|via|outline|decoration|placeholder)-surface\b/$1-surface-raised/g;
  s/\b(bg|text|border|ring|divide|from|to|via|outline|decoration|placeholder)-border\b/$1-border-subtle/g;
  s/\b(bg|text|border|ring|divide|from|to|via|outline|decoration|placeholder)-espresso\b/$1-surface-page/g;
  s/\b(bg|text|border|ring|divide|from|to|via|outline|decoration|placeholder)-cream\b/$1-ink-primary/g;
  s/\bred-[0-9]+\b/danger/g;
  s/\bamber-[0-9]+\b/warn/g;
' <file>

# Step 4 (green): assert zero OLD token substrings remain
grep -nE -- '-(espresso|cream|surface-card|surface-2|surface|border|accent-lighter|accent-darker|text-primary|text-secondary|text-tertiary)\b|(text|bg|border)-(red|amber)-[0-9]+' <file>
# Expected: no matches
```

Run `bun run typecheck` and `bun run lint` after each file's rename — both
must stay green (renamed classes are still valid Tailwind syntax since the
new tokens exist in `app.css` as of Task B.1; typecheck only catches TSX
syntax errors from the edit, not missing CSS).

### Task E.1: `AppChat.tsx`
**Group:** E (parallel with E.2–E.4)
Files: Modify `apps/web/src/components/AppChat.tsx`.
Confirmed sites: `bg-espresso` (line 741), `border-border` (×4: lines 741,
943, 957, 1013). Apply the shared pattern above.
- [ ] Step 1: `grep -nE -- '-(espresso|border)\b' apps/web/src/components/AppChat.tsx` — expect 5 matches.
- [ ] Step 2: n/a (grep-based check, no separate "run and observe fail" step beyond Step 1's match count).
- [ ] Step 3: Apply the shared perl command to this file.
- [ ] Step 4: `grep -nE -- '-(espresso|border)\b' apps/web/src/components/AppChat.tsx` — expect 0 matches. Then `cd apps/web && bun run typecheck && bun run lint`.
- [ ] Step 5: `git add apps/web/src/components/AppChat.tsx && git commit -m "chore(#156): rename design tokens in AppChat.tsx"`

### Task E.2: `ArtifactOverlay.tsx`
**Group:** E (parallel)
Files: Modify `apps/web/src/components/ArtifactOverlay.tsx`.
Confirmed site: `border-border` (line 96).
- [ ] Step 1: `grep -nE -- '-border\b' apps/web/src/components/ArtifactOverlay.tsx` — expect 1 match.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: `grep -nE -- '-border\b' apps/web/src/components/ArtifactOverlay.tsx` — expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/ArtifactOverlay.tsx && git commit -m "chore(#156): rename design tokens in ArtifactOverlay.tsx"`

### Task E.3: `cards/CollapsedPreview.tsx`
**Group:** E (parallel)
Files: Modify `apps/web/src/components/cards/CollapsedPreview.tsx`.
Confirmed sites: `border-border` (line 30), `text-text-tertiary`.
- [ ] Step 1: `grep -nE -- '-(border|text-tertiary)\b' apps/web/src/components/cards/CollapsedPreview.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run Step 1's grep, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/cards/CollapsedPreview.tsx && git commit -m "chore(#156): rename design tokens in CollapsedPreview.tsx"`

### Task E.4: `cards/ExerciseSetCard.tsx`
**Group:** E (parallel)
Files: Modify `apps/web/src/components/cards/ExerciseSetCard.tsx`.
Confirmed sites: `border-border` (×4: lines 86, 90, 249, 254, 297).
- [ ] Step 1: `grep -nE -- '-border\b' apps/web/src/components/cards/ExerciseSetCard.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run Step 1's grep, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/cards/ExerciseSetCard.tsx && git commit -m "chore(#156): rename design tokens in ExerciseSetCard.tsx"`

### Task E.5: `cards/ExerciseSetExpanded.tsx`
**Group:** E2 (parallel with E.6–E.8; independent files from E.1–E.4/C/D)
Files: Modify `apps/web/src/components/cards/ExerciseSetExpanded.tsx`.
Confirmed sites: `border-border` (×2: lines 156, 230), `border-red-500
text-red-400 hover:bg-red-500/10` (line 26).
- [ ] Step 1: `grep -nE -- '-border\b|-(red|amber)-[0-9]+' apps/web/src/components/cards/ExerciseSetExpanded.tsx`.
- [ ] Step 3: Apply the shared perl command (the `red-[0-9]+`→`danger` rule
covers line 26's three sites in one pass).
- [ ] Step 4: re-run Step 1's grep, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/cards/ExerciseSetExpanded.tsx && git commit -m "chore(#156): rename design tokens in ExerciseSetExpanded.tsx"`

### Task E.6: `cards/KeyboardGuideCard.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/cards/KeyboardGuideCard.tsx`.
Confirmed site: `border-border` (line 16).
- [ ] Step 1: `grep -nE -- '-border\b' apps/web/src/components/cards/KeyboardGuideCard.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/cards/KeyboardGuideCard.tsx && git commit -m "chore(#156): rename design tokens in KeyboardGuideCard.tsx"`

### Task E.7: `cards/PlaceholderCard.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/cards/PlaceholderCard.tsx`.
Confirmed site: `border-border` (line 7).
- [ ] Step 1: `grep -nE -- '-border\b' apps/web/src/components/cards/PlaceholderCard.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/cards/PlaceholderCard.tsx && git commit -m "chore(#156): rename design tokens in PlaceholderCard.tsx"`

### Task E.8: `cards/SegmentLoopArtifact.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/cards/SegmentLoopArtifact.tsx`.
Confirmed sites: `border-border` (×4: lines 20, 30, 39, 110), `text-red-400`
(line 51).
- [ ] Step 1: `grep -nE -- '-border\b|-(red|amber)-[0-9]+' apps/web/src/components/cards/SegmentLoopArtifact.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/cards/SegmentLoopArtifact.tsx && git commit -m "chore(#156): rename design tokens in SegmentLoopArtifact.tsx"`

### Task E.9: `ChatInput.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/ChatInput.tsx`.
Confirmed site: `border-border/` (line 57, with an opacity modifier — the
shared perl pattern's `\b` boundary still matches before the `/`).
- [ ] Step 1: `grep -nE -- '-border\b' apps/web/src/components/ChatInput.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/ChatInput.tsx && git commit -m "chore(#156): rename design tokens in ChatInput.tsx"`

### Task E.10: `ChatMessages.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/ChatMessages.tsx`.
Confirmed sites: `border-border` (×4: lines 128, 134, 142, 281),
`border-red-500 text-red-400 hover:bg-red-500/10` (line 280).
- [ ] Step 1: `grep -nE -- '-border\b|-(red|amber)-[0-9]+' apps/web/src/components/ChatMessages.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/ChatMessages.tsx && git commit -m "chore(#156): rename design tokens in ChatMessages.tsx"`

### Task E.11: `ExerciseProofBlock.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/ExerciseProofBlock.tsx`.
Confirmed site: `text-text-secondary` (or similar `-text-secondary` form —
confirm exact site with grep before editing, since this file wasn't in the
earlier line-level greps).
- [ ] Step 1: `grep -nE -- '-(text-secondary|border|surface)\b' apps/web/src/components/ExerciseProofBlock.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/ExerciseProofBlock.tsx && git commit -m "chore(#156): rename design tokens in ExerciseProofBlock.tsx"`

### Task E.12: `landing/DeviceFrames.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/landing/DeviceFrames.tsx`.
Confirmed sites: `bg-espresso` (×2: lines 32, 48), `text-text-tertiary`
(line 26). Note: `border-b border-white/5` (line 19) is Tailwind's built-in
border-width + arbitrary-opacity-white utility, **not** a `--color-border`
consumer — do not touch it; the shared perl pattern's `\bborder-border\b`-
style anchoring already leaves it alone since it never matches
`(prefix)-border` (it's `border-white`, a different color name).
- [ ] Step 1: `grep -nE -- '-(espresso|text-tertiary)\b' apps/web/src/components/landing/DeviceFrames.tsx` — expect 3.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0; confirm `border-b border-white/5` is still present unchanged. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/landing/DeviceFrames.tsx && git commit -m "chore(#156): rename design tokens in DeviceFrames.tsx"`

### Task E.13: `LoopTransport.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/LoopTransport.tsx`.
Confirmed sites: `border-border` (×2: lines 30, 61), `border-border/` (line
25), `text-amber-400` (line 93).
- [ ] Step 1: `grep -nE -- '-border\b|-amber-[0-9]+' apps/web/src/components/LoopTransport.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/LoopTransport.tsx && git commit -m "chore(#156): rename design tokens in LoopTransport.tsx"`

### Task E.14: `ProofCard.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/ProofCard.tsx`.
Confirmed sites: `border-border` (×2: lines 256, 386).
- [ ] Step 1: `grep -nE -- '-border\b' apps/web/src/components/ProofCard.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/ProofCard.tsx && git commit -m "chore(#156): rename design tokens in ProofCard.tsx"`

### Task E.15: `ReflectionMessage.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/ReflectionMessage.tsx`.
Confirmed sites: `border-border` (line 69), `text-red-400` (line 77).
- [ ] Step 1: `grep -nE -- '-border\b|-red-[0-9]+' apps/web/src/components/ReflectionMessage.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/ReflectionMessage.tsx && git commit -m "chore(#156): rename design tokens in ReflectionMessage.tsx"`

### Task E.16: `Skeleton.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/Skeleton.tsx`.
Confirmed pattern: uses `surface`/`surface-2`-family classes for its shimmer
gradient (confirm exact sites first — this file wasn't in the earlier
line-level greps).
- [ ] Step 1: `grep -nE -- '-(surface|surface-2|surface-card)\b' apps/web/src/components/Skeleton.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/Skeleton.tsx && git commit -m "chore(#156): rename design tokens in Skeleton.tsx"`

### Task E.17: `ToastContainer.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/ToastContainer.tsx`.
Confirmed sites: `border-border"` (line 18), `border-red-500/40` (line 15).
- [ ] Step 1: `grep -nE -- '-border\b|-red-[0-9]+' apps/web/src/components/ToastContainer.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/ToastContainer.tsx && git commit -m "chore(#156): rename design tokens in ToastContainer.tsx"`

### Task E.18: `ToolCallBar.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/components/ToolCallBar.tsx`.
Confirmed sites: `border-red-500/40 bg-red-500/10 text-red-300` (line 67),
`text-red-200/80` (line 74).
- [ ] Step 1: `grep -nE -- '-red-[0-9]+' apps/web/src/components/ToolCallBar.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/components/ToolCallBar.tsx && git commit -m "chore(#156): rename design tokens in ToolCallBar.tsx"`

### Task E.19: `routes/app.sandbox.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/routes/app.sandbox.tsx`.
Confirmed sites: `bg-espresso` (line 1133), `border-border` (×9: lines 583,
590, 597, 713, 787, 966, 989, 1152, 1166, 1190), `bg-border` (line 732).
Leave the dev-only mock SVG fixture's own literal colors untouched — the
spec's exemption is for those literals, not for this file's genuine
Tailwind token-utility classes.
- [ ] Step 1: `grep -nE -- '-(espresso|border)\b' apps/web/src/routes/app.sandbox.tsx`.
- [ ] Step 3: Apply the shared perl command.
- [ ] Step 4: re-run, expect 0. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/routes/app.sandbox.tsx && git commit -m "chore(#156): rename design tokens in app.sandbox.tsx"`

### Task E.20: `routes/privacy.tsx` and `routes/terms.tsx`
**Group:** E2 (parallel)
Files: Modify `apps/web/src/routes/privacy.tsx`, `apps/web/src/routes/terms.tsx`.
Confirmed pattern: both use `text-text-secondary` (confirm exact sites
first).
- [ ] Step 1: `grep -nE -- '-text-secondary\b' apps/web/src/routes/privacy.tsx apps/web/src/routes/terms.tsx`.
- [ ] Step 3: Apply the shared perl command to both files.
- [ ] Step 4: re-run, expect 0 in both. Then typecheck+lint.
- [ ] Step 5: `git add apps/web/src/routes/privacy.tsx apps/web/src/routes/terms.tsx && git commit -m "chore(#156): rename design tokens in privacy.tsx and terms.tsx"`

---

# Task Group F — a11y Harness and Final Gate

### Task F.1: add `@axe-core/playwright` and the `test:a11y` script
**Group:** F (sequential, depends on every prior group — this is the final
end-to-end check across the fully-migrated app)

**Behavior being verified:** `bun run test:a11y` exists and runs Playwright.

**Files:**
- Modify: `apps/web/package.json`

- [ ] **Step 1: Write the failing check**

```bash
cd apps/web && bun run test:a11y
```
Expected: FAIL — `error: Script not found "test:a11y"`.

- [ ] **Step 2: (same as Step 1 for a script-existence check — no separate run/observe split applies)**

- [ ] **Step 3: Implement**

```bash
cd apps/web && bun add -d @axe-core/playwright
```

Add to `package.json` `scripts`:

```json
		"test:a11y": "playwright test --config playwright.a11y.config.ts",
```

- [ ] **Step 4: Run — verify the script now resolves (it will fail for a
different reason: the config file doesn't exist yet — that's Task F.2)**

```bash
cd apps/web && bun run test:a11y
```
Expected: FAIL — `Cannot find configuration file 'playwright.a11y.config.ts'`
(not `Script not found`). That change in failure reason is the pass
condition for this task.

- [ ] **Step 5: Commit**

```bash
git add apps/web/package.json apps/web/bun.lock
git commit -m "chore(#156): add @axe-core/playwright and test:a11y script"
```

---

### Task F.2: `playwright.a11y.config.ts` and `tests/a11y.spec.ts`
**Group:** F (sequential, depends on Task F.1)

**Behavior being verified:** axe's `color-contrast` rule reports zero
violations against the rendered app shell in both `light` and `dark` themes.

**Files:**
- Create: `apps/web/playwright.a11y.config.ts`
- Create: `apps/web/tests/a11y.spec.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/playwright.a11y.config.ts
import { defineConfig } from "@playwright/test";

export default defineConfig({
	testMatch: ["tests/a11y.spec.ts"],
	use: {
		headless: true,
		baseURL: "http://localhost:4173",
	},
	timeout: 60000,
	webServer: {
		command: "bun run build && bunx vite preview --port 4173 --strictPort",
		port: 4173,
		reuseExistingServer: !process.env.CI,
		timeout: 120000,
	},
});
```

```typescript
// apps/web/tests/a11y.spec.ts
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test.describe("color contrast", () => {
	for (const theme of ["light", "dark"] as const) {
		test(`app shell has no color-contrast violations (${theme})`, async ({
			page,
		}) => {
			await page.goto("/signin");
			await page.evaluate((t) => {
				document.documentElement.dataset.theme = t;
			}, theme);

			const results = await new AxeBuilder({ page })
				.withRules(["color-contrast"])
				.exclude("[data-axe-exempt]") // Google logo SVG, OAuth buttons, sandbox mock fixture
				.analyze();

			expect(results.violations).toEqual([]);
		});
	}
});
```

This will need a real reachable route to check — `/signin` is the shell
component reachable without auth, per the File Changes table's own reference
to `signin.tsx`. If `/signin` in this environment requires network calls
that fail in CI/local without a backend, switch the target route to
whichever unauthenticated route actually renders standalone (check with
`bun run dev` and visiting the route manually) before finalizing this file —
do not point the spec at a route that 404s or hangs on a fetch.

Add `data-axe-exempt` attributes to the two exempt regions this plan already
identified: the Google logo SVG / OAuth buttons in `signin.tsx` (from Task
C.11, add the attribute to their wrapping element) and, if `app.sandbox.tsx`
is ever reachable outside dev, its mock SVG fixture.

- [ ] **Step 2: Run test — verify it FAILS or PASSES based on real content**

```bash
cd apps/web && bun run test:a11y
```
Expected outcome depends on what's actually on `/signin` at this point in
the plan (all prior groups are done, so the page should already be
AA-compliant). If it unexpectedly reports violations, that is a **real
finding** — do not suppress it; it means some site wasn't covered by Task
Groups C/D/E and needs a follow-up task before this plan can close. Do not
add it to `data-axe-exempt` to make the test pass — that exemption is
reserved for the three named categories only (brand colors, OAuth buttons,
dev fixture).

- [ ] **Step 3: N/A if Step 2 already passes; otherwise fix the specific
site(s) axe reports, following the same pattern as the relevant Task Group C
task, and re-run.**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test:a11y
```
Expected: PASS, 0 violations, both themes.

- [ ] **Step 5: Commit**

```bash
git add apps/web/playwright.a11y.config.ts apps/web/tests/a11y.spec.ts
git commit -m "test(#156): add Playwright + axe color-contrast gate for both themes"
```

---

### Task F.3: Final repo-wide grep gate
**Group:** F (sequential, depends on Task F.2)

**Behavior being verified:** no source file (outside `.test.` files, which
may reference old names in comments describing the migration) still
references the deleted `espresso`/`cream` tokens.

**Files:** none modified — this is a verification-only task. If it fails, it
means a file was missed by Task Groups C/D/E; fix that file using the same
pattern as its nearest sibling task, then re-run this check. Do not mark this
task done until the grep is clean.

- [ ] **Step 1: Run the gate**

```bash
cd apps/web && grep -rE '\b(espresso|cream)\b' src --include="*.tsx" --include="*.ts" | grep -v '\.test\.'
```
Expected: no output.

- [ ] **Step 2: Run the full test suite and typecheck/lint as the closing gate**

```bash
cd apps/web && bun run test && bun run typecheck && bun run lint && bun run test:a11y
```
Expected: all green.

- [ ] **Step 3–4: N/A (verification-only task, no implementation step).**

- [ ] **Step 5: Commit only if Step 1 required a fix-up; otherwise this task
produces no commit of its own** — the fix-up (if any) is committed as part of
whichever file's task it belongs to, re-opened and amended with a new commit
(never `--amend` an already-pushed task commit; add a follow-up commit
instead):

```bash
git commit -m "fix(#156): close the last espresso/cream reference the rename sweep missed"
```

---

## Plan Self-Review

1. **Spec coverage:**
   - Two-column token table → Task B.1. ✓
   - `.score-container` uses `score-canvas` → Task B.1. ✓
   - Dimension colors single source of truth → Task A.2 + Tasks C.1–C.7. ✓
   - Shadow-copy RGB literals (5 sites) → Tasks C.8, C.9, C.10, C.11. ✓
   - Error/warning tokens → Tasks C.9, C.10, C.11, C.13, E.5, E.8, E.10,
     E.13, E.15, E.17, E.18. ✓
   - Time-aware theme resolution → Tasks A.1, D.1, D.2. ✓
   - Marketing routes explicit dark → Task D.2. ✓
   - `accent-lighter`/`accent-darker` deletion → Tasks C.9, C.12, C.13, B.1. ✓
   - Full rename sweep, ~32-35 files → Tasks C.2–C.13, E.1–E.20. ✓
   - Verification harness fails first, then passes → Tasks 0.3 (red) → B.1 (green). ✓
   - `bun run test:a11y` via Playwright+axe → Tasks F.1, F.2. ✓
   - Final grep gate → Task F.3. ✓
   - iOS drift — explicitly out of scope per spec; no task touches
     `apps/ios`. ✓ (nothing to do)

2. **Placeholder scan:** no "TBD"/"TODO"/"implement later" strings appear in
   any task's Step 3. Every mechanical task (Group E/E2) gives the exact
   `perl` command and exact `grep` verification command rather than a
   description. The two `read the component first` / `confirm sites first`
   instructions in Tasks A.2, C.4, C.6, C.8, C.9, E.11, E.16, E.20 are not
   placeholders — they are exact `grep` commands the subagent runs before
   editing, because those files were not read line-by-line while writing
   this plan (only their family-membership and rough site count were
   confirmed by grep). This is disclosed rather than hidden, consistent with
   "surface confusion, don't hide it."

3. **Type consistency:** `DIMENSION_COLOR_VAR` (Task A.2) is the single name
   used by every consumer task (C.2–C.7) — no task calls it
   `DIMENSION_COLORS_VAR` or similar. `resolveTheme`'s signature
   (`{ stored, now }` → `"light" | "dark"`) is identical across Tasks A.1,
   D.1, and D.2's `resolveDocumentTheme`, which composes it rather than
   reimplementing precedence. `contrastRatio`/`readTokenTable` signatures
   match between Tasks 0.1/0.2 and their use in Task 0.3.

4. **Group correctness:** re-checked file-by-file — no two tasks in the same
   parallel group (0, A, C, D-internal, E, E2) touch the same file. Group C
   and Group E/E2 are disjoint by construction (a file appears in exactly one
   of: the hand-treated list or the mechanical list). Task D.1 and D.2 are
   marked sequential (not parallel) because D.2 depends on D.1's new store
   shape being importable without breaking `__root.tsx`'s existing
   `useThemeStore.getState().theme` call — though they touch different files
   and *could* run in parallel safely, the ordering is kept sequential
   because D.2's flash-script rewrite assumes `resolveTheme`'s boundary hours
   (owned by A.1) are already the values in use, avoiding a race where D.2 is
   drafted against stale assumptions.

5. **Vertical slice check:** every task is one test (or one grep-based
   red/green check for the mechanical tasks, which have no behavior to unit
   test) → one implementation → one commit. Task 0.3 is the one deliberate
   exception the spec itself calls for: it commits in a FAILING state on
   purpose, with Task B.1 as the paired implementation+green commit — this
   is flagged inline in Task 0.3 rather than silently deviating from the
   pattern.

6. **Behavior test check:** every hand-written test (Groups 0, A, C, D)
   asserts through a public interface or rendered DOM/style output — none
   mocks an internal collaborator, calls a private method, or asserts
   "function was called with X." The mechanical tasks (Group E/E2)
   deliberately use grep instead of a behavior test because there is no
   behavior to assert on — a class rename with an unchanged underlying value
   has no observable difference for a component test to catch; the axe run
   (Task F.2) and the token-pair test (Task 0.3, already green after B.1) are
   what actually verify the values are correct. This was a plan-authoring
   decision, not an oversight.
