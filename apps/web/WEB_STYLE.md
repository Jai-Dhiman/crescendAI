# Web Style Guide

## CrescendAI browser client -- TanStack Start + React

Scope: every TypeScript/TSX file under `apps/web/src/` except generated files
(`routeTree.gen.ts`) and tests. Stack: TanStack Start/Router/Query, zustand,
Tailwind v4 (tokens in `app.css` `@theme`, see `docs/apps/05-ui-system.md`),
motion, Biome.

Normative statements carry a stable id and an RFC 2119 level; the compact
index is `docs/standards/rules.json`. Cite the id when reporting a violation.
UI/UX numbers (durations, hit targets) come from the userinterface-wiki skill.

Governing philosophy: minimum code that solves the problem. Nothing
speculative. Explicit failures over silent fallbacks -- never hide API,
WebSocket, or inference failures behind client-side defaults.

## 1. Errors

- **[WEB-ERR-001]** (MUST) A caught API/WebSocket/inference error re-throws,
  sets a caller-visible error state, or reports to Sentry. Never silently
  return a default (`[]`, `null`) that makes a backend failure look like an
  empty result.
- **[WEB-ERR-002]** (SHOULD) Every `catch` block does one of: Sentry capture,
  explicit UI error state, or a comment naming the specific benign condition
  it swallows (e.g. "SSR: localStorage unavailable").

## 2. Server state (react-query) and client state (zustand)

- **[WEB-QUERY-001]** (MUST) Query keys come from a shared per-domain factory
  (e.g. `conversationKeys.detail(id)`), never string-literal arrays typed
  independently at call sites -- one typo silently breaks invalidation with no
  compile error.
- **[WEB-QUERY-002]** (MUST) Server state (anything from `api.*`, the Hono
  client, or a WebSocket message) is never copied into a zustand store.
  Zustand holds client-only UI state; react-query owns caching and
  invalidation.
- **[WEB-STATE-001]** (MUST) No `useEffect` + `useState` to derive a value
  that is a pure function of props or other in-scope state; compute it during
  render.
- **[WEB-STATE-002]** (SHOULD) `useMemo`/`useCallback` require either a
  downstream dependency-array consumer or a comment citing a measured perf
  need. Default is no memoization.
- **[WEB-STATE-003]** (MUST) A new `createContext` requires a cross-tree
  concern with no natural prop path (the two existing contexts -- auth
  session, artifact scroll -- are the bar). Do not add context to avoid
  passing two props through two components.
- **[WEB-ROUTE-001]** (SHOULD) If route loaders are introduced, a loader only
  prefetches/seeds the query cache via the same shared query options the
  component subscribes to with `useQuery`; `useLoaderData` is never the sole
  data source while react-query is present.

## 3. UI discipline

- **[WEB-UI-001]** (MUST) No `any` in component props or event handlers; use
  the concrete DOM/React event type. `as never` is permitted only as the
  documented Hono RPC escape hatch.
- **[WEB-UI-002]** (MUST) No Tailwind arbitrary bracket values (`w-[42%]`,
  `text-[9px]`, `z-[100]`) where a token equivalent exists in the `@theme`
  table; a genuinely one-off geometry value carries a comment. The token
  table is the single source of the light/dark asymmetry.
- **[WEB-UI-003]** (SHOULD) `style={{...}}` only for values computed at
  runtime (measured positions, dimension colors); static inline styles are
  token bypass.

## 4. Accessibility and motion

- **[WEB-A11Y-001]** (MUST) Every `motion.*` component or CSS animation with
  movement respects `prefers-reduced-motion` (via `useReducedMotion()`,
  `<MotionConfig reducedMotion="user">`, or a CSS media override), preferring
  an opacity fade over removal.
- **[WEB-A11Y-002]** (MUST) Animation durations: press/hover feedback
  120-180ms, small state changes 180-260ms, user-initiated transitions at
  most 300ms; `ease-out` for entrances, `ease-in` for exits, linear only for
  progress.
- **[WEB-A11Y-003]** (MUST) Interactive targets have at least a 32px hit
  area; icons inside them are `aria-hidden` and never the sole accessible
  label.
- **[WEB-A11Y-004]** (MUST) New color/token pairs pass WCAG AA in a real
  browser via `bun run test:a11y` -- jsdom axe skips `color-contrast`, so a
  green unit run proves nothing about contrast.

## 5. Minimum code

- **[WEB-COMPONENT-001]** (SHOULD) A component over ~300 non-test lines is
  split, unless it is one deep module with a single responsibility and a
  top-of-file comment saying so.
- **[WEB-COMPONENT-002]** (MUST) No prop that no current caller passes, and
  no prop-count creep past ~6-8 without grouping; "for future flexibility" is
  not a caller.
