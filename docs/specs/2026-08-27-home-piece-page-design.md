# Home + Piece Page Design

**Goal:** A student opens `/app` and sees the pieces they have actually
practiced, can add a piece from the score library, and can open any piece to
see its score marked up across sessions -- instead of the single unlabelled
"Start practicing" button that `/app` is today.

**Not in scope:**

- A persisted `marks` table. Marks remain ephemeral WebSocket/DO state; the
  piece page's across-time marks are fixture-driven. Persistence is a
  follow-up issue.
- Real `openMarkCount` on repertoire cards (it needs the mark store above).
  The field exists in the contract and is `null` until that issue lands.
- Bar-anchored marks. Alignment is deferred (`docs/apps/02-pipeline.md#4`);
  live marks carry timestamp anchors.
- Passage-scoped ask threads (#161), baseline surfacing, iOS parity.
- Any change to `ScoreStand`, `ScoreMarkLayer`, `MarkGlyph`, `PieceLessMode`,
  `SessionReview`, or the mark model. They are reused as-is.

---

## Problem

Three concrete gaps, each verifiable in the tree today:

1. **`/app` has no home.** `apps/web/src/routes/app.tsx` renders a centered
   logo and one "Start practicing" button, then mounts `PracticeMode`
   full-screen. There is no repertoire, no piece list, no way to choose what
   you are about to play. Its own docstring says so: *"No piece picker / home
   surface yet -- that is #160."*

2. **The piece ladder's top two rungs are unreachable.** `PracticeMode` takes
   `userPickedPieceId` and `confidentGuess`, and
   `resolvePieceLadderState` ranks them correctly -- but `app.tsx:78-79`
   passes `null` for both, and `usePracticeSession.ts:312` handles
   `piece_identified` by `console.log`-ing free-text composer/title and
   discarding the `pieceId` the event already carries. Every live session is
   therefore pieceless, and `ScoreStand` never renders outside the
   `/practice-preview` dev harness. Recorded as an inbound dependency on this
   issue by #158.

3. **Sessions have no piece identity.** `apps/api/src/db/schema/sessions.ts`
   stores `studentId`, timestamps, six `avg*` dimension columns,
   `observationsJson` and `conversationId` -- but no `piece_id`. `piece_id`
   exists on `segment_loops`, `diagnosis_artifacts`, and `pending_exercises`,
   never on `sessions`. "Which pieces has this student practiced" is currently
   underivable from real data, so a repertoire cannot be rendered at all.

---

## Solution (from the user's perspective)

**Home (`/app`).** A list of pieces the student has practiced, most recent
first. Each card shows composer, title, and one plain-English arc line ("6
sessions - last practiced Tuesday"). Below the list: a search field that
queries the score library by composer or title, and a secondary "Just play"
control that starts a pieceless session. A student with no history sees only
the search field and "Just play".

**Picking a piece.** Tapping a card opens that piece's page. Tapping a search
result starts a practice session with that piece's score already on the stand
-- the ladder's `user-picked` rung, reachable from the live Record flow for
the first time.

**Piece page (`/app/piece/:pieceId`).** The score, with every mark that piece
has accumulated laid over it, faded by lifecycle: `active` marks at full
strength, `improving` at 0.7, `resolved` at 0.4. Below it, the session history
for that piece and any open drills. Tapping a mark expands its evidence, the
same interaction as in review.

**Recording.** A session started from a piece carries that piece's id for its
whole life, so it appears on that piece's card and history the next time home
loads.

---

## Design

### The write path is the load-bearing half

Everything on home is derived from `sessions.piece_id`. Adding the column is
necessary but not sufficient -- something has to write it. `POST
/api/practice/start` gains an optional `pieceId`, validated against the
catalog, and stores it on the row it already inserts
(`apps/api/src/routes/practice.ts:88`). A pieceless session simply omits it
and the column stays `NULL`.

`pieceId` is validated against `pieces` on the way in rather than trusted.
An unknown id is a 404, not a silently stored orphan: a row pointing at a
piece the catalog does not have would produce a repertoire entry with no
composer or title, and the join would have to invent one or drop the row.
Rejecting at the boundary means `buildRepertoire` never has to decide.

### Aggregation is a pure function, not a query

`listRepertoire` does exactly one thing: fetch the student's non-null-piece
sessions joined to `pieces`. `buildRepertoire` does the rest -- group by
piece, count sessions, take the most recent `startedAt`, order by recency.

This split is forced by how this codebase tests routes. `exercises.test.ts`
mocks the *service* with `vi.spyOn`; `sessions.test.ts` hand-builds a
`db.query.sessions.findFirst` stub. Neither technique can stub a Drizzle
`select().from().leftJoin()` chain, so aggregation written inline in the
service would be untestable without a live Postgres. As a pure function over
rows it is testable with real inputs and no mocks at all -- and it is the
correct Ousterhout split independently: the query is shallow, the grouping
and ordering policy is the depth.

### `openMarkCount` is `null`, not `0`

The card design calls for an open mark count, and there is no mark store to
count. `0` would be a lie a student could read as "nothing to work on";
omitting the field would force an API change when the store lands. So the
contract types it `number | null`, the service returns `null`, and
`RepertoireCard` renders the count only when it is non-null.

This follows the project's stated failure-mode principle -- *the system may
fall silent, never guess* (`docs/apps/05-ui-system.md`) -- and it means the
mark-persistence follow-up changes one service line and zero contracts.

### Marks on the piece page come from fixtures, and say so in code

The success criterion is explicit that the piece page shows *"marks aging
across >= 2 sessions of fixture data"*. `piece-page-fixtures.ts` supplies two
dated sessions in which the same three marks progress `active -> improving ->
resolved`. Every anchor is built by `resolveAnchor`, never by a cast: the
`MarkAnchor` brand symbol is unexported precisely so that "a wrong bar number
cannot reach the screen" is a compile-time property, and a fixture that
routed around it with `as MarkAnchor` would be the first hole in that
guarantee.

Fixtures live in their own leaf module with no component imports, matching
the existing `practice-preview-fixtures.ts` (whose comment records why: the
Playwright loader is not Vite and cannot evaluate `config.ts`'s
`import.meta.env.PROD`).

### The `/app` route becomes a real layout

`app.tsx` today is a layout route that renders a component directly and never
renders an `<Outlet/>`; `app.index.tsx` exists only to satisfy route matching
and returns `null`. `/app` therefore has no child routes.

Home and the piece page require children, so the arrangement inverts:

| Route | File | Renders |
|---|---|---|
| `/app` (layout) | `app.tsx` | auth guard + `<Outlet/>` |
| `/app` (index) | `app.index.tsx` | Home |
| `/app/practice` | `app.practice.tsx` | the practice surface, moved verbatim |
| `/app/piece/:pieceId` | `app.piece.$pieceId.tsx` | Piece page |

The practice surface moves **unchanged** except for reading its piece from a
`?pieceId=` search param. A search param, not a route param: a pieceless
session is a permanent first-class state
(`docs/apps/05-ui-system.md#2`), and `/app/practice/none` would be a lie
about the URL space.

The auth `beforeLoad` guard stays on `app.tsx` alone, where it already is, so
it keeps covering every child without being repeated.

### `confidentGuess` flows out of the hook

`usePracticeSession` gains `confidentGuess: ConfidentGuess | null`, set from
the `piece_identified` event -- which already carries `pieceId`, `composer`,
`title`, and `confidence`, exactly `ConfidentGuess`'s four fields. The
console.log is replaced, not supplemented. `app.practice.tsx` passes it
through, discharging the second half of #158's inbound dependency and making
the `confirm-chip` rung reachable live.

The hook does **not** decide whether the guess wins; `resolvePieceLadderState`
already owns that, and duplicating the precedence rule in the hook would give
the codebase two copies of one policy.

---

## Modules

### `buildRepertoire` / `listRepertoire` / `listPieceHistory` (`apps/api/src/services/repertoire.ts`)

- **Interface:**
  `buildRepertoire(rows: RepertoireRow[]): RepertoireEntry[]`,
  `listRepertoire(ctx, studentId): Promise<RepertoireEntry[]>`,
  `listPieceHistory(ctx, studentId, pieceId): Promise<PieceHistory>`
- **Hides:** the sessions-to-pieces join, grouping by piece id, session
  counting, most-recent-session selection, recency ordering, the
  `openMarkCount: null` policy, and exclusion of null-piece sessions.
- **Tested through:** `buildRepertoire` directly, with real row arrays.
- **Depth:** DEEP -- three exported names over a join, a grouping pass, and a
  policy decision.

### `searchPieces` (`apps/web/src/lib/piece-search.ts`)

- **Interface:** `searchPieces(pieces: readonly PieceSummary[], query: string, limit?: number): PieceSummary[]`
- **Hides:** case/diacritic-insensitive normalization, matching across both
  composer and title, prefix-before-substring ranking, the empty-query
  contract (returns nothing rather than everything), and result capping.
- **Tested through:** the function, with real catalog-shaped arrays.
- **Depth:** DEEP -- one call replaces filtering, ranking, and normalization
  at every call site.

### `describeArc` (`apps/web/src/lib/learning-arc.ts`)

- **Interface:** `describeArc(input: { sessionCount: number; lastPracticedAt: string | null; now: Date }): string`
- **Hides:** singular/plural session wording, relative-date phrasing (today /
  yesterday / weekday within a week / absolute date beyond), and the
  never-practiced case.
- **Tested through:** the function, with an injected `now` so the tests are
  not clock-dependent.
- **Depth:** DEEP -- the card carries no date logic at all.

`now` is a parameter, not `new Date()` inside: this repo's memory records that
the user moves timezones and that fixed clock assumptions have bitten before.
An injected clock also makes every relative-date branch testable without fake
timers.

### Leaf components (`RepertoireCard`, `AddPieceSearch`, `PieceSessionHistory`)

- **Interface:** props in, JSX out; no data fetching, no date maths, no
  ranking.
- **Depth:** SHALLOW, deliberately. All three are presentational leaves whose
  logic has been extracted into the three deep modules above. A leaf component
  with an interface as wide as its body is the expected shape; the failure
  mode to avoid is a leaf that *also* formats dates and ranks search results,
  which is what the extraction prevents.

---

## Verification Architecture

- **Canonical success state:**
  1. `GET /api/repertoire` returns one entry per distinct
     `sessions.piece_id` for the authenticated student, ordered by most recent
     session first, each carrying `pieceId`, `composer`, `title`,
     `sessionCount`, `lastPracticedAt`, and `openMarkCount: null`.
  2. `/app` renders exactly one `RepertoireCard` per returned entry.
  3. Choosing a piece navigates to `/app/practice?pieceId=<id>` and
     `PracticeMode` receives that id as `userPickedPieceId`.
  4. `/app/piece/:pieceId` renders marks drawn from two distinct fixture
     sessions, and all three lifecycle states are present in the DOM.

- **Automated check:**
  `cd apps/api && bun run test` and `cd apps/web && bun run test`, plus
  `cd apps/web && bun run test:a11y` as the WEB-A11Y-004 gate (enforced), and
  `bun run lint` / `bun run typecheck` in both apps.
  Point 4 is assertable in jsdom because `MarkGlyph` writes
  `data-lifecycle={mark.lifecycle}` onto the button and the opacity onto
  `data-testid="mark-lifecycle-dot"` -- both queryable without a real browser.
  Colour contrast is *not* checkable here (jsdom skips the axe
  `color-contrast` rule entirely) and is not claimed by these tests.

- **Harness:** **Task Group 0** -- `apps/web/src/lib/piece-page-fixtures.ts`,
  two dated sessions of lifecycle-progressing marks, written and asserted
  before any surface consumes it. It is what makes the success criterion a
  test rather than a judgement call.

- **Contrast gate:** `bun run test:a11y` (real-browser axe) must pass. It is
  the only place `color-contrast` is checkable -- jsdom skips the rule
  entirely, so the jsdom suites above make no contrast claim. The design
  avoids introducing new token pairs specifically so this gate stays green
  without new cases (see Style Guide Conformance, WEB-A11Y-004).

- **Manual verification step** (cannot be automated here): #158's deferred
  live click-through. Start the stack, sign in, pick a piece from home,
  record, and confirm the score stand renders with the picked piece's score
  and that pause-marks land on it. This needs a microphone, a live DO, and the
  Transkun path, none of which exist in the test environment.

---

## File Changes

| File | Change | Type |
|---|---|---|
| `apps/api/src/db/schema/sessions.ts` | add `pieceId` column + `idx_sessions_student_piece` | Modify |
| `apps/api/src/db/migrations/0006_*.sql` | drizzle-kit **generated** (never hand-written) | New |
| `apps/api/src/services/repertoire.ts` | `buildRepertoire`, `listRepertoire`, `listPieceHistory` | New |
| `apps/api/src/services/repertoire.test.ts` | pure-function tests for `buildRepertoire` | New |
| `apps/api/src/routes/repertoire.ts` | `GET /`, `GET /:pieceId` | New |
| `apps/api/src/routes/repertoire.test.ts` | auth gating + body shape | New |
| `apps/api/src/routes/practice.ts` | `POST /start` accepts + persists `pieceId` | Modify |
| `apps/api/src/routes/practice.test.ts` | unknown `pieceId` rejected | Modify |
| `apps/api/src/services/segment-loops.ts` | `listOpenLoopsForPiece`, consumed by `listPieceHistory` (no new route -- open drills ride along on `GET /api/repertoire/:pieceId`) | Modify |
| `apps/api/src/index.ts` | mount `/api/repertoire` | Modify |
| `apps/web/src/lib/piece-search.ts` | `searchPieces` | New |
| `apps/web/src/lib/piece-search.test.ts` | ranking + normalization | New |
| `apps/web/src/lib/learning-arc.ts` | `describeArc` | New |
| `apps/web/src/lib/learning-arc.test.ts` | every relative-date branch | New |
| `apps/web/src/lib/piece-page-fixtures.ts` | 2 sessions of lifecycle-varied marks | New |
| `apps/web/src/lib/piece-page-fixtures.test.ts` | fixture invariants | New |
| `apps/web/src/lib/api.ts` | `api.repertoire.*`, `api.scores.list` | Modify |
| `apps/web/src/hooks/useRepertoire.ts` | `repertoireQueryOptions`, `pieceHistoryQueryOptions` (WEB-QUERY-001) | New |
| `apps/web/src/hooks/usePieceCatalog.ts` | `pieceCatalogQueryOptions` (WEB-QUERY-001) | New |
| `apps/web/src/components/RepertoireCard.tsx` | one card | New |
| `apps/web/src/components/AddPieceSearch.tsx` | library search + pick | New |
| `apps/web/src/components/PieceSessionHistory.tsx` | session list for a piece | New |
| `apps/web/src/routes/app.tsx` | becomes auth guard + `<Outlet/>` | Modify |
| `apps/web/src/routes/app.index.tsx` | Home | Modify |
| `apps/web/src/routes/app.index.test.tsx` | cards render, pick navigates | New |
| `apps/web/src/routes/app.practice.tsx` | practice surface + `?pieceId=` | New |
| `apps/web/src/routes/app.piece.$pieceId.tsx` | piece page | New |
| `apps/web/src/routes/app.piece.test.tsx` | lifecycle fade across 2 sessions | New |
| `apps/web/src/hooks/usePracticeSession.ts` | expose `confidentGuess` | Modify |
| `apps/web/src/hooks/usePracticeSession.piece.test.ts` | `piece_identified` -> guess | New |
| `apps/web/src/routes/practice-preview.tsx` | unchanged; listed to confirm it is NOT touched | — |

---

## Style Guide Conformance

Checked against `apps/web/WEB_STYLE.md` and `apps/api/TS_STYLE.md`, with rule
status read from `docs/standards/rules.json`. Only rules this feature can
actually violate are listed; each states what the implementation must do.

### Enforced (blocking) rules

- **[WEB-QUERY-002]** (MUST, enforced) Server state never enters zustand.
  Repertoire, piece history, and catalog results live in react-query only.
  The single piece of client state this feature has -- the add-piece search
  query string -- is component-local `useState`, and the picked piece lives in
  the URL, not a store.
- **[WEB-STATE-003]** (MUST, enforced) No new `createContext`. None is added;
  every value reaches its consumer by prop or by route param.
- **[WEB-A11Y-001]** (MUST, enforced) Any motion respects
  `prefers-reduced-motion`. **This feature adds no animation at all** -- cards,
  search results, and the piece page are static. If a later change adds
  motion, this rule reactivates.
- **[WEB-A11Y-004]** (MUST, enforced) New colour/token pairs must pass WCAG AA
  in a real browser via `bun run test:a11y`; jsdom axe skips `color-contrast`,
  so a green `bun run test` proves nothing about it.
  **Resolution:** the new surfaces are restricted to token pairs already in
  use on shipped surfaces -- `surface-page`, `surface-raised`, `ink-primary`,
  `ink-secondary`, `ink-tertiary`, `border-subtle`, `accent`/`on-accent`,
  `danger`, exactly as `routes/app.tsx`, `ScoreStand`, and `MarkGlyph` already
  pair them. No new pair is introduced, so the rule's trigger condition is not
  met. `bun run test:a11y` is still run as a release gate (see Verification).
  A task that reaches for a token pairing outside that list must extend
  `tests/a11y.spec.ts` instead of inventing one. `/app` is reachable in the
  a11y preview build (`.env.production` sets `VITE_AUTH_MODE=waitlist`, so
  `app.tsx`'s `beforeLoad` returns early), so extending coverage is possible
  if it becomes necessary.

### Approved (advisory, surfaces as P2)

- **[WEB-QUERY-001]** (MUST) Query keys come from a shared factory, never
  string-literal arrays at call sites. This repo's established form is an
  exported `queryOptions` object whose `.queryKey` every call site references
  -- `authQueryOptions` in `hooks/useAuth.ts`, read by `lib/auth.tsx`,
  `routes/index.tsx`, and `routes/signin.tsx`. This feature follows it exactly
  with a new `hooks/useRepertoire.ts` exporting `repertoireQueryOptions` and
  `pieceHistoryQueryOptions(pieceId)`, plus `pieceCatalogQueryOptions` in
  `hooks/usePieceCatalog.ts`. **No route or component writes a `queryKey`
  literal.**
- **[WEB-STATE-001]** (MUST) No `useEffect` + `useState` to derive what is a
  pure function of in-scope state. `searchPieces(pieces, query)` is called
  **during render**. An effect that mirrors results into state would be the
  textbook violation and is explicitly forbidden by the tasks.
- **[WEB-STATE-002]** (SHOULD) Default is *no* memoization; `useMemo` requires
  a downstream dependency-array consumer or a measured perf need. `searchPieces`
  is therefore called bare, **not** wrapped in `useMemo`. Neither condition
  holds: nothing downstream takes it as a dependency, and the catalog slice is
  small.
- **[WEB-ERR-001]** (MUST) A caught API error re-throws, sets a caller-visible
  error state, or reports to Sentry -- never a silent `[]`. This is already
  the design: `lib/api.ts`'s `request()` captures to Sentry and throws
  `ApiError`, and home renders an explicit error state rather than an empty
  list (Open Questions).
- **[WEB-UI-001]** (MUST) No `any` in props or handlers; concrete DOM/React
  event types.
- **[WEB-UI-002]** (MUST) No Tailwind arbitrary bracket values where a
  `@theme` token exists. The full token set is `surface-page/raised/sunken`,
  `ink-primary/secondary/tertiary`, `border-subtle/strong`, `accent`,
  `on-accent`, `score-canvas`, `danger`, `warn`.
- **[WEB-A11Y-003]** (MUST) Interactive targets have at least a 32px hit area;
  icons inside them are `aria-hidden` and never the sole accessible label.
  Applies to the repertoire card (the whole card is the target), each search
  result, and "Just play".
- **[WEB-COMPONENT-001]** (SHOULD) Components over ~300 non-test lines get
  split. All three new leaves are far below it.
- **[WEB-COMPONENT-002]** (MUST) No prop no current caller passes; no
  prop-count creep past ~6-8. `RepertoireCard` takes 2 props (`entry`,
  `now`), `AddPieceSearch` 1 (`onPick`), `PieceSessionHistory` 1 (`sessions`).

### Proposed (recorded, never reported)

- **[WEB-ROUTE-001]** (SHOULD, *proposed*) Loaders only seed the query cache
  via the same shared query options the component subscribes to. Status is
  `proposed`, so it is not reportable -- but the design complies anyway: no
  data loader is added, and every surface subscribes with `useQuery`.
  `app.tsx`'s existing `beforeLoad` is an auth guard, not a data source, and
  is unchanged.

### API side (`apps/api/TS_STYLE.md`)

- **[TS-HONO-001]** (MUST) Validators and handler chained in one expression.
- **[TS-HONO-002]** (MUST) `.route()` calls chained so `AppType` keeps route
  info -- `/api/repertoire` is appended to the existing chain in `index.ts`,
  not registered separately.
- **[TS-HONO-005]** (MUST) Validation via the existing `lib/validate.ts`
  wrapper, which already returns JSON errors.
- **[TS-ERR-001/002]** (MUST) `services/repertoire.ts` throws `NotFoundError`
  and never imports `HTTPException` or returns status codes.
- **[TS-SVC-001/002]** (MUST) Services take `ServiceContext`, stay stateless.
- **[TS-DB-003]** (MUST) `db/migrations/` is drizzle-kit generated and never
  hand-edited -- the migration is produced by `bun run generate`, and its
  final filename is whatever drizzle-kit assigns.
- **[TS-DB-004]** (SHOULD) Index naming `idx_{table}_{columns}` ->
  `idx_sessions_student_piece`.
- **[TS-FILE-001]** (SHOULD) `kebab-case.ts`; the route module exports a named
  `{ repertoireRoutes }`, not a default.
- **[TS-TYPE-001]** (MUST) No `any`.

---

## Contracts

Exact shapes, so no task has to invent one.

```typescript
// apps/api/src/services/repertoire.ts

/** One joined sessions-to-pieces row. The input to buildRepertoire. */
export interface RepertoireRow {
  pieceId: string;
  composer: string;
  title: string;
  startedAt: Date;
}

export interface RepertoireEntry {
  pieceId: string;
  composer: string;
  title: string;
  sessionCount: number;
  /** ISO 8601. The most recent session's startedAt. */
  lastPracticedAt: string;
  /** null until a persisted mark store exists. Never 0-as-unknown. */
  openMarkCount: number | null;
}

export interface PieceSessionSummary {
  sessionId: string;
  /** ISO 8601 */
  startedAt: string;
  /** ISO 8601, or null for a session that never ended cleanly. */
  endedAt: string | null;
}

export interface OpenDrill {
  id: string;
  barsStart: number;
  barsEnd: number;
  dimension: string | null;
  requiredCorrect: number;
  attemptsCompleted: number;
}

export interface PieceHistory {
  pieceId: string;
  composer: string;
  title: string;
  /** Most recent first. */
  sessions: PieceSessionSummary[];
  openDrills: OpenDrill[];
}
```

```typescript
// apps/web/src/lib/piece-search.ts
// Structurally the subset of GET /api/scores rows this feature reads.
export interface PieceSummary {
  pieceId: string;
  composer: string;
  title: string;
}
```

`GET /api/repertoire` -> `200 { entries: RepertoireEntry[] }`, `401` unauthenticated.
`GET /api/repertoire/:pieceId` -> `200 PieceHistory`, `401` unauthenticated, `404` for a piece not in the catalog.
`POST /api/practice/start` body -> `{ conversationId?: string; pieceId?: string }`, `404` for a `pieceId` not in the catalog.

---

## Open Questions

- **Q:** Should a session started from the piece page's "practice this" also
  land on that piece, or only ones started from home?
  **Default:** both -- the piece page's record control links to
  `/app/practice?pieceId=<id>` exactly as a home card does. One code path, no
  second rule.

- **Q:** What does home show when `GET /api/repertoire` fails?
  **Default:** an explicit error state naming the failure, with the search
  field and "Just play" still usable. No empty-list fallback -- an empty list
  and a failed fetch must not look alike (`apps/CLAUDE.md`: "Do not hide API,
  WebSocket, inference, or persistence failures with client-side fallbacks").

- **Q:** Does the piece page need its own auth check?
  **Default:** no. `app.tsx`'s `beforeLoad` guard covers every `/app` child
  once it becomes a layout; repeating it per-route would be two copies of one
  policy. The API is separately guarded by `requireAuth`.
