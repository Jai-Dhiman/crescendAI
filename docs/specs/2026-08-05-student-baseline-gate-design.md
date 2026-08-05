# Student Baseline Gate Design

**Goal:** Decide, per dimension, whether a deviation in a student's playing is
worth marking — firing on repeated evidence, staying quiet on single-observation
noise, and retiring symmetrically when the deviation persistently returns to
normal.

**Not in scope:**

- Persistence to D1 / Durable Objects (this module is pure; the caller stores the
  returned state).
- Any HTTP route, service wiring, or pipeline integration.
- Any UI. Mark rendering and the `active | improving | resolved` visual treatment
  belong to #157.
- Replacing `fetch-student-baseline` / `compute-dimension-delta` or migrating
  their callers (follow-up; see Open Questions).
- Writing habit facts to `synthesized_facts`. This module emits the promotion and
  retirement *signal*; the memory service performs the write (follow-up).

---

## Problem

Today, `apps/api/src/harness/skills/atoms/fetch-student-baseline.ts` is the only
baseline logic in the codebase, and it is wrong on three counts:

1. **It returns `null` when fewer than 3 sessions exist.** Every caller must
   branch on "no baseline yet," so cold start and normal operation are two
   separate code paths. Bugs live in the seam.
2. **It is MuQ-based** (`session_means` are documented as "per-session MuQ mean
   scores"). MuQ was removed from the pipeline by the approved #154 design.
3. **It has no gate.** It reports a mean and a standard deviation. Nothing in the
   codebase decides whether a deviation is *worth saying something about*, so
   there is no mechanism behind the `active -> improving -> resolved` lifecycle
   that #157 renders and no mechanism behind fact retirement.

Without this module, the score-first redesign has no way to stay silent. Every
computed deviation would become a mark, which is the failure mode the entire
epic is built to avoid.

---

## Solution (from the user's perspective)

A student practises. Most of what they play sits inside their normal range and
the app says nothing.

- **First session.** The app has no history, but it has many observations from
  that one sitting. If the student blurs the pedal three separate times, that
  repetition is enough evidence and the app says so that night. If they blur it
  once, the app stays quiet — one observation has nothing to be unusual
  relative to.
- **Later sessions.** The student's own history becomes the reference. A single
  off night does not trigger anything. A pattern that holds across three sessions
  does.
- **Getting better.** When the flagged dimension comes back inside the normal
  range and stays there, the mark softens (`improving`), then retires
  (`resolved`). The student is never told to keep working on something they
  have fixed.
- **Habits.** A pattern that recurs across two or more distinct calendar weeks is
  promoted from "something that happened" to "something the student does." A
  promoted habit retires by the same rule that retires the mark — never by an
  LLM's judgement.

---

## Design

### The one mechanism

There is exactly one gate: a per-dimension pair of opposed persistence counters,
`consecutiveOutOfBand` and `consecutiveInBand`. Each folded session increments
exactly one of them and zeroes the other. Firing and retiring are the same
threshold test read in opposite directions — that is what "symmetric retirement"
means concretely.

Evidence arrives from two sources that are **not** two code paths — they are two
levels of the same fold, and both write to `consecutiveOutOfBand`:

| Level | Available from | Contributes |
|---|---|---|
| Within-session: samples far from that session's own centre | session 1 | +1 per deviant sample, capped at `MAX_WITHIN_SESSION_CONTRIBUTION` |
| Across-session: the short EWMA sitting outside the band | session 2+ | +1 |

This is the approved doc's own clause — *"2-3 consecutive sessions **or repeated
within-session evidence**"* — with the within-session arm treated as a first-class
path rather than a footnote.

**Consequence: no cohort priors, no self-reported level, no session counter.**
The reference is always the student's own data. On session 1 that data is
within-session; by session 8 it is mostly across-session. Nothing in the code
branches on how many sessions have been seen. Absence of history is expressed as
a *wide band*, not as a `null` and not as an `if`.

### Divergence from `docs/apps/03-memory-system.md`

The merged doc (#155) specifies *"Cold start (sessions 1-3): cohort priors from
self-reported level ... real gating from ~session 5-8."* This spec **replaces**
that clause, by decision of 2026-08-05:

- No cohort priors and no `level` parameter. No calibration data for per-level
  score distributions exists anywhere in the repo, so any per-level constant
  would be invented and would silently shape a student's first week.
- Gating is live from session 1 via within-session repetition, because a first
  session that says nothing reads as broken.

**Accepted trade-off:** the motor-learning argument in the epic (concurrent and
early feedback harms retention) favours the later gate. Session-1 flags rest on a
thinner spread estimate and will be less reliable than session-10 flags. The
`confidence` field records this so the teacher can frame early marks as
exploratory prose, but confidence **never** gates. `03-memory-system.md` must be
updated to match when this ships.

### Band construction

Per dimension, the band is centred on the long EWMA. Its half-width is:

```
halfWidth = max(noiseFloor, MIN_BAND_SD_FRACTION * sd)
```

- `noiseFloor` — the student's own typical error, tracked as an EWMA of the
  observed within-session spread. **This is what makes session 1 work:** the
  noise floor is measurable from a single sitting, whereas a session-to-session
  variance needs at least two sessions.
- `sd` — the accumulated spread estimate.
- `MIN_BAND_SD_FRACTION * sd` is a floor that stops the band collapsing to zero
  for a very consistent student, matching the approved `0.2 x SD` term.

Early on, few samples means a large observed spread means a wide band means the
gate rarely fires. Evidence narrows the band. That narrowing *is* the cold-start
mechanism, and it is behaviourally testable ("the band narrows monotonically
under consistent evidence"), which is a far stronger assertion than "a null is
not returned."

### Lifecycle state machine

Symmetric, driven by the single persistence counter:

```
absent    --  outOfBand counter >= FIRE_PERSISTENCE      --> active
active    --  inBand counter    >= IMPROVING_PERSISTENCE --> improving
improving --  inBand counter    >= RETIRE_PERSISTENCE    --> resolved
improving --  outOfBand counter >= FIRE_PERSISTENCE      --> active   (reset)
resolved  --  outOfBand counter >= FIRE_PERSISTENCE      --> active   (recurrence)
```

Entering `active` from `resolved` is a recurrence, and recurrence in a distinct
calendar week is what drives promotion.

### Promotion and retirement

- **Promote** to a durable habit when the dimension has recorded out-of-band
  evidence while `active` in at least `PROMOTION_DISTINCT_WEEKS` distinct ISO
  weeks. "Recurs across weeks," per the doc — not one session, not one week.
- **Retire** the promoted habit when the lifecycle reaches `resolved`. Explicit
  and rule-driven. Staleness is never left to LLM discretion — the one documented
  failure mode shared by MemGPT, mem0 and Zep.

Promotion state is carried on the returned state as `promoted: boolean` plus the
set of distinct ISO weeks that supplied evidence; retirement is simply
`lifecycle === "resolved"`, so there is no second retirement field to keep in
sync. This module emits the signal; writing `synthesized_facts` rows is the
memory service's job.

### Purity and serialisability

`updateBaseline` is a pure fold: `(state, session) -> state`. No clock, no
randomness, no I/O — the session carries its own ISO timestamp. `BaselineState`
is plain JSON validated by a Zod schema, so a caller can persist it in D1 or DO
storage without this module ever importing a database. #157 requires lifecycle
transitions to come from server state; this is what makes that possible.

### Explicit failure

Per repo policy, the module throws rather than silently defaulting on: an unknown
dimension, a non-finite score, an unparseable timestamp, or a session whose
timestamp precedes the last folded session. Silence about a broken input would be
indistinguishable from silence about a student playing well — the one confusion
this module exists to prevent.

### Constants

Doc-sourced, in one exported config object, overridable per call:

| Constant | Default | Source |
|---|---|---|
| `SHORT_HALF_LIFE_SESSIONS` | 4 | doc: "~3-5" |
| `LONG_HALF_LIFE_SESSIONS` | 20 | doc: "~15-25" |
| `MIN_BAND_SD_FRACTION` | 0.2 | doc: "0.2 x SD" |
| `FIRE_PERSISTENCE` | 3 | doc: "2-3 consecutive" |
| `IMPROVING_PERSISTENCE` | 2 | intermediate marker |
| `RETIRE_PERSISTENCE` | 3 | symmetric with fire |
| `PROMOTION_DISTINCT_WEEKS` | 2 | doc: "across weeks" |
| `MAX_WITHIN_SESSION_CONTRIBUTION` | 3 | equals `FIRE_PERSISTENCE` |
| `MIN_SAMPLES_FOR_SPREAD` | 3 | arithmetic: spread is undefined below 3 points |
| `DEVIANT_SAMPLE_MULTIPLE` | 1.5 | **uncalibrated** — see Open Questions |

Scale-agnostic: the module accepts any finite score scale. The pipeline emits
`0..1`; nothing in the module assumes it.

---

## Modules

### `apps/api/src/services/student-baseline.ts` — DEEP

- **Interface:** `initialBaselineState()`, `updateBaseline(state, session, config?)`,
  `BaselineStateSchema`, `DEFAULT_BASELINE_CONFIG`, and the `BaselineState` /
  `SessionSamples` types. Two functions and a schema.
- **Hides:** bias-corrected dual EWMA, within-session centre and spread
  estimation, noise-floor tracking, band construction, the two-level evidence
  fold, the persistence counter, the five-transition lifecycle machine, ISO-week
  bucketing for promotion, and input validation.
- **Depth verdict:** DEEP. The interface is a fold plus a schema; the hidden
  surface is the entire state clock. Callers never see an EWMA, a counter, or a
  week bucket.
- **Tested through:** the public interface only — synthetic session sequences fed
  through `updateBaseline`, asserting on the returned state's lifecycle,
  `markWorthy`, `promoted` and band fields. No test reaches an internal helper.

`src/services/` and not `src/lib/`: TS_STYLE §11 reserves `lib/` for "shared
utilities, types, error classes — no business logic," and a feedback gate is
business logic. `src/services/**/*.test.ts` is already matched by both
`vitest.config.ts` (workerd) and `vitest.node.config.ts` (node), so the suite runs
in both runtimes with no config change.

---

## Verification Architecture

- **Canonical success state:** a table-driven synthetic-sequence suite in which
  each row is a scripted sequence of sessions and the expected lifecycle string
  after each one. The four success criteria from the issue map to four rows:
  fires on persistent deviation, quiet on single-observation noise, retires on
  persistent return-to-band, and cold start uses the same call shape as every
  other session.
- **Automated check:** `cd apps/api && bun run test:scripts` and
  `bun run test`, plus `bun run typecheck` and `bun run lint`.
- **Harness:** buildable before the feature — **Task Group 0**. A `runSequence`
  helper folds an array of sessions through `updateBaseline` and returns the
  lifecycle trace. It fails at import (`updateBaseline is not exported`) until the
  module exists, which is the required "harness fails before the feature" proof.

---

## File Changes

| File | Change | Type |
|---|---|---|
| `apps/api/src/services/student-baseline.ts` | The module | New |
| `apps/api/src/services/student-baseline.test.ts` | Sequence harness + behaviour suite | New |

No existing file is modified. `docs/apps/03-memory-system.md` needs the cold-start
clause corrected, but that is a `/ship` documentation step, not an implementation
change.

---

## Open Questions

- **Q: What multiple of the noise floor makes a sample "deviant"?**
  `DEVIANT_SAMPLE_MULTIPLE` is the one constant with no doc or data behind it.
  **Default:** 1.5. Exported and overridable; a tuning issue once real session
  data exists.
- **Q: Should `fetch-student-baseline` and `compute-dimension-delta` be replaced
  by this module?**
  They are MuQ-era and null-returning, so yes eventually — but their callers are
  harness molecules outside this issue's scope.
  **Default:** leave both untouched, open a follow-up issue at `/ship`.
- **Q: Who writes the promoted habit to `synthesized_facts`?**
  **Default:** not this module. It emits `promoted` and `lifecycle` on the state;
  wiring to the memory service is a follow-up on the pipeline issue (#162).
- **Q: Does the epic's motor-learning premise survive session-1 gating?**
  **Default:** ship the product decision (immediate feedback), record the
  divergence in `03-memory-system.md`, and revisit if early marks prove noisy.
