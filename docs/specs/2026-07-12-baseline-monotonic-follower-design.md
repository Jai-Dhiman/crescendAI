# Baseline MONOTONIC Follower Design

**Goal:** Given a stream of AMT-transcribed performance notes and a score's note
list, `follow()` produces a deterministic, monotonic-by-construction
perf-note <-> score-note alignment that reproduces the day-0 spike's
symbolic-alignment result (62/82 clean in-position matches, zero teleports)
on the `bach_inv1_chunk0` fixture — replacing the killed audio-chroma
follower (5/82) as the foundation for epic #108.

**Not in scope:**
- Jump-awareness / non-monotonic re-locking after repeats, jumps, restarts (#118).
- HMM-based following (#119).
- Rust/WASM port (#120).
- The live audio -> AMT -> follower pipeline (#112).
- Any changes to the WASM `score-analysis` crate or its chroma-DTW follower (being retired, not touched here).

## Problem

Epic #108's day-0 de-risk spike proved a symbolic (note-sequence) follower
beats audio chroma-DTW by ~12x (62/82 vs 5/82) on a real recording, but the
spike's source code was never committed and is now lost. Only its recovered
acceptance numbers survive (Trackio project `score-follower-pivot`, run
`symbolic-align-derisk-day0`, posted to issue #115's STATE comment). Without
a committed, tested implementation, epic #108's later work (#118 jump-aware,
#119 HMM, #120 WASM port) has no P0 baseline to extend or regression-test
against.

## Solution (from the user's perspective)

There is no end-user-facing behavior yet — this is a model-side research
module. The "user" is the next engineer (future session or #118/#119/#120
work): they get a tested `follow()` function in `model/src/follower_bench/`
that:
- Takes AMT performance notes + score notes and returns a note-level
  alignment, auto-correcting for a global transposition error.
- Is provably monotonic (score position never regresses across chosen
  matches) — this is a documented *limitation*, not a bug: it will not
  re-lock after a repeat/jump/restart, and tests assert that failure
  explicitly so #118's improvement has a baseline to beat.
- Reproduces the recovered day-0 numbers on the committed golden fixture,
  within documented tolerance (see Verification Architecture).

## Design

### Algorithm: continuity-penalized monotonic fitting-DP with inner transpose search

This is a **fitting alignment** (bioinformatics sense: a short pattern
aligned against a long text, with free leading/trailing gaps in the text,
but internal gaps penalized) between the performance note sequence
(pattern, length N) and the score note sequence (text, length M >> N for a
sub-clip of a larger score).

**State:** `B[i][j]` = best cumulative match score for aligning
`perf_notes[:i]` against `score_notes[:j]`, *conditioned on at least one
match having already occurred* among `perf_notes[:i]`. The "nothing matched
yet" state is always 0 (free leading skip of both unmatched perf notes and
unconsumed score notes) and is not tracked in the table — it is the base
case `B[0][*] = 0` used by the first-match transition below.

**Recurrence**, for `i = 1..N`, `j = 1..M` (1-indexed row/col, 0-indexed
note arrays):

```
pitch_ok(i, j, t) := (score_notes[j-1].pitch + t) == amt_notes[i-1].pitch

B[i][j] = max(
    B[i][j-1] - prior.skip_penalty,                 # skip score note j (only defined once B[i][*] has a real value; -inf propagates)
    B[i-1][j],                                       # leave perf note i unmatched, cost 0 (B[0][j] := 0)
    (B[i-1][j-1] if i > 1 else 0.0) + 1.0             # match perf note i to score note j
        if pitch_ok(i, j, t) else -inf,
)
B[i][0] = -inf for i >= 1   # cannot have a match with zero score notes available
```

The final alignment score for a given transpose `t` is `max_j B[N][j]`
(trailing unconsumed score notes are free, matching the fitting-alignment
shape). Backtracking through whichever transition won at each cell recovers
the matched `(perf_index, score_index)` pairs; because both match
transitions (`B[i-1][j-1]`) and skip transitions (`B[i][j-1]`, `B[i-1][j]`)
only ever move `i` and/or `j` forward, the recovered match sequence is
**monotonic non-decreasing in `score_index` by construction** — this is
where "monotonic by construction, not jump-aware" comes from structurally,
not as a post-hoc filter.

`prior.skip_penalty` is the continuity prior: a fixed cost charged per
skipped score note *between two matches already established* (transitions
that move `j` forward without matching, once `B[i][*]` already holds a
real — non-base-case — value). Leading and trailing skips are always free
(fitting alignment); this is deliberate, since the golden fixture's score
(458 notes / 22 bars) vastly exceeds the perf excerpt's true span (~90
notes), and the excerpt does not start at the score's first note. Setting
`skip_penalty = 0.0` (the `NO_PRIOR` constant) disables the prior entirely
and reproduces the day-0 spike's "free wandering" failure mode: the DP is
indifferent between a nearby match and a far-away coincidental exact-pitch
match in repetitive two-voice texture, so it teleports.

**Transpose search:** `follow()` runs the full DP independently for each
candidate `t` in a fixed candidate set, and picks the `t` that yields the
most matches (ties broken by smallest `|t|`, preferring no transposition).
This is a brute-force *outer* search — deliberately dumb and
deterministic, not a smarter pitch-histogram heuristic — because the day-0
spike's finding was specifically that the correct value is at most ±1
semitone and a full small-integer sweep is cheap (at most 5 DP runs on an
82x458 table). The design intentionally does **not** hardcode which sign
(-1 or +1) is correct: issue #115's body says -1, the Trackio log says +1,
and the two are a labeling-convention difference (perf-relative-to-score
vs. score-relative-to-perf) that the empirical winner on the fixture
resolves without the implementation needing to know which convention is
"true" — tests assert `abs(transpose_semitones) == 1`, never a fixed sign.

### Why this counts as "weight-free / deterministic"

There are two free constants: `prior.skip_penalty` (a fixed algorithm
parameter, not learned from data) and the transpose candidate set. Neither
is fit via gradient descent or any training loop; both are simple
integers/floats chosen once and asserted against the fixture. This keeps
the algorithm portable to a future WASM port (#120, out of scope here)
without needing to ship model weights.

### "Teleport" is a diagnostic over the output, not a DP internal

Because matches are monotonic in `score_index` by construction, a
"teleport" cannot mean non-monotonicity (impossible). It means: the gap in
`score_position` between two *consecutive matched notes* is implausibly
large for a continuous performance — i.e., the DP "wandered" forward to a
distant coincidental pitch match instead of the nearby correct one. This is
computed post-hoc from `EstimatedTrajectory.matches` (see Modules below),
not tracked inside the DP, and the "large" threshold is a test-side
grading constant (see Verification Architecture), since it is used only to
grade reproduction of the historical numbers, not as part of `follow()`'s
public contract.

## Modules

### `model/src/follower_bench/score_notes.py` (new)

- **Interface:** `ScoreNote` (frozen dataclass: `pitch: int`,
  `position: float`); `load_golden_fixture_notes(json_path: Path) ->
  tuple[list[PerfNote], list[ScoreNote]]`; `load_score_notes_from_midi(path:
  Path) -> list[ScoreNote]`.
- **Hides:** Two different score-note source formats (a flattened
  bar/tick-based WASM fixture JSON, and a raw score MIDI file via
  partitura) behind one common `ScoreNote` shape. Callers never see
  `onset_tick`, `duration_ticks`, `bar_number`, or partitura's note-array
  column layout.
- **Depth verdict:** DEEP — two structurally different input formats
  collapse to one 2-field record; the parsing complexity (bar iteration,
  tick math, partitura note-array indexing) is real and fully hidden.
- **Tested through:** the loader functions' return values only (note
  counts, boundary pitches/positions), never through inspecting
  intermediate parsing state.

### `model/src/follower_bench/follower.py` (new)

- **Interface:** `ContinuityPrior` (frozen dataclass: `skip_penalty:
  float`), `NO_PRIOR` (module constant, `ContinuityPrior(skip_penalty=0.0)`),
  `MatchedNote` (frozen dataclass: `perf_index: int`, `score_index: int`,
  `perf_time: float`, `score_position: float`), `EstimatedTrajectory`
  (frozen dataclass: `transpose_semitones: int`, `matches:
  tuple[MatchedNote, ...]`, `unmatched_perf_indices: tuple[int, ...]`),
  `follow(amt_notes: list[PerfNote], score_notes: list[ScoreNote], prior:
  ContinuityPrior, transpose_candidates: tuple[int, ...] = (-2, -1, 0, 1,
  2)) -> EstimatedTrajectory`, `teleport_gaps(trajectory:
  EstimatedTrajectory) -> list[float]`.
- **Hides:** The full O(N*M) two-layer DP table, backtracking pointers, and
  the per-transpose-candidate outer loop. Callers never see the DP table,
  the tie-break logic, or the backtrace mechanics.
- **Depth verdict:** DEEP — a ~5-line call signature hides an O(N*M) DP
  with backtracking run up to 5 times per call (once per transpose
  candidate).
- **Tested through:** `follow()`'s return value (`EstimatedTrajectory`) and
  `teleport_gaps()`'s return value only. No test inspects the DP table or
  back-pointers directly.

### Existing modules reused unmodified

- `follower_bench.segments.PerfNote` — reused as-is for `amt_notes`; the
  golden fixture's `perf_notes` JSON keys (`pitch`, `onset`, `offset`,
  `velocity`) already match `PerfNote`'s field names exactly, so no
  adapter is needed on the performance side.
- `follower_bench.trajectory.TrueTrajectory` — reused (not modified) by
  characterization tests to interpolate an `EstimatedTrajectory`'s implied
  score-position-over-time curve, by constructing
  `TrueTrajectory(anchors=tuple((m.perf_time, m.score_position) for m in
  result.matches))` directly in the test. No new interpolation code is
  written for `EstimatedTrajectory`.
- `follower_bench.pathologies.build_plan` / `follower_bench.clip_generator.generate`
  — reused unmodified to generate jump/repeat/restart `SynthClip`s for the
  characterization tests.

## Verification Architecture

- **Canonical success state:** On the committed golden fixture
  (`bach_inv1_chunk0`), `follow()` returns an `EstimatedTrajectory` with
  `abs(transpose_semitones) == 1`, zero teleports (hard assertion, the
  north-star), and a match count in a documented tolerance band around the
  historical 62/82. On jump/repeat/restart `SynthClip`s, `follow()`'s
  implied trajectory measurably diverges from the clip's `true_trajectory`
  shortly after the injected pathology event (documenting the expected
  monotonic-follower failure mode).
- **Automated check:** `cd model && uv run python -m pytest
  tests/follower_bench/ -v` (matches the existing `just test-model` /
  #111 test suite; no new `just` recipe needed).
- **Harness:** buildable and required as Task Group 0 in the plan — the
  golden-fixture loader (`load_golden_fixture_notes`) is a prerequisite for
  every later DP-behavior task and is built and tested first, against the
  real, already-on-disk fixture at
  `apps/api/src/wasm/score-analysis/tests/fixtures/bach_inv1_chunk0.json`
  (verified this session: 82 `perf_notes` with onsets 0.70s-14.92s, 458
  total notes across `score_bars` — an exact match to the day-0 spike's
  "82 perf notes / 458 score notes" numbers). This file is read in place
  via a `__file__`-anchored path (matching `asap_alignment.py`'s
  `DEFAULT_ASAP_ROOT` convention) — it is **not** duplicated into
  `model/`, to avoid two copies of the same fixture drifting apart.

## Historical-number reproduction: exact tolerance is an open design choice

The day-0 spike's source code is lost; its recovered numbers (62/82,
3 teleports, max 6.9s) come from a run of that lost implementation, not
from an external, independently-computable ground truth. This spec commits
to reproducing the two **structural** claims exactly (monotonic
construction; the continuity prior eliminates teleports; the prior's
absence reintroduces at least one teleport) and treats the **exact
counts** as an approximate reproduction target with an explicit tolerance,
consistent with the issue's own hedged phrasing ("~0.756", "must
reproduce" used loosely). See Open Questions.

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/follower_bench/score_notes.py` | `ScoreNote`, `load_golden_fixture_notes`, `load_score_notes_from_midi` | New |
| `model/src/follower_bench/follower.py` | `ContinuityPrior`, `NO_PRIOR`, `MatchedNote`, `EstimatedTrajectory`, `follow`, `teleport_gaps` | New |
| `model/tests/follower_bench/test_score_notes.py` | Tests for both loaders | New |
| `model/tests/follower_bench/test_follower.py` | Unit-level DP behavior tests (basic match, continuity prior effect, transpose search, `teleport_gaps`) on small synthetic examples | New |
| `model/tests/follower_bench/test_follower_golden_fixture.py` | Golden-fixture reproduction test + continuity-prior ablation test | New |
| `model/tests/follower_bench/test_follower_characterization.py` | jump / repeat / restart characterization tests (documents expected failure) | New |

## Open Questions

- Q: What exact `skip_penalty` value should `follow()`'s golden-fixture
  test use?
  Default: introduce `DEFAULT_SKIP_PENALTY = 0.5` as a named constant in
  `follower.py`; the plan's golden-fixture task empirically verifies it
  drives teleports to zero on the real fixture and adjusts the constant
  (documenting the final chosen value in the task's commit message) if
  0.5 does not.
- Q: What tolerance counts as "reproducing" 62/82 clean matches given the
  lost source?
  Default: assert `len(matches)` falls in `[54, 70]` (62 ± 8) on the
  golden fixture — tight enough to catch a broken algorithm, loose enough
  to not require bit-for-bit reproduction of an unrecoverable
  implementation. Document this band's rationale inline in the test.
- Q: What counts as a "teleport" for grading purposes (threshold, in
  seconds, on the consecutive-match `score_position` gap)?
  Default: `TELEPORT_THRESHOLD_S = 2.0`, defined as a test-local constant
  in `test_follower_golden_fixture.py` (not part of `follow()`'s public
  API, since it is a grading concept, not an algorithm parameter).
- Q: What divergence magnitude counts as "fails to re-lock" in the
  characterization tests?
  Default: `DIVERGENCE_THRESHOLD_BEATS = 2.0`, defined as a test-local
  constant in `test_follower_characterization.py`; probe at
  `event.perf_time + 3.0` seconds after the injected pathology event using
  the ASAP piece's own beat scale (matches `TrueTrajectory`'s units).
