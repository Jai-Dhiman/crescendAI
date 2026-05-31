# Chroma-DTW score follower — eval-driven pivot

**Status:** HARNESS SHIPPED (2026-05-31, merge `3305babf`). Eval harness is on main; `just chroma-eval-verify` and `just chroma-eval-ratchet` are live. Next step: `/autoresearch` loop on continuity-aware improvements. Pre-MEC-2026 papers + bucket-1 amateur experiment surfaced concrete failures and a validated partial fix; eval harness provides the metric system to safely iterate.

**Last updated:** 2026-05-31. Picks up from the autopilot run that halted at the Task 0 gate.

---

## 1. Goal

Harden CrescendAI's live chroma-DTW score follower (`apps/api/src/wasm/score-analysis/src/chroma_dtw.rs`) into a system whose alignment quality we can measure and continually improve. The score follower runs per 15s audio chunk inside the `SessionBrain` Durable Object during a practice session; it is currently gated behind AMT (which will be up locally via `just amt` once we wire this end-to-end before beta).

The deeper goal is to convert future improvements (continuity bands, local-margin confidence, 88-bin pitch features, AMT note-level symbolic alignment, rehearsal-acquisition logic) into an `/autoresearch` loop: modify one knob, run the eval, keep-or-revert based on a single composite metric.

---

## 2. What we did, in order

1. **Mapped the system.** `chroma_dtw.rs` is a stateless per-chunk free-start subsequence DTW, 12-bin chroma (built from L2-normalized columns via `apps/inference/muq/chroma.py`), full `n_a × n_s` cost matrix, monotonic step pattern. Called from `apps/api/src/do/session-brain.ts:601` inside `if (perfNotes.length > 0)` (AMT-gated). The audio chroma is normalized at the MuQ source, so raw energy is destroyed before reaching DTW.
2. **Ran the production Rust DTW on a real bucket-1 amateur.** Built a temp Python harness + Rust `tests/amateur_align_probe.rs` (since deleted) calling `chroma_dtw_native` on `Jt2f6yEGcP4.wav` ("Chopin Ballade No 1 — 1 month progress", bucket 1, 419s) vs `HlHBUxlcWfk.wav` (Kevin Chen, professional, 624s) across identical chunk positions. Harness validated against the existing `ballade1_coldstart_111s` fixture (pro @111s → bars 30-34, matched expected 25-40).
3. **Reproduced three failures:**
   - **Teleport.** Amateur first 15s aligned to bars 261-262 of a 264-bar score; amateur's own forward-track (~bar 19 by 120s) contradicts its independent `cs_120` chunk (bars 39-42). Free-start over a tonally repetitive score with a noisy short input is underdetermined.
   - **Cost is not confidence.** Correct and wrong alignments overlap in `[0.13, 0.21]`. The pro's *correct* `cs_120` cost is 0.174; the amateur's *wrong* `cs_000` is 0.214. Cost mostly reflects player skill, not location correctness.
   - **Silence locks confidently to garbage.** Pro `cs_000` (intro silence, rms ≈ 0.01) → bar 48, cost 0.049 — the lowest cost in the whole experiment, on the wrong bar.
4. **Read the two MEC 2026 papers** (Peter/Hu/Widmer "Precise and Simple Audio-to-Score Alignment"; Chiruthapudi et al. "Flexible Encoding Model for Non-Unique Note Alignments"). Paper 1's audio-to-audio baseline IS our method; their pitch-resolved features + beat-period continuity term lift median error 49ms → 21ms and eliminate spurious alignments. Their MIDI-to-score result (6ms) is what AMT note-level symbolic alignment could give us. Paper 2 reframes practice as fundamentally non-monotonic, non-unique; our `bar_per_frame` non-decreasing assertion is structurally wrong for rehearsal.
5. **`/brainstorm` → approved design.** Single-DP-fill WASM returns both global and in-band candidates from one fill (the existing `d[last, j]` array is comparable across all `j`); in-WASM arbitration in-band/relocalize/abstain; DO carries `expectedScoreFrame`; chroma uniformity gate for silence; thresholds as call-parameters. Spec: `docs/specs/2026-05-28-continuity-aware-chroma-follower-design.md`. Plan: `docs/plans/2026-05-28-continuity-aware-chroma-follower.md`.
6. **`/autopilot` → plan → challenge (2 loops, PROCEED_WITH_CAUTION) → build halted at Task 0 hard gate.**
7. **Task 0 gate (global-margin probe) FAILED with a real finding.** The amateur `cs_000` correct opening has accumulated DTW cost **189.97** at the correct opening frame; the wrong teleport zone (bar 261+) has cost **160.64**. The opening is a *global loser by ~29 units* — so the planned "global separation margin" cannot confirm the correct alignment because the correct alignment isn't the global winner.
8. **Wrote and ran a local-margin probe** on the same fixture data (no production code changed). Result: the correct opening is a *clear local winner* inside any forward window from frame 0. Margin 1.02 in a 6-second window, 32.03 in a 12-second window, 6.61 in a 20-second-and-wider window (all vs threshold 0.02). Pro `cs_111` warm-start: every window agrees with global, margin 12.66. **The local-margin hypothesis is validated.**
9. **Decided to pivot to eval-driven autoresearch** rather than patch the plan and ship the targeted fix. Rationale: every fix so far was caught by an ad-hoc probe; without a real eval, future changes (88-bin features, AMT-symbolic, rehearsal acquisition) will repeat the same one-recording validation cycle.

---

## 3. Evidence (the numbers that matter)

**Production DTW on bucket-1 amateur vs pro, identical chunk positions:**

| recording | chunk | rms | barMin | barMax | cost | trajectory |
|---|---|---|---|---|---|---|
| amateur_b1 | forward 0–120s | 0.036 | 3 | 19 | 0.2025 | clean |
| amateur_b1 | cs 0s+15 | 0.030 | **261** | **262** | 0.2139 | **TELEPORT** |
| amateur_b1 | cs 60s+15 | 0.039 | **246** | **247** | 0.2005 | **TELEPORT** |
| amateur_b1 | cs 120s+15 | 0.052 | 39 | 42 | 0.1905 | contradicts forward |
| pro_ref | forward 0–120s | 0.026 | 3 | 33 | 0.1606 | clean |
| pro_ref | cs 0s+15 | **0.010** | **48** | **48** | **0.0494** | **wrong (silent intro)** |
| pro_ref | cs 111s+15 | 0.041 | 30 | 34 | 0.1527 | matches fixture |
| pro_ref | cs 180s+15 | **0.009** | **170** | **171** | 0.1178 | **wrong (silent gap)** |

**Global vs local margin on amateur `cs_000` (prior = frame 0):**

| window | argmin frame | bar | best cost | margin | verdict |
|---|---|---|---|---|---|
| GLOBAL | 26773 | **262** | 160.635 | **0.0000** | teleport |
| +300f (~6s) | 281 | **2** | 379.420 | **1.0194** | PASS |
| +600f (~12s) | 562 | **3** | 347.386 | **32.0335** | PASS |
| +1000f (~20s) | 812 | **4** | 189.971 | **6.6066** | PASS |
| +6000f (~120s) | 812 | **4** | 189.971 | **6.6066** | PASS |

Pro `cs_111` is unaffected: every local window from prior 5550 agrees with the global answer (frame 5840, bar 34, margin 12.66).

The cost of the correct local winner (189.97) is *higher* than the cost of the global loser (160.64); the only thing that changes between "ambiguous garbage" and "clear winner" is the comparison scope. This is the central empirical fact driving the pivot.

---

## 4. Two design candidates, both parked

**A. Original (single-fill, global margin) — failed Task 0 gate.** Documented in `docs/specs/2026-05-28-continuity-aware-chroma-follower-design.md` and the plan. Global separation margin cannot confirm a correct alignment that is not the global cost minimum.

**B. Revised targeted (local margin in band, session-start prior frame 0) — validated on one recording.** Same architecture, only the margin computation scope changes (in-band, not global) and cold-start seeds `expected_score_frame = 0` (performance assumption). Rehearsal-mid-piece first-chunk acquisition is an explicit known limitation. We chose NOT to ship this because validation is on a single fixture; generalization across pieces/skill buckets/repeats is unmeasured.

Both candidates and the worktree are preserved on branch `feat/continuity-aware-chroma-follower` (5 spec/plan/fix commits + 1 probe commit, listed in §6).

---

## 5. The pivot: eval-first autoresearch

The bottleneck is the *metric*, not the loop. `/autoresearch` is mechanical once a verify command returns a single composite number that genuinely tracks alignment quality. Defining that number is itself a brainstorm-worthy design.

### Constraints any eval must satisfy

- **No reliance on hand-annotated amateur ground truth.** We have 361 YouTube recordings across 17 pieces in 5 skill buckets at `model/data/evals/skill_eval/`, but no note-level alignments for them.
- **Must surface the three reproduced failures.** Teleport, cost-not-confidence, silence-locks. If the metric goes up on a system that still teleports, the metric is wrong.
- **Per-commit runnable.** Long evals kill the loop. Target a verify command in the seconds-to-minutes range, not hours.
- **Multi-piece and multi-skill-bucket.** A single-recording metric won't catch generalization regressions.

### The three eval sources, composed

1. **Gold-truth slice via (n)ASAP + parangonar** (Paper 1's recipe). MIDI performances exist; parangonar gives near-perfect symbolic alignment as ground truth; synthesize audio or use existing MAESTRO audio; chroma DTW alignment error in milliseconds is exact. Most defensible single number.
2. **Synthetic chunks with known truth.** Render score (or known-aligned MIDI) → chroma → take a chunk starting at known frame T → align → error = |argmin − T|. Cheaply generate at scale across pieces, with controlled perturbations (added noise, dropped notes, tempo jitter, silence injection, jumps).
3. **Self-consistency on real amateur recordings.** No ground truth required: forward-tracking trajectory must be monotone; per-chunk cold-start alignments must agree with forward-tracking at the same time index; abstention must fire on injected silence; chunk N alignment within ±band of forward alignment at the same time (anti-teleport).

A composite (weighted ms-error on gold + teleport-rate on amateur + abstention-correctness on injected silence + monotonicity on forward-tracking) is plausible. Designing the weights is the brainstorm.

### Phased path

1. **`/brainstorm` the eval** — datasets, metric definition, verify command, runtime budget, guard.
2. **`/plan` + `/build` the eval harness** — fixture loaders, parangonar bridge, synthetic-chunk generator, metric aggregator, verify CLI returning a single number.
3. **`/autoresearch` against the harness** — start with candidates we already have on the parked branch (local-margin scope, band widths, neighborhood, uniformity threshold), then expand to feature-level (88-bin / pitch-resolved, AMT note-level).

---

## 6. State of artifacts

### Worktree / branch — parked, not deleted

- **Path:** `/Users/jdhiman/Documents/crescendai/.worktrees/feat/continuity-aware-chroma-follower`
- **Branch:** `feat/continuity-aware-chroma-follower`
- **Commits** (most recent first):
  - `probe: add subseq_dtw_last_row diagnostic + amateur cs_000 fixture + global/local margin probes`
  - `docs: initialize implementation notes`
  - `fix: address challenge blocker (loop 2) — extract pure status-dispatch fn for real DO tests`
  - `fix: address challenge blockers (loop 1)`
  - `docs(plan): add continuity-aware chroma follower implementation plan`
  - `docs(spec): add continuity-aware chroma follower design`
- **Production code changed:** none.

### Files added on the branch (probes + diagnostic only)

- `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs` — adds `#[cfg(not(target_arch = "wasm32"))] pub fn subseq_dtw_last_row(...)` returning `d[last, *]` from the cost matrix (the array both probes consume).
- `apps/api/src/wasm/score-analysis/src/lib.rs` — re-exports above.
- `apps/api/src/wasm/score-analysis/tests/fixtures/generate.py` — extended for per-case `audio_wav` parameter; adds `ballade1_amateur_cs000` case.
- `apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/` — fixture dir:
  - `audio_chroma.bin` (12 × 751 float32 LE, row-major)
  - `score_bars.json` (full 264-bar Ballade 1 score)
  - `expected.json` (bounds for a later cargo regression test; not yet wired)
  - `margin_probe.py` — global-margin probe (the one that failed Task 0)
  - `local_margin_probe.py` — local-margin probe (the one that validated the fix direction)
- `docs/specs/2026-05-28-continuity-aware-chroma-follower-design.md` — design spec for the parked candidate B.
- `docs/plans/2026-05-28-continuity-aware-chroma-follower.md` — implementation plan for candidate B (Task 0 gating present; later tasks reflect the original global-margin design).

### Files on `main`

- This handoff doc: `docs/implementation/2026-05-31-chroma-dtw-eval-pivot.md`
- No other production changes on main from this work.

### Data available for eval design

- **Bucket-labeled YouTube corpus:** `model/data/evals/skill_eval/{piece}/audio/*.wav` across 17 pieces; `candidates.yaml` per piece carries `skill_bucket: 1–5` and `downloaded` flag. Chopin Ballade 1 has 159 downloaded WAVs spanning all buckets.
- **Score JSON corpus:** `model/data/scores/*.json` — Ballade 1 confirmed (264 bars). MusicXML masters under `model/scores/v1/`.
- **MAESTRO MIDI:** `model/data/midi/percepiano/*.mid` — usable for (n)ASAP-style ground-truth alignment.
- **Existing alignment surfaces:** `model/tests/score_library/test_reference_pipeline.py` uses an onset-based DTW (different from chroma DTW) for reference-profile generation. Worth surveying but not the production path.

### Reproducing the probes

```bash
# From repo root, using the model uv env (has librosa)
cd /Users/jdhiman/Documents/crescendai/.worktrees/feat/continuity-aware-chroma-follower

# Global-margin probe (will FAIL Task 0 gate):
uv run --project ../../../model python \
  apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/margin_probe.py

# Local-margin probe (PASSES — validates the direction):
uv run --project ../../../model python \
  apps/api/src/wasm/score-analysis/tests/fixtures/ballade1_amateur_cs000/local_margin_probe.py
```

Adjust `parents[N]` in scripts if the worktree path differs.

### Open architectural questions left over

1. **Rehearsal-mid-piece first-chunk acquisition.** No single 15s chunk can disambiguate; needs multi-chunk acquisition-by-agreement. Out of scope for the parked targeted fix; the eval must measure this case so autoresearch can attack it.
2. **Should chroma alignment be decoupled from the AMT gate?** The DTW only needs MuQ chroma, but the DO nests it inside `if (perfNotes.length > 0)`. Currently moot because AMT will be up via `just amt`, but the gate is structurally wrong and the eval should not depend on it.
3. **Bar-per-frame non-decreasing assertion (Paper 2).** Structurally incompatible with rehearsal jumps/restarts. Future work; flag it explicitly so a representation change doesn't surprise downstream consumers.

---

## 7. Brainstorm prompt for the next session

The prompt below is self-contained — it briefs a fresh `/brainstorm` agent on everything above and asks it to design the eval (not the DTW fix). Paste verbatim.

```
/brainstorm

Build an eval harness for CrescendAI's chroma-DTW score follower that
returns a SINGLE composite number suitable for /autoresearch loops. The
loop will measure each candidate change (local-vs-global margin scope,
band widths, uniformity threshold, eventually 88-bin pitch features and
AMT note-level symbolic alignment) against the same metric, keep-or-revert
based on the number.

Context (do not re-derive — full handoff at
docs/implementation/2026-05-31-chroma-dtw-eval-pivot.md):

- Production DTW: apps/api/src/wasm/score-analysis/src/chroma_dtw.rs.
  Stateless per-chunk free-start subsequence DTW, 12-bin chroma, monotonic
  step pattern. Called from apps/api/src/do/session-brain.ts:601 inside
  the AMT-gated path. Chroma is L2-normalized at the MuQ source so raw
  energy is destroyed before the DTW sees it.

- Three reproduced failures the eval MUST surface (a metric that improves
  while these still happen is wrong):
  1. Teleport: amateur opening 15s aligns to bar 261-262 of a 264-bar score.
  2. Cost is not confidence: correct (cost 0.174) and wrong (cost 0.214)
     alignments overlap.
  3. Silence locks: rms~0.01 chunks produce confident-looking wrong locks
     (cost 0.049 -> wrong bar 48).

- One revision validated on a single fixture, NOT shipped: local separation
  margin within an in-band window from a session-start prior of frame 0.
  Probe results in the handoff doc. Generalization across pieces/skill
  buckets/repeats is unmeasured — that is precisely what this eval is for.

- Two MEC 2026 papers worth mining for eval design (not for the DTW itself):
  * Peter/Hu/Widmer "Precise and Simple Audio-to-Score Alignment" — uses
    (n)ASAP + parangonar for ms-level alignment ground truth. Their
    audio-to-audio baseline IS our method (49ms median, 53% under 50ms,
    with spurious alignments). Their best method hits 21ms median, 83.7%
    under 50ms. MIDI-to-score is 6ms. These give us calibrated targets.
  * Chiruthapudi et al "Flexible Encoding Model for Non-Unique Note
    Alignments" — reframes practice as non-monotonic / non-unique. The
    eval should include repeats, restarts, jumps, partial play, errors.

- Datasets available locally:
  * model/data/evals/skill_eval/{piece}/audio/*.wav — 17 pieces, 5 skill
    buckets, 361 recordings, no ground-truth alignments. Chopin Ballade 1
    has 159 downloaded WAVs across buckets.
  * model/data/scores/*.json — score JSONs (Ballade 1 = 264 bars).
  * model/data/midi/percepiano/*.mid — MAESTRO MIDI usable for
    parangonar-style gold-truth alignment.
  * MusicXML masters under model/scores/v1/.

- Parked branch with the validated-but-unshipped revision and probes:
  feat/continuity-aware-chroma-follower in
  .worktrees/feat/continuity-aware-chroma-follower. Spec at
  docs/specs/2026-05-28-continuity-aware-chroma-follower-design.md, plan at
  docs/plans/2026-05-28-continuity-aware-chroma-follower.md. Do NOT
  reuse the spec; design the eval fresh.

Constraints any eval must satisfy:
- No reliance on hand-annotated amateur ground truth (we don't have it).
- Must surface all three reproduced failures listed above.
- Per-commit runnable (target seconds-to-minutes, not hours).
- Multi-piece and multi-skill-bucket (a one-recording metric won't catch
  generalization regressions).
- Honest about what it does NOT measure (rehearsal acquisition, etc.).

Three eval sources I'd compose; the brainstorm should pressure-test them:
1. Gold-truth slice via (n)ASAP + parangonar — synthesize or use MAESTRO
   audio, align with our DTW, error in ms vs parangonar gold.
2. Synthetic chunks with known truth — render score/MIDI to chroma, take
   chunks at known start frames, error = |argmin - T|. Cheaply scale with
   perturbations: noise, dropped notes, tempo jitter, silence injection,
   jumps.
3. Self-consistency on real amateurs — no ground truth needed:
   forward-tracking monotonicity, per-chunk-vs-forward agreement,
   abstention fires on injected silence, anti-teleport bound on
   chunk-vs-forward bar distance.

Open the brainstorm by drilling on the SINGLE most important question:
what is the one composite number that, if it goes up, tells us the
alignment system actually got better — and what is the dataset that
produces it cheaply enough to run per-commit?

Out of scope for this brainstorm (defer to a later session): the actual
DTW improvements; rehearsal non-monotonic representation; AMT-symbolic
alignment integration.

After the design is approved, the next steps are /plan and /build for the
EVAL HARNESS (not the DTW), then /autoresearch runs against it with the
parked branch as one of the first candidates.
```

---

## 8. To resume this work

1. Read this doc.
2. Read the two papers (paths/citations in §2, full texts in the conversation history that produced this doc).
3. Open a fresh session and paste §7 verbatim into `/brainstorm`.
4. The parked branch and its probes are reference artifacts; the eval design should not depend on them.
5. After the eval ships and `/autoresearch` runs, the parked candidate (local-margin + session-start prior) is the natural first iteration to measure.
