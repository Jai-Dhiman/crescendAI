# Implementation Notes — Deterministic Claim Verifier (#65)

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

Build started from baseline: 45 tests passing.

## Task 1: VerdictResult dataclass and UnverifiableError
Verbatim transcription of plan. No deviations. 4 tests pass. Commit 8ee818f2.
Spec review: PASS (all 10 fields correct types, no scope creep).

## Task 2: SubstrateErrorEngine (commit 2a9bcb0d, tests 663e55eb)
Seeded MC engine. KEY PROPERTY: applies a single SCALAR jitter to all anchors per MC
sample (global shift), so for identity alignment, alignment_uncertainty == jitter sigma
(~0.010s) independent of anchor density. Review flagged two confounded/tautological
alignment tests; replaced with behavioral tests (identity==sigma; dense==sparse under
global shift). Production code unchanged.

## Task 3: LocationResolver (commit 815d740c)
Bar-range -> audio-time via np.interp on anchors; MC failsafe raises unlocalizable when
uncertainty_bars >= location_span_bars. CHALLENGE RISK #4 RESOLVED: the plan's failsafe
fixture (2 anchors, 50ms bar, query at boundary 0.0) did NOT trigger — np.interp clips at
the boundary so uncertainty pinned to jitter sigma (~9.6ms), uncertainty_bars=0.19. Fix:
use a 4ms bar (finer than alignment precision) -> uncertainty_bars=2.4 >= 1, raises
unlocalizable deterministically (3x). Production failsafe NOT weakened. Semantic: cannot
localize to a bar finer than alignment precision.

## Task 9: taxonomy v0.1 (commit de3cfaf2)
taxonomy_version v0->v0.1; dynamics status active (rms_contour, tau=1.5 dB, tier 2,
librosa_rms_region_estimator, minimum_events=20). verdict_dispatch.py UNTOUCHED (invariant
held). EXPECTED pre-existing failures introduced (dynamics no longer gated) -> fixed in Task 8.

## Task 4: TimingMeasurer (commit d4b43c85)
DEVIATION (correct, caught a plan bug): plan's region formula `(region_bpm - established)/established*100`
gave rushed=>d>0, but spec docstring + SHIPPED route_verdict require rushed (polarity "-") => d<0.
Implementer flipped to `(established - region_bpm)/established*100`. Verified consistent with
verdict_dispatch polarity branch (polarity "-" SUPPORTED iff d<0 and abs(d)>tau). Without this
flip EVERY rush claim would REFUTE. d values: rush=-25, injection |d|>8, uniform CV<5%.
KNOWN MINOR (non-blocking, var is sign-invariant): _substrate_var MC loop still uses old
(bpm-established) direction and uses region tempo as reference rather than established — both
already flagged [OBS] in challenge; error_bar magnitude unaffected.

## Task 5: PedalingMeasurer (commit 49adf72c)
DEVIATION (SOUND): plan gated region_too_short on region-local event count, but plan's own
test_no_pedal_in_region_negative_d expects a 0-pedal region to give d<0 (sparse signal), not
region_too_short — a plan self-contradiction. Resolved by gating region_too_short on GLOBAL
sustain_on_times.size; region-local 0-event => d = 0 - self_density < 0. Returned event_count is
region-local. SAFETY NET: route_verdict Step 5 re-checks event_count<minimum_events at verdict
layer. KNOWN MINOR (non-blocking): _pedal_bar_fraction_from_times is a named no-op wrapper.

## Task 6: DynamicsMeasurer (commit 2c02f881, fix cb6e85ba)
Verbatim plan logic. Review removed 2 dead imports (dataclass, Path) and documented the region
substrate_var scalar-offset simplification (var(jitters) == var(perturbed_d) for scalar j).
except-Exception around librosa.load re-raised as typed UnverifiableError(substrate_failure) — ACCEPTABLE.
d: flat=>negative, wide/loud=>positive, 6x injection |d|>1.5 dB.

## Task 7: verify() orchestrator (commit 209feb95)
Wires LocationResolver + measurers to SHIPPED route_verdict (UNTOUCHED). _measurement dict
provides exactly the 6 keys route_verdict reads: d, tau, error_bar, event_count, localizable,
substrate_failure. NEVER raises — all UnverifiableError paths -> VerdictResult(UNVERIFIABLE,reason_code).
Lazy module-global measurer registry (challenge: non-thread-safe but fine single-process eval).
ADDED (challenge risk #3): test_timing_rush_claim_strong_behavioral asserts SPECIFIC verdict
SUPPORTED on a known-by-construction rush bundle (d=-42.86, tau=8) IN ADDITION to the wiring
smoke test. PROCESS NOTE: first two agent dispatches gave unreliable "DONE" reports (phantom
SHAs 870aa372, files briefly uncommitted); controller verified via git log/ls/3x test runs and
confirmed 209feb95 is the real landed commit, 6/6 deterministic. Taxonomy measurement key
"amt_onsets_region_tempo_fit" matches registry.

## Task 10: signed-d convention doc (commit 3539b785)
docs/model/claim-verifier-signed-d-conventions.md + doc-exists test in test_taxonomy_v01.py (6/6).
Timing row uses the IMPLEMENTED sign `(established - region_bpm)/established*100` (rushed=>d<0),
not the plan draft. Doc heading lowercased to "Sign convention" to match the test assertion.
Documents the global-shift alignment-uncertainty limitation (tier-2). PROCESS NOTE: agent's first
report had a phantom SHA; controller verified real commit 3539b785 landed + 6/6 pass before review.

## Task 8: CLI + BundleExtractor + taxonomy test updates (commit 54509cdb)
cli.py (verify subcommand -> JSON VerdictResult), model/src/claim_measurement/extractor.py
(skeleton reusing chroma_dtw_eval.amt_regen; pedal_events:[] until AMT exposes CC64; .tmp
write-then-replace; raises BundleExtractionError). CLI test uses Path(__file__)-anchored cwd
(challenge risk #1 RESOLVED - no hardcoded path). test_round_trip dynamics-gated test replaced
with test_dynamics_active_routes_correctly. Extractor smoke test SKIPS (no bundle) and does not
import extractor.py (collection-safe). Full claim_taxonomy suite 94 passed; verdict_dispatch.py
UNTOUCHED. All git via -C worktree (challenge risk #5 honored).
