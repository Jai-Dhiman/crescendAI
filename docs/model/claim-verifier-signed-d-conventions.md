# Claim Verifier: Per-Dimension Signed-d Conventions and Error-Bar Table

**Issue:** #65, #94 (first real-audio measurement), #101 (GATE 3 ground-truth baseline + hard gates)
**Taxonomy version:** v0.3
**Validation boundary:** This document covers substrate-level measurement. The error bars in
"Measured Error Bars" below are MEASURED on real cached audio (AMT held fixed, #94); the
"Substrate Error Distributions" table remains the per-source design input. Proxy-to-perception is
validated for dynamics+pedaling (GATE 2 / #66); error-rich localization is coverage-gated (GATE 1).
**GATE 3 (#101) is the current frontier and the honest baseline: at whole_piece — where 99.6% of
real generator claims land — the verdict is measurement-driven for pedaling ONLY; dynamics and
timing are degenerate (verdict determined by polarity, not performance). See the GATE 3 section
and the Path #1 hard gates at the end of this document.**

---

## Sign convention Table

| Dimension | d formula (region) | d formula (whole_piece) | d < 0 means | d > 0 means | tau (provisional) | units |
|-----------|-------------------|------------------------|-------------|-------------|-------------------|-------|
| timing | `(established_tempo - region_median_bpm) / established_tempo * 100` | `CV% = std(bpms) / mean(bpms) * 100` | faster than reference (rushed) | slower than reference (dragging) | 8.0 | percent |
| pedaling | `region_on_fraction - whole_piece_on_fraction` | `on_fraction - REFERENCE_FRACTION` | drier / less sustain (under-pedaled) | wetter / more sustain (over-pedaled) | 0.25 | fraction |
| dynamics | `mean(region_rms_db) - mean(piece_rms_db)` | `std(rms_db) / dynamic_range` (dispersion) | quieter / flatter than whole piece | louder / wider than whole piece | 1.5 | dB |

**Note on the timing sign:** The region formula subtracts the measured region BPM from the established tempo, so a rushed region (higher BPM) yields d < 0 and a dragging region yields d > 0. This matches the shipped `route_verdict` polarity contract: a "you rushed" claim carries polarity "-" and is SUPPORTED only when d < 0 and abs(d) > tau.

**Note on whole_piece:** Self-referential references (established_tempo, self_density, within_region_range) degenerate for whole_piece location. Each measurer switches to an intrinsic dispersion or presence statistic for whole_piece claims.

---

## Substrate Error Distributions

Documented error distributions used by `SubstrateErrorEngine` for Monte-Carlo error propagation:

| Source | Distribution | Parameters | Rationale |
|--------|-------------|------------|-----------|
| AMT onset jitter (timing, pedaling localization) | Gaussian | mean=0, sigma=0.010s | Aria-AMT onset error on clean audio (10ms 1-sigma estimate from MAESTRO validation) |
| RMS frame variance (dynamics) | Gaussian | mean=0, sigma=0.3 dB | Empirical: constant-signal buffer shows ~0.3 dB std from librosa hop-length framing |
| CC64 threshold uncertainty (pedaling) | Uniform | low=-10, high=+10 value counts | MIDI CC quantization: threshold 64 +/- 10 covers typical soft-pedal ambiguity |

**Note on alignment uncertainty:** `alignment_uncertainty_sec` perturbs all anchors by the same scalar jitter per Monte-Carlo sample (a global shift, not independent per-note jitter). This conservatively overestimates correlated error and makes the unlocalizable failsafe more aggressive. For an identity alignment the resulting uncertainty equals the jitter sigma (~0.010s) independent of anchor density. This is a known tier-2 limitation.

---

## error_bar Formula

```
error_bar = sqrt(sampling_var + substrate_var)
```

- **sampling_var**: `var(bootstrap_d(within_region_events, stat_fn))` -- bootstrap (N=500, seeded) over events in the region.
- **substrate_var**: `var(MC_perturbed_d_values)` -- Monte-Carlo (N=500, same seeded engine) perturbing raw measurements by the distributions above.

The `SubstrateErrorEngine` is initialized with `seed=42` by default, making all error estimates deterministic for a given bundle.

---

## Near-Threshold Dead-Band

`route_verdict` applies: if `abs(abs(d) - tau) <= error_bar` then UNVERIFIABLE(near_threshold).

This dead-band prevents confident SUPPORTED/REFUTED verdicts when the measured deviation is within one error bar of the tolerance threshold.

---

## Accuracy Boundary

The mistake-injection harness (bundle-level signal perturbation) tests measurer recovery from:
- Timing: onset shifts (20% region speedup / slowdown)
- Pedaling: CC event injection/removal in region
- Dynamics: programmed RMS envelope (6x amplitude injection)

**Captioned:** "Accuracy under AMT transcription error deferred to GATE 1." The injection harness holds the AMT bundle fixed and perturbs it by a known delta. It does not simulate AMT transcription errors on raw audio.

---

## Minimum Events

| Dimension | minimum_events | Unit | Equivalent duration |
|-----------|---------------|------|---------------------|
| timing | 8 | note onsets | ~4 bars at moderate tempo |
| pedaling | 2 | sustain-on events (CC64 >= 64) | ~2 bars with any pedaling |
| dynamics | 20 | RMS frames (hop=512, SR=16kHz) | ~640ms audio |

---

## Measured Error Bars (real cached audio, AMT fixed) -- #94

The error bars below are **MEASURED**, not estimated. They come from running the shipped
`verify()` over real measurement bundles extracted from cached practice-eval audio via real
Aria-AMT transcription (`model/src/claim_measurement/extract_cli.py`), then re-run with the
real-claim harness (`apps/evals/claim_taxonomy/run_real_verify.py`). The prior sections give
the per-source design inputs; this section gives the realized end-to-end `error_bar` per
dimension. **Caveat:** AMT is held fixed -- these bars capture sampling + substrate-model
variance on *clean* transcription, NOT accuracy under AMT transcription error on error-rich
audio (GATE 1, still deferred).

**Substrate:** 3 bundles extracted (`bach_invention_1` x2, `bach_prelude_c_wtc1` x1; the other
18 cached clips lack a `_load_bach_json_score`-compatible single-tempo 4/4 score and are
reported `no_score`, not silently skipped). 12 `verify()` runs (4 real teacher-prose claims x
3 bundles). Each run uses `SubstrateErrorEngine(seed=42)`, so the bars are deterministic.

| Dimension | location kind | measured `error_bar` range | units | n (events) | notes |
|-----------|---------------|----------------------------|-------|------------|-------|
| timing | bar-range (bars 9-12) | 0.18 -- 0.34 | percent | 141 -- 237 | tight when onsets exist in the region; one bundle hit `UNVERIFIABLE(region_too_short)` at n=0 |
| timing | whole_piece (CV%) | 2.38 -- 19.78 | percent | 557 -- 1807 | **large**: raw AMT onset IOI on fast Bach is noisy; the prelude's bar (19.78%) exceeds tau=8% |
| pedaling | whole_piece (density) | 0.00 -- 0.11 | fraction | 178 -- 350 | measurable on real CC64 events; verdicts spanned d=0.09 (REFUTED) to d=1.0 (SUPPORTED) |
| dynamics | whole_piece (dispersion) | 0.026 -- 0.035 | dB | 3095 -- 9111 | tightest and most stable dimension across bundles |

### Pedaling decoded successfully -- dimension stays `active`

The #65 extractor stubbed `pedal_events = []` on the false premise that "the AMT server returns
notes only." That premise is wrong: Aria-AMT emits CC64 sustain events
(`apps/inference/amt/server.py:485-487`). Measured pedal-event counts on the three real bundles:

| bundle | clip ~dur | pedal_events | note: on/off split | bar density |
|--------|-----------|--------------|--------------------|-------------|
| bach_prelude_c_wtc1/mfN8ZEYWdqs | 152s | 354 | 178 on / 176 off | 1.00 |
| bach_invention_1/2cbYFp9kNpg | 292s | 453 | -- | ~0.09 region |
| bach_invention_1/7zVlDxBO5q4 | 97s | 1246 | 350 on / 896 off | 0.46 |

Events are clean 0/127 on-off alternation and yield real density verdicts, so pedaling remains
`status: active` (no re-gate). **Honest caveat:** the invention `7zVlDxBO5q4` decodes ~12.8
events/s, which likely over-segments pedal on/off on busy passages; the coarse
*presence-per-bar density* proxy is robust to this, but fine half-pedal/flutter remains out of
scope as documented.

---

## GATE 1: Localization Robustness on Error-Rich Audio (#95)

**Verdict: bar-level localization FAILS the ~90% bar. Taxonomy restricted to v0.2
(single-bar inadmissible; region degraded; whole_piece reliable).**

GATE 1 is the localization-in-errors attack the plan §3.4/§7 names as the dominant
reviewer risk. Since `ood_practice/` is empty, error-rich audio was *manufactured* by
degrading the real cached Bach clips with **construction-known** corruptions and measuring
whether the shipped `LocationResolver` moves each bar's resolved time the way the known
transform predicts.

### Method (truth by construction, no hand annotation)

`delta(bar) = M_corrupt(bar) - W(M_clean(bar))`, where `M_*` is the shipped resolver's
resolved bar-start audio time and `W` is the known corruption warp (a piecewise time-warp's
exact `WarpMap` for tempo, identity for noise/dropout). `W` is the ground truth; the clean
alignment is not assumed correct. Harness: `model/src/claim_measurement/gate1/`
(`corruption.py`, `build_corrupt_bundles.py`) + `apps/evals/claim_taxonomy/gate1/`
(`localization.py`, `analyze.py`). Corpus: 2 real Bach Invention clips
(`7zVlDxBO5q4`, `2cbYFp9kNpg`), sweep `{clean, tempo rush 1.3x/1.6x, dropout, noise SNR10}`.
The C-major prelude is **excluded**: its arpeggiated transcription made parangonar's matcher
blow up combinatorially (>1 h, never converged) -- itself a substrate-fragility finding.

### Measured bar-level localization (over the bars the resolver attempts)

| corruption | resolvable rate | within ±0.5 s | within ±1.5 s | median \|delta\| | p90 \|delta\| |
|------------|-----------------|---------------|---------------|------------------|---------------|
| clean (noise floor, identical audio) | 0.45 | 0.80* | 0.87* | 0.000 s | 1.93 s |
| tempo (rush 1.3x/1.6x) | 0.45 | 0.57* | 0.63* | 0.57 s | 6.58 s |
| dropout (missing notes) | 0.41 | 0.46* | 0.54* | 0.53 s | 15.1 s |
| noise (SNR 10 dB) | 0.41 | 0.23* | 0.31* | 3.17 s | 17.3 s |

`*` = over **resolvable** bars (single-clip 7zVlDxBO5q4 figures). The 2-clip pooled
`within_over_total` is lower still (≤0.25 at ±1.5 s) because out-of-anchor-span bars cap the
resolvable pool at 0.44.

### Three substrate causes, none fixable at the verifier layer

1. **Partial alignment** -- ~32-56% of bars fall outside the matched parangonar anchor span and
   are now **abstained** (UNVERIFIABLE), not extrapolated. This is the `LocationResolver`
   extrapolation guard added under #95: previously `np.interp` clamped out-of-span bars to the
   last anchor, yielding confident, wrong, run-to-run-unstable times.
2. **AMT nondeterminism** -- decoding is greedy (`torch.argmax`) yet re-transcribing
   *bit-identical* audio yields different onsets run-to-run (GPU float ties flip tokens),
   **amplified by parangonar match-set churn** (one flipped onset reshuffles the anchor set,
   swinging nearby bars by seconds). This sets the clean noise floor BELOW 90%.
3. **Corruption fragility** -- SNR-10 noise collapses localization to ~31%; the −14 to −20 F1
   distribution-shift degradation the plan predicted, realized.

### Region-width sweep (does widening recover it?)

Widening the claim from a single bar to a W-bar region (boundary error judged relative to the
region's duration) helps but does **not** robustly clear 90% under corruption (over-resolvable):

| kind | W=1 | W=2 | W=4 | W=8 |
|------|-----|-----|-----|-----|
| clean | 0.75 | 0.78 | 0.86 | 1.00 |
| tempo | 0.53 | 0.38 | 0.35 | 0.46 |
| dropout | 0.56 | 0.78 | 0.86 | 0.67 |
| noise | 0.28 | 0.22 | 0.14 | 0.33 |

Only `clean` reaches ~0.86-1.0; tempo/noise never do, and `over_total` stays ≤0.34 (the
out-of-span cap). W=16 collapses to 0 because no 16-bar all-resolvable window exists on a
22-bar clip with this resolvable rate.

### Decision (taxonomy v0.2, `localization_granularity` block)

- **whole_piece** -> RELIABLE tier (no bar indexing; always resolvable with >=2 anchors). This
  is where dynamics dispersion / pedaling density / timing CV% live -- the stable checks.
- **region** (bar-range, width >= `min_region_bars` = 4) -> DEGRADED, admissible but low-yield;
  safety comes from abstention, not accuracy.
- **single_bar** -> INADMISSIBLE.

The verifier is **safe-by-abstention**: on partially-aligned or error-rich clips it returns
UNVERIFIABLE rather than committing a mislocalized verdict. The headline for M3 therefore
shifts toward faithfulness *among committed claims* plus an abstention-rate, with whole_piece as
the dependable substrate. **Caveat:** GATE 1 measures degradation under corruption on only 2
clips (1 piece family); broadening the corpus past the 2 Bach pieces remains open.

### Corpus breadth (#98): resolvable-rate is piece-dependent (0.13-0.99)

The score loader was generalized to variable-tempo / non-4/4 (#98), unlocking the full
ASAP-derived catalog. Clean bundles were extracted for 10 clips across 6 composers. The
per-clip **resolvable-rate** (fraction of bars whose score-time falls inside the matched
parangonar anchor span -- the dominant cap on bar localization) is **not uniformly low; it
ranges 0.13-0.99 and tracks anchor coverage almost exactly**:

| clip | bars | resolvable rate | anchors | span coverage |
|------|-----:|----------------:|--------:|--------------:|
| pathetique_mvt2           |  73 | 0.99 | 379 | 1.01 |
| moonlight_sonata_mvt1     |  69 | 0.99 |  96 | 1.00 |
| schumann_traumerei        |  25 | 0.92 | 330 | 0.98 |
| fur_elise                 | 105 | 0.89 | 140 | 0.89 |
| bach_invention_1 (clip B) |  22 | 0.68 | 158 | 0.75 |
| bach_prelude_c_wtc1       |  35 | 0.54 | 114 | 0.57 |
| chopin_waltz_csm          | 193 | 0.46 | 611 | 0.48 |
| chopin_etude_op10no4      |  83 | 0.35 | 686 | 0.36 |
| bach_invention_1 (clip A) |  22 | 0.23 | 125 | 0.24 |
| liszt_liebestraum_3       | 131 | 0.13 | 152 | 0.13 |

**Methodological caveat on the GATE 1 verdict above:** the bar-level corruption sweep was run
on the two Bach *invention* clips, which sit at resolvable 0.23 and 0.68 -- among the
WORST-coverage clips in the corpus. Four clips (pathetique, moonlight, schumann, fur_elise)
have 0.89-0.99 coverage, where the out-of-anchor-span abstention would not dominate. So the
bar-level FAIL is partly a worst-case artifact of the test clips. Whether it holds on a
high-coverage clip is being measured directly (corruption sweep on pathetique, `gate1_hicov/`):
the AMT-nondeterminism noise floor and corruption fragility still apply regardless of coverage,
so a FAIL there would be the stronger substrate-level result, while a PASS would mean bar-level
viability is recording-dependent (gated on coverage) -- arguing for a per-clip coverage gate
rather than a blanket bar-level ban. **Operational note:** the local AMT server is
throughput-bound (~64 s/chunk) and unstable under sustained batch load (it died once mid-run);
the full 16-piece sweep is regenerable but multi-hour. The score loader, 16-piece map, and a
per-clip `--timeout-sec` extraction guard shipped under #98.

### Coverage gate (#100): the bar-level FAIL was coverage-dependent

The #95 corruption sweep ran on the two low-coverage Bach invention clips. Re-running it on
the two highest-coverage clips (pathetique 0.99 / 379 anchors, moonlight 0.99) overturns the
blanket bar-level FAIL: on well-aligned clips, **clean bar localization is essentially
perfect** and even noise is tolerable. Pooled over the two high-coverage clips (n=2):

| corruption | within ±0.1 s | within ±1.0 s | within ±1.5 s | median \|delta\| | p90 \|delta\| |
|------------|---------------|---------------|---------------|------------------|---------------|
| clean      | **0.96**      | **0.99**      | 0.99          | 0.000 s          | 0.003 s |
| noise SNR10| 0.49          | 0.70          | 0.73          | 0.111 s          | 2.50 s |
| dropout    | 0.44          | 0.55          | 0.56          | 0.649 s          | 10.2 s |
| tempo      | 0.27          | 0.44          | 0.46          | 0.814 s          | 11.2 s |

Compare clean on the Bach clips (~0.55 within ±0.1 s): the "AMT-nondeterminism noise floor"
of #95 was largely **coverage-induced parangonar churn**, not irreducible model noise -- with
dense full-coverage anchors, clean re-extraction is deterministic to p90 0.003 s. Tempo
corruption stays the lone hard mode (~0.45) even at high coverage (a strong artificial warp
genuinely outruns the aligner), and is the documented residual limitation; region-level *timing*
measures (percent deviation over the region) tolerate the resulting window jitter.

**Decision (taxonomy v0.3):** the v0.2 blanket `single_bar: inadmissible` is replaced by a
**per-clip coverage gate**. `LocationResolver(min_coverage=...)` computes
`anchor_span / measure_table_span`; bar/region claims on clips below the threshold abstain as
`UNVERIFIABLE(low_coverage)`; whole_piece is exempt. The orchestrator threads
`coverage_gate.threshold` (provisional **0.9**) from the taxonomy. The threshold is calibrated
on the extremes (high-coverage 0.99 -> 0.96-0.99 clean; low-coverage 0.23-0.68 -> 0.34-0.55);
the 0.68-0.99 gap is unsampled, so 0.9 is conservative pending mid-coverage clips. Default
`min_coverage=0.0` leaves raw-localization measurement and prior tests unchanged; the gate is
active only through the orchestrator. `verdict_dispatch.py` is untouched (the gate is a resolver
abstention flowing through the orchestrator's UnverifiableError path).

---

## GATE 2 (#66): expert-anchor validation — do the measurements track perception?

GATE 2 (the Path-#2 go/no-go) asks whether the verifier's deterministic measurements correlate
with expert-judged quality. PercePiano (1202 segments, human perceptual labels, MIDI) is the
anchor. Because PercePiano is MIDI but the shipped dynamics measurer is audio/librosa-RMS, a
MIDI-native adapter computes the measurements directly:
`model/src/claim_measurement/expert_anchor/{midi_measures,correlate}.py`. **No LLM is involved**
— PercePiano labels are human ratings (the `dual_judge` artifact is the *teacher-feedback* judge,
an M3 concern, NOT a GATE 2 blocker as the plan implied).

Each measurement is correlated (Spearman) against its `composite_labels.json` perceptual
dimension, both raw and **partial controlling for the mean of the other 5 dims** (halo control —
the dimension-specific validity number that distinguishes a real proxy from a global-quality
confound):

| dimension | validated MIDI proxy | raw rho | **partial rho** | verdict |
|-----------|---------------------|--------:|----------------:|---------|
| dynamics  | mean MIDI velocity  | +0.625  | **+0.562**      | **PASS** |
| pedaling  | CC64 on-fraction    | +0.583  | **+0.478**      | **PASS** |
| timing    | inter-onset-interval CV | +0.292 | **+0.252**   | marginal |

All p < 1e-24 at n=1202. **Calibration context: PercePiano inter-rater reliability is ~0.5**
(plan §7), so dynamics (0.56) and pedaling (0.48) sit **at/near the label-noise ceiling** — a
pass, not a weak result. Timing (0.25) is well below: the perceptual "timing" construct
(rubato/pulse/appropriateness) does not reduce to onset-interval variability.

Process notes worth keeping: (1) the *first* proxies were poor — tempo-CV gave timing rho 0.09,
velocity-dispersion gave dynamics −0.09; the validated proxies (IOI-CV, mean-velocity) emerged
only from a multi-feature sweep, so a null first result is about the proxy, not the dimension.
(2) The discriminant matrix initially looked like pedaling was a halo confound (pedal_frac
correlated ~0.55 with the all-dims mean), but the **partial correlation rehabilitated it** (0.48
specific signal survives) — the partial is the rigorous test, not the raw cross-correlation.
(3) **Forward-looking:** the validated dynamics proxy is mean *level*, not the verifier's current
whole_piece dynamics *dispersion* — perceived dynamics tracks loudness level more than spread;
worth revisiting the verifier's dynamics statistic.

**GATE 2 decision:** dynamics and pedaling are validated against human perception; timing is not.
Path #2 (RLVR) is greenlightable on the **dynamics+pedaling** subset (a narrower but real basis);
timing-based verifiable rewards are not yet justified. Report JSON (regenerable):
`model/data/results/gate2_expert_anchor.json`.

---

## GATE 3 (#101): the whole_piece verdict is degenerate for 2 of 3 dimensions — the honest baseline

GATE 3 is the M3 feasibility checkpoint (can we report a measured per-dimension faithfulness rate
for an LLM feedback generator?). Two findings reframe everything downstream. Both are code-grounded
(`verdict_dispatch.py`, the three measurers) and empirically confirmed (`verify()` run directly).

### Finding 1 — claim yield: generators almost never localize

LLM-extractor (Sonnet 4.6, truth labels NOT involved) exhaustive decomposition of 48
deterministically-sampled `baseline_v1.jsonl` prose records (6 bundle-overlapping pieces), 250 atomic
claims:

| Metric | Value | Read |
|---|---|---|
| claims / record | 5.21 mean | prose IS decomposable into atomic claims |
| validated-scope (dyn+ped) / record | 2.40 mean, **100% of records ≥1** | supply is sufficient for a stable whole_piece rate |
| **bar-localizable claims** | **1 / 250 (0.4%)** | **a *localized* faithfulness headline is dead; the headline is whole_piece** |
| whole_piece | 249 / 250 (99.6%) | the rate lives at whole_piece |
| vague-region language (no bar #s) | 15% | generators gesture at regions but never index bars |

Artifact: `scratchpad/m3_yield_feasibility.json` (regenerable). Implication: the GATE-1 localization
machinery characterizes the *benchmark's capability*, not the measured generator; "LLM piano teachers
essentially never localize their claims (0.4%)" is itself a publishable measured finding.

### Finding 2 — the whole_piece verdict is polarity-determined, not measurement-driven (the BLOCKER)

The frozen `route_verdict` is a SIGNED-anomaly test: `+` SUPPORTED iff `d>0 and |d|>tau`; `-` iff
`d<0 and |d|>tau`; `neutral` iff `|d|≤tau`. But each measurer computes a DIFFERENT quantity at
whole_piece than at a region — a **non-negative dispersion/presence statistic** with no signed
reference. Plugging a non-negative `d` into a signed test produces pathologies. Empirically confirmed
(fur_elise bundle, all three polarities):

| dim | whole_piece statistic | tau (unit) | behavior | status |
|---|---|---|---|---|
| dynamics | `std/range` (unitless, ≤0.5) | 1.5 **dB** | `+`/`-` UNREACHABLE (`d>1.5` impossible); `neutral` ALWAYS SUPPORTED | **DEGENERATE (unit mismatch)** |
| timing | IOI-BPM CV% (≥0, 100–260% on real polyphony) | 8.0 percent | `+` auto-SUPPORTED; `neutral`/`-` auto-REFUTED | **DEGENERATE (CV floor ≫ tau; noisy statistic)** |
| pedaling | pedal-bar fraction ∈[0,1] | 0.25 fraction | verdict tracks the performance (fraction vs 0.25) | **SUPERSEDED 2026-06-26** (front 3) |

Pedaling was the only dimension whose whole_piece statistic was in units matching its tau, but the
shipped statistic was pedal-BAR-fraction — an unvalidated *cousin* of the quantity GATE 2 actually
validated (CC64 *time*-on-fraction, partial ρ 0.478). **Front 3 (#101, 2026-06-26)** corrects this:
the whole_piece statistic is now signed CC64 time-on-fraction vs a corpus-median reference, so the
verifier checks the exact validated quantity and `-` (under-pedaled) claims are adjudicable instead
of structurally auto-REFUTED (see FRONT 3 UPDATE below). Residual caveat: virtue ("luminous
resonance," `+`) is still conflated with excess ("over-pedaled," also `+`) — backlog item 8.

**Bottom line:** a faithfulness rate computed today would mostly measure the generator's *polarity
distribution*, not its truthfulness (dynamics `±` → 0% by construction; timing `+` → 100% by
construction; only pedaling is real). Two compounding weaknesses underneath: every `tau` is
`locked:false` (hand-set, uncalibrated), and error bars propagate *assumed* noise (onset 10ms / RMS
0.3dB / pedal CC ±10), which GATE 1 shows understates true AMT match-churn (13–20% bar loss on
bit-identical audio). The localization/coverage gate, the typed-abstention machinery, and
pedaling-presence are the parts that ARE solid.

---

## GATE 3 UPDATE (#101, 2026-06-26): dynamics rescued — degeneracy was the statistic, not the dimension

The "dynamics ± → 0% by construction" degeneracy above is **superseded**. It was an artifact of the
*chosen statistic* (`std/range`, a non-negative unitless dispersion measure against a dB tau), not a
property of the dimension. Replacing it closed both hard gates for dynamics@whole_piece.

**The empirical chain (all measured, not assumed):**
1. *Statistic sweep* (PercePiano, n=1202, halo-controlled partial-ρ vs perceived dynamics): every
   **gain-invariant** statistic fails the ~0.5 ceiling (std +0.04, range −0.01, IQR −0.05, crest
   −0.09, dB-range −0.34 wrong-signed); every **passing** statistic is gain-bound *level*
   (mean-velocity +0.562, median +0.533, soft-floor +0.523, mean-dB +0.558). The old `std/range`
   sits at perceptual-null (+0.042). **Perceived "dynamics" is a loudness-LEVEL judgment, not a
   contrast/spread one** (a publishable sub-finding).
2. *Substrate trap*: absolute audio level is recording-gain-bound (the headline corpus is heterogeneous
   YouTube audio → gain = mastering artifact, not the pianist), and frame-RMS conflates strike velocity
   with note density (rendered-audio mean-RMS partial-ρ only +0.16). Neither is usable.
3. *The bridge*: the verifier bundle already carries AMT-transcribed `notes` with per-note **velocity**
   — an onset-level, density-free, gain-robust analog of the GATE-2-validated mean velocity. **The
   transcription model's auxiliary velocity head is itself a non-LLM perceptual measurement** (stays
   inside the non-circularity rule).

**G-B (perceptual validity, end-to-end on rendered audio).** Fixed-gain piano render (fluidsynth +
MuseScore_General.sf3) → aria-amt → mean note-velocity, correlated vs PercePiano perceived dynamics:
**partial-ρ 0.544, n=180, 95% CI [0.417, 0.655]** — at the inter-rater ceiling and *statistically
indistinguishable from ground-truth MIDI velocity* (0.525). AMT↔GT velocity 0.965 (loudness order
preserved; the RMS density/normalize failure mode is dead). AMT transcription noise costs only ~0.05
partial-ρ. Report: `model/data/results/gb_amt_velocity_gate.json`.

**G-A (non-degeneracy, construction-known).** Velocity-scaling corruption ×{0.55, 1.0, 1.45} → render
→ AMT → the *real* `DynamicsMeasurer` + *frozen* `route_verdict` (n=30): monotonicity 30/30 (median d
−23.8 / −0.9 / +17.9), performance-flip **0.85** (≥0.80), polarity-shuffle collapse **0.22** (≥0.20,
aligned 0.85 → shuffled 0.37). Report: `model/data/results/ga_dynamics_velocities.json`.

**Production change** (commit on `issue-101-dynamics-signed-level`): `measurers/dynamics.py`
whole_piece `d = mean(bundle note velocities) − 51.5` (corpus-median ref); region
`d = mean(region vel) − mean(all vel)`. **Velocity units for BOTH tiers** (one tau — fixed a latent
unit-mismatch where region was dB and whole_piece unitless). librosa/RMS removed. `verdict_dispatch.py`
untouched (frozen). Taxonomy tolerance → `provisional 8.0, unit midi_velocity, locked:false`.

**Net GATE-3 reality now:** pedaling-presence (solid) **+ dynamics-level (G-A+G-B pass)** are real
signals; only **timing@whole_piece remains degenerate**. Residual dynamics caveats: tau=8 is provisional
(front 4 calibrates — drives the flip⁺ 22/30 < flip⁻ 29/30 asymmetry), error bar uses an *assumed*
velocity-quantization jitter (front 5 / G-C measures the real AMT churn), and the scope is controlled-gain
audio (AMT velocity calibration on heterogeneous real recordings is a separate G-F question).

---

## FRONT 3 UPDATE (#101, 2026-06-26): pedaling signed + statistic-corrected, scoped to under-pedal

Front 3 had two jobs: (a) make `-` (under-pedaled) claims **adjudicable** instead of structurally
auto-REFUTED, and (b) run pedaling **G-A** for the first time (the prior "provisional PASS" was never
empirically measured — only dynamics had the harness). Both are done; G-A revealed a hard substrate limit.

**Statistic correction (the latent cousin).** The shipped whole_piece statistic was pedal-**bar**-fraction
(fraction of bars with ≥1 CC64 event), but GATE-2's 0.478 was measured on CC64 **time**-on-fraction
(`pedaling_on_fraction`, `gate2_expert_anchor.json`). These are cousins, not the same statistic — the
exact trap the dynamics rescue caught. Both tiers now use **time-on-fraction**, so the verifier checks
the same *family* of statistic. **CORRECTED by FRONT 4 (below):** this section claimed "0.478 is
inherited verbatim (frac − const is monotone in frac)" — that is WRONG. The monotonicity argument holds
only if the verifier's **AMT** on-fraction equals the **MIDI** on-fraction 0.478 was measured on; it does
not (lossy AMT pedal head). Measured directly, AMT pedaling G-B is only **0.181**, not 0.478. The signed
under-pedal *construction-known* detection (G-A) still holds; the *perceptual* validity (G-B) does not.
The original (now-corrected) reasoning:
- whole_piece: `d = on_fraction − REFERENCE_FRACTION`, `REFERENCE_FRACTION = 0.4623` (corpus-median AMT
  on-fraction over the 30 fixed-gain neutral renders; AMT-derived, not MIDI-native, because AMT inflates
  low pedal and saturates high — a MIDI median would mis-zero the AMT-substrate measurement).
- region: `d = region_on_fraction − whole_piece_on_fraction` (within-clip, gain-free).
- `minimum_events: 0` (deliberate): a zero-pedal performance now measures `d = −0.4623` (an informative
  signed measurement), NOT an abstention — this is what unblocks `-` claims. (The frozen router's Step-5
  `event_count < minimum_events` gate would otherwise still force UNVERIFIABLE.)

**Signed test works end-to-end** (real `PedalingMeasurer` + frozen `route_verdict`): dry+`-`→SUPPORTED
(was auto-REFUTED), wet+`+`→SUPPORTED, both wrong polarities→REFUTED.

**G-A (construction-known, maximal corruption, n=30).** sparse = remove ALL CC64 (true dry, frac→0),
dense = full pedal down (frac→1), neutral = original; render → aria-amt → real measurer + frozen router.
First verified AMT **preserves on-fraction ordering** (the gating risk; sparse<neutral<dense held). Result
is **directional**:
- **under-pedal `-` PASS**: performance-flip **0.90** (dry `-` SUPPORTED & wet `-` not, 27/30), polarity-
  shuffle collapse **+0.23** (aligned 0.93 → shuffled 0.30), direction-ordering 29/30.
- **over-pedal `+` FAIL**: flip⁺ only 7/30. AMT pedal on-fraction **SATURATES** (~0.55 ceiling): even
  full-pedal (true frac=1.0) reaches only median `d = +0.214 < tau`. Over-pedaling is physically
  unrecoverable from audio, regardless of tau (a tau sweep 0.05–0.25 never lifts flip⁺ above 0.27).
  Unlike velocity (AMT↔GT 0.965, linear), AMT pedal is a saturating detector.

**Scoping (production).** `pedaling.substrate_insensitive_polarity = {whole_piece: "+"}`; the **orchestrator**
(not the frozen router) returns `UNVERIFIABLE(substrate_insensitive_direction)` for whole_piece `+`
pedaling claims, so a true over-pedaling observation is abstained-on, not wrongly REFUTED (which would
poison the G-D rate). The router is untouched.

**Net:** pedaling whole_piece is now a real, G-A+G-B-passing signal **for under-pedal detection**; over-pedal
is scoped out as substrate-insensitive. Residual caveats: tau=0.25 provisional (front 4 calibrates;
under-pedal flip is 0.90 at tau=0.25 but degrades for gentler corruptions), error bar uses *assumed*
boundary jitter (front 5 / G-C measures real re-transcription churn), region-tier saturation is unmeasured
(only whole_piece was tested), and virtue-vs-excess is moot now that `+` is scoped out. Reports:
`model/data/results/ga_pedal_onfraction.json` (scaled) + `ga_pedal_onfraction_maximal.json` (the gate);
harnesses `model/src/claim_measurement/ga_validation/amt_pedaling_ga_{render,metrics}.py`.

---

## FRONT 4 UPDATE (#101, 2026-06-27): tau calibration — dynamics LOCKED, pedaling G-B corrected

Front 4 calibrated the two `tau` against human-labeled anomaly slices and flipped `locked:true`.
Method (`tau_calibration/tau_calibrate.py`): signed-anomaly detection — a clip is a `+` anomaly if its
composite perceptual rating > Q75, `-` if < Q25, else normal; the verifier fires ±1 if `d` clears ±tau;
`tau* = argmax(TPR − FPR)` (Youden), with bootstrap CI and per-direction tau. n=180 on the AMT substrate.

**Dynamics — tau 8.0 → 6.5, LOCKED.** Youden point estimate 6.35 (boot 95% CI [4, 17] — wide because the
composite "dynamics" label is a *quality* rating, only a noisy proxy for loudness level). The directional
medians are clean (`+anom +5.7 / −anom −9.3 / normal +0.1`) and the per-direction optima (loud ~1.4 <
soft ~5.0) explain the front-3 flip⁺<flip⁻ asymmetry: tau=8 was too high, especially for loud. Cross-checked
on the construction-known G-A set — lowering tau 8→6.5 *raises* flip 0.85→0.90, narrows flip⁺ 22→24/30, and
lifts shuffle-collapse 0.22→0.30. 6.5 is the symmetric compromise satisfying both human-calibration and G-A.

**Pedaling — G-B FAIL on AMT; tau NOT locked (correction of a front-3 claim).** Front 3 asserted pedaling
"G-B inherited 0.478 (frac − const is monotone in frac)". That argument is **wrong**: it assumes the
verifier's *AMT* on-fraction equals the *MIDI* on-fraction that 0.478 was measured on. Measured directly on
the production substrate (rendered audio → aria-amt, n=180 natural performances), pedaling's halo-controlled
partial-ρ is only **0.181 [boot 95% CI 0.03–0.33]**, vs **0.386** for true MIDI on-fraction on the same
clips — below the ~0.5 ceiling. Root cause (`gb_pedaling_amt_check.py`): the AMT pedal head recovers true
on-fraction at only **Spearman 0.389** (regression-to-the-middle — +0.19 hallucinated pedal on dry clips,
−0.34 on full-pedal clips), and `AMT ⊥ perception | TRUE = +0.106` (adds only noise). **Decisive contrast:**
on the *same* fluidsynth→AMT renders the AMT *velocity* head recovers GT at **0.965** — so it is the
weakly-supervised pedal head, not the render or postprocessing. **Pedaling is therefore NOT perceptually
validated on the AMT substrate.** It stays `active` only for the COARSE construction-known under-pedal
detection that G-A established (the verifier responds to large pedal swings), `tolerance.locked:false`,
pending a better pedal-transcription substrate. The under-pedal tau itself calibrates to ~0.16–0.29
(consistent with the provisional 0.25), but locking is premature while the dimension's perceptual validity
is unestablished. Reports: `model/data/results/tau_cal_pedaling.json`; harnesses
`model/src/claim_measurement/tau_calibration/`.

---

## FRONT 5 / G-D UPDATE (#101, 2026-06-27): the first faithfulness-rate attempt — dynamics rate is UNMEASURABLE (construct mismatch)

G-D is the paper milestone: the first measured per-dimension faithfulness RATE. Because pedaling FAILS
G-B on AMT (front-4) and timing is degenerate, G-D was scoped **dynamics-only, whole_piece**. The result
is a clean, fully-reproducible **negative**: the rate cannot be measured on this generator — not for lack
of corpus or CI, but because the generator never makes a claim the G-B-validated statistic can adjudicate.

**Pairing resolved (the front-3 blocker).** The generator prose and the cached bundles had ZERO same-clip
overlap. Fix = *extract-on-baseline-audio*: `baseline_v1.jsonl` (legacy `synthesize` generator) has **94
chopin_ballade_1 performances** whose `recording_id` has BOTH prose AND local `skill_eval` audio (162 prose
docs total; the other 10 baseline pieces lack local audio, so the corpus is single-piece — a G-F/G-E
generalization limit, not a G-D one). This clears the ≥30-distinct-performances floor.

**Pipeline (truth label deterministic; LLM only extracts).** (1) In-process aria-amt transcription of
stratified 27s windows (`gd_rate/transcribe_bundles.py`) → minimal bundles carrying pooled `notes` with
velocity (stub `measure_table`/`anchors` clear LocationResolver's whole_piece precondition but are never
read by the whole_piece dynamics measurement). On MPS, dense-polyphony decode is ~30–130s/window, so full
coverage of ~8.6-min clips is ~28h/94; 3 stratified windows ([0.15,0.45,0.75]·dur) remove the temporal
bias (Ballade mean velocity varies 47–78 by section) at ~57s/clip. (2) Sonnet-4.6 extractors decompose
prose into dynamics claims (`gd_rate/extract_prompt.md`). (3) `gd_rate/route_and_score.py` runs the real
`verify()` + frozen `route_verdict` (tau 6.5) and computes the rate + performance-clustered bootstrap CI +
abstention histogram.

**The measured yield (the headline, `model/data/results/gd_dynamics_rate.json`):**

| quantity | value | read |
|---|---|---|
| dynamics claims | 146 (162 docs, 94 perfs, 83 with ≥1) | prose decomposes fine |
| bar-localized | **0 / 146** | corroborates front-3's 0.4% localization on a fresh corpus+dimension |
| subtype split | **131 contrast (90%) / 15 level / 0 ambiguous** | the generator talks *shaping*, not *level* |
| level-claim scope | **0 overall-loudness / 8 register-specific / 7 non-falsifiable** | even the "level" claims aren't whole-piece-overall |
| **in scope** for whole-piece mean velocity | **0** | committed n=0 → **rate UNMEASURABLE** |
| abstention histogram | `out_of_scope_statistic`: contrast 131, register 8, non-falsifiable 7 | 100% abstention |

**Why this is a construct mismatch, not a supply-size problem.** The only G-B-passing dynamics statistic is
whole-piece mean *level* (the doc's statistic sweep proved every contrast/spread statistic fails the ~0.5
ceiling: std +0.04, range −0.01, IQR −0.05). But the generator's dynamics feedback is *about contrast/
shaping/range* (90%) or *register-specific/exploratory* ("make the pp more hushed", "push the forte",
"how soft can you go?" — 10%). These address a construct the validated statistic cannot test even in
principle. The 17 transcribed bundles span mean velocity 47–78, so the statistic IS live and discriminating
on this corpus — the gap is purely **claim supply of the in-scope construct**. Because level claims are a
fixed ~7% of dynamics claims and only ~96 performances have local audio, no corpus expansion within reach
clears the ≥30-committed gate; the binding constraint is the generator's claim distribution.

**Net.** This extends the GATE-3 narrative one layer: front 1–4 *rescued the dynamics statistic*
(G-A+G-B pass); G-D shows the *rate* is still unmeasurable because of a generator↔statistic **construct
mismatch**. "An LLM piano teacher's dynamics feedback is ~90% contrast and ~0% falsifiable whole-piece
loudness, so a perceptually-validated loudness-LEVEL verifier has no claims to score" is itself a measured,
publishable finding — the faithfulness-rate analog of the localization-yield finding. **G-D stays NOT
PASSED.** Forks (a user decision, none on the validated critical path): (i) reframe the paper's measured
contribution around the claim-supply/validity gap; (ii) open a new G-B front to validate a dynamic-*contrast*
statistic so the 131 contrast claims become adjudicable; (iii) better pedal substrate to revive pedaling.
Harnesses: `model/src/claim_measurement/gd_rate/`; regenerable inputs: `model/data/results/gd_rate/`.

---

## FRONT 6 / contrast-G-B UPDATE (#101, 2026-06-27): no deterministic dynamic-CONTRAST statistic validates — dynamics is DOUBLY blocked

Front 5 (G-D) showed the generator's dynamics feedback is ~90% CONTRAST/shaping and ~0%
falsifiable whole-piece LEVEL — so the G-B-validated mean-velocity LEVEL statistic has no
claims to score. The obvious rescue is to perceptually-validate a CONTRAST statistic so the
131 contrast claims become adjudicable. Front 6 tested that and it **does not work**.

**The bet vs the earlier sweep.** The GATE-3 statistic sweep already showed *global-spread*
contrast statistics fail to predict perceived dynamics (= PercePiano `dynamic_range`): std
+0.04, range −0.01, IQR −0.05. But "ebb and swell" is a *temporal-envelope* property a global
std is blind to. So front 6 added a *temporal-shaping* family — swell depth (peak-to-trough of
the time-binned velocity envelope), envelope-modulation std, quadratic arc curvature, terracing
(envelope direction-changes), normalized swell — and correlated all of them (halo-controlled
partial Spearman, control = mean of the other 18 dims) against all 19 granular PercePiano dims,
n=1202. Harness `model/src/claim_measurement/contrast_gb/sweep.py`; report
`model/data/results/contrast_gb_sweep.json`. Sanity: mean_velocity vs `dynamic_range` raw
ρ 0.569 reproduces GATE-2 (0.625), confirming the index mapping and method.

**Result: NO-GO.** Every shaping statistic's best partial against any dynamic-shaping dim
(`dynamic_range`, `drama`, `mood_energy`, `sophistication`, `timbre_loudness`, `balance`) is
**≤0.26** (swell_depth_norm↔sophistication; most ≤0.19) — far below the ~0.5 inter-rater ceiling.
The only ≥0.45 correlations the temporal-shaping statistics achieve are with **`tempo`** (swell_depth
0.467, envelope_std 0.467) — a note-density/speed confound, not dynamic shaping. Perceived dynamic
shaping on 4–8-bar segments does not reduce to a deterministic MIDI-velocity-envelope statistic
(likely because "drama"/"shaping" integrates timing + pedaling + timbre jointly, not loudness alone;
and PercePiano segments are too short to express a whole-piece arc).

**Net — dynamics is DOUBLY blocked, and the two blocks are complementary:**
- **LEVEL**: a validated statistic exists (mean velocity, partial-ρ 0.54) but the generator makes
  ~0 in-scope claims → rate unmeasurable (claim-SUPPLY gap, front 5).
- **CONTRAST**: the generator makes ~90% of its dynamics claims here, but no deterministic statistic
  validates against perception → no non-circular truth label possible (measurement-VALIDITY gap, front 6).

A dynamics faithfulness RATE is therefore unattainable today by either route. This is decisive
evidence for reframing Path #1's measured contribution around the **claim-supply / measurement-validity
gap itself** (localization-yield + the dynamics double-block + the typed-abstention machinery) rather
than a headline per-dimension faithfulness number — unless a fundamentally different substrate lands
(a multi-modal learned shaping predictor, which reintroduces the circularity burden; or longer-segment
human shaping labels). Caveats (leave a narrow door): single-velocity-feature only; PercePiano segments
are short; shaping labels are themselves ~0.5-noisy.

---

## Dimension status matrix — what is verifiable today (#101, 2026-06-27)

The honest, consolidated picture across every dimension the teacher emits feedback on. "Verifiable"
= a deterministic, non-LLM statistic that (a) clears halo-controlled partial-ρ ~0.5 vs human
perception AND (b) survives the production AMT-audio substrate AND (c) has falsifiable generator
claims to score. **Today exactly zero dimensions are fully verifiable end-to-end** — each fails at a
different stage, and knowing *which* stage is the contribution.

| Dimension | Measurer built? | G-A (non-degen) | G-B perceptual validity | Substrate (AMT) | Claim supply | Net status |
|---|---|---|---|---|---|---|
| **dynamics — LEVEL** | yes (mean velocity) | PASS (flip 0.90) | **PASS** (ρ 0.54) | **survives** (AMT≈GT 0.965) | **~0 in-scope** (front 5) | validated stat, **no rate** — supply gap |
| **dynamics — CONTRAST** | no (swept) | — | **FAIL** (≤0.26, front 6) | — | 90% of claims | **no validatable stat** — validity gap |
| **pedaling** | yes (CC64 on-frac) | PASS under-pedal | PASS on MIDI (0.48) | **FAILS** (AMT 0.18, front 4) | moderate | **substrate-blocked**; coarse under-pedal only |
| **timing** | yes (IOI-CV) | FAIL | marginal (0.25, < ceiling) | perceptual stat degenerate; **onset substrate clean** (AMT−GT 4.7ms ≪ 30ms, n=50 #101) | n/a | **degenerate for perception**; onset substrate-viable for the score-relative #64 path |
| **articulation** | **no** (`gated_on_measurement`) | — | never tested | **AMT dur ✓ bar-mean** (clip-mean ρ0.90; per-note 0.76, n=50 #101) | — | perception unbuilt; **duration substrate-viable** (WASM perf/score ratio, #64) |
| **timbre** (4 PercePiano dims) | **no** (`scoped_out`) | — | never tested | — | — | **unexamined**; needs SPECTRAL features, not velocity/MIDI |
| **phrasing** | **no** (`scoped_out`) | — | never tested | — | — | **unexamined**; structural, likely hardest |
| **interpretation** (+sophistication) | **no** (`scoped_out`) | — | never tested | — | — | **unexamined**; most holistic/subjective |

**PercePiano granular dims probed for a deterministic correlate:** only `dynamic_range` (level ✓ /
contrast ✗), `timing` (0.25), `pedal_amount/clarity` (MIDI ✓ / AMT ✗). The other 14 — `timbre_*`,
`tempo`, `space`, `balance`, `drama`, `mood_valence/energy/imagination`, `sophistication`,
`articulation_length/touch` — have **never been probed with their natural features** (front-6's velocity
sweep touched some incidentally and they were ≤0.26, but velocity is the wrong feature for timbre/tempo/
mood). So "unexamined" ≠ "impossible": timbre→spectral-centroid, articulation→key-overlap, tempo→beat-
tracking are plausible unbuilt candidates; mood/sophistication/interpretation/drama are likely
deterministically-unmeasurable (they integrate everything).

**App implication (not just paper).** The deterministic verifier can stand behind essentially **no**
expressive feedback today: dynamics-level is validated but the teacher speaks in contrast terms;
pedaling works only for coarse construction-known under-pedal on clean audio; everything else is
LLM-asserted, not ground-truth-checked. Any "grounded feedback" claim the app makes is currently
honest only for that thin slice. The cheapest expansions of the verifiable surface, in rough order of
likely payoff: (1) articulation measurer (MIDI key-overlap is well-defined and AMT gives note on/off);
(2) timbre via spectral features (audio-native, no AMT-velocity dependence); (3) a better pedal substrate
to rescue pedaling; (4) a beat-tracker to rescue timing. Dynamics-contrast and the mood/interpretation
dims look deterministically out of reach.

**Complement — score+reference grounding (the product path, #64).** Everything above grounds against
*perception* (PercePiano). There is a SECOND, independent grounding already built in the product:
`apps/api/src/wasm/score-analysis/` compares the performance against the **notated score** and against
**reference-performer profiles** (MAESTRO-derived, `model/src/score_library/reference_cache.py`) — perf-vs-
notated velocity + crescendo shape, onset deviation (rush/drag), perf/score duration ratio (legato/staccato),
pedal on-fraction, all "within/outside reference range". It needs NO perception labels and the neural
`model_score` is decorative. This sidesteps the perceptual-validity bottleneck entirely: the relevant test
becomes **AMT-fidelity** (does the AMT measure recover the ground-truth-MIDI measure? — velocity ✓0.97,
pedal ✗0.39, **onset ✓ and offset/duration ✓** as of 2026-06-27, #64), not perceptual correlation. The
onset/duration map (`model/src/claim_measurement/amt_fidelity/`, **n=50** PercePiano fluidsynth→aria-amt renders,
median note recall 0.99, 5731 pooled notes): **onset noise 4.7ms** (std of AMT−GT onset; bias +2.0ms is a removable constant)
vs the ±30ms rush/drag band — a **~6× margin, so timing is substrate-viable**; **note duration recovers at
clip-mean-dur ρ 0.90** — the bar-mean statistic `analyze_articulation_tier1` actually consumes, cf velocity
0.965 — with per-note ρ 0.76, so
**articulation is substrate-viable at the bar-aggregate level the engine uses**. Critically, neither the AMT
onset nor the offset head behaves like the saturating pedal head (0.39); the untested-offset risk does not
materialize. Confirmed at n=50 (#101, 2026-07-04, disk pre-checked): the earlier n=15 numbers (onset 4.2ms,
clip-mean-dur ρ 0.925, per-note 0.57 w/ one 3.2× outlier) held to within noise — onset 4.2→4.7ms, ρ 0.925→0.90,
per-note 0.57→0.76 (the outlier averaged out); verdict unchanged. The tier-1 (score+reference) path is now WIRED in the live
practice path (#64, 2026-06-27): `finalizeChunk` calls `alignChunkNotes` -> `analyzeTier1`, so per-bar
onset/duration deviations vs the notated score reach the teacher (a new Rust `align_chunk_notes` derives
per-note `onset_deviation_ms` from the chroma frame-warp + an affine tempo fit; coarse/directional by
design). Remaining: reference-performer profiles still deferred (score-relative only), and prod stays on
Tier 3 until a deliberate deploy sets `AMT_ENDPOINT`. Verified at unit + integration + DO-suite level AND
live local-AMT-session E2E (2026-07-05): tier-1 engages on the real DO path — alignment spans multiple
advancing bars, `chunk_bar_map` reaches the client, and signed per-note rush/drag reaches the teacher. The
live run additionally surfaced and fixed two workerd `serde_wasm_bindgen` ABI marshaling bugs in the
`analyze_tier1` path (cargo tests miss the class — they never cross the workerd boundary). Signal is coarse
on diatonic material (±200-360ms per-note residuals, #21 chroma-gap limit). See `docs/model/04-north-star.md` → "Two grounding philosophies" for the full
strategic framing and the "no neural encoder" possibility.

---

## Path #1 operating mode and hard gates (#101)

**Operating mode (re-scoped 2026-06-24).** Open-ended, no time limit. The fixed M0–M3 timeline in the
plan §6 is SUPERSEDED. We treat the GATE-3 baseline above as ground truth and iterate each measurement
FRONT — rebuild statistics, recalibrate thresholds, expand corpus, measure error empirically, acquire
error-rich audio — until the per-dimension faithfulness numbers are publishable-proud. The original
paper thesis is unchanged: *non-circular, deterministic faithfulness ground truth in a soft perceptual
domain* (music as the existence proof). **No paper is drafted until every hard gate below passes.**

**Hard gates (ALL must pass before drafting the paper).** Thresholds are proposed defaults
(adjustable); the gate STRUCTURE is the commitment.

- **G-A — Non-degeneracy / measurement-drivenness.** For every reported (dimension, location-tier),
  the verdict must depend on the performance. Proven by two controls on a construction-known set:
  (i) *polarity-shuffle* — permuting claim polarity must collapse the faithfulness signal toward
  chance (proposal: shifts |rate−0.5| by ≥0.20); (ii) *performance-flip sensitivity* — on
  corruption-harness clean-vs-corrupted pairs, the verdict flips in the expected direction in ≥0.80
  of construction-known cases. A dimension whose verdict is unmoved by either control FAILS.
  *Status: **pedaling construction-known PASS for under-pedal `-`** (#101 front-3: performance-flip 0.90,
  polarity-shuffle collapse +0.23, ordering 29/30, maximal corruption n=30) — but note this is
  construction-known detection only; pedaling FAILS G-B on natural data (front-4, below). Over-pedal `+`
  FAILS (AMT pedal saturation), scoped out as substrate-insensitive; **dynamics PASS** (#101:
  performance-flip 0.90 at the locked tau=6.5 / 0.85 at the old tau=8, polarity-shuffle collapse 0.30,
  monotonicity 30/30 — see GATE 3 + FRONT 4 UPDATE); timing FAIL (degenerate).*
- **G-B — Perceptual validity of the EXACT statistic.** The specific statistic the verifier checks
  (not a cousin) clears halo-controlled partial-Spearman ≥ the measured PercePiano inter-rater
  ceiling (~0.5) on its perceptual dimension, p<1e-6. *Status: **pedaling FAIL on the AMT substrate**
(#101 front-4: AMT on-fraction partial-Spearman **0.181** [CI 0.03–0.33], n=180 natural — the GATE-2
0.478 was MIDI-native and does NOT transfer through the lossy AMT pedal head, which recovers true
on-fraction at only 0.389; see FRONT 4 UPDATE). Pedaling stays active for construction-known detection
only, not graded perceptual claims;
  **dynamics PASS** (#101: switched the statistic to mean AMT note-velocity → partial-Spearman 0.544,
  n=180, 95% CI [0.417, 0.655], indistinguishable from ground-truth MIDI velocity 0.525 — see GATE 3
  UPDATE below); timing FAILS (0.25) → out unless a new proxy passes.*
- **G-C — Empirical (not assumed) error bars.** Substrate error per dimension MEASURED by
  re-transcribe→re-measure variance over ≥10 clips; the near-threshold dead-band set to ≥ the
  measured 1σ. No assumed-noise error bars in the headline. *Status: **dynamics DONE** (#101,
  2026-07-04; `model/src/claim_measurement/gc_error_bars/`, n=12). Re-transcribed each clip under
  perceptually neutral recording nuisances (sub-JND ±0.5 dB gain jitter + 40 dB-SNR additive noise;
  aria-amt decodes greedily so identical audio is a no-op — churn is nuisance-driven, verified per
  clip). Key finding: the mean-velocity churn is **correlated across notes** (a global gain shift
  moves every note together), so the statistic 1σ (0.68 median / 1.39 p90 / 2.39 max velocity units)
  is ~5× the old assumed `VELOCITY_QUANT_STEP/√12/√N` bar (0.14 at N≈103) and does NOT shrink with N.
  Replaced the placeholder with a two-term `substrate_var = max(σ_note²/N, floor²)` (σ_note 2.69,
  floor 1.39, both measured p90) in `DynamicsMeasurer` — dead-band now ≥ measured 1σ at every note
  count; frozen router untouched; `claim_taxonomy` suite still 125-green. Timing G-C deferred (its
  measurer, FRONT 7b, has not landed); pedaling is substrate-blocked (front 4) so no rate to bar.*
- **G-D — Claim supply for a stable rate.** Per reported dimension: ≥30 distinct performances and a
  bootstrap 95% CI half-width ≤0.05 on the faithfulness rate; report yield + the abstention
  histogram (`out_of_scope_dim`, `gated_dim`, `unlocalizable`, `low_coverage`, `region_too_short`,
  `near_threshold`). *Status: **MEASURED — NOT PASSED, supply-blocked** (#101 front-5 / #67). Pairing
  resolved (extract-on-baseline-audio: 94 chopin_ballade_1 performances with both baseline_v1 generator
  prose AND local skill_eval audio). 146 dynamics claims LLM-extracted → **0 in scope** for the
  G-B-validated whole-piece mean-velocity statistic → committed n=0, rate UNMEASURABLE. The blocker is
  a generator↔statistic CONSTRUCT MISMATCH, not corpus size or CI. See the FRONT 5 / G-D UPDATE below.*
- **G-E — Reproducibility.** The per-dimension rate reproduces within CI across (a) a generator
  re-run and (b) two disjoint clip halves; a second independent decomposition (different LLM or a
  human-checked slice) agrees on (dim, polarity, location) at Cohen's κ ≥0.6. *Status: not started.*
- **G-F — Generalization honesty (the §7 dominant risk).** EITHER localization + ≥1 dimension
  validated on error-rich/student audio, OR the paper scope is explicitly limited to
  clean/professional audio with the limitation stated and the localized-claim contribution demoted.
  All in-repo audio is currently clean/pro (`ood_practice/` empty). *Status: not started.*

**Improvement fronts (the backlog — work in any order, one at a time, measure-keep-or-revert).**
1. Redesign whole_piece **dynamics** statistic → signed loudness *level* vs a reference (GATE-2-
   validated), dB-matched tau. (unblocks G-A, G-B for dynamics)
2. Redesign / scope whole_piece **timing** → real beat-tracked tempo-stability statistic, or formally
   scope whole_piece timing OUT until a beat tracker exists. (G-A, G-B) **SUPERSEDED 2026-07-04 by
   FRONT 7 below: the paper's timing route is now SCORE-RELATIVE onset deviation (substrate cleared
   #64), not a perception-anchored beat-tracker statistic.**
3. Add a **signed whole_piece test per dimension** so `-` claims (54% of validated-scope) are
   adjudicable instead of auto-REFUTED.
4. **Calibrate the three tau** against human-labeled anomaly/no-anomaly slices; flip `locked:true`. (G-B)
5. **Measure substrate error** empirically; widen the dead-band. (G-C)
6. **Resolve pairing + scale corpus** for claim supply and CI width. (G-D)
7. **Acquire/simulate error-rich student audio.** (G-F)
8. Disambiguate pedaling **virtue vs excess** (cleanliness/density bands).

**The improvement loop (per front).** Goal: move a named gate from FAIL→PASS (or tighten a passing
margin). Scope: one front; never modify `verdict_dispatch.py` or its tests (frozen). Metric: the
gate's measurable criterion above. Verify: `cd apps/evals && uv run --extra all pytest claim_taxonomy/ -q`
(113 passing) green AND the gate metric improved on a held-out construction-known set. Guard: extractor
is LLM-OK but the truth label is NEVER an LLM; TDD; revert any change that does not move its gate.

---

## FRONT 7 + dependency graph — the path to the first measurable rate (#101, 2026-07-04)

**Mental-model updates this section records (supersedes earlier scope notes):**
1. **The #64 score-relative substrate is now ON the paper critical path.** The 2026-06-26 note
   ("#64 is PRODUCT-path, NOT paper critical path") is superseded: fronts 5/6 proved
   perception-anchoring cannot produce a headline rate for the claims teachers actually make, while
   the AMT-fidelity map (onset 4.7ms ≪ ±30ms band; duration bar-mean ρ0.90, n=50) cleared the
   score-anchored route. The paper's contribution is the SYSTEM: two grounding backends
   (perception-anchored where a validated statistic exists; score-anchored where a score exists)
   plus legible abstention everywhere else.
2. **Timing's "degenerate" verdict was statistic-specific, not dimension-specific.** IOI-CV vs
   perception is dead; onset-deviation-vs-score is a different statistic with truth by construction
   ("rushing" is definitionally score-relative). Kill-verdicts get re-checked when new substrate
   evidence lands.
3. **The perfect system verifies a wide KNOWN subset and abstains legibly** — dynamics-contrast and
   the holistic dims (mood/interpretation/sophistication/drama) stay out BY DESIGN; the abstention
   histogram is a first-class output, not an apology.

**FRONT 7 — score-relative timing verifier (the keystone front).** Unique intersection of
substrate-proven + supply-rich (rush/drag is the most common teacher claim) + machinery-built
(`align_chunk_notes` / offline aria-AMT→parangonar bar-map). Success = the first measurable
per-dimension faithfulness rate = the paper exists. Steps, in order:
- **7a. Claim-supply probe FIRST (hours; GO/NO-GO).** Extract timing claims from the 162
  baseline_v1 prose docs (the front-5 lesson institutionalized: check supply BEFORE building the
  statistic). If timing claims are also ~90% out-of-scope for an onset-deviation statistic, the
  front re-plans before any measurer work.
- **7b. Offline timing measurer**: signed onset_deviation_ms vs the aligned score (offline
  AMT→parangonar path, NOT the live DO), whole_piece + bar tiers.
- **7c. G-A by construction**: tempo-warp corruptions (rush/drag injected in MIDI, render, AMT,
  measure) — performance-flip ≥0.80 + polarity-shuffle collapse ≥0.20.
- **7d. Construct validity + tau**: verify the teacher's "rushing/dragging" sentences map onto the
  statistic; calibrate tau (start unlocked).
- **FRONT 8 (after 7b skeleton): articulation** via perf/score duration ratio — rides the same
  alignment infra; two dimensions make the rate a result, not an anecdote.

**Dependency graph (what is unblocked TODAY vs blocked, and by what):**

```
UNBLOCKED NOW — parallel lanes:
  L1 (critical): 7a supply probe (hours) ──GO──> 7b measurer -> 7c G-A -> 7d tau
  L2 (mechanical, independent): [DONE #101 2026-07-04] G-C error bars — dynamics MEASURED (n=12),
                                  correlated-floor two-term model wired into DynamicsMeasurer
                                [DONE #101 2026-07-04] n=50 confirmatory AMT-fidelity run (onset 4.7ms,
                                  dur ρ0.90; verdict unchanged)
                                [NOT A CODE TASK — score-loader already handles variable-tempo/non-4/4
                                  since #98, regression-tested] chopin_ballade_1 is BLOCKED ON DATA, not
                                  code: no local ASAP score MIDI AND no local audio (practice_eval/
                                  chopin_ballade_1/ has only candidates.yaml). Unblock = clone ASAP +
                                  acquire audio, OR generate fresh prose on the Bach clips (on-bundles).
  L3 (product/shared): #64 live local-AMT session verify (de-risks the alignment machinery)

BLOCKED — and by what:
  timing G-D rate        <- 7b-7d + prose+audio+SCORE triple overlap (score-loader OR generate-on-bundles)
  articulation verifier  <- 7b skeleton (soft; reuses infra + lessons)
  G-E reproducibility    <- a rate to reproduce (G-D)
  G-F student audio      <- acquisition (ood_practice/ empty); independent, deferred
  paper draft            <- ALL hard gates
  pedal rescue           <- a better pedal-transcription substrate (model work, external to verifier)
  timbre front           <- deliberate deferral (research gamble; audio-native spectral features)
  paper #2 (RLVR)        <- paper #1 verifier adjudicating something real
```

**Priority ruling:** 7a first (same day), then 7b–7d as the main thread; L2 + L3 items run in
parallel at any time (all independent). Defer timbre / pedal / G-F until the first rate exists.
Known external risk: both MIREX tracks (#104/#105, Oct 1 deadline) compete for the same sessions —
FRONT 7 is weeks of work and loses by default unless explicitly scheduled.

## FRONT 7a UPDATE (#101, 2026-07-04): claim-supply probe = **NO_GO**. 7b–7d NOT STARTED.

**Gate status change:** FRONT 7's signed onset-deviation-vs-score route is **supply-blocked**, the
same failure mode as dynamics G-D. The keystone-front premise ("supply-rich: rush/drag is the most
common teacher claim") is **FALSIFIED on the generator corpus.** No measurer was built (the front-5
lesson held: the probe stopped the front before 7b).

**What was measured (harness `model/src/claim_measurement/timing_supply/`, LLM extract → deterministic
classify; truth path is not an LLM):**
- **Corpus A — chopin_ballade_1** (the exact 162-doc / 94-perf front-5 corpus): 77 timing claims across
  55 perfs. Subtype: **rubato 70 (91%) / evenness 5 (6%) / rush_drag 2 (3%) / note_value 0 / hesitation
  0 / ambiguous 0.** In-scope for signed onset-deviation (`rush_drag`, whole_piece|bar) = **2 claims,
  both `neutral` polarity** ("your timing is steady/solid"). **0 directional rush-or-drag. 0
  bar-localized** (100% whole_piece — corroborates the 0.4% localization finding on a fresh dim).
- **Corpus B — bach_invention_1 + bach_prelude_c_wtc1** (60-doc / 55-perf metronomic probe, prose-only,
  run to de-confound "chopin is a rubato piece"): **2 timing claims total** (rush_drag 1 — again
  `neutral` "grounded and steady, exactly what Bach asks for"; evenness 1). Metronomic repertoire does
  NOT rescue directional supply; timing is barely salient (2/60 docs) when the pulse is unremarkable.
- **Net across both corpora:** 222 docs, 79 timing claims, **3 `rush_drag` — all 3 `neutral`. Directional
  rush-or-drag error claims = 0.** A signed statistic needs a signed claim; the corpus supplies none.

**Root cause = the GENERATOR VOICE, not the piece or the dimension.** The two confounds resolve cleanly:
(1) piece — Bach (metronomic) also yields ~0 directional supply, so it is not a Chopin-rubato artifact;
(2) voice — `baseline_v1` is the warm/"celebrate-strengths" synthesizer, which produces *aspirational
shaping* advice ("add rubato, let it breathe") and *steadiness praise*, never *error-correction* ("you
rushed bar 12"). Directional error is the one thing a signed-onset-deviation-vs-score statistic
adjudicates, and this teacher voice structurally does not make it. **Caveat for the paper:** this NO_GO
is proven on the GENERATOR corpus (what the production system actually adjudicates), NOT on real human
masterclass prose — a real teacher may make more "you rushed" claims. If the paper's claim is about the
SYSTEM's faithfulness, the generator corpus is the right denominator and this is decisive; if it is
about verifying human teaching broadly, the generator-voice caveat must be stated.

**The supply that DOES exist is rubato (91%)** — a *shaping/spread* quantity (deviation magnitude &
structure across a phrase), not a signed mean (rubato stretches then compresses → signed mean ≈ 0). This
is the exact structural twin of dynamics (contrast 90% / level ~0). **Do not naively pivot to a
"rubato-magnitude" statistic:** FRONT 6 already proved no deterministic *shaping* statistic validates
against perception for dynamics (best partial ≤0.26); a rubato-shape statistic inherits that unproven-
validity risk AND lacks a clean perception anchor. That is a research gamble, not a next step.

**Timing gate status now:** perception route = degenerate (IOI-CV, GATE 2/3); score-relative route =
**substrate-clean but supply-blocked (7a NO_GO)**. Both timing routes are now closed on the available
corpus. 7b (measurer), 7c (G-A), 7d (tau) — NOT STARTED, correctly un-built.

**Re-plan fork (strategic; the front does NOT auto-continue):**
- (a) **Accept + reframe** — the paper's measured contribution becomes the *supply/validity gap map*
  across dimensions (dynamics-level supply-blocked, dynamics-contrast validity-blocked, timing
  supply-blocked, pedal AMT-lossy), i.e. "what teacher feedback is even verifiable," with legible
  abstention as the result. This is the honest, already-earned finding.
- (b) **Change the corpus, not the statistic** — generate-on-bundles with a teacher prompt that is
  permitted to make directional error claims (or acquire real masterclass prose), then re-probe supply.
  Tests whether the NO_GO is the voice (fixable) vs the pedagogy (fundamental).
- (c) **New G-B front for a rubato-shape statistic** — high-risk per FRONT 6; only if (b) also fails.
FRONT 8 (articulation / duration-ratio) is independent of this NO_GO and remains available.

## FRONT 7a-bis UPDATE (#101, 2026-07-04): fork (b) ran — the blocker is the INPUT, not the voice.

Ran fork (b) as a paired two-arm supply re-probe over the 17 chopin_ballade_1 performances that
have real AMT bundles. Prompt held IDENTICAL and permissive across arms (explicitly allowed to name
directional rush/drag); Sonnet held constant; only the INPUT varied:
- **ARM A** = permissive prompt + the ORIGINAL scalar input (`muq_means["timing"] ~0.57`, no direction).
- **ARM B** = permissive prompt + a CONSTRUCTION-KNOWN directional cue (7 rush / 7 drag / 3 steady;
  4 bar-localized). Construction-known because a cheap self-relative onset signal is texture-confounded
  (raw AMT `median(60/IOI)` gave "established ~375 BPM", −47..−75% drift = polyphony density, not tempo —
  independent confirmation that a trustworthy directional signal needs the score-relative bar-map = 7b).

| condition | input | rush_drag | rubato | directional supply |
|---|---|---|---|---|
| original (warm prompt, 162 docs) | scalar timing 0.57 | 0 | 70 | **0** |
| ARM A (permissive, 17 docs) | scalar timing 0.57 | **0** | 0 | **0** |
| ARM B (permissive, 17 docs) | + directional cue | **19** (7 rush/7 drag/5 neutral) | 0 | **19** (4 bar-localized) |

Construct-match (injected direction vs extracted claim): **17/17 direction, 4/4 bar-localization.**

**Diagnosis — H1 (voice) REJECTED, H2 (input) CONFIRMED, H3 (fundamental) REJECTED.** Making the prompt
permissive changes nothing (Arm A = original = 0): the teacher is HONEST and will not fabricate a
direction from a scalar quality score. Add a real directional signal and directional claims appear
immediately (0 → 19), correctly signed and located. The FRONT-7 NO_GO was never the statistic, the claim
taxonomy, or the voice — **the production teacher's input carries no timing direction.**

**What this does to FRONT 7:** it is not dead; its center of gravity moves from "is there supply?" to a
PIPELINE change — compute score-relative onset deviation (7b's measurer) and feed it to the teacher, and
directional (verifiable) claims flow. **But three caveats reshape the paper goal, and must be stated:**
1. **Circularity.** If the SAME measurer both feeds the teacher and adjudicates it, the "faithfulness
   rate" is near-tautological — the teacher parrots the fed signal (construct-match already 17/17). The
   meaningful evaluation shifts from "is the claim true?" to "does the teacher DISTORT the signal it was
   given (wrong magnitude / mislocated / dropped)?" — a signal-fidelity question, not a discovery one.
2. **Construction-known, not real.** Arm B used INVENTED directions. This proves plumbing + voice
   capability, NOT that real performances carry directional errors worth reporting, nor that a real 7b
   measurer would find them. The 17 real performances' true directional timing remains unmeasured.
3. **Model + n.** Sonnet-as-teacher (not production glm-4.7-flash; held constant so A-vs-B is clean but
   transfer assumed); n=17, single piece.

**Refined fork (the real decision now):**
- (b1) **Build 7b for its own sake** — the score-relative onset measurer, validated by 7c G-A
  (construction-known tempo-warp), becomes the paper's SYSTEM contribution: "a score-anchored directional
  timing signal that, wired into the teacher, unlocks verifiable directional feedback absent by default,"
  with the honest signal-fidelity (not discovery-faithfulness) framing. Real deliverable, but accept the
  circularity reframing of caveat 1.
- (a) **Accept + reframe** (unchanged) — fold 7a + 7a-bis into the supply/validity gap map; the finding
  "directional timing feedback is input-limited, not voice- or verifier-limited" is itself a clean paper
  result and needs no more building.
- FRONT 8 (articulation) remains independent and available.

Harness: `model/src/claim_measurement/timing_supply/{build_teacher_inputs.py, teacher_prompt.md}`;
audit artifacts in `model/data/results/timing_supply_arm_{a,b}*.json` (gitignored).

## FRONT 7b UPDATE (#101, 2026-07-05): measurer core + verifier integration LANDED (owner chose b1).

Owner picked **build 7b**. The score-relative timing measurer now exists and is wired end-to-end
through the FROZEN router; the remaining 7b work is the offline alignment PIPELINE that feeds it.

**Shipped this session (TDD, verdict_dispatch untouched, 138 tests = 125 baseline + 13 new):**
- `verifier/measurers/onset_deviation.py` — `OnsetDeviationMeasurer`: `d = mean(perf_onset -
  score_onset)` ms, whole_piece + bar tiers; rush `d<0` / drag `d>0` (matches frozen polarity
  contract: '-'=rush SUPPORTED iff d<0&|d|>tau; '+'=drag; neutral=steady iff |d|<=tau). error_bar
  folds sampling + AMT onset-jitter + **alignment-uncertainty** in quadrature (weak alignment widens
  the bar honestly). 7 unit tests.
- Wired into `orchestrator._build_registry` under key `amt_score_relative_onset_deviation` (additive).
  6 end-to-end `verify()` integration tests (inline taxonomy): rush→SUPPORTED, wrong-polarity→REFUTED,
  below-tau→REFUTED, drag→SUPPORTED, neutral→SUPPORTED, **unaligned bundle→UNVERIFIABLE(substrate_
  failure)** = legible abstention (timing without a score is unverifiable BY DESIGN, not REFUTED).

**Key design decision (the crux — for whoever builds the pipeline): AFFINE detrend, not raw, not full
warp.** Raw `perf_onset - score_seconds` is dominated by the global tempo difference (a real perf runs
~1.5-2.2x off the score's nominal seconds) — that is NOT rush/drag. The measurer's `score_onset`
CONTRACT is the score onset passed through a GLOBAL AFFINE fit `a*score_onset+b` (least squares over
parangonar matches), so `perf_onset - score_onset` = the affine RESIDUAL = local rush/drag. It must NOT
be the full monotone DTW warp (`anchors` / chroma follower) — that tracks local tempo and would ABSORB
the rush/drag, collapsing d→0. Mirrors the shipped Rust `align_notes_from_warp` →
`NoteAlignment.onset_deviation_ms` (affine detrend, note_align.rs), which is the LIVE-DO path; 7b is the
offline analog.

**Remaining 7b (offline pipeline — NOT the live DO; the heavy next slice):** an offline Python function
(new, in `model/src/chroma_dtw_eval/` or `claim_measurement/`) that: parangonar-matches AMT perf notes
to the score (`_match`, amt_regen.py:334, exists), fits the global affine (a,b), emits per matched note
`score_onset = a*score_beat+b` (perf-time) + a `bar_number`, and writes a SCORE-ALIGNED bundle
(`notes[*].score_onset`). `_build_pairs` (amt_regen.py:340) currently DISCARDS the per-note
correspondence — must be replaced/forked to keep it. Score loader already supports variable-tempo/non-4/4
(extended in #98; chopin_ballade_1 loadable). THEN: repoint the shipped `timing` dimension's
`measurement` to the new key + unit `ms` + provisional tau (update the two dimension-count tests then);
run 7c G-A (tempo-warp corruption through the REAL measurer, flip≥0.80 / shuffle≥0.20); 7d tau + construct
validity (7a-bis already previewed construct-match 17/17). G-D timing rate still needs the prose→signal
pipeline of 7a-bis fork-b caveat 1 (circularity: signal-fidelity, not discovery).

---

## FRONT 7b UPDATE 2 (#101, 2026-07-07): pipeline landed, global affine EMPIRICALLY INVALIDATED -> windowed frame + whole-piece degeneracy declared

**Pipeline shipped** (`model/src/claim_measurement/score_align/`): fork of `_build_pairs` keeping the
per-note parangonar correspondence; annotates the cached bundles with `score_onset` + `bar_number`
(pure post-processing on stored AMT notes — no AMT server needed; CLI idempotent via
`score_align.schema`, per-clip timeout-guarded).

**Finding 1 — the whole-piece GLOBAL affine of the 7b core contract does not survive real data.**
First real batch (10 bundles): residual RMS 3.0-48.7 SECONDS, tempo ratios up to a=16.5. Diagnosis on
the cleanest clip (Traumerei, 330 matches): matches are GOOD (1% backward jumps, 100% consistent with
the accepted pseudo-truth anchors) — the residual is RUBATO. A real performance's tempo breathes;
against one global line that integrates to seconds (clean-core median |res| 2.2s), two orders above
any plausible tau. Per-window refit collapses it: windowed-30s median 297ms (7x), windowed-15s is the
shipped frame (mirrors the live Rust per-chunk affine, note_align.rs).

**Finding 2 — whole-piece mean-d is degenerate BY CONSTRUCTION under any same-set LSQ frame.**
LSQ-with-intercept residuals over the fitted notes have mean EXACTLY zero, so `d(whole_piece) = 0`
identically on pipeline bundles — the IOI-CV degeneracy reborn one abstraction up (7c's controls
would have caught it; running real data first caught it sooner). Encoding: the DEGENERACY IS A
PIPELINE PROPERTY, so the bundle declares `score_align.reference_frame = "windowed_affine"` and the
measurer abstains at whole_piece for `SAME_SET_LSQ_FRAMES` with reason_code
`degenerate_reference_frame` (construction-known hand-annotated bundles keep measuring). Bar/region
tiers are subsets of each window — they carry the real signal.

**Consequences for the front (re-planned):**
- The 7b statistic is BAR/REGION-tier score-relative onset deviation. Whole-piece directional timing
  needs a DIFFERENT statistic (e.g. trend of per-window tempo ratios a_w, or an out-of-sample /
  score-beat-grid reference) — design that WITH 7c controls, not before.
- Noise floor: windowed-15s residuals on real bundles are the empirical per-note sigma for 7d tau
  calibration; tau=30ms provisional is likely too tight for bar-tier claims on this substrate
  (bar-mean sigma ~60-100ms at 8-30 notes/bar). 7c corruption sensitivity will set the honest floor.
- Timing dim REPOINTED in the taxonomy (measurement `amt_score_relative_onset_deviation`, tolerance
  30ms provisional `locked:false`); frozen `verdict_dispatch` untouched; the two behavioral timing
  tests now run on score-aligned construction-known bundles.

**UPDATE-2 addendum (same session): anchor gate + the substrate verdict.** Gating matches to the
accepted pseudo-truth anchor envelope (+-1.5s) improves 9/10 clips ~40% (waltz 1.57->0.90s,
pathetique 2.03->0.85s median) but the floor stays 0.85-1.9s everywhere except Traumerei (0.23s,
already clean). Etude op10no4 is METRONOMIC and still 1.43s -> the residual is per-note MATCH ERROR,
not rubato: parangonar+AMT per-note correspondence on real YouTube audio is +-1s-coarse (consistent
with GATE-1's +-1.5s tolerance). CONSEQUENCES: (1) bar-tier timing claims on REAL audio are
substrate-blocked on 9/10 clips (bar-mean sigma >> any honest tau); Traumerei-class alignment
(~230ms note / ~50-80ms bar-mean) is the existence proof of the usable regime. (2) 7c G-A runs on
RENDERED construction-known audio (clean AMT) — unaffected. (3) The real-audio rate inherits a new
dependency: **#108's continuity-penalized symbolic matcher replacing parangonar inside this offline
pipeline** (its day-0 probe already beat parangonar's global matchers, which scatter with no local
constraint). The #101 paper path and the #108 product path CONVERGE at the matcher.

---

## FRONT 8 UPDATE (#101, 2026-07-19): dynamics level-cue supply probe — the front-5 gap is INPUT-side and PROMPTABLE (rate is signal-fidelity, not independent)

Front 5 declared the dynamics *rate* UNMEASURABLE: the generator makes ~90% dynamic-CONTRAST
claims and ~0 falsifiable whole-piece LOUDNESS-LEVEL claims, so the G-B-validated mean-velocity
LEVEL statistic had nothing to score (a construct mismatch). Front 6 then killed the obvious rescue
(no deterministic dynamic-CONTRAST statistic validates). Front 5 left ONE door un-run: is that
supply gap the generator's VOICE (won't state overall loudness) or its INPUT (never *given* an
overall-loudness signal)? — the front-7 question, asked for dynamics. This front ran it and the
answer is INPUT: the gap is promptable.

**Method — paired two-arm test (mirrors 7a-bis) over the SAME 17 `gd_bundles` performances.** Held
the permissive-but-honest teacher prompt IDENTICAL across arms; only the per-performance INPUT
differs, so the arms isolate exactly one variable.
- **ARM A**: `muq_means` scalars only (the ORIGINAL production input).
- **ARM B**: + a **MEASURED** overall-loudness cue derived from THIS bundle's mean AMT note velocity
  (`d = mean_vel − 51.5`, direction vs tau 6.5; 9 loud / 8 balanced / 0 soft).

**KEY DISTINCTION from the timing probe:** timing's ARM-B cue had to be INVENTED (raw AMT onsets are
texture-confounded; a real directional signal needed 7b's score alignment). Dynamics' validated
statistic is mean note VELOCITY — density-free and gain-robust on the raw bundle — so the ARM-B cue
is a REAL self-relative measurement. **CONSEQUENCE (the honest caveat):** the SAME statistic both
CUES and SCORES, so any resulting rate is a **signal-fidelity** rate (the measure feeds and
adjudicates itself), NOT an independent faithfulness rate. Teacher generation + claim extraction are
LLM (Sonnet); the truth label is the real `DynamicsMeasurer` + frozen `route_verdict` (tau 6.5) — the
Path-#1 non-circularity rule holds for the *verdict*, not for the cue↔statistic loop.

**Results (all measured; `model/data/results/dynamics_level_cue_supply.json`, `dyn_arm_{a,b}_rate.json`):**

| quantity | ARM A (no cue) | ARM B (measured cue) | read |
|---|---:|---:|---|
| level@whole_piece claims | **0** (13 contrast) | **17** (+14 contrast) | **supply LIFT 0→17: the gap is promptable, not a voice block** |
| committed (real router) | 0 | 14 (3 near_threshold abstain) | 3 near-boundary claims correctly quarantined |
| faithfulness rate | UNMEASURABLE | **1.000** CI [1.000, 1.000] | signal-fidelity ceiling (circular) — NOT the headline |
| polarity (committed) | — | + 8/8, neutral 6/6 SUPPORTED | spans two verdict classes (soft/− untested: 0 soft cues) |
| sign-fidelity vs cue | — | **15/17 (0.88)** | teacher inflated 2/8 *balanced* cues to a loud (+) claim |

**What this measures (and what it does not).** ARM A **reproduces front 5** on these 17 (0 falsifiable
overall-loudness claims — the teacher is honest and won't invent a level uncued). ARM B shows the
teacher WILL make grounded whole-piece loudness claims when given a measured signal. The **1.000 is a
signal-fidelity ceiling, not an independent faithfulness rate** — expected by construction and NOT a
paper headline. The genuinely measured contributions are: (1) **the dynamics-level supply gap is
INPUT-side and promptable** (0→17), reframing front 5's "unmeasurable" as "unmeasurable *on the uncued
production input*"; (2) **teacher sign-fidelity is 0.88, not 1.0** — even given a clean cue the voice
inflated 2 balanced performances to a directional loud claim (a measured teacher-side distortion);
(3) **the near-threshold dead-band worked** — both inflated claims had true |d|<tau and were abstained,
not committed (0 REFUTED among committed is the abstention machinery doing its job, not luck).

**G-D PASS = False**, for two independent reasons: n=14 committed < 30 (only 17 bundles are
transcribed; ≥30 needs more AMT transcription, ~57s/clip), AND the rate is signal-fidelity, not
independent. A gate-passing INDEPENDENT dynamics-level rate still requires an **independent loudness
signal** to break the cue↔statistic circularity (e.g. ground-truth MIDI velocity, or a human loudness
label distinct from the scored statistic) — the same circularity caveat timing carries.

**Net.** Dynamics-level moves from "validated statistic, no claims" (front 5) to "validated statistic,
**claims are input-promptable**, independent rate still circularity-blocked." **Product consequence
(concrete):** if CrescendAI feeds the teacher a measured overall-loudness input, the teacher emits
whole-piece loudness claims the verifier CAN check — a cheap grounded-feedback surface, unlike
dynamic-contrast (front 6: no valid statistic exists). Harnesses:
`model/src/claim_measurement/dynamics_supply/{build_teacher_inputs,classify_supply}.py` +
`teacher_prompt.md` (tests `test_dynamics_supply_{inputs,classify}.py`, 11 green).

---

## FRONT 8b UPDATE (#101, 2026-07-19): the FIRST INDEPENDENT (non-circular) dynamics faithfulness rate = 0.919 — substrate-faithful, gate-adjacent

Front 8's rate (1.000) was a signal-fidelity ceiling: it cued the teacher from AMT mean velocity
and scored with AMT mean velocity, so the measure adjudicated itself. This front breaks the
circularity with an **independent truth signal**: the claim's polarity is fixed by **ground-truth
MIDI velocity** and the SCORE is the production AMT-velocity measurer + frozen router — two
independent measurements of the same performance. It answers the real question: *when ground truth
says a performance is loud / soft / balanced overall, does the deployed AMT verifier agree?*

**Substrate move (forced).** GT MIDI velocity exists only for PercePiano (which IS MIDI); the real
YouTube `gd_bundles` have no ground truth. So this runs on PercePiano: `render_percepiano_bundles.py`
renders each MIDI (fluidsynth, fixed gain 0.5) → aria-amt → a bundle carrying AMT `notes` +
`gt_mean_velocity` + `gt_corpus_median`. 168 segments **stratified soft/balanced/loud** by GT velocity
vs the labeled-corpus median (58.2), seed 42 (rendered in two nested batches, 45 then +123, ~40-70s/seg)
— so all three verdict classes are represented (front 8 had 0 soft cues). This also **fixes the G-B
non-persistence gap** (that gate rendered to a deleted tempdir and cached only 180 anonymous scalars);
the per-segment AMT arrays are now persisted.

**ORACLE (no-LLM) design.** The claim polarity IS the GT label (loud→+, soft→−, balanced→neutral),
so the rate isolates the verifier's **substrate faithfulness** (AMT-vs-GT at the tau decision
boundary) with zero teacher/extractor noise. Truth-label purity holds — the verdict is only the
deterministic measurer + frozen `route_verdict` (tau 6.5); GT MIDI velocity is a non-LLM signal.

**Result (`model/data/results/dynamics_independent_rate.json`, n=45):**

| quantity | value | read |
|---|---:|---|
| **independent faithfulness rate** | **0.919**, 95% CI [0.870, 0.959] | AMT verifier agrees with ground truth 92% (committed) |
| committed | 123 / 168 (113 SUPPORTED, 10 REFUTED) | 45 near-threshold abstentions (16 soft / 17 balanced / 12 loud) |
| by verdict class | loud 39✓/5✗, **soft 36✓/4✗**, balanced 38✓/1✗ | all three tested; soft (−) covered |
| tau_gt sweep (4 / 6.5 / 9) | 0.862 / 0.919 / 0.902 | stable ~0.90, not a single-threshold artifact |
| **G-D gate** | **PASS — committed 123 ≥ 30 AND CI half 0.045 ≤ 0.05** | **first gate-passing independent per-dimension rate** |

**What it means. This is the paper's first GATE-PASSING independent per-dimension faithfulness number,
and 0.919 < 1.000 is the whole point** — the circular front-8 rate could not see substrate error; here
the AMT velocity genuinely disagrees with GT on 10/123 committed segments (5 loud read as not-loud,
4 soft as not-soft, 1 balanced as directional — real boundary errors). The rate is essentially the
AMT↔GT velocity agreement (Spearman 0.965, G-B) expressed as a whole-piece loudness DECISION rate, with
the near-threshold dead-band honestly abstaining the 45 boundary cases. **The G-D gate now PASSES on both
criteria**: 123 committed clears the ≥30 count bar (a first for any dimension) AND the CI half-width 0.045
clears the ≤0.05 precision bar. (The initial n=45 batch passed count but missed precision at half 0.095;
one nested +123-segment render batch closed it with the rate stable at 0.919.)

**Scope / caveats (honest).** (1) This is **substrate** faithfulness (oracle), NOT teacher faithfulness:
the claim = the GT label. The deployed end-to-end rate would be this × the teacher's sign-fidelity
(~0.88, front 8) ≈ 0.8 — the teacher arm (cue the teacher from the GT label, extract, score) is the
un-run next step. (2) Controlled-gain fluidsynth renders, not heterogeneous real audio (the G-F
question). (3) PercePiano 8-bar excerpts. Harnesses:
`model/src/claim_measurement/dynamics_supply/{render_percepiano_bundles,independent_rate}.py` (tests
`test_dynamics_independent_rate.py`, 5 green; full `dynamics_supply` suite 19 green). Bundles persisted
at `model/data/evals/percepiano_indep_bundles/`.

---

## FRONT 8c UPDATE (#101, 2026-07-22): the END-TO-END (deployed) rate = 0.919 — the teacher is NOT the bottleneck; the substrate is

8b measured the VERIFIER's substrate faithfulness (oracle: claim = GT label, no LLM). 8c layers the
teacher back in to get the deployed end-to-end rate: the teacher is CUED by the ground-truth loudness
label (independent of the AMT statistic that scores it), writes prose, and its LLM-extracted claims are
scored by the real AMT measurer + frozen router. Same 168 PercePiano bundles as 8b; balanced GT-cue mix
56 loud / 56 soft / 56 balanced. Harness `build_gt_cued_inputs.py` (reuses front-8 cue text + `gt_polarity`);
teacher/extractor are Sonnet (4 batches each); scored by the shipped `gd_rate/route_and_score.py`.
Non-circular: cue = GT MIDI velocity, score = AMT velocity.

**Result (`model/data/results/dyn_gt_teacher_rate.json`):**

| quantity | value | read |
|---|---:|---|
| level-claim supply | **168 / 168** segments (+59 contrast, abstain) | every cued segment yields a whole-piece level claim |
| **teacher sign-fidelity vs GT cue** | **1.000 (168/168)** | 56/56 loud→+, 56/56 soft→−, 56/56 balanced→neutral |
| **end-to-end deployed rate** | **0.919**, 95% CI [0.870, 0.959] | committed 123/168 (113 SUP, 10 REF); **G-D PASS = True** |

**The end-to-end rate is IDENTICAL to the 8b oracle substrate rate (0.919, same committed set, same
confusion)** — because the teacher's sign-fidelity is 1.000, it adds ZERO error, so the AMT substrate is
the sole error source. **The teacher is not the bottleneck.** This is *higher* than front 8's teacher
sign-fidelity (0.88): front 8's 2 sign errors were *balanced* cues inflated to loud, where the
**AMT-derived** cue was ambiguous near the boundary; here the cue is a deadband-cleaned **GT** label
(unambiguous), which the teacher echoes perfectly (incl. affirming all 56 balanced levels).

**Honest deployment caveat.** 0.919 is the **clean-input** ceiling. In production the teacher is fed the
**AMT-derived** loudness signal (noisy near tau), where front-8 measured ~0.88 sign-fidelity → a deployed
rate ~0.8. So the deployed rate is **substrate-bounded at 0.919 when the loudness input is clean**, and
the gap down to ~0.8 is entirely a function of how noisy the fed signal is near the decision boundary —
NOT the teacher's honesty. The teacher also still adds ~0.35 contrast claims/segment (all abstain),
echoing the front-5/8 finding that teachers gravitate to shaping language even when cued on level.

**Net across FRONT 8 / 8b / 8c — dynamics-level is the one fully-characterized verifiable dimension:**
supply is input-promptable (8: 0→17; 8c: 168/168); the verifier is perceptually validated (G-B 0.54) and
substrate-faithful vs ground truth at **0.919 [0.870, 0.959], G-D PASS** (8b); and the deployed teacher
pipeline hits that same 0.919 on clean input, ~0.8 on realistic noisy input (8c). The remaining open
edge is real heterogeneous-gain audio (G-F), untested here (controlled fluidsynth renders only). Harness:
`build_gt_cued_inputs.py` (test `test_build_gt_cued_inputs.py`, 3 green; full `dynamics_supply` suite 19
green). Result: `model/data/results/dyn_gt_teacher_rate.json`.

**SUBSTRATE CAVEAT — these numbers are aria-amt-specific (re-run trigger: #125 Transkun adoption).**
Every AMT-substrate number in FRONT 8/8b/8c (and the upstream G-B 0.544 / G-C error bars) is measured on
**aria-amt** velocity — the current production transcriber (as of 2026-07-23 transkun has ZERO code
references; it is a candidate, not wired in). The 0.919 rate IS the aria-amt AMT↔GT velocity agreement
(0.965) expressed as a decision rate. **Issue #125 proposes adopting Transkun** (vel F1 0.926, emits
velocity+pedal) as the transcriber, explicitly to improve the velocity channel this rate depends on; its
#104-difficulty arm STOPped (tau-c-neutral) but the CrescendAI-core/verifier adoption is unresolved. IF
Transkun replaces aria-amt in the bundle path, re-run the whole chain — the transcribe→score split makes
it a one-transcriber swap in `render_percepiano_bundles.py` then re-score `independent_rate.py` /
`route_and_score.py` (bundles + rate JSONs regenerate; no code change to the scorers). The rate would
likely RISE (better velocity fidelity → tighter AMT↔GT agreement). Until then, read every number here as
"aria-amt substrate," not "the transcriber substrate."
