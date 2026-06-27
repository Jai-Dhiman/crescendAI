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
  measured 1σ. No assumed-noise error bars in the headline. *Status: not started.*
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
   scope whole_piece timing OUT until a beat tracker exists. (G-A, G-B)
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
