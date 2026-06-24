# Claim Verifier: Per-Dimension Signed-d Conventions and Error-Bar Table

**Issue:** #65, #94 (first real-audio measurement)
**Taxonomy version:** v0.1
**Validation boundary:** This document covers substrate-level measurement. The error bars in
"Measured Error Bars" below are now MEASURED on real cached audio (AMT held fixed, #94); the
"Substrate Error Distributions" table remains the per-source design input. Real-claim faithfulness
(#67), proxy-to-perception (#66), and error-rich localization (GATE 1) are still NOT validated here.

---

## Sign convention Table

| Dimension | d formula (region) | d formula (whole_piece) | d < 0 means | d > 0 means | tau (provisional) | units |
|-----------|-------------------|------------------------|-------------|-------------|-------------------|-------|
| timing | `(established_tempo - region_median_bpm) / established_tempo * 100` | `CV% = std(bpms) / mean(bpms) * 100` | faster than reference (rushed) | slower than reference (dragging) | 8.0 | percent |
| pedaling | `region_pedal_bar_fraction - self_density` | `pedal_bar_fraction` (0.0-1.0) | sparse pedaling vs piece average | dense pedaling vs piece average | 0.25 | fraction |
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
