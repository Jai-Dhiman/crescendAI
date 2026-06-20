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
