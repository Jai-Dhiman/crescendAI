# Deterministic Claim Verifier Design

**Goal:** Given a structured claim (dimension + location + polarity) and a piece reference, produce SUPPORTED | REFUTED | UNVERIFIABLE with measured d, tau, error_bar, and event_count — by populating the `_measurement` dict that the shipped `route_verdict` consumes.

**Not in scope:**
- Bar-N single-bar localization tightening (GATE 1, deferred to follow-up issue)
- Error-rich / student audio acquisition or simulation (GATE 1)
- Real-claim faithfulness validation (issue #67)
- Proxy-to-perception correlation study (issue #66)
- articulation dimension activation (AMT offset reliability not yet validated)
- Any LLM call in the truth label path
- Any modification to `apps/evals/claim_taxonomy/verdict_dispatch.py` or `test_verdict_dispatch.py`

---

## Problem

The shipped `route_verdict` in `verdict_dispatch.py` is a complete dispatch chain but it requires a caller to populate `claim["_measurement"]` with `{d, tau, error_bar, event_count, localizable, substrate_failure}`. Without the verifier, these fields must be hand-authored for tests and there is no way to produce verdicts from real audio.

Separately, two data gaps block two active dimensions:
1. **Pedaling**: `claim_taxonomy.json` lists pedaling as `status: active` with `measurement: amt_sustain_pedal_events`, but the existing extraction path (`amt_regen.py` + `pseudo_truth_cache.py`) discards CC64 sustain-pedal events entirely — the bundle only contains note onsets.
2. **Dynamics**: `claim_taxonomy.json` lists dynamics as `gated_on_measurement` because no audio-domain loudness estimator exists in the repo. MIDI velocity is not perceived loudness. A librosa RMS estimator would unlock this dimension.

---

## Solution (from the user's perspective)

After this issue ships:

1. A developer runs `python -m claim_measurement.extract_bundle --piece bach_prelude_c_wtc1 --video-id mfN8ZEYWdqs` and gets a JSON bundle at `model/data/evals/claim_bundles/{piece}/{video_id}.json` that contains notes, pedal events, measure table, alignment anchors, audio path, and substrate versions.

2. The developer runs `python -m claim_taxonomy.verifier.cli verify --claim claim.json --bundle bundle.json` and gets a `VerdictResult` JSON object with verdict, reason_code, measured_value, tau, error_bar, event_count, units, substrate_versions, dimension, and location.

3. `claim_taxonomy.json` is bumped to `v0.1`. The dynamics dimension moves from `gated_on_measurement` to `active`. The net active dimensions are: timing, pedaling, dynamics.

4. A per-dimension error-bar and signed-d convention table is committed to `docs/model/claim-verifier-signed-d-conventions.md`.

---

## Design

### Two-Environment Split

The design enforces an explicit boundary between extraction (model/ uv env) and verification (apps/evals/ uv env).

**EXTRACTION** (`model/src/claim_measurement/`): runs parangonar, partitura, librosa for AMT + alignment. Produces a bundle JSON. Lives in the model/ env because parangonar and partitura are already deps there. The bundle is the stable, auditable interface between substrate and checker.

**VERIFIER** (`apps/evals/claim_taxonomy/verifier/`): reads the bundle, applies dimension-specific statistics, assembles `_measurement`, calls `route_verdict`. Lives in apps/evals/ env because numpy/scipy/librosa/soundfile are in the `[inference]` extra there.

### Bundle Schema

The bundle JSON produced by extraction is the auditable checker/substrate interface:

```json
{
  "piece_id": "bach_prelude_c_wtc1",
  "video_id": "mfN8ZEYWdqs",
  "audio_path": "/abs/path/to/audio.wav",
  "notes": [{"onset": 0.0, "offset": 0.5, "pitch": 60, "velocity": 80}],
  "pedal_events": [{"time": 0.3, "value": 127}],
  "measure_table": [{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}],
  "anchors": {
    "perf_audio_sec": [0.0, 1.2],
    "score_audio_sec": [0.0, 1.0]
  },
  "substrate_versions": {
    "amt_checkpoint_hash": "aria_amt_v1_pilot_2026_06_01",
    "parangonar_version": "3.3.2",
    "bundle_schema": "v1"
  }
}
```

`pedal_events` contains raw CC64 events (time, value). Sustain-on threshold is value >= 64. If the AMT server does not expose CC64 events, `pedal_events` is `[]` and the PedalingMeasurer raises `UnverifiableError("substrate_failure", "AMT endpoint does not expose CC64 events")`.

### Active Dimensions After v0.1

| Dimension | d (whole_piece) | d (region) | tau | Sign convention |
|-----------|-----------------|------------|-----|-----------------|
| timing | local-tempo CV% (coefficient of variation of inter-onset intervals) | signed % deviation from established_tempo | 8.0% | negative = faster than reference (rushed) |
| pedaling | pedal-bar fraction (fraction of bars with sustain-on event) | signed deviation from self_density | 0.25 fraction | negative = sparse, positive = over-pedaled |
| dynamics | RMS-contour std normalized by within-piece range | signed dB deviation from within_region_range | 1.5 dB (provisional) | negative = flat/narrow, positive = wide |

**Whole-piece reference degeneracy:** For `whole_piece` location, self-referential references degenerate. Each Measurer computes an intrinsic dispersion or presence statistic compared against an absolute provisional tau, rather than a deviation from a reference.

### Error Bar

```
error_bar = sqrt(sampling_var + substrate_var)
```

- **sampling_var**: bootstrap (N=500, seed=42 + call counter) over within-region events. Resamples with replacement; std of bootstrapped d values.
- **substrate_var**: Monte-Carlo (N=500, same seeded engine). Perturbs raw measurements by documented error distributions:
  - Timing: AMT onset jitter ~10ms (1-sigma, Gaussian, sigma=0.010s)
  - Dynamics: RMS frame variance, empirically ~0.3 dB std
  - Pedaling: CC threshold uncertainty ±10 value counts (Uniform), threshold 64 ± 10

### Near-Threshold Dead-Band

`abs(abs(d) - tau) <= error_bar` → UNVERIFIABLE(near_threshold). Computed inside `route_verdict`; the verifier supplies accurate d, tau, error_bar.

### Localization (LocationResolver)

1. `measure_table` maps bar_number → score_sec (start of bar in score-time)
2. `anchors` (parangonar perf/score pairs) interpolate score_sec → audio_sec via `np.interp`
3. `SubstrateErrorEngine` propagates alignment uncertainty: perturbs anchor pairs by timing jitter (Gaussian, sigma=0.010s), computes std of resulting bar-start audio_sec estimates → `alignment_uncertainty_sec`
4. Convert to bar units: `uncertainty_bars = alignment_uncertainty_sec / bar_duration_sec` where `bar_duration_sec` is inferred from adjacent measure_table entries
5. Failsafe: if `uncertainty_bars >= location_span` (bar_end - bar_start + 1), raise `UnverifiableError("unlocalizable", ...)`. Whole_piece span = infinity → always passes.

### Taxonomy v0.1 Changes

`claim_taxonomy.json` changes:
- `taxonomy_version`: `"v0"` → `"v0.1"`
- `dynamics`: replace `gated_on_measurement` entry with `active` entry:
  - `status`: `"active"`
  - `reference`: `"within_region_range"`
  - `check`: `"rms_contour_std_normalized"`
  - `tolerance`: `{name: "rms_contour_deviation", provisional: 1.5, unit: "dB", calibration_source: "#65/M1 error-bar study", locked: false}`
  - `reliability_tier`: 2
  - `measurement`: `"librosa_rms_region_estimator"`
  - `minimum_events`: 20 (RMS frames, at 512-hop 16kHz → ~32ms per frame)
  - `notes`: "RMS-based loudness proxy. Absolute dB inadmissible (uncontrolled recording gain). Within-region dynamic range only."

The schema (`claim_taxonomy.schema.json`) is **not changed**.

The existing test `test_dynamics_gated_returns_unverifiable` in `test_round_trip.py` tests the old taxonomy. It must be updated to `test_dynamics_active_routes_correctly` after v0.1 is committed. This is the only file outside the verifier/ tree that changes in the test suite.

### VerdictResult

```python
@dataclass
class VerdictResult:
    verdict: str           # SUPPORTED | REFUTED | UNVERIFIABLE
    reason_code: str | None
    measured_value: float  # d
    tau: float
    error_bar: float
    event_count: int
    units: str             # "percent", "fraction", "dB"
    substrate_versions: dict
    dimension: str
    location: dict | str   # bar range dict or "whole_piece"
```

### UnverifiableError

```python
class UnverifiableError(Exception):
    reason_code: str   # one of the taxonomy's unverifiable_reason_codes
    detail: str
```

Raised by LocationResolver and Measurers when verification cannot proceed. The orchestrator catches `UnverifiableError` and returns `VerdictResult(verdict="UNVERIFIABLE", reason_code=e.reason_code, ...)` without calling `route_verdict` for pre-measurement failures (unlocalizable, substrate_failure, region_too_short). For post-measurement failures (near_threshold), `route_verdict` handles it internally.

---

## Modules

### 1. BundleExtractor (`model/src/claim_measurement/extractor.py`)

**Interface:**
```python
def extract_bundle(
    piece_id: str,
    video_id: str,
    *,
    audio_path: Path,
    cache_root: Path,     # pseudo_truth cache root (existing anchors)
    bundle_root: Path,    # output root for bundle JSON
    amt_url: str,
    force: bool = False,
) -> Path:
    """Produce bundle JSON at bundle_root/piece_id/video_id.json. Returns path.
    Idempotent: returns existing path if bundle exists and force=False.
    Raises AmtRegenError on AMT failures.
    """
```

**Hides:** AMT transcription (reuses `_transcribe_clip` + `_dedup_amt_notes` from `chroma_dtw_eval.amt_regen`), CC64 pedal event capture, parangonar alignment (reuses `_match` + `_build_pairs`), bundle serialization, idempotency check via sha256 of audio path.

**Depth verdict:** DEEP — one function call hides ~150 lines of multi-system orchestration.

---

### 2. SubstrateErrorEngine (`apps/evals/claim_taxonomy/verifier/substrate_error.py`)

**Interface:**
```python
class SubstrateErrorEngine:
    def __init__(self, seed: int = 42, n_samples: int = 500): ...

    def timing_onset_jitter_sec(self) -> np.ndarray:
        """n_samples of AMT onset jitter: Gaussian(mean=0, sigma=0.010)."""

    def dynamics_rms_jitter_db(self) -> np.ndarray:
        """n_samples of RMS frame variance: Gaussian(mean=0, sigma=0.3)."""

    def pedal_threshold_jitter(self) -> np.ndarray:
        """n_samples of CC threshold offset: Uniform(-10, +10)."""

    def bootstrap_d(
        self, values: np.ndarray, stat_fn: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """Bootstrap stat_fn over values with replacement. Returns n_samples bootstrapped d values."""

    def alignment_uncertainty_sec(
        self,
        perf_audio_sec: np.ndarray,
        score_audio_sec: np.ndarray,
        bar_start_score_sec: float,
    ) -> float:
        """MC: perturb anchor pairs by timing_onset_jitter_sec, interp bar_start_score_sec
        for each sample, return std of resulting audio_sec estimates."""
```

**Hides:** NumPy RNG seeding and state, MC loop, bootstrap resampling.

**Depth verdict:** DEEP — deterministic interface over stochastic internals.

---

### 3. LocationResolver (`apps/evals/claim_taxonomy/verifier/location_resolver.py`)

**Interface:**
```python
@dataclass
class ResolvedRegion:
    audio_start_sec: float
    audio_end_sec: float
    alignment_uncertainty_sec: float
    location_span_bars: float  # float('inf') for whole_piece

class LocationResolver:
    def __init__(self, bundle: dict, engine: SubstrateErrorEngine): ...

    def resolve(self, location: dict | str) -> ResolvedRegion:
        """Maps claim location to audio time range + uncertainty.

        Raises UnverifiableError("unlocalizable", detail) if:
        - bar_number not in measure_table
        - anchors have < 2 points
        - alignment_uncertainty_bars >= location_span_bars
        """
```

**Hides:** Measure-table lookup, score_sec → audio_sec interpolation, bar_duration_sec inference from adjacent table entries, MC uncertainty propagation via engine, failsafe comparison.

**Depth verdict:** DEEP.

---

### 4. TimingMeasurer (`apps/evals/claim_taxonomy/verifier/measurers/timing.py`)

**Interface:**
```python
@dataclass
class Measurement:
    d: float
    error_bar: float
    event_count: int
    substrate_failure: bool

class TimingMeasurer:
    def measure(
        self,
        location: dict | str,
        bundle: dict,
        region: ResolvedRegion,
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        """Timing deviation measurement.

        Whole-piece: d = CV% of local IOI-derived tempo values across all notes.
        Region: d = signed % deviation of region median BPM from piece established_tempo.
        d < 0 means faster than reference (rushed).
        event_count = number of note onsets in region.

        Raises UnverifiableError("region_too_short") if event_count < 8.
        Raises UnverifiableError("substrate_failure") if notes array is empty.
        """
```

**Hides:** IOI computation, median-BPM formula, established_tempo computation (median BPM across whole piece using sliding 8-note windows), percent-deviation formula, bootstrap + MC error bar assembly via `sqrt(sampling_var + substrate_var)`.

**Depth verdict:** DEEP.

---

### 5. PedalingMeasurer (`apps/evals/claim_taxonomy/verifier/measurers/pedaling.py`)

**Interface:** Same `measure()` protocol as TimingMeasurer (same `Measurement` dataclass).

**d sign convention:** negative = sparse vs self_density (or low pedal-bar fraction for whole_piece).

**Hides:** CC64 threshold (>= 64 = sustain on), pedal-bar fraction computation (fraction of bars in region with >= 1 sustain-on event), self_density computation (pedal-bar fraction over full piece), signed deviation formula, error bar (bootstrap over per-bar binary presence + MC threshold jitter).

**Depth verdict:** DEEP.

---

### 6. DynamicsMeasurer (`apps/evals/claim_taxonomy/verifier/measurers/dynamics.py`)

**Interface:** Same `measure()` protocol as TimingMeasurer.

**d sign convention:** negative = flat/narrow vs reference. Whole_piece: d = RMS-contour std normalized by full-piece dynamic range (small std = flat = negative).

**Hides:** librosa.load at 16kHz mono, librosa.feature.rms(hop_length=512), dB conversion (10*log10), within-region frame extraction, contour std computation, whole-piece normalization, error bar (bootstrap over RMS frames + MC jitter).

**Depth verdict:** DEEP.

---

### 7. verify() orchestrator (`apps/evals/claim_taxonomy/verifier/orchestrator.py`)

**Interface:**
```python
def verify(
    claim: dict,
    bundle: dict,
    taxonomy: dict,
    engine: SubstrateErrorEngine | None = None,
) -> VerdictResult:
    """Full verification pipeline for one claim against one bundle.

    Returns VerdictResult for all outcomes including UNVERIFIABLE.
    Never raises (catches UnverifiableError internally).
    If engine is None, creates SubstrateErrorEngine(seed=42).
    """
```

**Measurer registry** (keyed by `dimension.measurement` field in taxonomy):
- `"amt_onsets_region_tempo_fit"` → `TimingMeasurer`
- `"amt_sustain_pedal_events"` → `PedalingMeasurer`
- `"librosa_rms_region_estimator"` → `DynamicsMeasurer`

**Hides:** Registry lookup, pre-measurement UNVERIFIABLE catches (scoped_out, gated, unlocalizable, substrate_failure, region_too_short via UnverifiableError), `_measurement` dict assembly with `{d, tau, error_bar, event_count, localizable: True, substrate_failure: False}`, `route_verdict` call, VerdictResult construction.

**Depth verdict:** DEEP — the entire pipeline collapses to one function call.

---

### 8. CLI (`apps/evals/claim_taxonomy/verifier/cli.py`)

**Interface:**
```
python -m claim_taxonomy.verifier.cli verify \
  --claim claim.json \
  --bundle bundle.json \
  [--taxonomy path/to/claim_taxonomy.json]
```

Prints VerdictResult as JSON to stdout. Default taxonomy path: the committed `claim_taxonomy.json` in the package.

**Depth verdict:** SHALLOW (intentional glue). Justified: zero logic lives here.

---

## Verification Architecture

**Canonical success state:** `verify(claim, bundle, taxonomy)` returns the correct verdict for known-by-construction inputs where the true answer is analytically computable (e.g., inject a 20% tempo rush → expect SUPPORTED for a polarity="-" timing claim).

**Automated check:** `cd apps/evals && uv run pytest claim_taxonomy/tests/ -k "verifier or measurer or resolver or substrate"` — runs all verifier tests.

**Harness:** Task 1 (Group A) builds a `FixtureBundle` factory function that all measurer tests import. This is a prerequisite for Tasks 2-8 but not for the taxonomy v0.1 changes (Tasks 9-10). See Task Groups in the plan.

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/claim_measurement/__init__.py` | Package init | New |
| `model/src/claim_measurement/extractor.py` | BundleExtractor | New |
| `model/src/claim_measurement/tests/__init__.py` | Test package | New |
| `model/src/claim_measurement/tests/test_extractor.py` | Bundle schema smoke test | New |
| `apps/evals/claim_taxonomy/claim_taxonomy.json` | Bump to v0.1; dynamics active | Modify |
| `apps/evals/claim_taxonomy/tests/test_round_trip.py` | Update dynamics test for v0.1 | Modify |
| `apps/evals/claim_taxonomy/tests/test_schema_validates.py` | taxonomy_version v0.1 assertion | Modify |
| `apps/evals/claim_taxonomy/verifier/__init__.py` | Package init; exports VerdictResult, UnverifiableError, verify | New |
| `apps/evals/claim_taxonomy/verifier/models.py` | VerdictResult, UnverifiableError | New |
| `apps/evals/claim_taxonomy/verifier/substrate_error.py` | SubstrateErrorEngine | New |
| `apps/evals/claim_taxonomy/verifier/location_resolver.py` | LocationResolver | New |
| `apps/evals/claim_taxonomy/verifier/measurers/__init__.py` | Package init | New |
| `apps/evals/claim_taxonomy/verifier/measurers/timing.py` | TimingMeasurer | New |
| `apps/evals/claim_taxonomy/verifier/measurers/pedaling.py` | PedalingMeasurer | New |
| `apps/evals/claim_taxonomy/verifier/measurers/dynamics.py` | DynamicsMeasurer | New |
| `apps/evals/claim_taxonomy/verifier/orchestrator.py` | verify() | New |
| `apps/evals/claim_taxonomy/verifier/cli.py` | CLI | New |
| `apps/evals/claim_taxonomy/tests/test_substrate_error.py` | SubstrateErrorEngine tests | New |
| `apps/evals/claim_taxonomy/tests/test_location_resolver.py` | LocationResolver tests | New |
| `apps/evals/claim_taxonomy/tests/test_timing_measurer.py` | TimingMeasurer + injection harness | New |
| `apps/evals/claim_taxonomy/tests/test_pedaling_measurer.py` | PedalingMeasurer + injection harness | New |
| `apps/evals/claim_taxonomy/tests/test_dynamics_measurer.py` | DynamicsMeasurer + injection harness | New |
| `apps/evals/claim_taxonomy/tests/test_verifier_orchestrator.py` | verify() integration | New |
| `docs/model/claim-verifier-signed-d-conventions.md` | Signed-d table + error-bar table | New |

**DO NOT MODIFY:**
- `apps/evals/claim_taxonomy/verdict_dispatch.py`
- `apps/evals/claim_taxonomy/tests/test_verdict_dispatch.py`

---

## Open Questions

- Q: Does the AMT server (`apps/inference/amt/server.py`) expose CC64 pedal events in its response?
  Default: Assume no. BundleExtractor records `pedal_events: []` when AMT does not return CC data. PedalingMeasurer raises `UnverifiableError("substrate_failure", "AMT endpoint does not expose CC64 events; re-run with MIDI export")`. The plan's pedaling measurer tests use injected bundles with synthetic pedal_events regardless.
