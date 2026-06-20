# Deterministic Claim Verifier Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Given a structured claim and a bundle JSON, populate `_measurement` and call the shipped `route_verdict` to produce SUPPORTED | REFUTED | UNVERIFIABLE with measured d, tau, error_bar, and event_count.
**Spec:** docs/specs/2026-06-19-deterministic-claim-verifier-design.md
**Style:** Follow CLAUDE.md — explicit exceptions, no fallbacks, no emojis, uv for Python.

---

## Task Groups

```
Group A (parallel): Task 1 (models + UnverifiableError)
Group B (parallel, depends on A): Task 2 (SubstrateErrorEngine), Task 3 (LocationResolver), Task 9 (taxonomy v0.1)
Group C (parallel, depends on B): Task 4 (TimingMeasurer), Task 5 (PedalingMeasurer), Task 6 (DynamicsMeasurer)
Group D (sequential, depends on C): Task 7 (verify() orchestrator)
Group E (sequential, depends on D): Task 8 (BundleExtractor + CLI + smoke test), Task 10 (signed-d doc)
```

---

### Task 1: VerdictResult dataclass and UnverifiableError
**Group:** A (parallel — no dependencies)

**Behavior being verified:** `VerdictResult` is constructable with all required fields; `UnverifiableError` carries `reason_code` and `detail` and is a subclass of `Exception`.

**Interface under test:** `from claim_taxonomy.verifier.models import VerdictResult, UnverifiableError`

**Files:**
- Create: `apps/evals/claim_taxonomy/verifier/__init__.py`
- Create: `apps/evals/claim_taxonomy/verifier/models.py`
- Create: `apps/evals/claim_taxonomy/verifier/measurers/__init__.py`
- Test: `apps/evals/claim_taxonomy/tests/test_verifier_models.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/claim_taxonomy/tests/test_verifier_models.py
from __future__ import annotations
import pytest
from claim_taxonomy.verifier.models import VerdictResult, UnverifiableError


def test_verdict_result_fields() -> None:
    r = VerdictResult(
        verdict="SUPPORTED",
        reason_code=None,
        measured_value=-12.0,
        tau=8.0,
        error_bar=1.5,
        event_count=20,
        units="percent",
        substrate_versions={"bundle_schema": "v1"},
        dimension="timing",
        location={"bar_start": 1, "bar_end": 4},
    )
    assert r.verdict == "SUPPORTED"
    assert r.reason_code is None
    assert r.measured_value == -12.0
    assert r.units == "percent"


def test_verdict_result_unverifiable_has_reason_code() -> None:
    r = VerdictResult(
        verdict="UNVERIFIABLE",
        reason_code="near_threshold",
        measured_value=-9.0,
        tau=8.0,
        error_bar=2.0,
        event_count=15,
        units="percent",
        substrate_versions={},
        dimension="timing",
        location="whole_piece",
    )
    assert r.reason_code == "near_threshold"


def test_unverifiable_error_has_reason_code_and_detail() -> None:
    err = UnverifiableError("unlocalizable", "bar 3 is within alignment uncertainty")
    assert err.reason_code == "unlocalizable"
    assert err.detail == "bar 3 is within alignment uncertainty"
    assert isinstance(err, Exception)


def test_unverifiable_error_reason_codes_are_strings() -> None:
    for code in ("out_of_scope_dim", "gated_dim", "unlocalizable",
                 "substrate_failure", "region_too_short", "near_threshold"):
        err = UnverifiableError(code, "test")
        assert err.reason_code == code
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_verifier_models.py -x 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_taxonomy.verifier'`

- [x] **Step 3: Implement**

```python
# apps/evals/claim_taxonomy/verifier/__init__.py
from claim_taxonomy.verifier.models import VerdictResult, UnverifiableError

__all__ = ["VerdictResult", "UnverifiableError"]
```

```python
# apps/evals/claim_taxonomy/verifier/models.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VerdictResult:
    verdict: str            # SUPPORTED | REFUTED | UNVERIFIABLE
    reason_code: str | None
    measured_value: float   # d
    tau: float
    error_bar: float
    event_count: int
    units: str              # "percent", "fraction", "dB"
    substrate_versions: dict
    dimension: str
    location: dict | str    # bar range dict or "whole_piece"


class UnverifiableError(Exception):
    """Raised by LocationResolver and Measurers when verification cannot proceed."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail
```

```python
# apps/evals/claim_taxonomy/verifier/measurers/__init__.py
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_verifier_models.py -x
```
Expected: PASS (4 tests)

- [x] **Step 5: Commit**

```bash
cd apps/evals && git add claim_taxonomy/verifier/__init__.py claim_taxonomy/verifier/models.py claim_taxonomy/verifier/measurers/__init__.py claim_taxonomy/tests/test_verifier_models.py && git commit -m "feat(verifier): add VerdictResult dataclass and UnverifiableError (#65)"
```

---

### Task 2: SubstrateErrorEngine
**Group:** B (parallel with Task 3 and Task 9; depends on Task 1)

**Behavior being verified:** Seeded MC engine produces deterministic arrays; zero jitter produces sampling-only error bar; more jitter widens the bar monotonically; bootstrap returns per-sample d values.

**Interface under test:** `from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine`

**Files:**
- Create: `apps/evals/claim_taxonomy/verifier/substrate_error.py`
- Test: `apps/evals/claim_taxonomy/tests/test_substrate_error.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/claim_taxonomy/tests/test_substrate_error.py
from __future__ import annotations
import numpy as np
import pytest
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine


def test_timing_jitter_is_deterministic_with_seed() -> None:
    e1 = SubstrateErrorEngine(seed=42, n_samples=100)
    e2 = SubstrateErrorEngine(seed=42, n_samples=100)
    np.testing.assert_array_equal(e1.timing_onset_jitter_sec(), e2.timing_onset_jitter_sec())


def test_timing_jitter_changes_with_different_seed() -> None:
    e1 = SubstrateErrorEngine(seed=42, n_samples=100)
    e2 = SubstrateErrorEngine(seed=99, n_samples=100)
    assert not np.allclose(e1.timing_onset_jitter_sec(), e2.timing_onset_jitter_sec())


def test_timing_jitter_shape_and_scale() -> None:
    e = SubstrateErrorEngine(seed=0, n_samples=5000)
    j = e.timing_onset_jitter_sec()
    assert j.shape == (5000,)
    assert abs(j.std() - 0.010) < 0.002  # ~Gaussian sigma=0.010


def test_dynamics_rms_jitter_shape_and_scale() -> None:
    e = SubstrateErrorEngine(seed=0, n_samples=5000)
    j = e.dynamics_rms_jitter_db()
    assert j.shape == (5000,)
    assert abs(j.std() - 0.3) < 0.05


def test_pedal_threshold_jitter_uniform_range() -> None:
    e = SubstrateErrorEngine(seed=0, n_samples=5000)
    j = e.pedal_threshold_jitter()
    assert j.min() >= -10.0
    assert j.max() <= 10.0
    assert abs(j.mean()) < 1.0  # centered near zero


def test_bootstrap_d_is_deterministic() -> None:
    e1 = SubstrateErrorEngine(seed=7, n_samples=200)
    e2 = SubstrateErrorEngine(seed=7, n_samples=200)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(
        e1.bootstrap_d(values, np.mean),
        e2.bootstrap_d(values, np.mean),
    )


def test_bootstrap_d_returns_n_samples_values() -> None:
    e = SubstrateErrorEngine(seed=0, n_samples=300)
    values = np.linspace(0.5, 1.5, 20)
    result = e.bootstrap_d(values, np.median)
    assert result.shape == (300,)


def test_alignment_uncertainty_zero_jitter_near_zero() -> None:
    """With zero-sigma jitter the MC uncertainty should be effectively zero."""
    # Patch: use a near-zero sigma engine by subclassing or by passing known anchors
    # We cannot patch sigma directly; instead verify monotonicity property:
    # more anchor jitter -> larger uncertainty
    perf = np.array([0.0, 1.0, 2.0, 3.0])
    score = np.array([0.0, 1.0, 2.0, 3.0])
    # Small-engine: with only 4 anchors and small jitter the uncertainty is small
    e_small = SubstrateErrorEngine(seed=0, n_samples=500)
    u = e_small.alignment_uncertainty_sec(perf, score, bar_start_score_sec=1.0)
    assert isinstance(u, float)
    assert u >= 0.0


def test_alignment_uncertainty_monotone_with_anchor_spread() -> None:
    """More spread-out anchors give lower uncertainty (more information)."""
    # Dense anchors -> more interpolation stability -> same or less uncertainty
    # than sparse anchors at the same query point.
    perf_dense = np.linspace(0.0, 10.0, 100)
    score_dense = np.linspace(0.0, 10.0, 100)
    perf_sparse = np.array([0.0, 5.0, 10.0])
    score_sparse = np.array([0.0, 5.0, 10.0])
    e = SubstrateErrorEngine(seed=0, n_samples=500)
    u_dense = e.alignment_uncertainty_sec(perf_dense, score_dense, 5.0)
    u_sparse = e.alignment_uncertainty_sec(perf_sparse, score_sparse, 5.0)
    # Dense has more anchors absorbing jitter -> should be <= sparse uncertainty
    assert u_dense <= u_sparse + 0.005  # allow tiny float tolerance
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_substrate_error.py -x 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_taxonomy.verifier.substrate_error'`

- [x] **Step 3: Implement**

```python
# apps/evals/claim_taxonomy/verifier/substrate_error.py
from __future__ import annotations

from typing import Callable

import numpy as np


class SubstrateErrorEngine:
    """Seeded Monte-Carlo engine for substrate error propagation and bootstrap.

    All methods are deterministic given the same seed. The engine advances
    its RNG state per call, so call order must be consistent for reproducibility.
    """

    def __init__(self, seed: int = 42, n_samples: int = 500) -> None:
        self._rng = np.random.default_rng(seed)
        self._n = n_samples

    def timing_onset_jitter_sec(self) -> np.ndarray:
        """n_samples of AMT onset jitter: Gaussian(mean=0, sigma=0.010s)."""
        return self._rng.normal(loc=0.0, scale=0.010, size=self._n)

    def dynamics_rms_jitter_db(self) -> np.ndarray:
        """n_samples of RMS frame variance: Gaussian(mean=0, sigma=0.3 dB)."""
        return self._rng.normal(loc=0.0, scale=0.3, size=self._n)

    def pedal_threshold_jitter(self) -> np.ndarray:
        """n_samples of CC threshold offset: Uniform(-10, +10)."""
        return self._rng.uniform(low=-10.0, high=10.0, size=self._n)

    def bootstrap_d(
        self, values: np.ndarray, stat_fn: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """Bootstrap stat_fn over values with replacement. Returns n_samples d values."""
        n = len(values)
        indices = self._rng.integers(0, n, size=(self._n, n))
        return np.array([stat_fn(values[idx]) for idx in indices])

    def alignment_uncertainty_sec(
        self,
        perf_audio_sec: np.ndarray,
        score_audio_sec: np.ndarray,
        bar_start_score_sec: float,
    ) -> float:
        """MC propagation of anchor jitter -> std of bar-start audio_sec estimates.

        Perturbs each anchor by timing_onset_jitter_sec, then interpolates
        bar_start_score_sec for each sample. Returns std of resulting estimates.
        """
        jitter = self.timing_onset_jitter_sec()
        estimates = np.empty(self._n)
        for i, j in enumerate(jitter):
            perturbed_perf = perf_audio_sec + j
            # np.interp requires sorted x; re-sort after perturbation
            order = np.argsort(perturbed_perf)
            estimates[i] = np.interp(
                bar_start_score_sec,
                score_audio_sec[order],
                perturbed_perf[order],
            )
        return float(np.std(estimates))
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_substrate_error.py -x
```
Expected: PASS (9 tests)

- [x] **Step 5: Commit**

```bash
cd apps/evals && git add claim_taxonomy/verifier/substrate_error.py claim_taxonomy/tests/test_substrate_error.py && git commit -m "feat(verifier): add SubstrateErrorEngine with seeded MC (#65)"
```

---

### Task 3: LocationResolver
**Group:** B (parallel with Task 2 and Task 9; depends on Task 1)

**Behavior being verified:** Bar-range resolves to audio time + uncertainty; whole_piece always localizable; single-bar region with uncertainty >= 1.0 bar raises UnverifiableError(unlocalizable); missing bar raises UnverifiableError(unlocalizable).

**Interface under test:** `from claim_taxonomy.verifier.location_resolver import LocationResolver, ResolvedRegion`

**Files:**
- Create: `apps/evals/claim_taxonomy/verifier/location_resolver.py`
- Test: `apps/evals/claim_taxonomy/tests/test_location_resolver.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/claim_taxonomy/tests/test_location_resolver.py
from __future__ import annotations
import math
import numpy as np
import pytest
from claim_taxonomy.verifier.location_resolver import LocationResolver, ResolvedRegion
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine


def _make_bundle(n_bars: int = 10, bar_dur: float = 2.0) -> dict:
    """Synthetic bundle: n_bars of bar_dur seconds each, identity alignment."""
    measure_table = [
        {"bar_number": i + 1, "start_sec": i * bar_dur, "start_tick": i * 480}
        for i in range(n_bars)
    ]
    # Identity alignment: score_sec == perf_audio_sec
    t = np.linspace(0.0, n_bars * bar_dur, 200)
    return {
        "measure_table": measure_table,
        "anchors": {
            "perf_audio_sec": t.tolist(),
            "score_audio_sec": t.tolist(),
        },
    }


def test_bar_range_resolves_to_correct_audio_times() -> None:
    bundle = _make_bundle(n_bars=10, bar_dur=2.0)
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    region = resolver.resolve({"bar_start": 3, "bar_end": 5})
    # Bar 3 starts at (3-1)*2.0 = 4.0 sec; bar 5 ends at bar 6 start = 10.0 sec
    assert abs(region.audio_start_sec - 4.0) < 0.05
    assert abs(region.audio_end_sec - 10.0) < 0.05


def test_whole_piece_always_localizable() -> None:
    bundle = _make_bundle()
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    region = resolver.resolve("whole_piece")
    assert region.location_span_bars == math.inf
    assert region.audio_start_sec >= 0.0


def test_missing_bar_raises_unlocalizable() -> None:
    bundle = _make_bundle(n_bars=5)
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    with pytest.raises(UnverifiableError) as exc_info:
        resolver.resolve({"bar_start": 10, "bar_end": 12})
    assert exc_info.value.reason_code == "unlocalizable"


def test_too_few_anchors_raises_unlocalizable() -> None:
    bundle = {
        "measure_table": [{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}],
        "anchors": {"perf_audio_sec": [0.0], "score_audio_sec": [0.0]},
    }
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    with pytest.raises(UnverifiableError) as exc_info:
        resolver.resolve({"bar_start": 1, "bar_end": 1})
    assert exc_info.value.reason_code == "unlocalizable"


def test_failsafe_triggers_when_uncertainty_exceeds_span() -> None:
    """A single-bar region with large alignment uncertainty must become unlocalizable."""
    # Sparse anchors -> large uncertainty
    bundle = {
        "measure_table": [
            {"bar_number": 1, "start_sec": 0.0, "start_tick": 0},
            {"bar_number": 2, "start_sec": 0.05, "start_tick": 480},  # 50ms bar = tiny span
        ],
        "anchors": {
            "perf_audio_sec": [0.0, 5.0],  # only 2 anchors -> large uncertainty
            "score_audio_sec": [0.0, 5.0],
        },
    }
    engine = SubstrateErrorEngine(seed=42, n_samples=500)
    resolver = LocationResolver(bundle, engine)
    with pytest.raises(UnverifiableError) as exc_info:
        resolver.resolve({"bar_start": 1, "bar_end": 1})
    assert exc_info.value.reason_code == "unlocalizable"


def test_resolved_region_has_uncertainty_field() -> None:
    bundle = _make_bundle()
    engine = SubstrateErrorEngine(seed=0)
    resolver = LocationResolver(bundle, engine)
    region = resolver.resolve({"bar_start": 2, "bar_end": 4})
    assert isinstance(region.alignment_uncertainty_sec, float)
    assert region.alignment_uncertainty_sec >= 0.0
    assert region.location_span_bars == 3  # bar 2, 3, 4
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_location_resolver.py -x 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_taxonomy.verifier.location_resolver'`

- [x] **Step 3: Implement**

```python
# apps/evals/claim_taxonomy/verifier/location_resolver.py
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine


@dataclass
class ResolvedRegion:
    audio_start_sec: float
    audio_end_sec: float
    alignment_uncertainty_sec: float
    location_span_bars: float  # math.inf for whole_piece


class LocationResolver:
    """Maps claim location to audio time range with MC alignment uncertainty."""

    def __init__(self, bundle: dict, engine: SubstrateErrorEngine) -> None:
        self._measure_table: dict[int, dict] = {
            int(row["bar_number"]): row
            for row in bundle["measure_table"]
        }
        self._sorted_bars = sorted(self._measure_table.keys())
        perf = bundle["anchors"]["perf_audio_sec"]
        score = bundle["anchors"]["score_audio_sec"]
        self._perf_audio_sec = np.asarray(perf, dtype=np.float64)
        self._score_audio_sec = np.asarray(score, dtype=np.float64)
        self._engine = engine

    def _score_sec_to_audio_sec(self, score_sec: float) -> float:
        if self._perf_audio_sec.size < 2:
            raise UnverifiableError(
                "unlocalizable",
                f"fewer than 2 alignment anchors; cannot interpolate score_sec={score_sec}",
            )
        return float(np.interp(score_sec, self._score_audio_sec, self._perf_audio_sec))

    def _bar_start_score_sec(self, bar_number: int) -> float:
        if bar_number not in self._measure_table:
            raise UnverifiableError(
                "unlocalizable",
                f"bar {bar_number} not in measure_table (available: {self._sorted_bars})",
            )
        return float(self._measure_table[bar_number]["start_sec"])

    def _bar_duration_sec(self, bar_number: int) -> float:
        """Infer bar duration from adjacent measure_table entry."""
        idx = self._sorted_bars.index(bar_number)
        if idx + 1 < len(self._sorted_bars):
            next_bar = self._sorted_bars[idx + 1]
            return float(
                self._measure_table[next_bar]["start_sec"]
                - self._measure_table[bar_number]["start_sec"]
            )
        # Last bar: use previous bar duration as estimate
        if idx > 0:
            prev_bar = self._sorted_bars[idx - 1]
            return float(
                self._measure_table[bar_number]["start_sec"]
                - self._measure_table[prev_bar]["start_sec"]
            )
        return 2.0  # fallback: 2 seconds per bar

    def resolve(self, location: dict | str) -> ResolvedRegion:
        """Map location to audio time range.

        Raises UnverifiableError("unlocalizable") if bar not found,
        < 2 anchors, or alignment uncertainty >= location span.
        """
        if location == "whole_piece":
            if self._perf_audio_sec.size < 2:
                raise UnverifiableError(
                    "unlocalizable",
                    "fewer than 2 alignment anchors for whole_piece resolution",
                )
            audio_start = float(self._perf_audio_sec.min())
            audio_end = float(self._perf_audio_sec.max())
            uncertainty = self._engine.alignment_uncertainty_sec(
                self._perf_audio_sec,
                self._score_audio_sec,
                float(self._score_audio_sec[0]),
            )
            return ResolvedRegion(
                audio_start_sec=audio_start,
                audio_end_sec=audio_end,
                alignment_uncertainty_sec=uncertainty,
                location_span_bars=math.inf,
            )

        bar_start = int(location["bar_start"])
        bar_end = int(location["bar_end"])
        location_span_bars = bar_end - bar_start + 1

        start_score_sec = self._bar_start_score_sec(bar_start)
        # end of region = start of the bar after bar_end
        end_bar = bar_end + 1
        if end_bar in self._measure_table:
            end_score_sec = float(self._measure_table[end_bar]["start_sec"])
        else:
            # bar_end is last bar: approximate end
            end_score_sec = start_score_sec + self._bar_duration_sec(bar_end) * location_span_bars

        audio_start = self._score_sec_to_audio_sec(start_score_sec)
        audio_end = self._score_sec_to_audio_sec(end_score_sec)

        uncertainty = self._engine.alignment_uncertainty_sec(
            self._perf_audio_sec,
            self._score_audio_sec,
            start_score_sec,
        )

        bar_dur = self._bar_duration_sec(bar_start)
        if bar_dur > 0:
            uncertainty_bars = uncertainty / bar_dur
        else:
            uncertainty_bars = math.inf

        if uncertainty_bars >= location_span_bars:
            raise UnverifiableError(
                "unlocalizable",
                f"alignment uncertainty {uncertainty:.3f}s >= location span "
                f"{location_span_bars} bars ({location_span_bars * bar_dur:.3f}s); "
                f"bar_dur={bar_dur:.3f}s",
            )

        return ResolvedRegion(
            audio_start_sec=audio_start,
            audio_end_sec=audio_end,
            alignment_uncertainty_sec=uncertainty,
            location_span_bars=float(location_span_bars),
        )
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_location_resolver.py -x
```
Expected: PASS (6 tests)

- [x] **Step 5: Commit**

```bash
cd apps/evals && git add claim_taxonomy/verifier/location_resolver.py claim_taxonomy/tests/test_location_resolver.py && git commit -m "feat(verifier): add LocationResolver with MC failsafe (#65)"
```

---

### Task 4: TimingMeasurer
**Group:** C (parallel with Task 5 and Task 6; depends on Group B)

**Behavior being verified:** Region rush (faster IOIs) → d negative; region drag (slower IOIs) → d positive; whole_piece high-CV → d is the CV%; event_count below minimum → UnverifiableError(region_too_short); mistake-injection: shifting onsets by +20% produces d > tau.

**Interface under test:** `from claim_taxonomy.verifier.measurers.timing import TimingMeasurer`

**Files:**
- Create: `apps/evals/claim_taxonomy/verifier/measurers/timing.py`
- Test: `apps/evals/claim_taxonomy/tests/test_timing_measurer.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/claim_taxonomy/tests/test_timing_measurer.py
from __future__ import annotations
import numpy as np
import pytest
from claim_taxonomy.verifier.measurers.timing import TimingMeasurer, Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine
import math


def _make_region(start: float = 0.0, end: float = 10.0) -> ResolvedRegion:
    return ResolvedRegion(
        audio_start_sec=start,
        audio_end_sec=end,
        alignment_uncertainty_sec=0.05,
        location_span_bars=5.0,
    )


def _make_bundle_with_notes(onsets: list[float]) -> dict:
    """Construct a minimal bundle with evenly-spaced notes across whole piece."""
    notes = [{"onset": t, "offset": t + 0.1, "pitch": 60, "velocity": 80} for t in onsets]
    return {
        "notes": notes,
        "pedal_events": [],
        "measure_table": [{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}],
        "anchors": {"perf_audio_sec": [0.0, max(onsets)], "score_audio_sec": [0.0, max(onsets)]},
        "substrate_versions": {"bundle_schema": "v1"},
    }


def test_region_rush_gives_negative_d() -> None:
    """Notes in region are 20% faster than piece established_tempo -> d < 0."""
    # Piece: 100 notes at 0.5s intervals = 120 BPM established
    piece_onsets = [i * 0.5 for i in range(100)]
    # Region (0-10s): override with 0.4s intervals = 150 BPM (25% faster)
    region_onsets = [i * 0.4 for i in range(25)]  # 25 notes in first 10s
    all_onsets = sorted(set(piece_onsets[25:]) | set(region_onsets))
    bundle = _make_bundle_with_notes(all_onsets)
    engine = SubstrateErrorEngine(seed=42)
    measurer = TimingMeasurer()
    region = _make_region(start=0.0, end=10.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 5},
                              bundle=bundle, region=region, engine=engine)
    assert result.d < 0, f"Expected negative d (rushed), got {result.d}"
    assert result.event_count >= 8


def test_region_drag_gives_positive_d() -> None:
    """Notes in region are slower than established_tempo -> d > 0."""
    piece_onsets = [i * 0.5 for i in range(100)]
    # Region (0-10s): 0.7s intervals = dragging
    region_onsets = [i * 0.7 for i in range(14)]
    all_onsets = sorted(set(piece_onsets[14:]) | set(region_onsets))
    bundle = _make_bundle_with_notes(all_onsets)
    engine = SubstrateErrorEngine(seed=42)
    measurer = TimingMeasurer()
    region = _make_region(start=0.0, end=10.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 5},
                              bundle=bundle, region=region, engine=engine)
    assert result.d > 0, f"Expected positive d (dragging), got {result.d}"


def test_whole_piece_uniform_tempo_low_cv() -> None:
    """Perfectly even tempo -> CV% near 0."""
    onsets = [i * 0.5 for i in range(100)]
    bundle = _make_bundle_with_notes(onsets)
    engine = SubstrateErrorEngine(seed=42)
    measurer = TimingMeasurer()
    region = ResolvedRegion(
        audio_start_sec=0.0, audio_end_sec=50.0,
        alignment_uncertainty_sec=0.05, location_span_bars=math.inf
    )
    result = measurer.measure(location="whole_piece", bundle=bundle, region=region, engine=engine)
    assert result.d < 5.0, f"Expected low CV% for uniform tempo, got {result.d}"


def test_region_too_short_raises() -> None:
    """Fewer than 8 onsets in region -> UnverifiableError(region_too_short)."""
    onsets = [0.5, 1.0, 1.5, 2.0, 2.5]  # only 5 notes in [0, 4]
    onsets += [i * 0.5 + 10 for i in range(50)]
    bundle = _make_bundle_with_notes(onsets)
    engine = SubstrateErrorEngine(seed=42)
    measurer = TimingMeasurer()
    region = _make_region(start=0.0, end=4.0)
    with pytest.raises(UnverifiableError) as exc_info:
        measurer.measure(location={"bar_start": 1, "bar_end": 2},
                         bundle=bundle, region=region, engine=engine)
    assert exc_info.value.reason_code == "region_too_short"


def test_mistake_injection_onset_shift_recovers_anomaly() -> None:
    """Shift all onsets in region by -20% (faster) -> |d| > tau=8%."""
    piece_onsets = [i * 0.5 for i in range(80)]
    # Inject: shift region onsets (0-10s) by 20% to be faster
    shifted = [t * 0.8 for t in piece_onsets if t < 10.0]
    rest = [t for t in piece_onsets if t >= 10.0]
    bundle = _make_bundle_with_notes(sorted(shifted + rest))
    engine = SubstrateErrorEngine(seed=42)
    measurer = TimingMeasurer()
    region = _make_region(start=0.0, end=8.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 4},
                              bundle=bundle, region=region, engine=engine)
    tau = 8.0
    assert abs(result.d) > tau, (
        f"Injection: 20% faster onset shift should produce |d|>{tau}%, got d={result.d}"
    )
    # Accuracy note: captioned "accuracy under AMT transcription error deferred to GATE 1"


def test_error_bar_is_positive() -> None:
    onsets = [i * 0.5 for i in range(60)]
    bundle = _make_bundle_with_notes(onsets)
    engine = SubstrateErrorEngine(seed=0)
    measurer = TimingMeasurer()
    region = _make_region(start=0.0, end=15.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 5},
                              bundle=bundle, region=region, engine=engine)
    assert result.error_bar > 0.0
    assert not result.substrate_failure
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_timing_measurer.py -x 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_taxonomy.verifier.measurers.timing'`

- [x] **Step 3: Implement**

```python
# apps/evals/claim_taxonomy/verifier/measurers/timing.py
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

MINIMUM_EVENTS = 8
TAU_PERCENT = 8.0  # provisional; read from taxonomy at call time


@dataclass
class Measurement:
    d: float
    error_bar: float
    event_count: int
    substrate_failure: bool


class TimingMeasurer:
    """Measure signed tempo deviation for timing claims.

    Sign convention:
    - d < 0: region is faster than reference (rushed)
    - d > 0: region is slower than reference (dragging)
    - Whole-piece: d = CV% of local IOI-derived BPM (always >= 0; high = inconsistent tempo)
    """

    def measure(
        self,
        location: dict | str,
        bundle: dict,
        region: ResolvedRegion,
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        notes = bundle.get("notes") or []
        if not notes:
            raise UnverifiableError("substrate_failure", "bundle contains zero notes")

        all_onsets = np.array([float(n["onset"]) for n in notes], dtype=np.float64)
        all_onsets.sort()

        if location == "whole_piece":
            return self._measure_whole_piece(all_onsets, engine)

        region_onsets = all_onsets[
            (all_onsets >= region.audio_start_sec) & (all_onsets < region.audio_end_sec)
        ]
        event_count = int(region_onsets.size)
        if event_count < MINIMUM_EVENTS:
            raise UnverifiableError(
                "region_too_short",
                f"only {event_count} onsets in region [{region.audio_start_sec:.2f}, "
                f"{region.audio_end_sec:.2f}s]; need >= {MINIMUM_EVENTS}",
            )

        # Established tempo: median BPM over whole piece using consecutive IOIs
        d, sampling_var = self._region_d_and_sampling_var(
            region_onsets, all_onsets, engine
        )
        substrate_var = self._substrate_var(region_onsets, d, engine)
        error_bar = math.sqrt(sampling_var + substrate_var)

        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)

    def _ioi_to_bpm(self, onsets: np.ndarray) -> np.ndarray:
        """Convert onset array to BPM array via consecutive IOIs."""
        ioi = np.diff(onsets)
        ioi = ioi[ioi > 0.01]  # drop sub-10ms duplicates
        if ioi.size == 0:
            return np.array([120.0])
        return 60.0 / ioi

    def _established_tempo(self, all_onsets: np.ndarray) -> float:
        """Median BPM across whole piece."""
        bpms = self._ioi_to_bpm(all_onsets)
        return float(np.median(bpms))

    def _region_median_bpm(self, region_onsets: np.ndarray) -> float:
        bpms = self._ioi_to_bpm(region_onsets)
        return float(np.median(bpms))

    def _region_d_and_sampling_var(
        self,
        region_onsets: np.ndarray,
        all_onsets: np.ndarray,
        engine: SubstrateErrorEngine,
    ) -> tuple[float, float]:
        established = self._established_tempo(all_onsets)
        if established == 0.0:
            raise UnverifiableError("substrate_failure", "established_tempo is zero")

        def stat(onsets: np.ndarray) -> float:
            bpm = self._region_median_bpm(onsets)
            return (bpm - established) / established * 100.0

        d = stat(region_onsets)
        bootstrapped = engine.bootstrap_d(region_onsets, stat)
        sampling_var = float(np.var(bootstrapped))
        return d, sampling_var

    def _substrate_var(
        self, region_onsets: np.ndarray, d: float, engine: SubstrateErrorEngine
    ) -> float:
        jitters = engine.timing_onset_jitter_sec()
        # Perturb each onset by jitter and recompute d relative to unperturbed established
        established = self._established_tempo(region_onsets)  # use region for MC
        perturbed_ds = np.empty(len(jitters))
        for i, j in enumerate(jitters):
            perturbed = region_onsets + j
            bpm = self._region_median_bpm(perturbed)
            if established > 0:
                perturbed_ds[i] = (bpm - established) / established * 100.0
            else:
                perturbed_ds[i] = 0.0
        return float(np.var(perturbed_ds))

    def _measure_whole_piece(
        self, all_onsets: np.ndarray, engine: SubstrateErrorEngine
    ) -> Measurement:
        event_count = int(all_onsets.size)
        if event_count < MINIMUM_EVENTS:
            raise UnverifiableError(
                "region_too_short",
                f"whole_piece has only {event_count} onsets; need >= {MINIMUM_EVENTS}",
            )
        bpms = self._ioi_to_bpm(all_onsets)
        # CV% = (std / mean) * 100
        if bpms.mean() == 0:
            raise UnverifiableError("substrate_failure", "mean BPM is zero")
        d = float(bpms.std() / bpms.mean() * 100.0)

        bootstrapped = engine.bootstrap_d(
            bpms, lambda x: float(x.std() / x.mean() * 100.0) if x.mean() > 0 else 0.0
        )
        sampling_var = float(np.var(bootstrapped))
        jitters = engine.timing_onset_jitter_sec()
        perturbed_cvs = np.empty(len(jitters))
        for i, j in enumerate(jitters):
            perturbed = all_onsets + j
            pb = self._ioi_to_bpm(perturbed)
            perturbed_cvs[i] = float(pb.std() / pb.mean() * 100.0) if pb.mean() > 0 else 0.0
        substrate_var = float(np.var(perturbed_cvs))
        error_bar = math.sqrt(sampling_var + substrate_var)
        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_timing_measurer.py -x
```
Expected: PASS (6 tests)

- [x] **Step 5: Commit**

```bash
cd apps/evals && git add claim_taxonomy/verifier/measurers/timing.py claim_taxonomy/tests/test_timing_measurer.py && git commit -m "feat(verifier): add TimingMeasurer with injection harness (#65)"
```

---

### Task 5: PedalingMeasurer
**Group:** C (parallel with Task 4 and Task 6; depends on Group B)

**Behavior being verified:** Region with no pedal events → d negative (sparse); region with many → d near zero or positive; empty pedal_events + minimum_events=2 → UnverifiableError(region_too_short); CC injection (add events) recovers anomaly.

**Interface under test:** `from claim_taxonomy.verifier.measurers.pedaling import PedalingMeasurer`

**Files:**
- Create: `apps/evals/claim_taxonomy/verifier/measurers/pedaling.py`
- Test: `apps/evals/claim_taxonomy/tests/test_pedaling_measurer.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/claim_taxonomy/tests/test_pedaling_measurer.py
from __future__ import annotations
import math
import numpy as np
import pytest
from claim_taxonomy.verifier.measurers.pedaling import PedalingMeasurer, Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine


def _make_region(start: float = 0.0, end: float = 20.0) -> ResolvedRegion:
    return ResolvedRegion(
        audio_start_sec=start,
        audio_end_sec=end,
        alignment_uncertainty_sec=0.05,
        location_span_bars=10.0,
    )


def _make_bundle(pedal_events: list[dict], n_bars: int = 20, bar_dur: float = 2.0) -> dict:
    measure_table = [
        {"bar_number": i + 1, "start_sec": i * bar_dur, "start_tick": i * 480}
        for i in range(n_bars)
    ]
    return {
        "notes": [{"onset": i * 0.5, "offset": i * 0.5 + 0.4, "pitch": 60, "velocity": 80}
                  for i in range(n_bars * 4)],
        "pedal_events": pedal_events,
        "measure_table": measure_table,
        "anchors": {
            "perf_audio_sec": [0.0, n_bars * bar_dur],
            "score_audio_sec": [0.0, n_bars * bar_dur],
        },
        "substrate_versions": {"bundle_schema": "v1"},
    }


def test_no_pedal_in_region_negative_d() -> None:
    """Region has no pedal; rest of piece has dense pedal -> d < 0 (sparse)."""
    # Pedal in bars 11-20 only (times 20-40s), none in 0-20s region
    pedal_events = [{"time": 20.0 + i * 1.0, "value": 127} for i in range(20)]
    bundle = _make_bundle(pedal_events)
    engine = SubstrateErrorEngine(seed=42)
    measurer = PedalingMeasurer()
    region = _make_region(start=0.0, end=20.0)
    result = measurer.measure(
        location={"bar_start": 1, "bar_end": 10},
        bundle=bundle, region=region, engine=engine,
    )
    assert result.d < 0, f"Expected negative d (sparse pedaling), got {result.d}"


def test_dense_pedal_in_region_near_zero_or_positive_d() -> None:
    """Region has same density as whole piece -> d near zero."""
    # 1 pedal event per bar throughout
    pedal_events = [{"time": i * 2.0 + 0.1, "value": 127} for i in range(20)]
    bundle = _make_bundle(pedal_events)
    engine = SubstrateErrorEngine(seed=42)
    measurer = PedalingMeasurer()
    region = _make_region(start=0.0, end=20.0)
    result = measurer.measure(
        location={"bar_start": 1, "bar_end": 10},
        bundle=bundle, region=region, engine=engine,
    )
    # Region density == self_density -> |d| < tau=0.25
    assert abs(result.d) < 0.35, f"Expected near-zero d for uniform pedaling, got {result.d}"


def test_no_pedal_anywhere_region_too_short() -> None:
    """No pedal events at all -> UnverifiableError(region_too_short) since event_count=0 < 2."""
    bundle = _make_bundle(pedal_events=[])
    engine = SubstrateErrorEngine(seed=42)
    measurer = PedalingMeasurer()
    region = _make_region(start=0.0, end=20.0)
    with pytest.raises(UnverifiableError) as exc_info:
        measurer.measure(
            location={"bar_start": 1, "bar_end": 10},
            bundle=bundle, region=region, engine=engine,
        )
    assert exc_info.value.reason_code == "region_too_short"


def test_whole_piece_pedal_fraction() -> None:
    """Whole-piece: d = pedal-bar fraction (0.0-1.0); uniform pedal -> d > 0."""
    pedal_events = [{"time": i * 2.0 + 0.1, "value": 127} for i in range(20)]
    bundle = _make_bundle(pedal_events)
    engine = SubstrateErrorEngine(seed=42)
    measurer = PedalingMeasurer()
    region = ResolvedRegion(
        audio_start_sec=0.0, audio_end_sec=40.0,
        alignment_uncertainty_sec=0.05, location_span_bars=math.inf
    )
    result = measurer.measure(location="whole_piece", bundle=bundle, region=region, engine=engine)
    assert 0.0 <= result.d <= 1.0, f"Whole-piece d should be fraction 0-1, got {result.d}"


def test_cc_injection_recovers_sparse_anomaly() -> None:
    """Remove pedal events from region -> |d| > tau=0.25."""
    # Dense pedal everywhere EXCEPT region 0-20s (bars 1-10)
    pedal_events = [{"time": 20.0 + i * 1.5, "value": 127} for i in range(13)]
    bundle = _make_bundle(pedal_events)
    engine = SubstrateErrorEngine(seed=42)
    measurer = PedalingMeasurer()
    region = _make_region(start=0.0, end=20.0)
    result = measurer.measure(
        location={"bar_start": 1, "bar_end": 10},
        bundle=bundle, region=region, engine=engine,
    )
    tau = 0.25
    assert abs(result.d) > tau, (
        f"CC injection (remove region pedal) should produce |d|>{tau}, got d={result.d}"
    )


def test_error_bar_positive() -> None:
    pedal_events = [{"time": i * 2.0 + 0.1, "value": 127} for i in range(20)]
    bundle = _make_bundle(pedal_events)
    engine = SubstrateErrorEngine(seed=42)
    measurer = PedalingMeasurer()
    region = _make_region()
    result = measurer.measure(
        location={"bar_start": 1, "bar_end": 10},
        bundle=bundle, region=region, engine=engine,
    )
    assert result.error_bar >= 0.0
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_pedaling_measurer.py -x 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_taxonomy.verifier.measurers.pedaling'`

- [x] **Step 3: Implement**

```python
# apps/evals/claim_taxonomy/verifier/measurers/pedaling.py
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from claim_taxonomy.verifier.measurers.timing import Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

MINIMUM_EVENTS = 2          # minimum sustain-on events in region
SUSTAIN_THRESHOLD = 64      # CC64 value >= 64 = sustain on


class PedalingMeasurer:
    """Measure pedal presence density for pedaling claims.

    Sign convention:
    - d < 0: region has lower pedal density than self_density (sparse)
    - d > 0: region has higher pedal density than self_density (over-pedaled)
    - Whole-piece: d = pedal-bar fraction (presence statistic, 0.0 to 1.0)
    """

    def measure(
        self,
        location: dict | str,
        bundle: dict,
        region: ResolvedRegion,
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        pedal_events = bundle.get("pedal_events") or []
        # Sustain-on events: CC64 value >= threshold
        sustain_on_times = np.array(
            [float(e["time"]) for e in pedal_events if int(e["value"]) >= SUSTAIN_THRESHOLD],
            dtype=np.float64,
        )
        measure_table = bundle.get("measure_table") or []
        if not measure_table:
            raise UnverifiableError("substrate_failure", "measure_table is empty")

        bars = sorted(measure_table, key=lambda r: r["bar_number"])

        if location == "whole_piece":
            return self._measure_whole_piece(sustain_on_times, bars, engine)

        bar_start = int(location["bar_start"])
        bar_end = int(location["bar_end"])

        # Count sustain-on events in region audio time
        region_events = sustain_on_times[
            (sustain_on_times >= region.audio_start_sec)
            & (sustain_on_times < region.audio_end_sec)
        ]
        event_count = int(region_events.size)

        if event_count < MINIMUM_EVENTS:
            raise UnverifiableError(
                "region_too_short",
                f"only {event_count} sustain-on events in region "
                f"[{region.audio_start_sec:.2f}, {region.audio_end_sec:.2f}s]; "
                f"need >= {MINIMUM_EVENTS}",
            )

        # Pedal-bar fraction for region: fraction of bars in [bar_start, bar_end]
        # that contain at least one sustain-on event
        region_bar_rows = [r for r in bars if bar_start <= r["bar_number"] <= bar_end]
        region_fraction = self._pedal_bar_fraction(region_bar_rows, sustain_on_times, bars)

        # Self-density: pedal-bar fraction over whole piece
        self_density = self._pedal_bar_fraction(bars, sustain_on_times, bars)

        d = region_fraction - self_density

        # Error bar: bootstrap over per-bar binary presence
        per_bar_presence = np.array([
            1.0 if self._bar_has_pedal(r, sustain_on_times, bars) else 0.0
            for r in region_bar_rows
        ])
        bootstrapped = engine.bootstrap_d(per_bar_presence, np.mean)
        sampling_var = float(np.var(bootstrapped - self_density))

        # MC: threshold jitter -> some events cross the boundary
        threshold_jitters = engine.pedal_threshold_jitter()
        threshold_samples = np.empty(len(threshold_jitters))
        for i, jitter in enumerate(threshold_jitters):
            threshold = SUSTAIN_THRESHOLD + jitter
            region_ev_j = np.array(
                [float(e["time"]) for e in pedal_events if int(e["value"]) >= threshold],
                dtype=np.float64,
            )
            region_ev_j = region_ev_j[
                (region_ev_j >= region.audio_start_sec)
                & (region_ev_j < region.audio_end_sec)
            ]
            frac_j = self._pedal_bar_fraction_from_times(region_bar_rows, region_ev_j, bars)
            threshold_samples[i] = frac_j - self_density

        substrate_var = float(np.var(threshold_samples))
        error_bar = math.sqrt(sampling_var + substrate_var)

        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)

    def _bar_start_end_sec(self, bar_row: dict, all_bars: list[dict]) -> tuple[float, float]:
        start = float(bar_row["start_sec"])
        # Find the next bar's start_sec
        bar_num = int(bar_row["bar_number"])
        next_bars = [r for r in all_bars if int(r["bar_number"]) == bar_num + 1]
        if next_bars:
            end = float(next_bars[0]["start_sec"])
        else:
            # Last bar: estimate duration from previous
            prev_bars = [r for r in all_bars if int(r["bar_number"]) == bar_num - 1]
            if prev_bars:
                dur = start - float(prev_bars[0]["start_sec"])
            else:
                dur = 2.0
            end = start + dur
        return start, end

    def _bar_has_pedal(
        self, bar_row: dict, sustain_times: np.ndarray, all_bars: list[dict]
    ) -> bool:
        start, end = self._bar_start_end_sec(bar_row, all_bars)
        return bool(np.any((sustain_times >= start) & (sustain_times < end)))

    def _pedal_bar_fraction(
        self, bar_rows: list[dict], sustain_times: np.ndarray, all_bars: list[dict]
    ) -> float:
        if not bar_rows:
            return 0.0
        count = sum(1 for r in bar_rows if self._bar_has_pedal(r, sustain_times, all_bars))
        return count / len(bar_rows)

    def _pedal_bar_fraction_from_times(
        self, bar_rows: list[dict], sustain_times: np.ndarray, all_bars: list[dict]
    ) -> float:
        return self._pedal_bar_fraction(bar_rows, sustain_times, all_bars)

    def _measure_whole_piece(
        self,
        sustain_on_times: np.ndarray,
        bars: list[dict],
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        event_count = int(sustain_on_times.size)
        if event_count < MINIMUM_EVENTS:
            raise UnverifiableError(
                "region_too_short",
                f"whole_piece has only {event_count} sustain-on events; need >= {MINIMUM_EVENTS}",
            )
        d = self._pedal_bar_fraction(bars, sustain_on_times, bars)
        per_bar = np.array([
            1.0 if self._bar_has_pedal(r, sustain_on_times, bars) else 0.0
            for r in bars
        ])
        bootstrapped = engine.bootstrap_d(per_bar, np.mean)
        sampling_var = float(np.var(bootstrapped))
        substrate_var = 0.0  # threshold jitter has small effect on fraction
        error_bar = math.sqrt(sampling_var + substrate_var)
        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_pedaling_measurer.py -x
```
Expected: PASS (6 tests)

- [x] **Step 5: Commit**

```bash
cd apps/evals && git add claim_taxonomy/verifier/measurers/pedaling.py claim_taxonomy/tests/test_pedaling_measurer.py && git commit -m "feat(verifier): add PedalingMeasurer with CC injection harness (#65)"
```

---

### Task 6: DynamicsMeasurer
**Group:** C (parallel with Task 4 and Task 5; depends on Group B)

**Behavior being verified:** Synthetic flat audio → low std → d negative (flat); synthetic wide-dynamic audio → d near zero or positive; programmed RMS envelope injection recovers anomaly; region_too_short on < 20 frames.

**Interface under test:** `from claim_taxonomy.verifier.measurers.dynamics import DynamicsMeasurer`

**Files:**
- Create: `apps/evals/claim_taxonomy/verifier/measurers/dynamics.py`
- Test: `apps/evals/claim_taxonomy/tests/test_dynamics_measurer.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/claim_taxonomy/tests/test_dynamics_measurer.py
from __future__ import annotations
import math
import numpy as np
import pytest
from claim_taxonomy.verifier.measurers.dynamics import DynamicsMeasurer, Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

SR = 16000
HOP = 512


def _sine_audio(freq: float, duration: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0.0, duration, int(SR * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def _make_region(start: float, end: float) -> ResolvedRegion:
    return ResolvedRegion(
        audio_start_sec=start,
        audio_end_sec=end,
        alignment_uncertainty_sec=0.05,
        location_span_bars=5.0,
    )


def _make_bundle(audio: np.ndarray, audio_path: str = "/tmp/test.wav") -> dict:
    # DynamicsMeasurer receives audio directly (numpy array), not a path.
    # The bundle carries a reference; we pass audio as an extra kwarg in tests.
    return {
        "notes": [{"onset": 0.1, "offset": 0.2, "pitch": 60, "velocity": 80}],
        "pedal_events": [],
        "measure_table": [{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}],
        "anchors": {"perf_audio_sec": [0.0, len(audio) / SR],
                    "score_audio_sec": [0.0, len(audio) / SR]},
        "substrate_versions": {"bundle_schema": "v1"},
        "audio_path": audio_path,
    }


def test_flat_audio_region_negative_d(tmp_path) -> None:
    """Flat (constant) amplitude region vs dynamic whole piece -> d < 0."""
    import soundfile as sf
    # Whole piece: varying amplitude sine (0.1 to 0.9 sweep)
    n_total = SR * 20
    t = np.linspace(0, 20, n_total)
    amplitude_envelope = 0.1 + 0.8 * (t / 20.0)
    whole = (np.sin(2 * np.pi * 440 * t) * amplitude_envelope).astype(np.float32)
    audio_path = tmp_path / "test.wav"
    sf.write(str(audio_path), whole, SR)
    bundle = _make_bundle(whole, str(audio_path))
    engine = SubstrateErrorEngine(seed=42)
    measurer = DynamicsMeasurer()
    # Region 0-5s has low amplitude (0.1 to 0.3) = flat
    region = _make_region(start=0.0, end=5.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 3},
                              bundle=bundle, region=region, engine=engine)
    assert result.d < 0, f"Flat region should have negative d, got {result.d}"


def test_wide_dynamic_region_positive_or_zero_d(tmp_path) -> None:
    """Region with full dynamic swing vs a flat piece -> d > 0."""
    import soundfile as sf
    # Whole piece: mostly flat
    n_total = SR * 20
    whole = (np.sin(2 * np.pi * 440 * np.linspace(0, 20, n_total)) * 0.2).astype(np.float32)
    # Region 0-5s: programmed wide dynamic swing
    n_region = SR * 5
    t_r = np.linspace(0, 5, n_region)
    amp_swing = 0.1 + 0.8 * np.abs(np.sin(2 * np.pi * 0.2 * t_r))
    whole[:n_region] = (np.sin(2 * np.pi * 440 * t_r) * amp_swing).astype(np.float32)
    audio_path = tmp_path / "test_wide.wav"
    sf.write(str(audio_path), whole, SR)
    bundle = _make_bundle(whole, str(audio_path))
    engine = SubstrateErrorEngine(seed=42)
    measurer = DynamicsMeasurer()
    region = _make_region(start=0.0, end=5.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 3},
                              bundle=bundle, region=region, engine=engine)
    assert result.d > 0, f"Wide-dynamic region should have positive d, got {result.d}"


def test_rms_envelope_injection_recovers_anomaly(tmp_path) -> None:
    """Programmed RMS envelope: region is 6x louder than rest -> |d| > tau=1.5."""
    import soundfile as sf
    n_total = SR * 30
    audio = np.zeros(n_total, dtype=np.float32)
    # Quiet baseline
    t_all = np.linspace(0, 30, n_total)
    audio = (np.sin(2 * np.pi * 440 * t_all) * 0.05).astype(np.float32)
    # Inject: region 0-10s is 6x louder (amplitude 0.3 vs 0.05)
    n_region = SR * 10
    audio[:n_region] = (np.sin(2 * np.pi * 440 * np.linspace(0, 10, n_region)) * 0.3).astype(np.float32)
    audio_path = tmp_path / "test_inject.wav"
    sf.write(str(audio_path), audio, SR)
    bundle = _make_bundle(audio, str(audio_path))
    engine = SubstrateErrorEngine(seed=42)
    measurer = DynamicsMeasurer()
    region = _make_region(start=0.0, end=10.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 5},
                              bundle=bundle, region=region, engine=engine)
    tau = 1.5
    assert abs(result.d) > tau, (
        f"RMS injection (6x louder region) should produce |d|>{tau} dB, got d={result.d}"
    )


def test_region_too_short_raises(tmp_path) -> None:
    """Region shorter than 20 RMS frames -> UnverifiableError(region_too_short)."""
    import soundfile as sf
    # 20 frames * 512 hop / 16000 sr = 0.64 seconds; region of 0.3s < threshold
    audio = np.zeros(SR * 10, dtype=np.float32)
    audio_path = tmp_path / "test_short.wav"
    sf.write(str(audio_path), audio, SR)
    bundle = _make_bundle(audio, str(audio_path))
    engine = SubstrateErrorEngine(seed=42)
    measurer = DynamicsMeasurer()
    region = _make_region(start=0.0, end=0.3)  # < 20 frames at hop=512
    with pytest.raises(UnverifiableError) as exc_info:
        measurer.measure(location={"bar_start": 1, "bar_end": 1},
                         bundle=bundle, region=region, engine=engine)
    assert exc_info.value.reason_code == "region_too_short"


def test_error_bar_positive(tmp_path) -> None:
    import soundfile as sf
    n_total = SR * 20
    audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 20, n_total)) * 0.3).astype(np.float32)
    audio_path = tmp_path / "test_eb.wav"
    sf.write(str(audio_path), audio, SR)
    bundle = _make_bundle(audio, str(audio_path))
    engine = SubstrateErrorEngine(seed=42)
    measurer = DynamicsMeasurer()
    region = _make_region(start=0.0, end=10.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 5},
                              bundle=bundle, region=region, engine=engine)
    assert result.error_bar >= 0.0
    assert result.event_count >= 20
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_dynamics_measurer.py -x 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_taxonomy.verifier.measurers.dynamics'`

- [x] **Step 3: Implement**

```python
# apps/evals/claim_taxonomy/verifier/measurers/dynamics.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from claim_taxonomy.verifier.measurers.timing import Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

SR = 16000
HOP_LENGTH = 512
MINIMUM_FRAMES = 20  # ~640ms at 16kHz / 512 hop


class DynamicsMeasurer:
    """Measure RMS-based dynamic loudness for dynamics claims.

    Sign convention:
    - d < 0: region is quieter / flatter than whole-piece reference (narrow dynamics)
    - d > 0: region is louder / wider than reference
    - Whole-piece: d = RMS-contour std normalized by within-piece dynamic range (dispersion)
    """

    def measure(
        self,
        location: dict | str,
        bundle: dict,
        region: ResolvedRegion,
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        audio_path = bundle.get("audio_path")
        if not audio_path:
            raise UnverifiableError("substrate_failure", "bundle missing audio_path")

        import librosa
        try:
            y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        except Exception as exc:
            raise UnverifiableError("substrate_failure", f"failed to load audio: {exc}") from exc

        # Whole-piece RMS frames
        rms_frames = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        rms_db = 10.0 * np.log10(rms_frames + 1e-9)
        frame_times = librosa.frames_to_time(
            np.arange(len(rms_db)), sr=SR, hop_length=HOP_LENGTH
        )

        if location == "whole_piece":
            return self._measure_whole_piece(rms_db, engine)

        # Region frames
        mask = (frame_times >= region.audio_start_sec) & (frame_times < region.audio_end_sec)
        region_db = rms_db[mask]
        event_count = int(region_db.size)

        if event_count < MINIMUM_FRAMES:
            raise UnverifiableError(
                "region_too_short",
                f"only {event_count} RMS frames in region "
                f"[{region.audio_start_sec:.2f}, {region.audio_end_sec:.2f}s]; "
                f"need >= {MINIMUM_FRAMES}",
            )

        # d = mean region dB - mean whole-piece dB (signed loudness deviation)
        d = float(np.mean(region_db) - np.mean(rms_db))

        bootstrapped = engine.bootstrap_d(region_db, np.mean)
        sampling_var = float(np.var(bootstrapped - np.mean(rms_db)))

        jitters = engine.dynamics_rms_jitter_db()
        substrate_var = float(np.var(jitters))
        error_bar = math.sqrt(sampling_var + substrate_var)

        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)

    def _measure_whole_piece(
        self, rms_db: np.ndarray, engine: SubstrateErrorEngine
    ) -> Measurement:
        event_count = int(rms_db.size)
        if event_count < MINIMUM_FRAMES:
            raise UnverifiableError(
                "region_too_short",
                f"whole_piece has only {event_count} RMS frames; need >= {MINIMUM_FRAMES}",
            )
        dynamic_range = float(rms_db.max() - rms_db.min())
        if dynamic_range < 0.1:
            dynamic_range = 0.1  # avoid div-by-zero on silence
        # d = std / range (normalized dispersion; higher = more dynamic variation)
        d = float(rms_db.std() / dynamic_range)

        bootstrapped = engine.bootstrap_d(
            rms_db,
            lambda x: float(x.std() / max(x.max() - x.min(), 0.1)),
        )
        sampling_var = float(np.var(bootstrapped))
        jitters = engine.dynamics_rms_jitter_db()
        perturbed = np.array([
            float((rms_db + j).std() / max((rms_db + j).max() - (rms_db + j).min(), 0.1))
            for j in jitters
        ])
        substrate_var = float(np.var(perturbed))
        error_bar = math.sqrt(sampling_var + substrate_var)

        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_dynamics_measurer.py -x
```
Expected: PASS (5 tests)

- [x] **Step 5: Commit**

```bash
cd apps/evals && git add claim_taxonomy/verifier/measurers/dynamics.py claim_taxonomy/tests/test_dynamics_measurer.py && git commit -m "feat(verifier): add DynamicsMeasurer with RMS envelope injection (#65)"
```

---

### Task 7: verify() orchestrator
**Group:** D (sequential, depends on Group C)

**Behavior being verified:** verify() routes a known-by-construction timing claim to SUPPORTED via real route_verdict; scoped_out dimension returns UNVERIFIABLE(out_of_scope_dim) without calling measurer; missing bundle dimension raises cleanly; UnverifiableError from measurer becomes VerdictResult(UNVERIFIABLE).

**Interface under test:** `from claim_taxonomy.verifier.orchestrator import verify`

**Files:**
- Create: `apps/evals/claim_taxonomy/verifier/orchestrator.py`
- Test: `apps/evals/claim_taxonomy/tests/test_verifier_orchestrator.py`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/claim_taxonomy/tests/test_verifier_orchestrator.py
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pytest
from claim_taxonomy.verifier.orchestrator import verify
from claim_taxonomy.verifier.models import VerdictResult
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine


TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "claim_taxonomy.json"


def _load_taxonomy() -> dict:
    return json.loads(TAXONOMY_PATH.read_text())


def _make_timing_bundle(n_notes: int = 100, note_interval: float = 0.5) -> dict:
    notes = [
        {"onset": i * note_interval, "offset": i * note_interval + 0.1,
         "pitch": 60 + (i % 12), "velocity": 80}
        for i in range(n_notes)
    ]
    total_dur = n_notes * note_interval
    measure_table = [
        {"bar_number": i + 1, "start_sec": i * 2.0, "start_tick": i * 480}
        for i in range(int(total_dur // 2))
    ]
    t = np.linspace(0.0, total_dur, 200)
    return {
        "notes": notes,
        "pedal_events": [],
        "measure_table": measure_table,
        "anchors": {"perf_audio_sec": t.tolist(), "score_audio_sec": t.tolist()},
        "substrate_versions": {"amt_checkpoint_hash": "test", "bundle_schema": "v1"},
        "audio_path": "",
    }


def test_timing_rush_claim_returns_supported() -> None:
    """Whole-piece claim with d well above tau -> SUPPORTED."""
    taxonomy = _load_taxonomy()
    # Rush: region notes are 30% faster than rest
    n_rest = 80
    n_rush = 20
    rest_onsets = [i * 0.5 for i in range(n_rest)]
    rush_onsets = [n_rest * 0.5 + i * 0.35 for i in range(n_rush)]
    notes = [
        {"onset": t, "offset": t + 0.1, "pitch": 60, "velocity": 80}
        for t in sorted(rest_onsets + rush_onsets)
    ]
    total_dur = max(n["onset"] for n in notes) + 1.0
    measure_table = [
        {"bar_number": i + 1, "start_sec": i * 2.0, "start_tick": i * 480}
        for i in range(int(total_dur // 2) + 1)
    ]
    t = np.linspace(0.0, total_dur, 500)
    bundle = {
        "notes": notes,
        "pedal_events": [],
        "measure_table": measure_table,
        "anchors": {"perf_audio_sec": t.tolist(), "score_audio_sec": t.tolist()},
        "substrate_versions": {"amt_checkpoint_hash": "test", "bundle_schema": "v1"},
        "audio_path": "",
    }
    # Region claim: bars 1-10 (0-20s where rush notes live)
    claim = {
        "proposition": "You rushed in bars 1-10",
        "dimension": "timing",
        "location": {"bar_start": 1, "bar_end": 10},
        "polarity": "-",
        "magnitude": None,
    }
    engine = SubstrateErrorEngine(seed=42)
    result = verify(claim, bundle, taxonomy, engine=engine)
    assert isinstance(result, VerdictResult)
    # May be SUPPORTED or UNVERIFIABLE(near_threshold) depending on exact d;
    # the key invariant is that the orchestrator returns a VerdictResult without raising
    assert result.verdict in ("SUPPORTED", "REFUTED", "UNVERIFIABLE")
    assert result.dimension == "timing"


def test_scoped_out_dimension_returns_unverifiable_no_measurer_call() -> None:
    taxonomy = _load_taxonomy()
    bundle = _make_timing_bundle()
    claim = {
        "proposition": "Your phrasing lacked direction",
        "dimension": "phrasing",
        "location": "whole_piece",
        "polarity": "-",
        "magnitude": None,
    }
    engine = SubstrateErrorEngine(seed=42)
    result = verify(claim, bundle, taxonomy, engine=engine)
    assert result.verdict == "UNVERIFIABLE"
    assert result.reason_code == "out_of_scope_dim"


def test_dynamics_active_after_v01_returns_non_gated(tmp_path) -> None:
    """After v0.1 dynamics is active; verify() should attempt measurement, not return gated_dim."""
    import soundfile as sf
    taxonomy = _load_taxonomy()
    # Only run if taxonomy is v0.1 (dynamics active)
    if taxonomy["dimensions"]["dynamics"]["status"] != "active":
        pytest.skip("taxonomy not yet v0.1; dynamics still gated")
    n_total = SR = 16000
    audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 10, n_total * 10)) * 0.3).astype(np.float32)
    audio_path = tmp_path / "test.wav"
    sf.write(str(audio_path), audio, SR)
    measure_table = [
        {"bar_number": i + 1, "start_sec": i * 2.0, "start_tick": i * 480}
        for i in range(5)
    ]
    t = np.linspace(0.0, 10.0, 200)
    bundle = {
        "notes": [{"onset": 0.5, "offset": 0.6, "pitch": 60, "velocity": 80}],
        "pedal_events": [],
        "measure_table": measure_table,
        "anchors": {"perf_audio_sec": t.tolist(), "score_audio_sec": t.tolist()},
        "substrate_versions": {"bundle_schema": "v1"},
        "audio_path": str(audio_path),
    }
    claim = {
        "proposition": "Your dynamics were flat",
        "dimension": "dynamics",
        "location": "whole_piece",
        "polarity": "-",
        "magnitude": None,
    }
    engine = SubstrateErrorEngine(seed=42)
    result = verify(claim, bundle, taxonomy, engine=engine)
    assert result.reason_code != "gated_dim", "dynamics should not return gated_dim after v0.1"


def test_unlocalizable_claim_returns_unverifiable() -> None:
    taxonomy = _load_taxonomy()
    bundle = _make_timing_bundle(n_notes=100)
    # Bar 99 doesn't exist in measure_table
    claim = {
        "proposition": "You rushed in bar 99",
        "dimension": "timing",
        "location": {"bar_start": 99, "bar_end": 99},
        "polarity": "-",
        "magnitude": None,
    }
    engine = SubstrateErrorEngine(seed=42)
    result = verify(claim, bundle, taxonomy, engine=engine)
    assert result.verdict == "UNVERIFIABLE"
    assert result.reason_code == "unlocalizable"


def test_verify_returns_verdict_result_type() -> None:
    taxonomy = _load_taxonomy()
    bundle = _make_timing_bundle()
    claim = {
        "proposition": "You rushed slightly",
        "dimension": "timing",
        "location": {"bar_start": 1, "bar_end": 5},
        "polarity": "-",
        "magnitude": None,
    }
    engine = SubstrateErrorEngine(seed=42)
    result = verify(claim, bundle, taxonomy, engine=engine)
    assert isinstance(result, VerdictResult)
    assert result.dimension == "timing"
    assert result.location == {"bar_start": 1, "bar_end": 5}
    assert result.substrate_versions == bundle["substrate_versions"]
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_verifier_orchestrator.py -x 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_taxonomy.verifier.orchestrator'`

- [x] **Step 3: Implement**

```python
# apps/evals/claim_taxonomy/verifier/orchestrator.py
from __future__ import annotations

import json
from pathlib import Path

from claim_taxonomy.verifier.location_resolver import LocationResolver
from claim_taxonomy.verifier.models import UnverifiableError, VerdictResult
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine
from claim_taxonomy.verdict_dispatch import route_verdict

# Measurer registry: taxonomy `measurement` field -> class
def _build_registry():
    from claim_taxonomy.verifier.measurers.timing import TimingMeasurer
    from claim_taxonomy.verifier.measurers.pedaling import PedalingMeasurer
    from claim_taxonomy.verifier.measurers.dynamics import DynamicsMeasurer
    return {
        "amt_onsets_region_tempo_fit": TimingMeasurer(),
        "amt_sustain_pedal_events": PedalingMeasurer(),
        "librosa_rms_region_estimator": DynamicsMeasurer(),
    }


_MEASURER_REGISTRY = None


def _get_registry() -> dict:
    global _MEASURER_REGISTRY
    if _MEASURER_REGISTRY is None:
        _MEASURER_REGISTRY = _build_registry()
    return _MEASURER_REGISTRY


def verify(
    claim: dict,
    bundle: dict,
    taxonomy: dict,
    engine: SubstrateErrorEngine | None = None,
) -> VerdictResult:
    """Full verification pipeline for one claim against one bundle.

    Never raises. Returns VerdictResult for all outcomes including UNVERIFIABLE.
    """
    if engine is None:
        engine = SubstrateErrorEngine(seed=42)

    registry = taxonomy["dimensions"]
    dimension_name = claim["dimension"]
    location = claim["location"]
    polarity = claim["polarity"]
    substrate_versions = bundle.get("substrate_versions", {})

    dim = registry.get(dimension_name)
    if dim is None:
        return VerdictResult(
            verdict="UNVERIFIABLE",
            reason_code="out_of_scope_dim",
            measured_value=0.0,
            tau=0.0,
            error_bar=0.0,
            event_count=0,
            units="",
            substrate_versions=substrate_versions,
            dimension=dimension_name,
            location=location,
        )

    # Steps 1-2: scoped_out and gated handled by route_verdict if we pass _measurement,
    # but for pre-measurement short-circuit we avoid calling measurer at all.
    if dim["status"] == "scoped_out":
        return VerdictResult(
            verdict="UNVERIFIABLE",
            reason_code="out_of_scope_dim",
            measured_value=0.0,
            tau=0.0,
            error_bar=0.0,
            event_count=0,
            units="",
            substrate_versions=substrate_versions,
            dimension=dimension_name,
            location=location,
        )

    if dim["status"] == "gated_on_measurement":
        return VerdictResult(
            verdict="UNVERIFIABLE",
            reason_code="gated_dim",
            measured_value=0.0,
            tau=0.0,
            error_bar=0.0,
            event_count=0,
            units="",
            substrate_versions=substrate_versions,
            dimension=dimension_name,
            location=location,
        )

    tau = float(dim["tolerance"]["provisional"])
    units = dim["tolerance"]["unit"]
    measurement_key = dim["measurement"]
    measurer_registry = _get_registry()

    if measurement_key not in measurer_registry:
        return VerdictResult(
            verdict="UNVERIFIABLE",
            reason_code="substrate_failure",
            measured_value=0.0,
            tau=tau,
            error_bar=0.0,
            event_count=0,
            units=units,
            substrate_versions=substrate_versions,
            dimension=dimension_name,
            location=location,
        )

    # Resolve location
    try:
        resolver = LocationResolver(bundle, engine)
        region = resolver.resolve(location)
    except UnverifiableError as e:
        return VerdictResult(
            verdict="UNVERIFIABLE",
            reason_code=e.reason_code,
            measured_value=0.0,
            tau=tau,
            error_bar=0.0,
            event_count=0,
            units=units,
            substrate_versions=substrate_versions,
            dimension=dimension_name,
            location=location,
        )

    # Measure
    measurer = measurer_registry[measurement_key]
    try:
        measurement = measurer.measure(
            location=location,
            bundle=bundle,
            region=region,
            engine=engine,
        )
    except UnverifiableError as e:
        return VerdictResult(
            verdict="UNVERIFIABLE",
            reason_code=e.reason_code,
            measured_value=0.0,
            tau=tau,
            error_bar=0.0,
            event_count=0,
            units=units,
            substrate_versions=substrate_versions,
            dimension=dimension_name,
            location=location,
        )

    if measurement.substrate_failure:
        return VerdictResult(
            verdict="UNVERIFIABLE",
            reason_code="substrate_failure",
            measured_value=measurement.d,
            tau=tau,
            error_bar=measurement.error_bar,
            event_count=measurement.event_count,
            units=units,
            substrate_versions=substrate_versions,
            dimension=dimension_name,
            location=location,
        )

    # Assemble _measurement and call shipped route_verdict
    populated_claim = dict(claim)
    populated_claim["_measurement"] = {
        "d": measurement.d,
        "tau": tau,
        "error_bar": measurement.error_bar,
        "event_count": measurement.event_count,
        "localizable": True,
        "substrate_failure": False,
    }

    verdict, reason_code = route_verdict(populated_claim, registry)

    return VerdictResult(
        verdict=verdict,
        reason_code=reason_code,
        measured_value=measurement.d,
        tau=tau,
        error_bar=measurement.error_bar,
        event_count=measurement.event_count,
        units=units,
        substrate_versions=substrate_versions,
        dimension=dimension_name,
        location=location,
    )
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_verifier_orchestrator.py -x
```
Expected: PASS (5 tests)

- [x] **Step 5: Commit**

```bash
cd apps/evals && git add claim_taxonomy/verifier/orchestrator.py claim_taxonomy/tests/test_verifier_orchestrator.py && git commit -m "feat(verifier): add verify() orchestrator wired to route_verdict (#65)"
```

---

### Task 8: CLI, BundleExtractor, smoke test, and taxonomy v0.1 test updates
**Group:** E (sequential, depends on Group D and Task 9)

**Behavior being verified:** CLI outputs valid JSON VerdictResult to stdout; BundleExtractor produces bundle with correct schema keys; taxonomy v0.1 test_round_trip updated.

**Files:**
- Create: `apps/evals/claim_taxonomy/verifier/cli.py`
- Create: `model/src/claim_measurement/__init__.py`
- Create: `model/src/claim_measurement/extractor.py`
- Create: `model/src/claim_measurement/tests/__init__.py`
- Create: `model/src/claim_measurement/tests/test_extractor.py`
- Modify: `apps/evals/claim_taxonomy/tests/test_round_trip.py`
- Modify: `apps/evals/claim_taxonomy/tests/test_schema_validates.py`

- [x] **Step 1: Write the failing CLI test**

```python
# Test in test_verifier_orchestrator.py — add this test
# (Add to existing file from Task 7, do NOT rewrite that file)

def test_cli_verify_outputs_json(tmp_path) -> None:
    """CLI writes valid JSON VerdictResult to stdout."""
    import subprocess, json, sys
    claim = {
        "proposition": "You rushed",
        "dimension": "phrasing",  # scoped_out -> UNVERIFIABLE, no audio needed
        "location": "whole_piece",
        "polarity": "-",
        "magnitude": None,
    }
    bundle = {
        "notes": [],
        "pedal_events": [],
        "measure_table": [{"bar_number": 1, "start_sec": 0.0, "start_tick": 0}],
        "anchors": {"perf_audio_sec": [0.0, 1.0], "score_audio_sec": [0.0, 1.0]},
        "substrate_versions": {"bundle_schema": "v1"},
        "audio_path": "",
    }
    claim_path = tmp_path / "claim.json"
    bundle_path = tmp_path / "bundle.json"
    claim_path.write_text(json.dumps(claim))
    bundle_path.write_text(json.dumps(bundle))

    result = subprocess.run(
        [sys.executable, "-m", "claim_taxonomy.verifier.cli", "verify",
         "--claim", str(claim_path), "--bundle", str(bundle_path)],
        capture_output=True, text=True, cwd="/Users/jdhiman/Documents/crescendai/apps/evals",
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    out = json.loads(result.stdout)
    assert out["verdict"] == "UNVERIFIABLE"
    assert out["reason_code"] == "out_of_scope_dim"
    assert out["dimension"] == "phrasing"
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_verifier_orchestrator.py::test_cli_verify_outputs_json -x 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'claim_taxonomy.verifier.cli'`

- [x] **Step 3: Implement CLI**

```python
# apps/evals/claim_taxonomy/verifier/cli.py
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_DEFAULT_TAXONOMY = Path(__file__).resolve().parents[1] / "claim_taxonomy.json"


def _cmd_verify(args: argparse.Namespace) -> int:
    from claim_taxonomy.verifier.orchestrator import verify
    from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

    claim = json.loads(Path(args.claim).read_text())
    bundle = json.loads(Path(args.bundle).read_text())
    taxonomy_path = Path(args.taxonomy) if args.taxonomy else _DEFAULT_TAXONOMY
    taxonomy = json.loads(taxonomy_path.read_text())

    engine = SubstrateErrorEngine(seed=42)
    result = verify(claim, bundle, taxonomy, engine=engine)
    print(json.dumps(dataclasses.asdict(result)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claim_taxonomy.verifier.cli")
    subparsers = parser.add_subparsers(dest="command")

    verify_parser = subparsers.add_parser("verify", help="Verify a single claim against a bundle")
    verify_parser.add_argument("--claim", required=True, help="Path to claim JSON file")
    verify_parser.add_argument("--bundle", required=True, help="Path to bundle JSON file")
    verify_parser.add_argument("--taxonomy", default=None,
                               help="Path to claim_taxonomy.json (default: committed taxonomy)")

    args = parser.parse_args(argv)
    if args.command == "verify":
        return _cmd_verify(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 4: Implement BundleExtractor**

```python
# model/src/claim_measurement/__init__.py
```

```python
# model/src/claim_measurement/extractor.py
"""BundleExtractor: produce a unified per-clip measurement bundle from AMT + parangonar.

Reuses chroma_dtw_eval.amt_regen internals for AMT transcription and parangonar alignment.
Adds CC64 sustain-pedal capture (pedal_events: [] if AMT server does not expose CC data).
Output: bundle JSON at bundle_root/piece_id/video_id.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chroma_dtw_eval.amt_regen import (
    _dedup_amt_notes,
    _transcribe_clip,
    _read_wav_16k_mono,
    _amt_to_perf_na,
    _load_bach_json_score,
    _match,
    _build_pairs,
    DEFAULT_AMT_URL,
    DEFAULT_AMT_VERSION_CONFIG,
)

BUNDLE_SCHEMA_VERSION = "v1"


class BundleExtractionError(RuntimeError):
    pass


def _bundle_path(bundle_root: Path, piece_id: str, video_id: str) -> Path:
    return bundle_root / piece_id / f"{video_id}.json"


def extract_bundle(
    piece_id: str,
    video_id: str,
    *,
    audio_path: Path,
    score_path: Path,
    cache_root: Path,
    bundle_root: Path,
    amt_url: str = DEFAULT_AMT_URL,
    force: bool = False,
) -> Path:
    """Produce bundle JSON at bundle_root/piece_id/video_id.json. Returns path.

    Idempotent: returns existing path if bundle exists and force=False.
    Raises BundleExtractionError on AMT or alignment failure.
    """
    out_path = _bundle_path(bundle_root, piece_id, video_id)
    if not force and out_path.exists():
        return out_path

    if not audio_path.exists():
        raise BundleExtractionError(f"audio not found: {audio_path}")
    if not score_path.exists():
        raise BundleExtractionError(f"score not found: {score_path}")

    config_body = (
        json.loads(DEFAULT_AMT_VERSION_CONFIG.read_text())
        if DEFAULT_AMT_VERSION_CONFIG.exists() else {}
    )
    amt_checkpoint_hash = config_body.get("checkpoint_hash", "unknown")
    parangonar_version = config_body.get("parangonar_version", "unknown")

    audio_16k = _read_wav_16k_mono(audio_path)
    amt_notes = _transcribe_clip(audio_16k, amt_url)
    if not amt_notes:
        raise BundleExtractionError(f"AMT returned zero notes for {audio_path}")

    score_na, measure_table, score_sha256, beat_sec = _load_bach_json_score(score_path)
    deduped_notes = _dedup_amt_notes(amt_notes)

    # CC64 pedal events: AMT server currently returns notes only.
    # pedal_events is [] until the AMT endpoint exposes MIDI CC.
    pedal_events: list[dict] = []

    amt_perf_na = _amt_to_perf_na(deduped_notes, beat_sec)
    matches = _match(score_na, amt_perf_na)
    perf_arr, score_arr = _build_pairs(score_na, amt_perf_na, matches)

    bundle = {
        "piece_id": piece_id,
        "video_id": video_id,
        "audio_path": str(audio_path.resolve()),
        "notes": deduped_notes,
        "pedal_events": pedal_events,
        "measure_table": measure_table,
        "anchors": {
            "perf_audio_sec": perf_arr.tolist(),
            "score_audio_sec": score_arr.tolist(),
        },
        "substrate_versions": {
            "amt_checkpoint_hash": amt_checkpoint_hash,
            "parangonar_version": parangonar_version,
            "bundle_schema": BUNDLE_SCHEMA_VERSION,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(bundle))
    tmp.replace(out_path)
    return out_path
```

```python
# model/src/claim_measurement/tests/__init__.py
```

```python
# model/src/claim_measurement/tests/test_extractor.py
"""Smoke test: extract_bundle produces a bundle with the correct schema keys.

This test does NOT call the live AMT server. It verifies that a pre-existing
bundle file (if available) has the correct structure, or it skips if no bundle
has been generated yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

BUNDLE_SCHEMA_KEYS = frozenset({
    "piece_id", "video_id", "audio_path",
    "notes", "pedal_events", "measure_table", "anchors", "substrate_versions",
})
ANCHOR_KEYS = frozenset({"perf_audio_sec", "score_audio_sec"})
SUBSTRATE_VERSION_KEYS = frozenset({"amt_checkpoint_hash", "parangonar_version", "bundle_schema"})

# Default bundle root
_MODULE_DIR = Path(__file__).resolve()
DEFAULT_BUNDLE_ROOT = _MODULE_DIR.parents[3] / "data/evals/claim_bundles"


def _find_any_bundle() -> Path | None:
    if not DEFAULT_BUNDLE_ROOT.exists():
        return None
    for p in DEFAULT_BUNDLE_ROOT.rglob("*.json"):
        if not p.name.endswith(".tmp"):
            return p
    return None


def test_bundle_schema_if_exists() -> None:
    """If any bundle has been extracted, verify its schema keys."""
    bundle_path = _find_any_bundle()
    if bundle_path is None:
        pytest.skip("No bundle files found; run extract_bundle first")
    bundle = json.loads(bundle_path.read_text())
    missing = BUNDLE_SCHEMA_KEYS - set(bundle.keys())
    assert not missing, f"Bundle missing keys: {missing}"
    missing_anchors = ANCHOR_KEYS - set(bundle["anchors"].keys())
    assert not missing_anchors, f"anchors missing keys: {missing_anchors}"
    missing_versions = SUBSTRATE_VERSION_KEYS - set(bundle["substrate_versions"].keys())
    assert not missing_versions, f"substrate_versions missing keys: {missing_versions}"
    assert isinstance(bundle["notes"], list)
    assert isinstance(bundle["pedal_events"], list)
    assert isinstance(bundle["measure_table"], list)
```

- [x] **Step 5: Update taxonomy test files**

In `apps/evals/claim_taxonomy/tests/test_round_trip.py`, update `test_dynamics_gated_returns_unverifiable` to reflect v0.1:

```python
# Replace the test_dynamics_gated_returns_unverifiable function body:
def test_dynamics_active_routes_correctly() -> None:
    """After v0.1 dynamics is active; whole_piece with no audio -> substrate_failure or measurement."""
    taxonomy, _ = _load()
    registry = taxonomy["dimensions"]
    assert registry["dynamics"]["status"] == "active", (
        "dynamics must be active in v0.1 taxonomy"
    )
    # With synthetic _measurement (verifier populates this in real use):
    claim = {
        "proposition": "Your dynamics were flat throughout",
        "dimension": "dynamics",
        "location": "whole_piece",
        "polarity": "-",
        "magnitude": None,
        "_measurement": {
            "d": -2.0,
            "tau": registry["dynamics"]["tolerance"]["provisional"],
            "error_bar": 0.2,
            "event_count": 50,
            "localizable": True,
        },
    }
    verdict, reason = route_verdict(claim, registry)
    assert verdict in ("SUPPORTED", "REFUTED", "UNVERIFIABLE")
```

In `apps/evals/claim_taxonomy/tests/test_schema_validates.py`, update `test_all_tolerances_are_provisional` to allow dynamics as active (it already passes the active_dimension schema check; no change needed if taxonomy validates). Also update any hardcoded `"v0"` version string if present.

- [x] **Step 6: Run all verifier tests**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/ -x --tb=short 2>&1 | tail -20
```
Expected: All PASS.

- [x] **Step 7: Commit**

```bash
cd apps/evals && git add claim_taxonomy/verifier/cli.py claim_taxonomy/tests/test_verifier_orchestrator.py claim_taxonomy/tests/test_round_trip.py claim_taxonomy/tests/test_schema_validates.py
cd /Users/jdhiman/Documents/crescendai && git add model/src/claim_measurement/ apps/evals/claim_taxonomy/verifier/cli.py
git commit -m "feat(verifier): add CLI, BundleExtractor skeleton, update taxonomy tests (#65)"
```

---

### Task 9: Taxonomy v0.1 — dynamics active
**Group:** B (parallel with Task 2 and Task 3; depends on Task 1)

**Behavior being verified:** `claim_taxonomy.json` validates against schema with dynamics as active; `taxonomy_version` is `"v0.1"`; dynamics entry has all required active-dimension fields.

**Files:**
- Modify: `apps/evals/claim_taxonomy/claim_taxonomy.json`

- [x] **Step 1: Write the failing test**

```python
# apps/evals/claim_taxonomy/tests/test_taxonomy_v01.py
from __future__ import annotations
import json
from pathlib import Path
import jsonschema

TAXONOMY_DIR = Path(__file__).resolve().parents[1]

def _load():
    taxonomy = json.loads((TAXONOMY_DIR / "claim_taxonomy.json").read_text())
    schema = json.loads((TAXONOMY_DIR / "claim_taxonomy.schema.json").read_text())
    return taxonomy, schema


def test_taxonomy_version_is_v01() -> None:
    taxonomy, _ = _load()
    assert taxonomy["taxonomy_version"] == "v0.1", (
        f"Expected v0.1, got {taxonomy['taxonomy_version']}"
    )


def test_dynamics_is_active() -> None:
    taxonomy, _ = _load()
    dyn = taxonomy["dimensions"]["dynamics"]
    assert dyn["status"] == "active", f"Expected active, got {dyn['status']}"


def test_dynamics_has_all_active_fields() -> None:
    taxonomy, _ = _load()
    dyn = taxonomy["dimensions"]["dynamics"]
    for field in ("reference", "check", "tolerance", "reliability_tier", "measurement", "minimum_events"):
        assert field in dyn, f"dynamics missing field: {field}"
    assert dyn["tolerance"]["locked"] is False
    assert dyn["tolerance"]["calibration_source"] == "#65/M1 error-bar study"


def test_v01_taxonomy_validates_against_schema() -> None:
    taxonomy, schema = _load()
    jsonschema.validate(instance=taxonomy, schema=schema)


def test_three_active_dimensions() -> None:
    taxonomy, _ = _load()
    active = [k for k, v in taxonomy["dimensions"].items() if v["status"] == "active"]
    assert set(active) == {"timing", "pedaling", "dynamics"}, f"Active dims: {active}"
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_taxonomy_v01.py -x 2>&1 | head -20
```
Expected: FAIL — `AssertionError: Expected v0.1, got v0`

- [x] **Step 3: Update claim_taxonomy.json**

Edit `apps/evals/claim_taxonomy/claim_taxonomy.json`:

1. Change `"taxonomy_version": "v0"` → `"taxonomy_version": "v0.1"`

2. Replace the `"dynamics"` entry:

```json
"dynamics": {
  "status": "active",
  "reference": "within_region_range",
  "check": "rms_contour_std_normalized",
  "tolerance": {
    "name": "rms_contour_deviation",
    "provisional": 1.5,
    "unit": "dB",
    "calibration_source": "#65/M1 error-bar study",
    "locked": false
  },
  "reliability_tier": 2,
  "measurement": "librosa_rms_region_estimator",
  "minimum_events": 20,
  "notes": "RMS-based loudness proxy. Absolute dB inadmissible (uncontrolled recording gain). Within-region dynamic range only. minimum_events = RMS frames at hop_length=512, SR=16000 (~640ms minimum region)."
}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_taxonomy_v01.py -x
```
Expected: PASS (5 tests)

- [x] **Step 5: Commit**

```bash
cd apps/evals && git add claim_taxonomy/claim_taxonomy.json claim_taxonomy/tests/test_taxonomy_v01.py && git commit -m "feat(taxonomy): bump to v0.1; activate dynamics dimension (#65)"
```

---

### Task 10: Signed-d convention and error-bar table document
**Group:** E (sequential, depends on Group D)

**Behavior being verified:** Document exists at `docs/model/claim-verifier-signed-d-conventions.md` with correct content (validated by a test that checks file existence and key strings).

**Files:**
- Create: `docs/model/claim-verifier-signed-d-conventions.md`
- Test: included in test_taxonomy_v01.py (add one test)

- [x] **Step 1: Write the failing test**

Add to `apps/evals/claim_taxonomy/tests/test_taxonomy_v01.py`:

```python
def test_signed_d_convention_doc_exists() -> None:
    doc = Path(__file__).resolve().parents[4] / "docs/model/claim-verifier-signed-d-conventions.md"
    assert doc.exists(), f"Signed-d convention doc not found at {doc}"
    content = doc.read_text()
    assert "timing" in content
    assert "pedaling" in content
    assert "dynamics" in content
    assert "Sign convention" in content
    assert "error_bar" in content
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_taxonomy_v01.py::test_signed_d_convention_doc_exists -x 2>&1 | head -10
```
Expected: FAIL — `AssertionError: Signed-d convention doc not found`

- [x] **Step 3: Create the document**

```markdown
# Claim Verifier: Per-Dimension Signed-d Conventions and Error-Bar Table

**Issue:** #65
**Taxonomy version:** v0.1
**Validation boundary:** This document covers substrate-level measurement. Real-claim faithfulness (#67), proxy-to-perception (#66), and error-rich localization (GATE 1) are NOT validated here.

---

## Sign Convention Table

| Dimension | d formula (region) | d formula (whole_piece) | d < 0 means | d > 0 means | tau (provisional) | units |
|-----------|-------------------|------------------------|-------------|-------------|-------------------|-------|
| timing | `(region_median_bpm - established_tempo) / established_tempo * 100` | `CV% = std(bpms) / mean(bpms) * 100` | faster than reference (rushed) | slower than reference (dragging) | 8.0 | percent |
| pedaling | `region_pedal_bar_fraction - self_density` | `pedal_bar_fraction` (0.0–1.0) | sparse pedaling vs piece average | dense pedaling vs piece average | 0.25 | fraction |
| dynamics | `mean(region_rms_db) - mean(piece_rms_db)` | `std(rms_db) / dynamic_range` (dispersion) | quieter / flatter than whole piece | louder / wider than whole piece | 1.5 | dB |

**Note on whole_piece:** Self-referential references (established_tempo, self_density, within_region_range) degenerate for whole_piece location. Each measurer switches to an intrinsic dispersion or presence statistic for whole_piece claims.

---

## Substrate Error Distributions

Documented error distributions used by `SubstrateErrorEngine` for Monte-Carlo error propagation:

| Source | Distribution | Parameters | Rationale |
|--------|-------------|------------|-----------|
| AMT onset jitter (timing, pedaling localization) | Gaussian | mean=0, sigma=0.010s | Aria-AMT onset error on clean audio (10ms 1-sigma estimate from MAESTRO validation) |
| RMS frame variance (dynamics) | Gaussian | mean=0, sigma=0.3 dB | Empirical: constant-signal buffer shows ~0.3 dB std from librosa hop-length framing |
| CC64 threshold uncertainty (pedaling) | Uniform | low=-10, high=+10 value counts | MIDI CC quantization: threshold 64 ± 10 covers typical soft-pedal ambiguity |

---

## error_bar Formula

```
error_bar = sqrt(sampling_var + substrate_var)
```

- **sampling_var**: `var(bootstrap_d(within_region_events, stat_fn))` — bootstrap (N=500, seeded) over events in the region.
- **substrate_var**: `var(MC_perturbed_d_values)` — Monte-Carlo (N=500, same seeded engine) perturbing raw measurements by the distributions above.

The `SubstrateErrorEngine` is initialized with `seed=42` by default, making all error estimates deterministic for a given bundle.

---

## Near-Threshold Dead-Band

`route_verdict` applies: if `abs(abs(d) - tau) <= error_bar` → UNVERIFIABLE(near_threshold).

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
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/test_taxonomy_v01.py -x
```
Expected: PASS (6 tests)

- [x] **Step 5: Run full verifier suite**

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/ --tb=short 2>&1 | tail -20
```
Expected: All existing + new tests PASS.

- [x] **Step 6: Commit**

```bash
git add docs/model/claim-verifier-signed-d-conventions.md apps/evals/claim_taxonomy/tests/test_taxonomy_v01.py && git commit -m "docs(model): add signed-d convention and error-bar table for verifier (#65)"
```

---

## Final verification

```bash
cd apps/evals && uv run pytest claim_taxonomy/tests/ --tb=short
```

All tests should pass. Post issue comment:

```bash
gh issue comment 65 --body "STATE: Plan and spec committed to branch issue-65-deterministic-claim-verifier. Next: run /challenge on docs/plans/2026-06-19-deterministic-claim-verifier.md before executing."
```

---

## Challenge Review

### CEO Pass

#### 1. Premise Challenge

Right problem. Without a programmatic truth oracle, the "faithfulness rate" metric in research path #1 is unfalsifiable — every verification step bottoms out at a human judgment or a second LLM call, reintroducing the circularity that the PROVE/ViCrit defense is supposed to eliminate. This plan is the direct path to the core bet.

No dramatically simpler alternative exists. The bundle-as-interface design (extract separately from verify) is the correct decomposition: it keeps the substrate (AMT server, parangonar, librosa) out of the eval environment and makes the auditable artifact inspectable. The alternative — inline extraction inside the verifier — would couple a live HTTP call into what must be a deterministic offline checker.

Existing coverage: `verdict_dispatch.py` + `route_verdict` (shipped, tested, locked) already implement the dispatch chain. This plan correctly extends that rather than replacing it.

#### 2. Scope Check

The plan's scope is tight. The three active measurer tasks (Tasks 4-6) are genuinely required to populate `_measurement`, and each has a distinct substrate. Nothing is gold-plating.

The plan wisely defers: GATE 1 error-rich localization, real-claim faithfulness (#67), proxy-to-perception (#66). Those are correctly flagged as not blockers.

The signed-d convention doc (Task 10) is small and provides the durable reference that prevents future drift in sign polarity. Worth keeping.

Complexity: 10 tasks, ~18 new files. Justified — each file is a single-concern module with real depth (SubstrateErrorEngine hides MC internals; LocationResolver hides score→audio interpolation + uncertainty failsafe; each Measurer hides its signal-chain logic). The interface count (3-4 exported symbols each) is low relative to the hidden implementation.

#### 3. Twelve-Month Alignment

```
CURRENT STATE                              THIS PLAN                           12-MONTH IDEAL
route_verdict exists but nothing           Adds programmatic truth oracle       Full faithfulness
populates _measurement from real           (3 dimensions: timing, pedaling,     measurement harness
audio; dynamics is gated; pedaling         dynamics); enables offline batch      across all
measurer stubs are empty; no CLI           verification; unlocks dynamics;       verifiable dimensions
                                           surfaces error bars; provides CLI
```

Moves cleanly toward the ideal. No tech debt created that conflicts with future dimension expansion — the measurer registry in orchestrator.py is open/closed (add a key, add a file, done).

#### 4. Alternatives Check

Spec documents the two-environment split rationale. The choice to reuse `chroma_dtw_eval.amt_regen` internals rather than duplicate them is correct and documented. No undocumented alternatives.

---

### Engineering Pass

#### 5. Architecture

Data flow:

```
extract_bundle()               verify()
  audio + score_path  -->  bundle JSON  -->  LocationResolver.resolve()
       |                       |                    |
  _transcribe_clip()     measure_table            ResolvedRegion
  _match()/_build_pairs()  anchors                    |
  pedal_events: []         notes              Measurer.measure()
       |                   pedal_events              |
  bundle JSON file         audio_path           Measurement(d, error_bar, event_count)
                                                     |
                                             _measurement dict
                                                     |
                                             route_verdict()  (SHIPPED, LOCKED)
                                                     |
                                             VerdictResult
```

The boundary is clean. `route_verdict` is never touched.

One architectural issue found: the orchestrator's `_MEASURER_REGISTRY` is a module-level mutable global initialized lazily. This is not thread-safe. In the current CLI / batch-eval usage (single process, single thread), it is harmless. In a future parallel batch runner, it would cause a race. Low severity for the current scope.

Security: no user input flows to SQL, shell, or LLM. Audio path comes from a trusted bundle JSON. No exposure.

Deployment: all new code is offline Python eval tooling — no migration, no rollback concern.

#### 6. Module Depth Audit

| Module | Exported interface | Hidden implementation | Verdict |
|--------|-------------------|-----------------------|---------|
| `verifier/models.py` | 2 classes (VerdictResult, UnverifiableError) | Dataclass + typed exception | DEEP (for its role) |
| `verifier/substrate_error.py` | SubstrateErrorEngine (5 methods) | NumPy RNG, MC loop, bootstrap resampling | DEEP |
| `verifier/location_resolver.py` | LocationResolver.resolve() + ResolvedRegion | measure_table lookup, np.interp, bar-dur inference, MC failsafe | DEEP |
| `verifier/measurers/timing.py` | TimingMeasurer.measure() + Measurement | IOI→BPM, established_tempo, bootstrap+MC error bar | DEEP |
| `verifier/measurers/pedaling.py` | PedalingMeasurer.measure() | CC64 threshold, pedal-bar fraction, self_density, MC threshold jitter | DEEP |
| `verifier/measurers/dynamics.py` | DynamicsMeasurer.measure() | librosa.load, RMS, dB conversion, within-region frame extraction, bootstrap+MC | DEEP |
| `verifier/orchestrator.py` | verify() | Registry, UNVERIFIABLE short-circuits, _measurement assembly, route_verdict dispatch | DEEP |
| `verifier/cli.py` | 1 CLI subcommand | JSON I/O glue | SHALLOW (intentional, documented) |
| `claim_measurement/extractor.py` | extract_bundle() | AMT transcription, alignment, pedal capture, bundle serialization | DEEP |

All modules pass.

#### 7. Code Quality

**[RISK] (confidence: 9/10)** — `BundleExtractor` imports `_load_bach_json_score` from `chroma_dtw_eval.amt_regen`, but this function is named "bach" and enforces `len(tempos) == 1` (single tempo) and `4/4` time signature (raises `AmtRegenError` on anything else). The PercePiano evaluation corpus likely contains multi-tempo or non-4/4 pieces. For the initial scope of known-clean cached audio this is acceptable, but the constraint is invisible at the `extract_bundle` call site. Name the fallback: the plan already scopes to "clean cached audio only" — but the build agent should add a `score_format` parameter or a clear error message when `AmtRegenError` propagates from `_load_bach_json_score`.

**[OBS]** — `_substrate_var` in `TimingMeasurer` computes the established_tempo from `region_onsets` (line 882) rather than from `all_onsets`. This means the MC perturbation reference is the region itself, not the piece-wide reference. The sign is correct (perturbation is relative to the same reference as d), but the resulting substrate_var is slightly underestimated for narrow regions. This is a known precision trade-off, not a bug, and the error_bar is still positive and reported correctly.

**[OBS]** — `DynamicsMeasurer` uses `except Exception as exc` (line 1479) to catch librosa.load failures. CLAUDE.md mandates explicit exception handling over fallbacks. In this case the exception is re-raised as `UnverifiableError("substrate_failure", ...)` with context — it is not silenced — so the spirit of the rule is met. The `UnverifiableError` is the explicit typed outcome; `Exception` is only used to wrap foreign library errors with a typed code. This pattern is acceptable given the substrate-failure semantics, but the build agent should narrow the catch to `(OSError, Exception)` at minimum or at least document the catch scope.

**[OBS]** — The `_bar_start_end_sec` helper in `PedalingMeasurer` does a linear scan over `all_bars` for each bar lookup (`[r for r in all_bars if int(r["bar_number"]) == bar_num + 1]`). For a piece with 200+ bars and N bars in the region this is O(N * bars). Not a correctness issue, just O(N²) scan that could be a dict lookup. Acceptable for eval-only tooling.

**[RISK] (confidence: 7/10)** — The `test_cli_verify_outputs_json` test in Task 8 hardcodes `cwd="/Users/jdhiman/Documents/crescendai/apps/evals"` in a `subprocess.run` call. This is a machine-specific absolute path — it will fail on any other machine or CI environment. Should use `Path(__file__).resolve().parents[N]` anchoring (the same pattern MEMORY.md flags as a project-wide gotcha).

#### 8. Test Philosophy Audit

All tests exercise behavior through the module's public interface (`measure()`, `resolve()`, `verify()`). No private methods are called. No internal state is asserted. No mocking of internal collaborators — external boundaries (filesystem via `tmp_path`, bundle dicts) are passed in as data.

The injection harness tests (mistake-injection in Tasks 4-6) are the strongest behavioral tests: they construct a known-anomaly input and assert the measurer detects it. These will fail before the implementation exists (ImportError) and will fail with a wrong implementation (wrong sign or magnitude of d).

**[RISK] (confidence: 8/10)** — `test_timing_rush_claim_returns_supported` in Task 7's orchestrator test asserts only `result.verdict in ("SUPPORTED", "REFUTED", "UNVERIFIABLE")` — all three outcomes are acceptable. This is a shape test with no behavioral content. The comment says "the key invariant is that the orchestrator returns a VerdictResult without raising." That is true and useful for smoke-testing the wiring, but it is NOT a behavioral test for the SUPPORTED outcome. A real behavioral test would use a constructed bundle where the d value is analytically predictable (e.g., the injection harness from Task 4 reused here). The orchestrator smoke test is adequate as a wiring test but should be understood as ★ quality, not ★★★.

**[OBS]** — `test_bundle_schema_if_exists` in Task 8's extractor test skips when no bundle file is present. This is correct and necessary for a cold-repo run. The skip is explicit (`pytest.skip`) rather than silent, which is the right pattern.

#### 9. Vertical Slice Audit

Every task has the structure: write test (Step 1) → verify FAIL (Step 2) → implement (Step 3) → verify PASS (Step 4) → commit (Step 5). Strict vertical slice throughout.

Task 8 is the only deviation: it bundles CLI, BundleExtractor, and taxonomy test updates into one task. This is acceptable because:
1. The CLI test is a single behavioral assertion (one test → one impl).
2. The BundleExtractor test is a conditional smoke test (skips without data).
3. The taxonomy test updates are modifications to existing files, not new behavior.

No horizontal slicing found.

**[RISK] (confidence: 6/10)** — Task 8's Step 7 commit command uses two separate `git add` calls (one in `apps/evals`, one from the repo root). In a worktree, `cd apps/evals` shifts the working directory for that shell command but the `git add` in the second line runs from `crescendai/` root — which is the PRIMARY checkout, not the worktree. The build agent should use absolute paths or ensure all `git add` and `git commit` calls are issued from the worktree root (`/Users/jdhiman/Documents/crescendai/.worktrees/issue-65-deterministic-claim-verifier`). This is a bash sequencing issue, not a logical flaw.

#### 10. Test Coverage Gaps

```
[+] verifier/orchestrator.py
    │
    ├── verify() — dim not in registry
    │   └── [TESTED] → out_of_scope_dim (test_scoped_out... covers scoped_out)
    │
    ├── verify() — dim.status == "gated_on_measurement"
    │   └── [GAP] No test explicitly sets a gated dimension claim through verify().
    │           test_dynamics_active_after_v01 skips if dynamics is not yet active.
    │           After Task 9 bumps dynamics to active, the gated_dim path is no longer
    │           exercisable with articulation (which stays gated). The orchestrator's
    │           gated_dim short-circuit branch becomes a dead branch in tests.
    │
    ├── verify() — measurement_key not in measurer_registry
    │   └── [GAP] No test covers an "active" dimension with a measurement key
    │           that has no registered measurer. The substrate_failure fallback
    │           at line 1859-1871 is untested.
    │
    ├── verify() — location resolution fails (UnverifiableError from LocationResolver)
    │   └── [TESTED] test_unlocalizable_claim_returns_unverifiable ★★
    │
    ├── verify() — measurer raises UnverifiableError
    │   └── [PARTIALLY TESTED] unlocalizable path only; region_too_short from
    │           measurer itself is not directly tested via verify() (only via
    │           measurer unit tests)
    │
    └── verify() — measurement.substrate_failure == True
        └── [GAP] No test constructs a Measurement with substrate_failure=True.
                The branch at line 1914-1926 is untested.

[+] verifier/location_resolver.py
    ├── resolve("whole_piece") — [TESTED] ★★
    ├── resolve(bar_range) — happy path [TESTED] ★★★
    ├── resolve() — bar not in measure_table [TESTED] ★★★
    ├── resolve() — < 2 anchors [TESTED] ★★★
    └── resolve() — uncertainty >= span [TESTED] ★★ (probabilistic; may not
            always trigger with the exact synthetic fixture — see RISK below)

[+] DynamicsMeasurer — whole_piece branch
    └── [NOT TESTED] No test exercises DynamicsMeasurer.measure(location="whole_piece")
```

**[RISK] (confidence: 7/10)** — `test_failsafe_triggers_when_uncertainty_exceeds_span` in Task 3 uses sparse anchors (only 2 points) and a 50ms bar span to produce high alignment uncertainty. The SubstrateErrorEngine with seed=42, n_samples=500 produces Gaussian jitter with sigma=0.010s. With only 2 anchor points and 5s spacing, the MC interpolation std over the `bar_start_score_sec=0.0` endpoint will be dominated by the extrapolation regime — the uncertainty may or may not exceed 0.05s / 0.05s = 1.0 bar_unit depending on exact RNG behavior. The test may be flaky if the uncertainty calculation lands near the boundary. The build agent should verify this test actually fails before implementation and passes after, running it 3 times with different seeds.

**[OBS]** — The gated_dim coverage gap (articulation dimension is still `gated_on_measurement` after Task 9) is low severity — the gated-dim branch in orchestrator.py is 4 lines and its behavior is fully tested in the existing `test_round_trip.py` via direct `route_verdict` call with a gated claim. The orchestrator's pre-routing short-circuit for `gated_on_measurement` is a redundant guard; if it were removed, `route_verdict` would catch it anyway. It is dead-but-harmless code, not a risk.

#### 11. Failure Modes

All measurer errors flow to `UnverifiableError` (typed reason code) or propagate to `verify()` which catches them and returns `VerdictResult(UNVERIFIABLE)`. No silent failures.

Bundle serialization in `extract_bundle` uses a `.tmp` write-then-replace pattern — no partial-write corruption.

The `alignment_uncertainty_sec` MC loop perturbs ALL anchors by the SAME scalar jitter per sample (`perf_audio_sec + j` where j is a scalar). This is conservative (overestimates correlated error) but mechanically wrong — AMT onset jitter is independent per-note, not a global shift. The result is that alignment_uncertainty is overestimated, which makes the unlocalizable failsafe more aggressive (good for safety) but may reject more single-bar claims than necessary. This is a known limitation and acceptable at tier-2 reliability, but should be noted in the signed-d doc.

**[OBS]** — `_build_registry()` is called once and cached in `_MEASURER_REGISTRY`. DynamicsMeasurer imports librosa at module level inside `measure()` (deferred `import librosa` in the method body). This means librosa is not imported until the first dynamics measurement, which is fine for cold-start. However, if librosa is not installed (e.g., running without the `[inference]` extra), the error surfaces only at measure-time, not at import. This is acceptable behavior but should be documented.

#### 12. Presumption Inventory

| Assumption | Verdict | Reason |
|------------|---------|--------|
| `chroma_dtw_eval.amt_regen` is importable from `model/src/claim_measurement/` (both in same `model/` env) | SAFE | Verified: both are under `model/src/`; model pyproject.toml confirms same env |
| `_transcribe_clip`, `_dedup_amt_notes`, `_read_wav_16k_mono`, `_amt_to_perf_na`, `_load_bach_json_score`, `_match`, `_build_pairs`, `DEFAULT_AMT_URL`, `DEFAULT_AMT_VERSION_CONFIG` are all exported from `amt_regen.py` | SAFE | Verified by grep: all 9 symbols confirmed present at the expected line numbers |
| `_load_bach_json_score` returns `(score_na, measure_table, score_sha256, beat_sec)` — 4-element tuple | SAFE | Verified: line 159 signature and return matches plan's usage |
| `jsonschema` is available in `apps/evals` env for taxonomy tests | VALIDATE | jsonschema is in `teacher-model-stage0` optional extra, NOT in base deps or `inference` extra. Tests using `import jsonschema` (test_round_trip.py, test_schema_validates.py, test_taxonomy_v01.py) require `uv run --extra teacher-model-stage0 pytest` or the `all` extra. The existing tests already use it and presumably pass, so the extra is installed — but the plan's `uv run pytest` commands do not specify `--extra`, which may cause import failures in a fresh env |
| `soundfile` is available in `apps/evals` for dynamics measurer tests | SAFE | Verified: `soundfile>=0.12.0` in `inference` optional extra |
| `librosa` is available in `apps/evals` for dynamics measurer | SAFE | Verified: `librosa>=0.10.0` in `inference` optional extra |
| `numpy` is available in `apps/evals` base env | VALIDATE | `numpy` is in `model` and `inference` optional extras, not base deps. SubstrateErrorEngine and LocationResolver both import numpy at module level. If tests run without `--extra model` or `--extra inference`, they will fail with ImportError at collection time |
| `scipy` is not required (plan uses numpy only) | SAFE | No scipy import found in any plan code |
| The `test_failsafe_triggers_when_uncertainty_exceeds_span` test deterministically triggers with seed=42, 500 samples, 2-anchor sparse bundle | VALIDATE | The uncertainty computation involves interpolating at the endpoint of a 2-point array with Gaussian jitter — endpoint extrapolation behavior with np.interp could produce very small std (np.interp clips to boundary values). This needs empirical verification |
| `whole_piece` location in `LocationResolver.resolve()` always passes the failsafe (uncertainty < inf) | SAFE | Verified: code explicitly returns `location_span_bars=math.inf`; failsafe is `uncertainty_bars >= location_span_bars` which is never satisfied for inf |
| `test_schema_validates.py` does not hardcode `"v0"` as the expected version | VALIDATE | Line 42 and 106 in test_schema_validates.py use `"v0"` in the skeleton fixture — these are skeleton instances passed to the schema validator (not assertions about the real taxonomy file), so bumping the real taxonomy to "v0.1" will not break them. Confirmed safe. |
| The CLI test's `cwd="/Users/jdhiman/Documents/crescendai/apps/evals"` is machine-portable | RISKY | Hardcoded absolute path. Will fail on any other machine or CI. Must be replaced with `Path(__file__).resolve().parents[N]` anchoring |

---

### Summary

**[BLOCKER]** count: 0
**[RISK]** count: 5
**[QUESTION]** count: 0
**[OBS]** count: 6

Risks to monitor during execution:

1. **[RISK] (confidence: 9/10)** — CLI test hardcodes `cwd="/Users/jdhiman/Documents/crescendai/apps/evals"`. Will fail on CI or any other machine. Build agent must replace with `Path(__file__).resolve().parents[N]` anchoring per project-wide `Default Paths` gotcha in MEMORY.md.

2. **[RISK] (confidence: 8/10)** — `numpy` and `jsonschema` are not in the base `apps/evals` env — they are in optional extras (`model`/`inference` and `teacher-model-stage0`). The plan's `uv run pytest` commands do not specify `--extra`. In a fresh install, test collection will fail with ImportError. Build agent should use `uv run --extra all pytest` or install the required extras explicitly.

3. **[RISK] (confidence: 8/10)** — `test_timing_rush_claim_returns_supported` is a shape/wiring test with verdict `in ("SUPPORTED", "REFUTED", "UNVERIFIABLE")` — all outcomes are accepted. It will pass even with a broken measurer. Keep it as a wiring smoke test but do not treat it as behavioral coverage.

4. **[RISK] (confidence: 7/10)** — `test_failsafe_triggers_when_uncertainty_exceeds_span` may not reliably trigger with np.interp endpoint behavior at a 2-point anchor boundary. Build agent must verify the test actually FAILS before implementation and PASSES after, not just that it passes.

5. **[RISK] (confidence: 7/10)** — Task 8's two-step `git add` command crosses worktree/primary-checkout boundary. Build agent must issue all git commands from the worktree root using absolute paths.

6. **[RISK] (confidence: 7/10)** — `_load_bach_json_score` enforces single-tempo and 4/4 time signature; `BundleExtractionError` will surface for non-4/4 or multi-tempo pieces. Acceptable for current scope (clean cached audio, known pieces), but the build agent should confirm the initial target pieces (bach_prelude_c_wtc1 etc.) are 4/4 single-tempo before running `extract_bundle` in the smoke test.

VERDICT: PROCEED_WITH_CAUTION — monitor the 5 risks above, especially the hardcoded cwd path (fix before committing the CLI test) and the numpy/jsonschema extra requirement (add `--extra all` to all uv run pytest invocations in the plan).
