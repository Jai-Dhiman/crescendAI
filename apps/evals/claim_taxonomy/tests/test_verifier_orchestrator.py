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
SR = 16000


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


def _make_rush_bundle() -> dict:
    """Known-by-construction: bars 1-10 are markedly faster than the rest.

    Measure table uses 1s/bar so bars 1-10 span 0-10s exactly.
    Rush notes (0.35s IOI, 25 notes) occupy 0-8.4s — entirely within bars 1-10.
    Rest notes (0.5s IOI, 80 notes) start at 10s — entirely outside bars 1-10.
    Established tempo (median across all notes) = 120 BPM (0.5s IOI dominates).
    Region BPM = 171 BPM -> d ~ -43% << tau=8% -> SUPPORTED.
    """
    n_rush = 25
    n_rest = 80
    rush_onsets = [i * 0.35 for i in range(n_rush)]        # 0..8.4s, all in bars 1-10
    rest_onsets = [10.0 + i * 0.5 for i in range(n_rest)]  # 10..49.5s, bars 11+
    notes = [
        {"onset": t, "offset": t + 0.1, "pitch": 60, "velocity": 80}
        for t in sorted(rush_onsets + rest_onsets)
    ]
    total_dur = max(n["onset"] for n in notes) + 1.0
    # 1s/bar so bars 1-10 = 0-10s (rush region), bars 11+ = rest region
    measure_table = [
        {"bar_number": i + 1, "start_sec": float(i), "start_tick": i * 480}
        for i in range(int(total_dur) + 2)
    ]
    t = np.linspace(0.0, total_dur, 500)
    return {
        "notes": notes,
        "pedal_events": [],
        "measure_table": measure_table,
        "anchors": {"perf_audio_sec": t.tolist(), "score_audio_sec": t.tolist()},
        "substrate_versions": {"amt_checkpoint_hash": "test", "bundle_schema": "v1"},
        "audio_path": "",
    }


def test_low_coverage_clip_bar_claim_is_low_coverage() -> None:
    """#100 coverage gate: a clip whose anchors cover only a fraction of the score
    span abstains on bar/region claims with reason_code 'low_coverage' (driven by the
    taxonomy's coverage_gate.threshold through the orchestrator)."""
    taxonomy = _load_taxonomy()
    bundle = _make_rush_bundle()
    # Shrink anchor coverage well below threshold while keeping the measure_table span.
    span = max(m["start_sec"] for m in bundle["measure_table"])
    low = np.linspace(0.0, 0.2 * span, 100)  # ~0.2 coverage
    bundle["anchors"] = {"perf_audio_sec": low.tolist(), "score_audio_sec": low.tolist()}
    claim = {
        "proposition": "You rushed in bars 1-10",
        "dimension": "timing",
        "location": {"bar_start": 1, "bar_end": 10},
        "polarity": "-",
        "magnitude": None,
    }
    result = verify(claim, bundle, taxonomy, engine=SubstrateErrorEngine(seed=42))
    assert result.verdict == "UNVERIFIABLE"
    assert result.reason_code == "low_coverage"


def test_whole_piece_claim_exempt_from_coverage_gate() -> None:
    """whole_piece never trips the coverage gate, even on a low-coverage clip."""
    taxonomy = _load_taxonomy()
    bundle = _make_rush_bundle()
    span = max(m["start_sec"] for m in bundle["measure_table"])
    low = np.linspace(0.0, 0.2 * span, 100)
    bundle["anchors"] = {"perf_audio_sec": low.tolist(), "score_audio_sec": low.tolist()}
    claim = {
        "proposition": "Your timing was uneven overall",
        "dimension": "timing",
        "location": "whole_piece",
        "polarity": "-",
        "magnitude": None,
    }
    result = verify(claim, bundle, taxonomy, engine=SubstrateErrorEngine(seed=42))
    assert result.reason_code != "low_coverage"


def test_timing_rush_claim_returns_verdict_result_wiring() -> None:
    """WIRING SMOKE TEST: orchestrator returns a VerdictResult without raising. Accepts all verdicts."""
    taxonomy = _load_taxonomy()
    bundle = _make_rush_bundle()
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
    assert result.verdict in ("SUPPORTED", "REFUTED", "UNVERIFIABLE")
    assert result.dimension == "timing"


def test_timing_rush_claim_strong_behavioral() -> None:
    """BEHAVIORAL: a strongly rushed region (polarity '-') yields measured d<0 with |d|>tau.

    This asserts the SPECIFIC measured value and verdict on a known-by-construction bundle,
    not just that *some* VerdictResult comes back. The region is ~43% faster (0.35s vs 0.5s
    IOI) so |d| comfortably exceeds tau and the error bar.
    """
    taxonomy = _load_taxonomy()
    bundle = _make_rush_bundle()
    claim = {
        "proposition": "You rushed in bars 1-10",
        "dimension": "timing",
        "location": {"bar_start": 1, "bar_end": 10},
        "polarity": "-",
        "magnitude": None,
    }
    engine = SubstrateErrorEngine(seed=42)
    result = verify(claim, bundle, taxonomy, engine=engine)
    # measured d must be negative (rushed) and exceed the timing tau (8%)
    assert result.measured_value < 0, f"rushed region must give d<0, got {result.measured_value}"
    assert abs(result.measured_value) > result.tau, (
        f"|d|={abs(result.measured_value)} must exceed tau={result.tau}"
    )
    # With d<0, polarity '-', and |d|>tau outside the dead-band, route_verdict yields SUPPORTED
    assert result.verdict == "SUPPORTED", (
        f"expected SUPPORTED for strong rush, got {result.verdict} ({result.reason_code}); "
        f"d={result.measured_value}, tau={result.tau}, error_bar={result.error_bar}"
    )


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
    import soundfile as sf
    taxonomy = _load_taxonomy()
    if taxonomy["dimensions"]["dynamics"]["status"] != "active":
        pytest.skip("taxonomy not yet v0.1; dynamics still gated")
    n_total = SR
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


def test_unknown_measurement_key_returns_substrate_failure_unverifiable() -> None:
    """Orchestrator hits the unknown-measurement-key branch (lines 75-80 of orchestrator.py).

    An active dimension whose 'measurement' value is not registered in the measurer
    registry must return UNVERIFIABLE with reason_code='substrate_failure'.
    """
    taxonomy = _load_taxonomy()
    # Inject a fake active dimension with an unrecognized measurement key into a copy.
    import copy
    fake_taxonomy = copy.deepcopy(taxonomy)
    fake_taxonomy["dimensions"]["fake_dim"] = {
        "status": "active",
        "measurement": "nonexistent_measurement_key",
        "tolerance": {"provisional": 0.05, "unit": "fraction"},
    }
    bundle = _make_timing_bundle()
    claim = {
        "proposition": "Something unregistered",
        "dimension": "fake_dim",
        "location": {"bar_start": 1, "bar_end": 5},
        "polarity": "-",
        "magnitude": None,
    }
    engine = SubstrateErrorEngine(seed=42)
    result = verify(claim, bundle, fake_taxonomy, engine=engine)
    assert result.verdict == "UNVERIFIABLE"
    assert result.reason_code == "substrate_failure"


def test_cli_verify_outputs_json(tmp_path) -> None:
    """CLI writes valid JSON VerdictResult to stdout."""
    import subprocess, sys
    evals_root = Path(__file__).resolve().parents[2]  # .../apps/evals
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
        capture_output=True, text=True, cwd=str(evals_root),
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    out = json.loads(result.stdout)
    assert out["verdict"] == "UNVERIFIABLE"
    assert out["reason_code"] == "out_of_scope_dim"
    assert out["dimension"] == "phrasing"
