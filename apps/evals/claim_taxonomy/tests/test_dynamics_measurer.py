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
    import soundfile as sf
    n_total = SR * 20
    t = np.linspace(0, 20, n_total)
    amplitude_envelope = 0.1 + 0.8 * (t / 20.0)
    whole = (np.sin(2 * np.pi * 440 * t) * amplitude_envelope).astype(np.float32)
    audio_path = tmp_path / "test.wav"
    sf.write(str(audio_path), whole, SR)
    bundle = _make_bundle(whole, str(audio_path))
    engine = SubstrateErrorEngine(seed=42)
    measurer = DynamicsMeasurer()
    region = _make_region(start=0.0, end=5.0)
    result = measurer.measure(location={"bar_start": 1, "bar_end": 3},
                              bundle=bundle, region=region, engine=engine)
    assert result.d < 0, f"Flat region should have negative d, got {result.d}"


def test_wide_dynamic_region_positive_or_zero_d(tmp_path) -> None:
    import soundfile as sf
    n_total = SR * 20
    whole = (np.sin(2 * np.pi * 440 * np.linspace(0, 20, n_total)) * 0.2).astype(np.float32)
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
    import soundfile as sf
    n_total = SR * 30
    audio = np.zeros(n_total, dtype=np.float32)
    t_all = np.linspace(0, 30, n_total)
    audio = (np.sin(2 * np.pi * 440 * t_all) * 0.05).astype(np.float32)
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
    import soundfile as sf
    audio = np.zeros(SR * 10, dtype=np.float32)
    audio_path = tmp_path / "test_short.wav"
    sf.write(str(audio_path), audio, SR)
    bundle = _make_bundle(audio, str(audio_path))
    engine = SubstrateErrorEngine(seed=42)
    measurer = DynamicsMeasurer()
    region = _make_region(start=0.0, end=0.3)
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
