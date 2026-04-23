"""Compare three segment filtering approaches on noisy audio.

Tests on a masterclass recording (speech + piano + silence + applause)
and a competition recording (mostly piano but with some dead space).

Usage:
    uv run python scripts/test_segment_filters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_ROOT = Path(__file__).resolve().parent.parent

MASTERCLASS_PATH = MODEL_ROOT / "data/raw/filter_test/zimerman_masterclass.wav"
CLIBURN_PATH = (
    MODEL_ROOT
    / "data/manifests/competition/cliburn_2022/audio"
    / "cliburn2022_preliminary_yunchanlim_hough_couperin_mozart_chopin.wav"
)

SEGMENT_DURATION = 30.0
MIN_SEGMENT_DURATION = 5.0


def load_and_segment(path: Path) -> list[dict]:
    """Load audio and split into 30-second segments."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    segment_samples = int(SEGMENT_DURATION * sr)
    min_samples = int(MIN_SEGMENT_DURATION * sr)
    segments = []
    offset = 0

    while offset < len(audio):
        end = min(offset + segment_samples, len(audio))
        chunk = audio[offset:end]

        if len(chunk) < min_samples:
            break

        segments.append({
            "audio": chunk,
            "start_sec": offset / sr,
            "end_sec": end / sr,
            "sr": sr,
        })
        offset = end

    return segments


# ---- Filter A: RMS energy ----

def filter_a_rms(segments: list[dict], min_rms: float = 0.002) -> list[bool]:
    """Filter by absolute RMS energy threshold.

    Uses absolute threshold instead of percentile to avoid dropping
    quiet piano passages (ppp in Couperin, dying Chopin phrases).

    Threshold 0.002 is well below the quietest piano playing (~0.005)
    but above digital silence + room noise (~0.0001).
    """
    rms_values = []
    for seg in segments:
        rms = np.sqrt(np.mean(seg["audio"] ** 2))
        rms_values.append(rms)

    keep = [rms > min_rms for rms in rms_values]
    return keep, rms_values, min_rms


# ---- Filter B: Harmonic ratio ----

def filter_b_spectral(segments: list[dict], min_harmonic_ratio: float = 0.3) -> list[bool]:
    """Filter by harmonic content ratio.

    Piano produces strong harmonic series (fundamental + overtones at
    integer multiples). Speech has formants but weaker harmonic structure.
    Applause/noise is broadband with no harmonic peaks.

    Approach: autocorrelation-based pitch confidence. High confidence
    = strong periodic (harmonic) content = likely pitched instrument.
    """
    scores = []

    for seg in segments:
        audio = seg["audio"]
        sr = seg["sr"]
        rms = np.sqrt(np.mean(audio ** 2))

        if rms < 1e-5:
            scores.append(0.0)
            continue

        # Compute pitch confidence via autocorrelation on short frames
        frame_len = int(0.050 * sr)  # 50ms (need longer for low pitches)
        hop = int(0.025 * sr)
        confidences = []

        # Piano range: A0 (27.5 Hz) to C8 (4186 Hz)
        min_lag = int(sr / 4186)  # highest piano note
        max_lag = int(sr / 27.5)  # lowest piano note
        max_lag = min(max_lag, frame_len - 1)

        for i in range(0, len(audio) - frame_len, hop):
            frame = audio[i:i + frame_len]
            frame_rms = np.sqrt(np.mean(frame ** 2))

            if frame_rms < 1e-5:
                confidences.append(0.0)
                continue

            # Normalized autocorrelation
            frame = frame - np.mean(frame)
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(frame) - 1:]  # keep positive lags only
            if corr[0] > 0:
                corr = corr / corr[0]  # normalize

            # Find strongest peak in piano pitch range
            search = corr[min_lag:max_lag + 1]
            if len(search) > 0:
                peak_val = np.max(search)
                confidences.append(max(0, peak_val))
            else:
                confidences.append(0.0)

        # Harmonic ratio: fraction of frames with strong periodicity
        conf_arr = np.array(confidences)
        harmonic_frames = np.sum(conf_arr > 0.4)  # 0.4 = reasonably periodic
        harmonic_ratio = harmonic_frames / len(conf_arr) if len(conf_arr) > 0 else 0.0

        # Weight by energy so silence doesn't accidentally score high
        score = harmonic_ratio * min(1.0, rms / 0.005)
        scores.append(score)

    keep = [s >= min_harmonic_ratio for s in scores]
    return keep, scores, min_harmonic_ratio


# ---- Filter C: Piano-specific onset density ----

def filter_c_onsets(
    segments: list[dict],
    min_onsets: int = 3,
) -> list[bool]:
    """Filter by piano-specific onset density.

    Piano onsets are sharp broadband transients (hammer strikes) followed
    by harmonic decay. We detect these by looking for high-amplitude
    spectral flux concentrated in the attack (sharp rise, fast decay).

    Key improvements over naive spectral flux:
    - Use absolute flux threshold based on segment energy, not relative
    - Require onset amplitude significantly above local background
    - Use wider minimum gap (200ms) to avoid counting speech syllables
    """
    onset_counts = []
    onset_strengths = []  # avg strength of detected onsets

    for seg in segments:
        audio = seg["audio"]
        sr = seg["sr"]
        rms = np.sqrt(np.mean(audio ** 2))

        if rms < 1e-5:
            onset_counts.append(0)
            onset_strengths.append(0.0)
            continue

        # Spectral flux with focus on piano-relevant frequencies
        frame_len = int(0.025 * sr)
        hop = int(0.010 * sr)
        prev_spectrum = None
        flux_values = []

        # Piano fundamental range: ~27 Hz (A0) to ~4186 Hz (C8)
        # At 24kHz sr, rfft gives bins up to 12kHz
        # Focus on 50-5000 Hz for onset detection
        freq_resolution = sr / frame_len
        low_bin = max(1, int(50 / freq_resolution))
        high_bin = min(frame_len // 2, int(5000 / freq_resolution))

        for i in range(0, len(audio) - frame_len, hop):
            frame = audio[i:i + frame_len]
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame_len)))
            piano_band = spectrum[low_bin:high_bin]

            if prev_spectrum is not None:
                diff = piano_band - prev_spectrum
                flux = np.sum(np.maximum(0, diff))
                flux_values.append(flux)

            prev_spectrum = piano_band.copy()

        if not flux_values:
            onset_counts.append(0)
            onset_strengths.append(0.0)
            continue

        flux_arr = np.array(flux_values)

        # Adaptive threshold: median + 3*MAD (robust to outliers)
        median_flux = np.median(flux_arr)
        mad = np.median(np.abs(flux_arr - median_flux))
        threshold = median_flux + 3.0 * mad

        # Also require absolute minimum flux (avoids noise triggers)
        abs_min = rms * 5.0  # onset should be much stronger than background
        threshold = max(threshold, abs_min)

        # Wider gap for piano: 200ms (speech syllables are ~100ms apart)
        min_gap = int(0.2 * sr / hop)

        onsets = 0
        strengths = []
        last_onset = -min_gap
        for j, f in enumerate(flux_arr):
            if f > threshold and (j - last_onset) >= min_gap:
                onsets += 1
                strengths.append(f)
                last_onset = j

        onset_counts.append(onsets)
        onset_strengths.append(np.mean(strengths) if strengths else 0.0)

    keep = [count >= min_onsets for count in onset_counts]
    return keep, onset_counts, min_onsets


# ---- Combined: A + C ----

def filter_combined(segments: list[dict]) -> list[bool]:
    """RMS energy AND onset density must both pass."""
    keep_a, rms_values, rms_thresh = filter_a_rms(segments)
    keep_c, onset_counts, onset_thresh = filter_c_onsets(segments)

    keep = [a and c for a, c in zip(keep_a, keep_c)]
    return keep, rms_values, onset_counts


def analyze_recording(path: Path, label: str) -> None:
    """Run all filters on a recording and print comparison."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  {path.name}")
    print(f"{'=' * 70}")

    audio_info = sf.info(str(path))
    print(f"  Duration: {audio_info.duration / 60:.1f} min ({audio_info.duration:.0f}s)")

    segments = load_and_segment(path)
    n_total = len(segments)
    print(f"  Total 30s segments: {n_total}")
    print()

    # Run all filters
    keep_a, rms_vals, rms_thresh = filter_a_rms(segments)
    keep_b, spec_scores, spec_thresh = filter_b_spectral(segments)
    keep_c, onset_counts, onset_thresh = filter_c_onsets(segments)
    keep_ac, _, _ = filter_combined(segments)

    n_a = sum(keep_a)
    n_b = sum(keep_b)
    n_c = sum(keep_c)
    n_ac = sum(keep_ac)

    print(f"  Filter A (RMS energy, >p15):     {n_a:3d} kept / {n_total - n_a:3d} dropped ({100*n_a/n_total:.0f}%)")
    print(f"  Filter B (spectral, z>-1.5):     {n_b:3d} kept / {n_total - n_b:3d} dropped ({100*n_b/n_total:.0f}%)")
    print(f"  Filter C (onsets, >={int(onset_thresh)}):         {n_c:3d} kept / {n_total - n_c:3d} dropped ({100*n_c/n_total:.0f}%)")
    print(f"  Filter A+C (combined):           {n_ac:3d} kept / {n_total - n_ac:3d} dropped ({100*n_ac/n_total:.0f}%)")
    print()

    # Detailed per-segment view (first 20 + last 5)
    print(f"  {'Seg':>4s}  {'Time':>10s}  {'RMS':>8s}  {'Spec':>8s}  {'Onsets':>6s}  {'A':>2s} {'B':>2s} {'C':>2s} {'A+C':>3s}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*2} {'-'*2} {'-'*2} {'-'*3}")

    indices = list(range(min(20, n_total)))
    if n_total > 25:
        indices += list(range(n_total - 5, n_total))

    for i in indices:
        seg = segments[i]
        t0 = seg["start_sec"]
        t1 = seg["end_sec"]
        mark = lambda b: "ok" if b else "--"
        print(
            f"  {i:4d}  {t0/60:4.1f}-{t1/60:4.1f}m  "
            f"{rms_vals[i]:.6f}  {spec_scores[i]:.6f}  {onset_counts[i]:6d}  "
            f"{mark(keep_a[i]):>2s} {mark(keep_b[i]):>2s} {mark(keep_c[i]):>2s} {mark(keep_ac[i]):>3s}"
        )

    if n_total > 25:
        print(f"  ... ({n_total - 25} segments omitted) ...")

    # Agreement analysis
    all_agree_keep = sum(a and b and c for a, b, c in zip(keep_a, keep_b, keep_c))
    all_agree_drop = sum(not a and not b and not c for a, b, c in zip(keep_a, keep_b, keep_c))
    disagreements = n_total - all_agree_keep - all_agree_drop

    print(f"\n  Agreement: {all_agree_keep} all-keep, {all_agree_drop} all-drop, {disagreements} disagreements")


def main() -> None:
    if not MASTERCLASS_PATH.exists():
        print(f"Masterclass audio not found: {MASTERCLASS_PATH}")
        sys.exit(1)

    analyze_recording(MASTERCLASS_PATH, "MASTERCLASS: Zimerman (speech + piano + silence)")

    if CLIBURN_PATH.exists():
        analyze_recording(CLIBURN_PATH, "COMPETITION: Yunchan Lim preliminary (mostly piano)")
    else:
        print(f"\nCliburn audio not found at {CLIBURN_PATH}, skipping comparison.")


if __name__ == "__main__":
    main()
