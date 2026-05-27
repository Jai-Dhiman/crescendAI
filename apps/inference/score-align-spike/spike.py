"""Chroma subsequence DTW spike for score following.

Diagnostic only. Not production code. See conversation for context.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import librosa
import numpy as np

SR = 22050
HOP = 441            # 50 Hz frame rate at 22050 Hz
FRAME_RATE = SR / HOP  # ~50.0

AUDIO_WAV = Path("/Users/jdhiman/Documents/crescendai/model/data/evals/skill_eval/chopin_ballade_1/audio/HlHBUxlcWfk.wav")
SCORE_JSON = Path("/Users/jdhiman/Documents/crescendai/model/data/scores/chopin.ballades.1.json")
SCORE_CHROMA_CACHE = Path("/tmp/chopin_ballade1_score_chroma.npy")
AUDIO_CHROMA_CACHE = Path("/tmp/chopin_ballade1_audio_chroma.npy")


def build_score_chroma(score_json_path: Path, frame_rate: float) -> tuple[np.ndarray, list[dict]]:
    """Construct a 12 x N chroma matrix directly from the note list."""
    score = json.loads(score_json_path.read_text())
    bars = score["bars"]
    notes = []
    for bar in bars:
        for n in bar["notes"]:
            onset = float(n["onset_seconds"])
            dur = max(float(n["duration_seconds"]), 0.05)
            notes.append((onset, onset + dur, int(n["pitch"]) % 12))
    if not notes:
        raise RuntimeError("score has no notes")
    end_time = max(n[1] for n in notes)
    n_frames = int(np.ceil(end_time * frame_rate)) + 1
    chroma = np.zeros((12, n_frames), dtype=np.float32)
    for onset, offset, pc in notes:
        f0 = int(np.floor(onset * frame_rate))
        f1 = int(np.ceil(offset * frame_rate))
        f0 = max(0, f0)
        f1 = min(n_frames, f1)
        if f1 <= f0:
            continue
        chroma[pc, f0:f1] += 1.0
    # Floor: prevents all-zero columns at rests, which break cosine distance.
    chroma += 1e-3
    norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma /= norm
    return chroma, bars


def load_audio_chroma() -> np.ndarray:
    if AUDIO_CHROMA_CACHE.exists():
        return np.load(AUDIO_CHROMA_CACHE)
    print(f"[load] decoding {AUDIO_WAV.name} -> chroma at {FRAME_RATE:.1f} Hz")
    y, _ = librosa.load(str(AUDIO_WAV), sr=SR, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    chroma = chroma.astype(np.float32) + 1e-3
    norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma /= norm
    np.save(AUDIO_CHROMA_CACHE, chroma)
    return chroma


def load_score_chroma() -> tuple[np.ndarray, list[dict]]:
    bars = json.loads(SCORE_JSON.read_text())["bars"]
    if SCORE_CHROMA_CACHE.exists():
        return np.load(SCORE_CHROMA_CACHE), bars
    chroma, bars = build_score_chroma(SCORE_JSON, FRAME_RATE)
    np.save(SCORE_CHROMA_CACHE, chroma)
    return chroma, bars


def frame_to_bar(frame_idx: int, frame_rate: float, bars: list[dict]) -> int:
    t = frame_idx / frame_rate
    last = 1
    for b in bars:
        if b["start_seconds"] <= t:
            last = b["bar_number"]
        else:
            break
    return last


def run_subseq_dtw(audio_chroma_slice: np.ndarray, score_chroma: np.ndarray):
    """librosa subsequence DTW: aligns the *short* sequence Y as a subseq of X.

    Returns (path, wp_cost, wall_seconds, monotonic_flag).
    """
    t0 = time.perf_counter()
    # X = score (long), Y = audio chunk (short). subseq=True finds best
    # contiguous-ish region of X matching Y.
    _D, wp = librosa.sequence.dtw(
        X=score_chroma,
        Y=audio_chroma_slice,
        subseq=True,
        backtrack=True,
        metric="cosine",
    )
    wall = time.perf_counter() - t0
    # wp is returned reversed: [(score_idx, audio_idx), ...] from end to start
    wp = wp[::-1]
    score_path = wp[:, 0]
    # Monotonicity check on score-side path (allow tiny noise from step pattern)
    deltas = np.diff(score_path)
    nonmono_frac = float(np.mean(deltas < 0))
    # Mean cosine distance along the warping path (axis-independent).
    s_idx = wp[:, 0]
    a_idx = wp[:, 1]
    dots = np.sum(score_chroma[:, s_idx] * audio_chroma_slice[:, a_idx], axis=0)
    cost = float(np.mean(1.0 - dots))
    return wp, cost, wall, nonmono_frac


def run_test(name: str, audio_chroma: np.ndarray, score_chroma: np.ndarray,
             bars: list[dict], start_s: float, dur_s: float,
             second_segment: tuple[float, float] | None = None):
    f0 = int(round(start_s * FRAME_RATE))
    f1 = int(round((start_s + dur_s) * FRAME_RATE))
    chunk = audio_chroma[:, f0:f1]
    label = f"audio[{start_s:.1f}s..{start_s + dur_s:.1f}s]"
    if second_segment is not None:
        s0, sd = second_segment
        g0 = int(round(s0 * FRAME_RATE))
        g1 = int(round((s0 + sd) * FRAME_RATE))
        chunk = np.concatenate([chunk, audio_chroma[:, g0:g1]], axis=1)
        label = f"{label} ++ audio[{s0:.1f}s..{s0 + sd:.1f}s]"

    wp, cost, wall, nonmono_frac = run_subseq_dtw(chunk, score_chroma)
    score_frames = wp[:, 0]
    bars_hit = [frame_to_bar(int(f), FRAME_RATE, bars) for f in score_frames]
    bars_arr = np.array(bars_hit)

    # Per-quartile summary (so we can see jumps within the chunk)
    q = np.linspace(0, len(bars_arr) - 1, 5).astype(int)
    quart_bars = bars_arr[q].tolist()

    print(f"\n=== {name} ===")
    print(f"  input        : {label}  ({chunk.shape[1]} frames)")
    print(f"  bars covered : min={int(bars_arr.min())} max={int(bars_arr.max())} "
          f"unique={len(set(bars_hit))}")
    print(f"  bar trajectory at [0%, 25%, 50%, 75%, 100%] = {quart_bars}")
    print(f"  non-monotonic frame fraction on score axis = {nonmono_frac:.3f}")
    print(f"  mean DTW cost (cosine, lower=better) = {cost:.4f}")
    print(f"  wall-clock = {wall * 1000:.0f} ms")
    return {
        "name": name,
        "bar_min": int(bars_arr.min()),
        "bar_max": int(bars_arr.max()),
        "bar_quartiles": quart_bars,
        "nonmono_frac": nonmono_frac,
        "cost": cost,
        "wall_ms": wall * 1000,
    }


def main():
    print(f"[setup] frame rate = {FRAME_RATE:.3f} Hz, hop = {HOP}, sr = {SR}")
    score_chroma, bars = load_score_chroma()
    print(f"[setup] score chroma : 12 x {score_chroma.shape[1]} "
          f"({score_chroma.shape[1] / FRAME_RATE:.1f}s, {len(bars)} bars)")
    audio_chroma = load_audio_chroma()
    print(f"[setup] audio chroma : 12 x {audio_chroma.shape[1]} "
          f"({audio_chroma.shape[1] / FRAME_RATE:.1f}s)")

    results = []
    # Test 1: 2-minute forward tracking, expect bars 1 -> ~33
    results.append(run_test("T1 forward 2min from 0s", audio_chroma, score_chroma,
                            bars, 0.0, 120.0))
    # Test 2: cold start mid-piece at 111s, expect bars ~25-40
    results.append(run_test("T2 cold start 15s @ 111s", audio_chroma, score_chroma,
                            bars, 111.0, 15.0))
    # Test 3: drilling -- same 15s aligned 3x
    for i in range(3):
        results.append(run_test(f"T3 drill rep {i+1} (15s @ 60s)", audio_chroma,
                                score_chroma, bars, 60.0, 15.0))
    # Test 4: jump -- 7.5s @ 0s ++ 7.5s @ 200s
    results.append(run_test("T4 jump 7.5s@0 + 7.5s@200", audio_chroma, score_chroma,
                            bars, 0.0, 7.5, second_segment=(200.0, 7.5)))
    # Test 5: 15s perf benchmark already covered by T2/T3 wall_ms

    print("\n=== Summary table ===")
    print(f"{'Test':<35} {'bar_min':>8} {'bar_max':>8} {'nonmono':>8} {'cost':>8} {'ms':>6}")
    for r in results:
        print(f"{r['name']:<35} {r['bar_min']:>8} {r['bar_max']:>8} "
              f"{r['nonmono_frac']:>8.3f} {r['cost']:>8.4f} {r['wall_ms']:>6.0f}")


if __name__ == "__main__":
    main()
