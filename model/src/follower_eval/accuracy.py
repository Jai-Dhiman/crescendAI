# model/src/follower_eval/accuracy.py
"""Gold (accuracy) track of the real-audio score-follower eval (issue #133 S3).

The proxy track (``follower_eval.realaudio``) reports anchor-free structure
(coverage / span / confidence) but CANNOT tell a correct alignment from a
confidently-wrong one -- both look the same. This module supplies the one
NON-CIRCULAR accuracy number: a human taps each bar downbeat while listening to
the real recording (``follower_eval.tap_tool``), producing
``(bar_number -> audio_sec)`` ground truth. We then ask the follower, at each
tapped ``audio_sec``, "where in the score do you think we are?" and compare that
decoded score-second to the bar's TRUE score-second from the score's
``measure_table``.

WHY THE TWO CLOCKS LINE UP (the crux):
  * A gold tap's ``audio_sec`` is the WAV playback time the labeler tapped at.
  * ``MatchedNote.perf_time`` is the Transkun note onset in that SAME WAV clock
    (the WAV the tap tool serves IS the transcription's source audio).
  So interpolating the follower's ``perf_time -> score_position`` staircase at a
  tapped ``audio_sec`` yields the follower's decoded score-second, directly
  comparable to ``measure_table[bar_number].start_sec`` (score-render seconds).
  Both sides live in score-render seconds -> the error is a real "how far off is
  the follower's cursor," not a proxy.

METRICS (per clip, then pooled -- distributions, never medians-of-medians):
  * bar-localization error -- |decoded_score_sec - true_bar_start_sec|, reported
    in seconds AND in bars (error / local bar duration; tempo-invariant).
  * within-tolerance rate -- fraction of taps within +/-1 bar and +/-0.5 bar.
  * relock latency after restarts -- when the gold taps show the performer went
    BACKWARD (bar_number drops = a repeat/restart), how many audio-seconds until
    the follower's decoded position falls back within tolerance of the tapped
    bar. This is the property the live cursor depends on.

RUNNING (from the PRIMARY checkout so data/ + the venv resolve):

  cd /Users/jdhiman/Documents/crescendai/model
  PYTHONPATH=<worktree>/model/src .venv/bin/python -m follower_eval.gold_report \
    --bundles-root data/evals/realaudio_bundles --scores-root data/scores
"""
from __future__ import annotations

import bisect
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from follower_bench.hmm import TUNED_HMM_PARAMS, follow_hmm

from follower_eval.realaudio import RealAudioEvalError, load_bundle_notes

# A tap is "correctly localized" if the follower's decoded score-second is within
# this many bars of the tapped bar's true downbeat. One bar is the natural unit:
# the live cursor only needs to sit on the right measure. Half a bar is the
# stricter beat-level bound. Both are REPORTED; the PASS gate (gold_report) is
# built on the observed distributions, not hard-coded here.
TOL_BARS_LENIENT = 1.0
TOL_BARS_STRICT = 0.5


@dataclass(frozen=True)
class BarTap:
    """One human bar-downbeat tap: which score bar, at what WAV playback second."""
    bar_number: int
    audio_sec: float


@dataclass(frozen=True)
class TapError:
    """Per-tap localization result (score-render seconds unless noted)."""
    bar_number: int
    audio_sec: float
    true_score_sec: float
    decoded_score_sec: float | None  # None if no follower match brackets this tap
    abs_err_sec: float | None
    abs_err_bars: float | None       # abs_err_sec / local bar duration
    local_bar_sec: float
    is_restart: bool                 # this tap's bar_number < the previous tap's


@dataclass(frozen=True)
class ClipAccuracy:
    """One gold-labeled clip's accuracy against the follower."""
    piece: str
    bundle: str
    n_taps: int
    n_decoded: int                   # taps the follower could place (bracketed)
    median_abs_err_sec: float | None
    p90_abs_err_sec: float | None
    median_abs_err_bars: float | None
    p90_abs_err_bars: float | None
    within_1bar_frac: float | None
    within_half_bar_frac: float | None
    n_restarts: int
    relock_latencies_sec: tuple[float, ...]   # one per restart that relocked
    n_restart_no_relock: int                  # restarts that never came back within tol
    transpose_semitones: int
    tap_errors: tuple[TapError, ...]


def load_gold(gold_path: Path) -> list[BarTap]:
    """Read a ``<vid>.gold.json`` (``{bar_taps:[{bar_number,audio_sec}]}``) into
    BarTaps sorted by audio_sec. Loud on an empty/malformed file -- a gold file
    with no taps is a labeling error, not a zero-accuracy clip.

    Raises:
        RealAudioEvalError: missing ``bar_taps`` or an empty tap list.
    """
    body = json.loads(gold_path.read_text())
    taps = body.get("bar_taps")
    if not taps:
        raise RealAudioEvalError(f"{gold_path.name}: no 'bar_taps' (unlabeled?)")
    out = [BarTap(bar_number=int(t["bar_number"]), audio_sec=float(t["audio_sec"])) for t in taps]
    out.sort(key=lambda t: t.audio_sec)
    return out


def bar_second_table(measure_table: list[dict]) -> tuple[dict[int, float], dict[int, float]]:
    """(bar_number -> true start_sec, bar_number -> local bar duration in sec).

    Local duration = next bar's start - this bar's start; the final bar reuses
    the previous span. Used both to look up a tap's true score-second and to
    express its error in tempo-invariant bar units."""
    starts = {int(r["bar_number"]): float(r["start_sec"]) for r in measure_table}
    ordered = sorted(starts.items())
    durs: dict[int, float] = {}
    for i, (bn, s) in enumerate(ordered):
        if i + 1 < len(ordered):
            durs[bn] = ordered[i + 1][1] - s
        elif i > 0:
            durs[bn] = s - ordered[i - 1][1]
        else:
            durs[bn] = 0.0
    return starts, durs


def decode_at(perf_times: list[float], score_positions: list[float], t: float) -> float | None:
    """Follower's decoded score-second at WAV time ``t``: linear interpolation of
    the ``perf_time -> score_position`` match staircase.

    Clamps to the first/last match outside the matched span (the follower placed
    nothing there, so its best estimate is the nearest edge). Returns None only
    when there are no matches at all. ``perf_times`` MUST be ascending."""
    if not perf_times:
        return None
    if t <= perf_times[0]:
        return score_positions[0]
    if t >= perf_times[-1]:
        return score_positions[-1]
    k = bisect.bisect_left(perf_times, t)
    # perf_times[k-1] < t <= perf_times[k]
    a_t, b_t = perf_times[k - 1], perf_times[k]
    a_s, b_s = score_positions[k - 1], score_positions[k]
    if b_t == a_t:
        return b_s
    frac = (t - a_t) / (b_t - a_t)
    return a_s + frac * (b_s - a_s)


def _pctl(values: list[float], q: float) -> float:
    """Simple linear-interpolation percentile (q in [0,1]); avoids a numpy import
    for a handful of taps."""
    if not values:
        raise ValueError("empty")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def evaluate_taps(taps: list[BarTap], matches, bar_starts: dict[int, float],
                  bar_durs: dict[int, float]) -> list[TapError]:
    """Compare each gold tap to the follower's decoded position.

    Args:
        matches: the follower's ``EstimatedTrajectory.matches`` (any order).
        bar_starts / bar_durs: from :func:`bar_second_table`.

    Raises:
        RealAudioEvalError: a tapped ``bar_number`` is not in the score's
            measure table (a mislabeled bar -- loud, never silently dropped).
    """
    ms = sorted(matches, key=lambda m: m.perf_time)
    perf_times = [m.perf_time for m in ms]
    score_positions = [m.score_position for m in ms]

    errors: list[TapError] = []
    prev_bar: int | None = None
    for tap in taps:
        if tap.bar_number not in bar_starts:
            raise RealAudioEvalError(
                f"tapped bar {tap.bar_number} not in score measure_table "
                f"(bars {min(bar_starts)}-{max(bar_starts)})")
        true_sec = bar_starts[tap.bar_number]
        local_bar = bar_durs[tap.bar_number]
        decoded = decode_at(perf_times, score_positions, tap.audio_sec)
        abs_err = abs(decoded - true_sec) if decoded is not None else None
        abs_err_bars = (abs_err / local_bar) if (abs_err is not None and local_bar > 0) else None
        errors.append(TapError(
            bar_number=tap.bar_number,
            audio_sec=tap.audio_sec,
            true_score_sec=round(true_sec, 3),
            decoded_score_sec=round(decoded, 3) if decoded is not None else None,
            abs_err_sec=round(abs_err, 3) if abs_err is not None else None,
            abs_err_bars=round(abs_err_bars, 3) if abs_err_bars is not None else None,
            local_bar_sec=round(local_bar, 3),
            is_restart=(prev_bar is not None and tap.bar_number < prev_bar),
        ))
        prev_bar = tap.bar_number
    return errors


def _relock(tap_errors: list[TapError], tol_bars: float) -> tuple[list[float], int]:
    """(relock latencies, n_restarts_that_never_relocked).

    For each restart tap (bar_number dropped vs the previous tap), scan forward
    to the first tap whose error is within ``tol_bars`` and take the audio-second
    gap. A restart already within tolerance relocks with latency 0."""
    latencies: list[float] = []
    no_relock = 0
    n = len(tap_errors)
    for i, te in enumerate(tap_errors):
        if not te.is_restart:
            continue
        relocked = False
        for j in range(i, n):
            ej = tap_errors[j]
            if ej.abs_err_bars is not None and ej.abs_err_bars <= tol_bars:
                latencies.append(round(ej.audio_sec - te.audio_sec, 3))
                relocked = True
                break
        if not relocked:
            no_relock += 1
    return latencies, no_relock


def evaluate_clip(piece: str, bundle_path: Path, gold_path: Path,
                  score_notes, bar_boundaries, measure_table: list[dict]) -> ClipAccuracy:
    """Full gold accuracy for one clip: run the follower, compare to its taps."""
    taps = load_gold(gold_path)
    perf = load_bundle_notes(bundle_path)
    est = follow_hmm(perf, score_notes, TUNED_HMM_PARAMS, bar_boundaries=bar_boundaries)

    bar_starts, bar_durs = bar_second_table(measure_table)
    tap_errors = evaluate_taps(taps, est.matches, bar_starts, bar_durs)

    decoded_errs_sec = [te.abs_err_sec for te in tap_errors if te.abs_err_sec is not None]
    decoded_errs_bars = [te.abs_err_bars for te in tap_errors if te.abs_err_bars is not None]
    n_decoded = len(decoded_errs_sec)
    within_1 = ([te.abs_err_bars <= TOL_BARS_LENIENT for te in tap_errors if te.abs_err_bars is not None])
    within_half = ([te.abs_err_bars <= TOL_BARS_STRICT for te in tap_errors if te.abs_err_bars is not None])
    latencies, no_relock = _relock(tap_errors, TOL_BARS_LENIENT)

    return ClipAccuracy(
        piece=piece,
        bundle=bundle_path.stem,
        n_taps=len(taps),
        n_decoded=n_decoded,
        median_abs_err_sec=round(statistics.median(decoded_errs_sec), 3) if decoded_errs_sec else None,
        p90_abs_err_sec=round(_pctl(decoded_errs_sec, 0.9), 3) if decoded_errs_sec else None,
        median_abs_err_bars=round(statistics.median(decoded_errs_bars), 3) if decoded_errs_bars else None,
        p90_abs_err_bars=round(_pctl(decoded_errs_bars, 0.9), 3) if decoded_errs_bars else None,
        within_1bar_frac=round(sum(within_1) / len(within_1), 4) if within_1 else None,
        within_half_bar_frac=round(sum(within_half) / len(within_half), 4) if within_half else None,
        n_restarts=sum(1 for te in tap_errors if te.is_restart),
        relock_latencies_sec=tuple(latencies),
        n_restart_no_relock=no_relock,
        transpose_semitones=est.transpose_semitones,
        tap_errors=tuple(tap_errors),
    )


def clip_to_jsonable(acc: ClipAccuracy) -> dict:
    """ClipAccuracy -> plain dict (tap_errors expanded), for the JSON report."""
    out = asdict(acc)
    out["tap_errors"] = [asdict(te) for te in acc.tap_errors]
    out["relock_latencies_sec"] = list(acc.relock_latencies_sec)
    return out
