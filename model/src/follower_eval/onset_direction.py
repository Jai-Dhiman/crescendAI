# model/src/follower_eval/onset_direction.py
"""Clean-audio baseline for the DIRECTIONAL onset-timing signal (#108 / #148).

#108's second unmeasured deliverable: given a performed note, is the student
EARLY or LATE, and by how much. This module measures the sign accuracy and the
magnitude error of that call on clean audio, so #148's out-of-distribution table
has a baseline to be subtracted from.

DIRECTION RELATIVE TO WHAT REFERENCE TEMPO -- the design question, answered
--------------------------------------------------------------------------
"Early" is meaningless without a reference. Three were available:

  * **The score's notated tempo.** Rejected. An amateur working at 60% of the
    marked tempo would have EVERY note labelled late. The signal would encode
    the student's tempo choice, which is not an error, and would say nothing
    about the note.

  * **A global affine fit** over the whole performance. Rejected. A genuine
    ritardando makes the entire closing section read late. This measures tempo
    interpretation, again not the note.

  * **A LOCAL reference** -- chosen. Deviation is measured against the tempo the
    performer was actually playing at, right there. The resulting claim is "you
    rushed THIS note relative to how you were playing around it", which is the
    only one of the three a teacher would make and the only one the product can
    act on.

Concretely, the two sides use the same definition and differ only in whose
local tempo map they consult:

  * **Truth deviation** = perf onset - the perf time ASAP's human-verified beat
    alignment predicts for that note's TRUE score position. ASAP's anchors are
    per-beat, so this reference already follows the performer's rubato at beat
    resolution; the residual is sub-beat timing, which is exactly the quantity.

  * **System deviation** = perf onset - the perf time the FOLLOWER's decoded
    matches predict for the score position the FOLLOWER assigned, from a local
    linear fit over its own matches within +/-``LOCAL_WINDOW_S``.

Sign accuracy compares the two signs; magnitude error compares the two values.
Truth never touches the follower, per #101's gate1 lesson.

THE DEADBAND, AND WHY IT IS NOT A CONVENIENCE
---------------------------------------------
A note played exactly on time has no meaningful direction -- its sign is
whichever way the noise fell. Scoring those inflates or deflates sign accuracy
for reasons unrelated to the signal. Notes whose TRUTH deviation is under
``DEADBAND_S`` are therefore excluded, and the excluded fraction is reported
with every result. The deadband is applied on the truth side only; using the
system's own deviation to decide what to score would let it choose its exam.

THE NULL CONTROL
----------------
Sign accuracy has no interpretable scale on its own: if 80% of notes are late,
a constant "late" answers 80% correctly while carrying no information. Every
run therefore reports two controls alongside the real number:

  * **shuffled** -- system deviations permuted across notes. This destroys the
    per-note pairing while preserving both marginals.

    **Its expected value is NOT 0.5.** When truth is a fraction ``q`` late and
    the system answers late a fraction ``p`` of the time, a shuffled arm still
    agrees at ``p*q + (1-p)(1-q)``. On clean ASAP that is around 0.57-0.60, not
    chance. Reading 0.57 as "barely above chance" would be wrong; it is exactly
    what a signal carrying zero per-note information looks like here.

  * **majority** -- always answer whichever direction is more common in truth.
    This is the floor the real number has to beat to mean anything.

A run that does not beat both is reported as uninformative, not as a result.

WHAT POPULATION THE NUMBER DESCRIBES
------------------------------------
The deadband is not a small trim. On clean ASAP performance MIDI it removes
about 84% of matched notes, because ASAP's beat anchors predict most onsets to
within 20 ms. The scored population is therefore the minority of notes that are
genuinely displaced -- between-beat and rubato notes -- which is the population
the early/late call is FOR, but it is not "all notes", and the headline must
never be quoted as if it were.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from follower_bench.hmm import TUNED_HMM_PARAMS, follow_hmm

from follower_eval.asap_eval import AsapEvalError, aligned_pieces, load_asap_clip
from follower_eval.note_correspondence import (
    NoteCorrespondenceError,
    _interp,
    derive_truth,
)

# Half-width of the local window the follower's tempo map is fit over, in
# performance seconds. Wide enough to hold several matches at any tempo, narrow
# enough that a rubato phrase is not averaged with its neighbours.
LOCAL_WINDOW_S = 2.0

# Minimum follower matches inside the window for a local fit. Two points define
# a line with zero residual and no way to tell a good fit from a bad one.
MIN_LOCAL_MATCHES = 4

# Notes whose TRUTH deviation is under this are on time; their direction is
# noise, so they are excluded and counted.
DEADBAND_S = 0.02

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 148


class OnsetDirectionError(RuntimeError):
    """Raised when a performance cannot be scored as specified. Loud."""


@dataclass(frozen=True)
class ClipDirection:
    """One performance's directional-timing result."""

    asap_piece: str
    n_scored: int
    n_deadband_excluded: int
    n_no_local_fit: int
    sign_accuracy: float | None
    median_abs_magnitude_err_s: float | None
    truth_late_frac: float | None
    shuffled_sign_accuracy: float | None
    majority_sign_accuracy: float | None


def _local_fit(points: list[tuple[float, float]], at_score_pos: float) -> float | None:
    """Least-squares perf_time ~ score_position over `points`, evaluated at
    `at_score_pos`. Returns None when the window is degenerate (all matches at
    one score position gives no slope)."""
    n = len(points)
    if n < MIN_LOCAL_MATCHES:
        return None
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope * at_score_pos + intercept


def _inverse_beat_map(truth) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(score_secs, perf_times) -- ASAP's alignment read the other way, so a
    score position can be mapped to the perf time the performer reached it."""
    pairs = sorted(zip(truth.score_secs, truth.perf_times))
    return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)


def score_clip(
    asap_piece: str,
    asap_root: Path,
    audio_cache: Path | None = None,
    seed: int = BOOTSTRAP_SEED,
) -> ClipDirection:
    """Measure the early/late call on one ASAP performance."""
    perf_notes, score_notes, bar_boundaries, truth = load_asap_clip(
        asap_piece, asap_root, audio_cache=audio_cache
    )
    pairs, _, _ = derive_truth(perf_notes, score_notes, truth)
    if not pairs:
        raise OnsetDirectionError(f"{asap_piece}: no truth pairs")

    est = follow_hmm(
        perf_notes, score_notes, TUNED_HMM_PARAMS, bar_boundaries=bar_boundaries
    )
    matches = sorted(est.matches, key=lambda m: m.perf_time)
    match_by_perf = {m.perf_index: m for m in matches}
    map_points = [(m.score_position, m.perf_time) for m in matches]
    match_times = [m.perf_time for m in matches]

    score_xs, perf_ys = _inverse_beat_map(truth)

    truth_devs: list[float] = []
    sys_devs: list[float] = []
    n_deadband = n_no_fit = 0

    for i, true_j in pairs.items():
        m = match_by_perf.get(i)
        if m is None:
            continue
        p = perf_notes[i].onset

        # truth side: ASAP's beat alignment, at the TRUE score position
        expected_truth = _interp(score_xs, perf_ys, score_notes[true_j].position)
        d_truth = p - expected_truth
        if abs(d_truth) < DEADBAND_S:
            n_deadband += 1
            continue

        # system side: the follower's own local tempo map, at the score
        # position the FOLLOWER assigned
        lo = _bisect(match_times, p - LOCAL_WINDOW_S)
        hi = _bisect(match_times, p + LOCAL_WINDOW_S)
        expected_sys = _local_fit(map_points[lo:hi], m.score_position)
        if expected_sys is None:
            n_no_fit += 1
            continue

        truth_devs.append(d_truth)
        sys_devs.append(p - expected_sys)

    if not truth_devs:
        raise OnsetDirectionError(
            f"{asap_piece}: no note survived the deadband and local-fit filters "
            f"({n_deadband} in deadband, {n_no_fit} without a local fit)"
        )

    correct = sum(1 for t, s in zip(truth_devs, sys_devs) if (t > 0) == (s > 0))
    late_frac = sum(1 for t in truth_devs if t > 0) / len(truth_devs)
    mag_errs = sorted(abs(t - s) for t, s in zip(truth_devs, sys_devs))

    rng = random.Random(seed)
    shuffled = list(sys_devs)
    rng.shuffle(shuffled)
    shuffled_acc = sum(
        1 for t, s in zip(truth_devs, shuffled) if (t > 0) == (s > 0)
    ) / len(truth_devs)

    return ClipDirection(
        asap_piece=asap_piece,
        n_scored=len(truth_devs),
        n_deadband_excluded=n_deadband,
        n_no_local_fit=n_no_fit,
        sign_accuracy=round(correct / len(truth_devs), 4),
        median_abs_magnitude_err_s=round(mag_errs[len(mag_errs) // 2], 5),
        truth_late_frac=round(late_frac, 4),
        shuffled_sign_accuracy=round(shuffled_acc, 4),
        majority_sign_accuracy=round(max(late_frac, 1 - late_frac), 4),
    )


def _bisect(xs: list[float], v: float) -> int:
    from bisect import bisect_left

    return bisect_left(xs, v)


def _pooled(clips: list[ClipDirection]) -> dict:
    n = sum(c.n_scored for c in clips)
    if not n:
        return {"sign_accuracy": None, "shuffled": None, "majority": None}
    return {
        "sign_accuracy": round(sum(c.sign_accuracy * c.n_scored for c in clips) / n, 4),
        "shuffled": round(
            sum(c.shuffled_sign_accuracy * c.n_scored for c in clips) / n, 4
        ),
        "majority": round(
            sum(c.majority_sign_accuracy * c.n_scored for c in clips) / n, 4
        ),
    }


def bootstrap_ci(
    clips: list[ClipDirection], n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> dict:
    """Cluster bootstrap over PERFORMANCES -- notes inside one performance share
    a pianist and a tempo, so they are not independent draws."""
    if not clips:
        raise OnsetDirectionError("no clips to bootstrap")
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {"sign_accuracy": [], "shuffled": []}
    for _ in range(n):
        sample = [clips[rng.randrange(len(clips))] for _ in clips]
        pooled = _pooled(sample)
        draws["sign_accuracy"].append(pooled["sign_accuracy"])
        draws["shuffled"].append(pooled["shuffled"])
    out = {}
    for key, vals in draws.items():
        vals.sort()
        out[key] = {
            "ci95": [
                round(vals[int(0.025 * (len(vals) - 1))], 4),
                round(vals[int(0.975 * (len(vals) - 1))], 4),
            ]
        }
    return out


def run(
    asap_root: Path, limit: int | None = None, audio_cache: Path | None = None
) -> dict:
    pieces = aligned_pieces(asap_root / "asap_annotations.json", limit=limit)
    clips: list[ClipDirection] = []
    failures: list[dict] = []
    for piece in pieces:
        try:
            clips.append(score_clip(piece, asap_root, audio_cache=audio_cache))
        except (AsapEvalError, NoteCorrespondenceError, OnsetDirectionError) as exc:
            failures.append({"piece": piece, "error": f"{type(exc).__name__}: {exc}"})
    if not clips:
        raise OnsetDirectionError(f"no clip scored; {len(failures)} failures")

    pooled = _pooled(clips)
    mags = sorted(
        c.median_abs_magnitude_err_s
        for c in clips
        if c.median_abs_magnitude_err_s is not None
    )
    informative = (
        pooled["sign_accuracy"] > pooled["majority"]
        and pooled["sign_accuracy"] > pooled["shuffled"]
    )
    return {
        "system_under_test": "follower_bench.hmm.follow_hmm (#119), local tempo fit "
        f"over its own matches within +/-{LOCAL_WINDOW_S}s",
        "truth_provenance": (
            "ASAP human-verified beat alignment read score->perf, evaluated at the "
            "note's TRUE score position (note_correspondence.derive_truth). The "
            "beat anchors are human-verified; the interpolation is ours. Not "
            "produced by the system under test."
        ),
        "reference_tempo": (
            "LOCAL, not notated and not a global fit. Notated tempo would label "
            "every note of a slow practice performance 'late'; a global fit would "
            "label a whole ritardando 'late'. Only a local reference supports the "
            "claim 'you rushed this note'."
        ),
        "note_source": "ASAP performance MIDI"
        if audio_cache is None
        else "MAESTRO audio -> Transkun",
        "params": {
            "local_window_s": LOCAL_WINDOW_S,
            "min_local_matches": MIN_LOCAL_MATCHES,
            "deadband_s": DEADBAND_S,
        },
        "n_clips": len(clips),
        "n_scored": sum(c.n_scored for c in clips),
        "n_deadband_excluded": sum(c.n_deadband_excluded for c in clips),
        "n_no_local_fit": sum(c.n_no_local_fit for c in clips),
        "scored_frac_of_matched": round(
            sum(c.n_scored for c in clips)
            / sum(c.n_scored + c.n_deadband_excluded + c.n_no_local_fit for c in clips),
            4,
        ),
        "null_shuffled_expected_note": (
            "the shuffled null's expectation is p*q+(1-p)(1-q), NOT 0.5 -- a "
            "value near 0.57-0.60 here means zero per-note information, not "
            "'barely above chance'"
        ),
        "sign_accuracy": pooled["sign_accuracy"],
        "null_shuffled": pooled["shuffled"],
        "null_majority": pooled["majority"],
        "informative": informative,
        "median_abs_magnitude_err_s": round(mags[len(mags) // 2], 5) if mags else None,
        "bootstrap": bootstrap_ci(clips),
        "failures": failures,
        "clips": [asdict(c) for c in clips],
    }


def _format(r: dict) -> str:
    b = r["bootstrap"]
    verdict = (
        "INFORMATIVE (beats both nulls)"
        if r["informative"]
        else "UNINFORMATIVE -- does not beat its nulls; do not quote as a result"
    )
    return "\n".join(
        [
            "=" * 78,
            "DIRECTIONAL ONSET TIMING -- CLEAN-AUDIO BASELINE (#108 / #148)",
            "=" * 78,
            f"system under test : {r['system_under_test']}",
            f"note source       : {r['note_source']}",
            f"reference tempo   : LOCAL (+/-{r['params']['local_window_s']}s)",
            f"clips {r['n_clips']}   scored notes {r['n_scored']:,}   "
            f"deadband excluded {r['n_deadband_excluded']:,}   "
            f"no local fit {r['n_no_local_fit']:,}",
            f"  -> scored population is {r['scored_frac_of_matched']:.1%} of matched "
            f"notes: the ones genuinely displaced, NOT all notes",
            "",
            f"sign accuracy        {r['sign_accuracy']:.4f}  "
            f"95% CI {b['sign_accuracy']['ci95']}",
            f"  null: shuffled     {r['null_shuffled']:.4f}  "
            f"95% CI {b['shuffled']['ci95']}",
            f"  null: majority     {r['null_majority']:.4f}",
            f"median |magnitude err| {r['median_abs_magnitude_err_s'] * 1000:.1f} ms",
            "",
            verdict,
            f"failures: {len(r['failures'])}",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Directional onset-timing baseline (#108/#148)"
    )
    ap.add_argument("--asap-root", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--audio-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    result = run(args.asap_root, limit=args.limit, audio_cache=args.audio_cache)
    print(_format(result))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
