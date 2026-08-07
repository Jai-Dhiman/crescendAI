# model/src/follower_eval/note_correspondence.py
"""Clean-audio baseline for PER-NOTE correspondence (issues #108, #148).

#108's remaining scope is two deliverables it never measured: per-note
correspondence precision/recall, and the directional onset-timing signal. Both
were stated as blocked on #148, but only the *out-of-distribution* half is --
the clean-audio baseline that #148's per-factor table must be subtracted from
does not exist for either, and can be measured today. This module is the first
of the two.

Track A's existing clean baseline covers score POSITION only (cold-start
within-1-beat 0.9205, median 0.005 beats). Position accuracy does not imply
correspondence accuracy: a follower can sit at the right score time while
pairing performed notes with the wrong score notes, and the product features
that need "which note did you play" -- wrong-note flagging, per-note timing
feedback -- depend on the pairing, not the position.

THE TRUTH IS NOT THE SYSTEM UNDER TEST
--------------------------------------
System under test: the #119 Viterbi HMM (``follower_bench.hmm.follow_hmm``),
which emits ``(perf_index, score_index)`` pairs directly.

Truth: derived from **ASAP's human-verified beat alignment** -- per performance,
a hand-checked pairing of performance-time beats to score-time beats -- and
never from any aligner. This is the lesson #101's gate1 carried: its anchors
were parangonar output scored against parangonar, so "residual-vs-anchors =
agreement, not accuracy". Nothing here consults the HMM, parangonar, or any
other matcher when building truth.

TRUTH PROVENANCE, STATED EXACTLY
--------------------------------
The derivation is two steps, and only the first is human-verified:

  1. **Human-verified (ASAP).** Each performed note's onset is mapped to a score
     time by piecewise-linear interpolation over ASAP's paired beat anchors.
     The anchors are hand-checked by ASAP's authors; the interpolation BETWEEN
     anchors assumes locally constant tempo, which is ours, not theirs.
  2. **Deterministic rule (ours).** The true score note is the same-pitch score
     note nearest that interpolated time, if one lies within ``TOL_BEATS``.

So the honest label is "ASAP human-verified beat alignment + same-pitch
nearest-neighbour assignment within tolerance", not "human-verified note
correspondence". Two consequences are measured rather than assumed:

  * A performed note with no same-pitch score note in tolerance has **no true
    correspondent**. That is a real category (wrong notes, ornaments, repeats
    the score does not contain), and the follower is not charged for pairing it.
  * When two same-pitch score notes sit within tolerance and within
    ``AMBIGUITY_BEATS`` of each other, the rule cannot decide, so the note is
    **excluded and counted**. The excluded fraction is reported alongside every
    result; a baseline that quietly dropped its hard cases would flatter itself.

BOOTSTRAP OVER PERFORMANCES, NOT NOTES
--------------------------------------
Notes within one performance share a tempo, a pianist and a score, so they are
nowhere near independent. Resampling notes would give an interval far too tight.
The bootstrap resamples PERFORMANCES (a cluster bootstrap), which is the honest
unit of independence here.

RUNNING (from the PRIMARY checkout -- data/ is gitignored in worktrees):

  cd /Users/jdhiman/Documents/crescendai/model
  PYTHONPATH=<worktree>/model/src .venv/bin/python \
    -m follower_eval.note_correspondence \
    --asap-root data/raw/asap --limit 55 --out /tmp/note_corr.json
"""

from __future__ import annotations

import argparse
import json
import random
from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path

from follower_bench.hmm import TUNED_HMM_PARAMS, follow_hmm

from follower_eval.asap_eval import AsapEvalError, aligned_pieces, load_asap_clip

# A candidate score note must lie within this many beats of the interpolated
# score time to be the note's truth. Wide enough to absorb the local-tempo
# assumption in the interpolation, narrow enough that it does not reach a
# different bar.
TOL_BEATS = 0.5

# Two same-pitch candidates closer together than this are indistinguishable to
# the rule; the note is excluded and counted rather than assigned by a coin flip.
AMBIGUITY_BEATS = 0.25

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 148


class NoteCorrespondenceError(RuntimeError):
    """Raised when a performance cannot be scored as specified -- loud, never a
    silent skip that would let the corpus shrink without anyone noticing."""


@dataclass(frozen=True)
class ClipCorrespondence:
    """One performance's correspondence result against derived truth."""

    asap_piece: str
    n_perf_notes: int
    n_truth_pairs: int  # perf notes WITH a true score note
    n_no_truth: int  # perf notes with no same-pitch score note in tolerance
    n_ambiguous: int  # excluded: two candidates too close to separate
    n_predicted: int  # follower pairs on notes that have truth
    n_correct: int
    precision: float | None
    recall: float | None
    f1: float | None


def _interp(xs: tuple[float, ...], ys: tuple[float, ...], x: float) -> float:
    """Piecewise-linear interpolation, clamped at both ends. Written out rather
    than pulled from numpy so the clamping behavior at the edges is explicit:
    notes before the first beat anchor or after the last are mapped to the
    endpoint rather than extrapolated into invented score time."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect_left(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _local_beat_sec(truth, score_sec: float) -> float:
    """Seconds per beat near a score time, from ASAP's own beat spacing."""
    i = min(
        range(len(truth.score_secs)), key=lambda k: abs(truth.score_secs[k] - score_sec)
    )
    sp = truth.beat_spacing[i]
    if sp > 0:
        return sp
    positive = [s for s in truth.beat_spacing if s > 0]
    if not positive:
        raise NoteCorrespondenceError("ASAP alignment has no positive beat spacing")
    return sorted(positive)[len(positive) // 2]


def derive_truth(perf_notes, score_notes, truth) -> tuple[dict[int, int], int, int]:
    """(perf_index -> true score_index, n_no_truth, n_ambiguous).

    Truth comes from ASAP's beat anchors plus the same-pitch nearest-neighbour
    rule; see the module docstring for the provenance split. The follower is
    never consulted.
    """
    by_pitch: dict[int, list[tuple[float, int]]] = {}
    for j, s in enumerate(score_notes):
        by_pitch.setdefault(s.pitch, []).append((s.position, j))
    for v in by_pitch.values():
        v.sort()

    pairs: dict[int, int] = {}
    n_no_truth = n_ambiguous = 0
    for i, p in enumerate(perf_notes):
        s_true = _interp(truth.perf_times, truth.score_secs, p.onset)
        beat_sec = _local_beat_sec(truth, s_true)
        tol = TOL_BEATS * beat_sec

        cands = [(abs(pos - s_true), pos, j) for pos, j in by_pitch.get(p.pitch, [])]
        cands = [c for c in cands if c[0] <= tol]
        if not cands:
            n_no_truth += 1
            continue
        cands.sort()
        if (
            len(cands) > 1
            and abs(cands[1][1] - cands[0][1]) < AMBIGUITY_BEATS * beat_sec
        ):
            n_ambiguous += 1
            continue
        pairs[i] = cands[0][2]
    return pairs, n_no_truth, n_ambiguous


def score_clip(
    asap_piece: str, asap_root: Path, audio_cache: Path | None = None
) -> ClipCorrespondence:
    """Follow one ASAP performance and score its note pairing against truth."""
    perf_notes, score_notes, bar_boundaries, truth = load_asap_clip(
        asap_piece, asap_root, audio_cache=audio_cache
    )
    pairs, n_no_truth, n_ambiguous = derive_truth(perf_notes, score_notes, truth)

    est = follow_hmm(
        perf_notes, score_notes, TUNED_HMM_PARAMS, bar_boundaries=bar_boundaries
    )
    predicted = {m.perf_index: m.score_index for m in est.matches}

    # Only notes that HAVE truth can be scored either way. A follower pairing a
    # note the score does not contain is a different error class (it belongs to
    # the no-truth count) and is not charged as a precision miss here.
    scorable = {i: j for i, j in predicted.items() if i in pairs}
    n_correct = sum(1 for i, j in scorable.items() if pairs[i] == j)

    precision = n_correct / len(scorable) if scorable else None
    recall = n_correct / len(pairs) if pairs else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else (0.0 if precision is not None and recall is not None else None)
    )
    return ClipCorrespondence(
        asap_piece=asap_piece,
        n_perf_notes=len(perf_notes),
        n_truth_pairs=len(pairs),
        n_no_truth=n_no_truth,
        n_ambiguous=n_ambiguous,
        n_predicted=len(scorable),
        n_correct=n_correct,
        precision=round(precision, 4) if precision is not None else None,
        recall=round(recall, 4) if recall is not None else None,
        f1=round(f1, 4) if f1 is not None else None,
    )


def _pooled(
    clips: list[ClipCorrespondence],
) -> tuple[float | None, float | None, float | None]:
    """Note-pooled precision/recall/F1 over a set of clips."""
    correct = sum(c.n_correct for c in clips)
    pred = sum(c.n_predicted for c in clips)
    tru = sum(c.n_truth_pairs for c in clips)
    p = correct / pred if pred else None
    r = correct / tru if tru else None
    f = (
        2 * p * r / (p + r)
        if p and r
        else (0.0 if p is not None and r is not None else None)
    )
    return p, r, f


def bootstrap_ci(
    clips: list[ClipCorrespondence], n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> dict:
    """Cluster bootstrap 95% CI, resampling PERFORMANCES.

    Notes inside one performance share a pianist, a tempo and a score, so a
    note-level bootstrap would treat thousands of correlated observations as
    independent and return an interval far too tight. The performance is the
    unit that varies.
    """
    if not clips:
        raise NoteCorrespondenceError("no clips to bootstrap")
    rng = random.Random(seed)
    draws = {"precision": [], "recall": [], "f1": []}
    for _ in range(n):
        sample = [clips[rng.randrange(len(clips))] for _ in clips]
        p, r, f = _pooled(sample)
        if p is None or r is None:
            continue
        draws["precision"].append(p)
        draws["recall"].append(r)
        draws["f1"].append(f)

    out = {}
    for key, vals in draws.items():
        vals.sort()
        lo = vals[int(0.025 * (len(vals) - 1))]
        hi = vals[int(0.975 * (len(vals) - 1))]
        out[key] = {"ci95": [round(lo, 4), round(hi, 4)], "n_draws": len(vals)}
    return out


def run(
    asap_root: Path, limit: int | None = None, audio_cache: Path | None = None
) -> dict:
    pieces = aligned_pieces(asap_root / "asap_annotations.json", limit=limit)
    clips: list[ClipCorrespondence] = []
    failures: list[dict] = []
    for piece in pieces:
        try:
            clips.append(score_clip(piece, asap_root, audio_cache=audio_cache))
        except (AsapEvalError, NoteCorrespondenceError) as exc:
            failures.append({"piece": piece, "error": f"{type(exc).__name__}: {exc}"})
    if not clips:
        raise NoteCorrespondenceError(f"no clip scored; {len(failures)} failures")

    p, r, f = _pooled(clips)
    n_notes = sum(c.n_perf_notes for c in clips)
    return {
        "system_under_test": "follower_bench.hmm.follow_hmm (#119) + TUNED_HMM_PARAMS",
        "truth_provenance": (
            "ASAP human-verified beat alignment (performance_beats <-> "
            "midi_score_beats), piecewise-linearly interpolated to each performed "
            "note's onset, then assigned to the same-pitch score note nearest that "
            "time within TOL_BEATS. The beat anchors are human-verified; the "
            "interpolation and the nearest-same-pitch rule are ours. NOT produced "
            "by any aligner, and specifically not by the system under test."
        ),
        "note_source": "ASAP performance MIDI"
        if audio_cache is None
        else "MAESTRO audio -> Transkun",
        "params": {"tol_beats": TOL_BEATS, "ambiguity_beats": AMBIGUITY_BEATS},
        "n_clips": len(clips),
        "n_perf_notes": n_notes,
        "n_truth_pairs": sum(c.n_truth_pairs for c in clips),
        "n_no_truth": sum(c.n_no_truth for c in clips),
        "n_ambiguous_excluded": sum(c.n_ambiguous for c in clips),
        "ambiguous_frac": round(sum(c.n_ambiguous for c in clips) / n_notes, 4)
        if n_notes
        else None,
        "no_truth_frac": round(sum(c.n_no_truth for c in clips) / n_notes, 4)
        if n_notes
        else None,
        "precision": round(p, 4) if p is not None else None,
        "recall": round(r, 4) if r is not None else None,
        "f1": round(f, 4) if f is not None else None,
        "bootstrap": bootstrap_ci(clips),
        "failures": failures,
        "clips": [asdict(c) for c in clips],
    }


def _format(result: dict) -> str:
    b = result["bootstrap"]
    return "\n".join(
        [
            "=" * 78,
            "PER-NOTE CORRESPONDENCE -- CLEAN-AUDIO BASELINE (#108 / #148)",
            "=" * 78,
            f"system under test : {result['system_under_test']}",
            f"note source       : {result['note_source']}",
            f"clips {result['n_clips']}   perf notes {result['n_perf_notes']:,}   "
            f"truth pairs {result['n_truth_pairs']:,}",
            f"no-truth notes    : {result['n_no_truth']:,} "
            f"({result['no_truth_frac']:.1%})",
            f"ambiguous, excluded: {result['n_ambiguous_excluded']:,} "
            f"({result['ambiguous_frac']:.1%})",
            "",
            f"precision {result['precision']:.4f}  95% CI {b['precision']['ci95']}",
            f"recall    {result['recall']:.4f}  95% CI {b['recall']['ci95']}",
            f"F1        {result['f1']:.4f}  95% CI {b['f1']['ci95']}",
            "",
            f"failures: {len(result['failures'])}",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-note correspondence baseline (#108/#148)"
    )
    ap.add_argument("--asap-root", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--audio-cache",
        type=Path,
        default=None,
        help="score through MAESTRO audio -> Transkun bundles instead of ASAP MIDI",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    result = run(args.asap_root, limit=args.limit, audio_cache=args.audio_cache)
    print(_format(result))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
