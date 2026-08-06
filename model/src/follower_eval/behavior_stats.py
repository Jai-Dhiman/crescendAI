# model/src/follower_eval/behavior_stats.py
"""Behavior statistics for G-OOD-6 representativeness (issue #148).

G-OOD-6 asks whether the simulated practice takes recorded in #148's Phase 2
resemble the amateur practice population the product actually serves. The gate
is ">= 4 of 6 behavior statistics with simulated median inside the 279-clip
corpus IQR". This module defines those six statistics and computes the corpus
side of the comparison.

WHY THESE SIX, AND WHY THEY LOOK LIKE THIS
------------------------------------------
The gate's binding constraint is that a statistic must be computable with **no
truth on either side** -- the 279 amateur YouTube clips have no position truth
and never will, and the simulated takes must be judged before their truth
alignment exists. That rules out anything score-relative.

It also rules out, less obviously, every proxy in ``realaudio.py`` (coverage,
score_span_frac, backward_frac, confidence). Those are *follower outputs*. If a
simulated take matched the corpus on ``backward_frac``, the demonstrated fact
would be that the follower reacts similarly to both -- not that the playing is
similar. Representativeness must be established upstream of the system under
test, or it is circular in exactly the way #101's gate1 anchors were
("residual-vs-anchors = agreement, not accuracy").

So all six statistics are functions of the transcribed note stream alone:
no score, no alignment, no follower. Both sides of the comparison run the same
production transcriber (Transkun, #128), so transcription noise is a shared
constant rather than a confound between arms.

THE SIX STATISTICS
------------------
Two describe session shape, two describe stop-start behavior, one describes
repetition, one describes within-phrase timing steadiness:

  1. ``active_duration_s``    -- last onset - first onset. How long a take runs.
  2. ``note_rate_per_min``    -- note events per minute of active duration.
                                 Tempo x density; separates a slow beginner
                                 take from a fast one of the same length.
  3. ``pause_rate_per_min``   -- events/min where the inter-onset gap is
                                 >= PAUSE_SEC. The hesitation/stop rate.
  4. ``longest_pause_s``      -- the single longest inter-onset gap. Distinguishes
                                 "paused briefly, often" from "stopped dead once".
  5. ``repeat_event_frac``    -- fraction of chord-events belonging to a pitch
                                 n-gram that occurs more than once in the clip.
                                 The truth-free signature of repeats and restarts.
  6. ``local_tempo_jitter``   -- median |log2 (IOI / local geometric-mean IOI)| over
                                 *continuous* stretches only (IOIs < PAUSE_SEC).
                                 Unsteadiness within a phrase, with the macro
                                 stop-start structure of (3)/(4) excluded.

WHAT THIS DELIBERATELY CANNOT COVER
-----------------------------------
**Wrong-note rate is absent, and cannot be added.** Detecting a wrong note
requires the score, which is truth. G-OOD-6 therefore bounds session shape,
stopping, repetition, and steadiness -- not note accuracy. If the recorded takes
are cleaner than the corpus in note accuracy, this gate will not detect it, and
that must be stated as a limitation rather than left implied.

CHORD HANDLING
--------------
Piano audio produces near-simultaneous notes. All timing statistics run over
*chord events* (onsets collapsed within CHORD_SEC), not raw notes, so a
four-note chord is one event and does not read as a zero-IOI flurry.

RUNNING (from the PRIMARY checkout -- data/ is gitignored, absent in worktrees):

  cd /Users/jdhiman/Documents/crescendai/model
  PYTHONPATH=<worktree>/model/src .venv/bin/python -m follower_eval.behavior_stats \
    --bundles-root data/evals/realaudio_bundles \
    --out src/follower_eval/corpus_behavior_iqr.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from paths import DATA_ROOT

# Onsets within this many seconds are one chord event. 50 ms is wider than
# Transkun's onset jitter within a struck chord and narrower than any played
# interval at plausible tempi.
CHORD_SEC = 0.05

# An inter-onset gap at or above this is a pause, not playing. 2.0 s is long
# enough that ordinary rubato and long notes do not trip it, short enough to
# catch a beginner stopping to find the next chord.
PAUSE_SEC = 2.0

# Chord-event n-gram length for repeat detection. 4 is a compromise: long enough
# that ordinary harmonic coincidence does not register as a repeat, short enough
# to survive the wrong notes an amateur puts inside a repeated passage.
REPEAT_NGRAM = 4

# Window over which the "local" median IOI is taken for jitter, in events.
JITTER_WINDOW = 9

# Per-statistic minimum evidence. A clip below the floor contributes None to
# that statistic and is counted in ``n_missing``, never silently as a zero.
MIN_EVENTS_TIMING = 8  # duration / rate / pause statistics
MIN_EVENTS_REPEAT = 2 * REPEAT_NGRAM  # a repeat needs room for two occurrences
MIN_IOIS_JITTER = JITTER_WINDOW  # one full local window of continuous playing

STATISTIC_NAMES = (
    "active_duration_s",
    "note_rate_per_min",
    "pause_rate_per_min",
    "longest_pause_s",
    "repeat_event_frac",
    "local_tempo_jitter",
)


class BehaviorStatsError(RuntimeError):
    """Raised when a bundle cannot be read as required. Loud: a clip that fails
    to load is reported, never dropped into the denominator as a zero."""


@dataclass(frozen=True)
class ClipBehavior:
    """One clip's six statistics. A field is None when the clip lacks the
    evidence that statistic needs (see the MIN_* floors) -- missing, not zero."""

    piece: str
    clip: str
    n_notes: int
    n_events: int
    active_duration_s: float | None
    note_rate_per_min: float | None
    pause_rate_per_min: float | None
    longest_pause_s: float | None
    repeat_event_frac: float | None
    local_tempo_jitter: float | None


def chord_events(
    onsets_pitches: list[tuple[float, int]],
) -> list[tuple[float, tuple[int, ...]]]:
    """Collapse (onset, pitch) pairs into chord events: (event_onset, pitches).

    Input need not be sorted. Events are greedy: an event absorbs every note
    within CHORD_SEC of the event's FIRST onset, so a rolled chord spanning more
    than CHORD_SEC correctly becomes more than one event rather than an
    unbounded chain.
    """
    if not onsets_pitches:
        return []
    ordered = sorted(onsets_pitches)
    events: list[tuple[float, list[int]]] = []
    start, pitches = ordered[0][0], [ordered[0][1]]
    for onset, pitch in ordered[1:]:
        if onset - start <= CHORD_SEC:
            pitches.append(pitch)
        else:
            events.append((start, pitches))
            start, pitches = onset, [pitch]
    events.append((start, pitches))
    return [(t, tuple(sorted(p))) for t, p in events]


def _iois(event_times: list[float]) -> list[float]:
    return [event_times[i] - event_times[i - 1] for i in range(1, len(event_times))]


def repeat_event_frac(pitch_sets: list[tuple[int, ...]]) -> float | None:
    """Fraction of chord events that sit inside a REPEAT_NGRAM-gram of pitch
    sets occurring more than once in the clip.

    Truth-free proxy for repeats and restarts: playing a passage twice emits the
    same pitch-set sequence twice, whatever the tempo or the timing. Insensitive
    to *where* in the score the repeat is, which is the point -- no score is
    consulted.

    Amateur wrong notes inside a repeat break the exact match, so this
    UNDERSTATES repetition. It understates it on both sides of the comparison,
    which is what the gate requires; it is not a calibrated repeat count.
    """
    n = len(pitch_sets)
    if n < MIN_EVENTS_REPEAT:
        return None
    grams: dict[tuple[tuple[int, ...], ...], list[int]] = {}
    for i in range(n - REPEAT_NGRAM + 1):
        grams.setdefault(tuple(pitch_sets[i : i + REPEAT_NGRAM]), []).append(i)
    covered: set[int] = set()
    for starts in grams.values():
        if len(starts) > 1:
            for s in starts:
                covered.update(range(s, s + REPEAT_NGRAM))
    return len(covered) / n


def local_tempo_jitter(event_times: list[float]) -> float | None:
    """Median |log2(IOI / local geometric-mean IOI)| over continuous playing.

    Continuous means IOIs below PAUSE_SEC: the macro stop-start structure is
    already carried by pause_rate_per_min and longest_pause_s, and leaving it in
    here would make the two pairs redundant. log2 makes the measure
    tempo-invariant and symmetric (twice as fast and half as fast both read
    1.0), so a steady slow take and a steady fast take score the same.

    The local reference is taken over a JITTER_WINDOW-event centred window, so a
    deliberate ritardando is absorbed as local tempo rather than charged as
    jitter; only beat-to-beat unevenness registers.

    The reference is the GEOMETRIC mean, not the median. A median has a blind
    spot for strictly alternating long-short IOIs -- the window majority
    alternates with the values, every IOI equals its own reference, and one of
    the commonest amateur patterns (uneven hands, dotted-rhythm drift) reads as
    perfectly steady. Measured on constructed clips: median scores an
    alternating 0.65/0.35 clip at 0.000, the geometric mean at 0.397, and both
    still score a metronomic clip and a smooth ritardando at 0.000. The
    geometric mean is also the natural centre for a log-ratio deviation.
    """
    continuous = [d for d in _iois(event_times) if 0 < d < PAUSE_SEC]
    if len(continuous) < MIN_IOIS_JITTER:
        return None
    half = JITTER_WINDOW // 2
    devs = []
    for i, d in enumerate(continuous):
        lo, hi = max(0, i - half), min(len(continuous), i + half + 1)
        window = continuous[lo:hi]
        local = math.exp(sum(math.log(x) for x in window) / len(window))
        devs.append(abs(math.log2(d / local)))
    return statistics.median(devs) if devs else None


def clip_behavior(piece: str, clip: str, notes: list[dict]) -> ClipBehavior:
    """Compute the six statistics for one transcribed clip's note list."""
    events = chord_events([(float(n["onset"]), int(n["pitch"])) for n in notes])
    times = [t for t, _ in events]
    pitch_sets = [p for _, p in events]
    n_events = len(events)

    duration = rate = pause_rate = longest = None
    if n_events >= MIN_EVENTS_TIMING:
        duration = times[-1] - times[0]
        iois = _iois(times)
        longest = max(iois)
        if duration > 0:
            per_min = 60.0 / duration
            rate = n_events * per_min
            pause_rate = sum(1 for d in iois if d >= PAUSE_SEC) * per_min

    return ClipBehavior(
        piece=piece,
        clip=clip,
        n_notes=len(notes),
        n_events=n_events,
        active_duration_s=duration,
        note_rate_per_min=rate,
        pause_rate_per_min=pause_rate,
        longest_pause_s=longest,
        repeat_event_frac=repeat_event_frac(pitch_sets),
        local_tempo_jitter=local_tempo_jitter(times),
    )


def corpus_clips(bundles_root: Path) -> list[tuple[str, str, Path]]:
    """The 279-clip corpus, as (piece, video_id, bundle_path), sorted.

    The corpus is defined by the BUILD MANIFESTS, not by a directory glob.
    ``data/evals/realaudio_bundles/`` holds more bundle files than the corpus
    (366 at time of writing) because earlier and partial builds left artifacts
    behind; globbing the directory silently yields a wrong denominator and
    therefore wrong quantiles. A manifest row with status ``ok`` or ``skip`` is
    a corpus member; ``download_fail`` / ``transcribe_fail`` / ``empty`` are not.

    Raises:
        BehaviorStatsError: no manifests found, or a manifest names a bundle
            that is not on disk (a corpus member we cannot measure is an error,
            not a quiet 278).
    """
    manifests = sorted(bundles_root.glob("_*manifest*.json"))
    if not manifests:
        raise BehaviorStatsError(f"no _*manifest*.json under {bundles_root}")

    members: dict[tuple[str, str], Path] = {}
    for m in manifests:
        for row in json.loads(m.read_text()).get("outcomes", []):
            if row.get("status") not in ("ok", "skip"):
                continue
            piece, vid = row["piece"], row["video_id"]
            members[(piece, vid)] = bundles_root / piece / f"{vid}.json"

    missing = sorted(
        f"{p}/{v}" for (p, v), path in members.items() if not path.exists()
    )
    if missing:
        raise BehaviorStatsError(
            f"{len(missing)} corpus bundle(s) named by a manifest are absent from "
            f"disk: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return [(p, v, members[(p, v)]) for p, v in sorted(members)]


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy 'linear' / R type 7). Written out so
    the committed numbers do not move with a numpy version bump."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def summarize(behaviors: list[ClipBehavior]) -> dict:
    """Per-statistic median, IQR and n over the corpus.

    n is reported PER STATISTIC, with ``n_missing`` naming how many corpus clips
    lacked the evidence for it. A statistic's denominator is its own n, never
    the corpus total -- reporting 279 for a statistic measured on 240 clips is
    the mistake #133 made once already with gold_subset rates.
    """
    out: dict[str, dict] = {}
    total = len(behaviors)
    for name in STATISTIC_NAMES:
        vals = sorted(v for v in (getattr(b, name) for b in behaviors) if v is not None)
        if not vals:
            out[name] = {
                "n": 0,
                "n_missing": total,
                "median": None,
                "q1": None,
                "q3": None,
                "iqr": None,
                "p10": None,
                "p90": None,
            }
            continue
        q1, q3 = _quantile(vals, 0.25), _quantile(vals, 0.75)
        out[name] = {
            "n": len(vals),
            "n_missing": total - len(vals),
            "median": round(_quantile(vals, 0.5), 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": [round(q1, 4), round(q3, 4)],
            "p10": round(_quantile(vals, 0.10), 4),
            "p90": round(_quantile(vals, 0.90), 4),
        }
    return out


def inside_iqr(name: str, simulated_median: float, corpus: dict) -> bool:
    """G-OOD-6's per-statistic test: is the simulated median inside the corpus
    IQR (inclusive of both bounds)?"""
    band = corpus[name]["iqr"]
    if band is None:
        raise BehaviorStatsError(f"corpus statistic {name!r} has no IQR (n=0)")
    return band[0] <= simulated_median <= band[1]


def score_arm(behaviors: list[ClipBehavior], corpus: dict) -> dict:
    """Run one arm (a set of takes/clips) through G-OOD-6 against the corpus
    reference. Returns per-statistic medians, inside/outside, and the pass count
    the gate's ">= 4 of 6" bar is read against."""
    arm = summarize(behaviors)
    per_stat = {}
    for name in STATISTIC_NAMES:
        med = arm[name]["median"]
        per_stat[name] = {
            "arm_median": med,
            "arm_n": arm[name]["n"],
            "corpus_iqr": corpus["statistics"][name]["iqr"],
            "inside": None
            if med is None
            else inside_iqr(name, med, corpus["statistics"]),
        }
    n_inside = sum(1 for s in per_stat.values() if s["inside"])
    return {
        "n_clips": len(behaviors),
        "per_statistic": per_stat,
        "n_inside": n_inside,
        "passes_bar": n_inside >= 4,
    }


def separation_auc(
    corpus_behaviors: list[ClipBehavior], control_behaviors: list[ClipBehavior]
) -> dict[str, float | None]:
    """Per-statistic AUC of corpus-vs-control separation (P[corpus > control],
    ties at 0.5). 0.5 = the statistic cannot tell the two populations apart.

    This exists because G-OOD-6's "median inside the corpus IQR" test is only
    evidence of representativeness if a NON-representative arm would fail it.
    An IQR is the middle 50% of a heterogeneous population -- a wide target --
    so the gate can be passed by an arm with no behavioral resemblance at all.
    Measured, not hypothesized: the 56 ASAP/MAESTRO competition performances
    (professional, linear, zero practice behavior) pass 5 of 6 against a bar of
    4, with AUC 0.42-0.58 on five of the six statistics. See ``main --control``.
    """
    out: dict[str, float | None] = {}
    for name in STATISTIC_NAMES:
        a = [v for v in (getattr(b, name) for b in corpus_behaviors) if v is not None]
        b = [v for v in (getattr(x, name) for x in control_behaviors) if v is not None]
        if not a or not b:
            out[name] = None
            continue
        wins = sum((1.0 if x > y else 0.5 if x == y else 0.0) for x in a for y in b)
        out[name] = round(wins / (len(a) * len(b)), 4)
    return out


def run(bundles_root: Path) -> dict:
    """Compute the corpus-side G-OOD-6 reference. Deterministic: the clip list
    comes from the manifests in sorted order and every statistic is a pure
    function of the bundle's notes."""
    clips = corpus_clips(bundles_root)
    behaviors: list[ClipBehavior] = []
    for piece, vid, path in clips:
        notes = json.loads(path.read_text()).get("notes")
        if not notes:
            raise BehaviorStatsError(f"{piece}/{vid}: bundle has no 'notes'")
        behaviors.append(clip_behavior(piece, vid, notes))
    return {
        "corpus": "realaudio_bundles 279-clip amateur practice corpus (#133)",
        "corpus_definition": "union of build-manifest rows with status ok|skip",
        "gate": "G-OOD-6 (#148): >=4 of 6 simulated medians inside the corpus IQR",
        "transcriber": "transkun",
        "params": {
            "chord_sec": CHORD_SEC,
            "pause_sec": PAUSE_SEC,
            "repeat_ngram": REPEAT_NGRAM,
            "jitter_window": JITTER_WINDOW,
            "min_events_timing": MIN_EVENTS_TIMING,
            "min_events_repeat": MIN_EVENTS_REPEAT,
            "min_iois_jitter": MIN_IOIS_JITTER,
        },
        "n_clips": len(behaviors),
        "statistics": summarize(behaviors),
        "per_clip": [b.__dict__ for b in behaviors],
    }


def _format_report(result: dict) -> str:
    lines = [
        "=" * 78,
        "G-OOD-6 CORPUS BEHAVIOR REFERENCE (#148)",
        "=" * 78,
        f"corpus clips: {result['n_clips']}   ({result['corpus_definition']})",
        "",
        f"{'statistic':<22}{'n':>5}{'miss':>6}{'median':>11}"
        f"{'IQR lo':>11}{'IQR hi':>11}",
        "-" * 66,
    ]
    for name, s in result["statistics"].items():
        if s["median"] is None:
            lines.append(f"{name:<22}{s['n']:>5}{s['n_missing']:>6}{'--':>11}")
            continue
        lines.append(
            f"{name:<22}{s['n']:>5}{s['n_missing']:>6}"
            f"{s['median']:>11.3f}{s['q1']:>11.3f}{s['q3']:>11.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="G-OOD-6 corpus behavior statistics (#148)"
    )
    ap.add_argument(
        "--bundles-root",
        type=Path,
        default=DATA_ROOT / "evals" / "realaudio_bundles",
        help="dir of <piece>/<video_id>.json bundles plus their build manifests",
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="write the JSON reference here"
    )
    ap.add_argument(
        "--no-per-clip",
        action="store_true",
        help="omit the per-clip rows from --out (summary only)",
    )
    ap.add_argument(
        "--control",
        type=Path,
        default=None,
        help="dir of flat <name>.json bundles to score as an arm against the "
        "corpus (negative control: run the ASAP competition bundles here and "
        "confirm they FAIL before trusting a pass)",
    )
    args = ap.parse_args()

    result = run(args.bundles_root)
    print(_format_report(result))

    if args.control:
        controls = []
        for path in sorted(args.control.glob("*.json")):
            notes = json.loads(path.read_text()).get("notes")
            if not notes:
                raise BehaviorStatsError(f"{path.name}: control bundle has no 'notes'")
            controls.append(clip_behavior("control", path.stem, notes))
        arm = score_arm(controls, result)
        auc = separation_auc([ClipBehavior(**c) for c in result["per_clip"]], controls)
        print(f"\nCONTROL ARM ({args.control.name}, n={len(controls)})")
        print(f"{'statistic':<22}{'arm median':>12}{'inside':>8}{'AUC':>7}")
        for name, s in arm["per_statistic"].items():
            med = "--" if s["arm_median"] is None else f"{s['arm_median']:.3f}"
            a = "--" if auc[name] is None else f"{auc[name]:.2f}"
            print(f"{name:<22}{med:>12}{str(s['inside']):>8}{a:>7}")
        verdict = (
            "PASSES -- gate does NOT discriminate"
            if arm["passes_bar"]
            else "FAILS -- gate discriminates"
        )
        print(f"control passes {arm['n_inside']} of 6 (bar >=4) -> {verdict}")
    if args.out:
        body = dict(result)
        if args.no_per_clip:
            body.pop("per_clip")
        args.out.write_text(json.dumps(body, indent=1) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
