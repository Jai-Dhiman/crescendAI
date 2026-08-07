# model/src/follower_eval/ood_eval.py
"""Take -> per-factor degradation table (issue #148).

#148's success criterion is "a per-factor degradation table computed through the
**unchanged** ``accuracy.py`` metric core, directly subtractable from Track A's
clean-audio baseline". "Unchanged" is the load-bearing word. If this module
reimplemented the metrics -- even faithfully -- the resulting numbers would not
be subtractable from Track A's, because nobody could prove the two definitions
had not drifted apart. A metric that is copied is a metric that will diverge.

So this module owns exactly one thing: the FACTOR TABLE. Every number in it is
produced by importing ``asap_eval`` and ``accuracy`` and calling them. The
imports below are the whole guarantee:

    from follower_eval.asap_eval import _beat_errors, _summarize, follow_window

Nothing here recomputes an error, a median, or a within-tolerance rate.

THE ARM ABSTRACTION
-------------------
A **take** is one performance recorded on N channels with ONE truth (#148's
central abstraction). An **arm** is one (take, channel) pair -- the same
performance and the same truth, a different note source. The degradation table
subtracts arms sharing a take, so the performance, the pianist, the score and
the truth all cancel and only the factor under test remains.

PROVING THE SHARING BEFORE REAL TAKES EXIST
-------------------------------------------
Real phone takes do not exist yet, so the wiring is exercised on the one factor
ASAP already supplies at two levels: the NOTE SOURCE. Every ASAP performance is
available both as clean performance MIDI and as MAESTRO audio -> Transkun, and
Track A has already published the paired result (median 0.000 -> 0.005 beats,
cold-start within-1-beat 0.9242 -> 0.9205, -0.37 pp). Running that same
comparison through this module must reproduce those numbers.

That reproduction is the test that matters: it demonstrates the take -> table
path end to end AND proves the metric core is genuinely shared, because a copied
metric would land somewhere else. When phone channels arrive they slot in as
additional levels of a ``channel`` factor with no change to the table code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# The metric core, imported UNCHANGED. Nothing below recomputes any of it.
from follower_eval.asap_eval import (
    AsapEvalError,
    aligned_pieces,
    follow_window,
    load_asap_clip,
)


class OodEvalError(RuntimeError):
    """Raised when an arm cannot be evaluated. Loud: an arm that silently
    vanished would make the subtraction unpaired without anyone noticing."""


@dataclass(frozen=True)
class ArmSpec:
    """One level of one factor: how to obtain the notes for a take."""

    factor: str  # e.g. "note_source", later "channel"
    level: str  # e.g. "midi", "audio", later "phone_near"
    audio_cache: Path | None  # None = ASAP performance MIDI


@dataclass(frozen=True)
class ArmResult:
    """One (take, arm) evaluation, straight from the shared metric core."""

    take_id: str
    factor: str
    level: str
    n_beats_eval: int
    median_abs_err_beats: float | None
    p90_abs_err_beats: float | None
    within_1beat_frac: float | None


def evaluate_arm(take_id: str, arm: ArmSpec, asap_root: Path) -> ArmResult:
    """Evaluate one arm through the shared core. ``follow_window`` runs the
    follower and hands its matches to ``_beat_errors`` / ``_summarize``; this
    function only relabels the result."""
    perf_notes, score_notes, bar_boundaries, truth = load_asap_clip(
        take_id, asap_root, audio_cache=arm.audio_cache
    )
    w = follow_window(
        perf_notes,
        score_notes,
        bar_boundaries,
        truth,
        t_start=0.0,
        window_sec=None,
        label=f"{arm.factor}={arm.level}",
    )
    return ArmResult(
        take_id=take_id,
        factor=arm.factor,
        level=arm.level,
        n_beats_eval=w.n_beats_eval,
        median_abs_err_beats=w.median_abs_err_beats,
        p90_abs_err_beats=w.p90_abs_err_beats,
        within_1beat_frac=w.within_1beat_frac,
    )


def paired_table(results: list[ArmResult], baseline_level: str) -> dict:
    """Per-factor degradation table over takes present in EVERY level.

    Only takes with a result for every level are used. An unpaired take would
    let the comparison drift on which performances each arm happened to cover,
    which is precisely the confound the take abstraction exists to remove --
    #148's hard requirement that channel conditions be identical between phases
    is the same idea one level up.
    """
    by_level: dict[str, dict[str, ArmResult]] = {}
    for r in results:
        by_level.setdefault(r.level, {})[r.take_id] = r
    if baseline_level not in by_level:
        raise OodEvalError(
            f"baseline level {baseline_level!r} absent; have {sorted(by_level)}"
        )

    common = set.intersection(*(set(v) for v in by_level.values()))
    if not common:
        raise OodEvalError(
            f"no take is present in every level {sorted(by_level)}; "
            f"the subtraction would be unpaired"
        )
    takes = sorted(common)

    def _usable(field: str) -> list[str]:
        """Takes carrying a non-None value for ``field`` at EVERY level.

        Being present at every level is not enough: a metric is None whenever a
        window evaluated zero beats (``asap_eval._summarize``), which is a
        plausible phone-channel outcome and therefore exactly what this module
        exists to measure. Averaging each level over whatever it happened to
        have non-None would compare different sets of takes and unpair the
        subtraction -- the confound the take abstraction exists to remove. It
        can also flip the sign: a take that follows on the clean channel and
        yields nothing on the phone channel leaves its (large) clean error in
        the baseline mean and its absent phone error out of the other, making
        the degraded channel read BETTER.
        """
        return [
            t
            for t in takes
            if all(getattr(by_level[lv][t], field) is not None for lv in by_level)
        ]

    def _mean(level: str, field: str, subset: list[str]) -> float | None:
        vals = [getattr(by_level[level][t], field) for t in subset]
        return round(sum(vals) / len(vals), 4) if vals else None

    med_takes = _usable("median_abs_err_beats")
    w1_takes = _usable("within_1beat_frac")

    base_med = _mean(baseline_level, "median_abs_err_beats", med_takes)
    base_w1 = _mean(baseline_level, "within_1beat_frac", w1_takes)

    rows = []
    for level in sorted(by_level):
        med = _mean(level, "median_abs_err_beats", med_takes)
        w1 = _mean(level, "within_1beat_frac", w1_takes)
        rows.append(
            {
                "level": level,
                # Per FIELD, not per level: every level's mean for a field is
                # over the same takes, so the row counts are shared by
                # construction rather than by coincidence.
                "n_takes_median": len(med_takes),
                "n_takes_within_1beat": len(w1_takes),
                "median_abs_err_beats": med,
                "within_1beat_frac": w1,
                "delta_median_beats": round(med - base_med, 4)
                if med is not None and base_med is not None
                else None,
                "delta_within_1beat_pp": round((w1 - base_w1) * 100, 2)
                if w1 is not None and base_w1 is not None
                else None,
            }
        )
    return {
        "baseline_level": baseline_level,
        "n_takes_paired": len(takes),
        "n_takes_dropped_unpaired": len({r.take_id for r in results} - common),
        # Paired takes that still carry no comparable value at some level. Counted
        # separately from the unpaired drop because they are a different failure:
        # the take was recorded on every arm and one arm produced nothing.
        "n_takes_dropped_null_median": len(takes) - len(med_takes),
        "n_takes_dropped_null_within_1beat": len(takes) - len(w1_takes),
        "rows": rows,
    }


def available_takes(
    asap_root: Path, audio_cache: Path, limit: int | None = None
) -> list[str]:
    """ASAP takes that can carry EVERY level of the note_source factor.

    Restricting up front is not the same as dropping failures afterwards. Only
    ~56 of ASAP's 500+ aligned performances have a MAESTRO-audio bundle on disk,
    so iterating all of them would attempt 900 arms that cannot exist and bury
    the real errors under hundreds of "no audio bundle" rows. A take that cannot
    supply every level was never a candidate for a PAIRED table.
    """
    from follower_eval.asap_audio import bundle_path

    aligned = aligned_pieces(asap_root / "asap_annotations.json")
    takes = [p for p in aligned if bundle_path(audio_cache, p).exists()]
    return takes[:limit] if limit else takes


def run(asap_root: Path, audio_cache: Path, limit: int | None = None) -> dict:
    """Run the note_source factor over ASAP, which is the factor available today.

    Only pieces with an audio bundle can carry the audio level; the rest are
    dropped from the PAIRED set and counted, never scored on one level only.
    """
    arms = [
        ArmSpec(factor="note_source", level="midi", audio_cache=None),
        ArmSpec(factor="note_source", level="audio", audio_cache=audio_cache),
    ]
    takes = available_takes(asap_root, audio_cache, limit=limit)
    if not takes:
        raise OodEvalError(
            f"no ASAP take has a bundle under {audio_cache}; build them first "
            f"with `python -m follower_eval.asap_audio`"
        )

    results: list[ArmResult] = []
    failures: list[dict] = []
    for take in takes:
        for arm in arms:
            try:
                results.append(evaluate_arm(take, arm, asap_root))
            except (AsapEvalError, OodEvalError) as exc:
                failures.append(
                    {
                        "take": take,
                        "level": arm.level,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    if not results:
        raise OodEvalError(f"no arm evaluated; {len(failures)} failures")

    return {
        "metric_core": (
            "follower_eval.asap_eval.follow_window / _beat_errors / _summarize, "
            "IMPORTED UNCHANGED. No metric is recomputed in ood_eval."
        ),
        "factor": "note_source",
        "table": paired_table(results, baseline_level="midi"),
        "failures": failures,
        "arms": [asdict(r) for r in results],
    }


def _format(r: dict) -> str:
    t = r["table"]
    lines = [
        "=" * 78,
        f"OOD PER-FACTOR TABLE (#148) -- factor: {r['factor']}",
        "=" * 78,
        f"metric core: {r['metric_core']}",
        f"paired takes: {t['n_takes_paired']}   "
        f"dropped unpaired: {t['n_takes_dropped_unpaired']}   "
        f"dropped null (median/within-1): {t['n_takes_dropped_null_median']}/"
        f"{t['n_takes_dropped_null_within_1beat']}   "
        f"failures: {len(r['failures'])}",
        "",
        f"{'level':<16}{'median beats':>14}{'d median':>11}"
        f"{'within-1beat':>14}{'d pp':>8}",
        "-" * 63,
    ]
    for row in t["rows"]:
        dm = (
            "--"
            if row["delta_median_beats"] is None
            else f"{row['delta_median_beats']:+.4f}"
        )
        dp = (
            "--"
            if row["delta_within_1beat_pp"] is None
            else f"{row['delta_within_1beat_pp']:+.2f}"
        )
        # Guarded like the deltas above: these are None whenever no paired take
        # carries the field, and a report that crashes tells you less than one
        # that prints the gap.
        med = (
            "--"
            if row["median_abs_err_beats"] is None
            else f"{row['median_abs_err_beats']:.4f}"
        )
        w1 = (
            "--"
            if row["within_1beat_frac"] is None
            else f"{row['within_1beat_frac']:.4f}"
        )
        lines.append(f"{row['level']:<16}{med:>14}{dm:>11}{w1:>14}{dp:>8}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="OOD per-factor degradation table (#148)")
    ap.add_argument("--asap-root", type=Path, required=True)
    ap.add_argument("--audio-cache", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    result = run(args.asap_root, args.audio_cache, limit=args.limit)
    print(_format(result))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
