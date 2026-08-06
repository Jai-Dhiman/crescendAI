# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.24.0"]
# ///
"""FRONT 10 (#101): calibrate the articulation statistic, its tau, and its error bar off
the measured Transkun offset tail on real MAESTRO audio.

FRONT 9 Finding 4 ungated articulation with a qualification: Transkun's note-OFFSET
error has a median of 9.4ms but a **p90 of 90ms** (releases are acoustically ambiguous
-- intrinsically ~3x onset noise for any transcriber), and the offset-derived
articulation statistic tracked GT at corr 0.876. It left the measurer unrouted and the
tau uncalibrated. This script supplies both.

Two things get decided here, in order.

1. **STATISTIC CONDITIONING.** FRONT 9's probe statistic is median(note_duration / IOI)
over all
   note pairs with IOI > 1ms. That denominator is pathological on piano: notes inside a
   CHORD are
   near-simultaneous, so IOI -> 0 and the ratio explodes (measured max |AMT-GT| error
   13.4 ratio
   units against a corpus MAD of 0.37). An IOI FLOOR removes the pathology. The floor is
   swept
   here rather than assumed; the shipped value must sit on a plateau, not a cliff.

2. **TAU, off the offset tail.** The analytic propagation of the 90ms offset p90 through
a single
   ratio is 0.090s / median_IOI -- about 1.1 ratio units, which is uselessly large. That
   analytic
   bound is wrong for this statistic because the window statistic is a MEDIAN over
   hundreds of
   notes, which absorbs the tail. So tau is calibrated EMPIRICALLY off the same tail: it
   is the
   p90 of the per-window |AMT statistic - GT statistic| discrepancy, i.e. the amount the
   offset
   error alone can move the statistic on 90% of windows. A claim must beat that to be
   adjudicable.
   Both numbers are reported; the analytic one is the caveat, the empirical one is the
   tau.

The error bar uses the same measurement, at 1 sigma rather than the p90 tail, as a flat
floor -- mirroring the dynamics SUBSTRATE_STATISTIC_FLOOR (G-C), except that this floor
is measured against GROUND TRUTH rather than against re-transcription churn, which is a
strictly stronger reference.

Run:
    uv run --no-project --with numpy python \\
      model/src/claim_measurement/dynamics_supply/articulation_tau_calibrate.py \\
      --bundles model/data/evals/maestro_indep_bundles \\
      --out     model/data/results/articulation_tau_calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[4]
DEFAULT_BUNDLES = REPO / "model/data/evals/maestro_indep_bundles"
DEFAULT_OUT = REPO / "model/data/results/articulation_tau_calibration.json"

# Candidate IOI floors (seconds). 0.001 is FRONT 9's probe value (effectively no floor).
IOI_FLOOR_SWEEP = (0.001, 0.010, 0.020, 0.030, 0.050, 0.080, 0.120, 0.200, 0.300)

MINIMUM_PAIRS = 5  # below this the per-window median is not a statistic

# A floor discards notes, so any criterion that rewards a WIDER corpus spread rewards
# measuring a sparser, noisier sub-population (at a 0.12s floor only ~25% of note pairs
# survive and the statistic is effectively "slow melodic notes only"). The floor is
# therefore chosen by MINIMISING substrate error subject to a retention bar, not by
# maximising spread-over-noise.
MINIMUM_PAIR_RETENTION = 0.50


def duration_ioi_pairs(
    notes: list[dict], ioi_floor: float
) -> list[tuple[float, float]]:
    """(duration, IOI) for each onset-sorted note whose IOI to the next clears
    the floor.
    """
    ns = sorted(notes, key=lambda n: n["onset"])
    pairs = []
    for i in range(len(ns) - 1):
        ioi = ns[i + 1]["onset"] - ns[i]["onset"]
        if ioi > ioi_floor:
            pairs.append((ns[i]["offset"] - ns[i]["onset"], ioi))
    return pairs


def articulation_ratio(notes: list[dict], ioi_floor: float) -> float | None:
    """Median duration/IOI. >1 legato (notes overlap), <1 detached. None if too
    few pairs.
    """
    pairs = duration_ioi_pairs(notes, ioi_floor)
    if len(pairs) < MINIMUM_PAIRS:
        return None
    return float(np.median([d / i for d, i in pairs]))


def paired_statistics(
    bundles: list[dict], ioi_floor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window (AMT statistic, GT statistic) over the windows where both are defined.
    """
    amt, gt = [], []
    for b in bundles:
        if "gt_notes" not in b:
            raise ValueError(
                f"bundle {b.get('video_id')} carries no gt_notes; not an oracle bundle"
            )
        ra = articulation_ratio(b["notes"], ioi_floor)
        rg = articulation_ratio(b["gt_notes"], ioi_floor)
        if ra is not None and rg is not None:
            amt.append(ra)
            gt.append(rg)
    return np.array(amt), np.array(gt)


def conditioning_row(bundles: list[dict], ioi_floor: float) -> dict:
    """Substrate error and corpus spread for one candidate IOI floor.

    `abs_err_p90` is the selection number (minimise it) and `pair_retention` is the
    guard that stops the minimisation from being won by throwing notes away.
    `mad_over_err_p90` is reported for interpretation only -- it is NOT the selection
    criterion, because a floor inflates both the numerator (sparser population) and the
    window-to-window sampling noise. """
    amt, gt = paired_statistics(bundles, ioi_floor)
    kept = sum(len(duration_ioi_pairs(b["notes"], ioi_floor)) for b in bundles)
    total = sum(max(len(b["notes"]) - 1, 0) for b in bundles)
    retention = round(kept / total, 4) if total else 0.0
    # A floor above every IOI leaves no measurable window. That is a legitimate sweep
    # outcome -- the eligibility filter rejects it on retention -- so report an empty
    # row rather than letting np.percentile raise an opaque IndexError from inside the
    # sweep.
    if amt.size == 0:
        return {
            "ioi_floor_sec": ioi_floor,
            "n_windows": 0,
            "pair_retention": retention,
            "corr_amt_vs_gt": None,
            "amt_median": None,
            "gt_median": None,
            "gt_mad": None,
            "gt_p10_p90_spread": None,
            "amt_p10_p90_spread": None,
            "signed_err_mean": None,
            "signed_err_sd": None,
            "abs_err_median": None,
            "abs_err_p68": None,
            "abs_err_p90": None,
            "abs_err_max": None,
            "mad_over_err_p90": None,
        }
    err = amt - gt
    abs_err = np.abs(err)
    gt_mad = float(np.median(np.abs(gt - np.median(gt))))
    err_p90 = float(np.percentile(abs_err, 90))
    return {
        "ioi_floor_sec": ioi_floor,
        "n_windows": int(amt.size),
        "pair_retention": retention,
        "corr_amt_vs_gt": round(float(np.corrcoef(amt, gt)[0, 1]), 4),
        # The SHIPPED whole_piece reference must be the AMT median, not the GT median:
        # the measured quantity is the AMT statistic, so anchoring to GT would bake the
        # systematic AMT-vs-GT release bias into every d -- FRONT 8d Cause 1
        # (calibration debt) reborn.
        "amt_median": round(float(np.median(amt)), 4),
        "gt_median": round(float(np.median(gt)), 4),
        "gt_mad": round(gt_mad, 4),
        "gt_p10_p90_spread": round(
            float(np.percentile(gt, 90) - np.percentile(gt, 10)), 4
        ),
        "amt_p10_p90_spread": round(
            float(np.percentile(amt, 90) - np.percentile(amt, 10)), 4
        ),
        "signed_err_mean": round(float(np.mean(err)), 4),
        "signed_err_sd": round(float(np.std(err, ddof=1)), 4),
        "abs_err_median": round(float(np.median(abs_err)), 4),
        "abs_err_p68": round(float(np.percentile(abs_err, 68)), 4),
        "abs_err_p90": round(err_p90, 4),
        "abs_err_max": round(float(np.max(abs_err)), 4),
        "mad_over_err_p90": round(gt_mad / err_p90, 3) if err_p90 > 0 else None,
    }


def offset_tail_ms(bundles: list[dict]) -> dict:
    """The FRONT 9 offset-error tail, recomputed here so tau's anchor sits in
    one artifact.
    """
    from claim_measurement.dynamics_supply.articulation_offset_probe import match_notes

    onset_errs, offset_errs, iois = [], [], []
    for b in bundles:
        for a, g in match_notes(b["notes"], b["gt_notes"]):
            onset_errs.append(abs(a["onset"] - g["onset"]))
            offset_errs.append(abs(a["offset"] - g["offset"]))
        ns = sorted(b["notes"], key=lambda n: n["onset"])
        iois.extend(ns[i + 1]["onset"] - ns[i]["onset"] for i in range(len(ns) - 1))
    onset_errs, offset_errs, iois = (
        np.array(onset_errs),
        np.array(offset_errs),
        np.array(iois),
    )
    median_ioi = float(np.median(iois))
    off_p90 = float(np.percentile(offset_errs, 90))
    return {
        "n_matched_notes": int(offset_errs.size),
        "onset_error_ms": {
            "median": round(float(np.median(onset_errs)) * 1000, 2),
            "p90": round(float(np.percentile(onset_errs, 90)) * 1000, 2),
        },
        "offset_error_ms": {
            "median": round(float(np.median(offset_errs)) * 1000, 2),
            "p90": round(off_p90 * 1000, 2),
        },
        "median_ioi_sec": round(median_ioi, 4),
        "analytic_single_note_ratio_error_at_offset_p90": round(
            off_p90 / median_ioi, 3
        ),
        "analytic_note": (
            "offset_p90 / median_IOI -- the tail's effect on ONE note's ratio. The "
            "window statistic is a median over hundreds of notes, so this analytic "
            "bound massively over-states the statistic error; it is the caveat, not "
            "the tau. The empirical per-window p90 below is the tau."
        ),
    }


def calibrate(bundles: list[dict]) -> dict:
    sweep = [conditioning_row(bundles, f) for f in IOI_FLOOR_SWEEP]
    n_windows = len(bundles)
    eligible = [
        r
        for r in sweep
        if r["pair_retention"] >= MINIMUM_PAIR_RETENTION and r["n_windows"] == n_windows
    ]
    if not eligible:
        raise SystemExit(
            f"no IOI floor retains >= {MINIMUM_PAIR_RETENTION:.0%} of note pairs "
            "on all "
            f""
            f""
            f"{n_windows} windows; the statistic is not conditionable on this corpus"
        )
    # Minimise substrate error; ties (within 5%) go to the floor retaining the most
    # notes.
    row = min(eligible, key=lambda r: r["abs_err_p90"])
    best_p90 = row["abs_err_p90"]
    row = max(
        [r for r in eligible if r["abs_err_p90"] <= best_p90 * 1.05],
        key=lambda r: r["pair_retention"],
    )
    chosen = row["ioi_floor_sec"]
    tail = offset_tail_ms(bundles)
    return {
        "front": "FRONT 10 (#101): articulation measurer calibration",
        "corpus": "MAESTRO test split, real audio, transkun (188 27s windows)",
        "truth_signal": (
            "ground_truth_midi_note_offsets (INDEPENDENT of the scored AMT offsets)"
        ),
        "offset_tail": tail,
        "ioi_floor_sweep": sweep,
        "chosen_ioi_floor_sec": chosen,
        "amt_corpus_median": row["amt_median"],
        "chosen_rationale": (
            f"minimises substrate error p90 subject to retaining >= "
            f"{MINIMUM_PAIR_RETENTION:.0%} "
            "of note pairs on all windows. The unfloored FRONT 9 value (0.001) is "
            ""
            ""
            "chord-pathological: near-simultaneous chord notes drive IOI -> 0 and the "
            "ratio "
            "explodes. Floors above the chosen value keep improving corr but only by "
            "discarding "
            "notes -- at 0.12s just ~25% of pairs survive and the statistic silently "
            "becomes "
            "'slow melodic notes only'."
        ),
        "reference_ratio": row["amt_median"],
        "reference_rationale": (
            "the AMT (Transkun) corpus median, NOT the GT median. d = AMT statistic - "
            "reference, "
            "so a GT-anchored reference would add Transkun's systematic release bias "
            f"({row['signed_err_mean']:+} ratio units) to every measurement."
        ),
        "tau_ratio": row["abs_err_p90"],
        "tau_rationale": (
            "p90 of the per-window |AMT statistic - GT statistic| discrepancy: the "
            "amount "
            "Transkun's offset tail alone can move the statistic on 90% of windows. A "
            "claim must "
            "beat the substrate's own tail to be adjudicable."
        ),
        "substrate_ratio_sigma": row["signed_err_sd"],
        "substrate_statistic_floor": row["abs_err_p68"],
        "discriminability": {
            "gt_mad": row["gt_mad"],
            "tau_over_gt_mad": round(row["abs_err_p90"] / row["gt_mad"], 3)
            if row["gt_mad"]
            else None,
            "read": (
                "tau/MAD >= 1 means the substrate noise is as large as the "
                "between-performance "
                "spread, so only the corpus tails will ever commit -- articulation is "
                "the "
                "least-clean of the verifiable dimensions, exactly as FRONT 9 "
                "predicted."
            ),
        },
    }


def _load_bundles(bundle_dir: Path) -> list[dict]:
    bundles = [
        json.loads(p.read_text())
        for p in sorted(bundle_dir.glob("*.json"))
        if not p.name.endswith(".tmp")
    ]
    if not bundles:
        raise SystemExit(f"no bundles in {bundle_dir}")
    return bundles


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dynamics_supply.articulation_tau_calibrate")
    ap.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    res = calibrate(_load_bundles(args.bundles))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))

    print(
        "\n=== articulation statistic + tau calibration "
        "(transkun, real MAESTRO audio) ==="
    )
    print(
        f"  offset p90 = {res['offset_tail']['offset_error_ms']['p90']}ms; analytic "
        f"single-note "
        f"ratio error = "
        f"{res['offset_tail']['analytic_single_note_ratio_error_at_offset_p90']}"
    )
    print(
        f"  {'ioi_floor':>10} {'n_win':>6} {'kept':>7} {'corr':>7} {'gt_mad':>8} "
        f"{'err_p90':>8}"
    )

    def _f(v: float | None, width: int) -> str:
        return f"{v:>{width}.4f}" if v is not None else f"{'--':>{width}}"

    for r in res["ioi_floor_sweep"]:
        print(
            f"  {r['ioi_floor_sec']:>10.3f} {r['n_windows']:>6d} "
            f"{r['pair_retention']:>7.1%} "
            f"{_f(r['corr_amt_vs_gt'], 7)} {_f(r['gt_mad'], 8)} "
            f"{_f(r['abs_err_p90'], 8)}"
        )
    print(
        f"  CHOSEN ioi_floor = {res['chosen_ioi_floor_sec']}s  reference = "
        f"{res['reference_ratio']}  "
        f"tau = {res['tau_ratio']}"
    )
    print(
        f"  substrate sigma = {res['substrate_ratio_sigma']}  floor = "
        f"{res['substrate_statistic_floor']}"
    )
    print(f"  tau / GT MAD = {res['discriminability']['tau_over_gt_mad']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
