# model/src/piece_id_eval/bakeoff.py
"""Piece-ID bakeoff orchestrator.

Sweeps: window_lengths x matchers x corruption_grid.
For each (matcher, window_length, corruption): compute recall@{1,5,10} across recordings.
Computes leave-one-out open-set FA/TA curve.
Emits: BakeoffReport (recall_table, corruption_curves, open_set_point, verdict).
Writes sidecar JSON if --output-json is given.
Logs to Trackio unless --no-track.

CLI: python -m piece_id_eval.bakeoff [--no-track] [--synthetic-only] [--output-json PATH]
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from piece_id_eval.corruption import corrupt_notes
from piece_id_eval.decision import decide
from piece_id_eval.matchers import (
    ChromaSeqDtwMatcher,
    DtwCeilingMatcher,
    LandmarkMatcher,
    NoteChromaMatcher,
)
from piece_id_eval.matchers.base import Matcher
from piece_id_eval.metrics import Rankings, recall_at_k
from piece_id_eval.notes import Note
from piece_id_eval.open_set import best_point, operating_points
from piece_id_eval.windowing import sample_windows

_MODULE_DIR = Path(__file__).resolve().parent
# parents[0] = piece_id_eval/, parents[1] = model/ (crescendai/model/data/...)
_DEFAULT_PRACTICE_ROOT = _MODULE_DIR.parents[1] / "data/evals/practice_eval"
_DEFAULT_NOTES_ROOT = _MODULE_DIR.parents[1] / "data/evals/practice_eval_pseudo"
_DEFAULT_SCORES_ROOT = _MODULE_DIR.parents[1] / "data/scores"
_DEFAULT_PIECE_MAP = _MODULE_DIR.parents[1] / "data/evals/piece_id/eval_piece_map.json"

_WINDOW_LENGTHS: list[float | None] = [15.0, 30.0, 60.0, 90.0, None]
_N_STARTS = 5
_SEED = 42
# NoteChromaMatcher produces cosine scores in [-1, 1]; sweep [0, 1] (positive matches).
_OPEN_SET_THRESHOLDS = [i / 100 for i in range(0, 101)]  # 0.00, 0.01, ..., 1.00
_MAX_FA = 0.05
_MIN_TA = 0.60


@dataclass
class BakeoffReport:
    """Output of a bakeoff run."""
    # recall_table: {(matcher_name, window_label): {"recall@1": float, "recall@5": float, "recall@10": float}}
    recall_table: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    # corruption_curves: {(matcher_name, corruption_label): {"recall@10": float}}
    corruption_curves: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    # open_set_ok: True if a threshold exists with FA<=0.05, TA>=0.60
    open_set_ok: bool = False
    # verdict: KILL / TUNE / PROCEED
    verdict: str = "TUNE"
    # dtw_ceiling_recall10: best recall@10 for DtwCeilingMatcher (ceiling)
    dtw_ceiling_recall10: float = 0.0
    # best_indexable_recall10: max recall@10 across C1/C2/C4
    best_indexable_recall10: float = 0.0


def _window_label(window_seconds: float | None) -> str:
    return "full" if window_seconds is None else f"{int(window_seconds)}s"


def _corruption_label(grid_entry: dict[str, float]) -> str:
    return (
        f"del={grid_entry['deletion_rate']:.1f}"
        f"_ins={grid_entry['insertion_rate']:.1f}"
        f"_jit={grid_entry['jitter_seconds']:.2f}"
    )


def run(
    catalog: dict[str, list[Note]],
    recordings: dict[str, list[Note]],
    window_lengths: list[float | None] | None = None,
    n_starts: int = _N_STARTS,
    corruption_grid: list[dict[str, float]] | None = None,
    seed: int = _SEED,
    no_track: bool = False,
) -> BakeoffReport:
    """Run the full bakeoff and return a BakeoffReport.

    Args:
        catalog: {piece_id: list[Note]} for all 254 (or synthetic) catalog pieces.
        recordings: {piece_id: list[Note]} for each query recording (true piece_id as key).
        window_lengths: list of window durations in seconds; None = full piece.
            Defaults to [15, 30, 60, 90, None].
        n_starts: number of random start offsets per window length.
        corruption_grid: list of corruption parameter dicts with keys
            deletion_rate, insertion_rate, jitter_seconds.
            Defaults to a small preset grid.
        seed: RNG seed for windowing and corruption.
        no_track: if True, skip Trackio logging.

    Returns:
        BakeoffReport with recall tables, corruption curves, and verdict.
    """
    if window_lengths is None:
        window_lengths = _WINDOW_LENGTHS
    if corruption_grid is None:
        corruption_grid = [
            {"deletion_rate": 0.0, "insertion_rate": 0.0, "jitter_seconds": 0.0},
            {"deletion_rate": 0.2, "insertion_rate": 0.0, "jitter_seconds": 0.05},
            {"deletion_rate": 0.4, "insertion_rate": 0.1, "jitter_seconds": 0.1},
        ]

    matchers: list[Matcher] = [
        NoteChromaMatcher(catalog),
        LandmarkMatcher(catalog),
        DtwCeilingMatcher(catalog),
        ChromaSeqDtwMatcher(catalog),
    ]

    report = BakeoffReport()

    # Recall sweep: window x matcher (no corruption)
    for window_seconds in window_lengths:
        wlabel = _window_label(window_seconds)
        for matcher in matchers:
            rankings: Rankings = []
            for true_id, notes in recordings.items():
                windows = sample_windows(notes, window_seconds, n_starts, seed)
                for win in windows:
                    if not win:
                        continue
                    ranked = matcher.rank(win)
                    rankings.append((true_id, [(r.piece_id, r.score) for r in ranked]))
            if not rankings:
                continue
            r1 = recall_at_k(rankings, 1)
            r5 = recall_at_k(rankings, 5)
            r10 = recall_at_k(rankings, 10)
            report.recall_table[(matcher.name, wlabel)] = {
                "recall@1": r1,
                "recall@5": r5,
                "recall@10": r10,
            }

    # Corruption sweep: use full window; compare across matchers
    for entry in corruption_grid:
        clabel = _corruption_label(entry)
        for matcher in matchers:
            rankings = []
            for true_id, notes in recordings.items():
                corrupted = corrupt_notes(
                    notes,
                    deletion_rate=entry["deletion_rate"],
                    insertion_rate=entry["insertion_rate"],
                    jitter_seconds=entry["jitter_seconds"],
                    seed=seed,
                )
                if not corrupted:
                    continue
                ranked = matcher.rank(corrupted)
                rankings.append((true_id, [(r.piece_id, r.score) for r in ranked]))
            if not rankings:
                continue
            report.corruption_curves[(matcher.name, clabel)] = {
                "recall@10": recall_at_k(rankings, 10)
            }

    # Compute ceiling + best indexable recall@10 (best across all window lengths).
    # Read .name from the existing matchers list — no extra instantiations.
    dtw_name = matchers[2].name  # "dtw_ceiling"
    indexable_names = {matchers[0].name, matchers[1].name, matchers[3].name}

    for (mname, wlabel), vals in report.recall_table.items():
        r10 = vals["recall@10"]
        if mname == dtw_name:
            report.dtw_ceiling_recall10 = max(report.dtw_ceiling_recall10, r10)
        if mname in indexable_names:
            report.best_indexable_recall10 = max(report.best_indexable_recall10, r10)

    # Open-set: leave-one-out (remove true piece from catalog, top-1 score).
    # Uses NoteChromaMatcher (C1, cosine in [-1, 1]) as the open-set oracle:
    # matched cosines are positive, so the [0, 1] threshold sweep is meaningful.
    # DtwCeilingMatcher scores are -normalized_cost (always <= 0), making it
    # incompatible with a [0, 1] threshold sweep (TA and FA both stuck at 0).
    in_scores: list[float] = []
    loo_scores: list[float] = []
    open_set_oracle = matchers[0]  # NoteChromaMatcher (C1, cosine similarity)
    for true_id, notes in recordings.items():
        # In-catalog: full catalog
        ranked = open_set_oracle.rank(notes)
        in_scores.append(ranked[0].score if ranked else 0.0)
        # LOO: catalog without true piece
        loo_catalog = {pid: n for pid, n in catalog.items() if pid != true_id}
        if loo_catalog:
            loo_matcher = NoteChromaMatcher(loo_catalog)
            loo_ranked = loo_matcher.rank(notes)
            loo_scores.append(loo_ranked[0].score if loo_ranked else 0.0)

    if in_scores and loo_scores:
        pts = operating_points(in_scores, loo_scores, _OPEN_SET_THRESHOLDS)
        bp = best_point(pts, max_fa=_MAX_FA, min_ta=_MIN_TA)
        report.open_set_ok = bp is not None

    report.verdict = decide(
        dtw_recall10=report.dtw_ceiling_recall10,
        best_indexable_recall10=report.best_indexable_recall10,
        open_set_ok_flag=report.open_set_ok,
    )

    if not no_track:
        try:
            import trackio as tr
            tr.log({
                "dtw_ceiling_recall10": report.dtw_ceiling_recall10,
                "best_indexable_recall10": report.best_indexable_recall10,
                "open_set_ok": report.open_set_ok,
                "verdict": report.verdict,
            })
        except (ImportError, RuntimeError) as exc:
            print(f"WARNING: Trackio logging failed: {exc}", file=sys.stderr)

    return report


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Piece-ID note-based bakeoff. Emits VERDICT: KILL|TUNE|PROCEED."
    )
    parser.add_argument("--no-track", action="store_true", help="Skip Trackio logging.")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Run on a 2-piece synthetic catalog instead of real data (for CI smoke).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write BakeoffReport metrics to this JSON path.",
    )
    parser.add_argument(
        "--piece-map",
        type=Path,
        default=_DEFAULT_PIECE_MAP,
        help="JSON map of slug->piece_id.",
    )
    parser.add_argument(
        "--scores-root",
        type=Path,
        default=_DEFAULT_SCORES_ROOT,
        help="Directory of score JSON files.",
    )
    parser.add_argument(
        "--notes-root",
        type=Path,
        default=_DEFAULT_NOTES_ROOT,
        help="Root of practice_eval_pseudo cache directories.",
    )
    parser.add_argument(
        "--practice-root",
        type=Path,
        default=_DEFAULT_PRACTICE_ROOT,
        help="Root of practice_eval directories (for candidates.yaml).",
    )
    args = parser.parse_args()

    if args.synthetic_only:
        # 2-piece synthetic run for CI smoke
        def _p(root: int, n: int = 40) -> list[Note]:
            return [Note(onset=i * 0.5, offset=i * 0.5 + 0.4, pitch=root + (i % 7), velocity=80) for i in range(n)]

        catalog = {"piece_a": _p(60), "piece_b": _p(67)}
        recordings = {"piece_a": _p(60), "piece_b": _p(67)}
        report = run(
            catalog=catalog,
            recordings=recordings,
            window_lengths=[None, 10.0],
            n_starts=2,
            corruption_grid=[{"deletion_rate": 0.0, "insertion_rate": 0.0, "jitter_seconds": 0.0}],
            seed=42,
            no_track=True,
        )
    else:
        from piece_id_eval.notes import load_amt_notes, load_score_notes

        piece_map: dict[str, str] = json.loads(args.piece_map.read_text())

        # Load catalog from score JSONs
        catalog: dict[str, list[Note]] = {}
        for score_json in args.scores_root.glob("*.json"):
            piece_id = score_json.stem
            try:
                catalog[piece_id] = load_score_notes(score_json)
            except Exception as exc:
                print(f"[WARN] skipping {score_json.name}: {exc}", file=sys.stderr)

        # Load recordings from amt_notes.json cache
        import yaml

        recordings: dict[str, list[Note]] = {}
        for slug, piece_id in piece_map.items():
            slug_dir = args.practice_root / slug
            candidates_file = slug_dir / "candidates.yaml"
            if not candidates_file.exists():
                print(f"[SKIP] {slug}: no candidates.yaml", file=sys.stderr)
                continue
            with candidates_file.open() as f:
                candidates = yaml.safe_load(f)
            approved = [r for r in (candidates.get("recordings") or []) if r.get("approved")]
            for rec in approved:
                video_id = rec["video_id"]
                notes_path = args.notes_root / slug / video_id / "amt_notes.json"
                if not notes_path.exists():
                    print(f"[SKIP] {slug}/{video_id}: amt_notes.json missing (run transcribe first)", file=sys.stderr)
                    continue
                try:
                    notes = load_amt_notes(notes_path)
                    recordings[piece_id] = notes
                    break  # use first available approved recording per slug
                except Exception as exc:
                    print(f"[WARN] {slug}/{video_id}: {exc}", file=sys.stderr)

        if not recordings:
            print("ERROR: no recordings loaded. Run `python -m piece_id_eval.transcribe` first.", file=sys.stderr)
            sys.exit(1)

        report = run(
            catalog=catalog,
            recordings=recordings,
            no_track=args.no_track,
        )

    # Print recall table
    print("\n=== Recall Table ===")
    for (mname, wlabel), vals in sorted(report.recall_table.items()):
        print(f"  {mname:30s} window={wlabel:6s}  R@1={vals['recall@1']:.3f}  R@5={vals.get('recall@5', 0):.3f}  R@10={vals['recall@10']:.3f}")

    print(f"\nDTW ceiling recall@10: {report.dtw_ceiling_recall10:.3f}")
    print(f"Best indexable recall@10: {report.best_indexable_recall10:.3f}")
    print(f"Open-set ok (FA<=0.05 @ TA>=0.60): {report.open_set_ok}")
    print(f"\nVERDICT: {report.verdict}")

    if args.output_json:
        out: dict[str, Any] = {
            "verdict": report.verdict,
            "dtw_ceiling_recall10": report.dtw_ceiling_recall10,
            "best_indexable_recall10": report.best_indexable_recall10,
            "open_set_ok": report.open_set_ok,
            "recall_table": {f"{m}|{w}": v for (m, w), v in report.recall_table.items()},
            "corruption_curves": {f"{m}|{c}": v for (m, c), v in report.corruption_curves.items()},
        }
        args.output_json.write_text(json.dumps(out, indent=2))
        print(f"\nSidecar JSON written to {args.output_json}")


if __name__ == "__main__":
    _cli_main()
