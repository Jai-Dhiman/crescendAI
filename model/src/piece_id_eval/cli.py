"""CLI entry point for the piece-ID feasibility harness.

Usage:
    python -m piece_id_eval.cli \\
        --slugs bach_prelude_c_wtc1 fur_elise ... \\
        --eval-root model/data/evals/practice_eval \\
        --scores-dir model/data/scores \\
        --piece-map model/data/evals/piece_id/eval_piece_map.json \\
        [--holdout slug1 slug2] \\
        [--sidecar path/to/result.json] \\
        [--window-seconds 2.0] \\
        [--hop-seconds 1.0] \\
        [--no-track]

Prints a per-matcher metrics table and a final VERDICT: KILL|TUNE|PROCEED line.
Exit code: always 0 on successful run (the verdict is research output, not a gate).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parents[2]
_DEFAULT_EVAL_ROOT = _REPO_ROOT / "model" / "data" / "evals" / "practice_eval"
_DEFAULT_SCORES_DIR = _REPO_ROOT / "model" / "data" / "scores"
_DEFAULT_PIECE_MAP = _REPO_ROOT / "model" / "data" / "evals" / "piece_id" / "eval_piece_map.json"
_DEFAULT_SIDECAR = _REPO_ROOT / "model" / "data" / "evals" / "piece_id" / "last_run.json"

_ALL_SLUGS = [
    "bach_invention_1", "bach_prelude_c_wtc1", "chopin_ballade_1",
    "chopin_etude_op10no4", "chopin_waltz_csm", "clair_de_lune",
    "debussy_arabesque_1", "fantaisie_impromptu", "fur_elise",
    "liszt_liebestraum_3", "moonlight_sonata_mvt1", "mozart_k545_mvt1",
    "nocturne_op9no2", "pathetique_mvt2", "rachmaninoff_prelude_csm",
    "schumann_traumerei",
]


def _load_catalog(
    scores_dir: Path,
    holdout_piece_ids: set[str],
    frame_rate_hz: float = 50.0,
) -> dict[str, np.ndarray]:
    from piece_id_eval.score_chroma import load_catalog_score_chroma

    catalog: dict[str, np.ndarray] = {}
    for score_path in sorted(scores_dir.glob("*.json")):
        piece_id = score_path.stem
        if piece_id in holdout_piece_ids:
            continue
        try:
            catalog[piece_id] = load_catalog_score_chroma(score_path, frame_rate_hz)
        except (KeyError, ValueError):
            # Skip non-score JSON files that lack 'bars'/'notes' keys (e.g. titles.json)
            # and score files whose note lists are empty or otherwise invalid.
            pass
    return catalog


def _print_table(matcher_results) -> None:
    header = f"{'Matcher':<30} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6}"
    print(header)
    print("-" * len(header))
    for mr in matcher_results:
        print(
            f"{mr.matcher_name:<30} "
            f"{mr.recall_at_1:>6.3f} "
            f"{mr.recall_at_5:>6.3f} "
            f"{mr.recall_at_10:>6.3f} "
            f"{mr.mrr:>6.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="piece_id_eval.cli")
    parser.add_argument("--slugs", nargs="+", default=_ALL_SLUGS,
                        help="Practice-eval slugs to evaluate (default: all 16)")
    parser.add_argument("--eval-root", type=Path, default=_DEFAULT_EVAL_ROOT)
    parser.add_argument("--scores-dir", type=Path, default=_DEFAULT_SCORES_DIR)
    parser.add_argument("--piece-map", type=Path, default=_DEFAULT_PIECE_MAP)
    parser.add_argument("--holdout", nargs="*", default=[],
                        help="Slugs to hold out for open-set probe")
    parser.add_argument("--sidecar", type=Path, default=_DEFAULT_SIDECAR)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--frame-rate-hz", type=float, default=50.0)
    parser.add_argument("--ngram-n", type=int, default=3,
                        help="N-gram size for ChordNgramMatcher")
    parser.add_argument("--no-track", action="store_true",
                        help="Suppress Trackio experiment logging")
    args = parser.parse_args(argv)

    from piece_id_eval.matchers import ChordNgramMatcher, DtwCeilingMatcher, TwoDFTMatcher
    from piece_id_eval.query_set import QuerySet
    from piece_id_eval.report import EvalReport

    # Load query windows
    print(f"Loading query windows for {len(args.slugs)} slugs...")
    load_result = QuerySet.load(
        slugs=args.slugs,
        eval_root=args.eval_root,
        piece_map_path=args.piece_map,
        audio_cache_root=args.eval_root,
        holdout_slugs=args.holdout,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
    )
    print(
        f"  {len(load_result.windows)} windows loaded, "
        f"{load_result.excluded_count} recordings excluded (missing audio)"
    )
    if not load_result.windows:
        print("ERROR: no query windows available. Run audio acquisition first.", file=sys.stderr)
        return 1

    # Resolve piece_ids for holdout exclusion from catalog
    piece_map: dict[str, str] = json.loads(args.piece_map.read_text())
    holdout_piece_ids: set[str] = {
        piece_map[s] for s in args.holdout if s in piece_map
    }

    # Load catalog
    print(f"Loading catalog from {args.scores_dir}...")
    catalog = _load_catalog(
        args.scores_dir,
        holdout_piece_ids,
        args.frame_rate_hz,
    )
    print(f"  {len(catalog)} catalog pieces loaded")

    # Build matchers (each variant: with and without OTI)
    matchers = [
        DtwCeilingMatcher(catalog, oti=False),
        ChordNgramMatcher(catalog, oti=False, n=args.ngram_n),
        ChordNgramMatcher(catalog, oti=True, n=args.ngram_n),
        TwoDFTMatcher(catalog, oti=False),
        TwoDFTMatcher(catalog, oti=True),
    ]

    thresholds = np.linspace(0.0, 1.0, 101)
    print("\nRunning evaluation...")
    report = EvalReport.run(
        load_result.windows,
        matchers,
        thresholds=thresholds,
    )

    # Print results table
    print()
    _print_table(report.matcher_results)
    print()
    print(f"Open-set OK (FA<=0.10 @ TA>=0.75): {report.open_set_ok_flag}")
    print()
    print(f"VERDICT: {report.verdict}")

    # Write sidecar JSON
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar_data = {
        "verdict": report.verdict,
        "open_set_ok": report.open_set_ok_flag,
        "matchers": [
            {
                "name": mr.matcher_name,
                "recall_at_1": mr.recall_at_1,
                "recall_at_5": mr.recall_at_5,
                "recall_at_10": mr.recall_at_10,
                "mrr": mr.mrr,
            }
            for mr in report.matcher_results
        ],
        "n_windows": len(load_result.windows),
        "n_excluded": load_result.excluded_count,
        "n_catalog_pieces": len(catalog),
        "holdout_slugs": args.holdout,
        "window_seconds": args.window_seconds,
        "hop_seconds": args.hop_seconds,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    args.sidecar.write_text(json.dumps(sidecar_data, indent=2))

    # Trackio logging
    if not args.no_track:
        try:
            import trackio
            run = trackio.init(project="crescendai-piece-id-feasibility")
            for mr in report.matcher_results:
                run.log({
                    f"{mr.matcher_name}/recall_at_10": mr.recall_at_10,
                    f"{mr.matcher_name}/mrr": mr.mrr,
                })
            run.log({"verdict": report.verdict, "open_set_ok": int(report.open_set_ok_flag)})
            run.finish()
        except Exception as exc:
            print(f"WARNING: Trackio logging failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
