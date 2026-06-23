"""Runner: extract measurement bundles over the cached practice-eval clips.

Resolves each (piece, video) clip under practice_eval_pseudo to its cached audio WAV
and its _load_bach_json_score-compatible score (only a subset of pieces have one), then
calls extract_bundle. Missing audio fails loudly (no silent skip); a piece without a
compatible score is reported as 'no_score' (a known structural limit, explicitly logged).

Usage:
    AMT_URL=http://127.0.0.1:8001/transcribe \
        uv run python -m claim_measurement.extract_cli [--force] [--limit-piece bach_invention_1]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from chroma_dtw_eval.amt_regen import DEFAULT_AMT_URL, DEFAULT_SCORE_BY_PIECE

from claim_measurement.extractor import BundleExtractionError, extract_bundle


@contextlib.contextmanager
def _time_limit(seconds: int):
    """Wall-clock limit via SIGALRM. seconds <= 0 disables the guard. Unix, main
    thread only. Guards parangonar alignment, which can blow up combinatorially on
    some real transcriptions (a piece hung >1h in #95)."""
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"exceeded {seconds}s time limit")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

_MODULE_DIR = Path(__file__).resolve()
DEFAULT_DATA_ROOT = _MODULE_DIR.parents[2] / "data" / "evals"
DEFAULT_PSEUDO_ROOT = DEFAULT_DATA_ROOT / "practice_eval_pseudo"
DEFAULT_PRACTICE_EVAL_ROOT = DEFAULT_DATA_ROOT / "practice_eval"
DEFAULT_BUNDLE_ROOT = DEFAULT_DATA_ROOT / "claim_bundles"


@dataclass
class ClipSpec:
    piece_id: str
    video_id: str
    audio_path: Path
    score_path: Path | None


def _resolve_clips(
    pseudo_root: Path,
    practice_eval_root: Path,
    score_by_piece: dict[str, Path],
) -> list[ClipSpec]:
    """Enumerate (piece, video) clips and bind each to its audio WAV + optional score."""
    clips: list[ClipSpec] = []
    for piece_dir in sorted(pseudo_root.iterdir()):
        if not piece_dir.is_dir():
            continue
        piece_id = piece_dir.name
        for vid_dir in sorted(piece_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            video_id = vid_dir.name
            audio_path = practice_eval_root / piece_id / "audio" / f"{video_id}.wav"
            clips.append(
                ClipSpec(
                    piece_id=piece_id,
                    video_id=video_id,
                    audio_path=audio_path,
                    score_path=score_by_piece.get(piece_id),
                )
            )
    return clips


def run(
    clips: list[ClipSpec],
    *,
    bundle_root: Path,
    amt_url: str,
    force: bool,
    timeout_sec: int = 0,
) -> list[dict]:
    """Extract a bundle per scored clip. Returns one result dict per clip.

    Raises BundleExtractionError immediately if a scored clip's audio is missing.
    Clips without a compatible score are recorded as 'no_score' (explicit, not silent).
    A clip exceeding timeout_sec (e.g. parangonar combinatorial blow-up) is recorded
    as 'timeout' rather than blocking the run. timeout_sec <= 0 disables the guard.
    """
    results: list[dict] = []
    for clip in clips:
        base = {"piece": clip.piece_id, "video": clip.video_id}
        if clip.score_path is None:
            results.append({**base, "status": "no_score",
                            "reason": "no _load_bach_json_score-compatible score for piece"})
            continue
        if not clip.audio_path.exists():
            raise BundleExtractionError(
                f"audio missing for {clip.piece_id}/{clip.video_id}: {clip.audio_path}"
            )
        try:
            with _time_limit(timeout_sec):
                out = extract_bundle(
                    clip.piece_id, clip.video_id,
                    audio_path=clip.audio_path, score_path=clip.score_path,
                    cache_root=bundle_root, bundle_root=bundle_root,
                    amt_url=amt_url, force=force,
                )
        except TimeoutError as e:
            results.append({**base, "status": "timeout", "reason": str(e)})
            continue
        except BundleExtractionError as e:
            results.append({**base, "status": "failed", "reason": str(e)})
            continue
        bundle = json.loads(out.read_text())
        results.append({
            **base, "status": "ok", "bundle": str(out),
            "n_notes": len(bundle.get("notes", [])),
            "n_pedal_events": len(bundle.get("pedal_events", [])),
        })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claim_measurement.extract_cli")
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--practice-eval-root", type=Path, default=DEFAULT_PRACTICE_EVAL_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--amt-url", default=DEFAULT_AMT_URL)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-piece", default=None,
                        help="If set, only extract clips for this piece_id.")
    parser.add_argument("--timeout-sec", type=int, default=0,
                        help="Per-clip wall-clock limit (parangonar hang guard); 0 disables.")
    args = parser.parse_args(argv)

    if not args.pseudo_root.exists():
        raise FileNotFoundError(f"pseudo-truth clip root not found: {args.pseudo_root}")

    clips = _resolve_clips(args.pseudo_root, args.practice_eval_root, dict(DEFAULT_SCORE_BY_PIECE))
    if args.limit_piece:
        clips = [c for c in clips if c.piece_id == args.limit_piece]

    results = run(clips, bundle_root=args.bundle_root, amt_url=args.amt_url,
                  force=args.force, timeout_sec=args.timeout_sec)

    args.bundle_root.mkdir(parents=True, exist_ok=True)
    index_path = args.bundle_root / "_index.json"
    index_path.write_text(json.dumps({"amt_url": args.amt_url, "results": results}, indent=2))

    ok = [r for r in results if r["status"] == "ok"]
    no_score = [r for r in results if r["status"] == "no_score"]
    failed = [r for r in results if r["status"] == "failed"]
    timed_out = [r for r in results if r["status"] == "timeout"]
    total_pedals = sum(r["n_pedal_events"] for r in ok)
    print(json.dumps({
        "extracted": len(ok), "no_score": len(no_score),
        "failed": len(failed), "timeout": len(timed_out),
        "total_pedal_events_over_extracted": total_pedals,
        "index": str(index_path),
    }, indent=2))
    for r in ok:
        print(f"  OK   {r['piece']}/{r['video']}: notes={r['n_notes']} pedals={r['n_pedal_events']}")
    for r in failed + timed_out:
        print(f"  {r['status'].upper()} {r['piece']}/{r['video']}: {r['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
