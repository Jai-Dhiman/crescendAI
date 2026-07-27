# model/src/follower_eval/build_corpus.py
"""Build the real-audio follower-eval corpus (issue #133, Slice 2).

Turns the pre-approved practice-video review (``data/evals/practice_eval/*/
candidates.yaml``, ``approved: true``) into transcribed bundles the proxy eval
(``follower_eval.realaudio``) reads. Pipeline per video, matching PRODUCTION:

    approved video_id
      -> acquire.acquire_audio      (yt-dlp -> 16kHz mono WAV, cache-miss only)
      -> transkun_cli.transcribe_wav (PRODUCTION transcriber, #128; isolated env)
      -> bundle JSON                 (claim_bundles-compatible: piece_id, notes, ...)

Resumable and loud: an existing bundle is skipped; every download/transcription
failure is recorded in the run manifest and the loop continues (299 real videos
WILL include link-rot and transcription errors -- they are reported, not hidden,
never written as an empty-notes bundle).

RUNNING (from the PRIMARY checkout so data/ + the venv resolve; long job --
Transkun is CPU whole-clip, minutes per multi-minute video):

  cd /Users/jdhiman/Documents/crescendai/model
  PYTHONPATH=<worktree>/model/src .venv/bin/python -m follower_eval.build_corpus \
    --bundles-root data/evals/realaudio_bundles \
    --manifest data/evals/realaudio_bundles/_build_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from piece_id_eval.acquire import AcquireError, acquire_audio

TRANSCRIBER_ID = "transkun"  # production /transcribe engine (#128); stamped into every bundle


def _import_transcribe_wav():
    """Locate apps/inference/amt (import-safe transkun_cli) from CWD-up or
    file-up and return its transcribe_wav. Kept lazy so unit tests that never
    transcribe don't need the path present."""
    for base in (Path.cwd(), Path(__file__).resolve()):
        for parent in [base, *base.parents]:
            cand = parent / "apps" / "inference" / "amt"
            if (cand / "transkun_cli.py").exists():
                sys.path.insert(0, str(cand))
                from transkun_cli import transcribe_wav  # type: ignore
                return transcribe_wav
    raise RuntimeError("could not locate apps/inference/amt/transkun_cli.py from CWD or module path")


@dataclass
class BuildOutcome:
    """One video's build result. status in {ok, skip, download_fail,
    transcribe_fail, empty}. Exactly the fields the manifest records."""
    piece: str
    video_id: str
    status: str
    n_notes: int = 0
    elapsed_s: float = 0.0
    error: str | None = None


def approved_videos(practice_root: Path, pieces: list[str] | None = None) -> dict[str, list[dict]]:
    """piece_id -> approved recordings, read from every candidates.yaml under
    practice_root. Only ``approved: true`` rows are returned."""
    out: dict[str, list[dict]] = {}
    for cand in sorted(practice_root.glob("*/candidates.yaml")):
        piece = cand.parent.name
        if pieces and piece not in pieces:
            continue
        data = yaml.safe_load(cand.read_text()) or {}
        approved = [r for r in (data.get("recordings") or []) if r.get("approved") is True]
        if approved:
            out[piece] = approved
    return out


def build_one(piece: str, rec: dict, audio_dir: Path, bundle_path: Path,
              transcribe_wav, force: bool = False) -> BuildOutcome:
    """Download + transcribe one approved video into a bundle. Idempotent: an
    existing bundle short-circuits unless force. Never writes an empty-notes
    bundle -- an empty transcription is a loud ``empty`` outcome."""
    vid = rec["video_id"]
    if bundle_path.exists() and not force:
        return BuildOutcome(piece, vid, "skip")

    t0 = time.perf_counter()
    try:
        wav = acquire_audio(vid, audio_dir)
    except AcquireError as exc:
        return BuildOutcome(piece, vid, "download_fail", elapsed_s=round(time.perf_counter() - t0, 1),
                            error=str(exc)[:400])

    try:
        notes, pedals = transcribe_wav(wav)
    except Exception as exc:  # TranskunError or anything the subprocess throws
        return BuildOutcome(piece, vid, "transcribe_fail", elapsed_s=round(time.perf_counter() - t0, 1),
                            error=f"{type(exc).__name__}: {exc}"[:400])

    if not notes:
        return BuildOutcome(piece, vid, "empty", elapsed_s=round(time.perf_counter() - t0, 1),
                            error="transkun returned zero notes")

    bundle = {
        "piece_id": piece,
        "video_id": vid,
        "audio_path": str(wav),
        "title": rec.get("title"),
        "duration_seconds": rec.get("duration_seconds"),
        "notes": notes,
        "pedal_events": pedals,
        "source": "practice_eval_approved",
        "substrate_versions": {"transcriber": TRANSCRIBER_ID, "device": "cpu"},
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(bundle))
    return BuildOutcome(piece, vid, "ok", n_notes=len(notes),
                        elapsed_s=round(time.perf_counter() - t0, 1))


def run(practice_root: Path, audio_root: Path, bundles_root: Path, manifest_path: Path,
        pieces: list[str] | None = None, limit_per_piece: int | None = None,
        limit: int | None = None, force: bool = False) -> list[BuildOutcome]:
    """Build bundles for every approved video (optionally capped). Writes the
    manifest after EACH video so an interrupted long run keeps its record."""
    by_piece = approved_videos(practice_root, pieces)
    transcribe_wav = _import_transcribe_wav()

    outcomes: list[BuildOutcome] = []
    n_done = 0
    total = sum(min(len(v), limit_per_piece or len(v)) for v in by_piece.values())
    if limit is not None:
        total = min(total, limit)

    for piece in sorted(by_piece):
        recs = by_piece[piece][:limit_per_piece] if limit_per_piece else by_piece[piece]
        audio_dir = audio_root / piece / "audio"
        for rec in recs:
            if limit is not None and n_done >= limit:
                break
            bundle_path = bundles_root / piece / f"{rec['video_id']}.json"
            outcome = build_one(piece, rec, audio_dir, bundle_path, transcribe_wav, force=force)
            outcomes.append(outcome)
            n_done += 1
            print(f"[{n_done}/{total}] {piece}/{outcome.video_id}: {outcome.status}"
                  f"{f' ({outcome.n_notes} notes, {outcome.elapsed_s}s)' if outcome.status == 'ok' else ''}"
                  f"{f' -- {outcome.error}' if outcome.error else ''}", flush=True)
            _write_manifest(manifest_path, outcomes)
        if limit is not None and n_done >= limit:
            break
    return outcomes


def _write_manifest(manifest_path: Path, outcomes: list[BuildOutcome]) -> None:
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(
        {"transcriber": TRANSCRIBER_ID, "counts": counts, "n": len(outcomes),
         "outcomes": [asdict(o) for o in outcomes]}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build real-audio follower-eval corpus (#133 Slice 2)")
    ap.add_argument("--practice-root", type=Path, default=Path("data/evals/practice_eval"))
    ap.add_argument("--audio-root", type=Path, default=Path("data/evals/practice_eval"),
                    help="per-piece <root>/<piece>/audio/<vid>.wav download cache")
    ap.add_argument("--bundles-root", type=Path, default=Path("data/evals/realaudio_bundles"))
    ap.add_argument("--manifest", type=Path, default=None,
                    help="run manifest path (default: <bundles-root>/_build_manifest.json)")
    ap.add_argument("--pieces", nargs="+", default=None)
    ap.add_argument("--limit-per-piece", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="global cap (smoke runs)")
    ap.add_argument("--force", action="store_true", help="rebuild even if a bundle exists")
    args = ap.parse_args()

    manifest = args.manifest or (args.bundles_root / "_build_manifest.json")
    t0 = time.perf_counter()
    outcomes = run(args.practice_root, args.audio_root, args.bundles_root, manifest,
                   pieces=args.pieces, limit_per_piece=args.limit_per_piece,
                   limit=args.limit, force=args.force)
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    print(f"\nDONE in {time.perf_counter() - t0:.0f}s  counts={counts}  manifest={manifest}")


if __name__ == "__main__":
    main()
