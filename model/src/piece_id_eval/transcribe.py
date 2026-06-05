# model/src/piece_id_eval/transcribe.py
"""AMT-notes cache builder.

Calls the local AMT server to transcribe an audio file and writes the result
to a JSON cache. Idempotent: skips the HTTP call if the cache already exists
(unless force=True).

Reuses _read_wav_16k_mono and _transcribe_clip from chroma_dtw_eval.amt_regen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from chroma_dtw_eval.amt_regen import (
    DEFAULT_AMT_URL,
    _read_wav_16k_mono,
    _transcribe_clip,
)

_MODULE_DIR = Path(__file__).resolve().parent
# parents[0] = piece_id_eval/, parents[1] = model/ (crescendai/model/data/...)
DEFAULT_PRACTICE_ROOT = _MODULE_DIR.parents[1] / "data/evals/practice_eval"
DEFAULT_AMT_NOTES_ROOT = _MODULE_DIR.parents[1] / "data/evals/practice_eval_pseudo"


def ensure_amt_notes(
    audio_path: Path,
    out_path: Path,
    amt_url: str = DEFAULT_AMT_URL,
    force: bool = False,
) -> Path:
    """Transcribe audio_path via AMT and write notes to out_path as JSON.

    Idempotent: if out_path already exists and force=False, returns immediately.

    Args:
        audio_path: path to a WAV file (any sample rate; resampled to 16kHz).
        out_path: destination JSON path; parent directory is created if needed.
        amt_url: URL of the local AMT /transcribe endpoint.
        force: if True, re-transcribe even if out_path already exists.

    Returns:
        out_path (the written or pre-existing cache file).

    Raises:
        FileNotFoundError: if audio_path does not exist.
        AmtRegenError: if the AMT server fails after retries.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if out_path.exists() and not force:
        return out_path

    audio_16k = _read_wav_16k_mono(audio_path)
    notes = _transcribe_clip(audio_16k, amt_url)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(notes, indent=2))
    return out_path


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe cached audio for piece-ID slugs via local AMT server."
    )
    parser.add_argument(
        "--slugs",
        nargs="+",
        required=True,
        help="Slug names under data/evals/practice_eval/ (e.g. bach_prelude_c_wtc1)",
    )
    parser.add_argument(
        "--amt-url",
        default=DEFAULT_AMT_URL,
        help=f"AMT endpoint URL (default: {DEFAULT_AMT_URL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe even if cache already exists.",
    )
    parser.add_argument(
        "--practice-root",
        type=Path,
        default=DEFAULT_PRACTICE_ROOT,
        help="Root of practice_eval audio directories.",
    )
    parser.add_argument(
        "--notes-root",
        type=Path,
        default=DEFAULT_AMT_NOTES_ROOT,
        help="Root of practice_eval_pseudo cache directories.",
    )
    args = parser.parse_args()

    import yaml  # pyyaml; already in pyproject.toml deps

    for slug in args.slugs:
        slug_dir = args.practice_root / slug
        candidates_file = slug_dir / "candidates.yaml"
        if not candidates_file.exists():
            print(f"[SKIP] {slug}: no candidates.yaml at {candidates_file}", file=sys.stderr)
            continue
        with candidates_file.open() as f:
            candidates = yaml.safe_load(f)
        recordings = [r for r in (candidates.get("recordings") or []) if r.get("approved")]
        for rec in recordings:
            video_id = rec["video_id"]
            audio_path = slug_dir / "audio" / f"{video_id}.wav"
            if not audio_path.exists():
                print(f"[SKIP] {slug}/{video_id}: audio not cached at {audio_path}", file=sys.stderr)
                continue
            out_path = args.notes_root / slug / video_id / "amt_notes.json"
            print(f"[transcribe] {slug}/{video_id} -> {out_path}")
            ensure_amt_notes(audio_path, out_path, amt_url=args.amt_url, force=args.force)
            print(f"[done]      {slug}/{video_id}")


if __name__ == "__main__":
    _cli_main()
