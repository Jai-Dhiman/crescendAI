"""#166: build the pathological tail the container's failure rate is measured on.

The contract excludes a submission failing on >5% of items, and our whole
corpus is one kind of thing: complete YouTube recordings of graded piano
repertoire, 1-10 minutes, decent audio. The test set is *"curated by expert
pedagogues"* from *"sources outside existing public piano difficulty datasets"*,
so its edges are exactly the shapes we have never fed the pipeline.

Each case below is a real failure mode with a named mechanism, not a random
perturbation:

    very_short   20s   MoonBeam sees far fewer than max_len tokens; a piece
                       shorter than one window has never been scored.
    tiny          2s   Below any plausible musical content. Transkun may return
                       few or no notes, which score_wav treats as a failure --
                       this checks that it is the LOUD kind.
    very_long   40min  Concatenation. Full-piece mean-pool chunks to max_len and
                       averages over ALL chunks, so a 40-minute input is ~40x
                       the forward passes of a typical one. This is the case
                       most likely to break the 24h budget, not to crash.
    silence     30s    Digital silence. Zero notes by construction.
    noisy        --    Heavy additive noise: transcription degrades rather than
                       fails, which is the more dangerous outcome (a confident
                       score from a garbage transcription).
    lowfi       --     8kHz mono, 32kbps round trip. Approximates a poor archival
                       recording, which "sources outside existing datasets"
                       plausibly includes.
    truncated   --     A WAV header with the audio cut off mid-stream. Corrupt
                       input, which any real test set eventually contains.

NOT covered here: fixed-soundfont synthesis, which is HALF the MIREX test audio
and the single largest unmeasured risk we carry. That needs fluidsynth over
score MIDI and is #167's job, not this file's -- it is a distribution question,
not a robustness one, and conflating them would hide it.

    uv run python apps/inference/mirex-difficulty/make_tail.py \\
        --wav-dir <data>/results/amt_gap_curve/wav --out-dir /tmp/mirex-tail
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _ffmpeg(args: list[str]) -> None:
    result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args],
                            capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")


def make_tail(source_wav: Path, out_dir: Path, long_minutes: int = 40) -> list:
    """Write the pathological cases and return their paths.

    Loud on failure: this is a research/QA harness, not the container, so
    model/CLAUDE.md's normal rule applies and a case that cannot be built must
    stop the run rather than silently shrink the tail.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = str(Path(source_wav))
    built = []

    def case(name: str, args: list[str]) -> None:
        path = out_dir / f"{name}.wav"
        _ffmpeg(args + [str(path)])
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"{name}: ffmpeg produced nothing")
        built.append(path)

    case("very_short", ["-i", src, "-t", "20"])
    case("tiny", ["-i", src, "-t", "2"])
    case("silence", ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "30"])
    # Additive white noise at a level that degrades without erasing the signal.
    case("noisy", ["-i", src, "-f", "lavfi", "-i",
                   "anoisesrc=r=44100:a=0.20:c=white", "-filter_complex",
                   "[0:a][1:a]amix=inputs=2:duration=first", "-t", "60"])
    # 8kHz mono at 32kbps and back: the resample and the codec both lose
    # information, which is the point.
    tmp_mp3 = out_dir / "_lowfi.mp3"
    _ffmpeg(["-i", src, "-t", "60", "-ar", "8000", "-ac", "1", "-b:a", "32k",
             str(tmp_mp3)])
    case("lowfi", ["-i", str(tmp_mp3), "-ar", "44100"])
    tmp_mp3.unlink()

    # Concatenate the source with itself until it exceeds long_minutes. Uses a
    # concat list rather than a filter so memory stays flat regardless of length.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", src],
        capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {src}: {probe.stderr[-1000:]}")
    duration = float(probe.stdout.strip())
    repeats = max(2, int((long_minutes * 60) / duration) + 1)
    listing = out_dir / "_concat.txt"
    listing.write_text("".join(f"file '{src}'\n" for _ in range(repeats)))
    case("very_long", ["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy"])
    listing.unlink()

    # A valid header with the stream cut off partway through. Written by hand
    # because ffmpeg has no interest in emitting a broken file.
    intact = out_dir / "_intact.wav"
    _ffmpeg(["-i", src, "-t", "30", str(intact)])
    data = intact.read_bytes()
    (out_dir / "truncated.wav").write_bytes(data[: len(data) // 3])
    built.append(out_dir / "truncated.wav")
    intact.unlink()

    return built


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source", default=None,
                    help="filename inside --wav-dir; defaults to the first WAV")
    ap.add_argument("--long-minutes", type=int, default=40)
    args = ap.parse_args(argv)

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} is not on PATH")

    source = (args.wav_dir / args.source if args.source
              else next(iter(sorted(args.wav_dir.glob("*.wav"))), None))
    if source is None or not source.exists():
        raise FileNotFoundError(f"no source WAV found under {args.wav_dir}")

    built = make_tail(source, args.out_dir, args.long_minutes)
    listing = Path(args.out_dir) / "tail_list.txt"
    listing.write_text("\n".join(str(p) for p in built) + "\n")
    for path in built:
        print(f"{path.stat().st_size / 1e6:9.1f} MB  {path.name}")
    print(f"{len(built)} cases from {source.name}; list at {listing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
