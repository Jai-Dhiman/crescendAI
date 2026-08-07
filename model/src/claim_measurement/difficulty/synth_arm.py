"""#167 (#104 S2): run one synthesis arm over the 900-piece eval set.

    mid.zip MIDI -> [arm transform] -> fluidsynth -> Transkun -> a cached .mid

Transcription is the whole cost of this stage -- the eval set is 37.5 hours of
audio and Transkun reads ~0.44x realtime natively on this machine, so an arm is
roughly 16 hours of CPU. Three properties follow from that number and none of
them are optional:

**Resumable.** The completion marker is the transcribed MIDI on disk, written
via a temp file and `os.replace`, so a kill at hour 12 costs one item rather
than twelve hours. `pending()` is what a relaunch consults.

**Per-item streaming.** Rows append to a JSONL as they finish. #166 lost a
whole run's scores to an OOM kill that took the in-memory results with it; the
fix there was to stream, and the same applies at 8x the duration.

**Loud, but not fatal per item.** `model/CLAUDE.md` mandates explicit failure
over silent fallback, and the container's inverted policy stops at
`score_wav_or_fallback` and must not reach here. The compromise that keeps both:
a failed item records its exception text in its row and the run continues, then
`run_arm` returns a non-zero failure count and the CLI exits non-zero. Nothing
is substituted for a failed item -- it is absent from the arm and the evaluation
refuses to score an arm with missing rows.

Embedding is deliberately NOT in this pass. It needs the 1.6GB backbone loaded
once per fold and runs at ~1s/piece, so it is a separate cheap pass over the
cached transcriptions rather than 900 model loads interleaved with 16 hours of
Transkun.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import tempfile
import threading
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from claim_measurement.difficulty.synth_render import (
    deadpan,
    read_performance,
    render_wav,
    write_performance_midi,
)

# "source" is the B0 control: mid.zip embedded directly, no render and no
# re-transcription. It exists to subtract a confound rather than to measure a
# shift -- arm A passes through one transcription (of the human audio) and the
# rendered arms pass through two, so a bare synth-minus-human delta would bundle
# the synthesis penalty with double-transcription loss. It is listed here for
# completeness but never reaches this module: it has no audio stage.
RENDERED_ARMS = ("expressive", "deadpan")

ROWS_FILENAME = "rows.jsonl"

# The (n_folds, seed) pair the five adapters are welded to. Not a knob: a set
# of per-fold adapters is only fold-honest against the folds that produced its
# training pools (ft_eval.py's docstring has the full rationale).
N_FOLDS, SEED = 5, 2026


@dataclass(frozen=True)
class ArmItem:
    seg_id: str
    midi_name: str
    fold: int


def plan_items(seg_ids, manifest_rows, fold_by_seg) -> list[ArmItem]:
    """The work list, in the canonical eval row order.

    Raises on a seg_id the manifest or the fold map cannot place, rather than
    dropping it: a silently shorter arm would still produce a tau-c, and it
    would be a tau-c over a different population than the one it is compared
    against.
    """
    by_seg = {row["seg_id"]: row for row in manifest_rows}
    items = []
    for seg_id in seg_ids:
        if seg_id not in by_seg:
            raise KeyError(
                f"{seg_id!r} is an eval piece with no manifest row, so its "
                f"mid.zip member is unknown")
        if seg_id not in fold_by_seg:
            raise KeyError(
                f"{seg_id!r} is an eval piece assigned to no fold, so no "
                f"adapter held it out and it cannot be scored fold-honestly")
        items.append(ArmItem(seg_id=seg_id,
                             midi_name=by_seg[seg_id]["midi_name"],
                             fold=int(fold_by_seg[seg_id])))
    return items


def transcription_path(out_dir: Path, seg_id: str) -> Path:
    return Path(out_dir) / "tx" / f"{seg_id}.mid"


def pending(items, out_dir: Path) -> list[ArmItem]:
    """Items with no cached transcription yet. The marker is the file, written
    atomically, so a half-written MIDI never counts as done."""
    return [it for it in items if not transcription_path(out_dir, it.seg_id).is_file()]


def _atomic_write_midi(notes, path: Path) -> None:
    from claim_measurement.difficulty.audio_emb_extract import write_notes_midi

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".mid.part")
    os.close(fd)
    try:
        write_notes_midi(notes, Path(tmp))
        os.replace(tmp, path)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()


def run_item(item: ArmItem, arm: str, mid_zip: Path, soundfont: Path,
             out_dir: Path, transcribe, render=render_wav,
             keep_wav: bool = False) -> dict:
    """Render and transcribe one piece. Returns the JSONL row.

    `transcribe` and `render` are injected so the offline suite never needs
    Transkun or fluidsynth; the CLI wires the real ones.
    """
    if arm not in RENDERED_ARMS:
        raise ValueError(f"arm must be one of {RENDERED_ARMS}, got {arm!r}")

    out_dir = Path(out_dir)
    row = {"seg_id": item.seg_id, "arm": arm, "fold": item.fold, "ok": False}
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    midi_path = work / f"{item.seg_id}.mid"
    wav_path = work / f"{item.seg_id}.wav"
    try:
        with zipfile.ZipFile(mid_zip) as zf:
            perf = read_performance(zf.read(item.midi_name))
        if arm == "deadpan":
            perf = deadpan(perf)
        row["source_notes"] = len(perf.notes)
        row["source_pedals"] = len(perf.pedals)

        write_performance_midi(perf, midi_path)
        render(midi_path, soundfont, wav_path)
        row["wav_bytes"] = wav_path.stat().st_size

        notes, _pedals = transcribe(wav_path)
        if not notes:
            raise ValueError(
                f"{item.seg_id} rendered to audio but transcribed to no notes; "
                f"an empty MIDI carries no difficulty signal")
        row["transcribed_notes"] = len(notes)
        _atomic_write_midi(notes, transcription_path(out_dir, item.seg_id))
        row["ok"] = True
    except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed; see docstring
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()[-2000:]
    finally:
        for path in (midi_path, wav_path):
            # A 44.1kHz render of the mean 150s piece is ~26MB; 900 of them is
            # 24GB, so the intermediates go unless explicitly kept for
            # inspection.
            if path.exists() and not keep_wav:
                path.unlink()
    return row


def append_row(out_dir: Path, row: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / ROWS_FILENAME).open("a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_rows(out_dir: Path) -> list[dict]:
    path = Path(out_dir) / ROWS_FILENAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_arm(items, arm: str, mid_zip: Path, soundfont: Path, out_dir: Path,
            transcribe, workers: int = 1, render=render_wav, log=print) -> int:
    """Render + transcribe every pending item. Returns the failure count.

    Threads, not processes: `transkun_cli.transcribe_wav` shells out to an
    isolated env via `subprocess.run`, so the GIL is released for the whole
    expensive part and the pool needs no pickling of the injected callables.
    The caller caps per-child thread counts (see `main`) -- Transkun's torch
    otherwise grabs every core in every worker and the workers fight.
    """
    todo = pending(items, out_dir)
    log(f"arm={arm}: {len(todo)} pending of {len(items)} "
        f"({len(items) - len(todo)} cached)")
    if not todo:
        return 0

    failures = 0
    done = 0
    lock = threading.Lock()

    def work(item: ArmItem) -> dict:
        return run_item(item, arm, mid_zip, soundfont, out_dir, transcribe,
                        render=render)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(work, todo):
            with lock:
                append_row(out_dir, row)
                done += 1
                if not row["ok"]:
                    failures += 1
                    log(f"  ITEM_FAILURE [{done}/{len(todo)}] {row['seg_id']}: "
                        f"{row.get('error')}")
                elif done % 25 == 0 or done == len(todo):
                    log(f"  [{done}/{len(todo)}] {failures} failures so far")
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=RENDERED_ARMS)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, required=True,
                    help="per-arm subdirectories are created under this")
    ap.add_argument("--soundfont", type=Path, default=None,
                    help="default: <data-root>/weights/soundfonts/FluidR3_GM.sf2")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads-per-worker", type=int, default=3)
    ap.add_argument("--timeout-s", type=int, default=3600,
                    help="per-item Transkun timeout; the default clears the "
                         "2285s eval outlier even under worker contention")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N pending items only; for smoke runs")
    args = ap.parse_args(argv)

    from claim_measurement.difficulty.audio_emb_extract import fold_of_seg_ids
    from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
    from claim_measurement.difficulty.bakeoff_paths import (
        features37_dir,
        features37_seg_ids,
        resolve_paths,
    )

    # Set before the first transcription subprocess is spawned; children
    # inherit it. Without this each worker's torch claims all 10 cores and the
    # pool is slower than running serially.
    os.environ["OMP_NUM_THREADS"] = str(args.threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(args.threads_per_worker)
    os.environ["TRANSKUN_TIMEOUT_S"] = str(args.timeout_s)

    paths = resolve_paths(args.data_root)
    soundfont = args.soundfont or (
        paths.labels.parents[1] / "weights" / "soundfonts" / "FluidR3_GM.sf2")

    seg_ids = features37_seg_ids(features37_dir(paths.emb_root))
    composers = np.array([
        read_embedding_npz(features37_dir(paths.emb_root) / f"{s}.npz").composer_id
        for s in seg_ids])
    fold_by_seg = fold_of_seg_ids(seg_ids, composers, N_FOLDS, SEED)
    manifest_rows = json.loads(Path(paths.manifest).read_text())

    items = plan_items(seg_ids, manifest_rows, fold_by_seg)
    if args.limit is not None:
        items = pending(items, Path(args.out_root) / args.arm)[:args.limit]

    from claim_measurement.difficulty.realaudio_check import _import_transcribe_wav

    failures = run_arm(items, args.arm, paths.labels.parent / "mid.zip", soundfont,
                       Path(args.out_root) / args.arm,
                       _import_transcribe_wav(), workers=args.workers)
    print(f"arm={args.arm} complete: {failures} failures")
    # Non-zero on any failure: an arm with holes cannot be scored, and a green
    # exit would let the evaluation run against a silently shorter population.
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
