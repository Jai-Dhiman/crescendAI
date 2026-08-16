"""Issue #169 part 1: recover the missing PSyllabus recordings.

Paced, resumable, cookie-passing refetch over missing.json (manifest seg_ids
minus transkun_mid stems), with Transkun transcription pipelined behind the
downloader. WAVs are deleted after their transcription succeeds; the MIDI is
the artifact. Every per-piece outcome is appended to recover_report.jsonl --
a silent skip would unpair any later measurement.

Run from anywhere; pass --data-dir when the checkout does not hold the data
(worktrees do not see model/data):

  python3 recover_missing.py --data-dir /path/to/model/data/results/amt_gap_curve \
      --cookies-from-browser chrome --limit 20 --prior-failures-first
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "results" / "amt_gap_curve"
)
TRANSKUN_TIMEOUT_S = int(os.environ.get("TRANSKUN_TIMEOUT_S", "3600"))

_REPORT_LOCK = threading.Lock()


def _log(report_path: Path, **row):
    row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _REPORT_LOCK, open(report_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _download(m, wav_dir: Path, cookies_browser: str | None):
    wav = wav_dir / f"{m['seg_id']}.wav"
    if wav.exists():
        return True, "cached", False
    url = f"https://www.youtube.com/watch?v={m['video_id']}"
    cmd = ["yt-dlp", "-q", "--no-warnings", "-f", "bestaudio", "-x",
           "--audio-format", "wav", "--postprocessor-args", "-ar 24000 -ac 1",
           "-o", str(wav_dir / f"{m['seg_id']}.%(ext)s")]
    if cookies_browser:
        cmd += ["--cookies-from-browser", cookies_browser]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and wav.exists()
    return ok, (r.stderr or "").strip()[-200:], True


def _transcribe(wav: Path, tk_dir: Path):
    out = tk_dir / f"{wav.stem}.mid"
    cmd = ["uv", "run", "--no-project", "--with", "transkun", "--with", "setuptools",
           "--python", "3.11", "transkun", str(wav), str(out), "--device", "cpu"]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TRANSKUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {TRANSKUN_TIMEOUT_S}s"
    return (r.returncode == 0 and out.exists()), (r.stderr or "").strip()[-200:]


def _tk_worker(q: queue.Queue, tk_dir: Path, report_path: Path, stats: Counter):
    while True:
        item = q.get()
        if item is None:
            return
        wav, grade = item
        ok, err = _transcribe(wav, tk_dir)
        if ok:
            wav.unlink()
            stats["transcribed"] += 1
        else:
            stats["tk_failed"] += 1
            print(f"  TK FAIL {wav.stem[:50]}: {err[-90:]}", flush=True)
        _log(report_path, seg_id=wav.stem, grade=grade, stage="transcribe", ok=ok,
             reason=None if ok else err)


def main():
    ap = argparse.ArgumentParser(
        description="Recover missing PSyllabus recordings (issue #169)")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--cookies-from-browser", default=None,
                    help="browser whose YouTube cookies yt-dlp uses (e.g. chrome)")
    ap.add_argument("--sleep-min", type=float, default=10.0)
    ap.add_argument("--sleep-max", type=float, default=30.0)
    ap.add_argument("--tk-workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None,
                    help="attempt at most N downloads (smoke)")
    ap.add_argument("--prior-failures-first", action="store_true",
                    help="order the 191 refetch_report.json failures ahead of the rest")
    ap.add_argument("--exclude-prior-failures", action="store_true",
                    help="skip the 191 refetch_report.json failures entirely")
    args = ap.parse_args()

    data = args.data_dir
    wav_dir, tk_dir = data / "wav", data / "transkun_mid"
    report_path = data / "recover_report.jsonl"
    missing = json.loads((data / "missing.json").read_text())

    todo = [m for m in missing if not (tk_dir / f"{m['seg_id']}.mid").exists()]
    if args.prior_failures_first or args.exclude_prior_failures:
        prior = {f["seg_id"] for f in json.loads(
            (data / "refetch_report.json").read_text())["failures"]}
        if args.exclude_prior_failures:
            todo = [m for m in todo if m["seg_id"] not in prior]
        else:
            todo.sort(key=lambda m: m["seg_id"] not in prior)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} pieces to recover (of {len(missing)} missing); "
          f"cookies={args.cookies_from_browser or 'none'} "
          f"sleep={args.sleep_min}-{args.sleep_max}s tk_workers={args.tk_workers}",
          flush=True)

    stats = Counter()
    q: queue.Queue = queue.Queue()
    with ThreadPoolExecutor(max_workers=args.tk_workers) as ex:
        for _ in range(args.tk_workers):
            ex.submit(_tk_worker, q, tk_dir, report_path, stats)
        for i, m in enumerate(todo, 1):
            ok, reason, hit_network = _download(m, wav_dir, args.cookies_from_browser)
            _log(report_path, seg_id=m["seg_id"], grade=m["grade"], stage="download",
                 ok=ok, video_id=m["video_id"], reason=None if ok else reason)
            if ok:
                stats["downloaded"] += 1
                q.put((wav_dir / f"{m['seg_id']}.wav", m["grade"]))
            else:
                stats["dl_failed"] += 1
                print(f"  DL FAIL {m['seg_id'][:50]}: {reason[-90:]}", flush=True)
            if i % 20 == 0 or i == len(todo):
                print(f"  progress {i}/{len(todo)} {dict(stats)}", flush=True)
            if hit_network and i < len(todo):
                time.sleep(random.uniform(args.sleep_min, args.sleep_max))
        for _ in range(args.tk_workers):
            q.put(None)
    print(f"done: {dict(stats)} -> report {report_path}", flush=True)


if __name__ == "__main__":
    main()
