"""Disaster-recovery: regenerate per-piece score JSONs (data/scores/<id>.json)
from the surviving fingerprint catalog + staging source MIDIs.

The fingerprint (piece_index.json) is the authoritative piece set (it carries
piece_id + composer + title per piece). For each piece we resolve its source MIDI
and re-run parse_score_midi -- deterministic, so the regenerated score is faithful
and the note set matches the fingerprint (bypasses re-dedup). Resumable: skips any
piece whose JSON already exists.

Resolvers (the 92% with a clean MIDI<->piece_id map):
  giantmidi : pid = giantmidi.<surname>.<title>_<ytid>; ytid = last 11 chars
              (YouTube IDs may contain '_' so split('_')[-1] is WRONG).
  pdmx      : pid = pdmx.<comp>.<title>_<cid8>; cid8 = first 8 of the on-disk CID.
  mutopia   : pid = mutopia.<sanitized staged-midi stem>.
Kern-family pieces (chopin/scriabin/beethoven/bach/...) have a path-based pid and
are NOT handled here -- recover their JSONs from R2 or re-run the kernscores/
fullcorpus ingest recipes.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from score_library.parse import parse_score_midi

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_SCORES = _MODEL_ROOT / "data" / "scores"
_STAGE = Path.home() / "crescendai_corpus_staging"
_YTID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _san(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "untitled"


def _build_indices() -> dict[str, dict]:
    gm: dict[str, Path] = {}
    for sub in ["full_midis/midis", "surname_midis/surname_checked_midis"]:
        d = _STAGE / "giantmidi/GiantMIDI-PIano" / sub
        if d.exists():
            for m in d.glob("*.mid"):
                parts = m.stem.split(",")
                if len(parts) >= 3 and _YTID.match(parts[-1].strip()):
                    gm.setdefault(parts[-1].strip(), m)
    pdmx = {m.stem[:8]: m for m in (_STAGE / "pdmx/extracted/mid").rglob("*.mid")}
    muto = {f"mutopia.{_san(m.stem)}": m for m in (_STAGE / "mutopia_midi").glob("*.mid")}
    return {"giantmidi": gm, "pdmx": pdmx, "mutopia": muto}


def _resolve(pid: str, idx: dict) -> Path | None:
    src = pid.split(".")[0]
    if src == "giantmidi":
        return idx["giantmidi"].get(pid[-11:])
    if src == "pdmx":
        return idx["pdmx"].get(pid.split("_")[-1][:8])
    if src == "mutopia":
        return idx["mutopia"].get(pid)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate score JSONs from fingerprint + staging MIDIs.")
    ap.add_argument("--fingerprint", required=True, type=Path, help="piece_index.json")
    ap.add_argument("--scores-dir", type=Path, default=_SCORES)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.scores_dir.mkdir(parents=True, exist_ok=True)

    pi = json.load(open(args.fingerprint))
    pieces = pi.get("pieces", pi)
    entries = (
        [(k, v.get("composer", ""), v.get("title", "")) for k, v in pieces.items()]
        if isinstance(pieces, dict)
        else [(p["piece_id"], p.get("composer", ""), p.get("title", "")) for p in pieces]
    )
    idx = _build_indices()
    print(f"indices: giantmidi={len(idx['giantmidi'])} pdmx={len(idx['pdmx'])} mutopia={len(idx['mutopia'])}", flush=True)

    todo = []
    for pid, composer, title in entries:
        if (args.scores_dir / f"{pid}.json").exists():
            continue
        midi = _resolve(pid, idx)
        if midi is not None:
            todo.append((pid, composer, title, midi))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} score JSONs to regenerate (resolvable + missing)", flush=True)

    ok = 0
    fails: list[tuple[str, str]] = []
    done = 0

    def task(item):
        pid, composer, title, midi = item
        try:
            score = parse_score_midi(str(midi), pid, composer, title)
            (args.scores_dir / f"{pid}.json").write_text(json.dumps(score.model_dump(), indent=2))
            return pid, True, ""
        except Exception as e:  # noqa: BLE001 -- surface every parse failure
            return pid, False, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for fut in as_completed([ex.submit(task, it) for it in todo]):
            pid, success, msg = fut.result()
            done += 1
            if success:
                ok += 1
            else:
                fails.append((pid, msg))
            if done % 500 == 0:
                print(f"  {done}/{len(todo)} ({ok} ok, {len(fails)} failed)", flush=True)
    print(f"\nDONE: {ok}/{len(todo)} regenerated; {len(fails)} failed")
    for pid, msg in fails[:20]:
        print(f"  FAIL {pid}: {msg}")


if __name__ == "__main__":
    main()
