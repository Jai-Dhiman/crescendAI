"""Render recognize-only Mutopia keyboard pieces to standalone SVG via LilyPond.

Mutopia ships LilyPond .ly source (no clean kern/MEI path), so these pieces are
display-less in the Verovio (.mei/.mxl) tiers. LilyPond 2.26 renders the legacy
.ly directly (`-dcrop` -> a single whole-piece cropped SVG, ideal for inline
scrolling display). Output is LOCAL-ONLY: Mutopia engravings are CC-BY-SA / CC-BY,
not PD, so they are not pushed to prod R2 and the UI must show attribution.

Mapping: the Mutopia keyboard manifest row's third column is
`<Composer>__<Work>__<File>.mid`; the .ly sits beside the .mid in the local
mutopia ftp clone, and the catalog piece_id is `mutopia.` + the sanitized
composer/work/file. A piece needs an SVG iff it has a score JSON but no .mei.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_SCORES = _MODEL_ROOT / "data" / "scores"
_MEI = _MODEL_ROOT / "scores" / "v1"
_SVG_OUT = _MODEL_ROOT / "scores" / "v1"
_MUTOPIA = Path.home() / "crescendai_corpus_staging" / "mutopia"
_MIDI_STAGE = Path.home() / "crescendai_corpus_staging" / "mutopia_midi"


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _ly_for_stem(stem: str) -> Path | None:
    """Locate the .ly for a staged .mid whose name flattens the ftp path with `__`
    (e.g. BeethovenLv__WoO55__prelude_WoO55__prelude_WoO55.mid -> ftp/BeethovenLv/
    WoO55/prelude_WoO55/prelude_WoO55.ly)."""
    rel = "/".join(stem.split("__"))
    for cand in (_MUTOPIA / "ftp" / f"{rel}.ly", _MUTOPIA / f"{rel}.ly"):
        if cand.exists():
            return cand
    parent = (_MUTOPIA / "ftp" / rel).parent
    lys = sorted(parent.glob("*.ly")) if parent.exists() else []
    return lys[0] if lys else None


def iter_candidates(midi_stage: Path | None = None):
    """Yield (piece_id, ly_path) for recognize-only Mutopia pieces: a staged .mid
    whose piece_id (`mutopia.<sanitized stem>`, matching mutopia_ingest) has a
    score JSON, no .mei, and a locatable .ly."""
    stage = midi_stage or _MIDI_STAGE
    for mid in sorted(stage.glob("*.mid")):
        stem = mid.stem
        pid = f"mutopia.{_sanitize(stem)}"
        if not (_SCORES / f"{pid}.json").exists():
            continue
        if (_MEI / f"{pid}.mei").exists():
            continue
        ly = _ly_for_stem(stem)
        if ly is not None:
            yield pid, ly


def render_svg(pid: str, ly: Path, out_dir: Path, work_dir: Path) -> tuple[bool, str]:
    """Render a .ly to <out_dir>/<pid>.svg (whole-piece cropped). Returns (ok, msg)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = work_dir / "score"
    try:
        proc = subprocess.run(
            ["lilypond", "-dcrop", "-dbackend=svg", "-o", str(stem), str(ly)],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    cropped = work_dir / "score.cropped.svg"
    if not cropped.exists():
        tail = (proc.stderr or proc.stdout or "")[-200:]
        return False, f"no svg (exit {proc.returncode}): {tail}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cropped.replace(out_dir / f"{pid}.svg")
    # clean per-page svgs
    for p in work_dir.glob("score*.svg"):
        p.unlink(missing_ok=True)
    return True, "ok"


def main() -> None:
    """Render every recognize-only Mutopia keyboard piece to <out-dir>/<pid>.svg.
    LOCAL-ONLY: these engravings are CC-BY-SA/CC-BY (not PD) -- never push to prod
    R2; the UI shows Mutopia attribution for them."""
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description="Render Mutopia .ly -> SVG (local-only).")
    ap.add_argument("--scores-dir", type=Path, help="override score JSON dir")
    ap.add_argument("--mei-dir", type=Path, help="override .mei dir")
    ap.add_argument("--midi-stage", type=Path, help="override staged .mid dir")
    ap.add_argument("--out-dir", type=Path, help="SVG output dir (default scores/v1)")
    ap.add_argument("--limit", type=int, default=0, help="render at most N (0=all)")
    args = ap.parse_args()

    global _SCORES, _MEI
    if args.scores_dir:
        _SCORES = args.scores_dir
    if args.mei_dir:
        _MEI = args.mei_dir
    out_dir = args.out_dir or _SVG_OUT

    cands = list(iter_candidates(args.midi_stage))
    if args.limit:
        cands = cands[: args.limit]
    print(f"rendering {len(cands)} Mutopia pieces -> SVG (LOCAL-ONLY, CC-BY-SA)")
    ok = 0
    fails: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for i, (pid, ly) in enumerate(cands):
            success, msg = render_svg(pid, ly, out_dir, work)
            if success:
                ok += 1
            else:
                fails.append((pid, msg))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(cands)} ({ok} ok, {len(fails)} failed)")
    print(f"\nDONE: {ok}/{len(cands)} rendered; {len(fails)} failed")
    for pid, msg in fails[:20]:
        print(f"  FAIL {pid}: {msg}")


if __name__ == "__main__":
    main()
