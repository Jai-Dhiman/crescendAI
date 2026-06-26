"""Stage craigsapp **kern repos as per-piece MIDI for the piece-ID catalog.

The catalog-expansion pipeline (kernscores_expand.py) reads score-engraved
MIDIs from ~/crescendai_corpus_staging/kernscores_midi/<collection>/. This
module is the durable, reproducible record of how that staging directory is
populated: clone (or pull) the craigsapp humdrum repo, then convert every
source **kern to MIDI via Verovio's native Humdrum importer, preserving the
.krn stem so kernscores_expand's piece_id functions line up with
render_kern_assets' .krn -> MEI render of the same stem.

All sources are unambiguously public domain (composers pre-1928) and
commercial-clean (craigsapp humdrum is released into the public domain).

Per-repo `kern_glob` handles the layout variants:
  - most repos keep .krn in `kern/` (kern_glob="kern/*.krn")
  - scriabin nests by opus (kern_glob="op*/*.krn")

Stems must be unique within a repo (the staged MIDI keyspace is flat). A
collision aborts loudly rather than silently overwriting.

Run:  cd model && uv run python -m score_library.kernscores_stage
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import verovio

_CLONE = Path(os.environ.get("KERNSCORES_CLONE_DIR", str(Path.home() / "crescendai_corpus_staging" / "kernscores")))
_MIDI_BASE = Path(os.environ.get("KERNSCORES_STAGING_DIR", str(Path.home() / "crescendai_corpus_staging"))) / "kernscores_midi"

# repo -> source **kern glob (relative to the repo root). New catalog
# collections only; the original six are already staged on disk.
_REPOS: list[tuple[str, str]] = [
    ("hummel-preludes", "kern/*.krn"),
    ("art-of-the-fugue", "kern/*.krn"),
    ("scriabin", "op*/*.krn"),
]


def stage_repo(repo: str, kern_glob: str) -> tuple[int, int]:
    """Convert all matching .krn in a cloned repo to MIDI under the staging dir.

    Verovio's Humdrum importer segfaults (SIGSEGV) on a small number of
    pathological **kern files, which a try/except cannot catch (it kills the
    process). This loop is therefore SEGFAULT-RESUMABLE: before each conversion
    it drops a `<stem>.inprogress` sentinel; a sentinel still present on the
    next run means that file crashed the interpreter, so it is promoted to a
    permanent `<stem>.segfault` skip-marker and excluded LOUDLY. Successful
    MIDIs are skip-existing, so re-running resumes past completed work. Wrap the
    module in an `until` loop (the just recipe does) to drive it to completion.

    Returns (written_this_run, already_done). Raises SystemExit on a missing
    clone, zero source files, or a stem collision (fail-loud).
    """
    repo_dir = _CLONE / repo
    if not repo_dir.exists():
        raise SystemExit(f"ABORT: clone missing at {repo_dir} -- clone craigsapp/{repo} first")

    krns = sorted(repo_dir.glob(kern_glob))
    if not krns:
        raise SystemExit(f"ABORT: no .krn matched {repo}/{kern_glob}")

    out_dir = _MIDI_BASE / repo
    out_dir.mkdir(parents=True, exist_ok=True)

    tk = verovio.toolkit()
    tk.setInputFrom("humdrum")

    seen: dict[str, Path] = {}
    ok = done = 0
    segfaults: list[str] = []
    errors: list[str] = []
    for krn in krns:
        if krn.stem in seen:
            raise SystemExit(
                f"ABORT: duplicate stem {krn.stem!r} in {repo} "
                f"({krn} vs {seen[krn.stem]}) -- flat MIDI keyspace would collide"
            )
        seen[krn.stem] = krn
        mid = out_dir / (krn.stem + ".mid")
        segfault_marker = out_dir / (krn.stem + ".segfault")
        inprogress = out_dir / (krn.stem + ".inprogress")

        if mid.exists():
            done += 1
            continue
        if segfault_marker.exists():
            segfaults.append(f"{repo}/{krn.name}")
            continue
        if inprogress.exists():
            # The previous run crashed (SIGSEGV) on this exact file: promote to
            # a permanent skip and surface it loudly.
            inprogress.rename(segfault_marker)
            segfaults.append(f"{repo}/{krn.name} (SIGSEGV)")
            continue

        inprogress.write_bytes(b"")  # flushed to disk before the risky call
        try:
            if not tk.loadFile(str(krn)):
                raise RuntimeError("verovio loadFile returned False")
            mid.write_bytes(base64.b64decode(tk.renderToMIDI()))
        except Exception as e:  # noqa: BLE001 -- surface every catchable failure
            inprogress.unlink(missing_ok=True)
            errors.append(f"{repo}/{krn.name}: {e}")
            continue
        inprogress.unlink(missing_ok=True)
        ok += 1

    print(f"  {repo:22s} new={ok:3d} done={done:3d} segfault={len(segfaults):2d} err={len(errors):2d}  -> {out_dir}")
    for f in segfaults:
        print(f"    SKIP(segfault) {f}")
    for f in errors:
        print(f"    FAIL {f}")
    total = ok + done
    if total == 0:
        raise SystemExit(f"ABORT: {repo} produced ZERO MIDIs")
    return ok, done


def main() -> None:
    new = total = 0
    for repo, kern_glob in _REPOS:
        n, d = stage_repo(repo, kern_glob)
        new += n
        total += n + d
    print(f"\nStaged {total} MIDIs into {_MIDI_BASE}/ ({new} new this run) across {len(_REPOS)} collections.")


if __name__ == "__main__":
    main()
