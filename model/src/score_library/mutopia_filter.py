"""Authoritative solo-keyboard filter for the Mutopia second-wave MIDIs.

Mutopia exports nearly every LilyPond score to GM program 0, so a MIDI-program
introspection cannot tell a string quartet from a piano sonata. The real
instrumentation lives in the source `.ly` header field `mutopiainstrument`.
This maps each downloaded MIDI -> its .ly source (manifest URL -> local clone)
and keeps ONLY pieces whose instrument tokens are all keyboard (piano /
harpsichord / clavichord / clavier / cembalo / fortepiano, incl. duets). Organ
is EXCLUDED (pedalboard, not pianistic); Voice/Guitar/strings/orchestra rejected.

Fail-loud: a MIDI with no manifest entry or no readable instrument is reported
UNKNOWN and excluded, never silently accepted.

Run:  cd model && uv run python -m score_library.mutopia_filter
Writes <staging>/mutopia_keyboard_accepted.tsv (filename<TAB>instrument).
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _MODEL_ROOT / "data" / "manifests" / "mutopia_keyboard_manifest.tsv"
_STAGE = Path(os.environ.get("MUTOPIA_STAGING_DIR", str(Path.home() / "crescendai_corpus_staging")))
_MIDI_DIR = _STAGE / "mutopia_midi"
_FTP = _STAGE / "mutopia" / "ftp"
_ACCEPTED_OUT = _STAGE / "mutopia_keyboard_accepted.tsv"
_REJECTED_OUT = _STAGE / "mutopia_nonkeyboard_rejected.tsv"

KEYBOARD = {"piano", "harpsichord", "clavichord", "clavier", "cembalo", "fortepiano"}
MODIFIERS = {"duet", "solo", "duo", "4", "2", "hands", "hand", "four", "two", "and", "or", "left", "right"}
_INSTR_RE = re.compile(r'mutopiainstrument\s*=\s*"([^"]*)"', re.IGNORECASE)


def load_manifest() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _MANIFEST.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            out[parts[2].strip()] = parts[1].strip()
    return out


def instrument_for(url: str) -> str | None:
    rel = url.replace("https://www.mutopiaproject.org/ftp/", "")
    d = (_FTP / rel).parent
    if not d.exists():
        return None
    for src in list(d.glob("*.ly")) + list(d.glob("*.ily")) + list(d.rglob("*.ly")):
        m = _INSTR_RE.search(src.read_text(errors="ignore"))
        if m:
            return m.group(1).strip()
    return None


def is_keyboard(instr: str) -> bool:
    toks = [t for t in re.split(r"[,\s]+", instr.lower()) if t and t not in MODIFIERS]
    return bool(toks) and all(t in KEYBOARD for t in toks)


def main() -> None:
    if not _MIDI_DIR.exists():
        raise SystemExit(f"ABORT: {_MIDI_DIR} missing -- run acquire_mutopia.sh first")
    manifest = load_manifest()
    midis = sorted(_MIDI_DIR.glob("*.mid"))
    if not midis:
        raise SystemExit(f"ABORT: no MIDIs in {_MIDI_DIR}")
    print(f"{len(midis)} MIDIs; {len(manifest)} manifest rows", flush=True)

    accepted: list[tuple[str, str]] = []
    rejected: list[tuple[str, str]] = []
    unknown: list[str] = []
    for p in midis:
        url = manifest.get(p.name)
        if url is None:
            unknown.append(f"{p.name}\tno manifest entry")
            continue
        instr = instrument_for(url)
        if instr is None:
            unknown.append(f"{p.name}\tno mutopiainstrument in source")
            continue
        (accepted if is_keyboard(instr) else rejected).append((p.name, instr))

    print(f"ACCEPT (keyboard): {len(accepted)}")
    print(f"REJECT (non-keyboard): {len(rejected)}")
    print(f"UNKNOWN (fail-loud, excluded): {len(unknown)}")

    acc_instr: dict[str, int] = defaultdict(int)
    for _, i in accepted:
        acc_instr[i] += 1
    print("\n-- accepted instrument breakdown --")
    for k in sorted(acc_instr, key=lambda x: -acc_instr[x]):
        print(f"  {acc_instr[k]:4d}  {k}")

    _ACCEPTED_OUT.write_text("\n".join(f"{n}\t{i}" for n, i in accepted) + "\n")
    _REJECTED_OUT.write_text("\n".join(f"{n}\t{i}" for n, i in rejected) + "\n")
    if unknown:
        (_STAGE / "mutopia_unknown.tsv").write_text("\n".join(unknown) + "\n")
    print(f"\nwrote {len(accepted)} accepted -> {_ACCEPTED_OUT}")


if __name__ == "__main__":
    main()
