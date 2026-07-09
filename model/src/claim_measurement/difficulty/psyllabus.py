"""PianoSyllabus dataset loader for MIREX 2026 difficulty (#104).

PSyllabus (Zenodo 14794592) is the primary audio-task label source: 7,901 pieces,
each with an integer difficulty grade `ps` in 0..10 (11 levels) and a matching MIDI
in `mid.zip` named `mid/<key>.mid` where <key> is the label-dict key. We train on
these MIDIs directly (skip AMT); at test the organizers give raw audio that our Docker
runs through aria-amt, so there is a train(their MIDI)/test(aria-amt) transcriber gap
to check separately -- but the difficulty *signal* is measured here first.

`build_records` is pure (label dict + available MIDI names -> records) so grade coercion
and MIDI-presence filtering are unit-testable without the 70MB zip. `composer_disjoint_split`
guarantees no composer straddles the train/test boundary -- the safe self-eval protocol
given the private MIREX test set (captains did not confirm public/private composer overlap).
"""
from __future__ import annotations

import io
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PsRecord:
    key: str
    grade: int          # ps, 0..10
    composer: str
    duration: float     # seconds (from the label JSON; 0.0 if absent)
    midi_name: str       # name inside the zip, e.g. "mid/Faure G....mid"


def build_records(labels: dict, available_midis: set[str]) -> tuple[list[PsRecord], dict]:
    """Pure: label dict + set of MIDI names in the zip -> (records, skip_report).

    Skips (loudly, via the report) any entry whose grade is not an int-coercible value
    or whose MIDI is absent from the zip. No silent drops -- the report counts every loss.
    """
    records: list[PsRecord] = []
    skipped_no_midi = 0
    skipped_bad_grade = 0
    for key, meta in labels.items():
        midi_name = f"mid/{key}.mid"
        if midi_name not in available_midis:
            skipped_no_midi += 1
            continue
        raw = meta.get("ps")
        try:
            grade = int(raw)
        except (TypeError, ValueError):
            skipped_bad_grade += 1
            continue
        duration = float(meta.get("duration") or 0.0)
        records.append(PsRecord(key=key, grade=grade,
                                composer=str(meta.get("composer", "")).strip(),
                                duration=duration, midi_name=midi_name))
    report = {"n_records": len(records), "n_labels": len(labels),
              "skipped_no_midi": skipped_no_midi, "skipped_bad_grade": skipped_bad_grade}
    return records, report


def load_records(labels_path: Path, mid_zip_path: Path) -> tuple[list[PsRecord], dict]:
    """Read the label JSON + the MIDI-zip's name list, then `build_records`."""
    labels = json.loads(Path(labels_path).read_text())
    with zipfile.ZipFile(mid_zip_path) as zf:
        available = {n for n in zf.namelist() if n.endswith(".mid")}
    return build_records(labels, available)


def notes_from_midi_bytes(raw: bytes) -> list[dict]:
    """Note dicts {pitch,onset,offset,velocity} from raw MIDI bytes (order by onset)."""
    import pretty_midi  # lazy: keep the pure loader/split functions dependency-light
    pm = pretty_midi.PrettyMIDI(io.BytesIO(raw))
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append({"pitch": int(n.pitch), "onset": float(n.start),
                          "offset": float(n.end), "velocity": int(n.velocity)})
    notes.sort(key=lambda d: d["onset"])
    return notes


def read_notes(mid_zip_path: Path, midi_name: str) -> list[dict]:
    """Note dicts for one MIDI, read straight from the zip (reopens the zip per call --
    for bulk iteration open the ZipFile once and use `notes_from_midi_bytes`)."""
    with zipfile.ZipFile(mid_zip_path) as zf:
        raw = zf.read(midi_name)
    return notes_from_midi_bytes(raw)


def composer_disjoint_split(
    records: list[PsRecord], test_frac: float = 0.2, seed: int = 2026
) -> tuple[list[PsRecord], list[PsRecord]]:
    """Split so that NO composer appears in both train and test.

    Composers are shuffled deterministically and whole composers are assigned to the
    test side until it reaches ~test_frac of records. This is the composer-disjoint
    protocol (A9): difficulty must generalize beyond memorizing a composer's style.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    by_composer: dict[str, list[PsRecord]] = defaultdict(list)
    for r in records:
        by_composer[r.composer].append(r)
    composers = sorted(by_composer)
    # deterministic shuffle without Math.random-style state: seed a numpy Generator
    import numpy as np
    order = np.random.default_rng(seed).permutation(len(composers))
    target_test = int(round(len(records) * test_frac))
    test: list[PsRecord] = []
    test_composers: set[str] = set()
    for idx in order:
        if len(test) >= target_test:
            break
        c = composers[idx]
        test.extend(by_composer[c])
        test_composers.add(c)
    train = [r for r in records if r.composer not in test_composers]
    return train, test
