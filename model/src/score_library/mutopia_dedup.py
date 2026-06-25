"""Content-based semantic dedup: Mutopia keyboard MIDIs vs the existing catalog.

String piece-IDs cannot detect that a Mutopia piece is the SAME musical work as
an existing catalog entry under different naming. So each Mutopia piece is run
through the SAME open-set matcher the production piece-ID gate uses (chroma top-K
recall -> pitch-only elastic subsequence DTW, lower cost = closer match, reusing
piece_id_eval.stage0c_elastic_dtwgate). A piece whose best-match cost is at/below
DUP_THRESHOLD is a duplicate of an existing work and dropped; the rest are
net-new and written to the candidates list for ingest.

DUP_THRESHOLD = 0.2885 was set at the natural gap (0.2591 -> 0.3179, ~3x the next
largest) in the bimodal best-cost distribution over the 529-piece wave, and audited:
every piece below it is a verified true-duplicate, the first above it a verified
net-new (a Czerny etude with only a spurious Scarlatti best-match).

Run:  cd model && uv run python -m score_library.mutopia_dedup
Reads <staging>/mutopia_keyboard_accepted.tsv; writes mutopia_dedup_costs.tsv +
mutopia_netnew_candidates.txt to <staging>.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from piece_id_eval.matchers.note_chroma_matcher import NoteChromaMatcher
from piece_id_eval.notes import Note, load_score_notes
from piece_id_eval.stage0c_elastic_dtwgate import ElasticGate, _notes_to_events
from score_library.parse import parse_score_midi

_MODEL_ROOT = Path(__file__).resolve().parents[2]
_SCORES_DIR = _MODEL_ROOT / "data" / "scores"
_STAGE = Path(os.environ.get("MUTOPIA_STAGING_DIR", str(Path.home() / "crescendai_corpus_staging")))
_MIDI_DIR = _STAGE / "mutopia_midi"
_ACCEPTED = _STAGE / "mutopia_keyboard_accepted.tsv"
_COSTS_OUT = _STAGE / "mutopia_dedup_costs.tsv"
_NETNEW_OUT = _STAGE / "mutopia_netnew_candidates.txt"

TOP_K = 5
W_PITCH = 1.0
W_TIME = 0.0  # pitch_only
DUP_THRESHOLD = 0.2885


def _score_to_notes(score_data) -> list[Note]:
    notes: list[Note] = []
    for bar in score_data.bars:
        for n in bar.notes:
            notes.append(Note(
                onset=n.onset_seconds,
                offset=n.onset_seconds + n.duration_seconds,
                pitch=n.pitch,
                velocity=n.velocity,
            ))
    notes.sort(key=lambda n: n.onset)
    return notes


def load_catalog() -> dict[str, list[Note]]:
    skip = {"titles.json", "seed.sql"}
    catalog: dict[str, list[Note]] = {}
    for path in sorted(f for f in _SCORES_DIR.glob("*.json") if f.name not in skip):
        catalog[path.stem] = load_score_notes(path)
    return catalog


def main() -> None:
    t0 = time.time()
    catalog = load_catalog()
    print(f"catalog: {len(catalog)} pieces", flush=True)
    chroma = NoteChromaMatcher(catalog)
    gate = ElasticGate(catalog)

    names = [l.split("\t")[0] for l in _ACCEPTED.read_text().splitlines() if l.strip()]
    print(f"mutopia accepted: {len(names)}", flush=True)

    results: list[dict] = []
    failed: list[tuple[str, str]] = []
    for i, fname in enumerate(names):
        midi = _MIDI_DIR / fname
        if not midi.exists():
            failed.append((fname, "MIDI not found"))
            continue
        try:
            notes = _score_to_notes(parse_score_midi(midi, fname[:-4], fname.split("__")[0], fname))
            if len(notes) < 2:
                raise ValueError(f"too few notes: {len(notes)}")
            top = [r.piece_id for r in chroma.rank(notes)[:TOP_K]]
            q_pc, q_li = _notes_to_events(notes)
            if q_pc.shape[0] < 2:
                raise ValueError("too few chord-events")
            costs = []
            for cid in top:
                c = gate.cost(q_pc, q_li, cid, W_PITCH, W_TIME)
                if c is not None and np.isfinite(c):
                    costs.append((cid, float(c)))
            if not costs:
                raise ValueError("no finite DTW costs")
        except Exception as e:  # noqa: BLE001 -- surface, never silently skip
            failed.append((fname, f"{type(e).__name__}: {e}"))
            continue
        costs.sort(key=lambda x: x[1])
        best_id, best_cost = costs[0]
        results.append({"f": fname, "cost": best_cost, "match": best_id})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)

    with open(_COSTS_OUT, "w") as f:
        f.write("mutopia_filename\tbest_cost\tbest_match_catalog_piece_id\n")
        for r in results:
            f.write(f"{r['f']}\t{r['cost']:.6f}\t{r['match']}\n")

    netnew = [r["f"] for r in results if r["cost"] > DUP_THRESHOLD]
    dups = len(results) - len(netnew)
    _NETNEW_OUT.write_text("\n".join(netnew) + "\n")

    print(f"\nDUP (<= {DUP_THRESHOLD}): {dups}   NET-NEW: {len(netnew)}   FAILED: {len(failed)}")
    if failed:
        print("-- failed (fail-loud) --")
        for fn, r in failed[:20]:
            print(f"  {fn}: {r}")
    print(f"wrote net-new candidates -> {_NETNEW_OUT}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
