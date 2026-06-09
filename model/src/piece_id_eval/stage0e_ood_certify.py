# model/src/piece_id_eval/stage0e_ood_certify.py
"""Stage-0e (H3): certify the gate on DIVERSE, long, leave-no-leak out-of-catalog negatives (#26).

Stage-0d verdict was PASS_POINT_ONLY: the frozen pitch-only elastic margin gate
held TA>=0.60 with near-zero false-accept, but could not be CERTIFIED at FA<=0.05
because the negatives were too few/narrow -- 16 in-catalog leave-one-out works
(H1) + only 3 short out-of-catalog works (H2, PercePiano). Certifying a <=0.05
rate from zero observed false-accepts needs >=60 INDEPENDENT negative works
(rule of three: 3/60 = 0.05).

This experiment supplies that: full-length MAESTRO performances of works whose
canonical_composer is NOT one of the 16 catalog composers -- so they are provably
out-of-catalog with ZERO title-matching / alias risk (the join key is the clean
`canonical_composer` categorical, not free-text titles). ASAP/MAESTRO finding that
forced this design: the 254-piece catalog IS the ASAP repertoire (242/242 works
matched), so the only out-of-catalog canon is MAESTRO minus ASAP; the same-composer
slice needs a leak-prone free-text de-dup, so it is EXCLUDED from certification here
and left as future work. The foreign-composer slice is the certifiable tier.

Gate is FROZEN to the Stage-0c/0d winner (pitch-only chord-Jaccard elastic margin).
H1 in-catalog LOO is carried over UNCHANGED from Stage-0d; only the OOD set changes.

HONEST SCOPE: foreign-composer OOD is the "different repertoire" rejection case
(a user plays Scarlatti we don't have). It is EASIER than the hardest adversary
(same composer, different piece), which is covered only by H1's 16 in-catalog works.
A clean PASS here certifies the gate rejects diverse foreign repertoire at scale; it
does NOT by itself certify same-composer rejection at scale.

SUCCESS: an operating point with TA>=0.60, both FA upper-95%-CI <= 0.05, and the OOD
axis having >=60 independent works (so certification is mathematically possible).

Run:  cd model && PYTHONUNBUFFERED=1 caffeinate -i uv run python -m piece_id_eval.stage0e_ood_certify
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import partitura as pa

from piece_id_eval.matchers.note_chroma_matcher import NoteChromaMatcher
from piece_id_eval.notes import Note
from piece_id_eval.stage0c_elastic_dtwgate import ElasticGate, load_data
from piece_id_eval.stage0d_gate_hardening import (
    _MIN_CLUSTERS_TO_CERTIFY,
    _MIN_TA,
    _WINDOW_SECONDS,
    _collect_in_catalog,
    _evcount_summary,
    _margin,
    _operating_point,
)

_MODULE_DIR = Path(__file__).resolve().parent
_MODEL_ROOT = _MODULE_DIR.parents[1]
_MAESTRO_DIR = _MODEL_ROOT / "data/raw/maestro-v3.0.0"
_MAESTRO_CSV = _MAESTRO_DIR / "maestro-v3.0.0.csv"
_OUTPUT = _MODEL_ROOT / "data/evals/piece_id/stage0e_ood_certify_results.json"

# The 16 catalog composers (lowercased substrings of canonical_composer).
_CAT_COMPOSERS = frozenset({
    "bach", "beethoven", "brahms", "chopin", "debussy", "glinka", "haydn", "liszt",
    "mozart", "prokofiev", "rachmaninoff", "ravel", "schubert", "schumann", "scriabin", "balakirev",
})
_MAX_OOD_EVENTS = 1500  # cap query length to bound elastic-DTW cost-matrix size (~positive scale)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _is_foreign(canonical_composer: str) -> bool:
    """True iff the performer's canonical_composer contains none of the 16 catalog composers."""
    c = canonical_composer.lower()
    return not any(name in c for name in _CAT_COMPOSERS)


def _load_perf_midi(path: Path, cap_events: int) -> list[Note]:
    ppart = pa.load_performance_midi(str(path))
    na = ppart.note_array()
    notes = [
        Note(onset=float(r["onset_sec"]), offset=float(r["onset_sec"]) + max(float(r["duration_sec"]), 1e-3),
             pitch=int(r["pitch"]), velocity=int(r["velocity"]))
        for r in na
    ]
    notes.sort(key=lambda n: n.onset)
    if cap_events and len(notes) > cap_events:
        notes = notes[:cap_events]  # first N notes ~ a production-length window
    return notes


def _load_foreign_ood(full_chroma: NoteChromaMatcher, gate: ElasticGate) -> list[dict]:
    """One MAESTRO performance per foreign-composer canonical_title -> margin vs full catalog.

    Each title is one independent OOD work (cluster). Any accept is a false accept.
    """
    rows = list(csv.DictReader(_MAESTRO_CSV.open()))
    # pick one performance per foreign canonical_title (the first encountered)
    by_title: dict[str, dict] = {}
    for r in rows:
        if _is_foreign(r["canonical_composer"]) and r["canonical_title"] not in by_title:
            by_title[r["canonical_title"]] = r
    if len(by_title) < _MIN_CLUSTERS_TO_CERTIFY:
        _log(f"[warn] only {len(by_title)} foreign OOD works (< {_MIN_CLUSTERS_TO_CERTIFY} to certify)")

    out: list[dict] = []
    t0 = time.time()
    for i, (title, r) in enumerate(sorted(by_title.items())):
        midi = _MAESTRO_DIR / r["midi_filename"]
        if not midi.exists():
            _log(f"[skip] missing {r['midi_filename']}")
            continue
        notes = _load_perf_midi(midi, _MAX_OOD_EVENTS)
        res = _margin(notes, full_chroma, gate)
        if res is not None:
            m, _bid, n_ev = res
            out.append({"margin": m, "work": title, "composer": r["canonical_composer"], "n_ev": n_ev})
        if (i + 1) % 25 == 0:
            _log(f"[ood] {i+1}/{len(by_title)} works {time.time()-t0:.1f}s")
    _log(f"[ood] {len(out)} foreign-composer OOD works from {len(by_title)} titles {time.time()-t0:.1f}s")
    return out


def main() -> None:
    t_start = time.time()
    catalog, recordings = load_data()
    _log(f"[build] chroma + elastic index over {len(catalog)} catalog pieces ...")
    full_chroma = NoteChromaMatcher(catalog)
    gate = ElasticGate(catalog)

    _log("\n=== H3: diverse foreign-composer out-of-catalog (MAESTRO) ===")
    ood = _load_foreign_ood(full_chroma, gate)
    from collections import Counter
    ood_composers = Counter(o["composer"].split()[-1] for o in ood)

    results: dict[str, dict] = {}
    for window_seconds, mode_label in [(None, "full"), (_WINDOW_SECONDS, "90s")]:
        _log(f"\n=== H1 (carried from Stage-0d): in-catalog LOO (mode={mode_label}) ===")
        positives, loo = _collect_in_catalog(catalog, recordings, full_chroma, gate, window_seconds, mode_label)
        op = _operating_point(positives, loo, ood)
        results[mode_label] = {
            "positives": _evcount_summary(positives),
            "loo_negatives": _evcount_summary(loo),
            "ood_negatives": _evcount_summary(ood),
            "operating_point": op,
        }
        _log(f"  [{mode_label}] chosen={op.get('chosen_point')}")
        if op.get("bootstrap_at_chosen"):
            _log(f"  [{mode_label}] bootstrap={op['bootstrap_at_chosen']}")

    certified = False
    point_pass = False
    for _mode, r in results.items():
        op = r["operating_point"]
        if op.get("passes_point_estimate"):
            point_pass = True
        bs = op.get("bootstrap_at_chosen")
        if op.get("passes_point_estimate") and bs and bs.get("fa_loo") and bs.get("fa_ood"):
            if bs["fa_loo"]["certifiable"] and bs["fa_ood"]["certifiable"] and bs["ta_strict"]["point"] >= _MIN_TA:
                certified = True

    if certified:
        verdict = "PASS_CERTIFIED_FOREIGN_OOD"
        verdict_line = (
            "PASS (certified on foreign-composer OOD): the frozen pitch-only elastic margin gate holds "
            "TA>=0.60 with the out-of-catalog false-accept upper-95%-CI <= 0.05 across >=60 diverse "
            "independent works. The gate rejects diverse foreign repertoire at scale. Caveat: the OOD "
            "false-accept upper-CI is still bounded by the in-catalog LOO axis (16 works) for the "
            "HARDEST same-composer adversary; same-composer rejection at scale remains future work."
        )
    elif point_pass:
        verdict = "PASS_POINT_ONLY"
        verdict_line = (
            "PASS (point only): TA>=0.60 @ FA<=0.05 point estimate, but an upper-95%-CI still exceeds 0.05."
        )
    else:
        verdict = "FAIL"
        verdict_line = (
            "FAIL: the gate does not hold TA>=0.60 at FA<=0.05 against diverse foreign-composer OOD."
        )

    out = {
        "experiment": "stage0e_ood_certify",
        "gate_under_test": "Stage-0c/0d winner: pitch-only chord-Jaccard elastic subseq-DTW, v3 margin (FROZEN)",
        "ood_source": "MAESTRO v3 performances whose canonical_composer is NOT one of the 16 catalog "
        "composers (provably out-of-catalog; zero title/alias matching).",
        "ood_distinct_works": len(ood),
        "ood_by_composer": dict(ood_composers.most_common()),
        "ood_event_cap": _MAX_OOD_EVENTS,
        "catalog_pieces": len(catalog),
        "recordings": len(recordings),
        "min_clusters_to_certify": _MIN_CLUSTERS_TO_CERTIFY,
        "criterion": "TA>=0.60 @ FA<=0.05; certified iff FA upper-95%-CI<=0.05 AND >=60 independent OOD works",
        "results": results,
        "verdict": verdict,
        "verdict_line": verdict_line,
        "caveats": {
            "foreign_is_easier": "Foreign-composer OOD is the 'different repertoire' case -- easier to reject "
            "than the hardest adversary (same composer, different piece). The hard case is covered only by "
            "H1's 16 in-catalog LOO works; certifying same-composer rejection at scale needs a verified "
            "same-composer MAESTRO de-dup (free-text, alias-prone) and is left as future work.",
            "catalog_is_asap": "The 254-piece catalog IS the ASAP repertoire (242/242 works matched), so ASAP "
            "yields zero OOD; only MAESTRO-minus-ASAP provides out-of-catalog canon.",
            "positives_capped_at_16": "TA is still bounded by 16 eval recordings; this experiment hardens the "
            "rejection (FA) axis, not the recall (TA) axis.",
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    _OUTPUT.write_text(json.dumps(out, indent=2))
    _log(f"\nVERDICT: {verdict}")
    _log(verdict_line)
    _log(f"Wrote {_OUTPUT}")


if __name__ == "__main__":
    main()
