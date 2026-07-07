"""Cross-performance recall + open-set (LOO) verification of the piece-ID margin
gate at 11K-catalog scale, using held-out ASAP performance MIDIs.

The confusability proxy (confusability_check.py) proved the catalog is
SELF-distinguishable: each piece's own opening window retrieves itself. It did
NOT prove the gate recognizes a DIFFERENT performance of a catalog piece, nor
that it REJECTS a piece absent from the catalog. This harness closes both gaps
against the FROZEN production gate (piece_id_eval.export_parity_fixtures, the
WASM parity contract):

    chroma top-K shortlist  ->  pitch-only elastic-DTW cost per candidate
    ->  recognized id = lowest-cost candidate
    ->  accept ("locked") iff margin (2nd_best_cost - best_cost) >= 0.0935

Queries: ASAP performance MIDIs (1066 across 242 works; multiple performers per
work = natural correlation clusters), genuinely held-out renditions. Many ASAP
works are now sourced in the 11K catalog from a DIFFERENT engraving
(KernScores/Mutopia) -> the match is cross-source as well as cross-performance.

Ground truth is established GATE-INDEPENDENTLY first:
  L1 deterministic : derive_piece_id(ASAP folder) if that id is in the catalog
                     (gold; no gate involved).
  L2 score oracle  : fingerprint the work's own midi_score.mid against the
                     catalog; accept the best id only on a TIGHT self-match
                     (cost <= SELF_MATCH_MAX and margin >= SELF_MATCH_MARGIN),
                     which recovers works re-sourced under a different stem.
  Agreement on the L1 INTERSECT L2 overlap validates the oracle is not circular.

Metrics (per performance query, against the FULL catalog):
  chroma_recall@K : true id present in the chroma top-K shortlist (the ceiling
                    above which the elastic gate can never recover).
  top1_correct    : lowest-cost candidate == true id.
  locked          : margin >= 0.0935 (the production accept decision).
  recognized      : locked AND top1_correct (the user-visible success).
  LOO false-accept: mask the true id (simulate "piece not in catalog"); a lock
                    is then a false accept -- the open-set rejection failure.

Splits reported: by matched-catalog source (engraved canon vs giantmidi AMT vs
pdmx/mutopia) and by composer-neighborhood density (common = many same-composer
catalog neighbors vs uncommon = few).

Run:  cd model && uv run python -m score_library.pieceid_crossperf_verify \
          --per-work 0 --note-cap 600
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import partitura as pa

from piece_id_eval.note_chroma import chroma_vector
from piece_id_eval.notes import Note, load_score_notes
from piece_id_eval.stage0c_elastic_dtwgate import ElasticGate, _notes_to_events
from score_library.bulk_ingest import DUP_THRESHOLD, TOP_K, W_PITCH, W_TIME, _RunningChromaIndex
from score_library.discover import derive_piece_id

_MODULE_DIR = Path(__file__).resolve().parent
_MODEL_ROOT = _MODULE_DIR.parents[1]
_SCORES_DIR = _MODEL_ROOT / "data" / "scores"
_ASAP_DIR = _MODEL_ROOT / "data" / "raw" / "asap"
_OUT = _MODEL_ROOT / "data" / "evals" / "piece_id" / "crossperf_verify.json"

# Production accept threshold, frozen in export_parity_fixtures.py.
MARGIN_THRESHOLD = 0.0935
# Score-oracle self-match guards (a score matching its OWN catalog entry is near 0).
SELF_MATCH_MAX = 0.12
SELF_MATCH_MARGIN = 0.05
# Engraved-source composer prefixes (vs giantmidi/pdmx recognize-only AMT sources).
_ENGRAVED_PREFIXES = {
    "bach", "beethoven", "chopin", "mozart", "scarlatti", "haydn", "joplin",
    "scriabin", "hummel", "liszt", "schubert", "schumann", "rachmaninoff",
    "ravel", "debussy", "prokofiev", "glinka", "brahms", "balakirev", "mutopia",
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _source(pid: str) -> str:
    return pid.split(".")[0]


def _source_class(pid: str) -> str:
    s = _source(pid)
    if s == "giantmidi":
        return "giantmidi"
    if s == "pdmx":
        return "pdmx"
    if s in _ENGRAVED_PREFIXES:
        return "engraved"
    return "other"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def build_catalog(note_cap: int, scores_dir: Path = _SCORES_DIR,
                  exclude: set[str] | None = None) -> dict[str, list[Note]]:
    skip = {"titles.json", "seed.sql"}
    exclude = exclude or set()
    paths = sorted(f for f in scores_dir.glob("*.json") if f.name not in skip)
    catalog: dict[str, list[Note]] = {}
    for p in paths:
        if p.stem in exclude:
            continue
        notes = load_score_notes(p)[:note_cap]
        if notes:
            catalog[p.stem] = notes
    if not catalog:
        raise RuntimeError(f"empty catalog from {scores_dir}")
    return catalog


def load_dedup_manifest(path: Path, high_confidence_only: bool = True) -> tuple[set[str], dict[str, str]]:
    """Return (drop_ids, remap drop_id -> cluster keep) from a dedup_scan manifest.

    By default masks only the HIGH-confidence drops (tight cost <=0.2885), the set
    safe to auto-apply; the medium AMT-near-dup tier (0.2885-0.40) is a review pool,
    not masked, to avoid corrupting the re-measure by dropping genuinely distinct pieces.
    """
    body = json.loads(path.read_text())
    remap: dict[str, str] = {}
    for cl in body.get("clusters", []):
        for d in cl.get("drop", []):
            remap[d] = cl["keep"]
    if high_confidence_only and "high_confidence_drops" in body:
        drop = set(body["high_confidence_drops"])
        remap = {d: k for d, k in remap.items() if d in drop}
    else:
        drop = set(remap)
    return drop, remap


# ---------------------------------------------------------------------------
# Gate primitives -- faithful to export_parity_fixtures (pitch-only margin gate)
# ---------------------------------------------------------------------------

def _events(notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
    return _notes_to_events(notes)


def _ranked_costs(
    q_pc: np.ndarray,
    q_li: np.ndarray,
    cand_ids: list[str],
    gate: ElasticGate,
) -> list[tuple[str, float]]:
    """Pitch-only elastic cost for each candidate, ascending by cost (best first)."""
    out: list[tuple[str, float]] = []
    for cid in cand_ids:
        c = gate.cost(q_pc, q_li, cid, W_PITCH, W_TIME)
        if c is not None and np.isfinite(c):
            out.append((cid, float(c)))
    out.sort(key=lambda x: x[1])
    return out


def _decision(ranked: list[tuple[str, float]]) -> tuple[str, float] | None:
    """(best_id, margin) from cost-ascending candidates, or None if < 2 usable."""
    if len(ranked) < 2:
        return None
    return ranked[0][0], ranked[1][1] - ranked[0][1]


def _catalog_pair_cost(gate: ElasticGate, id_a: str, id_b: str) -> float | None:
    """Pitch-only elastic cost between two CATALOG entries (for dup detection).

    Used to classify a LOO false-accept: if the falsely-locked neighbor is within
    the certified dedup distance (DUP_THRESHOLD) of the true piece, it is a residual
    catalog DUPLICATE of the same work -- a cleanliness artifact, NOT a genuine
    open-set rejection failure between two different pieces.
    """
    ev_a = gate._events.get(id_a)
    if ev_a is None:
        return None
    return gate.cost(ev_a[0], ev_a[1], id_b, W_PITCH, W_TIME)


# ---------------------------------------------------------------------------
# Ground truth: L1 deterministic + L2 score oracle
# ---------------------------------------------------------------------------

def _oracle_label(
    score_notes: list[Note],
    chroma: _RunningChromaIndex,
    gate: ElasticGate,
) -> str | None:
    """Best catalog id for a work's own score MIDI, accepted only on a tight self-match."""
    q_pc, q_li = _events(score_notes)
    if q_pc.shape[0] < 2:
        return None
    top = chroma.top_k(chroma_vector(score_notes), TOP_K)
    ranked = _ranked_costs(q_pc, q_li, top, gate)
    if len(ranked) < 2:
        # single usable candidate that matches near-perfectly still counts
        if ranked and ranked[0][1] <= SELF_MATCH_MAX:
            return ranked[0][0]
        return None
    best_id, best_cost = ranked[0]
    margin = ranked[1][1] - best_cost
    if best_cost <= SELF_MATCH_MAX and margin >= SELF_MATCH_MARGIN:
        return best_id
    return None


def label_works(
    catalog: dict[str, list[Note]],
    chroma: _RunningChromaIndex,
    gate: ElasticGate,
    note_cap: int,
    remap: dict[str, str] | None = None,
) -> tuple[dict[str, dict], dict]:
    """Map each ASAP work folder -> {true_id, method, det_id, oracle_id}.

    Reads metadata.csv for the authoritative folder list. Returns (labels, stats).
    `remap` (drop_id -> cluster keep) redirects labels onto the surviving entry
    when running against a simulated deduped catalog.
    """
    remap = remap or {}
    meta = _ASAP_DIR / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"ASAP metadata.csv missing: {meta}")
    folders: dict[str, dict] = {}  # folder -> row info
    with meta.open() as fh:
        for row in csv.DictReader(fh):
            folders.setdefault(row["folder"], row)

    labels: dict[str, dict] = {}
    n_det = n_oracle = n_both = n_agree = n_none = 0
    for folder in sorted(folders):
        det_raw = derive_piece_id(_ASAP_DIR / folder, _ASAP_DIR)
        det_id = remap.get(det_raw, det_raw)
        det_id = det_id if det_id in catalog else None
        score_path = _ASAP_DIR / folder / "midi_score.mid"
        oracle_id = None
        if score_path.exists():
            sn = _load_perf_midi(score_path)[:note_cap]
            if sn:
                oracle_id = _oracle_label(sn, chroma, gate)
        if det_id and oracle_id:
            n_both += 1
            n_agree += int(det_id == oracle_id)
        true_id = det_id or oracle_id
        method = "det" if det_id else ("oracle" if oracle_id else "none")
        if det_id:
            n_det += 1
        elif oracle_id:
            n_oracle += 1
        else:
            n_none += 1
            continue
        labels[folder] = {"true_id": true_id, "method": method,
                          "det_id": det_id, "oracle_id": oracle_id}
    stats = {"n_works": len(folders), "labeled": len(labels), "by_det": n_det,
             "by_oracle_only": n_oracle, "unlabeled": n_none,
             "both_fired": n_both, "both_agree": n_agree}
    return labels, stats


def _load_perf_midi(path: Path) -> list[Note]:
    ppart = pa.load_performance_midi(str(path))
    na = ppart.note_array()
    notes = [
        Note(onset=float(r["onset_sec"]),
             offset=float(r["onset_sec"]) + max(float(r["duration_sec"]), 1e-3),
             pitch=int(r["pitch"]), velocity=int(r["velocity"]))
        for r in na
    ]
    notes.sort(key=lambda n: n.onset)
    return notes


# ---------------------------------------------------------------------------
# Per-query evaluation (closed-set + LOO open-set in one pass)
# ---------------------------------------------------------------------------

def eval_query(
    notes: list[Note],
    true_id: str,
    chroma: _RunningChromaIndex,
    gate: ElasticGate,
    k: int,
) -> dict | None:
    q_pc, q_li = _events(notes)
    n_ev = int(q_pc.shape[0])
    if n_ev < 2:
        return None
    # chroma top-(k+1): enough to reconstruct the LOO top-k after masking true id.
    shortlist = chroma.top_k(chroma_vector(notes), k + 1)
    chroma_rank = shortlist.index(true_id) if true_id in shortlist else -1

    # Closed set: production uses chroma top-k.
    closed_cands = shortlist[:k]
    closed_ranked = _ranked_costs(q_pc, q_li, closed_cands, gate)
    closed = _decision(closed_ranked)

    # LOO: remove true id from the chroma ranking, take the new top-k.
    loo_cands = [c for c in shortlist if c != true_id][:k]
    loo_ranked = _ranked_costs(q_pc, q_li, loo_cands, gate)
    loo = _decision(loo_ranked)

    rec = {
        "true_id": true_id,
        "true_source": _source_class(true_id),
        "n_ev": n_ev,
        "chroma_rank": chroma_rank,         # -1 if true id absent from top-(k+1)
        "chroma_recall_at_k": 0 <= chroma_rank < k,
    }
    if closed is not None:
        best, margin = closed
        rec.update({"closed_best": best, "closed_margin": margin,
                    "closed_correct": best == true_id,
                    "closed_locked": margin >= MARGIN_THRESHOLD})
    if loo is not None:
        lbest, lmargin = loo
        locked = lmargin >= MARGIN_THRESHOLD
        rec.update({"loo_margin": lmargin, "loo_best": lbest,
                    "loo_best_source": _source_class(lbest),
                    "loo_locked": locked})
        # Decompose a false-accept: is the falsely-locked neighbor a residual
        # duplicate of the true piece (cleanliness) or a genuinely different
        # piece (a real open-set rejection failure)?  Computed UNCONDITIONALLY
        # (not only when locked at 0.0935) so the dup/genuine split is correct at
        # ANY downstream operating threshold -- a sub-0.0935 sweep would otherwise
        # find the flag absent and miscount every dup FA as genuine.
        pair = _catalog_pair_cost(gate, true_id, lbest)
        is_dup = pair is not None and np.isfinite(pair) and pair <= DUP_THRESHOLD
        rec["loo_fa_dup_of_true"] = bool(is_dup)
        rec["loo_fa_pair_cost"] = round(float(pair), 4) if pair is not None and np.isfinite(pair) else None
    return rec


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _rate(items: list[dict], pred) -> float:
    vals = [pred(it) for it in items if pred(it) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _neighbor_density(catalog: dict[str, list[Note]]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter(_source(pid) for pid in catalog)
    return counts


def summarize(records: list[dict], catalog: dict[str, list[Note]]) -> dict:
    n = len(records)
    dens = _neighbor_density(catalog)

    def block(items: list[dict]) -> dict:
        m = len(items)
        if not m:
            return {"n": 0}
        recall_k = sum(1 for it in items if it.get("chroma_recall_at_k")) / m
        recall_1 = sum(1 for it in items if it.get("chroma_rank") == 0) / m
        top1 = _rate(items, lambda it: it.get("closed_correct"))
        locked = _rate(items, lambda it: it.get("closed_locked"))
        recognized = sum(1 for it in items
                         if it.get("closed_locked") and it.get("closed_correct")) / m
        loo_items = [it for it in items if "loo_locked" in it]
        loo_fa = sum(1 for it in loo_items if it.get("loo_locked")) / len(loo_items) if loo_items else 0.0
        # Decompose false accepts: genuine (different piece) vs duplicate-of-true.
        fa_locked = [it for it in loo_items if it.get("loo_locked")]
        fa_genuine = sum(1 for it in fa_locked if not it.get("loo_fa_dup_of_true"))
        fa_dup = sum(1 for it in fa_locked if it.get("loo_fa_dup_of_true"))
        loo_fa_genuine = fa_genuine / len(loo_items) if loo_items else 0.0
        return {"n": m,
                "chroma_recall@k": round(recall_k, 4),
                "chroma_recall@1": round(recall_1, 4),
                "top1_correct": round(top1, 4),
                "locked_rate": round(locked, 4),
                "recognized": round(recognized, 4),
                "loo_false_accept": round(loo_fa, 4),
                "loo_fa_genuine": round(loo_fa_genuine, 4),
                "loo_fa_dup_count": fa_dup,
                "loo_fa_genuine_count": fa_genuine}

    by_source: dict[str, dict] = {}
    for cls in ("engraved", "giantmidi", "pdmx", "other"):
        by_source[cls] = block([r for r in records if r.get("true_source") == cls])

    # common vs uncommon by same-composer catalog neighborhood density
    common = [r for r in records if dens.get(_source(r["true_id"]), 0) >= 100]
    uncommon = [r for r in records if dens.get(_source(r["true_id"]), 0) < 100]

    # What do open-set false-accepts lock ONTO? (do giantmidi AMT distractors steal matches?)
    fa_locked = [r for r in records if r.get("loo_locked")]
    fa_src = collections.Counter(r.get("loo_best_source") for r in fa_locked)
    fa_genuine_src = collections.Counter(
        r.get("loo_best_source") for r in fa_locked if not r.get("loo_fa_dup_of_true"))

    return {
        "n_queries": n,
        "overall": block(records),
        "by_source": by_source,
        "by_density": {"common(>=100 neighbors)": block(common),
                       "uncommon(<100 neighbors)": block(uncommon)},
        "fa_lock_target_source": dict(fa_src),
        "fa_genuine_lock_target_source": dict(fa_genuine_src),
    }


def sweep_thresholds(records: list[dict]) -> list[dict]:
    """TA (recognized = locked & correct) vs FA (LOO locked) across margin thresholds."""
    pos = [r for r in records if "closed_margin" in r]
    loo = [r for r in records if "loo_margin" in r]
    out = []
    for t in [round(x, 4) for x in np.linspace(0.0, 0.30, 31)]:
        ta = sum(1 for r in pos if r["closed_margin"] >= t and r.get("closed_correct")) / len(pos) if pos else 0.0
        ta_lock = sum(1 for r in pos if r["closed_margin"] >= t) / len(pos) if pos else 0.0
        fa = sum(1 for r in loo if r["loo_margin"] >= t) / len(loo) if loo else 0.0
        # genuine FA only: exclude false-accepts onto a residual duplicate of the true piece
        fa_gen = sum(1 for r in loo if r["loo_margin"] >= t and not r.get("loo_fa_dup_of_true")) / len(loo) if loo else 0.0
        out.append({"threshold": t, "ta_recognized": round(ta, 4),
                    "ta_locked": round(ta_lock, 4), "fa_loo": round(fa, 4),
                    "fa_loo_genuine": round(fa_gen, 4)})
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-work", type=int, default=0,
                    help="performances per work to test (0 = all)")
    ap.add_argument("--note-cap", type=int, default=600)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--limit-works", type=int, default=0, help="debug: cap works")
    ap.add_argument("--scores-dir", type=str, default=str(_SCORES_DIR),
                    help="catalog score JSONs (default: module-relative data/scores)")
    ap.add_argument("--exclude-manifest", type=str, default="",
                    help="dedup_scan manifest: drop its twins + remap labels (simulated dedup)")
    ap.add_argument("--out", type=str, default=str(_OUT))
    args = ap.parse_args()

    t0 = time.time()
    drop: set[str] = set()
    remap: dict[str, str] = {}
    if args.exclude_manifest:
        drop, remap = load_dedup_manifest(Path(args.exclude_manifest))
        _log(f"dedup manifest: dropping {len(drop)} twins (simulated deduped catalog)")
    _log(f"building catalog (note_cap={args.note_cap}) from {args.scores_dir} ...")
    catalog = build_catalog(args.note_cap, Path(args.scores_dir), exclude=drop)
    _log(f"  catalog: {len(catalog)} pieces  [{time.time()-t0:.1f}s]")

    t1 = time.time()
    chroma = _RunningChromaIndex(catalog)
    gate = ElasticGate(catalog)
    _log(f"  indexed chroma + gate events  [{time.time()-t1:.1f}s]")

    t2 = time.time()
    labels, lstats = label_works(catalog, chroma, gate, args.note_cap, remap=remap)
    _log(f"  labeled {lstats['labeled']}/{lstats['n_works']} works "
         f"(det={lstats['by_det']} oracle-only={lstats['by_oracle_only']} "
         f"none={lstats['unlabeled']}); oracle agreement on overlap "
         f"{lstats['both_agree']}/{lstats['both_fired']}  [{time.time()-t2:.1f}s]")

    folders = sorted(labels)
    if args.limit_works:
        folders = folders[: args.limit_works]

    records: list[dict] = []
    t3 = time.time()
    for i, folder in enumerate(folders):
        true_id = labels[folder]["true_id"]
        wdir = _ASAP_DIR / folder
        perfs = sorted(p for p in wdir.glob("*.mid") if p.name != "midi_score.mid")
        if args.per_work:
            perfs = perfs[: args.per_work]
        for perf in perfs:
            notes = _load_perf_midi(perf)[: args.note_cap]
            if not notes:
                continue
            rec = eval_query(notes, true_id, chroma, gate, args.top_k)
            if rec is None:
                continue
            rec["work"] = folder
            rec["perf"] = perf.name
            rec["label_method"] = labels[folder]["method"]
            records.append(rec)
        if (i + 1) % 25 == 0:
            _log(f"  [{i+1}/{len(folders)}] works, {len(records)} queries "
                 f"[{time.time()-t3:.1f}s]")

    summary = summarize(records, catalog)
    sweep = sweep_thresholds(records)
    out = {
        "config": {"note_cap": args.note_cap, "top_k": args.top_k,
                   "per_work": args.per_work, "margin_threshold": MARGIN_THRESHOLD,
                   "catalog_size": len(catalog)},
        "labeling": lstats,
        "summary": summary,
        "sweep": sweep,
        "records": records,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    # Console report
    print("\n" + "=" * 72)
    print(f"CROSS-PERFORMANCE PIECE-ID VERIFICATION  (catalog={len(catalog)}, "
          f"queries={len(records)})")
    print("=" * 72)
    print(f"labeling: {lstats['labeled']}/{lstats['n_works']} works | "
          f"det={lstats['by_det']} oracle-only={lstats['by_oracle_only']} "
          f"none={lstats['unlabeled']} | oracle agree "
          f"{lstats['both_agree']}/{lstats['both_fired']}")
    ov = summary["overall"]
    print(f"\nOVERALL (n={ov['n']}):")
    print(f"  chroma recall@{args.top_k}: {ov['chroma_recall@k']*100:.1f}%   "
          f"recall@1: {ov['chroma_recall@1']*100:.1f}%")
    print(f"  top-1 correct (cost argmin): {ov['top1_correct']*100:.1f}%")
    print(f"  locked@{MARGIN_THRESHOLD} (any): {ov['locked_rate']*100:.1f}%   "
          f"RECOGNIZED (locked & correct): {ov['recognized']*100:.1f}%")
    print(f"  LOO false-accept@{MARGIN_THRESHOLD}: {ov['loo_false_accept']*100:.1f}% "
          f"(GENUINE different-piece: {ov['loo_fa_genuine']*100:.1f}%, "
          f"dup-of-true: {ov['loo_fa_dup_count']}, genuine: {ov['loo_fa_genuine_count']})")
    print(f"  FA locks onto sources: {summary['fa_lock_target_source']}")
    print(f"  GENUINE FA locks onto: {summary['fa_genuine_lock_target_source']}")
    print("\nBY SOURCE:")
    for cls, b in summary["by_source"].items():
        if b.get("n"):
            print(f"  {cls:10s} n={b['n']:4d} recall@k={b['chroma_recall@k']*100:5.1f}% "
                  f"top1={b['top1_correct']*100:5.1f}% recog={b['recognized']*100:5.1f}% "
                  f"loo_fa={b['loo_false_accept']*100:5.1f}%")
    print("\nBY DENSITY:")
    for cls, b in summary["by_density"].items():
        if b.get("n"):
            print(f"  {cls:26s} n={b['n']:4d} recall@k={b['chroma_recall@k']*100:5.1f}% "
                  f"recog={b['recognized']*100:5.1f}% loo_fa={b['loo_false_accept']*100:5.1f}%")
    print("\nTHRESHOLD SWEEP (recognized TA vs LOO FA, total + genuine):")
    for s in sweep:
        if s["threshold"] in (0.0, 0.05, 0.09, 0.1, 0.15, 0.2, 0.25):
            print(f"  t={s['threshold']:.4f}  ta_recog={s['ta_recognized']*100:5.1f}%  "
                  f"ta_locked={s['ta_locked']*100:5.1f}%  fa_loo={s['fa_loo']*100:5.1f}%  "
                  f"fa_genuine={s['fa_loo_genuine']*100:5.1f}%")
    print(f"\nwrote {outp}   [total {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
