"""Comprehensive, production-fidelity piece-ID evaluation across edge-case axes.

The cross-performance harness (pieceid_crossperf_verify) measured the FROZEN
production margin gate on CLEAN ASAP performance MIDIs from the OPENING window.
That is an UPPER BOUND: production input is real audio -> AMT (noisy), often
starting mid-piece, and the user may transpose. This harness quantifies the gap
on every axis the clean-MIDI eval cannot see, against the same 11K catalog and
the same frozen gate (chroma top-K shortlist -> pitch-only chord-Jaccard elastic
subsequence-DTW -> lock iff margin(2nd_cost - best_cost) >= threshold):

  B  PARTIAL / MID-PIECE STARTS   query from random offsets (bar ~20, B-section),
                                  not just the opening. The elastic gate is
                                  subsequence-capable, but the 12-dim bag-of-chroma
                                  SHORTLIST is a whole-piece vector -> a mid-piece
                                  section's pitch distribution may fall off top-K.
  C  TRANSPOSITION                chroma is key-DEPENDENT (pitch%12, no cyclic
                                  normalization) -> a transposed performance should
                                  fail. Quantify the cliff vs +/- semitones.
  D  SAME-COMPOSER CONFUSION      which pieces get confused with which (closed-set
                                  wrong-top1 + LOO false-accepts) -> the worst
                                  confusable clusters.
  E  CONFIDENCE CALIBRATION       the DO exposes confidence = margin. Bin by margin,
                                  measure P(correct) -> is margin a usable
                                  probability? Report reliability + ECE.
  G  NOTES/SECONDS-TO-LOCK        how many notes until a confident, stable lock;
                                  does the locked id flip as more notes arrive?

Axes A (real-audio->AMT) and F (foreign-OOD at 11K) are gated on MAESTRO
rehydration + the AMT service and live in sibling commits.

Design: build the catalog + chroma index + elastic gate ONCE, then run every
axis by TRANSFORMING the query notes (slice / transpose / prefix) and reusing the
exact per-query evaluator eval_query() from the cross-performance harness. The
gate ALGORITHM is never touched -- only the query is transformed and the operating
threshold is a parameter, faithful to the WASM parity contract.

CIs cluster-bootstrap by ASAP work folder (the independent unit), never by the
correlated per-query windows -- the stage0d certification gotcha.

Run (from model/, pointing at the 11K catalog in the issue-49 worktree):
  PYTHONPATH=src uv run python -m score_library.pieceid_comprehensive_eval \
      --scores-dir /path/to/issue-49/model/data/scores \
      --axes b,c,d,e,g --per-work 1 --note-cap 600 --threshold 0.13
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from piece_id_eval.notes import Note
from piece_id_eval.stage0c_elastic_dtwgate import ElasticGate
from score_library.bulk_ingest import TOP_K, _RunningChromaIndex
from score_library.pieceid_crossperf_verify import (
    _ASAP_DIR,
    _SCORES_DIR,
    _load_perf_midi,
    _source,
    build_catalog,
    eval_query,
    label_works,
    load_dedup_manifest,
)

# Production operating point after the 11K retune (session-brain.ts). The frozen
# parity threshold is 0.0935; the live gate was retuned to 0.13 post-dedup.
DEFAULT_THRESHOLD = 0.13

_OUT_DIR = _ASAP_DIR.parent.parent / "evals" / "piece_id"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Parsed-catalog disk cache. Parsing 11K x ~825KB score JSONs (~9GB) is the
# entire fixed cost (~8 min); chroma+gate indexing is ~20s. The cache stores the
# parsed {id: [Note]} dict keyed by a cheap stat-hash of the score files + note
# cap, turning re-parse into a ~10s pickle load -- essential for the hill-climb
# loop. The FULL catalog is cached; dedup excludes are applied after load so one
# cache serves both full and deduped runs.
# ---------------------------------------------------------------------------

def _catalog_key(scores_dir: Path, note_cap: int) -> str:
    skip = {"titles.json", "seed.sql"}
    files = sorted(f for f in scores_dir.glob("*.json") if f.name not in skip)
    h = hashlib.md5()
    h.update(f"{note_cap}:{len(files)}".encode())
    for p in files:
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def load_catalog(scores_dir: Path, note_cap: int, cache_path: str | None,
                 exclude: set[str] | None) -> dict[str, list[Note]]:
    """build_catalog with an optional parsed-catalog pickle cache. Caches the
    FULL catalog; applies `exclude` (dedup drops) to the returned view.

    The pickle is SELF-GENERATED (written then read back by this same harness,
    keyed by a content stat-hash) and local-only -- never loaded from an
    untrusted source -- so unpickling is safe here. It holds only Note
    NamedTuples (plain numbers)."""
    exclude = exclude or set()
    if not cache_path:
        return build_catalog(note_cap, scores_dir, exclude=exclude)
    key = _catalog_key(scores_dir, note_cap)
    cp = Path(cache_path)
    if cp.exists():
        blob = pickle.loads(cp.read_bytes())
        if blob.get("key") == key:
            _log(f"  catalog cache HIT ({cp.name})")
            full = blob["catalog"]
            return {k: v for k, v in full.items() if k not in exclude}
        _log(f"  catalog cache STALE (key mismatch), rebuilding")
    full = build_catalog(note_cap, scores_dir)  # full catalog (no exclude)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_bytes(pickle.dumps({"key": key, "catalog": full},
                                protocol=pickle.HIGHEST_PROTOCOL))
    _log(f"  catalog cached -> {cp}")
    return {k: v for k, v in full.items() if k not in exclude}


# ---------------------------------------------------------------------------
# Query transforms (the entire point: every axis is a transform on the query)
# ---------------------------------------------------------------------------

def transpose_notes(notes: list[Note], semitones: int) -> list[Note]:
    """Shift every note's pitch by `semitones`, dropping notes pushed off the
    88-key piano range (a real transposition would, too)."""
    out: list[Note] = []
    for n in notes:
        p = n.pitch + semitones
        if 21 <= p <= 108:  # A0..C8
            out.append(Note(onset=n.onset, offset=n.offset, pitch=p, velocity=n.velocity))
    return out


def start_at_fraction(notes: list[Note], frac: float, note_cap: int) -> list[Note]:
    """Return up to note_cap notes whose onset is at/after `frac` of the way
    through the performance's onset span -- a mid-piece start.

    frac=0.0 is the opening (the clean-MIDI baseline). frac=0.5 starts halfway.
    """
    if not notes or frac <= 0.0:
        return notes[:note_cap]
    t0 = notes[0].onset
    t1 = notes[-1].onset
    cut = t0 + frac * (t1 - t0)
    tail = [n for n in notes if n.onset >= cut]
    return tail[:note_cap]


# ---------------------------------------------------------------------------
# Re-thresholding helpers -- recompute the lock decision at ANY operating point
# from the raw margins eval_query returns (the gate math is frozen; threshold is
# a parameter). closed_margin / loo_margin are always present for an 11K catalog.
# ---------------------------------------------------------------------------

def recognized_at(rec: dict, thr: float) -> bool:
    return bool(rec.get("closed_correct")) and rec.get("closed_margin", -1.0) >= thr


def locked_at(rec: dict, thr: float) -> bool:
    return rec.get("closed_margin", -1.0) >= thr


def loo_fa_at(rec: dict, thr: float, genuine_only: bool = False) -> bool | None:
    """LOO false-accept at threshold thr. genuine_only excludes locks onto a
    residual catalog duplicate of the true piece (a cleanliness artifact, not a
    genuine different-piece rejection failure)."""
    if "loo_margin" not in rec:
        return None
    fa = rec["loo_margin"] >= thr
    if genuine_only:
        return bool(fa and not rec.get("loo_fa_dup_of_true"))
    return bool(fa)


# ---------------------------------------------------------------------------
# Cluster-bootstrap CI by work (the independent unit; never by query/window)
# ---------------------------------------------------------------------------

def _bootstrap_ci(records: list[dict], stat_fn, n_boot: int = 1000,
                  seed: int = 12345) -> tuple[float, float]:
    """95% CI for stat_fn(list_of_records), resampling whole works with
    replacement. stat_fn returns a float or None (None -> skip that resample)."""
    by_work: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_work[r.get("work", "?")].append(r)
    works = list(by_work.values())
    if len(works) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(works)
    samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        pooled = [r for j in idx for r in works[j]]
        v = stat_fn(pooled)
        if v is not None:
            samples.append(v)
    if not samples:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (round(float(lo), 4), round(float(hi), 4))


def _rate(records: list[dict], pred) -> float | None:
    vals = [pred(r) for r in records]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(bool(v) for v in vals) / len(vals)


# ---------------------------------------------------------------------------
# Per-query evaluation under a transform (reuses the frozen eval_query)
# ---------------------------------------------------------------------------

def _eval_transformed(notes, true_id, chroma, gate, k, *, transform, label,
                      folder, perf, thr) -> dict | None:
    """Apply a transform to the query notes and evaluate with the frozen gate.
    Returns an augmented record carrying the axis label + lock decision at thr."""
    q = transform(notes)
    if len(q) < 2:
        return None
    rec = eval_query(q, true_id, chroma, gate, k)
    if rec is None:
        return None
    rec["work"] = folder
    rec["perf"] = perf
    rec["axis_label"] = label
    rec["recognized"] = recognized_at(rec, thr)
    rec["locked"] = locked_at(rec, thr)
    return rec


# ---------------------------------------------------------------------------
# Axis B: partial / mid-piece starts
# ---------------------------------------------------------------------------

_B_FRACS = [0.0, 0.1, 0.25, 0.5, 0.75]


def axis_partial(queries, chroma, gate, k, note_cap, thr) -> dict:
    out: dict[str, list[dict]] = {f"{f:.2f}": [] for f in _B_FRACS}
    for (folder, perf, notes, true_id) in queries:
        for f in _B_FRACS:
            rec = _eval_transformed(
                notes, true_id, chroma, gate, k,
                transform=lambda ns, f=f: start_at_fraction(ns, f, note_cap),
                label=f"start={f:.2f}", folder=folder, perf=perf, thr=thr)
            if rec is not None:
                out[f"{f:.2f}"].append(rec)
    blocks = {}
    for key, recs in out.items():
        blocks[key] = _block(recs, thr)
    return {"by_start_fraction": blocks,
            "baseline_recognized": blocks.get("0.00", {}).get("recognized"),
            "note": "frac of the performance onset-span skipped before the query window"}


# ---------------------------------------------------------------------------
# Axis C: transposition
# ---------------------------------------------------------------------------

_C_SEMIS = [-6, -5, -3, -2, -1, 0, 1, 2, 3, 5, 6]


def axis_transposition(queries, chroma, gate, k, note_cap, thr) -> dict:
    out: dict[str, list[dict]] = {str(s): [] for s in _C_SEMIS}
    for (folder, perf, notes, true_id) in queries:
        base = notes[:note_cap]
        for s in _C_SEMIS:
            rec = _eval_transformed(
                base, true_id, chroma, gate, k,
                transform=lambda ns, s=s: transpose_notes(ns, s),
                label=f"semitones={s}", folder=folder, perf=perf, thr=thr)
            if rec is not None:
                out[str(s)].append(rec)
    blocks = {s: _block(recs, thr) for s, recs in out.items()}
    return {"by_semitones": blocks,
            "note": "chroma is key-dependent; expect a sharp cliff away from 0"}


# ---------------------------------------------------------------------------
# Axis G: notes/seconds to a confident, stable lock
# ---------------------------------------------------------------------------

_G_PREFIXES = [30, 50, 75, 100, 150, 200, 300, 450, 600]


def axis_latency(queries, chroma, gate, k, thr) -> dict:
    """For each query, evaluate at increasing prefixes and find (a) the first
    prefix that locks, (b) whether that lock is correct, (c) whether the locked
    id FLIPS at any larger prefix (instability)."""
    per_prefix: dict[int, list[dict]] = {p: [] for p in _G_PREFIXES}
    first_lock_events: list[int | None] = []
    first_lock_seconds: list[float | None] = []
    correct_at_first_lock: list[bool] = []
    flipped: list[bool] = []

    for (folder, _perf, notes, true_id) in queries:
        locked_id_by_prefix: list[tuple[int, str]] = []
        first_lock_n: int | None = None
        first_lock_sec: float | None = None
        first_lock_correct = False
        for p in _G_PREFIXES:
            q = notes[:p]
            if len(q) < 2:
                continue
            rec = eval_query(q, true_id, chroma, gate, k)
            if rec is None or "closed_margin" not in rec:
                continue
            rec["work"] = folder
            is_locked = locked_at(rec, thr)
            per_prefix[p].append({**rec, "recognized": recognized_at(rec, thr),
                                  "locked": is_locked})
            if is_locked:
                locked_id_by_prefix.append((p, rec["closed_best"]))
                if first_lock_n is None:
                    first_lock_n = p
                    first_lock_sec = round(q[-1].onset - q[0].onset, 2)
                    first_lock_correct = bool(rec.get("closed_correct"))
        first_lock_events.append(first_lock_n)
        first_lock_seconds.append(first_lock_sec)
        if first_lock_n is not None:
            correct_at_first_lock.append(first_lock_correct)
            ids = [lid for _, lid in locked_id_by_prefix]
            flipped.append(len(set(ids)) > 1)

    locked_ns = [n for n in first_lock_events if n is not None]
    return {
        "by_prefix_notes": {str(p): _block(per_prefix[p], thr) for p in _G_PREFIXES},
        "ever_locked_frac": round(len(locked_ns) / len(first_lock_events), 4) if first_lock_events else None,
        "first_lock_notes_median": int(np.median(locked_ns)) if locked_ns else None,
        "first_lock_notes_p90": int(np.percentile(locked_ns, 90)) if locked_ns else None,
        "first_lock_seconds_median": round(float(np.median([s for s in first_lock_seconds if s is not None])), 2) if any(s is not None for s in first_lock_seconds) else None,
        "correct_at_first_lock": round(sum(correct_at_first_lock) / len(correct_at_first_lock), 4) if correct_at_first_lock else None,
        "lock_flip_rate": round(sum(flipped) / len(flipped), 4) if flipped else None,
        "note": "flip = the locked id changed across prefixes (an unstable, dangerous lock)",
    }


# ---------------------------------------------------------------------------
# Axis E: confidence calibration (margin -> P(correct))
# ---------------------------------------------------------------------------

_E_BINS = [0.0, 0.05, 0.0935, 0.13, 0.20, 0.30, 0.50, 1.0, 10.0]


def axis_calibration(records: list[dict]) -> dict:
    """Reliability of margin as a probability of a CORRECT lock. Uses the closed
    queries (a real target in the catalog). For each margin bin: count, mean
    margin, and P(closed_correct). Also a coarse ECE treating a min-max-scaled
    margin as the predicted probability."""
    rs = [r for r in records if "closed_margin" in r]
    bins: list[dict] = []
    for lo, hi in zip(_E_BINS, _E_BINS[1:]):
        sel = [r for r in rs if lo <= r["closed_margin"] < hi]
        if not sel:
            continue
        acc = sum(1 for r in sel if r.get("closed_correct")) / len(sel)
        mean_margin = float(np.mean([r["closed_margin"] for r in sel]))
        bins.append({"margin_lo": lo, "margin_hi": hi, "n": len(sel),
                     "mean_margin": round(mean_margin, 4),
                     "p_correct": round(acc, 4)})
    # ECE with margin clipped to [0, 0.5] then scaled to [0,1] as a pseudo-prob.
    if rs:
        preds = np.clip([r["closed_margin"] for r in rs], 0.0, 0.5) / 0.5
        correct = np.array([1.0 if r.get("closed_correct") else 0.0 for r in rs])
        order = np.argsort(preds)
        preds, correct = preds[order], correct[order]
        n = len(preds)
        ece = 0.0
        for q in range(10):
            a, b = int(q * n / 10), int((q + 1) * n / 10)
            if b <= a:
                continue
            ece += (b - a) / n * abs(preds[a:b].mean() - correct[a:b].mean())
        ece_val = round(float(ece), 4)
    else:
        ece_val = None
    # Monotonicity: is P(correct) non-decreasing across populated bins?
    pcs = [b["p_correct"] for b in bins]
    monotone = all(x <= y + 1e-9 for x, y in zip(pcs, pcs[1:]))
    return {"reliability_bins": bins, "ece_scaled_margin": ece_val,
            "monotone_increasing": monotone,
            "note": "margin is usable as confidence iff P(correct) rises monotonically with it"}


# ---------------------------------------------------------------------------
# Axis D: same-composer confusion matrix
# ---------------------------------------------------------------------------

def axis_confusion(records: list[dict], thr: float) -> dict:
    """Where does the gate get confused? Two confusion sources:
      closed wrong-top1 : the lowest-cost candidate is NOT the true piece.
      LOO false-accept  : with the true id masked, a different piece locks.
    Aggregate composer(true) -> composer(confused) and the worst id->id pairs."""
    comp_cm: collections.Counter = collections.Counter()
    pair_cm: collections.Counter = collections.Counter()
    kind_cm: collections.Counter = collections.Counter()
    same_composer_confusions = 0
    total_confusions = 0
    for r in records:
        confused_id = None
        kind = None
        # closed-set: top1 wrong AND it would actually LOCK (margin >= thr) -- a
        # dangerous wrong-recognition. A wrong top1 the gate rejects (low margin ->
        # unknown -> Tier-3) is graceful, not a confusion, so require the lock.
        if (r.get("closed_best") and not r.get("closed_correct")
                and r.get("closed_margin", -1.0) >= thr):
            confused_id = r["closed_best"]
            kind = "closed_wrong_lock"
        # LOO open-set false-accept onto a genuinely different piece
        elif loo_fa_at(r, thr, genuine_only=True):
            confused_id = r.get("loo_best")
            kind = "loo_genuine_fa"
        if not confused_id:
            continue
        kind_cm[kind] += 1
        true_comp = _source(r["true_id"])
        conf_comp = _source(confused_id)
        comp_cm[(true_comp, conf_comp)] += 1
        pair_cm[(r["true_id"], confused_id)] += 1
        total_confusions += 1
        if true_comp == conf_comp:
            same_composer_confusions += 1
    top_pairs = [{"true": a, "confused_with": b, "count": c,
                  "same_composer": _source(a) == _source(b)}
                 for (a, b), c in pair_cm.most_common(25)]
    top_comp = [{"true_composer": a, "confused_composer": b, "count": c,
                 "same": a == b}
                for (a, b), c in comp_cm.most_common(20)]
    return {
        "total_confusions": total_confusions,
        "by_kind": dict(kind_cm),
        "same_composer_confusions": same_composer_confusions,
        "same_composer_frac": round(same_composer_confusions / total_confusions, 4) if total_confusions else None,
        "worst_piece_pairs": top_pairs,
        "worst_composer_pairs": top_comp,
        "note": "high same_composer_frac => the gate confuses siblings (the dangerous failure)",
    }


# ---------------------------------------------------------------------------
# Aggregation block (with CIs)
# ---------------------------------------------------------------------------

def _block(records: list[dict], thr: float, with_ci: bool = True) -> dict:
    m = len(records)
    if not m:
        return {"n": 0}
    recall_k = _rate(records, lambda r: r.get("chroma_recall_at_k"))
    recall_1 = _rate(records, lambda r: r.get("chroma_rank") == 0)
    top1 = _rate(records, lambda r: r.get("closed_correct"))
    recog = _rate(records, lambda r: recognized_at(r, thr))
    locked = _rate(records, lambda r: locked_at(r, thr))
    loo_fa = _rate(records, lambda r: loo_fa_at(r, thr))
    loo_fa_gen = _rate(records, lambda r: loo_fa_at(r, thr, genuine_only=True))
    block = {
        "n": m,
        "chroma_recall@k": _r(recall_k),
        "chroma_recall@1": _r(recall_1),
        "top1_correct": _r(top1),
        "recognized": _r(recog),
        "locked_rate": _r(locked),
        "loo_false_accept": _r(loo_fa),
        "loo_fa_genuine": _r(loo_fa_gen),
    }
    if with_ci and m >= 4:
        block["recognized_ci95"] = _bootstrap_ci(
            records, lambda rs: _rate(rs, lambda r: recognized_at(r, thr)))
        block["loo_fa_genuine_ci95"] = _bootstrap_ci(
            records, lambda rs: _rate(rs, lambda r: loo_fa_at(r, thr, genuine_only=True)))
    return block


def _r(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _build_queries(labels, note_cap, per_work, limit_works) -> list[tuple]:
    """Materialize (folder, perf_name, notes, true_id) query tuples from ASAP."""
    folders = sorted(labels)
    if limit_works:
        folders = folders[:limit_works]
    queries = []
    for folder in folders:
        true_id = labels[folder]["true_id"]
        wdir = _ASAP_DIR / folder
        perfs = sorted(p for p in wdir.glob("*.mid") if p.name != "midi_score.mid")
        if per_work:
            perfs = perfs[:per_work]
        for perf in perfs:
            notes = _load_perf_midi(perf)[:note_cap]
            if len(notes) >= 2:
                queries.append((folder, perf.name, notes, true_id))
    return queries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", type=str, default="b,c,d,e,g",
                    help="comma list from b,c,d,e,g (a,f are gated siblings)")
    ap.add_argument("--per-work", type=int, default=1,
                    help="performances per work (1 = one per work, fast; 0 = all)")
    ap.add_argument("--note-cap", type=int, default=600)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="margin operating point (production retuned to 0.13)")
    ap.add_argument("--limit-works", type=int, default=0)
    ap.add_argument("--scores-dir", type=str, default=str(_SCORES_DIR))
    ap.add_argument("--exclude-manifest", type=str, default="",
                    help="dedup_scan manifest: drop high-conf twins + remap labels")
    ap.add_argument("--catalog-cache", type=str, default="",
                    help="path to a parsed-catalog pickle cache (built on first run, "
                         "loaded fast thereafter); avoids re-parsing ~9GB of JSON")
    ap.add_argument("--out", type=str, default=str(_OUT_DIR / "comprehensive_eval.json"))
    args = ap.parse_args()

    axes = {a.strip().lower() for a in args.axes.split(",") if a.strip()}
    t0 = time.time()

    drop: set[str] = set()
    remap: dict[str, str] = {}
    if args.exclude_manifest:
        drop, remap = load_dedup_manifest(Path(args.exclude_manifest))
        _log(f"dedup manifest: dropping {len(drop)} high-conf twins")

    _log(f"building catalog (note_cap={args.note_cap}) from {args.scores_dir} ...")
    catalog = load_catalog(Path(args.scores_dir), args.note_cap,
                           args.catalog_cache or None, exclude=drop)
    _log(f"  catalog: {len(catalog)} pieces  [{time.time()-t0:.1f}s]")

    t1 = time.time()
    chroma = _RunningChromaIndex(catalog)
    gate = ElasticGate(catalog)
    _log(f"  indexed chroma + gate  [{time.time()-t1:.1f}s]")

    t2 = time.time()
    labels, lstats = label_works(catalog, chroma, gate, args.note_cap, remap=remap)
    _log(f"  labeled {lstats['labeled']}/{lstats['n_works']} works "
         f"(oracle agree {lstats['both_agree']}/{lstats['both_fired']})  [{time.time()-t2:.1f}s]")

    queries = _build_queries(labels, args.note_cap, args.per_work, args.limit_works)
    _log(f"  {len(queries)} query performances")

    results: dict = {
        "config": {"axes": sorted(axes), "per_work": args.per_work,
                   "note_cap": args.note_cap, "top_k": args.top_k,
                   "threshold": args.threshold, "catalog_size": len(catalog),
                   "frozen_parity_threshold": 0.0935},
        "labeling": lstats,
    }

    # Baseline closed/LOO at the opening window (frac 0) -- the reference number.
    t3 = time.time()
    baseline = [
        _eval_transformed(notes, true_id, chroma, gate, args.top_k,
                          transform=lambda ns: ns[:args.note_cap],
                          label="baseline", folder=folder, perf=perf,
                          thr=args.threshold)
        for (folder, perf, notes, true_id) in queries
    ]
    baseline = [r for r in baseline if r is not None]
    results["baseline"] = _block(baseline, args.threshold)
    _log(f"  baseline computed (n={len(baseline)})  [{time.time()-t3:.1f}s]")

    if "b" in axes:
        ts = time.time(); _log("axis B: partial / mid-piece starts ...")
        results["axis_B_partial_start"] = axis_partial(queries, chroma, gate, args.top_k, args.note_cap, args.threshold)
        _log(f"  axis B done  [{time.time()-ts:.1f}s]")
    if "c" in axes:
        ts = time.time(); _log("axis C: transposition ...")
        results["axis_C_transposition"] = axis_transposition(queries, chroma, gate, args.top_k, args.note_cap, args.threshold)
        _log(f"  axis C done  [{time.time()-ts:.1f}s]")
    if "d" in axes:
        ts = time.time(); _log("axis D: same-composer confusion matrix ...")
        results["axis_D_confusion"] = axis_confusion(baseline, args.threshold)
        _log(f"  axis D done  [{time.time()-ts:.1f}s]")
    if "e" in axes:
        ts = time.time(); _log("axis E: confidence calibration ...")
        results["axis_E_calibration"] = axis_calibration(baseline)
        _log(f"  axis E done  [{time.time()-ts:.1f}s]")
    if "g" in axes:
        ts = time.time(); _log("axis G: notes-to-lock latency ...")
        results["axis_G_latency"] = axis_latency(queries, chroma, gate, args.top_k, args.threshold)
        _log(f"  axis G done  [{time.time()-ts:.1f}s]")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))

    _print_report(results, args)
    _log(f"\nwrote {outp}   [total {time.time()-t0:.1f}s]")


def _print_report(results: dict, args) -> None:
    thr = args.threshold
    print("\n" + "=" * 76)
    print(f"COMPREHENSIVE PIECE-ID EVAL  (catalog={results['config']['catalog_size']}, "
          f"threshold={thr}, axes={results['config']['axes']})")
    print("=" * 76)
    b = results["baseline"]
    print(f"\nBASELINE (opening window, n={b['n']}):  recognized={_pct(b['recognized'])}  "
          f"recall@k={_pct(b['chroma_recall@k'])}  loo_fa_genuine={_pct(b['loo_fa_genuine'])}")
    if "recognized_ci95" in b:
        print(f"  recognized CI95={b['recognized_ci95']}  loo_fa_genuine CI95={b['loo_fa_genuine_ci95']}")

    if "axis_B_partial_start" in results:
        print("\nAXIS B -- MID-PIECE STARTS (recognized by start fraction):")
        for frac, blk in results["axis_B_partial_start"]["by_start_fraction"].items():
            if blk.get("n"):
                print(f"  start={frac}  n={blk['n']:4d}  recognized={_pct(blk['recognized'])}  "
                      f"recall@k={_pct(blk['chroma_recall@k'])}")
    if "axis_C_transposition" in results:
        print("\nAXIS C -- TRANSPOSITION (recognized by semitone shift):")
        for s, blk in results["axis_C_transposition"]["by_semitones"].items():
            if blk.get("n"):
                print(f"  semis={s:>3}  recognized={_pct(blk['recognized'])}  "
                      f"recall@k={_pct(blk['chroma_recall@k'])}")
    if "axis_D_confusion" in results:
        d = results["axis_D_confusion"]
        print(f"\nAXIS D -- CONFUSION: total={d['total_confusions']}  "
              f"same-composer={d['same_composer_confusions']} ({_pct(d['same_composer_frac'])})")
        for p in d["worst_piece_pairs"][:8]:
            tag = "SAME-COMPOSER" if p["same_composer"] else ""
            print(f"  {p['count']}x  {p['true']}  ->  {p['confused_with']}  {tag}")
    if "axis_E_calibration" in results:
        e = results["axis_E_calibration"]
        print(f"\nAXIS E -- CALIBRATION: ECE={e['ece_scaled_margin']}  monotone={e['monotone_increasing']}")
        for bn in e["reliability_bins"]:
            print(f"  margin[{bn['margin_lo']:.3f},{bn['margin_hi']:.3f})  n={bn['n']:4d}  "
                  f"P(correct)={_pct(bn['p_correct'])}")
    if "axis_G_latency" in results:
        g = results["axis_G_latency"]
        print(f"\nAXIS G -- LATENCY: ever_locked={_pct(g['ever_locked_frac'])}  "
              f"first_lock_notes_median={g['first_lock_notes_median']}  "
              f"(~{g['first_lock_seconds_median']}s)  "
              f"correct_at_first_lock={_pct(g['correct_at_first_lock'])}  "
              f"flip_rate={_pct(g['lock_flip_rate'])}")


def _pct(x) -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "n/a"


if __name__ == "__main__":
    main()
