"""Catalog-wide AMT-aware duplicate detection (non-destructive cleanliness manifest).

The cross-performance verification (pieceid_crossperf_verify) showed that the
apparent open-set "failures" at 11K are almost entirely the catalog holding the
SAME work more than once: 139 exact duplicates plus ~36 AMT near-copies that the
ingest-time dedup (a single 0.2885 gate, applied incrementally and only vs the
frozen catalog at the time) missed. Those twins split the match and inflate the
raw leave-one-out false-accept rate, but they are a cleanliness artifact, not a
recognition defect.

This scan finds the residual duplicate CLUSTERS catalog-wide and writes a manifest
the deferred legality/cleanliness pass can action. It does NOT delete anything --
the volume campaign's rule is "keep pieces, dedup/legality later".

Mechanism: for each piece, chroma top-K shortlist -> pitch-only elastic-DTW cost
(same gate the catalog was deduped with), edge if cost <= an AMT-AWARE threshold:
  * engraved <-> engraved : 0.2885 (the certified dedup gap; tight).
  * any side is giantmidi/pdmx : 0.40 (looser, to catch same-work copies whose
    cost is inflated by AMT transcription noise -- justified only because AMT
    noise, not genuine difference, is the plausible cause).
Edges -> union-find clusters. Each cluster picks a KEEP (source priority
engraved > pdmx > giantmidi, then most notes) and marks the rest DROP candidates.

Run:  cd model && uv run python -m score_library.dedup_scan --note-cap 600 \
          --scores-dir <catalog> --out data/evals/piece_id/dedup_manifest.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

from piece_id_eval.note_chroma import chroma_vector
from piece_id_eval.notes import Note, load_score_notes
from piece_id_eval.stage0c_elastic_dtwgate import ElasticGate, _notes_to_events
from score_library.bulk_ingest import DUP_THRESHOLD, W_PITCH, W_TIME, _RunningChromaIndex

_MODULE_DIR = Path(__file__).resolve().parent
_MODEL_ROOT = _MODULE_DIR.parents[1]
_SCORES_DIR = _MODEL_ROOT / "data" / "scores"
_OUT = _MODEL_ROOT / "data" / "evals" / "piece_id" / "dedup_manifest.json"

AMT_THRESHOLD = 0.40  # looser gate when AMT noise is the plausible cost inflator
_AMT_SOURCES = {"giantmidi", "pdmx"}
# KEEP priority: prefer an engraved (cleaner) source over AMT transcriptions.
_SOURCE_RANK = {"giantmidi": 0, "pdmx": 1}  # default (engraved) = 2 (highest)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _source(pid: str) -> str:
    return pid.split(".")[0]


def _is_amt(pid: str) -> bool:
    return _source(pid) in _AMT_SOURCES


def _pair_threshold(a: str, b: str) -> float:
    return AMT_THRESHOLD if (_is_amt(a) or _is_amt(b)) else DUP_THRESHOLD


def _keep_rank(pid: str, n_notes: int) -> tuple[int, int]:
    """Higher is better: engraved beats AMT; among equals, more notes wins."""
    return (_SOURCE_RANK.get(_source(pid), 2), n_notes)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note-cap", type=int, default=600)
    ap.add_argument("--top-k", type=int, default=8, help="chroma shortlist per piece")
    ap.add_argument("--scores-dir", type=str, default=str(_SCORES_DIR))
    ap.add_argument("--out", type=str, default=str(_OUT))
    args = ap.parse_args()

    t0 = time.time()
    skip = {"titles.json", "seed.sql"}
    paths = sorted(f for f in Path(args.scores_dir).glob("*.json") if f.name not in skip)
    catalog: dict[str, list[Note]] = {}
    for p in paths:
        notes = load_score_notes(p)[: args.note_cap]
        if notes:
            catalog[p.stem] = notes
    _log(f"catalog: {len(catalog)} pieces [{time.time()-t0:.1f}s]")

    chroma = _RunningChromaIndex(catalog)
    gate = ElasticGate(catalog)
    qvecs = {pid: chroma_vector(notes) for pid, notes in catalog.items()}
    events = {pid: _notes_to_events(notes) for pid, notes in catalog.items()}
    n_notes = {pid: len(notes) for pid, notes in catalog.items()}
    _log(f"indexed [{time.time()-t0:.1f}s]")

    uf = _UnionFind()
    edges: list[tuple[str, str, float]] = []
    seen_pairs: set[tuple[str, str]] = set()
    ids = list(catalog)
    t1 = time.time()
    for i, pid in enumerate(ids):
        q_pc, q_li = events[pid]
        if q_pc.shape[0] < 2:
            continue
        for cid in chroma.top_k(qvecs[pid], args.top_k + 1):
            if cid == pid:
                continue
            key = (pid, cid) if pid < cid else (cid, pid)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            c = gate.cost(q_pc, q_li, cid, W_PITCH, W_TIME)
            if c is None or not np.isfinite(c):
                continue
            if c <= _pair_threshold(pid, cid):
                edges.append((key[0], key[1], round(float(c), 4)))
                uf.union(pid, cid)
        if (i + 1) % 1000 == 0:
            _log(f"  scanned {i+1}/{len(ids)}  edges={len(edges)} [{time.time()-t1:.1f}s]")

    # Build clusters from union-find
    members: dict[str, list[str]] = collections.defaultdict(list)
    for pid in {p for e in edges for p in e[:2]}:
        members[uf.find(pid)].append(pid)

    clusters = []
    n_drop = 0
    dropped_by_source: collections.Counter[str] = collections.Counter()
    edge_by_root: dict[str, list] = collections.defaultdict(list)
    for a, b, c in edges:
        edge_by_root[uf.find(a)].append([a, b, c])
    for root, mem in members.items():
        if len(mem) < 2:
            continue
        keep = max(mem, key=lambda p: _keep_rank(p, n_notes[p]))
        drop = sorted(p for p in mem if p != keep)
        n_drop += len(drop)
        for d in drop:
            dropped_by_source[_source(d)] += 1
        amt_involved = any(_is_amt(p) for p in mem)
        clusters.append({
            "keep": keep,
            "drop": drop,
            "size": len(mem),
            "sources": dict(collections.Counter(_source(p) for p in mem)),
            "amt_involved": amt_involved,
            "pairs": sorted(edge_by_root[root], key=lambda e: e[2]),
        })
    clusters.sort(key=lambda cl: (-cl["size"], cl["keep"]))

    summary = {
        "catalog_size": len(catalog),
        "n_clusters": len(clusters),
        "n_drop_candidates": n_drop,
        "post_dedup_size": len(catalog) - n_drop,
        "dropped_by_source": dict(dropped_by_source),
        "clusters_with_amt": sum(1 for c in clusters if c["amt_involved"]),
        "thresholds": {"engraved_engraved": DUP_THRESHOLD, "amt_involved": AMT_THRESHOLD},
        "note_cap": args.note_cap, "top_k": args.top_k,
    }
    out = {"summary": summary, "clusters": clusters}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 64)
    print(f"DEDUP SCAN  catalog={len(catalog)}  [{time.time()-t0:.1f}s]")
    print("=" * 64)
    print(f"  clusters: {len(clusters)}  drop-candidates: {n_drop}  "
          f"post-dedup catalog: {summary['post_dedup_size']}")
    print(f"  dropped by source: {dict(dropped_by_source)}")
    print(f"  clusters involving AMT: {summary['clusters_with_amt']}/{len(clusters)}")
    print("  largest clusters:")
    for cl in clusters[:10]:
        print(f"    size={cl['size']} keep={cl['keep'][:48]} sources={cl['sources']}")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
