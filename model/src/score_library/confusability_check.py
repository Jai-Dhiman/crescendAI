"""Intra-catalog confusability proxy for the piece-ID gate re-certification.

The #26 open-set gate cert (stage0e) certifies FOREIGN-composer rejection and
needs offloaded MAESTRO query audio, so it can't be run here. But the risk the
volume campaign actually introduces is INTRA-catalog confusability: at ~10k
pieces -- thousands of same-composer works plus AMT-noisy GiantMIDI, ingested
through a windowed dedup whose 0.2885 gap nearly closed -- can the gate still
tell catalog pieces apart?

This proxy needs no external audio: for a stratified sample of catalog pieces it
takes each piece's own opening window, queries the chroma+DTW gate, and reports

  * self-recall: does a piece's own window retrieve ITSELF as chroma top-1?
  * confusable rate: how many pieces have a DIFFERENT catalog piece within the
    0.2885 dup threshold (residual duplicate / true collision risk)?
  * the nearest-OTHER DTW cost distribution, broken down by source prefix.

A high self-recall + a low confusable rate means the catalog stayed
distinguishable at scale. A spike in either (especially within giantmidi/pdmx)
quantifies what the deferred cleanliness/legality pass must address.

Run:  cd model && uv run python -m score_library.confusability_check --sample 800
"""
from __future__ import annotations

import argparse
import collections
import random
from pathlib import Path

import numpy as np

from piece_id_eval.note_chroma import chroma_vector
from piece_id_eval.notes import load_score_notes
from piece_id_eval.stage0c_elastic_dtwgate import ElasticGate, _notes_to_events
from score_library.bulk_ingest import DUP_THRESHOLD, TOP_K, W_PITCH, W_TIME, _RunningChromaIndex

_SCORES_DIR = Path(__file__).resolve().parents[2] / "data" / "scores"


def _source(pid: str) -> str:
    return pid.split(".")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=800, help="pieces to probe (0 = all)")
    ap.add_argument("--note-cap", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    skip = {"titles.json", "seed.sql"}
    paths = sorted(f for f in _SCORES_DIR.glob("*.json") if f.name not in skip)
    catalog = {p.stem: load_score_notes(p)[: args.note_cap] for p in paths}
    catalog = {k: v for k, v in catalog.items() if v}
    print(f"catalog: {len(catalog)} pieces (window={args.note_cap})", flush=True)

    chroma = _RunningChromaIndex(catalog)
    gate = ElasticGate(catalog)
    qvecs = {pid: chroma_vector(notes) for pid, notes in catalog.items()}
    events = {pid: _notes_to_events(notes) for pid, notes in catalog.items()}

    ids = list(catalog)
    rng = random.Random(args.seed)
    probe = ids if args.sample <= 0 else rng.sample(ids, min(args.sample, len(ids)))

    self_hit = 0
    confusable = 0
    nearest_costs: list[float] = []
    per_source_conf: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])  # [confusable, total]
    examples: list[tuple[str, str, float]] = []

    for pid in probe:
        q_pc, q_li = events[pid]
        if q_pc.shape[0] < 2:
            continue
        top = chroma.top_k(qvecs[pid], TOP_K + 1)  # +1 because self is in the index
        self_hit += int(top and top[0] == pid)
        best_other, best_cost = "", np.inf
        for cid in top:
            if cid == pid:
                continue
            c = gate.cost(q_pc, q_li, cid, W_PITCH, W_TIME)
            if c is not None and np.isfinite(c) and c < best_cost:
                best_cost, best_other = float(c), cid
        if not best_other:
            continue
        nearest_costs.append(best_cost)
        src = _source(pid)
        per_source_conf[src][1] += 1
        if best_cost <= DUP_THRESHOLD:
            confusable += 1
            per_source_conf[src][0] += 1
            if len(examples) < 12:
                examples.append((pid, best_other, best_cost))

    n = len(probe)
    arr = np.array(nearest_costs) if nearest_costs else np.array([0.0])
    print(f"\n=== confusability proxy (n={n}) ===")
    print(f"  self-recall (own window -> itself top-1): {self_hit}/{n} = {100*self_hit/max(n,1):.1f}%")
    print(f"  confusable (a DIFFERENT piece within {DUP_THRESHOLD}): {confusable}/{n} = {100*confusable/max(n,1):.1f}%")
    print(f"  nearest-other DTW cost: min={arr.min():.4f} p5={np.percentile(arr,5):.4f} "
          f"p25={np.percentile(arr,25):.4f} median={np.percentile(arr,50):.4f}")
    print("  confusable rate by source:")
    for src, (c, t) in sorted(per_source_conf.items(), key=lambda kv: -kv[1][1]):
        if t:
            print(f"    {src:14s} {c:4d}/{t:4d} = {100*c/t:5.1f}%")
    print("  example near-collisions (pid -> nearest-other @ cost):")
    for pid, other, cost in examples:
        print(f"    {cost:.4f}  {pid[:46]}  ~  {other[:46]}")


if __name__ == "__main__":
    main()
