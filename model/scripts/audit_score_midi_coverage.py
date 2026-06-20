# /// script
# requires-python = ">=3.10"
# dependencies = ["partitura>=1.4.0"]
# ///
"""Score-MIDI coverage audit for Aria score-conditioning (issue #73).

Aria score-conditioning needs a score MIDI per training piece so the encoder can
compute `delta = z_perf - z_score`. This audits which T1/T2/T5 training pieces have
a resolvable score MIDI and emits a piece -> score-MIDI map (the #73 deliverable),
failing loudly if any score-conditioning-eligible piece is uncovered.

Coverage model (see issue #73 for the full reasoning):

  T1 PercePiano -- 4 underlying works (Beethoven WoO80 32 Variations; Schubert
      D935 No.3; Schubert D960 mvt2; mvt3). PercePiano ships its OWN per-segment
      score MIDI as `*_Score*.mid` / `*_Score2*.mid` files, so T1 is covered
      natively with no external sourcing.

  T5 YouTube Skill -- 16 named pieces, each with clean per-piece identity. Every
      one maps to an existing score-library catalog piece_id (data/scores/*.json),
      which is itself the proof of a backing score MIDI (244 ASAP-derived + 11
      manual via manifests/manual_scores.lock.json). Covered.

  T2 competition -- EXCLUDED from score-conditioning. Competition segments carry no
      per-segment work identity: the `piece` metadata field is a round/recital
      title ("second round (18th Chopin Competition)", "preliminary recital -
      Hough, Couperin, Mozart, Chopin"), and a 30s slice of a multi-work recital
      cannot be attributed to one score without per-segment piece-ID (#26). T2
      contributes ORDINAL ranking signal, which needs no score MIDI. Documented
      exclusion, not a gap.

The score MIDI itself lives in ASAP (`data/raw/asap`, offloaded -- rehydrate via
`git clone`) or in the manual lock (Mutopia URL / committed manual_midis/); catalog
membership is what establishes coverage. Run with partitura available (NOT music21).

Usage:
    uv run model/scripts/audit_score_midi_coverage.py \
        --data-root /abs/path/to/model/data \
        --out       /abs/path/to/model/data/manifests/score_midi_coverage.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

# T5 named piece -> score-library catalog piece_id. Hand-curated (these are
# well-known works); the audit verifies each id actually exists in the catalog.
T5_TO_CATALOG: dict[str, str] = {
    "bach_invention_1": "bach.inventions.1",
    "bach_prelude_c_wtc1": "bach.prelude.bwv_846",
    "chopin_ballade_1": "chopin.ballades.1",
    "chopin_etude_op10no4": "chopin.etudes_op_10.4",
    "chopin_waltz_csm": "chopin.waltzes.64-2",
    "clair_de_lune": "debussy.suite_bergamasque.3_clair_de_lune",
    "debussy_arabesque_1": "debussy.deux_arabesques.1",
    "fantaisie_impromptu": "chopin.fantaisie_impromptu",
    "fur_elise": "beethoven.fur_elise",
    "liszt_liebestraum_3": "liszt.liebestraume.3",
    "moonlight_sonata_mvt1": "beethoven.piano_sonatas.14-1",
    "mozart_k545_mvt1": "mozart.piano_sonatas.16-1",
    "nocturne_op9no2": "chopin.nocturnes.9-2",
    "pathetique_mvt2": "beethoven.piano_sonatas.8-2",
    "rachmaninoff_prelude_csm": "rachmaninoff.preludes_op_3.2",
    "schumann_traumerei": "schumann.kinderszenen.7",
}

# T1 PercePiano underlying works -> prefix of their performance/score segment keys.
T1_WORKS: dict[str, str] = {
    "beethoven_woo80_32_variations": "Beethoven_WoO80_",
    "schubert_d935_no3": "Schubert_D935_no.3_",
    "schubert_d960_mvt2": "Schubert_D960_mv2_",
    "schubert_d960_mvt3": "Schubert_D960_mv3_",
}

# Non-eval entries in the skill_eval dir to ignore when listing T5 pieces.
_T5_NON_PIECE = {"ensemble_4fold", "review.html"}


def load_catalog_ids(data_root: Path) -> set[str]:
    ids = {Path(p).stem for p in glob.glob(str(data_root / "scores" / "*.json"))}
    if not ids:
        raise FileNotFoundError(
            f"No score catalog under {data_root/'scores'} (regenerate via parse-manual, "
            "or point --data-root at the primary checkout)."
        )
    return ids


def audit_t1(data_root: Path) -> list[dict]:
    midi_dir = data_root / "midi" / "percepiano"
    rows = []
    for work, prefix in T1_WORKS.items():
        score_midis = sorted(
            p.name for p in midi_dir.glob(f"{prefix}*Score*.mid")
        )
        status = "covered" if score_midis else "MISSING"
        rows.append(
            {
                "tier": "T1",
                "training_piece": work,
                "catalog_piece_id": None,
                "score_midi_source": "percepiano_native_score_midi",
                "score_midi_locator": f"data/midi/percepiano/{prefix}*Score*.mid",
                "n_score_segments": len(score_midis),
                "status": status,
            }
        )
    return rows


def audit_t5(catalog_ids: set[str], data_root: Path, manual_lock: dict) -> list[dict]:
    skill_dir = data_root / "evals" / "skill_eval"
    present = {
        p.name
        for p in skill_dir.iterdir()
        if p.is_dir() and p.name not in _T5_NON_PIECE
    } if skill_dir.exists() else set(T5_TO_CATALOG)
    # Audit the full intended 16 regardless of which manifests are staged locally.
    rows = []
    for t5_piece in sorted(T5_TO_CATALOG):
        cid = T5_TO_CATALOG[t5_piece]
        in_catalog = cid in catalog_ids
        if cid in manual_lock:
            source = f"manual:{manual_lock[cid].get('resolved_url', '?')}"
        elif in_catalog:
            source = "asap"  # everything not in the manual lock is ASAP-derived
        else:
            source = None
        rows.append(
            {
                "tier": "T5",
                "training_piece": t5_piece,
                "catalog_piece_id": cid,
                "score_midi_source": source,
                "score_midi_locator": f"data/scores/{cid}.json (-> {source})",
                "manifest_staged_locally": t5_piece in present,
                "status": "covered" if in_catalog else "MISSING",
            }
        )
    return rows


def audit_t2(data_root: Path) -> dict:
    labels = collections.Counter()
    n_segments = 0
    for meta in [
        data_root / "manifests" / "competition" / "metadata.jsonl",
        data_root / "manifests" / "competition" / "cliburn_2022" / "metadata.jsonl",
    ]:
        if not meta.exists():
            continue
        for line in meta.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            labels[d.get("piece", "?")] += 1
            n_segments += 1
    # A label that names a concrete single work would let us attribute a score;
    # round/recital titles cannot. Detect the round/recital pattern to quantify.
    round_like = re.compile(r"round|recital|competition", re.I)
    attributable = {k: v for k, v in labels.items() if not round_like.search(k)}
    return {
        "tier": "T2",
        "status": "excluded",
        "reason": (
            "Competition segments carry no per-segment work identity (the `piece` "
            "field is a round/recital/competition title, not a single work), so "
            "score-conditioning is infeasible without per-segment piece-ID (#26). "
            "T2 supplies ordinal ranking signal, which needs no score MIDI."
        ),
        "n_segments": n_segments,
        "distinct_piece_labels": len(labels),
        "distinct_labels_naming_a_single_work": len(attributable),
        "follow_up": "If T2 score-conditioning becomes desired, run per-segment "
        "piece-ID (#26) then map identified works to catalog piece_ids.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="model/data root (point at the PRIMARY checkout when run from a worktree).",
    )
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[1] / "data" / "manifests" / "score_midi_coverage.json"
        ),
    )
    ap.add_argument(
        "--validate-midi",
        action="store_true",
        help="partitura-load one present score MIDI per T1 work as a parse check.",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    catalog_ids = load_catalog_ids(data_root)
    manual_lock_path = data_root / "manifests" / "manual_scores.lock.json"
    manual_lock = json.loads(manual_lock_path.read_text()) if manual_lock_path.exists() else {}

    t1 = audit_t1(data_root)
    t5 = audit_t5(catalog_ids, data_root, manual_lock)
    t2 = audit_t2(data_root)

    if args.validate_midi:
        import partitura

        for row in t1:
            hits = sorted((data_root / "midi" / "percepiano").glob(
                row["score_midi_locator"].split("/")[-1]
            ))
            if hits:
                partitura.load_score_midi(str(hits[0]))  # raises if unparseable
                row["partitura_parse_ok"] = True

    pieces = t1 + t5
    gaps = [r for r in pieces if r["status"] != "covered"]
    coverage = {
        "summary": {
            "t1_works": len(t1),
            "t5_pieces": len(t5),
            "score_conditioning_eligible": len(pieces),
            "covered": len(pieces) - len(gaps),
            "gaps": len(gaps),
            "t2": "excluded (ordinal-only; no per-segment work identity)",
        },
        "pieces": pieces,
        "t2_exclusion": t2,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coverage, indent=2) + "\n")

    print(json.dumps(coverage["summary"], indent=2))
    print(f"\nWrote coverage map -> {out}")
    if gaps:
        for g in gaps:
            print(f"  GAP: {g['tier']} {g['training_piece']} -> {g.get('catalog_piece_id')}")
        raise SystemExit(f"{len(gaps)} score-conditioning-eligible piece(s) uncovered.")
    print("All score-conditioning-eligible pieces (T1+T5) covered.")


if __name__ == "__main__":
    main()
