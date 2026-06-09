# model/src/piece_id_eval/stage0f_hard_ood_certify.py
"""Stage-0f (H4): certify the gate on the HARDEST adversary -- SAME-COMPOSER out-of-catalog (#26).

Stage-0e certified rejection of FOREIGN-composer repertoire (the easy case). The
dangerous failure -- a wrong lock that poisons the whole session -- is a
HARMONICALLY-SIMILAR SAME-COMPOSER piece we do NOT have. This experiment supplies
that negative set at scale: MAESTRO performances by the 16 catalog composers whose
work is NOT in the catalog.

The de-dup is the entire risk (it leaked an eval piece in an earlier attempt when
signatures were built from ASAP instead of the real catalog). Defenses here:
  1. Exclusion signatures built from the REAL 254 catalog slugs.
  2. exact ASAP->MAESTRO title join (catches title-identical performances).
  3. (composer, opus) and bidirectional-substring (composer, family-word) wholesale
     exclusion -- aggressively OVER-exclude (shrinks OOD; safe for FA integrity).
  4. LEAK DETECTOR: every surviving candidate's chroma top-1 catalog match + elastic
     cost is reported; near-perfect alignments are flagged for manual title inspection
     (a missed duplicate -> exclude; a genuine near-identical different piece -> keep).

Two run modes:
  --analyze (default): de-dup + leak diagnostics ONLY. Prints suspicious candidates
            for manual inspection. Writes NO certification (so an unverified de-dup
            can't masquerade as a result).
  --certify <excluded.json>: after manual inspection, runs the frozen gate on the
            verified OOD set (minus any titles listed in excluded.json) and certifies.

Gate is FROZEN to the Stage-0c/0d/0e winner (pitch-only chord-Jaccard elastic margin).

Run analyze:
  cd model && PYTHONUNBUFFERED=1 caffeinate -i uv run python -m piece_id_eval.stage0f_hard_ood_certify
Run certify (after inspection):
  ... stage0f_hard_ood_certify --certify data/evals/piece_id/stage0f_manual_exclusions.json
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import partitura as pa

from piece_id_eval.matchers.note_chroma_matcher import NoteChromaMatcher
from piece_id_eval.notes import Note
from piece_id_eval.stage0c_elastic_dtwgate import (
    _TOP_K,
    _W_PITCH,
    ElasticGate,
    _notes_to_events,
    load_data,
)
from piece_id_eval.stage0d_gate_hardening import (
    _MIN_TA,
    _WINDOW_SECONDS,
    _collect_in_catalog,
    _evcount_summary,
    _operating_point,
)
from piece_id_eval.stage0e_ood_certify import _CAT_COMPOSERS

_MODULE_DIR = Path(__file__).resolve().parent
_MODEL_ROOT = _MODULE_DIR.parents[1]
_MAESTRO_DIR = _MODEL_ROOT / "data/raw/maestro-v3.0.0"
_MAESTRO_CSV = _MAESTRO_DIR / "maestro-v3.0.0.csv"
_ASAP_CSV = _MODEL_ROOT / "data/raw/asap-dataset/metadata.csv"
_SCORES = _MODEL_ROOT / "data/scores"
_ANALYZE_OUT = _MODEL_ROOT / "data/evals/piece_id/stage0f_hard_ood_analysis.json"
_CERTIFY_OUT = _MODEL_ROOT / "data/evals/piece_id/stage0f_hard_ood_certify_results.json"

_MAX_OOD_EVENTS = 1500
_W_TIME = 0.0
_LEAK_COST_FLAG = 0.20  # candidates with best elastic cost below this are flagged for manual review
# NOTE: matching is composer-SCOPED, so genre words like "sonata" are safe to match on
# (they exclude only the SAME composer's sonatas). "sonata"/"sonatas" are deliberately
# NOT stopped: catalog sonatas are indexed (e.g. beethoven 21-1) while MAESTRO labels them
# by opus/Hob/Köchel (Op.53 / Hob.XVI:49 / K.332), sharing no number token -- so the genre
# word is the ONLY bridge. Stopping it leaked 13 in-catalog sonatas (Stage-0f analyze v1).
_STOP = frozenset({
    "piano", "major", "minor", "sharp", "flat", "first", "second", "third", "complete",
    "movement", "theme", "book", "volume", "from", "with", "and", "the", "for", "posth",
})


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _composer_of(s: str) -> str | None:
    s = s.lower()
    for c in _CAT_COMPOSERS:
        if c in s:
            return c
    return None


def _family_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{5,}", text.lower()) if w not in _STOP and w not in _CAT_COMPOSERS}


def _opus(text: str) -> set[tuple[str, int]]:
    t = text.lower()
    s: set[tuple[str, int]] = set()
    for m in re.finditer(r"op\.?\s*_?\s*(\d+)", t):
        s.add(("op", int(m.group(1))))
    for m in re.finditer(r"bwv\s*_?\s*(\d+)", t):
        s.add(("bwv", int(m.group(1))))
    for m in re.finditer(r"\bk\.?\s*(\d{3})\b", t):
        s.add(("k", int(m.group(1))))
    for m in re.finditer(r"\bs\.?\s*(\d{3})\b", t):
        s.add(("s", int(m.group(1))))
    for m in re.finditer(r"\bd\.?\s*(\d{3})\b", t):
        s.add(("d", int(m.group(1))))
    return s


def _build_catalog_dedup() -> tuple[dict[str, set[str]], dict[str, set], set[str], dict[str, str]]:
    """Return (fam_by_comp, opus_by_comp, in_exact_titles, slug_text_by_comp)."""
    fam_by_comp: dict[str, set[str]] = defaultdict(set)
    opus_by_comp: dict[str, set] = defaultdict(set)
    slug_text_by_comp: dict[str, str] = defaultdict(str)
    for p in _SCORES.glob("*.json"):
        slug = p.stem
        comp = _composer_of(slug)
        if not comp:
            continue
        body = slug.split(".", 1)[1] if "." in slug else slug
        body = body.replace(".", " ").replace("_", " ")
        fam_by_comp[comp] |= _family_tokens(body)
        opus_by_comp[comp] |= _opus(body)
        slug_text_by_comp[comp] += " " + body.lower()

    rows = list(csv.DictReader(_MAESTRO_CSV.open()))
    fn2title = {r["midi_filename"]: r["canonical_title"] for r in rows}
    asap = list(csv.reader(_ASAP_CSV.open()))[1:]
    in_exact = {fn2title[r[8].replace("{maestro}/", "")] for r in asap if r[8].replace("{maestro}/", "") in fn2title}
    return fam_by_comp, opus_by_comp, in_exact, slug_text_by_comp


def _is_in_catalog_samecomposer(
    title: str, comp: str, fam_by_comp, opus_by_comp, in_exact, slug_text_by_comp
) -> bool:
    if title in in_exact:
        return True
    tl = title.lower()
    if _opus(title) & opus_by_comp.get(comp, set()):
        return True
    # bidirectional substring on family tokens (catches stem variants)
    cat_fams = fam_by_comp.get(comp, set())
    title_toks = _family_tokens(title)
    if title_toks & cat_fams:
        return True
    slug_text = slug_text_by_comp.get(comp, "")
    if any(tok in slug_text for tok in title_toks) or any(f in tl for f in cat_fams):
        return True
    return False


def _candidates() -> list[dict]:
    """MAESTRO same-composer works NOT excluded by the catalog de-dup. One per title."""
    fam, opus, in_exact, slug_text = _build_catalog_dedup()
    rows = list(csv.DictReader(_MAESTRO_CSV.open()))
    by_title: dict[str, dict] = {}
    for r in rows:
        comp = _composer_of(r["canonical_composer"])
        if comp is None:
            continue  # foreign-composer -> handled by Stage-0e, not here
        if r["canonical_title"] in by_title:
            continue
        if _is_in_catalog_samecomposer(r["canonical_title"], comp, fam, opus, in_exact, slug_text):
            continue
        by_title[r["canonical_title"]] = {"row": r, "composer": comp}
    return list(by_title.values())


def _load_perf_midi(path: Path, cap: int) -> list[Note]:
    na = pa.load_performance_midi(str(path)).note_array()
    notes = [Note(onset=float(r["onset_sec"]), offset=float(r["onset_sec"]) + max(float(r["duration_sec"]), 1e-3),
                  pitch=int(r["pitch"]), velocity=int(r["velocity"])) for r in na]
    notes.sort(key=lambda n: n.onset)
    return notes[:cap] if cap and len(notes) > cap else notes


def _score_candidate(notes: list[Note], full_chroma: NoteChromaMatcher, gate: ElasticGate) -> dict | None:
    q_pc, q_li = _notes_to_events(notes)
    if q_pc.shape[0] < 2:
        return None
    topk = [r.piece_id for r in full_chroma.rank(notes)[:_TOP_K]]
    costs = []
    for cid in topk:
        c = gate.cost(q_pc, q_li, cid, _W_PITCH, _W_TIME)
        if c is not None and c == c:  # not NaN
            costs.append((cid, c))
    if len(costs) < 2:
        return None
    costs.sort(key=lambda x: x[1])
    margin = costs[1][1] - costs[0][1]
    return {"margin": margin, "best_cat_id": costs[0][0], "best_cost": costs[0][1], "n_ev": q_pc.shape[0]}


def analyze() -> None:
    t0 = time.time()
    catalog, _ = load_data()
    full_chroma = NoteChromaMatcher(catalog)
    gate = ElasticGate(catalog)
    cands = _candidates()
    _log(f"[dedup] {len(cands)} same-composer OOD candidate titles after exclusion")
    by_comp = Counter(c["composer"] for c in cands)
    _log(f"[dedup] by composer: {dict(by_comp.most_common())}")

    scored = []
    for i, c in enumerate(cands):
        midi = _MAESTRO_DIR / c["row"]["midi_filename"]
        if not midi.exists():
            continue
        s = _score_candidate(_load_perf_midi(midi, _MAX_OOD_EVENTS), full_chroma, gate)
        if s is None:
            continue
        s.update({"title": c["row"]["canonical_title"], "composer": c["row"]["canonical_composer"]})
        scored.append(s)
        if (i + 1) % 25 == 0:
            _log(f"[score] {i+1}/{len(cands)} {time.time()-t0:.0f}s")

    flagged = sorted([s for s in scored if s["best_cost"] < _LEAK_COST_FLAG], key=lambda s: s["best_cost"])
    _log(f"\n[LEAK DETECTOR] {len(flagged)}/{len(scored)} candidates align near-perfectly (best_cost<{_LEAK_COST_FLAG}).")
    _log("Inspect each: is the MAESTRO title actually the matched catalog piece (LEAK->exclude) "
         "or a genuinely different same-composer piece (KEEP)?")
    for s in flagged:
        _log(f"  cost={s['best_cost']:.3f} margin={s['margin']:.3f}  '{s['title']}' ({s['composer']})  ~= catalog[{s['best_cat_id']}]")

    out = {
        "experiment": "stage0f_hard_ood_analysis",
        "candidate_titles": len(cands),
        "scored": len(scored),
        "by_composer": dict(by_comp.most_common()),
        "leak_flag_threshold": _LEAK_COST_FLAG,
        "flagged_for_manual_review": flagged,
        "all_scored": scored,
        "runtime_seconds": round(time.time() - t0, 1),
        "next": "Inspect flagged_for_manual_review; write confirmed-leak titles to "
        "stage0f_manual_exclusions.json as {\"excluded_titles\": [...]}, then run --certify.",
    }
    _ANALYZE_OUT.write_text(json.dumps(out, indent=2))
    _log(f"\nWrote {_ANALYZE_OUT}  ({len(scored)} candidates, {len(flagged)} flagged)")


def certify(exclusions_path: Path) -> None:
    excluded = set(json.loads(exclusions_path.read_text()).get("excluded_titles", [])) if exclusions_path.exists() else set()
    _log(f"[certify] manual exclusions: {len(excluded)}")
    catalog, recordings = load_data()
    full_chroma = NoteChromaMatcher(catalog)
    gate = ElasticGate(catalog)

    cands = [c for c in _candidates() if c["row"]["canonical_title"] not in excluded]
    ood = []
    for c in cands:
        midi = _MAESTRO_DIR / c["row"]["midi_filename"]
        if not midi.exists():
            continue
        s = _score_candidate(_load_perf_midi(midi, _MAX_OOD_EVENTS), full_chroma, gate)
        if s:
            ood.append({"margin": s["margin"], "work": c["row"]["canonical_title"], "n_ev": s["n_ev"]})
    _log(f"[certify] {len(ood)} verified same-composer OOD works")

    results: dict[str, dict] = {}
    for window_seconds, mode_label in [(None, "full"), (_WINDOW_SECONDS, "90s")]:
        positives, loo = _collect_in_catalog(catalog, recordings, full_chroma, gate, window_seconds, mode_label)
        op = _operating_point(positives, loo, ood)
        results[mode_label] = {
            "positives": _evcount_summary(positives), "loo_negatives": _evcount_summary(loo),
            "ood_negatives": _evcount_summary(ood), "operating_point": op,
        }
        _log(f"  [{mode_label}] bootstrap={op.get('bootstrap_at_chosen')}")

    certified = False
    for _m, r in results.items():
        bs = r["operating_point"].get("bootstrap_at_chosen")
        if r["operating_point"].get("passes_point_estimate") and bs and bs.get("fa_ood"):
            if bs["fa_ood"]["certifiable"] and bs["ta_strict"]["point"] >= _MIN_TA:
                certified = True
    verdict = "PASS_HARD_OOD_CERTIFIED" if certified else "HARD_OOD_NOT_CERTIFIED"
    out = {
        "experiment": "stage0f_hard_ood_certify",
        "ood_source": "MAESTRO same-composer (16 catalog composers) works NOT in catalog, de-dup verified + manual exclusions",
        "manual_exclusions": sorted(excluded),
        "ood_distinct_works": len(ood),
        "results": results,
        "verdict": verdict,
        "runtime_seconds": 0,
    }
    _CERTIFY_OUT.write_text(json.dumps(out, indent=2))
    _log(f"\nVERDICT: {verdict}\nWrote {_CERTIFY_OUT}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--certify":
        certify(Path(sys.argv[2]))
    else:
        analyze()
