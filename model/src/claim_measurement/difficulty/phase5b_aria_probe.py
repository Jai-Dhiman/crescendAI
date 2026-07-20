"""Phase 5b -- the Aria symbolic-embedding probe: the RIGHT-modality sequel to the MuQ null.

MuQ (audio) failed to rank difficulty (Phase 4: ties note_count). But difficulty is a SYMBOLIC
property, and Aria is a symbolic foundation model (TransformerEMB, 512-dim global embedding,
pretrained on ~820K MIDIs). This asks the MuQ question on the right modality: does a frozen Aria
embedding rank difficulty better than our 37 hand-crafted features (0.764)?

Same protocol as the MuQ probe: frozen embedding -> RidgeCV probe (the SAME head for Aria and for
the 37-feature symbolic baseline, so the comparison isolates the representation), composer-disjoint
grouped 5-fold repeated over seeds, scipy tau-c (variant='c'). Deployment story if it wins: Aria on
aria-amt-transcribed MIDI (both audio-derived), no worse than the symbolic path's AMT gap.

Stages (resumable):
  embed:  uv-less, run under the model .venv (needs aria + torch):
          .venv/bin/python phase5b_aria_probe.py --stage embed [--limit 600]
  eval:   .venv/bin/python phase5b_aria_probe.py --stage eval [--repeats 5]

Run from model/ (model .venv: aria + torch + sklearn).
"""
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase3c_explore import candidate_features  # noqa: E402
from phase3e_amt_gap import _seg_id  # noqa: E402
from psyllabus import load_records, notes_from_midi_bytes  # noqa: E402

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
PS_DIR = PRIMARY / "model/data/raw/psyllabus"
LABELS = PS_DIR / "new_clean_data.json"
MID_ZIP = PS_DIR / "mid.zip"
WORK = PRIMARY / "model/data/results/aria_probe"
EMB_DIR = WORK / "emb"          # per-piece {seg_id}.npz: aria(512), sym(37), grade, composer
MANIFEST = WORK / "manifest.json"
OUT = PRIMARY / "model/data/results/mirex_phase5b_aria_probe.json"

SYMBOLIC_FULL_TASK = 0.764
N_FOLDS = 5


def tau_c(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)


def _select(limit: int):
    """Deterministic stride sample across the grade-grouped label file (a prefix is single-grade)."""
    records, report = load_records(LABELS, MID_ZIP)
    print(f"loaded {report}", flush=True)
    if limit and limit < len(records):
        records = records[:: max(1, len(records) // limit)][:limit]
    return records


def stage_embed(limit: int):
    sys.path.insert(0, str(PRIMARY / "model/src"))
    from model_improvement.aria_embeddings import extract_embedding

    records = _select(limit)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    manifest = [{"seg_id": _seg_id(r.key), "key": r.key, "grade": r.grade,
                 "composer": r.composer, "midi_name": r.midi_name} for r in records]
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    todo = [m for m in manifest if not (EMB_DIR / f"{m['seg_id']}.npz").exists()]
    print(f"embed: {len(todo)}/{len(manifest)} to extract (Aria 512-d, CPU)", flush=True)

    ok, fail, t0 = 0, 0, time.time()
    with zipfile.ZipFile(MID_ZIP) as zf:
        for i, m in enumerate(todo, 1):
            try:
                raw = zf.read(m["midi_name"])
                notes = notes_from_midi_bytes(raw)
                if len(notes) < 5:
                    raise ValueError("too few notes")
                sym = candidate_features(notes)
                with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tf:
                    tf.write(raw)
                    tf.flush()
                    aria = extract_embedding(tf.name, variant="embedding").numpy()
                # store ONLY numeric arrays -> npz needs no pickle on load; grade/composer
                # live in the manifest and are joined by seg_id at eval time.
                np.savez(EMB_DIR / f"{m['seg_id']}.npz", aria=aria.astype(np.float32),
                         sym=np.array([sym[k] for k in sorted(sym)], np.float32))
                ok += 1
                if i % 25 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] ok={ok} fail={fail} "
                          f"({(time.time()-t0)/i:.1f}s/piece)", flush=True)
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"  FAIL {m['seg_id'][:40]}: {exc!r}", flush=True)
    print(f"\nembedded {ok} ({fail} failed). Next: --stage eval", flush=True)


def _oof_tau(X, y, composers, n_folds, seeds):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def _folds(comp, k, seed):
        uniq = sorted(set(comp))
        sizes = {c: int(np.sum(comp == c)) for c in uniq}
        order = np.random.default_rng(seed).permutation(len(uniq))
        counts = [0] * k
        fold_of = {}
        for idx in order:
            c = uniq[idx]
            f = int(np.argmin(counts))
            fold_of[c] = f
            counts[f] += sizes[c]
        return [np.array([i for i, c in enumerate(comp) if fold_of[c] == f]) for f in range(k)]

    alphas = np.logspace(-1, 5, 25)
    taus = []
    for seed in seeds:
        oof = np.full(len(y), np.nan)
        for te in _folds(composers, n_folds, seed):
            tr = np.setdiff1d(np.arange(len(y)), te)
            if len(tr) < 3 or len(te) == 0:
                continue
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
            model.fit(np.nan_to_num(X[tr]), y[tr])
            oof[te] = model.predict(np.nan_to_num(X[te]))
        t = tau_c(oof, y)
        if t is not None:
            taus.append(t)
    return {"mean": float(np.mean(taus)) if taus else None,
            "std": float(np.std(taus)) if taus else None, "n_seeds": len(taus)}


def stage_eval(repeats: int):
    manifest = json.loads(MANIFEST.read_text())
    seeds = list(range(2026, 2026 + repeats))
    aria, sym, y, comp = [], [], [], []
    for m in manifest:
        p = EMB_DIR / f"{m['seg_id']}.npz"
        if not p.exists():
            continue
        z = np.load(p)  # numeric-only cache -> no pickle needed
        aria.append(z["aria"])
        sym.append(z["sym"])
        y.append(int(m["grade"]))          # grade/composer from the manifest, joined by seg_id
        comp.append(str(m["composer"]))
    aria = np.stack(aria)
    sym = np.stack(sym)
    y = np.array(y, int)
    comp = np.array(comp)
    print(f"eval on {len(y)} pieces (Aria {aria.shape[1]}-d vs symbolic {sym.shape[1]}-d), "
          f"composer-disjoint {N_FOLDS}-fold x {repeats} seeds", flush=True)

    res = {
        "n": int(len(y)),
        "aria": _oof_tau(aria, y, comp, N_FOLDS, seeds),
        "symbolic_37": _oof_tau(sym, y, comp, N_FOLDS, seeds),
    }
    a, s = res["aria"]["mean"], res["symbolic_37"]["mean"]
    verdict = {
        "aria_minus_symbolic_same_sample": (a - s) if (a and s) else None,
        "aria_minus_full_task_0764": (a - SYMBOLIC_FULL_TASK) if a else None,
        "aria_beats_symbolic": bool(a and s and a > s),
    }
    OUT.write_text(json.dumps({"results": res, "verdict": verdict}, indent=2, default=float))

    def _f(d):
        return f"{d['mean']:.4f} +/- {d['std']:.4f}" if d and d["mean"] is not None else "None"
    print("\n=== PHASE 5b: ARIA SYMBOLIC-EMBEDDING PROBE ===")
    print(f"  n={len(y)}  (RidgeCV head for both, same folds)")
    print(f"    Aria embedding (512-d)   {_f(res['aria'])}")
    print(f"    symbolic 37-feat         {_f(res['symbolic_37'])}")
    print(f"    reference: symbolic full-task = {SYMBOLIC_FULL_TASK}")
    print(f"  Aria - symbolic (same sample) = {verdict['aria_minus_symbolic_same_sample']}")
    print(f"  Aria - full-task 0.764        = {verdict['aria_minus_full_task_0764']}")
    print(f"  Aria beats symbolic: {verdict['aria_beats_symbolic']}")
    print(f"  wrote {OUT}")


def main():
    argv = sys.argv
    stage = argv[argv.index("--stage") + 1] if "--stage" in argv else "embed"
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 600
    repeats = int(argv[argv.index("--repeats") + 1]) if "--repeats" in argv else 5
    if stage == "embed":
        stage_embed(limit)
    elif stage == "eval":
        stage_eval(repeats)
    else:
        raise SystemExit(f"unknown --stage {stage!r} (embed|eval)")


if __name__ == "__main__":
    main()
