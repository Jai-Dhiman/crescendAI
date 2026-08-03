# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0","scipy>=1.10.0","pretty_midi>=0.2.10",
#     "scikit-learn>=1.3.0","lightgbm>=4.3.0",
# ]
# ///
"""#137 -- do Transkun-unlocked features move difficulty tau-c off the ~0.76 wall?

MEASUREMENT PROTOCOL (read this before trusting any number it prints).

Features come from the ~5,800 TRANSKUN MIDIs, not the clean PSyllabus MIDIs, because
transcribed audio is what the MIREX Docker actually sees. Both arms read the identical
MIDIs, so the comparison isolates FEATURE SET and nothing else.

The metric is mean composer-disjoint 5-fold Kendall tau-c. Composer-disjoint because a
random split lets a model memorize "Czerny pieces are grade 4" and score well without
learning difficulty; 1,066 composers over 5,798 pieces makes the constraint bite.

The arms share ONE fixed GroupKFold split, so every comparison is paired: the same
pieces are held out for every arm, and the difference is bootstrapped over pieces on
the pooled out-of-fold predictions.

WHAT THIS DOES *NOT* COMPARE AGAINST. The 0.824 figure quoted from #135 is the deployed
head fit on ALL PSyllabus records and then scored on a subset of those same records
(stage2_refit_curve.py: `deployed` fits `full_rows`, predicts `tk_all`). That is
train-on-test and cannot be beaten honestly by a cross-validated number. The anchor
here is arm BASE -- the same 37 features, same folds, same model -- so the reported
lift is a like-for-like delta, which is the only quantity worth ratcheting.

Absolute tau-c levels also move with the grade mix, so trust the GAP, not the level.

Stages:
    uv run --script tk_ablation.py --stage extract [--workers 8] [--limit N]
    uv run --script tk_ablation.py --stage cv [--boot 2000]
"""
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3c_explore import candidate_features  # noqa: E402  (the 37-feature superset)
from psyllabus import notes_from_midi_bytes  # noqa: E402
from transkun_features import pedal_from_midi_bytes, transkun_features  # noqa: E402

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
PS_DIR = PRIMARY / "model/data/raw/psyllabus"
LABELS = PS_DIR / "new_clean_data.json"
WORK = PRIMARY / "model/data/results/amt_gap_curve"
TK_DIR = WORK / "transkun_mid"
MANIFEST = WORK / "manifest.json"
FEAT_CACHE = PRIMARY / "model/data/results/mirex_137_tk_features.json"
OUT = PRIMARY / "model/data/results/mirex_137_tk_ablation.json"

N_FOLDS = 5
SEED = 2026
# Identical to the #104 phase3c CV model, so arm BASE reproduces that protocol exactly.
REG_PARAMS = dict(objective="regression", n_estimators=300, learning_rate=0.03,
                  num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.9, reg_lambda=1.0, random_state=SEED,
                  n_jobs=-1, verbosity=-1)


def _extractor_fingerprint() -> str:
    """SHA of the two modules that define the feature values.

    This harness exists to be looped on: edit a feature, re-measure. The failure mode
    that ruins such a loop is silent -- edit `transkun_features.py`, run `--stage cv`,
    forget `--stage extract`, and the cache serves the OLD values while the report reads
    like a verdict on the new ones. Comparing feature NAMES would not catch it either,
    since a changed formula keeps its name. Hashing the source does.
    """
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ("transkun_features.py", "phase3c_explore.py"):
        h.update((here / name).read_bytes())
    return h.hexdigest()[:16]


def tau_c(x, y):
    """Kendall tau-c, nan-safe. Returns None when the input cannot support a rank
    correlation, rather than a misleading 0.0."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)


def _extract_one(entry):
    """Both feature families for one piece, from ONE Transkun MIDI. Returns (row, error)
    so a parse failure is counted and named, never silently dropped."""
    seg_id, key, grade, composer = entry
    path = TK_DIR / f"{seg_id}.mid"
    try:
        raw = path.read_bytes()
        notes = notes_from_midi_bytes(raw)
        if len(notes) < 5:
            return None, f"{seg_id}: too few notes ({len(notes)})"
        pedal = pedal_from_midi_bytes(raw)
        row = {"key": key, "grade": int(grade), "composer": composer}
        row.update(candidate_features(notes))
        row.update(transkun_features(notes, pedal["sustain"], pedal["soft"]))
        return row, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{seg_id}: {exc!r}"


def stage_extract(workers: int, limit: int | None):
    manifest = json.loads(MANIFEST.read_text())
    labels = json.loads(LABELS.read_text())
    entries = []
    for m in manifest:
        if not (TK_DIR / f"{m['seg_id']}.mid").exists():
            continue
        composer = str(labels.get(m["key"], {}).get("composer", "")).strip()
        if not composer:
            continue          # no composer -> cannot be placed in a disjoint fold
        entries.append((m["seg_id"], m["key"], m["grade"], composer))
    if limit:
        entries = entries[:: max(1, len(entries) // limit)][:limit]
    print(f"{len(entries)} pieces with a Transkun MIDI and a composer "
          f"({workers} workers)", flush=True)

    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_extract_one, e) for e in entries}
        for done, fut in enumerate(as_completed(futures), 1):
            row, err = fut.result()
            if row is not None:
                rows.append(row)
            else:
                errors.append(err)
            if done % 500 == 0:
                print(f"  {done}/{len(entries)} ({len(errors)} failed)", flush=True)

    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FEAT_CACHE.write_text(json.dumps({"n_rows": len(rows), "n_failed": len(errors),
                                      "extractor_sha": _extractor_fingerprint(),
                                      "errors": errors[:50], "rows": rows}))
    print(f"\nextracted {len(rows)} rows, {len(errors)} failures -> {FEAT_CACHE}", flush=True)
    for e in errors[:10]:
        print(f"  fail {e}", flush=True)


def _load_matrix():
    cached = json.loads(FEAT_CACHE.read_text())
    rows = cached["rows"]
    if not rows:
        raise SystemExit(f"no rows in {FEAT_CACHE}; run --stage extract first")
    current = _extractor_fingerprint()
    if cached.get("extractor_sha") != current:
        raise SystemExit(
            f"STALE FEATURE CACHE: {FEAT_CACHE} was built by extractor "
            f"{cached.get('extractor_sha')!r} but the code on disk is {current!r}.\n"
            f"Re-run `--stage extract` before `--stage cv`; scoring the old values under "
            f"the new code would report a verdict on features that were never measured.")
    base = [k for k in rows[0] if k not in ("key", "grade", "composer") and not k.startswith("tk_")]
    new = [k for k in rows[0] if k.startswith("tk_")]
    feats = base + new
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r["grade"] for r in rows], int)
    groups = np.array([r["composer"] for r in rows])
    return X, y, groups, feats, base, new


def _oof_predict(X, y, cols, folds, objective):
    """Out-of-fold predictions on FIXED folds, so every arm is scored on the same
    held-out pieces and arm differences are paired per piece."""
    import lightgbm as lgb
    Xs = X[:, cols]
    oof = np.full(len(y), np.nan)
    per_fold = []
    for tr, te in folds:
        if objective == "regression":
            model = lgb.LGBMRegressor(**REG_PARAMS).fit(Xs[tr], y[tr])
            pred = model.predict(Xs[te])
        elif objective == "lambdarank":
            # Rank-native: one query group per training fold, grades 0..10 as relevance.
            params = {k: v for k, v in REG_PARAMS.items() if k != "objective"}
            model = lgb.LGBMRanker(objective="lambdarank", **params)
            model.fit(Xs[tr], y[tr], group=[len(tr)])
            pred = model.predict(Xs[te])
        elif objective == "rank_target":
            # Regression onto the within-training-fold normalized rank of the grade --
            # a cheap ordinal surrogate that keeps the squared loss but removes the
            # assumption that grade steps are equally spaced.
            ranks = stats.rankdata(y[tr]) / len(tr)
            model = lgb.LGBMRegressor(**REG_PARAMS).fit(Xs[tr], ranks)
            pred = model.predict(Xs[te])
        else:
            raise ValueError(f"unknown objective {objective!r}")
        oof[te] = pred
        t = tau_c(pred, y[te])
        if t is not None:
            per_fold.append(t)
    return oof, per_fold


def _paired_boot(oof_a, oof_b, y, n_boot, seed=SEED):
    """Bootstrap the tau-c difference over PIECES, resampling the same indices for both
    arms so the fold-level noise they share cancels."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        ta = stats.kendalltau(oof_a[idx], y[idx], variant="c").statistic
        tb = stats.kendalltau(oof_b[idx], y[idx], variant="c").statistic
        diffs[b] = tb - ta
    lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
    return lo, hi, float(np.mean(diffs > 0))


def stage_cv(n_boot: int):
    X, y, groups, feats, base, new = _load_matrix()
    idx = {f: j for j, f in enumerate(feats)}
    print(f"{len(y)} pieces | {len(base)} base features | {len(new)} transkun features "
          f"| {len(np.unique(groups))} composers", flush=True)

    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups=groups))
    for i, (tr, te) in enumerate(folds):
        assert not (set(groups[tr]) & set(groups[te])), f"fold {i} leaks a composer"
    print(f"composer-disjoint {N_FOLDS}-fold split verified (no composer straddles a fold)",
          flush=True)

    arms = {
        "BASE_37": [idx[f] for f in base],
        "TK_ONLY": [idx[f] for f in new],
        "BASE_PLUS_TK": [idx[f] for f in feats],
    }
    results, oofs = {}, {}
    for objective in ("regression", "lambdarank", "rank_target"):
        for arm, cols in arms.items():
            oof, per_fold = _oof_predict(X, y, cols, folds, objective)
            name = f"{arm}|{objective}"
            oofs[name] = oof
            results[name] = {
                "arm": arm, "objective": objective, "n_features": len(cols),
                "mean_fold_tau_c": float(np.mean(per_fold)),
                "std_fold_tau_c": float(np.std(per_fold)),
                "pooled_oof_tau_c": tau_c(oof, y),
                "per_fold": per_fold,
            }
            r = results[name]
            print(f"  {name:28s} k={len(cols):>3}  mean-fold tau-c "
                  f"{r['mean_fold_tau_c']:.4f} +/- {r['std_fold_tau_c']:.4f}   "
                  f"pooled {r['pooled_oof_tau_c']:.4f}", flush=True)

    # The headline test: does adding the Transkun family beat the 37 alone, same folds?
    anchor = "BASE_37|regression"
    comparisons = {}
    for name in results:
        if name == anchor:
            continue
        lo, hi, p = _paired_boot(oofs[anchor], oofs[name], y, n_boot)
        delta = results[name]["pooled_oof_tau_c"] - results[anchor]["pooled_oof_tau_c"]
        comparisons[name] = {"vs": anchor, "delta_pooled_tau_c": delta,
                             "ci95": [lo, hi], "p_better": p, "significant": lo > 0.0}
        print(f"  {name:28s} vs {anchor}: {delta:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
              f"P={p:.3f} {'SIG' if lo > 0 else 'noise'}", flush=True)

    best = max(results, key=lambda k: results[k]["pooled_oof_tau_c"])
    summary = {"n_pieces": int(len(y)), "n_composers": int(len(np.unique(groups))),
               "n_base_features": len(base), "n_tk_features": len(new),
               "n_folds": N_FOLDS, "anchor": anchor, "arms": results,
               "comparisons_vs_anchor": comparisons, "best_arm": best,
               "protocol": "composer-disjoint GroupKFold, fixed folds shared by all arms; "
                           "paired bootstrap over pieces on pooled OOF predictions"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary}, indent=2))
    print(f"\n  best arm: {best} pooled tau-c {results[best]['pooled_oof_tau_c']:.4f}")
    print(f"  wrote {OUT}", flush=True)


def main():
    def _arg(flag, default):
        return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
    stage = _arg("--stage", "cv")
    if stage == "extract":
        limit = _arg("--limit", None)
        stage_extract(int(_arg("--workers", 8)), int(limit) if limit else None)
    elif stage == "cv":
        stage_cv(int(_arg("--boot", 2000)))
    else:
        raise SystemExit(f"unknown --stage {stage!r} (extract|cv)")


if __name__ == "__main__":
    main()
