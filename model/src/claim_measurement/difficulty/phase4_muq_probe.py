"""Phase 4 -- the MuQ-audio difficulty probe (#104): does a frozen audio encoder beat symbolic?

Every symbolic number (clean 0.764, Phase 3c) reads NOTES. The real MIREX pipeline gets raw
AUDIO; the symbolic path must transcribe first, costing ~0.09 (Phase 3e aria-amt gap). MuQ reads
the WAVEFORM directly, so it (a) has a potentially higher ceiling and (b) bypasses the AMT gap
entirely. This is the decisive experiment for whether an audio edge exists.

Protocol (parity with the symbolic phases):
  * grade-stratified PSyllabus pieces -> YouTube audio (24kHz mono).
  * frozen MuQ-large-msd-iter, layers 9-12 mean-aggregated -> [T, 1024] frame embeddings.
  * pool over time to a piece vector: mean, or mean+std concat (2048-dim).
  * light head (RidgeCV -- the SAME head for every representation, to remove head-confound).
  * composer-disjoint grouped K-fold (repeated over seeds for a stability band), scipy tau-c
    (variant='c', the official MIREX scorer).

Because MuQ embeddings only exist for the downloaded subset (we cannot embed all 7,899 YouTube
audios cheaply), the honest like-for-like is IN-SUBSET grouped K-fold for ALL THREE
representations on the identical pieces + identical folds:
    MuQ-audio   vs   clean-symbolic (notes)   vs   aria-amt-symbolic (transcribed notes).
The Phase-3e train-full/test-subset numbers (clean 0.883 / amt 0.792) are a DIFFERENT protocol
(head trained on 7,899 pieces) and are reported only as context, not head-to-head with MuQ.

STAGES (resumable):
  1. prep:   uv run python phase4_muq_probe.py --stage prep [--per-grade N]
       reuse the Phase-3e 44-piece manifest by default (like-for-like vs amt_mid); --per-grade N
       selects a fresh larger grade-stratified set. Downloads {seg_id}.wav (24k mono) to wav/.
  2. embed:  uv run python phase4_muq_probe.py --stage embed [--layers 9-13]
       frozen MuQ -> pooled piece vectors, cached to pooled/{seg_id}.npz.
  3. eval:   uv run python phase4_muq_probe.py --stage eval [--pool meanstd] [--repeats 5]
       grouped K-fold RidgeCV tau-c for MuQ + clean + amt on the common piece set; gate verdict.

Run from model/ (uses the model .venv: torch + muq + sklearn):
    cd model && uv run python src/claim_measurement/difficulty/phase4_muq_probe.py --stage <s>
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase3c_explore import candidate_features  # noqa: E402  (the 37-feature superset)
from phase3e_amt_gap import select_records, _seg_id, _stitch_amt_notes  # noqa: E402
from psyllabus import load_records, notes_from_midi_bytes  # noqa: E402

PRIMARY = Path("/Users/jdhiman/Documents/crescendai")
PS_DIR = PRIMARY / "model/data/raw/psyllabus"
LABELS = PS_DIR / "new_clean_data.json"
MID_ZIP = PS_DIR / "mid.zip"

# Phase-3e artifacts we reuse for the like-for-like AMT comparison.
AMT_MANIFEST = PRIMARY / "model/data/results/amt_gap/manifest.json"

WORK = PRIMARY / "model/data/results/muq_probe"
WAV_DIR = WORK / "wav"
FRAME_DIR = WORK / "muq_frames"   # raw [T,1024] frame cache (MuQExtractor's own cache)
POOLED_DIR = WORK / "pooled"      # per-piece pooled vectors (npz: mean, meanstd)
MANIFEST = WORK / "manifest.json"
OUT = PRIMARY / "model/data/results/mirex_phase4_muq_probe.json"

# Reference: the full-task symbolic ceiling and the Phase-3e (different-protocol) subset numbers.
SYMBOLIC_FULL_TASK = 0.764   # Phase 3c clean-MIDI, 7,899 pieces, composer-disjoint 5-fold
PHASE3E_CLEAN_SUBSET = 0.883  # train-full/test-44, clean features (context only)
PHASE3E_AMT_SUBSET = 0.792    # train-full/test-44, aria-amt features (context only)


def tau_c(x, y):
    """scipy Kendall tau-c (official MIREX scorer). None if degenerate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    t = stats.kendalltau(x, y, variant="c").statistic
    return None if np.isnan(t) else float(t)


def _composer_map() -> dict[str, str]:
    """key -> composer, from the same records the symbolic phases use."""
    records, _ = load_records(LABELS, MID_ZIP)
    return {r.key: r.composer for r in records}


# ----------------------------------------------------------------------------- prep
def stage_prep(per_grade: int | None):
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    cmap = _composer_map()

    if per_grade is None and AMT_MANIFEST.exists():
        # reuse the exact Phase-3e 44-piece set (so amt_mid/ transcriptions line up).
        src = json.loads(AMT_MANIFEST.read_text())
        manifest = [{**m, "composer": cmap.get(m["key"], "")} for m in src]
        print(f"reusing Phase-3e manifest: {len(manifest)} pieces (like-for-like vs amt_mid)", flush=True)
    else:
        pg = per_grade or 4
        picked = select_records(pg)  # phase3e's grade-stratified deterministic pick
        manifest = [{"seg_id": _seg_id(r.key), "key": r.key, "grade": r.grade,
                     "video_id": vid, "midi_name": r.midi_name,
                     "composer": cmap.get(r.key, "")} for r, vid in picked]
        print(f"selected {len(manifest)} pieces (per_grade={pg}) across grades "
              f"{sorted(set(m['grade'] for m in manifest))}", flush=True)

    MANIFEST.write_text(json.dumps(manifest, indent=2))

    ok, fail = 0, 0
    for m in manifest:
        wav = WAV_DIR / f"{m['seg_id']}.wav"
        if wav.exists():
            ok += 1
            continue
        url = f"https://www.youtube.com/watch?v={m['video_id']}"
        cmd = ["yt-dlp", "-q", "--no-warnings", "-f", "bestaudio", "-x",
               "--audio-format", "wav", "--postprocessor-args", "-ar 24000 -ac 1",
               "-o", str(WAV_DIR / f"{m['seg_id']}.%(ext)s"), url]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and wav.exists():
            ok += 1
            print(f"  dl ok   [{m['grade']}] {m['seg_id'][:40]}", flush=True)
        else:
            fail += 1
            print(f"  dl FAIL [{m['grade']}] {m['seg_id'][:40]}: {r.stderr.strip()[:80]}", flush=True)
    print(f"\ndownloaded {ok}/{len(manifest)} ({fail} failed). Next: --stage embed", flush=True)


# ----------------------------------------------------------------------------- embed
def _pool(frames: np.ndarray) -> dict:
    """[T, D] frame embeddings -> pooled piece vectors."""
    mean = frames.mean(axis=0)
    std = frames.std(axis=0)
    return {"mean": mean.astype(np.float32),
            "meanstd": np.concatenate([mean, std]).astype(np.float32)}


def stage_embed(layer_start: int, layer_end: int):
    import torch  # noqa: F401  (imported for side-effect parity / availability check)
    sys.path.insert(0, str(PRIMARY / "model/src"))
    from audio_experiments.extractors.muq import MuQExtractor

    manifest = json.loads(MANIFEST.read_text())
    POOLED_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    have_wav = [m for m in manifest if (WAV_DIR / f"{m['seg_id']}.wav").exists()]
    todo = [m for m in have_wav if not (POOLED_DIR / f"{m['seg_id']}.npz").exists()]
    cached = len(have_wav) - len(todo)
    print(f"embed: {len(todo)} to pool ({cached} cached) "
          f"layers {layer_start}-{layer_end - 1} mean-agg", flush=True)

    if not todo:
        print("  all pieces already pooled.", flush=True)
        return

    extractor = MuQExtractor(layer_start=layer_start, layer_end=layer_end,
                             layer_aggregation="mean", cache_dir=FRAME_DIR)
    print(f"  MuQ loaded on {extractor.device}; num_layers={extractor.num_layers}", flush=True)

    ok, fail = 0, 0
    for i, m in enumerate(todo, 1):
        wav = WAV_DIR / f"{m['seg_id']}.wav"
        try:
            frames = extractor.extract_from_file(wav).numpy()  # [T, 1024]
            pooled = _pool(frames)
            np.savez(POOLED_DIR / f"{m['seg_id']}.npz",
                     n_frames=frames.shape[0], **pooled)
            ok += 1
            print(f"  [{i}/{len(todo)}] {m['seg_id'][:40]}  T={frames.shape[0]} D={frames.shape[1]}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"  [{i}/{len(todo)}] FAIL {m['seg_id'][:40]}: {exc!r}", flush=True)
    print(f"\npooled {ok} ({fail} failed). Next: --stage eval", flush=True)


# ----------------------------------------------------------------------------- eval
def _clean_feats(midi_name: str, zf) -> dict | None:
    notes = notes_from_midi_bytes(zf.read(midi_name))
    return candidate_features(notes) if len(notes) >= 5 else None


def _amt_feats(seg_id: str) -> dict | None:
    notes = _stitch_amt_notes(seg_id)  # whole-piece from 30s chunks
    return candidate_features(notes) if notes and len(notes) >= 5 else None


def _grouped_folds(composers: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Composer-disjoint folds: whole composers assigned to folds, greedily balancing piece
    count. Deterministic per seed; repeating over seeds gives a stability band on tiny n."""
    uniq = sorted(set(composers))
    sizes = {c: int(np.sum(composers == c)) for c in uniq}
    order = np.random.default_rng(seed).permutation(len(uniq))
    counts = [0] * n_folds
    fold_of = {}
    for idx in order:
        c = uniq[idx]
        f = int(np.argmin(counts))
        fold_of[c] = f
        counts[f] += sizes[c]
    return [np.array([i for i, c in enumerate(composers) if fold_of[c] == f], dtype=int)
            for f in range(n_folds)]


def _oof_tau(X: np.ndarray, y: np.ndarray, composers: np.ndarray,
             n_folds: int, seeds: list[int]) -> dict:
    """Repeated composer-disjoint grouped K-fold, RidgeCV head, out-of-fold tau-c.

    RidgeCV is the common head for EVERY representation (MuQ 2048-d, symbolic 37-d) so the
    comparison isolates representation quality, not head capacity. p>>n for MuQ, so the alpha
    grid runs high; leave-one-group-out CV inside RidgeCV picks alpha per fold."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    alphas = np.logspace(-1, 5, 25)
    taus = []
    for seed in seeds:
        oof = np.full(len(y), np.nan)
        for te in _grouped_folds(composers, n_folds, seed):
            tr = np.setdiff1d(np.arange(len(y)), te)
            if len(tr) < 3 or len(te) == 0:
                continue
            Xtr = np.nan_to_num(X[tr], nan=0.0)
            Xte = np.nan_to_num(X[te], nan=0.0)
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
            model.fit(Xtr, y[tr])
            oof[te] = model.predict(Xte)
        t = tau_c(oof, y)
        if t is not None:
            taus.append(t)
    return {"mean": float(np.mean(taus)) if taus else None,
            "std": float(np.std(taus)) if taus else None,
            "n_seeds": len(taus)}


def stage_eval(pool: str, n_folds: int, repeats: int):
    manifest = json.loads(MANIFEST.read_text())
    seeds = list(range(2026, 2026 + repeats))

    # assemble per-piece: MuQ pooled vector + clean features + amt features + grade + composer
    rows = []
    missing = {"muq": 0, "clean": 0, "amt": 0}
    with zipfile.ZipFile(MID_ZIP) as zf:
        for m in manifest:
            seg = m["seg_id"]
            pooled_p = POOLED_DIR / f"{seg}.npz"
            if not pooled_p.exists():
                missing["muq"] += 1
                continue
            muq = np.load(pooled_p)[pool]
            try:
                cf = _clean_feats(m["midi_name"], zf)
            except Exception:  # noqa: BLE001
                cf = None
            af = _amt_feats(seg)
            if cf is None:
                missing["clean"] += 1
            if af is None:
                missing["amt"] += 1
            rows.append({"grade": m["grade"], "composer": m["composer"],
                         "muq": muq, "clean": cf, "amt": af})

    y = np.array([r["grade"] for r in rows], int)
    composers = np.array([r["composer"] for r in rows])

    # common set where all three representations are present -> the true head-to-head
    common = [i for i, r in enumerate(rows) if r["clean"] is not None and r["amt"] is not None]
    common = np.array(common, dtype=int)
    print(f"pieces with MuQ: {len(rows)}  | common (MuQ+clean+amt): {len(common)}", flush=True)
    print(f"missing: {missing}", flush=True)

    feats = None
    for r in rows:
        if r["clean"] is not None:
            feats = list(r["clean"].keys())
            break

    def _sym_matrix(key, idx):
        return np.array([[rows[i][key][f] for f in feats] for i in idx], float)

    results = {}

    # (1) MuQ on ALL pieces that have an embedding (its natural denominator)
    muq_all = np.stack([r["muq"] for r in rows])
    results["muq_all"] = {
        "n": len(rows), "pool": pool, "dim": int(muq_all.shape[1]),
        **_oof_tau(muq_all, y, composers, n_folds, seeds),
    }

    # (2) head-to-head on the common set: MuQ vs clean vs amt, identical pieces + folds
    if len(common) >= n_folds:
        yc, cc = y[common], composers[common]
        muq_c = np.stack([rows[i]["muq"] for i in common])
        clean_c = _sym_matrix("clean", common)
        amt_c = _sym_matrix("amt", common)
        nc = np.array([rows[i]["clean"]["note_count"] for i in common], float)
        results["head_to_head"] = {
            "n": int(len(common)),
            "muq": _oof_tau(muq_c, yc, cc, n_folds, seeds),
            "clean_symbolic": _oof_tau(clean_c, yc, cc, n_folds, seeds),
            "amt_symbolic": _oof_tau(amt_c, yc, cc, n_folds, seeds),
            "note_count_baseline": {"tau_c": tau_c(nc, yc)},
        }
    else:
        results["head_to_head"] = {"n": int(len(common)), "note": "too few common pieces"}

    # (3) MuQ vs clean-symbolic on THEIR intersection (no aria-amt needed) -- this is the
    # scaled-run head-to-head: aria-amt transcriptions only exist for the 44-piece set, but a
    # larger grade-stratified set still has clean MIDI, so this comparison scales.
    mc = np.array([i for i, r in enumerate(rows) if r["clean"] is not None], dtype=int)
    if len(mc) >= n_folds:
        results["muq_vs_clean"] = {
            "n": int(len(mc)),
            "muq": _oof_tau(np.stack([rows[i]["muq"] for i in mc]), y[mc], composers[mc],
                            n_folds, seeds),
            "clean_symbolic": _oof_tau(_sym_matrix("clean", mc), y[mc], composers[mc],
                                       n_folds, seeds),
        }

    # gate verdict
    h2h = results.get("head_to_head", {})
    mvc = results.get("muq_vs_clean", {})
    muq_tau = (h2h.get("muq") or {}).get("mean")
    amt_tau = (h2h.get("amt_symbolic") or {}).get("mean")
    # prefer the (larger) muq_vs_clean intersection for the MuQ-vs-clean deltas; fall back to h2h
    muq_mvc = (mvc.get("muq") or {}).get("mean")
    clean_mvc = (mvc.get("clean_symbolic") or {}).get("mean")
    muq_ref = muq_mvc if muq_mvc is not None else muq_tau
    gate = {
        "muq_beats_amt_symbolic_in_subset": (muq_tau is not None and amt_tau is not None
                                             and muq_tau > amt_tau),
        "muq_minus_clean_symbolic_in_subset": (None if (muq_mvc is None or clean_mvc is None)
                                               else muq_mvc - clean_mvc),
        "muq_vs_full_task_symbolic_0764": (None if muq_ref is None else muq_ref - SYMBOLIC_FULL_TASK),
        "verdict_note": ("in-subset n is small; a positive MuQ result warrants scaling, "
                         "a null is ambiguous (data-starved p>>n)"),
    }

    summary = {
        "n_pieces_embedded": len(rows), "n_common": int(len(common)),
        "pool": pool, "n_folds": n_folds, "repeats": repeats,
        "reference": {"symbolic_full_task_0764": SYMBOLIC_FULL_TASK,
                      "phase3e_clean_subset_diff_protocol": PHASE3E_CLEAN_SUBSET,
                      "phase3e_amt_subset_diff_protocol": PHASE3E_AMT_SUBSET},
        "results": results, "gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=float))

    def _fmt(d):
        return (f"{d['mean']:.4f} +/- {d['std']:.4f} (n_seeds={d['n_seeds']})"
                if d and d.get("mean") is not None else "None")

    print("\n=== PHASE 4: MuQ-AUDIO DIFFICULTY PROBE ===")
    print(f"  embedded={len(rows)}  common(MuQ+clean+amt)={len(common)}  pool={pool}  "
          f"folds={n_folds} x {repeats} seeds")
    print(f"\n  MuQ on all embedded pieces (n={len(rows)}): tau-c = {_fmt(results['muq_all'])}")
    if "muq" in h2h:
        print(f"\n  HEAD-TO-HEAD on common n={h2h['n']} (identical pieces + folds, RidgeCV head):")
        print(f"    MuQ-audio          {_fmt(h2h['muq'])}")
        print(f"    clean-symbolic     {_fmt(h2h['clean_symbolic'])}")
        print(f"    aria-amt-symbolic  {_fmt(h2h['amt_symbolic'])}")
        print(f"    note_count (floor) {h2h['note_count_baseline']['tau_c']}")
    if "muq" in mvc:
        print(f"\n  MuQ vs CLEAN on their intersection n={mvc['n']} (scales without aria-amt):")
        print(f"    MuQ-audio          {_fmt(mvc['muq'])}")
        print(f"    clean-symbolic     {_fmt(mvc['clean_symbolic'])}")
    print(f"\n  REFERENCE (different protocol, context only):")
    print(f"    symbolic full-task (7,899 pieces)     {SYMBOLIC_FULL_TASK}")
    print(f"    Phase-3e clean subset (train-full)    {PHASE3E_CLEAN_SUBSET}")
    print(f"    Phase-3e amt   subset (train-full)    {PHASE3E_AMT_SUBSET}")
    print(f"\n  GATE:")
    print(f"    MuQ > aria-amt-symbolic (in-subset)?  {gate['muq_beats_amt_symbolic_in_subset']}")
    dc = gate["muq_minus_clean_symbolic_in_subset"]
    print(f"    MuQ - clean-symbolic (in-subset):     {('%+.4f' % dc) if dc is not None else None}")
    d = gate["muq_vs_full_task_symbolic_0764"]
    print(f"    MuQ vs full-task symbolic 0.764:      {('%+.4f' % d) if d is not None else None}")
    print(f"    note: {gate['verdict_note']}")
    print(f"\n  wrote {OUT}")


def main():
    argv = sys.argv
    stage = argv[argv.index("--stage") + 1] if "--stage" in argv else "prep"
    per_grade = int(argv[argv.index("--per-grade") + 1]) if "--per-grade" in argv else None
    pool = argv[argv.index("--pool") + 1] if "--pool" in argv else "meanstd"
    repeats = int(argv[argv.index("--repeats") + 1]) if "--repeats" in argv else 5
    n_folds = int(argv[argv.index("--folds") + 1]) if "--folds" in argv else 5
    layers = argv[argv.index("--layers") + 1] if "--layers" in argv else "9-13"
    ls, le = (int(x) for x in layers.split("-"))

    if stage == "prep":
        stage_prep(per_grade)
    elif stage == "embed":
        stage_embed(ls, le)
    elif stage == "eval":
        stage_eval(pool, n_folds, repeats)
    else:
        raise SystemExit(f"unknown --stage {stage!r} (prep|embed|eval)")


if __name__ == "__main__":
    main()
