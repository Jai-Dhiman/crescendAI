# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "numpy>=1.24.0", "scipy>=1.10.0", "torch>=2.0.0", "peft==0.11.1",
#     "scikit-learn",   # fit_head_from_fold_embeddings only; unused at inference
#     "pretty_midi",    # write_notes_midi + transkun_cli's MIDI parse
#     "soundfile",      # transkun_cli imports it at module scope
#     "mido", "music21", "pandas", "tqdm", "regex", "requests",
#     "filelock", "pyyaml", "safetensors", "tokenizers==0.19.1",
#     "huggingface_hub",
# ]
# ///
"""#166 (#104 S1): the MIREX Track A submission seam.

    score_wav(path_to_wav) -> one real-valued difficulty score

That signature IS the competition contract, and until this module existed
nothing in the repo implemented it: `realaudio_check.py` owns the front of the
pipe (WAV -> Transkun) as a batch CLI, `audio_emb_extract.py` owns the back
(notes -> MoonBeam+adapter -> mean-pool) reading a transcription cache, and
`ft_eval.py` fits its ridge heads inside a CV loop and throws them away. This
module joins the two halves and persists the head.

    WAV -> Transkun -> notes -> MIDI -> MoonBeam+LoRA -> mean-pool -> ridge -> float

**No cache anywhere on this path.** A cached transcription is what made the
research harness fast and is exactly what MIREX will not give us.

## The head is shipped as arrays, not as a pickle

`ft_eval.py` scores through `StandardScaler | RidgeCV`. A pickled sklearn
pipeline is welded to the sklearn version that wrote it, and the container pins
its own env, so the head travels as four numpy arrays and inference is
`(x - mean) / scale @ coef + intercept`. `test_score_wav.py` pins that this
reproduces the sklearn pipeline it was fit from -- otherwise every tau-c we
measured would describe a system other than the one submitted.

## The failure policy, and why the default is still LOUD

`model/CLAUDE.md` mandates loud failure over silent fallback. The contract says
submissions *"failing on >5% of items are excluded from ranking"*, so on
submission day an uncaught exception on one bad WAV can cost everything.

Both behaviours exist, and **`--on-failure raise` is the default**: while we are
still building, a fallback would convert real bugs into plausible-looking median
scores, and we would ship the bug. `--on-failure fallback` is the submission-day
setting -- flip it when the deadline is close and the failure modes are known.
The integration suite exercises both, so the flip is covered, not a leap.

`score_wav` raises and is the honest primitive the research harness may use.
`score_wav_or_fallback` emits the corpus median and logs `SCORE_FAILURE` to
stderr -- loud *and* non-fatal. **Do not propagate `score_wav_or_fallback` back
into the research harness.**

## Environment

Run under the ISOLATED env this file's `# /// script` header defines, never the
shared `model/.venv`: the MoonBeam fork's vendored transformers hard-requires
`tokenizers>=0.19,<0.20` and the shared venv carries 0.22.1, so it dies on
import before any peft code runs. `peft==0.11.1` is the version that WROTE
these adapters.

    cd model/src/claim_measurement/difficulty
    uv run --no-project --script score_wav.py --model-dir <dir> --wav <file.wav> \\
        --checkpoint .../moonbeam_839M.pt --repo-root .../moonbeam/repo \\
        --model-config .../repo/src/llama_recipes/configs/model_config.json

`python -m claim_measurement.difficulty.score_wav` still works for the tests,
which inject the transcriber and the embedder and never touch the fork.

## Model directory layout

    <model-dir>/adapter/        the peft LoRA adapter (config + safetensors)
    <model-dir>/ridge_head.npz  mean, scale, coef, intercept, fallback_score
    <model-dir>/manifest.json   provenance: what trained it, what fit the head

Nothing here reports a tau-c. Per #104: the recipe is validated on folds; a
model directory is a deployment of that recipe, and a number measured on one is
not a result.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Under `uv run --script` this file runs as a loose script, so the package it
# imports from is not on sys.path. __file__-anchored, never CWD-relative.
# A no-op under `python -m`.
_SRC_ROOT = str(Path(__file__).resolve().parents[2])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from claim_measurement.difficulty.audio_emb_extract import (  # noqa: E402
    build_fold_embedder,
    write_notes_midi,
)

# The SAME grid every measured arm was fit through. A different grid is a
# different model wearing the same number; test_score_wav.py pins the equality.
ALPHAS = np.logspace(-1, 5, 25)

ADAPTER_SUBDIR = "adapter"
HEAD_FILENAME = "ridge_head.npz"
MANIFEST_FILENAME = "manifest.json"


# --------------------------------------------------------------------------
# The head
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RidgeHead:
    """A fitted StandardScaler + Ridge, flattened to arrays so it survives an
    sklearn version bump between the machine that fits it and the container
    that runs it.

    `fallback_score` travels with the head because the defensible score for a
    failed item is a property of the training corpus the head was fit on, and
    keeping it anywhere else lets the two drift apart.
    """

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    fallback_score: float

    @property
    def n_features(self) -> int:
        return int(self.mean.shape[0])

    def predict(self, vector) -> float:
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vector.shape[0] != self.n_features:
            raise ValueError(
                f"ridge head expects {self.n_features}-dim embeddings, got "
                f"{vector.shape[0]} -- the staged adapter and head do not match")
        z = (vector - self.mean) / self.scale
        return float(z @ self.coef + self.intercept)


def fit_ridge_head(X, y, fallback_score: float, alphas=ALPHAS) -> RidgeHead:
    """Fit the deployable head exactly as `ft_eval.oof_tau_per_fold` fits its
    per-fold ones -- same scaler, same RidgeCV, same alpha grid -- then flatten
    it. Fitting differently from how we measured would silently deploy an
    unmeasured system."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    model.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridgecv"]
    return RidgeHead(
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        # StandardScaler already maps a zero-variance column to scale 1.0
        # rather than 0.0; a constant embedding dimension is plausible and
        # dividing by its std would emit nan for every piece.
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coef=np.asarray(ridge.coef_, dtype=np.float64).reshape(-1),
        intercept=float(ridge.intercept_),
        fallback_score=float(fallback_score))


def write_ridge_head(path: Path, head: RidgeHead) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=head.mean, scale=head.scale, coef=head.coef,
             intercept=np.float64(head.intercept),
             fallback_score=np.float64(head.fallback_score))


def read_ridge_head(path: Path) -> RidgeHead:
    with np.load(path) as z:
        return RidgeHead(
            mean=z["mean"], scale=z["scale"], coef=z["coef"],
            intercept=float(z["intercept"]),
            fallback_score=float(z["fallback_score"]))


# --------------------------------------------------------------------------
# The seam
# --------------------------------------------------------------------------


def score_wav(wav_path, transcribe, embed, head: RidgeHead) -> float:
    """One WAV in, one real-valued difficulty score out.

    Raises on any failure -- this is the honest primitive. The container calls
    `score_wav_or_fallback` instead; see the module docstring for why that
    inversion is confined to the container.

    `transcribe(wav_path) -> (notes, pedals)` and `embed(midi_path) -> vector`
    are injected so the fast test suite never needs Transkun, the 1.6GB
    checkpoint, or the isolated env. `load_scorer` wires the real ones.
    """
    wav_path = Path(wav_path)
    notes, _pedals = transcribe(wav_path)
    if not notes:
        raise ValueError(
            f"{wav_path} transcribed to no notes at all -- an embedding of an "
            f"empty MIDI carries no difficulty signal, so this is a failure to "
            f"fall back from, not a score to report")

    # MoonBeam's MusicTokenizer.midi_to_compound consumes a FILE, not a note
    # list, so the transcription is materialised and thrown away per item. The
    # container scores a whole test set in one process; a leaked temp MIDI per
    # item would be a slow disk-fill rather than a visible failure.
    with tempfile.TemporaryDirectory() as tmp:
        midi_path = Path(tmp) / "piece.mid"
        write_notes_midi(notes, midi_path)
        vector = embed(midi_path)

    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)):
        # Deliberately NOT nan_to_num'd. ft_eval nan_to_num's features37
        # because hand features are legitimately undefined sometimes; a
        # non-finite MoonBeam embedding means the backbone produced garbage,
        # and zeroing it would emit a confident score from a broken forward
        # pass.
        raise ValueError(
            f"{wav_path} produced a non-finite embedding "
            f"({int((~np.isfinite(vector)).sum())} of {vector.size} entries)")

    score = head.predict(vector)
    if not math.isfinite(score):
        raise ValueError(f"{wav_path} produced a non-finite score: {score!r}")
    return score


def score_wav_or_fallback(wav_path, transcribe, embed, head: RidgeHead,
                          log=None) -> tuple[float, bool]:
    """The container's entry point. Returns `(score, ok)`; never raises.

    The contract excludes any submission failing on >5% of items, so a raise
    here can cost everything. The `SCORE_FAILURE` line on stderr is what keeps
    this from being the silent fallback `model/CLAUDE.md` forbids -- the run's
    failure rate must be readable off the log, and it is what the <5% budget is
    checked against.
    """
    try:
        return score_wav(wav_path, transcribe, embed, head), True
    except Exception as exc:  # noqa: BLE001 -- inverted on purpose; see docstring
        print(f"SCORE_FAILURE {wav_path}: {exc!r}",
              file=log if log is not None else sys.stderr, flush=True)
        return float(head.fallback_score), False


def load_scorer(model_dir: Path, checkpoint: Path, repo_root: Path,
                model_config: Path, transcribe=None, on_failure: str = "raise",
                device=None):
    """Build the real `score(wav_path) -> (score, ok)` from a model directory.

    `on_failure="raise"` (the default) propagates the exception: while the
    system is still being built, a fallback turns a real bug into a
    plausible-looking median score and we ship the bug. `"fallback"` is the
    submission-day setting the >5% clause requires.

    The 839M backbone and the LoRA adapter are loaded ONCE and closed over --
    reloading per item would dominate the 24h budget. `build_fold_embedder` is
    reused rather than reimplemented: it owns the loader -> stub -> peft import
    order (peft binds transformers at import, so a wrong order silently skips
    the model stubs) and the `.eval()` that keeps `lora_dropout=0.05` from
    making every score a fresh random draw.
    """
    if on_failure not in ("raise", "fallback"):
        raise ValueError(
            f"on_failure must be 'raise' or 'fallback', got {on_failure!r}")
    model_dir = Path(model_dir)
    head = read_ridge_head(model_dir / HEAD_FILENAME)
    embed = build_fold_embedder(model_dir / ADAPTER_SUBDIR, checkpoint,
                                repo_root, model_config, device=device)
    if transcribe is None:
        from claim_measurement.difficulty.realaudio_check import (
            _import_transcribe_wav,
        )

        transcribe = _import_transcribe_wav()

    def score(wav_path) -> tuple[float, bool]:
        if on_failure == "raise":
            return score_wav(wav_path, transcribe, embed, head), True
        return score_wav_or_fallback(wav_path, transcribe, embed, head)

    return score


# --------------------------------------------------------------------------
# Building a model directory
# --------------------------------------------------------------------------


def fit_head_from_fold_embeddings(fold_emb_path: Path, exclude_seg_ids=None
                                  ) -> RidgeHead:
    """Fit one deployable head from a `train_fold.py` emb_fold{F}.npz.

    `exclude_seg_ids` drops that fold's own test pieces, so a head built from a
    per-fold artifact is fit only on rows that adapter also trained on. Passing
    None fits on every row, which is what the all-data submission model wants
    and what a per-fold artifact must NOT do.
    """
    from claim_measurement.difficulty.train_fold import read_fold_embeddings

    fold = read_fold_embeddings(Path(fold_emb_path))
    keep = np.array([s not in set(exclude_seg_ids or ()) for s in fold["seg_ids"]])
    X = fold["embeddings"][keep]
    y = fold["grades"][keep].astype(np.float64)
    if len(y) == 0:
        raise ValueError(f"{fold_emb_path} has no rows left after exclusions")
    # The median GRADE of the training corpus, not the median prediction: a
    # failed item should land where an uninformative guess belongs on the
    # ordinal scale the tau-c is computed over.
    return fit_ridge_head(X, y, fallback_score=float(np.median(y)))


def write_manifest(model_dir: Path, **provenance) -> None:
    """Provenance is a submission requirement, not bookkeeping: MIREX 2026 adds
    training-data size, model size, and compute to the mandatory disclosure."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / MANIFEST_FILENAME).write_text(json.dumps(provenance, indent=2))


def main(argv=None, scorer=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--wav", type=Path, action="append", default=[],
                    help="a WAV to score; repeatable")
    ap.add_argument("--wav-list", type=Path, default=None,
                    help="a file of WAV paths, one per line")
    ap.add_argument("--out", type=Path, default=None,
                    help="write '<wav_path>\\t<score>' lines here. Prefer this "
                         "over stdout: the MoonBeam fork's MusicTokenizer prints "
                         "its entire vocabulary (~96KB) to stdout on construction")
    ap.add_argument(
        "--timing-out", type=Path, default=None,
        help="write '<wav_path>\\t<seconds>\\t<ok>' per item here. The 24h/1-GPU "
             "budget is an exclusion criterion, so it needs measuring rather "
             "than estimating")
    ap.add_argument(
        "--on-failure", choices=("raise", "fallback"), default="raise",
        help="raise (default): a failed item aborts the run loudly, which is "
             "what we want while the system is still being built. fallback: "
             "emit the corpus median and log SCORE_FAILURE -- the submission-day "
             "setting the contract's >5%% exclusion clause requires")
    ap.add_argument(
        "--device", default=None,
        help="torch device for the MoonBeam forward pass, e.g. cuda. Default "
             "None = CPU, which is what every #149 measurement ran on. The "
             "container passes cuda: 24h for the whole test set on one GPU does "
             "not fit an 839M CPU forward pass per piece")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--model-config", type=Path, default=None)
    args = ap.parse_args(argv)

    wavs = list(args.wav)
    if args.wav_list is not None:
        wavs += [Path(line) for line in args.wav_list.read_text().split("\n")
                 if line.strip()]
    if not wavs:
        ap.error("pass at least one --wav or a --wav-list")

    if scorer is None:
        for flag, value in (("--checkpoint", args.checkpoint),
                            ("--repo-root", args.repo_root),
                            ("--model-config", args.model_config)):
            if value is None:
                ap.error(f"{flag} is required when the real backbone is used")
        scorer = load_scorer(args.model_dir, args.checkpoint, args.repo_root,
                             args.model_config, on_failure=args.on_failure,
                             device=args.device)

    lines, timings, failures = [], [], 0
    for wav in wavs:
        started = time.perf_counter()
        score, ok = scorer(wav)
        elapsed = time.perf_counter() - started
        failures += not ok
        # repr, not a fixed number of decimals: it is the shortest string that
        # round-trips to the same float, so the determinism harness compares
        # bit-identity rather than agreement to 6 places.
        lines.append(f"{wav}\t{score!r}")
        timings.append((wav, elapsed, ok))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))

    if args.timing_out is not None:
        args.timing_out.parent.mkdir(parents=True, exist_ok=True)
        args.timing_out.write_text("\n".join(
            f"{wav}\t{elapsed:.4f}\t{int(ok)}" for wav, elapsed, ok in timings) + "\n")

    rate = failures / len(wavs)
    # Both numbers are printed on EVERY run because both are exclusion
    # criteria: >5% item failures, or exceeding 24 wall-clock hours on one GPU.
    # The per-item seconds exclude the one-time backbone load, which is
    # amortised across a batch -- so extrapolating them to a test-set size is
    # only honest for a run that scored many items in one process.
    seconds = [t for _w, t, _ok in timings]
    total = sum(seconds)
    print(f"scored={len(wavs)} failures={failures} rate={rate:.3%} "
          f"(MIREX excludes a submission above 5%)", file=sys.stderr)
    print(f"runtime total={total:.1f}s mean={total / len(wavs):.2f}s/item "
          f"max={max(seconds):.2f}s "
          f"-> {24 * 3600 / (total / len(wavs)):.0f} items in a 24h budget "
          f"at this rate", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
