# MoonBeam-839M LoRA training runbook

Operator sequence for training a MoonBeam-839M LoRA difficulty model on HF Jobs.
Every step below that spends money or touches a GPU is **human-lit**: the
operator runs it, reads the printed numbers, and decides whether to continue.
Nothing in this repo launches an HF Job automatically.

**Scope note (2026-08-06).** This began as the #149 Phase 1 gate runbook. Both
gates are now measured and recorded in
[track-a-difficulty-prediction.md](./track-a-difficulty-prediction.md)'s
decision log, so the completed gate stages have been pruned. What remains —
Stages 0 to 3.5 — is the **live retraining procedure**, and the submission
depends on it: the final model must be retrained on all compliant pieces, and
again if the forbidden-composer list touches our pools.

Evaluate any retrained set of adapters with the same gate protocol that measured
Phase 1 (free, CPU, ~1 min):

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run python -m \
    claim_measurement.difficulty.ft_eval \
    --data-root /Users/jdhiman/Documents/crescendai/model/data \
    --fold-emb-dir <fold_embeddings dir>
```

Reference values at seed 2026, n=900: `features37` 0.8038, `moonbeam_ft_mean`
0.8395, delta +0.0357 `SIG`. `matched_features37.py` adds the
supervision-matched baseline (0.8068) and the honest delta (+0.0325 `SIG`). A
`noise` verdict on a retrain is a real finding, not a bug to work around.

## Stage 0 — one-time setup (already covered by moonbeam_extract_script.py)

Clone the MoonBeam fork at the pinned commit and fetch the checkpoint (see
`model/src/claim_measurement/difficulty/moonbeam_extract_script.py`'s module
docstring for the exact commands):

```bash
W=/Users/jdhiman/Documents/crescendai/model/data/weights/moonbeam
git clone https://github.com/guozixunnicolas/moonbeam-midi-foundation-model $W/repo
git -C $W/repo checkout 4e2c015c89ae44a9542a7e9a67f9d7098f487ef1
hf download guozixunnicolas/moonbeam-midi-foundation-model moonbeam_839M.pt --local-dir $W
```

Confirm the checkpoint loads against `model_config.json` (the 839M config:
hidden 1920, 15 layers, NOT `model_config_small.json`, the 309M config) with
zero missing/unexpected keys — `_real_loader`'s strict check in both
`moonbeam_extract_script.py` and `train_fold.py` refuses to proceed otherwise.

Before staging anything, sanity-check the fold-plan sizes against the design
spec's verified facts (this worktree's `model/data` is empty; point
`--data-root` at the main checkout):

```bash
cd /Users/jdhiman/Documents/crescendai/model/src/claim_measurement/difficulty && \
    uv run --no-project --script features37_compare.py \
    --data-root /Users/jdhiman/Documents/crescendai/model/data
```

Expected: `features37|ridge` tau-c `0.8048`, `moonbeam_mean|ridge` tau-c `0.8257`
(the Phase 0 numbers this whole phase is measured against — Task Group 0's
harness). If either number has drifted, STOP: something about the data or the
fold seed has changed and the gate threshold below is no longer valid.

## Stage 1 — stage and upload the training bundle (once)

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run python -m \
    claim_measurement.difficulty.push_train_dataset \
    --manifest /Users/jdhiman/Documents/crescendai/model/data/results/amt_gap_curve/manifest.json \
    --labels /Users/jdhiman/Documents/crescendai/model/data/raw/psyllabus/new_clean_data.json \
    --sample-manifest /Users/jdhiman/Documents/crescendai/model/data/results/bakeoff/sample_manifest.json \
    --midi-dir /Users/jdhiman/Documents/crescendai/model/data/results/amt_gap_curve/transkun_mid \
    --features37-dir /Users/jdhiman/Documents/crescendai/model/data/results/bakeoff/emb/features37 \
    --repo-snapshot-dir /Users/jdhiman/Documents/crescendai/model/data/weights/moonbeam/repo \
    --staging-dir /Users/jdhiman/Documents/crescendai/model/data/results/phase1_lora/bundle \
    --repo-id Jai-D/phase1-lora-bundle
```

`--n-folds` (default 5), `--seed` (default 2026), and `--val-frac` (default
0.12) all have defaults matching the design — pass them explicitly only if
deviating. `--features37-dir` defaults to
`<--data-root>/results/bakeoff/emb/features37`; it is passed explicitly above
so the command does not depend on which checkout `--data-root` resolves to.

Read the printed `staged N MIDIs, 5 fold plans, 900 eval pieces, ..., 2 code
files, ...` report before it uploads. Abort criteria:

- `n_midis` far from ~5798 (the deduplicated union of every fold's
  train+val+test) — re-check the sample manifest and the labels join;
- `n_eval_pieces` not exactly 900 — `eval_manifest.json` fixes the row order
  `train_fold.py` emits `emb_fold{F}.npz` in, and `ft_eval.py` rejects any
  fold file whose seg_ids do not match features37's order exactly.

The bundle carries everything the job needs except the checkpoint:

- `midi/` — one `.mid` per referenced piece;
- `grades.json`, `fold_plans.json`;
- `eval_manifest.json` — `{seg_id, grade, composer_id}` for all 900 eval
  pieces, in features37's row order;
- `moonbeam_repo/` — the fork snapshot, with `.git`, `__pycache__` and `*.pyc`
  excluded;
- `code/` — `ranking_loss.py` and `bakeoff_cv.py` verbatim, because
  `hf jobs uv run <file>` uploads only the one file named on the command line,
  never the rest of this package.

`moonbeam_839M.pt` is deliberately NOT staged: it is 1.6 GB and public, so
`train_fold.py` downloads it from
`guozixunnicolas/moonbeam-midi-foundation-model` inside the container instead.

**Re-stage once if you uploaded a bundle before #149.** Earlier bundles were
~119 MB, almost all of it the fork's `.git` history plus 16 `__pycache__`
dirs of stale bytecode, and they have no `eval_manifest.json` — a job against
one of those aborts immediately on the missing default. Delete the staging dir
and re-run the command above; the upload overwrites the same repo.

## Stage 2 — the pilot fold

```bash
hf jobs uv run \
    --flavor a100-large --timeout 3h --secrets HF_TOKEN \
    model/src/claim_measurement/difficulty/train_fold.py \
    --fold 0 --bundle-repo Jai-D/phase1-lora-bundle \
    --output-repo Jai-D/phase1-lora-fold0 --out-dir /data/fold0
```

That is the whole submit line. `--fold-plan`, `--pool-grades`,
`--eval-manifest`, `--midi-dir`, `--repo-root` and `--model-config` all default
to their location inside the downloaded bundle, which is the only thing that
can work: the bundle lands at `snapshot_download`'s cache path, unknowable at
submit time. An explicitly-passed path still wins (that is what a local run
against loose files relies on), and a default resolving to a nonexistent file
aborts immediately, naming the path. `--checkpoint` is likewise optional: with
none given, `moonbeam_839M.pt` is fetched from `--checkpoint-repo`
(default `guozixunnicolas/moonbeam-midi-foundation-model`).

`--bundle-repo` is the SAME `--repo-id` Stage 1 uploaded to. `--output-repo`
uploads `adapter/` and `emb_fold0.npz` to a Hub model repo once training
finishes — **do not omit it for a real job run**: the job container's local
disk (`--out-dir`), and the GPU time that filled it, is discarded when the
container exits. (`--bundle-dir` is accepted in place of `--bundle-repo` for a
local/offline run against an already-downloaded bundle; it is not useful
inside a fresh job container.)

Add `--micro-batch 4` if the pilot does not fit `a100-large` at the default 8.

`--device` is not on the submit line because it defaults to `cuda` and **aborts
rather than falling back to CPU**. That default is the fix for job `6a73a7d7`,
which rented an `a100-large` and trained on its CPU for an hour because the
script had no device handling at all. A local run must say `--device cpu` out
loud. Trackio is initialised before the checkpoint download, so a telemetry
misconfiguration kills the job in seconds instead of after GPU spend; pass
`--trackio-space <user>/<space>` to keep the metrics after the container exits
(without it they are written to container disk and lost), or `--no-trackio` for
a local run.

Monitor with `hf jobs ps`, `hf jobs logs <job-id>`, `hf jobs inspect <job-id>`.
The first log lines are `device: cuda:0 (...)`, `trainable params: N on cuda:0`
and a `steps/epoch x epochs` total; then a throughput line every
`--log-every` (default 10) steps carrying `loss`, `s/step`, `pieces/s`,
`elapsed` and `eta`.

Abort criteria (design spec's Open Questions):
- **The device line says `cpu`, or `s/step` implies CPU speed** — kill the job
  immediately. This is checkable in minute one, before real money is spent.
- **Val ranking tau is flat or diverging** across the printed `epoch N:
  val_ranking_tau=...` lines — **this criterion misfired on the real pilot and
  is advisory only.** Fold 0 printed 0.8261 → 0.8196 → 0.8226 (flat) and still
  produced a gate-passing encoder: that tau measures the *discarded* head on a
  *single random window* per piece, not the mean-pooled embeddings the gate
  scores. If it looks flat, do not stop — download the fold's
  `emb_fold0.npz` and run the gate protocol on that one fold (free, ~10 s).
  A single fold is underpowered (n=180) but will catch a catastrophe.
- **Peak memory does not fit `a100-large` at `--micro-batch 8`** — drop to
  `--micro-batch 4` and retry the pilot before scaling to the remaining folds
  (do not switch GPU flavor first).
- **Measured wall-clock is wildly off from the ~1 GPU-hr/fold estimate** — use
  the MEASURED number, not the estimate, to budget folds 1-4 (`hf jobs stats`
  after completion).

If the pilot's `emb_fold0.npz` looks reasonable (900 rows, finite values), proceed.

## Stage 3 — folds 1-4 (same seed, ~$13 total for all 5)

```bash
cd /Users/jdhiman/Documents/crescendai && for f in 1 2 3 4; do
    hf jobs uv run --flavor a100-large --timeout 3h --secrets HF_TOKEN \
        model/src/claim_measurement/difficulty/train_fold.py \
        --fold $f --bundle-repo Jai-D/phase1-lora-bundle \
        --output-repo Jai-D/phase1-lora-fold$f --out-dir /data/fold$f
done
```

Same bundle, same seed; only `--fold`, `--output-repo` and `--out-dir` change.

## Stage 3.5 — bring the fold embeddings back to local disk

Each fold's `train_fold.py` run uploaded its artifacts to its own
`--output-repo` and left nothing useful on the job container's disk once it
exited. Download `emb_fold{F}.npz` for all 5 folds into the SAME
`fold_embeddings/` layout Stage 4 expects, before running it. Stage 5 needs
each fold's `adapter/` directory, which the same download pulls:

```bash
for f in 0 1 2 3 4; do
    hf download Jai-D/phase1-lora-fold$f --local-dir \
        /Users/jdhiman/Documents/crescendai/model/data/results/phase1_lora/fold_embeddings/fold$f
done
```
