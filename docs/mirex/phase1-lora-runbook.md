# #138 Phase 1 LoRA Fine-Tune Runbook

Operator sequence for the MoonBeam-839M LoRA fine-tune gate (#149). Every step
below that spends money or touches a GPU is **human-lit**: the operator runs it,
reads the printed numbers, and decides whether to continue. Nothing in this
repo launches an HF Job automatically.

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
  val_ranking_tau=...` lines — stop, the objective/LR is not working, do not
  spend money on folds 1-4.
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

## Stage 4 — gate (i): encoder-as-feature-extractor (local, CPU, free)

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run python -m \
    claim_measurement.difficulty.ft_eval \
    --data-root /Users/jdhiman/Documents/crescendai/model/data \
    --fold-emb-dir /Users/jdhiman/Documents/crescendai/model/data/results/phase1_lora/fold_embeddings
```
Expected output: `moonbeam_ft_mean|ridge - features37|ridge: +0.0XXX
CI95[+a,+b] P(diff<=0)=p SIG|noise`. **The gate passes only if `a > 0`
(`SIG`).** If `noise`, STOP — do not proceed to the real-audio gate or report
an end-to-end number; the fine-tune did not clear 0.8048.

## Stage 5 — gate (ii): real-audio second gate (local, resumable)

**5a. Build the WAV manifest** (709 of the 900 eval pieces have a local WAV;
pieces without one are omitted, never faked):

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run python -m \
    claim_measurement.difficulty.realaudio_check \
    --write-wav-manifest /Users/jdhiman/Documents/crescendai/model/data/results/phase1_lora/audio_wav_manifest.json \
    --wav-dir /Users/jdhiman/Documents/crescendai/model/data/results/amt_gap_curve/wav \
    --data-root /Users/jdhiman/Documents/crescendai/model/data
```
Expected: `wrote 709 of 900 eval pieces with a WAV ...`. The seg_ids come from
`emb/features37/`, so the manifest is in the same canonical order everything
else in this phase is.

**5b. Transcribe** (resumable — safe to interrupt and re-run):

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run python -m \
    claim_measurement.difficulty.realaudio_check \
    --wav-manifest /Users/jdhiman/Documents/crescendai/model/data/results/phase1_lora/audio_wav_manifest.json \
    --out-dir /Users/jdhiman/Documents/crescendai/model/data/results/phase1_lora/audio_midi_cache
```

**5c. Extract audio embeddings through each piece's OWN fold adapter**
(resumable; loads each fold's adapter at most once):

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run python -m \
    claim_measurement.difficulty.audio_emb_extract \
    --cache-dir /Users/jdhiman/Documents/crescendai/model/data/results/phase1_lora/audio_midi_cache \
    --out-dir /Users/jdhiman/Documents/crescendai/model/data/results/phase1_lora/audio_emb \
    --adapter-root /Users/jdhiman/Documents/crescendai/model/data/results/phase1_lora/fold_embeddings \
    --data-root /Users/jdhiman/Documents/crescendai/model/data \
    --checkpoint /Users/jdhiman/Documents/crescendai/model/data/weights/moonbeam/moonbeam_839M.pt \
    --repo-root /Users/jdhiman/Documents/crescendai/model/data/weights/moonbeam/repo \
    --model-config /Users/jdhiman/Documents/crescendai/model/data/weights/moonbeam/repo/src/llama_recipes/configs/model_config.json
```

The fold each piece belongs to comes from `bakeoff_cv.composer_disjoint_folds`
at the same `(5, 2026)` every other stage uses — imported, not reimplemented,
because "the same folds" only holds if it is literally the same function.
Scoring a piece through an adapter trained on it would be train-on-test.
Extraction runs in `eval()` mode (LoRA dropout would otherwise make the
embeddings stochastic) and chunks the piece to `max_len` and mean-pools over
all tokens, matching `moonbeam_extract_script.py`, so the arm stays paired
against frozen 0.8257. Output is one `.npz` per piece with pooling key
`"mean_pool"` — exactly what the gate snippet below reads.

**5d. The gate.** Audio vs. features37 on the SAME subset, scored through the
SAME composer-disjoint folds/seed, plus the matched symbolic comparison that
makes it interpretable:

```python
from pathlib import Path
from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
from claim_measurement.difficulty.ft_eval import _load_features37
from claim_measurement.difficulty.train_fold import read_fold_embeddings
from claim_measurement.difficulty.realaudio_check import score_audio_subset
from claim_measurement.difficulty.bakeoff_paths import resolve_paths

root = Path("/Users/jdhiman/Documents/crescendai/model/data")
phase1 = root / "results" / "phase1_lora"
emb_root = resolve_paths(root).emb_root
Xf, y, composers, seg_ids = _load_features37(emb_root)
emb_by_fold = {
    f: read_fold_embeddings(phase1 / f"fold_embeddings/fold{f}/emb_fold{f}.npz")["embeddings"]
    for f in range(5)}
audio_embeddings = {p.stem: read_embedding_npz(p).embeddings["mean_pool"]
                    for p in sorted((phase1 / "audio_emb").glob("*.npz"))}

result = score_audio_subset(emb_by_fold, audio_embeddings, Xf, y, composers, seg_ids,
                            n_folds=5, seed=2026)
print(result)
```
Run it from `/Users/jdhiman/Documents/crescendai/model` under `uv run python`,
so `claim_measurement` resolves.

Expected: `audio_tau_c`, `symbolic_tau_c`, and `features37_tau_c` all reported.
`delta_vs_features37`/`ci_lo_vs_features37`/`ci_hi_vs_features37` are **THE
GATE** (item (a) of the design spec's "Real-audio second gate": tau-c on the
audio subset, paired-bootstrapped against features37 on the same pieces).
**The gate passes only if `ci_lo_vs_features37 > 0`** on this n=709(-ish)
subset (half-width ≈ ±0.017 per the design spec, enough to resolve the
+0.024 margin). `delta_vs_symbolic`/`ci_lo_vs_symbolic`/`ci_hi_vs_symbolic`
are item (b) — context, not the gate — showing whether any audio-vs-symbolic
gap is attributable to audio provenance rather than the subset being
easier or harder.

**5e. MIDI drift** (item (b)'s companion check), against the stored Transkun
MIDIs, to confirm any audio-vs-symbolic gap is attributable to audio
provenance and not to transcription failure on this subset specifically. The
reference notes come from parsing the stored Transkun MIDI directly (the same
note-dict shape `psyllabus.notes_from_midi_bytes` returns —
`{pitch, onset, offset, velocity}`); the candidate notes come straight out of
the 5b cache (each file is `{"notes": [...], "pedals": [...]}`, written by
`_write_cache_atomic`):

```python
import json
import statistics
from pathlib import Path
from claim_measurement.difficulty.psyllabus import notes_from_midi_bytes
from claim_measurement.difficulty.realaudio_check import midi_drift

transkun_mid_dir = Path("/Users/jdhiman/Documents/crescendai/model/data/results/amt_gap_curve/transkun_mid")
audio_midi_cache = Path("/Users/jdhiman/Documents/crescendai/model/data/results/phase1_lora/audio_midi_cache")
ONSET_TOLERANCE = 0.05  # seconds; matches the test fixture's tolerance

deltas, f1s = [], []
for cache_path in sorted(audio_midi_cache.glob("*.json")):
    seg_id = cache_path.stem
    reference_notes = notes_from_midi_bytes(
        (transkun_mid_dir / f"{seg_id}.mid").read_bytes())
    candidate_notes = json.loads(cache_path.read_text())["notes"]
    drift = midi_drift(reference_notes, candidate_notes, ONSET_TOLERANCE)
    deltas.append(drift["note_count_delta"])
    f1s.append(drift["onset_f1"])

print(f"n={len(deltas)} mean note_count_delta={statistics.mean(deltas):+.1f} "
      f"mean onset_f1={statistics.mean(f1s):.3f}")
```
This is a genuine audio-provenance perturbation only if drift is measured
rather than assumed — report the aggregated `note_count_delta`/`onset_f1`
alongside the two tau-c deltas above, not as a separate afterthought.

## Deferred to after gate (i)

The design spec's gate (ii) discussion notes that features37's ridge in
Stage 4 is fit on ~720 train-fold pieces while the LoRA fine-tune trains on
~3,800 — so beating 0.8048 that way partly measures more supervision, not a
better encoder. The design spec's fix is: "if (i) passes, a matched
features37 arm refit on the same ~3,800 pieces makes (ii) honest."

No task in this plan builds that matched arm. That is deliberate, not an
oversight:

- it is conditional on gate (i) passing — there is no reason to build it if
  the fine-tune never clears Stage 4;
- it cannot be correctly specified before gate (i) produces numbers (which
  ~3,800-piece pool, which fold structure, and what to compare it against
  all depend on what Stage 4 actually measured);
- it is therefore explicitly out of scope for Phase 1's build. If gate (i)
  passes, scope and build the matched features37 arm as a follow-up task
  before treating gate (ii)'s features37 comparison as fully honest.

## If both gates pass

Report the measured deltas (not the FLOP-derived estimates) in
`docs/mirex/track-a-difficulty-prediction.md`'s decision log, per the design
spec's File Changes table. That edit is out of this plan's scope (it happens
at ship time, once real numbers exist).

## If either gate fails

Report the negative result plainly. A `noise` verdict on gate (i) or a
`ci_lo_vs_features37 <= 0` on gate (ii) is a real finding — #137's own history
is seven converging nulls; an eighth is not a failure of this plan, it is data.
