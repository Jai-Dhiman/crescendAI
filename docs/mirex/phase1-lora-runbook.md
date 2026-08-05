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
W=model/data/weights/moonbeam
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
cd model/src/claim_measurement/difficulty && uv run --no-project --script \
    features37_compare.py --data-root /Users/jdhiman/Documents/crescendai/model/data
```

Expected: `features37|ridge` tau-c `0.8048`, `moonbeam_mean|ridge` tau-c `0.8257`
(the Phase 0 numbers this whole phase is measured against — Task Group 0's
harness). If either number has drifted, STOP: something about the data or the
fold seed has changed and the gate threshold below is no longer valid.

## Stage 1 — stage and upload the training bundle (once)

```bash
cd model/src/claim_measurement/difficulty && uv run python -m \
    claim_measurement.difficulty.push_train_dataset \
    --manifest /path/to/model/data/results/amt_gap_curve/manifest.json \
    --labels /path/to/model/data/raw/psyllabus/new_clean_data.json \
    --sample-manifest /path/to/model/data/results/bakeoff/sample_manifest.json \
    --midi-dir /path/to/model/data/results/amt_gap_curve/transkun_mid \
    --repo-snapshot-dir /path/to/model/data/weights/moonbeam/repo \
    --staging-dir /path/to/staging/phase1-lora-bundle \
    --repo-id <your-hf-username>/phase1-lora-bundle
```

`--n-folds` (default 5), `--seed` (default 2026), and `--val-frac` (default
0.12) all have defaults matching the design — pass them explicitly only if
deviating.

Read the printed `staged N MIDIs, 5 fold plans, ...` report before it uploads.
Abort criterion: if `n_midis` is far from the expected ~4000-4300 per fold
(sum across all 5 folds' train+val, deduplicated union will be close to the
full 5798-piece pool minus per-fold exclusions), STOP and re-check the sample
manifest and labels join.

## Stage 2 — the pilot fold

```bash
hf jobs uv run --flavor a100-large --timeout 3h \
    model/src/claim_measurement/difficulty/train_fold.py \
    --fold 0 \
    --checkpoint /path/to/moonbeam_839M.pt \
    --repo-root /path/to/moonbeam/repo \
    --model-config /path/to/moonbeam/repo/src/llama_recipes/configs/model_config.json \
    --fold-plan /path/to/staging/phase1-lora-bundle/fold_plans.json \
    --pool-grades /path/to/staging/phase1-lora-bundle/grades.json \
    --eval-manifest /path/to/eval_manifest.json \
    --midi-dir /path/to/staging/phase1-lora-bundle/midi \
    --out-dir /path/to/fold_embeddings/fold0 \
    --micro-batch 8
```

Monitor with `hf jobs ps`, `hf jobs logs <job-id>`, `hf jobs inspect <job-id>`.
Abort criteria (design spec's Open Questions):
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

Repeat Stage 2 with `--fold 1`, `--fold 2`, `--fold 3`, `--fold 4`, same
`--fold-plan`/`--pool-grades`/`--eval-manifest`, different `--out-dir` per fold
(e.g. `fold_embeddings/fold{N}`).

## Stage 4 — gate (i): encoder-as-feature-extractor (local, CPU, free)

```bash
cd model && uv run python -m claim_measurement.difficulty.ft_eval \
    --data-root /path/to/model/data --fold-emb-dir /path/to/fold_embeddings
```
Expected output: `moonbeam_ft_mean|ridge - features37|ridge: +0.0XXX
CI95[+a,+b] P(diff<=0)=p SIG|noise`. **The gate passes only if `a > 0`
(`SIG`).** If `noise`, STOP — do not proceed to the real-audio gate or report
an end-to-end number; the fine-tune did not clear 0.8048.

## Stage 5 — gate (ii): real-audio second gate (local, resumable)

Transcribe the 709 available WAVs (resumable — safe to interrupt and re-run):

```bash
cd model/src/claim_measurement/difficulty && uv run python -m \
    claim_measurement.difficulty.realaudio_check \
    --wav-manifest /path/to/audio_wav_manifest.json \
    --out-dir /path/to/audio_midi_cache
```

Extract MoonBeam embeddings for each transcribed piece using ITS OWN fold's
saved adapter (a `moonbeam_extract_script.py`-style run per fold, pointed at
`--repo-root`/`--model-config` as before and the fold's `adapter/` directory
loaded via `peft`'s `PeftModel.from_pretrained`), writing one `.npz` per piece
into `audio_emb/` via the standard `bakeoff_npz.write_embedding_npz` contract
(key `"mean_pool"`). This step is a GPU-optional but compute-bearing step
outside `realaudio_check.py`'s tested scope; wire it as a short local script.

Then compute the real-audio gate — audio vs. features37 on the SAME subset,
scored through the SAME composer-disjoint folds/seed — plus the matched
symbolic comparison that makes it interpretable:

```python
import json
import numpy as np
from pathlib import Path
from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
from claim_measurement.difficulty.ft_eval import _load_features37
from claim_measurement.difficulty.train_fold import read_fold_embeddings
from claim_measurement.difficulty.realaudio_check import score_audio_subset
from claim_measurement.difficulty.bakeoff_paths import resolve_paths

emb_root = resolve_paths(Path("/path/to/model/data")).emb_root
Xf, y, composers, seg_ids = _load_features37(emb_root)
emb_by_fold = {f: read_fold_embeddings(f"/path/to/fold_embeddings/fold{f}/emb_fold{f}.npz")["embeddings"]
               for f in range(5)}
audio_dir = Path("/path/to/audio_emb")
audio_embeddings = {p.stem: read_embedding_npz(p).embeddings["mean_pool"]
                    for p in sorted(audio_dir.glob("*.npz"))}

result = score_audio_subset(emb_by_fold, audio_embeddings, Xf, y, composers, seg_ids,
                             n_folds=5, seed=2026)
print(result)
```
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

Also compute MIDI drift per piece (item (b)'s companion check) against the
stored Transkun MIDIs, to confirm any audio-vs-symbolic gap is attributable to
audio provenance and not to transcription failure on this subset specifically.
The reference notes come from parsing the stored Transkun MIDI directly (the
same note-dict shape `psyllabus.notes_from_midi_bytes` returns —
`{pitch, onset, offset, velocity}`); the candidate notes come straight out of
the transcription cache Stage 5's `realaudio_check.main` wrote to
`--out-dir` (each file is `{"notes": [...], "pedals": [...]}`, written by
`_write_cache_atomic`):

```python
import json
import statistics
from pathlib import Path
from claim_measurement.difficulty.psyllabus import notes_from_midi_bytes
from claim_measurement.difficulty.realaudio_check import midi_drift

transkun_mid_dir = Path("/path/to/model/data/results/amt_gap_curve/transkun_mid")
audio_midi_cache = Path("/path/to/audio_midi_cache")
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
