# 5-Adapter Pred-Mean Ensemble in the Submission Container

**Goal:** The MIREX submission container scores a WAV through all five Experiment B adapters and returns their prediction mean, instead of scoring through one adapter.
**Issue:** [#175](https://github.com/Jai-Dhiman/crescendAI/issues/175) (sub-issue of #104, S5)

**Not in scope:**
- Rebuilding the container, the WAV->float seam, the failure policy, or `predict.sh`. S1 (#166) built and merged all of these and they work.
- Reopening whether to ensemble. It is pre-registered on #168; Experiment A's post-hoc `noise` reading does not revise it.
- Reporting any tau-c from these artifacts. Train-on-test by construction.
- Changing the Dockerfile. It already `COPY model/ /app/model/` wholesale, so a new subdirectory ships without edits.
- Retraining. The five adapters exist and are verified.

## Problem

`load_scorer` (`model/src/claim_measurement/difficulty/score_wav.py:260-302`) builds exactly one embedder from `model_dir/adapter` and predicts through exactly one `RidgeHead`. `build_model_dir.py` writes exactly one `adapter/` + one `ridge_head.npz`.

Experiment B (#168) produced five all-data adapters at `Jai-D/phase1-lora-7665-alldata-s{2026..2030}`, and the pre-registered submission model is the prediction mean over five ridge heads, each fit on its own run's `emb_fold99.npz`. The measured ensemble machinery in `adapter_ensemble.py` is offline research code operating on stored embedding matrices; it has no path into the container.

So today the container can ship one-fifth of the model we committed to shipping.

## Solution (from the user's perspective)

`build_model_dir.py --from-ensemble <dir>...` assembles a model directory holding all five members. `build_context.py` stages it, the existing Dockerfile bakes it, and `predict.sh <wav>` prints a single score that is the mean of five members' predictions. A pre-existing single-adapter model directory keeps working unchanged.

## Design

### The scoring path never branches on K

A `RidgeHead` maps one embedding vector to one score. An `EnsembleHead` holds an ordered tuple of `RidgeHead` and maps a `(K, D)` matrix to one score by averaging member predictions. **K=1 is an `EnsembleHead` with one member**, so there is no single-vs-ensemble branch anywhere in `score_wav`. The only place layout is interpreted is `discover_members`.

Chosen over the alternative of a separate `score_wav_ensemble` function: two scoring paths means the single-adapter path is the tested one and the ensemble path is the one that ships, which is how a submission-day divergence happens.

### Layout: additive, not migrating

```
<model-dir>/members/<name>/adapter/        member LoRA adapter
<model-dir>/members/<name>/ridge_head.npz  member head
<model-dir>/manifest.json                  provenance
```

Member `<name>` is the **basename of the source artifact directory** (e.g. `s2026` from `.../alldata_s2026`), so provenance is readable off the layout and the sort order below is stable and meaningful. `--from-ensemble` rejects duplicate basenames rather than silently overwriting a member.

The legacy layout (`<model-dir>/adapter/` + `<model-dir>/ridge_head.npz`) still loads, as a one-member ensemble. `discover_members` resolves both and is the single tested seam that knows the difference.

Chosen over migrating everything to `members/`: the merged container, `runtime_probe.py`, and staged Hub bundles all use the legacy layout, and regressing a working submission path 26 days from the deadline to gain uniformity is a bad trade. The cost is one branch, confined to one function, covered by tests on both layouts.

### Member order is part of correctness

Float addition is not associative, so the mean over five members depends on summation order. `discover_members` returns members sorted by directory name, which makes two independent container runs reproduce scores. Determinism is a stated exit criterion of #166 and must survive this change.

### One base model, five adapters

`build_fold_embedder` (`audio_emb_extract.py:110-153`) calls `loader_factory(checkpoint, ...)`, which loads the full 839M base. Calling it five times loads 1.6GB five times, which is both a memory problem and a startup-cost problem.

`build_ensemble_embedder` loads the base **once** and attaches all K adapters through PEFT's named-adapter API (`PeftModel.from_pretrained(base, dir0, adapter_name=n0)`, then `load_adapter(dir_i, adapter_name=n_i)`), selecting per member with `set_adapter`. Five 3.2M-parameter LoRA deltas share one resident base.

### Transcription and tokenization happen once per WAV

Transcription owns the linear runtime term (0.036x realtime); the MoonBeam forward pass is the fixed term (0.19s/piece on A100). `score_wav` already transcribes once and writes one temp MIDI. `_extract_full_piece` takes **tokens**, not a path, so `build_ensemble_embedder` tokenizes once and reuses the tokens across all K members.

Expected cost of K=5: four extra forward passes at ~0.19s = **+0.76s on the fixed term only**, leaving the linear term untouched. Against the current `4.93s + 0.0360x` this predicts roughly `5.7s + 0.0360x`, still an order of magnitude inside the 24h/1-GPU clause. This is a prediction to be measured, not an assumption to ship on.

### `fallback_score` disagreement is fatal

Every member head is fit on the same 7,665 grades, so `np.median(y)` is identical across members by construction. `EnsembleHead` asserts agreement and raises on mismatch, because differing fallbacks mean adapters from different training pools were staged into one directory — a silent artifact-mixing bug that would otherwise ship as a plausible-looking average.

## Modules

**`EnsembleHead`** (`score_wav.py`)
- Interface: `from_members(heads) -> EnsembleHead`; `.predict(vectors: (K,D)) -> float`; `.fallback_score`; `.n_members`; `.n_features`
- Hides: member averaging, fallback-agreement validation, per-member dimension checks
- Tested through: `.predict` on constructed heads; no member internals touched
- Depth: DEEP

**`discover_members`** (`score_wav.py`)
- Interface: `discover_members(model_dir) -> list[MemberPaths]` (`.name`, `.adapter_dir`, `.head_path`)
- Hides: the two on-disk layouts, sorting, and the "no members found" failure
- Tested through: real temp directories in both layouts
- Depth: DEEP

**`build_ensemble_embedder`** (`audio_emb_extract.py`)
- Interface: `build_ensemble_embedder(adapter_dirs, checkpoint, repo_root, model_config, loader_factory, device, window_len) -> embed(midi_path) -> (K,D)`
- Hides: single base load, PEFT named-adapter attach/select, one tokenization shared across members, stacking
- Tested through: an injected `loader_factory` that counts base loads; asserts exactly one
- Depth: DEEP

## Verification Architecture

- **Canonical success state:** `predict.sh some.wav` inside a five-member image prints one float per WAV, equal to the mean of the five members' individual predictions on that WAV, with exactly one transcription and one tokenization per item.
- **Automated check:** `uv run python -m pytest model/src/claim_measurement/difficulty/ -q --no-cov` for everything that does not need Docker or the 1.6GB checkpoint; `CRESCENDAI_RUN_CONTAINER_TESTS=1 pytest apps/inference/mirex-difficulty/test_container.py` for the image.
- **Harness (Task Group 0):** buildable and required. A fixture builder writes a synthetic two-member model directory (small random adapters are NOT needed — the heads and layout are what the fast tests exercise; the embedder is tested with an injected loader). This gives every later task a real on-disk model dir to assert against instead of mocks.

## File Changes

| File | Change | Type |
|---|---|---|
| `model/src/claim_measurement/difficulty/score_wav.py` | `EnsembleHead`, `MemberPaths`, `discover_members`; `load_scorer` uses them | Modify |
| `model/src/claim_measurement/difficulty/test_score_wav.py` | tests for the above, both layouts | Modify |
| `model/src/claim_measurement/difficulty/audio_emb_extract.py` | `build_ensemble_embedder` | Modify |
| `model/src/claim_measurement/difficulty/test_audio_emb_extract.py` | one-base-load and shared-tokenize tests | Modify |
| `model/src/claim_measurement/difficulty/build_model_dir.py` | `--from-ensemble`, `build_ensemble_model_dir` | Modify |
| `model/src/claim_measurement/difficulty/test_build_model_dir.py` | ensemble build tests | Modify |
| `apps/inference/mirex-difficulty/build_context.py` | validate either layout | Modify |
| `apps/inference/mirex-difficulty/test_container.py` | five-member scoring test | Modify |
| `docs/mirex/track-a-difficulty-prediction.md` | record the re-measured runtime | Modify |

## Open Questions

- **Q: Does `peft==0.11.1` support `load_adapter`/`set_adapter` on the vendored MoonBeam fork's model class?** Default if it does not: fall back to K sequential `PeftModel.from_pretrained` calls against one shared base via `add_adapter`, and if that also fails, accept K base loads and record the memory cost — but only after measuring, never as the first choice. Task 3 resolves this empirically before any container work depends on it.
- **Q: Does the judged run invoke `predict.sh` per item or once with `--wav-list`?** MIREX has not said. Default: keep both shapes working, as `predict.sh` already does, and report runtime both ways as #166 did.
