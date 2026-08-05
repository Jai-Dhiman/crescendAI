# Track A (Task 1) — Music Performance Difficulty Prediction

**Status:** PIPELINE BUILT AND MEASURED; feature frontier closed, encoder chosen (MoonBeam-839M), Phase 1 fine-tune pending · **Issue:** [#104](https://github.com/Jai-Dhiman/crescendAI/issues/104) · **Updated:** 2026-08-03 · **Deadline:** 2026-10-01

> The spec below (task, datasets, I/O contract) is current. The **"Provisional approach" section is SUPERSEDED** — the shipped pipeline is not frozen MuQ + the A1-Max head. See "Measured state" for what actually runs and what it scores.

## Measured state (read this before proposing work)

**Shipped pipeline:** audio → **Transkun** transcription → **37 hand-crafted symbolic features** (`candidate_features`, onset/pitch/velocity only) → **LightGBM regression** head. Not MuQ, not the ordinal head.

**Measurement convention — composer-disjoint 5-fold CV Kendall tau-c.** A random split lets a model memorize "Czerny pieces are grade 4" and score well without learning difficulty, so folds are `GroupKFold` over composer (1,066 composers across 5,798 pieces). Absolute tau-c also moves with the grade mix and the subset, so **compare gaps under identical folds, never levels across harnesses**.

**⚠️ The 0.824 "deployed-on-transkun" figure is CONTAMINATED — do not use it as a target.** In `stage2_refit_curve.py` the "deployed" head is fit on *all* PSyllabus records and then scored on a subset of those same records, i.e. train-on-test. No honestly cross-validated number can beat it. The valid anchor is the 37-feature arm measured under the same folds as whatever it is being compared to.

**Honest anchor (2026-08-01, #137, 5,798 Transkun MIDIs):**

| Arm | mean-fold tau-c | pooled OOF | vs anchor |
|---|---|---|---|
| 37 base features, regression | **0.7929 ± 0.0159** | 0.7966 | — (anchor) |
| 37 base + 32 Transkun-unlocked | 0.7988 ± 0.0160 | 0.8031 | **+0.0064** CI [+0.0038, +0.0090] SIG |
| 32 Transkun-unlocked only | 0.7576 ± 0.0202 | 0.7619 | −0.0347 |
| 37 base, rank-transformed target | 0.7920 ± 0.0162 | 0.7968 | +0.0002 (tie) |

**These 0.79xx levels are `tk_ablation.py`'s protocol (LightGBM + GroupKFold, n=5,798) and are NOT the #138 Phase 1 bar.** The same 37 features scored through `bakeoff_cv.py`'s folds (RidgeCV, n=900) give **0.8048** — that is what a fine-tuned encoder must clear. See the 2026-08-03 decision-log entries.

Regenerate with `--stage extract --workers 10` (~8 min) then `--stage cv --boot 2000` (~2 min). The extractor SHA is recorded in the feature cache and `--stage cv` refuses to run against a stale one, so an edited feature can never be scored through old values.

**The feature wall is real and the frontier is close to exhausted.** #137 tested the last untested feature premise — that every prior null was measured on offset-blind features — and the new offset/pedal family delivered a significant but negligible **+0.0058**. It takes only **8.6% of LightGBM gain** while `pitch_lz_complexity` alone takes **66.9%**. Hand-crafted symbolic features have hit their limit; the remaining unrefuted lever is an end-to-end encoder fine-tune (#138), whose backbone is now settled: **MoonBeam-839M**, chosen 2026-08-03 over Aria by a confound-free frozen bake-off (see the decision log).

**Transcriber constraint worth remembering: Transkun's pedal head is BINARY.** It emits CC64 as only 0 and 127, alternating. Pedal *depth* features (depth mean, value entropy, half-pedal fraction) are constant across the entire corpus and are therefore not expressible from this transcriber — recovering them needs a different model, not a different feature. Pedal *timing* (change rate, on-fraction, segment length) does survive and is the strongest single contribution of the new family.

## Why we committed to this track (thesis-aligned)
The task is **difficulty scoring of solo-piano audio** — the same modality, encoder, and ordinal-ranking machinery CrescendAI already runs. Our assets map directly:
- **MuQ (frozen, on piano)** — our production audio encoder is *already* solo-piano-oriented; here that's an advantage, not a liability.
- **Ordinal ranking head** — our deployed A1-Max head already trains BCE-pairwise + **ListMLE** ranking + CCC regression on frozen MuQ embeddings (`model/src/model_improvement/audio_encoders.py`). The task's metric is **Kendall Tau-c** (ordinal agreement) — a near-exact match to what we already optimize.
- **PercePiano + T2/T5 ordinal data** — we already curate piano ordinal/skill data.
- **Piece-ID / score library (#21, #96)** — score-side difficulty signal if a score/audio hybrid helps.

## Task spec (from the MIREX task page)
- **Goal:** predict a **single real-valued difficulty score per solo-piano recording** (ordinal pedagogical scale, hidden granularity). Each test piece has ≥2 recordings (one human performance + one synthesized rendering), scored independently, aggregated per piece.
- **Input:** **WAV, 44.1 kHz, mono or stereo** (path to a WAV file).
- **Output:** a real-valued difficulty score per recording (treated as an ordering).
- **Metric:** **Kendall's Tau-c** (official ranking). Dev-time supplementary: MSE, Accuracy±1, balanced accuracy, Spearman ρ.
- **Submission:** **Docker container** with standardized inference interface; input = WAV path, output = single score. Must finish the full test set in **24h on one GPU**.
- **Rules:** train on any data (public/proprietary/internal); **no held-out eval repertoire** used directly/indirectly for training.
- **Timeline:** opens Jul 1, closes **Oct 1**, results Oct 15, 2026.
- **Captains:** Pedro Ramoneda (Songscription) pedro@songscription.ai · Huan Zhang (Clefer) huan@clefer.com.

## Named datasets
| Dataset | What | Size / labels | Link |
|---|---|---|---|
| **PSyllabus** (Ramoneda 2025) | audio recordings, classical piano | 7,901 recordings, 11-level pedagogical grades, 13 grading systems | zenodo.org/records/14794592 |
| **CIPI** (Ramoneda 2024) | MusicXML piano pieces | 652 scores, 9-level Henle | zenodo.org/records/8037327 |
| **Mikrokosmos-difficulty** (Ramoneda 2022) | piano difficulty | (Bartók Mikrokosmos) | github.com/PRamoneda/Mikrokosmos-difficulty |

Note: **PSyllabus is audio** (matches the task's audio input) — the primary training set. CIPI is score-based (MusicXML) — useful only if we go audio+score hybrid or transcribe.

## Provisional approach — SUPERSEDED 2026-08-01 (kept for decision history)

> What follows was the 2026-07-03 plan. It was **not** what shipped: the pipeline went symbolic
> (Transkun → hand-crafted features → LightGBM), not frozen-MuQ → A1-Max head. The MuQ probe is
> one of the nulls recorded in the decision log. See "Measured state" above for current reality.

- **Spine:** frozen MuQ (piano) embeddings → our existing ordinal-ranking head, retargeted to a single difficulty scalar, trained on **PSyllabus audio**, optimized for Kendall Tau-c ordering. This is almost entirely reuse of the A1-Max stack.
- **Baseline to beat:** Ramoneda et al.'s own published difficulty models (audio + symbolic). *Research needed — find their reported Tau/accuracy.*
- **Edge hypothesis:** our piano-specialized frozen encoder + ordinal head may be genuinely competitive here. This is the transferable-asset story worth testing.

## Research checklist
- [ ] Fetch/read the PSyllabus, CIPI, Mikrokosmos papers (Ramoneda et al.) — exact schemas, label semantics, licenses, train/test protocol.
- [ ] Find the **published SOTA** difficulty-prediction numbers (Tau-c / Acc±1) and architectures — what's the bar?
- [ ] Confirm the reference Docker template / inference wrapper once released; nail the exact I/O contract.
- [ ] Determine whether audio-only (PSyllabus) or audio+score hybrid is stronger; does transcription help or hurt?
- [ ] Map our A1-Max head → single-scalar difficulty regression/ranking; what changes?
- [ ] Licenses of PSyllabus/CIPI (commercial-use fork question).
- [ ] The "human vs synthesized rendering" aggregation — does it bias toward performance-quality vs score-difficulty? (This is subtle: the task says *difficulty*, but a human *performance* encodes execution quality too.)
- [ ] Open questions for the captains (Ramoneda/Zhang).

## Must-confirm / open questions
1. Reference Docker template details (not yet released as of research date).
2. Whether difficulty is meant as **score difficulty** (composition) or **performance difficulty** (execution) — the two-recordings-per-piece design suggests they want a piece-level score robust to performer, i.e. score difficulty. Confirm.
3. PSyllabus/CIPI licenses.

## Stage 2 — Transkun difficulty-head re-fit (#135, closed — final verdict MARGINAL, see the 2026-07-31 decision-log entry)

**Why:** aria-amt was replaced repo-wide by Transkun (#128). Swapping the transcriber *alone* is tau-c-neutral because the deployed LightGBM head was trained on **clean** score note-counts and overfit the transcriber's biases. The fix is to **re-fit the head on Transkun-transcribed features**. The Stage-2 learning-curve gate confirmed this is worth it:

| eval N | arm B (transkun-trained) | deployed 7.9k-clean on transkun | B − A (matched-N) | verdict |
|---|---|---|---|---|
| 39 | — | — | +0.035 CI[−0.058,+0.129] | underpowered (looked dead) |
| 161 | — | 0.812 | +0.043 CI[+0.007,+0.081] | marginally significant |
| **604** | **0.807** | **0.804** | **+0.050 CI[+0.029,+0.071]** | **STRONG green** |

At n=604 a head trained on just 604 Transkun pieces already matches the deployed head trained on all 7,899 clean pieces (reading Transkun), and is still climbing → the full re-fit should beat deployed by ~**+0.04–0.05 tau-c** (full task).

**Harness:** `model/src/claim_measurement/transcription_bench/stage2_refit_curve.py` (stages: `prep` | `transcribe` | `pipeline` | `curve`). Data lands in `model/data/results/amt_gap_curve/` (gitignored). Imports the #104 difficulty feature code by absolute path (the `.worktrees/issue-104-mirex-difficulty` worktree must exist).

### Runbook (full 7.9k transcription)

Run the pipeline (per piece: download → Transkun → drop wav; disk-safe, resumable, skips pieces that already have a MIDI). It auto-resumes on any non-zero exit and `caffeinate` blocks system sleep — **lid open, on AC**:

```
caffeinate -is bash -c 'until uv run --script /Users/jdhiman/Documents/crescendai/.worktrees/issue-135-transkun-refit/model/src/claim_measurement/transcription_bench/stage2_refit_curve.py --stage pipeline --all --workers 4; do echo "[$(date)] stopped; resuming in 15s"; sleep 15; done'
```

Watch progress in a second terminal (climbs toward ~7,100–7,300; ~8.5% of pieces are unavailable on YouTube):

```
watch -n 30 'ls /Users/jdhiman/Documents/crescendai/model/data/results/amt_gap_curve/transkun_mid/*.mid | wc -l; df -h /Users/jdhiman/Documents/crescendai | tail -1'
```

When the count plateaus, re-run the gate curve at full N:

```
uv run --script /Users/jdhiman/Documents/crescendai/.worktrees/issue-135-transkun-refit/model/src/claim_measurement/transcription_bench/stage2_refit_curve.py --stage curve
```

**Still to build** (after the data lands): the retrain+eval stage — extract the 37 difficulty features from all Transkun MIDIs, retrain the LightGBM head (`DEPLOYED_PARAMS`) on Transkun features, CV-eval vs the clean-trained deployed head; if it wins, that head is the MIREX Track A submission.

## Decision log (append-only)
- **2026-07-03** — Track A spec captured from MIREX task page; assets mapped (strong fit). Deep research NOT yet done — next session should run the Research checklist before /brainstorm. Provisional lean: this is the higher-value track (real asset transfer vs Track B's no-moat).
- **2026-07-22 (#125 Stage-1)** — Transkun adopted over aria-amt for the symbolic stream (offset F1 0.79 vs 0.37, MIT licence). Swap measured neutral on difficulty tau-c on its own; the adopt decision stood on transcription quality, not on a difficulty win.
- **2026-07-27** — Stage-2 Transkun re-fit gate = STRONG green at n=604 (#135); full 7.9k transcription launched (user-driven, see Runbook). Earlier #104 "closed research line / feature wall ~0.76" conclusions predate the head-re-fit lever, which the n=39 probe was too underpowered to detect.
- **2026-07-31 (#135)** — Full-scale Stage-2 re-fit gate came back **MARGINAL**: re-training the head on Transkun features rather than clean MIDI gave B−A = +0.016 at n=5,798. Diagnosis: the 37 features exclude offsets *by design*, so they are transcriber-robust and clean-vs-Transkun barely differs — the gate tested the wrong lever. Motivated the #137/#138 split.
- **2026-08-01 (#137) — FEATURE FRONTIER CLOSED.** Built 32 Transkun-unlocked features (articulation via duration/IOI ratio, duration entropy/LZ, true time-weighted voicing, chord-release dispersion, pedal timing) and a composer-disjoint CV tau-c ablation with fixed folds shared by all arms and a paired bootstrap over pieces. Result: **+0.0064 tau-c** (0.7929 → 0.7988), significant but negligible; the new family is ~90% redundant with the 37 and takes 8.6% of model gain. **This was the last untested feature premise** — every prior null was measured on offset-blind features, and removing that blindness did not move the wall. Two by-products: (a) established that the **0.824 anchor is train-on-test contaminated**; (b) established that **Transkun cannot express pedal depth** (binary CC64). Harness: `model/src/claim_measurement/difficulty/{transkun_features,tk_ablation}.py`, 26 unit tests.
  - *Not refuted, still open:* the lambdarank arm scored −0.139 but used **one query group per fold**, and NDCG is top-heavy while tau-c scores the whole list — a formulation bug, not a verdict on rank-native objectives. Multi-scale / per-section aggregation of the new features remains untried.
- **2026-08-03 (#138 Phase 0) — ENCODER CHOSEN: MoonBeam-839M replaces Aria.** Frozen bake-off on the Transkun domain (not clean MIDI — avoids a #135-style domain mirage), composer-disjoint 5-fold × 5 seeds, RidgeCV, n=900 across 900 distinct composers, both arms serialized against one composer index so the folds are byte-identical. **MoonBeam-839M/mean-pool tau-c 0.8257 ± 0.0018 vs Aria/EOS 0.7790 ± 0.0030**; paired bootstrap over pieces +0.0540 [+0.0379, +0.0710].
  - *Pooling confound tested and REFUTED.* MoonBeam gains +0.046 from mean-over-tokens vs last-token, which first read as the bake-off measuring pooling rather than backbone. A control on the **same Aria checkpoint, same 300-note chunks, both poolings from one forward pass** (`eos_pool` reproducing the deployed path **bit-exactly**) showed mean pooling does **not** help Aria (−0.0079, CI straddles 0). The gain is backbone-specific; under **matched** pooling MoonBeam's lead **grows** to **+0.0619 [+0.0446, +0.0787]**, P(diff≤0)=0.000. Mechanism (plausible, unmeasured): Aria's contrastively-trained EOS position is already a pooled global vector; MoonBeam is a plain causal LM whose signal is distributed across tokens.
  - *Cost:* $0 — both backbones run on CPU locally. The "HF Jobs A100" line in the Phase 0 plan was an untested assumption.
  - *Comparability, RESOLVED 2026-08-03 (see the next entry).* 0.8257 comes from `bakeoff_cv.py` (RidgeCV + seeded folds) and is **not** comparable to this doc's 0.7929/0.7988 anchors from `tk_ablation.py` (LightGBM + GroupKFold). The 37 features have now been scored through the bake-off folds; that number, not 0.7929, is the bar Phase 1 must clear.
- **2026-08-03 (#138 step 2) — SAME-FOLDS FEATURE REFERENCE: the 37 hand features score 0.8048, so the encoder's real lead is +0.024, not +0.047.** The #137 feature vectors (read from `mirex_137_tk_features.json`, extractor fingerprint verified, so the values are byte-identical to the ones #137 measured) were written into `emb/features37/` through the same `extract_embeddings` path as the encoder arms; grades and composer ids match the Aria and MoonBeam `.npz` rows exactly, so the folds are identical by construction, not by assumption.

  | arm (n=900, identical folds) | tau-c | ± |
  |---|---|---|
  | MoonBeam-839M / mean-pool, RidgeCV | **0.8257** | 0.0018 |
  | **37 hand features, RidgeCV** | **0.8048** | 0.0008 |
  | 37 hand features, LightGBM (#137 model class) | 0.8009 | 0.0030 |
  | Aria-medium / EOS, RidgeCV | 0.7790 | 0.0030 |

  Paired bootstrap over pieces (2000 resamples, pooled OOF, seed 2026):
  - `moonbeam_mean − features37_ridge` = **+0.0244 [+0.0091, +0.0397]**, P(diff≤0)=0.002 — real, but roughly **half** the +0.047 headline against Aria.
  - `features37_lgbm − features37_ridge` = +0.0020 [−0.0075, +0.0121], noise. Model class does **not** move the feature baseline on this sample, so ~0.80 is robust and RidgeCV is not understating it.
  - `features37_ridge − aria_eos` = **+0.0298 [+0.0099, +0.0496]**, P=0.001. **The shipped frozen Aria embedding was worse than the 37 hand features on this protocol** — the Phase 0 headline partly measured how weak the incumbent was.
  - *Caveat on this sample:* 900 pieces spanning 900 **distinct** composers (one piece each), so the composer-disjoint constraint is vacuous here and these folds are effectively random splits — unlike `tk_ablation.py`'s 5,798 pieces over 1,066 composers, where the constraint bites. Trust the paired within-protocol deltas, not the levels.
  - **Phase 1 gate:** a fine-tuned MoonBeam must clear **0.8048** on these folds, not 0.7790 and not 0.7929. Beating frozen 0.8257 without clearing the feature baseline by a paired-bootstrap-significant margin is a partial result.
- **2026-08-03 — TRACK B DROPPED; Track A is the sole MIREX 2026 submission.** The parallel Task-2 (CMI-RewardBench) exploration is closed: its issues (#105, #106, #107, #122, #123, #124) were already closed, and its standalone repo, cached corpora, and R2 archive have been deleted. Rationale and the one transferable finding (CLAP window granularity, not head architecture, was the lever) are recorded in [README.md](./README.md). All remaining MIREX effort goes to #137/#138 under this doc.
- **2026-08-04 (#138 Phase 1) — DESIGN APPROVED, no implementation yet. Tracked in [#149](https://github.com/Jai-Dhiman/crescendAI/issues/149), which holds the full design.** Four things settled here are load-bearing and were measured, not assumed:
  - **Composer-disjointness is a PER-FOLD constraint, not a global one.** An earlier framing held that only 765 of the 4,898 non-eval pieces are usable, because 4,133 share a composer with some eval piece. That bound applies only if a single fine-tune must serve every fold. Excluding just the *test fold's* 180 composers leaves **4,535 / 4,802 / 5,003 / 4,748 / 4,869** training pieces per fold at seed 2026. The adopted rule (option D) additionally excludes all 900 eval pieces, leaving **3,815 / 4,082 / 4,283 / 4,028 / 4,149** — chosen so no reviewer can object that the encoder trained on the pieces the ridge head was later fit on. This moved the compute plan by an order of magnitude.
  - **The gate costs 5 fine-tunes, not 25.** `features37_compare.py`'s `paired_boot` resamples the pooled OOF at `SEEDS[0]=2026` only; the 5-seed ± is display, not the verdict. A set of per-fold fine-tunes is *welded to one seed* — seed 2027's test fold contains composers that seed 2026's adapters trained on — so every extra seed costs a fresh set of five. Plan is staged: pilot → 5 (gate) → 20 more only if the gate passes.
  - **Compute priced against the real CLI, not an assumption.** The Phase 0 log flagged "HF Jobs A100" as an untested assumption; `hf` 1.7.1 confirms `hf jobs uv run|ps|logs|inspect|stats|cancel|hardware` with `-d/--detach`, `--timeout`, `--secrets`, `--flavor`. **`a100-large` is $2.50/hr**, so ≈$13 for the gate and ≈$63 including the ±. `h200` and `rtx-pro-6000` appear in `hf jobs hardware` but are **not** in `hf jobs uv run`'s `--flavor` enum. The ~1 GPU-hr per fold-model figure is a FLOP-based **estimate, not measured** — the pilot fold exists partly to check it.
  - **Real-audio eval is now a second gate, not a sanity floor.** Eval pieces with local audio went **83 → 709 of 900**, uniform 75–88% coverage across all 11 grades, by re-fetching from the manifest's `video_id` fields (`model/data/results/amt_gap_curve/refetch_report.json`). At n=709 the paired-difference CI half-width is ≈±0.017, enough to resolve the +0.024 margin; the n≥500 threshold separating "gate" from "floor" was fixed *before* the yield was known. The 191 failures are **yt-dlp bot-detection, not unavailable videos** — zero failures in the first 626 attempts, then 191 consecutive. The missing PSyllabus audio is **not in R2**: `mirex/track-a/amt-gap-curve/transkun-mid` is a MIDI backup, and `raw/amt_audio` / `raw/amt_recordings` are Chopin-competition (T2) audio. Do not search R2 for it again.
  - **HARNESS BUILT AND MERGED (2026-08-04) — but NOTHING IS TRAINED.** The Phase 1 code is on local main and CPU-verified by 72 tests. **No GPU has run and no money has been spent; the gate is UNMEASURED.** Operator instructions: [phase1-lora-runbook.md](./phase1-lora-runbook.md). Harness: `model/src/claim_measurement/difficulty/{fold_plan,ranking_loss,train_fold,ft_eval,push_train_dataset,realaudio_check}.py`.
  - *Anchors re-derived, not copied forward.* `features37|ridge` **0.8048 ± 0.0008** and `moonbeam_mean|ridge` **0.8257 ± 0.0018** (delta **+0.0244 [+0.0091, +0.0397]** SIG) were regenerated from scratch this session and reproduce to four decimals after 40+ intervening commits. The gate value 0.8048 is live, not historical.
  - *LoRA targeting verified against real peft 0.18.1.* `get_peft_model` mutates the outer model **in place** (`peft_model.model is base_model`), and `peft_model.model.model.layers[L].<proj>` becomes a peft LoRA `Linear`. Targets are layers **10–14 × {q,k,v,o,gate,up,down}_proj = 35 modules**; peft's **default** `target_modules` would wrongly adapt `lm_head`, `decoder_embedding`, `fc_out`, and `summary_projection` on this fork.
  - **GOTCHA — `torch.no_grad()` does NOT disable dropout.** `train_fold.py` originally extracted the graded embeddings without `eval()`, leaving `lora_dropout=0.05` active. That would have made `emb_fold{F}.npz` stochastic and biased the fine-tuned arm's tau-c **down** — a false negative on the gate that no test caught and no crash revealed.
  - **GOTCHA — a statistical fixture cannot detect "one adapter's embeddings reused for every fold."** A single matrix used consistently across a fold's train and test rows is a legitimate CV of a different experiment and scores identically (0.9986 either way). The guard must be **structural**: the committed one breaks the symmetry so only fold 0's file carries signal, making correct routing score LOW and the bug score 0.9876.
  - *Per-fold adapters are welded to ONE seed.* Seed 2027's test fold contains composers seed 2026's adapters trained on. Never re-score seed-2026 embeddings under another seed.
  - *Re-derived from live data:* option-D per-fold training pools **3,815 / 4,082 / 4,283 / 4,028 / 4,149**; **709 of 900** eval pieces have local WAVs for the real-audio second gate.
- **2026-08-05 (#149) — first three real HF Jobs launches. The gate is still UNMEASURED; no fold has trained.** The harness is on local main and 91 CPU tests are green, but every launch so far died on job-environment plumbing that no CPU test could express. Jobs: `6a738f91` ERROR, `6a73942f` ERROR, `6a73a7d7` CANCELED after ~1h.
  - **`scipy` was missing from `train_fold.py`'s `# /// script` header.** `hf jobs uv run` builds the container from that header ALONE, while the code it imports at runtime comes from the bundle's `code/` dir — so the two drift silently and only fail inside a paid container. `bakeoff_cv.py` does `from scipy import stats`; the header listed numpy and not scipy. A guard now parses the header and asserts every module-scope import of `push_train_dataset._CODE_FILES` is declared.
  - **GOTCHA — the MoonBeam fork vendors a PARTIAL transformers.** Its `__init__.py` advertises the full model zoo (`models.bloom` at line 235) but ships only `auto/bert/encoder_decoder/llama`, with no `models/__init__.py` at all. `moonbeam_extract_script.py` never noticed because it touches only `LlamaConfig`/`LlamaForCausalLM`. `peft/utils/constants.py:16` does `from transformers import BloomPreTrainedModel` — purely to feature-probe `hasattr(_convert_to_standard_cache)` — and dies. Stubbing alone is a rabbit hole: **peft 0.18 additionally needs `EncoderDecoderCache`, which the fork's ~4.41-era transformers predates.** The fix is **`peft==0.11.1` (era-matched) PLUS a `models.bloom` stub**, verified in an isolated env against the real fork. Note `model/.venv` still has peft 0.18.1, so the local suite validates wiring against a different peft than the job runs; `peft_model.model is base_model` and the `.model.model` inner path were checked on both.
  - **DEFECT, OPEN — `train_fold.py` has NO device handling.** `grep` for `cuda` / `torch.device` / `.to(` across `model/src/claim_measurement/difficulty/*.py` returns nothing. Model, score head, optimizer and tensors all stay on CPU, so `--flavor a100-large` rented a GPU to run CPU training. Job `6a73a7d7` burned ~1h producing nothing and was canceled.
  - **DEFECT, OPEN — no per-step progress logging.** The training loop's only print is `epoch N: val_ranking_tau=...`, after a full epoch (~477 steps at 3,815 pieces / micro-batch 8). A hung run and a working run are indistinguishable for the first hour, which is exactly why the CPU-training defect stayed invisible.
  - **DEFECT, OPEN — `trackio` is declared in the script header but never imported or called.** The design spec required it fatal at init (before GPU spend) and warn-and-continue mid-run, precisely so a misconfigured run dies cheap.
  - **LESSON — a CPU-only fixture structurally cannot verify a GPU requirement.** Every test injects a tiny fake model that runs fine on CPU. The build, two review loops, and post-merge verification all checked WHAT is computed (leakage, pairing, dropout, adapter routing); none checked WHERE it runs. Device placement needs an assertion on `.to()`/device calls — which IS CPU-testable — not on outputs.
  - *Still unproven:* `_real_loader`'s strict key check against the real 1.6 GB checkpoint, and LoRA injection on the actual 15-layer model (only a 2-layer toy has been exercised). `audio_emb_extract.py` (Stage 5) will hit the same peft/vendored-transformers wall; it only matters if gate (i) passes.
- **2026-08-05 (#149, later the same day) — the three job-environment defects above are FIXED in code; the gate is still UNMEASURED.** 100 CPU tests green (91 + 9 new). Nothing has trained; no GPU has run since `6a73a7d7`.
  - **`--device` defaults to `cuda` and REFUSES to fall back.** `resolve_device()` raises when `cuda` is requested and unavailable; only the explicit value `auto` may degrade. The default matters more than the flag: the runbook submit line passes no `--device`, so a default of `auto`/`cpu` would have re-created the exact silent CPU run on a rented GPU. `peft_model` and the score head are moved **before** the optimizer is constructed (AdamW allocates state lazily on each parameter's device); every input window, the grade tensor, and every extraction chunk are moved per use, and `_extract_full_piece` now returns via `.cpu()`.
  - **The guard is a call-recording assertion, not an output assertion.** On a CPU-only box `input_ids.device == cpu` is true whether or not the code moves anything — so the test asserts the moves *happen*: a `torch.Tensor` subclass records every `.to()` (7 device moves = 4 train + 1 val + 2 eval pieces), and the fake model records the device of `input_ids` and of its own parameters on every forward. Filter to args carrying a `torch.device`: LoRA casts dtype through the same `.to()` and contributes 70 `.to(torch.float32)` calls.
  - **Per-step logging with `--log-every` (default 10).** Each line carries `loss`, `s/step`, `pieces/s`, `elapsed`, `eta`, preceded at startup by the resolved device, `torch.cuda.is_available()`, the trainable-parameter count, and `steps/epoch x epochs`. Every `print` is `flush=True` — a job container's stdout is block-buffered, so an unflushed progress line is not a progress line. The runbook's first abort criterion is now "the device line says `cpu`, or `s/step` implies CPU speed", checkable in minute one.
  - **Trackio is fatal at init, warn-once mid-run.** `trackio.init` runs *before* the 1.6 GB checkpoint download and the model load (asserted by an ordering test), so a telemetry misconfiguration costs seconds of GPU time rather than minutes; `trackio.log` failures print one warning and training continues. `--trackio-space` is optional but metrics without it die with the container; `--no-trackio` exists for local runs, and the injected `trackio_init` follows the file's existing `loader_factory`/`uploader` seam.
  - *Fixture gotcha worth keeping:* `_fake_loader_factory` gives every piece of the same token length **identical** content, so val scores collide and `tau_c` returns `None` (it needs ≥3 points and a non-constant vector). A test that needs the val row emitted must offset each piece's tokens.
- **Standing implication** — with hand-crafted symbolic features exhausted, **#138 Phase 1 (LoRA fine-tune of MoonBeam-839M, pairwise ranking + ordinal aux) is the remaining unrefuted lever** before the 2026-10-01 deadline.
