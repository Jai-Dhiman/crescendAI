# MIREX Track B Submission System + Generator-Shortcut Study Design

**Goal:** An original music-preference reward model (frozen-encoder + Bradley-Terry head) we can submit to MIREX 2026 Track B by Oct 2 AOE, plus a measured LBD-quality finding on generator-identity shortcuts — replacing "we can run CMI-RM's checkpoint" with "we built and understand our own."

**Not in scope:**
- LLM-judge scoring stream (evidence ceiling ~70%, kills the budget).
- The 110k `cmi-pref-pseudo` corpus in wave 1 (budget-gated wave 2; its broken loader gets fixed only if wave 2 triggers).
- Beating 78.2% on the public CMI-Pref split as a goal (n=500 → ±3.6pp CI; statistically meaningless).
- Any crescendai product integration, deploy, or app change.
- Encoder finetuning (frozen encoders only; training code never ships in the submission).
- The LBD paper text itself (a writing task, not a build task).

## Problem

Issue #105's P0 proved the CMI-RewardBench harness is trustworthy (CMI-RM reproduced to the decimal: musicality 77.8%, compositional 79.2%), but we own no model: the 77.8% came from QMUL's 2.8GB checkpoint. Without our own system there is no submission, no LBD slot, no credential. Without a disciplined validation design (generator-holdout + eval budget), any system we do build will be silently overfit to the public 500-pair test — the known failure mode of reward-model leaderboards whose eval sets refresh (MIREX eval drops Oct 1, unseen generators are the hard axis). And without the shortcut study, the LBD story is a me-too ablation table.

## Solution (from the user's perspective)

When this is done, the user can:
1. Run `just extract encoder=clap corpus=cmi_pref` (locally overnight, or the same entrypoint on a rented GPU) and get a verified, content-hash-keyed feature store.
2. Run `just train configs/<name>.json` on the Mac in minutes and get a checkpoint + generator-holdout metrics; point /autoresearch at that loop overnight.
3. Run `just eval-ratchet` to compare against baseline and promote deliberate improvements — the chroma-eval idiom, same muscle memory.
4. Run `just probe` and get the generator-ID shortcut report (classifier accuracy, ID-vs-OOD decomposition, mitigation delta) — the LBD headline numbers.
5. Run `python main.py --path input.jsonl` on eval day and get contract-valid `{sample_id, preferred_candidate}` predictions from the frozen best checkpoint.

## Design

**Repo topology (resolved this session):** the system lives in a **nested standalone repo** at `crescendai/mirex-trackb/` — own `.git`, own private→public GitHub remote, zero history entanglement with the monorepo. Parent `.gitignore` gets one line (`mirex-trackb/`). The `guard-primary-tree-edits.py` hook was audited: it anchors at the edited file, so it judges the nested repo by the nested repo's own branch — the standard worktree discipline applies inside `mirex-trackb/` unchanged.

**The load-bearing wall is the compute cliff.** Everything upstream of encoder features is O(GPU-$); everything downstream is O(free on the Mac). So: extract features once per (encoder × corpus) into a content-hash-keyed store, and make every later question — head shape, `joint_tf_depth` ∈ {0,1} (pointwise vs CMI-RM's light cross-encoder — a hyperparameter, not an architecture fork), data mixture, sampler, mitigation strength — a minutes-long sweep over cached tensors. Wave 1 corpora: CMI-Pref train (4,027 pairs, local overnight extraction, ~9k clips ≈ 1-2 nights/encoder at the measured ~7s/clip MPS rate) + AIME-survey (15.6k pairs) + Music Arena (cloud, ~$25-40 total). Raw bulk audio never lands on the Mac (disk-full incident precedent); cloud jobs stream audio and ship home only feature tensors (a few GB).

**Nested validation is the win condition operationalized.** Rotating generator-holdout folds (entire generators held out, never random pairs) are what sweeps and /autoresearch optimize; ONE final fold (fixed generator set) plus the public 500 are never touched by any loop — the final fold is spent only on submission-decision judgments, and the public 500 sits behind a hard counter that refuses evaluation #6. Rationale: an optimization loop pointed at a holdout consumes it; judging must happen on folds the loop never saw.

**The shortcut study rides the same rails.** The generator-ID probe is a classifier trained on the same cached features; the mitigation (generator-balanced sampling and/or an ID-adversarial loss term) is a trainer config flag. No new infrastructure — `probe` and `submit` are consumers of `data`/`extract`/`train`/`evalx`, never owners.

**Error-handling stance (project preference: explicit over fallback):** unverified manifests refuse assembly naming the missing clip; pairs lacking generator metadata fail loudly naming the source corpus (a silent `generator=unknown` would poison both folds and probe); `main.py` degrades gracefully on absent optional fields (`lyrics`, `ref_audio` — the contract allows absence) but hard-fails on malformed rows, undecodable audio, or checkpoint/encoder mismatch; exact score ties emit "A" deterministically with a logged warning (CMI-RM's random tie-break is a reproducibility bug we do not copy).

**License stance:** NC/SA training data (CMI-Pref) is acceptable — this is a non-commercial research submission with disclosure, not a product. The repo's own code is Apache-2.0.

**Trade-offs chosen:** trainer is deliberately thin/sweepable (cleverness lives in configs — sweepability beats abstraction); corpus adapters are deliberately shallow translation layers (all invariants centralized in `schema.py`); MERT-330M and MuQ-MuLan extraction happen in the cloud rather than 2 weeks of Mac overnight (calendar time before the ~Jul 24 fork decision is the scarce resource, not the ~$40).

## Modules

**`schema` (DEEP)**
- Interface: `PairRecord` (frozen dataclass: `pair_id, source, prompt, lyrics, ref_audio, clip_a, clip_b, label, generator_a, generator_b, confidence, modality`), `validate_pairs(records) -> list[PairRecord]`.
- Hides: every corpus-invariant (mandatory generator metadata, label domain, hash formats, dedup on `pair_id`), so adapters stay dumb.
- Tested through: `validate_pairs` on fixture rows — accepts valid, raises `SchemaError` naming field+source on each violation class.

**`extract.store` — FeatureStore (DEEP)**
- Interface: `FeatureStore(root).put(content_hash, encoder, array)`, `.get(content_hash, encoder) -> ndarray`, `.write_manifest(encoder, corpus, hashes)`, `.verify(encoder, corpus) -> Manifest`.
- Hides: on-disk layout (`data/features/{encoder}/{hash}.npy`), manifest bookkeeping (counts, dims, hash checks), idempotent resume (skip existing hashes), refusal semantics on unverified reads.
- Tested through: put/get roundtrip; `verify` raising `ManifestError` naming the missing clip; idempotent re-put.

**`extract.encoders` (DEEP)**
- Interface: `embed(encoder_name, wav, sr) -> ndarray` + `text_embed(encoder_name, text) -> ndarray` behind a registry; `load_audio(path) -> (wav, sr)`.
- Hides: per-encoder chunking/pooling, device placement (MPS/CUDA/CPU), the torchcodec-ABI gotcha (decode via soundfile, the P0 fix), HF model loading.
- Tested through: registry rejects unknown encoder names; `load_audio` decodes a fixture wav; real-encoder paths exercised by the extraction CLI on real data, not by unit tests (no GPU in CI).

**`data` — corpus adapters (shallow BY DESIGN — justified)**
- Interface: `load_cmi_pref() / load_aime_survey() / load_music_arena() -> Iterator[PairRecord]`.
- Each is a pure translation from one HF dataset's quirks to `PairRecord`; all enforcement lives in `schema`. Depth here would mean duplicating validation three times.
- Tested through: fixture-slice invariants only (a checked-in 12-row jsonl per corpus in the corpus's native format → adapter yields schema-valid records with populated generator fields).

**`train` (thin BY DESIGN — justified)**
- Interface: `train(config: TrainConfig) -> TrainResult` (checkpoint path + metrics dict); `BTHead.score_pair(feat_a, feat_b, ctx) -> (score_a, score_b)`.
- Sweepability is the requirement — /autoresearch mutates configs, not code. Deterministic given seed. Config fields: `encoders, head_width, head_depth, joint_tf_depth (0|1), lr, epochs, mixture_weights, sampler (uniform|generator_balanced), adversarial_lambda, seed`.
- Tested through: symmetry (swapping a/b flips prediction, negates margin — kills the position-bias class); determinism (same config+seed+features → identical metrics); learns a planted signal on synthetic fixtures.

**`evalx` (DEEP)**
- Interface: `make_folds(pairs, k, final_fold_generators) -> Folds`; `score(checkpoint, fold) -> metrics`; `ratchet_check() / ratchet_promote()`; `budget.spend(reason)`.
- Hides: generator-grouped fold construction (a pair is held out if EITHER clip's generator is held out), the `baseline.json`/`last_run.json` compare-and-promote (chroma-eval idiom), the public-500 counter with persisted reasons that hard-refuses spend #6.
- Tested through: fold-purity (no held-out generator appears in any training pair); ratchet accepts improvement / rejects regression; budget raises `BudgetExhausted` on the 6th spend.

**`probe` (DEEP)**
- Interface: `generator_id_probe(features, labels) -> ProbeReport` (accuracy vs chance, per-generator confusion); `decompose(checkpoint, folds) -> {in_dist_acc, holdout_acc, gap}`.
- Hides: classifier training, the ID-vs-quality decomposition bookkeeping, report serialization.
- Tested through: on synthetic fixtures where features encode generator identity by construction, probe accuracy ≈ 1.0; where features are generator-independent, probe accuracy ≈ chance.

**`submit` — `main.py` (DEEP)**
- Interface: `python main.py --path input.jsonl` → `predictions.jsonl` rows `{sample_id, preferred_candidate: "A"|"B"}`.
- Hides: checkpoint+encoder loading with version pinning (mismatch → hard error), on-the-fly feature extraction (MPS batch ≤ 2, the P0 memory lesson), optional-field degradation, deterministic tie-break ("A" + warning), both-orderings averaging iff the frozen head has `joint_tf_depth=1`.
- Tested through: end-to-end on the P0 toy `input.jsonl` with a fixture checkpoint + stubbed features — valid output rows, deterministic across runs, hard-fails on a malformed row.

## Verification Architecture

- **Canonical success state:** the synthetic smoke pipeline — fixture mini-corpus (12 pairs, 3 fake generators, seeded feature vectors with a planted preference signal) flowing assemble → train → evalx → `main.py` — reaches holdout accuracy > 0.9 on the planted signal and emits contract-valid predictions. On real data: a trained head scores ≥ 0.72 mean accuracy on rotating generator-holdout folds (above the LLM-judge tier), verified by `just eval-ratchet`.
- **Automated check:** `uv run pytest` (all behavior tests, no GPU, no downloads) plus `just smoke` (the fixture end-to-end). Definitive, agent-runnable.
- **Harness:** buildable BEFORE the feature — Task Group 0 of the plan creates the repo scaffold + synthetic fixtures + the smoke pipeline skeleton. The P0 `toy_contract.py` input becomes the contract fixture.

## File Changes

All paths under `mirex-trackb/` are in the NEW nested repo; the two crescendai rows ride the `issue-106-mirex-trackb` branch.

| File | Change | Type |
|------|--------|------|
| `mirex-trackb/pyproject.toml`, `justfile`, `.gitignore`, `LICENSE`, `README.md` | uv project scaffold (torch, transformers, soundfile, numpy, safetensors, datasets, pytest; `--index-strategy unsafe-best-match` note for any CUDA host) | New |
| `mirex-trackb/src/trackb/schema.py` | `PairRecord` + `validate_pairs` + `SchemaError` | New |
| `mirex-trackb/src/trackb/corpora/{cmi_pref,aime_survey,music_arena}.py` | per-corpus adapters | New |
| `mirex-trackb/src/trackb/extract/{store,encoders,run}.py` | FeatureStore, encoder registry, extraction CLI | New |
| `mirex-trackb/src/trackb/train/{head,trainer}.py` | BTHead + config-driven trainer | New |
| `mirex-trackb/src/trackb/evalx/{folds,score,ratchet,budget}.py` | folds, scoring, ratchet, public-eval budget | New |
| `mirex-trackb/src/trackb/probe/{generator_id,mitigation}.py` | shortcut probe + mitigation sampler/loss | New |
| `mirex-trackb/main.py` | MIREX contract entrypoint | New |
| `mirex-trackb/tests/` + `tests/fixtures/` | behavior tests + synthetic mini-corpus + toy contract fixture | New |
| `mirex-trackb/configs/first.json` | first real training config (CLAP × CMI-Pref) | New |
| `crescendai/.gitignore` | add `mirex-trackb/` line | Modify |
| `crescendai/docs/mirex/track-b-cmi-rewardbench.md` | decision-log entry: Option C approved, repo topology, links | Modify |

## Open Questions

- **Q:** Do AIME-survey and Music Arena expose per-clip generator identity? **Default:** verify in each adapter task; a corpus that lacks it gets an explicit per-corpus decision (excluded from fold construction and treated as one pseudo-generator for training balance), recorded in the living doc — never a silent `unknown`.
- **Q:** Eval-JSONL schema (does MIREX expose `lyrics`/`ref_audio`?), efficiency ranked-vs-reported, eval-set composition. **Default:** email the captain (non-blocking); `main.py` already degrades gracefully either way.
- **Q:** Fourth (newer) encoder candidate worth a cloud slot? **Default:** skip — three encoders exhaust the published evidence; revisit only if wave-1 results leave budget AND the encoder axis dominates the ablation grid.
- **Q:** Wave-2 pseudo-110k trigger. **Default:** only if (a) wave-1 lands < 0.74 on holdout, (b) ≥ $40 budget remains, (c) the loader fix is confirmed bounded (< 1 day).
