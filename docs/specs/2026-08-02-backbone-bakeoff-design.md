# Frozen Backbone Bake-Off (MIREX Track A, #138 Phase 0) Design

**Goal:** Give a human a composer-disjoint, tau-c-scored comparison of frozen
Aria-medium vs. frozen MoonBeam-839M embeddings on the Transkun-domain
difficulty MIDIs, so they can pick which backbone to commit to an expensive
LoRA fine-tune (#138 Phase 1) without guessing.

**Not in scope:**
- The Phase-1 LoRA fine-tune trainer, its pairwise/ordinal loss, or any
  training loop.
- Running the bake-off itself. GPU extraction against the real 839M MoonBeam
  checkpoint and the real Aria weights is human-lit and happens after this
  harness ships.
- Re-running or re-litigating the deployed 0.824 hand-feature baseline
  (that number is train-on-test per #135/#137 and is not this harness's
  concern).
- The 37 hand-crafted `candidate_features` symbolic baseline. Phase 0 compares
  the two backbones to each other, not to the feature baseline.

## Problem

#138 (MIREX Track A difficulty) needs to pick a symbolic-MIDI backbone before
committing to a LoRA fine-tune. Two candidates exist: Aria-medium (already
integrated in this repo — `model/src/model_improvement/aria_embeddings.py`
loads real weights) and MoonBeam-839M (bigger, MIDI-native, Apache-2.0, but
**zero lines of integration exist anywhere in this repo** — no loader, no
tokenizer wiring, no measured difficulty baseline). Guessing which backbone to
spend a GPU fine-tune budget on is exactly the mistake #104's frozen-probe
program exists to avoid.

The evaluation protocol that must back this decision already exists in spirit
on an **unmerged branch**: `issue-104-mirex-difficulty`'s
`model/src/claim_measurement/difficulty/phase5b_aria_probe.py` (commit
`7976b5e6`) implements composer-disjoint grouped 5-fold RidgeCV + scipy
`tau_c(variant='c')` and used it to score frozen Aria at 0.744 against the 37
hand-crafted features. That file does not exist on `issue-138-encoder-finetune`
or on `main` — it was never merged. Its logic must be ported (not imported
cross-branch) into this branch, factored so the composer-disjoint splitter is
independently testable (the current `_folds` closure inside `_oof_tau` is not).

MoonBeam has no scaffold at all: no loader for `moonbeam_839M.pt`, no
tokenizer wiring against the `transformers_minimal` fork, no venv boundary
(the venv boundary matters because this repo has twice polluted its shared
`model/.venv` with a competing pretraining stack's deps — MuQ's ancient-numba
pin bit `#125`/`#130` — and MoonBeam's fork is exactly that kind of
dependency).

Concretely, without this harness: there is no way to run the bake-off at all
(MoonBeam), and the Aria half would have to be re-derived by hand from a
branch that is not checked out here (Aria).

## Solution (from the user's perspective)

After this ships, the human (not an agent — GPU/weights/network work is
human-lit per this repo's automation boundary) can:

1. `cd model && uv run python -m claim_measurement.difficulty.run_bakeoff --stage sample`
   to draw a composer-stratified ~800-1000-piece sample from the 5798 Transkun
   MIDIs and write it to `model/data/results/bakeoff/sample_manifest.json`.
2. Run the Aria extraction stage under the existing shared `model/.venv`
   (Aria's `aria` package + weights are already there).
3. Set up an isolated `uv`-managed Python 3.12 venv per this module's
   documented recipe, install MoonBeam's fork into *that* venv only, and run
   the MoonBeam extraction stage from inside it.
4. Run the eval stage, which reads whichever per-piece `.npz` files exist
   under `model/data/results/bakeoff/emb/{backbone}/` and prints/writes
   per-backbone tau-c mean/std over 5 seeds, so they can apply the decision
   rule: MoonBeam >= Aria within seed noise -> MoonBeam wins Phase 1, else
   Aria.

No agent runs step 2, 3, or the real invocation of step 1/4 against GPU
weights as part of *this* build — those require the 3.3GB MoonBeam checkpoint
and/or a GPU and are the human-lit "bake-off RUN" that follows this harness.
This build's job is that every module above is fully unit-tested offline,
against fakes, with zero real model weights.

## Design

### Key decisions

1. **Port, don't import, `_oof_tau`/`tau_c`.** `phase5b_aria_probe.py` lives
   only on `issue-104-mirex-difficulty`, an unrelated, unmerged branch —
   worktrees are separate checkouts and cannot import code that isn't on
   their own branch. The RidgeCV/composer-fold/tau-c logic is ported
   verbatim (same alphas grid `np.logspace(-1, 5, 25)`, same
   `StandardScaler -> RidgeCV` pipeline, same `scipy.stats.kendalltau(...,
   variant="c")`), but the private `_folds` closure is extracted into its own
   public function `composer_disjoint_folds` so it is independently testable
   per the approved TDD target ("assert no composer appears in both train and
   test of any fold"). This is the one intentional deviation from a literal
   port, and it only adds a name to what was already a self-contained block —
   no behavior changes.

2. **One `Backbone` protocol, two adapters, one fake.** Aria and MoonBeam
   have incompatible native APIs (Aria: `extract_embedding(path) -> Tensor`
   one pooling; MoonBeam: undocumented, two candidate poolings). A
   `typing.Protocol` with a single method,
   `embed(midi_path: Path) -> dict[str, np.ndarray]`, unifies them: Aria's
   adapter returns `{"embedding": vec}`, MoonBeam's returns
   `{"mean_pool": vec, "last_token": vec}`. The extraction orchestrator
   (`extract_embeddings`) and every offline test depend only on this
   protocol, never on Aria or MoonBeam internals — that is what makes the
   MoonBeam integration spike testable before its checkpoint ever loads.

3. **The `.npz` contract stores grade and composer, not just the embedding.**
   `phase5b_aria_probe.py`'s original contract kept grade/composer only in
   `manifest.json`, joined by `seg_id` at eval time. The approved design's
   TDD target explicitly requires the round trip to preserve
   embedding/grade/composer *through the npz itself*
   ("write-then-read round trip preserves embedding/grade/composer"), and
   composer must be numeric-only so `np.load` needs no `pickle=True`. This
   harness therefore keeps a small `composer_index.json` (ordered list of
   composer names) beside the `.npz` files; each `.npz` stores an integer
   `composer_id` that indexes into it. This is a deliberate, minor extension
   of the original contract, not a reinvention of it.

4. **`__file__`-anchored paths with override args, not a hardcoded absolute
   constant.** `phase5b_aria_probe.py` and `tk_ablation.py` (both on
   unmerged branches) hardcode `PRIMARY = Path("/Users/jdhiman/Documents/crescendai")`.
   That breaks under worktrees, whose `model/data/` is a separate,
   independently-populated directory (confirmed: `issue-138-encoder-finetune`'s
   `model/data/raw/psyllabus/`, `model/data/weights/`, and
   `model/data/results/amt_gap_curve/` do not exist locally — the 5798
   Transkun MIDIs and manifest currently only exist under the main
   checkout's `model/data/`). This repo's newer convention
   (`model/src/claim_measurement/score_align/align_cli.py`,
   `model/src/paths.py`) is `__file__`-anchored defaults plus CLI/function
   override args for exactly this worktree case. This harness follows the
   newer convention: `bakeoff_paths.py` defaults resolve under
   `Path(__file__).resolve().parents[3] / "data"` (i.e. this worktree's own
   `model/data/`), and `run_bakeoff.py` exposes `--data-root` so the human
   running the actual bake-off points it at the main checkout's
   `model/data/` (`/Users/jdhiman/Documents/crescendai/model/data`) where the
   real Transkun MIDIs and Aria weights live, without editing source.

5. **MoonBeam isolation via `uv run --script`, not a persistent venv the
   agent creates.** This repo's established pattern for one-off heavy deps
   (`tk_ablation.py`, `transcribe_bundles.py`) is a PEP 723 inline
   `# /// script` metadata block run via `uv run --script`, which uv resolves
   into its own cached, ephemeral environment — never the project's
   `model/.venv` that a bare `uv run` from `model/` would sync. That
   sidesteps this repo's known "`uv run --with X --python N` mutates the
   shared venv" gotcha, which is specific to the ad hoc `--with` flag, not
   `--script` mode. MoonBeam's extraction stage is written as its own
   PEP 723 script (`model/src/claim_measurement/difficulty/moonbeam_extract_script.py`,
   `requires-python = "==3.12.*"`, declaring the `transformers_minimal` fork
   and MoonBeam's tokenizer package as git dependencies) with the setup
   recipe documented in its module docstring, matching this repo's existing
   documentation-in-docstring convention (no new standalone doc file, per
   project convention against unrequested `.md` files).

### Modules

All new, under `model/src/claim_measurement/difficulty/` (this package does
not exist on `issue-138-encoder-finetune` yet and is created fresh; nothing
is copied wholesale from the unmerged `issue-104`/`issue-137` branches beyond
the ported CV/tau-c math described in decision 1).

- **`bakeoff_paths.py`**
  - Interface: `DEFAULT_DATA_ROOT: Path`; `resolve_paths(data_root: Path | None) -> BakeoffPaths` (a frozen dataclass with `manifest`, `labels`, `transkun_mid_dir`, `emb_root` fields).
  - Hides: the directory layout under `model/data/` and the worktree-vs-main-checkout override.
  - Tested through: `resolve_paths` with an explicit `data_root` override — pure function, no filesystem access required to test.

- **`bakeoff_cv.py`**
  - Interface: `tau_c(x, y) -> float | None`; `composer_disjoint_folds(composers: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]`; `oof_tau_ridge(X: np.ndarray, y: np.ndarray, composers: np.ndarray, n_folds: int, seeds: list[int]) -> dict`.
  - Hides: the RidgeCV/StandardScaler pipeline, the alpha grid, the greedy composer-bin-packing fold assignment, and the seed-repeated OOF-prediction bookkeeping.
  - Tested through: all three functions directly (public, pure/near-pure — `oof_tau_ridge` is the only one touching sklearn, and it is deterministic given a seed).

- **`bakeoff_npz.py`**
  - Interface: `write_embedding_npz(path: Path, embeddings: dict[str, np.ndarray], grade: int, composer_id: int) -> None`; `read_embedding_npz(path: Path) -> EmbeddingRecord` (a frozen dataclass with `embeddings: dict[str, np.ndarray]`, `grade: int`, `composer_id: int`).
  - Hides: the numeric-only key-prefixing scheme that lets one `.npz` hold a variable number of pooling vectors (1 for Aria, 2 for MoonBeam) plus two scalars, all `pickle=False`-loadable.
  - Tested through: write-then-read round trip.

- **`bakeoff_sampling.py`**
  - Interface: `load_bakeoff_manifest(manifest_path: Path, labels_path: Path, transkun_mid_dir: Path) -> list[ManifestEntry]` (join manifest + composer labels, filter to entries with both a composer and an on-disk Transkun MIDI); `composer_stratified_sample(entries: list[ManifestEntry], target_n: int, seed: int) -> list[ManifestEntry]`.
  - Hides: the composer-quota-with-cap sampling algorithm and the join-and-filter logic against `new_clean_data.json`.
  - Tested through: both functions with small synthetic fixtures (no real manifest/labels files needed).

- **`backbone.py`**
  - Interface: `class Backbone(Protocol): def embed(self, midi_path: Path) -> dict[str, np.ndarray]`; `class FakeBackbone` (test double, deterministic hash-based vectors).
  - Hides: nothing by design — this is the seam. Its entire job is to be a narrow, stable interface both real backbones and the fake implement identically.
  - Tested through: `FakeBackbone` is exercised by `extract.py`'s tests, not tested standalone (it has no logic worth a dedicated test beyond "produces the declared shape," folded into the extract test).

- **`extract.py`**
  - Interface: `extract_embeddings(backbone: Backbone, entries: list[ManifestEntry], out_dir: Path, composer_index_path: Path) -> ExtractionReport` (`ExtractionReport`: `ok: int`, `failed: list[str]`).
  - Hides: per-entry try/except-and-record-loudly iteration, `.npz` writing, and composer-id assignment/append to the shared `composer_index.json`.
  - Tested through: `extract_embeddings` given a `FakeBackbone` and synthetic entries — this is the TDD target "the backbone-interface mock: an extractor built on a fake backbone produces conformant .npz without any real model weights."

- **`aria_backbone.py`**
  - Interface: `class AriaBackbone(Backbone): def embed(self, midi_path: Path) -> dict[str, np.ndarray]`.
  - Hides: the call into `model_improvement.aria_embeddings.extract_embedding` and the `torch.Tensor -> np.ndarray` conversion.
  - Tested through: `embed()` with `model_improvement.aria_embeddings.extract_embedding` monkeypatched to a fake tensor — the ML weight load itself is the human-lit GPU boundary, not this adapter's job.

- **`moonbeam_backbone.py`**
  - Interface: `class MoonBeamBackbone(Backbone): def __init__(self, checkpoint_path: Path, loader=None): ...`; `def embed(self, midi_path: Path) -> dict[str, np.ndarray]` (returns `{"mean_pool": ..., "last_token": ...}`).
  - Hides: the (currently undocumented, spike-status) checkpoint/tokenizer loading behind an injectable `loader` callable, so the class is constructible and testable without the `transformers_minimal` fork installed.
  - Tested through: `embed()` with an injected fake `loader` returning fake per-token hidden states, asserting the two pooling outputs are computed correctly from those hidden states (mean-over-tokens vs. last-token slice) — this is real, non-trivial logic (the pooling math) even though the checkpoint load is faked.

- **`run_bakeoff.py`**
  - Interface: CLI, `main(argv: list[str] | None = None) -> int`, `--stage {sample,extract-aria,extract-moonbeam,eval}`, `--data-root`.
  - Hides: stage dispatch and wiring the above modules together into the four human-run stages.
  - Tested through: the CLI's stage-dispatch behavior (`main(["--stage", "sample", ...])` invokes the sampling path and exits 0) using a tmp directory and tiny fixture files — not a full bake-off run.

## Verification Architecture

- **Canonical success state:** every module above importable and its public
  function/class behavior verified by a colocated `test_*.py` (this repo's
  established pattern — see `model/src/claim_measurement/score_align/test_align_notes.py`
  — is tests living beside the module, not under `model/tests/`, run
  standalone rather than through `just test-model`'s `testpaths = ["tests"]`).
  All tests pass under the existing shared `model/.venv` with **no** MoonBeam
  or real-Aria-weights dependency.
- **Automated check:**
  `cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov`
  — every test file in the new package, one command, offline, no GPU, no
  network, no real checkpoints.
- **Harness:** buildable, and *is* the deliverable — there is no separate
  "verify the verifier" step needed beyond that pytest command, since the
  four TDD targets in the approved design map directly onto
  `test_bakeoff_cv.py`, `test_bakeoff_npz.py`, and `test_extract.py`
  (Task Group 0 below is the CV/tau-c/npz/mock-backbone slice; there is no
  separate pre-existing golden fixture to build against because this harness
  *is* the fixture-producing tool).

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/claim_measurement/difficulty/__init__.py` | package marker | New |
| `model/src/claim_measurement/difficulty/bakeoff_paths.py` | `__file__`-anchored path resolution | New |
| `model/src/claim_measurement/difficulty/bakeoff_cv.py` | ported tau-c + composer-disjoint folds + RidgeCV OOF eval | New |
| `model/src/claim_measurement/difficulty/bakeoff_npz.py` | `.npz` write/read contract | New |
| `model/src/claim_measurement/difficulty/bakeoff_sampling.py` | manifest+labels join, composer-stratified sample | New |
| `model/src/claim_measurement/difficulty/backbone.py` | `Backbone` protocol + `FakeBackbone` | New |
| `model/src/claim_measurement/difficulty/extract.py` | backbone-agnostic extraction orchestrator | New |
| `model/src/claim_measurement/difficulty/aria_backbone.py` | `AriaBackbone` adapter | New |
| `model/src/claim_measurement/difficulty/moonbeam_backbone.py` | `MoonBeamBackbone` adapter (injectable loader) | New |
| `model/src/claim_measurement/difficulty/moonbeam_extract_script.py` | PEP 723 isolated-venv MoonBeam extraction CLI (documents setup in its docstring) | New |
| `model/src/claim_measurement/difficulty/run_bakeoff.py` | CLI stage dispatch (`sample`/`extract-aria`/`extract-moonbeam`/`eval`) | New |
| `model/src/claim_measurement/difficulty/test_bakeoff_paths.py` | tests | New |
| `model/src/claim_measurement/difficulty/test_bakeoff_cv.py` | tests | New |
| `model/src/claim_measurement/difficulty/test_bakeoff_npz.py` | tests | New |
| `model/src/claim_measurement/difficulty/test_bakeoff_sampling.py` | tests | New |
| `model/src/claim_measurement/difficulty/test_extract.py` | tests | New |
| `model/src/claim_measurement/difficulty/test_aria_backbone.py` | tests | New |
| `model/src/claim_measurement/difficulty/test_moonbeam_backbone.py` | tests | New |
| `model/src/claim_measurement/difficulty/test_run_bakeoff.py` | tests | New |

## Open Questions

- Q: Should the composer-stratified sample size default to 800 or 1000 (the
  design says "~800-1000")?
  Default: `run_bakeoff.py --stage sample` defaults `--target-n 900` (the
  midpoint), overridable via flag — the human running the real bake-off can
  pick any value in range without a code change.
- Q: MoonBeam's real pooling API (does `transformers_minimal`'s forward pass
  even expose per-token hidden states the way this harness assumes?) is
  unknown until the human runs it against the real checkpoint.
  Default: `MoonBeamBackbone.embed()`'s injected `loader` contract is
  `loader(midi_path) -> np.ndarray` of shape `(seq_len, hidden_dim)` (raw
  per-token hidden states); if the real API differs, only the real
  (non-fake) `loader` implementation inside `moonbeam_extract_script.py`
  changes at run time — the pooling math and the `Backbone` interface do not.
