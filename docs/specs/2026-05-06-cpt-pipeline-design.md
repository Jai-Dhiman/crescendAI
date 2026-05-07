# CPT Pipeline Design

**Goal:** Build a deterministic, re-runnable pipeline that turns the raw 9K-file `apps/evals/teacher_model/data/corpus/` directory into a private HuggingFace Hub dataset (`Jai-D/Crescendai-piano-pedagogy-cpt-v1`) ready to feed downstream SFT-data synthesis (primary) or contingent CPT (secondary).

**Not in scope:**
- Tokenization (dataset is tokenizer-neutral; trainer handles it).
- Sequence packing (online via TRL `packing=True` at training time, not in the dataset).
- SFT data synthesis (separate downstream pipeline; this dataset is its source corpus).
- Direct CPT training (contingent on Phase 0 baseline per `apps/evals/teacher_model/TRAINING_PLAN.md`).
- Topical relevance filtering (latent-knowledge bet — base 35B-A3B is expected to absorb minor topical noise; the existing `relevance_classifier.py` is for harvesting, not postprocessing).
- Fixing harvester provenance gaps (out-of-scope; this pipeline accepts the existing data shape).
- Verovio MEI / music notation handling (text corpus only).
- Multi-revision / v2 support (v2 ships as a separate Hub dataset, not as a revision of v1).

## Problem

The teacher-model finetune plan needs a clean, auditable pedagogy corpus published to a private HF dataset repo. Currently:

1. The raw corpus (`apps/evals/teacher_model/data/corpus/*.txt`) has structural noise: legal disclaimers repeated across docs, page headers/footers from PDF extraction, repeating template lines, references sections from academic papers, OCR-style mojibake, files from non-English sources or below 100 chars.
2. The existing `apps/evals/teacher_model/dedup.py` catches whole-document near-duplicates at Jaccard 0.8 but is blind to within-document repetition (a 50-page PDF with running headers) and corpus-wide line-level boilerplate.
3. There is no path from `data/corpus/` to a HuggingFace dataset — `corpus_builder.py` reports stats and orchestrates dedup but does not produce a HF artifact.
4. Provenance JSONLs (`data/provenance_*.jsonl`) do not store the corpus filename, so joining a `.txt` file to its source provenance row is non-trivial and harvester-specific.

Without this pipeline, the teacher-finetune workstream cannot produce the dataset deliverable required by `TRAINING_PLAN.md`.

## Solution (from the user's perspective)

A single CLI command:

```
uv run python -m teacher_model.cpt_pipeline run
```

runs all 5 stages end-to-end against the raw corpus and pushes a private dataset to HuggingFace Hub. Output is auditable: each stage writes a manifest + JSON report to `apps/evals/teacher_model/data/cpt_pipeline/<stage>/`. Drop reasons are logged per-doc to `drops.jsonl`. Stratified `train` / `validation` splits are produced. The HF dataset card is auto-generated from the stage reports.

Restart and audit affordances:

```
uv run python -m teacher_model.cpt_pipeline run --from-stage 3   # restart from dedup
uv run python -m teacher_model.cpt_pipeline run --only-stage 5   # republish only
uv run python -m teacher_model.cpt_pipeline run --audit          # print sanity samples
```

The operator runs `--audit` and eyeballs ten random `drops.jsonl` rows per drop reason, top-50 corpus-wide line frequencies, and per-source split sizes before treating the run as done.

## Design

### Approach

A **5-stage linear DAG**. Each stage is a pure function: `stage_in_path -> stage_out_path`. No cycles, no shared mutable state. Each stage writes its output to disk so any later stage can be re-run from its predecessor's artifact. Per-stage manifests cost ~3× disk (~1.8 GB total intermediate state at 100M words) but buy debuggability, restart-from-failure, and audit trails worth far more than the disk cost for a single-run pipeline.

```
[corpus/*.txt + provenance_*.jsonl]
        |
   stage 1: ingest          -> 1_ingest/manifest.jsonl + ingest_report.json
        |
   stage 2: structural_filter -> 2_filter/manifest.jsonl + drops.jsonl + filter_report.json
        |
   stage 3: dedup           -> 3_dedup/manifest.jsonl + dedup_report.json + line_freq_global.tsv
        |
   stage 4: split           -> 4_split/{train,validation}.jsonl + split_report.json
        |
   stage 5: hf_publish      -> 5_publish/{train,validation}/*.parquet + README.md + publish.log
```

### Key decisions and trade-offs

**Source resolution by filename pattern, with provenance JSONL join as enrichment.** Provenance JSONLs lack a clean filename field. Coarse source is derived deterministically from filename (`pdf_*` → `academic_pdf`, `web_*` → `web_scrape`, 11-char alnum → `youtube`, else `unknown`). Fine source is best-effort via URL-hash and video-id matching against per-harvester JSONLs. Combined value (`"academic_pdf:openalex"`) is stored as the `source` field. Trade-off: simpler than retrofitting all harvesters with filename emission; accepts that some files end up with `:unknown` fine-source (still typed coarsely).

**No topical filter.** Latent-knowledge bet: base Qwen3.6-35B-A3B has absorbed enough music-domain text that minor topical noise (saxophone/drums papers leaking through OpenAlex relevance threshold) is negligible. Trade-off: ships with ~5% off-topic noise; saves a brittle classifier-tuning loop.

**Three-pass dedup (B in brainstorm options).** Stage 3a wraps existing `teacher_model/dedup.py:find_duplicates()` for whole-doc MinHash@0.8. Stage 3b strips lines repeating ≥3× *within* a doc (after normalization). Stage 3c drops lines appearing in >20 distinct docs corpus-wide. Trade-off vs. paragraph-level fuzzy MinHash (Option C): simpler, faster, explainable; misses paraphrased section-level overlaps but the brainstorm-time corpus sample showed dominant boilerplate is exact-match.

**References-only section stripping (B in brainstorm options).** Source-conditional: PDF-derived sources (`academic_pdf:*`) are truncated at the last `^(References|Bibliography|Works Cited|REFERENCES)\s*$` match. YouTube and web sources are left alone. Trade-off vs. aggressive academic cleanup (Option C): preserves abstracts and figure captions which often contain pedagogically dense content; refs are the worst offender by far.

**Online packing, raw text schema, seq_len 4096 (Option A).** Final HF dataset is `{text: str, source: str, doc_id: str}`. Tokenizer-agnostic. Trainer applies `packing=True` and chooses seq_len. Trade-off vs. offline pre-packing: trainer recomputes packing per run (~10 min on 100M tokens, negligible); buys tokenizer-version flexibility, critical because Qwen3.6 tokenizer is brand new (April 2026, may receive patches).

**1% stratified-by-source held-out validation (Option C).** Sampled per source so validation has the same source mix as training, with sources of <100 docs sent fully to train (no stratification). Fixed seed for byte-deterministic split. Trade-off vs. random 1%: 5 lines more code, materially better signal on heterogeneous corpus where YouTube colloquial register differs from OpenAlex prose.

**Per-stage manifest artifacts on disk.** Each stage writes `manifest.jsonl` + a JSON `report` + (for stages 2-3) audit sidecars. Trade-off vs. in-memory pipelining: ~1.8 GB intermediate disk; restartable from any stage; auditable drop reasons; clean test boundaries (input is a file format, not a Python object graph).

**Explicit error policy: `drops.jsonl` for known per-doc failures, fail-loud for everything else.** Per-doc decode errors and filter-rejected docs are logged to `drops.jsonl` and the pipeline continues. Configuration errors, library import errors, regex bugs, disk-full, network failures all crash with a named exception. Trade-off vs. defensive try/except everywhere: bugs surface at the stage that broke them; clean runs produce clean datasets.

**Sub-package layout (`teacher_model.cpt_pipeline.*`).** New code lives under `apps/evals/teacher_model/cpt_pipeline/`, separate from existing flat `teacher_model/*.py` modules. Trade-off vs. flat layout: 5 modules + driver + tests warrant cohesion; existing `teacher_model.dedup` is reused by import, not modified.

## Modules

### `source_resolver` (DEEP)
- **Interface:** `resolve_source(filename: str, provenance_index: dict) -> str` ; `build_provenance_index(provenance_dir: Path) -> dict[str, str]`
- **Hides:** filename pattern matching (`pdf_/web_/youtube/unknown` coarse classification), URL hashing (`sha256(url)[:12]`), YouTube video-id extraction from URL (`?v=` and `youtu.be/`), per-harvester JSONL walking, conflict resolution when multiple JSONLs claim the same hash.
- **Tested through:** `resolve_source` and `build_provenance_index` only.

### `ingest` (DEEP)
- **Interface:** `run_ingest(corpus_dir: Path, provenance_dir: Path, out_dir: Path) -> Path`
- **Hides:** walking corpus `.txt` files, calling `source_resolver.build_provenance_index` once, per-doc encoding fix via `ftfy.fix_text`, whitespace normalization (collapse runs, strip BOM), word counting via `len(text.split())`, drop-on-decode-error to `drops.jsonl`, doc_id derivation (filename stem), unified manifest emission.
- **Tested through:** `run_ingest` only.

### `structural_filter` (DEEP)
- **Interface:** `run_filter(manifest_in: Path, out_dir: Path) -> Path`
- **Hides:** four sequential filter passes (min_chars=100, non_ascii_ratio<0.5, repeated_char_ratio<0.3, language=en via `langdetect.detect`), source-conditional refs stripping (regex truncation at last `^(References|Bibliography|Works Cited|REFERENCES)\s*$` line, only when `source` starts with `academic_pdf:`), per-drop reason logging, filter report aggregation.
- **Tested through:** `run_filter` only.

### `dedup` (DEEP)
- **Interface:** `run_dedup(manifest_in: Path, out_dir: Path) -> Path`
- **Hides:** stage 3a invocation of existing `teacher_model.dedup.find_duplicates` for whole-doc MinHash@0.8, stage 3b within-doc line-frequency strip (line normalization: lowercase + collapse whitespace + strip page numbers via `re.sub(r"^\s*\d+\s*$", "", line)`, then drop lines repeating ≥3× per doc), stage 3c corpus-wide line-frequency strip (count normalized lines across all surviving docs; drop lines appearing in >20 distinct docs), report aggregation including sample stripped lines and corpus-wide top-50 frequencies dumped to `line_freq_global.tsv`.
- **Tested through:** `run_dedup` only.

### `split` (justified-shallow)
- **Interface:** `run_split(manifest_in: Path, out_dir: Path, seed: int = 42) -> tuple[Path, Path]`
- **Hides:** group-by-source, deterministic 1% sampling per source (Python `random.Random(seed)` per group, `min(1, len(group)//100)` validation count for groups ≥100, zero validation for groups <100 — all docs route to train), train/validation manifest emission, `split_report.json` with per-source split sizes.
- **Justification for shallow:** ~30 lines, but isolating the deterministic-seed split in its own module makes the byte-determinism contract testable in isolation. Folding into stage 5 would couple test concerns.
- **Tested through:** `run_split` only.

### `hf_publish` (DEEP)
- **Interface:** `run_publish(train_manifest: Path, val_manifest: Path, repo_id: str, private: bool = True) -> str` (returns Hub URL)
- **Hides:** building `datasets.DatasetDict({train, validation})` with explicit `Features({text: Value("string"), source: Value("string"), doc_id: Value("string")})`, parquet sharding (256 MB target shards), dataset card generation pulling counts from each stage's `*_report.json`, decision-log appendix from this spec, `HF_TOKEN` environment lookup with explicit `RuntimeError` on missing, `huggingface_hub.HfApi.create_repo(repo_id, private=private, repo_type="dataset", exist_ok=True)`, `push_to_hub(repo_id, private=private)`, publish receipt to `publish.log`.
- **Tested through:** `run_publish` only (Hub network mocked).

### `pipeline` (DEEP)
- **Interface:** `run_pipeline(argv: list[str] | None = None) -> int` (returns exit code)
- **Hides:** argparse subcommands (`run`, `--from-stage N`, `--only-stage N`, `--audit`), pre-flight `validate_config()` (corpus dir exists, provenance dir exists, `HF_TOKEN` set when stage 5 will run), stage dispatch wiring (calls `run_ingest` → `run_filter` → `run_dedup` → `run_split` → `run_publish` in order, each reading the previous stage's output path), tqdm progress integration, stage-banner printing, audit mode sample printing.
- **Tested through:** `run_pipeline` only (E2E test against `tiny_corpus` fixture).

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/evals/teacher_model/cpt_pipeline/__init__.py` | Package marker, re-export `run_pipeline` | New |
| `apps/evals/teacher_model/cpt_pipeline/source_resolver.py` | Source resolution module | New |
| `apps/evals/teacher_model/cpt_pipeline/ingest.py` | Stage 1 module | New |
| `apps/evals/teacher_model/cpt_pipeline/structural_filter.py` | Stage 2 module | New |
| `apps/evals/teacher_model/cpt_pipeline/dedup.py` | Stage 3 module (orchestrator over existing `teacher_model.dedup`) | New |
| `apps/evals/teacher_model/cpt_pipeline/split.py` | Stage 4 module | New |
| `apps/evals/teacher_model/cpt_pipeline/hf_publish.py` | Stage 5 module | New |
| `apps/evals/teacher_model/cpt_pipeline/pipeline.py` | CLI driver | New |
| `apps/evals/teacher_model/cpt_pipeline/tests/__init__.py` | Test package marker | New |
| `apps/evals/teacher_model/cpt_pipeline/tests/conftest.py` | `tiny_corpus` fixture (~30 hand-crafted .txt files + matching provenance JSONLs covering every observable behavior) | New |
| `apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py` | source_resolver behavior tests (5) | New |
| `apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py` | ingest behavior tests (5) | New |
| `apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py` | structural_filter behavior tests (5) | New |
| `apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py` | dedup behavior tests (5) | New |
| `apps/evals/teacher_model/cpt_pipeline/tests/test_split.py` | split behavior tests (4) | New |
| `apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py` | hf_publish behavior tests (5) | New |
| `apps/evals/teacher_model/cpt_pipeline/tests/test_pipeline.py` | E2E integration test (1) | New |
| `apps/evals/pyproject.toml` | Add `ftfy>=6.2.0`, `langdetect>=1.0.9`, `huggingface_hub>=0.27.0`, `datasets>=3.0.0` to `[project.dependencies]` | Modify |

## Open Questions

- **Q:** Will the `langdetect` library produce stable results on short docs (~100-300 chars)?  **Default:** Accept whatever `langdetect.detect()` returns; if Phase-0 audit shows >5% English docs misclassified as non-English on the real corpus, raise the per-doc minimum-char floor for the language gate from 100 to 300 (filter still uses 100 for everything else).
- **Q:** Does `Jai-D/Crescendai-piano-pedagogy-cpt-v1` exist on HF as a dataset repo?  **Default:** `hf_publish` calls `create_repo(exist_ok=True, repo_type="dataset")` so it works whether or not the repo exists. Operator verifies post-push.
- **Q:** Should the dataset card include a license field given mixed-source provenance (YouTube transcripts + OpenAlex PDFs + scraped web)?  **Default:** Card states "private dataset, internal use only, mixed provenance, see `provenance_*.jsonl` for per-source license claims" — defers the legal question to operator review before any public release.
- **Q:** When the corpus-wide line-frequency threshold N=20 strips a line that appears in 21 docs but is legitimately repeated content (e.g., common scale name "C major"), do we lose signal?  **Default:** Stage 3 normalizer requires line length ≥30 chars before considering it for stripping, so short identifiers like "C major" are preserved. Audit step prints top-50 stripped lines for operator sanity check.
