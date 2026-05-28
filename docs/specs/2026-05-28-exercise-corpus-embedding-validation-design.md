# Exercise Corpus Embedding Validation Design

**Goal:** Prove that Aria 512-dim EOS-pooled embeddings cluster piano pedagogy primitives by source well enough (purity >= 0.70 at k=5 nearest neighbors) to justify building a retrieval matcher in a later slice.

**Not in scope:**
- Auto-tagging (technique class, dimension affinity, difficulty)
- Production Postgres schema or any migration
- Matcher or transformation service
- Briefing/memory integration
- MEI format support
- Any web scraper or automated MusicXML download
- The actual ~125-primitive production corpus run (user does this manually post-ship)

---

## Problem

The exercise-proposal molecule in `apps/api/src/harness/skills/molecules/exercise-proposal.ts` is a rule-based template engine (6 dims × 3 severities = ~18 string templates) with no real exercise corpus and no retrieval. The multi-slice rebuild plan (index at `docs/plans/exercise-system-rebuild-index.md`) begins with an offline validation that Aria embeddings are semantically meaningful for pedagogy primitives before any production infrastructure is built. Without this validation, slices B-D (schema, matcher, briefing) rest on an untested premise.

## Solution (from the user's perspective)

After running the pipeline the user has:
1. A SQLite catalog at `model/data/exercise_primitives.db` containing rows for each segmented primitive with stored 512-dim embeddings.
2. Segmented MusicXML files at `model/data/scores/exercise_primitives/` and derived MIDI at `model/data/midi/exercise_primitives/`.
3. A UMAP plot (`model/data/results/exercise_primitives_umap.png`) and a 15-row human-review artifact (`model/data/results/exercise_primitives_neighbors.json`) listing the 5 within-source nearest-neighbor pairs per source.
4. A terminal PASS/FAIL verdict with the measured k-NN source purity score against the 0.70 threshold.

The user then manually reviews the 15 pairs to judge pedagogical sensibility (acceptance gate: >= 11/15 judged sensible).

## Design

**Approach:** Five-component linear pipeline with a thin orchestrator.

```
sources.toml
    → segment.py (partitura, per-source structural logic)
    → embed.py (adapter over aria_embeddings.extract_all_embeddings)
    → catalog.py (SQLite read/write)
    → validate.py (k-NN purity, UMAP, review artifact)
    → run.py (orchestrator)
```

**Key decisions:**

1. **partitura over music21** for segmentation: ML-native note-array representation, best-in-class MIDI export, MEC-community standard, strategic alignment with CPJKU score-following lineage. music21 is not used in slice A.

2. **Reuse `aria_embeddings.extract_all_embeddings`** rather than re-implementing: the function already handles the EOS-pooled 512-dim variant and raises on per-file failures. Our primitives (tens to hundreds of notes) stay within the 300-note chunk threshold, producing a single-chunk embedding identical to PercePiano treatment.

3. **SQLite not Postgres**: zero production blast radius. Slice B will design the production schema that ingests this output.

4. **Whole-exercise granularity**: each numbered Hanon exercise / Czerny etude / Burgmüller piece = one primitive. This is the coarsest sensible unit. Finer granularity (phrase-level) is deferred.

5. **No scraper**: sources.toml is a hand-listed manifest pointing to locally acquired MusicXML files. The user acquires MusicXML from IMSLP/OpenScore manually before running the pipeline.

6. **Explicit exception handling throughout**: segment.py raises `SegmentationError` (naming source + expected boundary pattern) if structure doesn't match. embed.py and catalog.py propagate all exceptions without swallowing.

7. **Purity metric chosen over silhouette or Davies-Bouldin**: directly answers the question "do neighbors come from the same pedagogical family?" Threshold 0.70 is above the ~0.43 random floor for a 3-class problem with 60/40/25 split.

## Modules

### segment.py
- **Interface:** `segment_source(musicxml_path: Path, source_name: str, output_score_dir: Path, output_midi_dir: Path) -> list[Primitive]` where `Primitive` is a dataclass: `(primitive_id: str, source: str, source_exercise_number: int, title: str, musicxml_path: Path, midi_path: Path, n_notes: int)`.
- **Hides:** partitura parse logic, per-source structural quirk dispatch (Hanon uses part/measure structure, Czerny and Burgmüller use movement or part segmentation), MusicXML export per segment, MIDI export per segment, note-count computation, primitive_id construction.
- **Depth verdict:** DEEP — the interface is 4 arguments → list of primitives; the implementation hides ~3 source-specific parsers, format exports, and boundary detection.

### catalog.py
- **Interface:** `write_primitives(primitives: list[Primitive], embeddings: dict[str, torch.Tensor], db_path: Path) -> None` and `read_primitives(db_path: Path) -> list[CatalogRow]` where `CatalogRow` is a dataclass with all fields including `embedding: np.ndarray`.
- **Hides:** SQLite schema DDL, embedding serialization to/from binary blob (numpy `tobytes()` / `frombuffer()`), created_at timestamp injection, connection lifecycle.
- **Depth verdict:** DEEP — callers never see SQL or blob encoding; they hand in Python objects and get Python objects back.

### embed.py
- **Interface:** `embed_primitives(midi_dir: Path) -> dict[str, torch.Tensor]` — returns mapping of primitive_id stem to 512-dim tensor.
- **Hides:** import of `aria_embeddings.extract_all_embeddings`, variant pinning to `"embedding"`.
- **Depth verdict:** SHALLOW by design — this is an intentional thin boundary adapter. Its value is isolating the import path and variant selection so the rest of the pipeline never references aria_embeddings directly. Justified because the complexity truly lives in aria_embeddings.py (already DEEP and tested).

### validate.py
- **Interface:** `run_validation(db_path: Path, output_dir: Path) -> ValidationResult` where `ValidationResult` is a dataclass: `(purity: float, verdict: str, pairs: list[dict], umap_path: Path, pairs_path: Path)`.
- **Hides:** loading embeddings from catalog, k-NN computation (sklearn NearestNeighbors, k=5), purity calculation, UMAP fit + plot (matplotlib), neighbor-pair extraction (5 per source), JSON serialization, PASS/FAIL verdict against 0.70 threshold.
- **Depth verdict:** DEEP — one call produces purity score, plot file, and review artifact; callers see none of the sklearn/umap/matplotlib internals.

### run.py
- **Interface:** CLI entry point `python -m exercise_corpus.run --sources sources.toml --output-dir model/data`.
- **Hides:** nothing — thin orchestrator wiring segments → embed → catalog → validate. Intentionally shallow.
- **Depth verdict:** SHALLOW by design (orchestrator).

## Verification Architecture

- **Canonical success state:** All tests pass against committed fixture MusicXML files. The pipeline produces a catalog, UMAP plot, and review artifact from the user's real corpus and prints a PASS/FAIL verdict.
- **Automated check:** `cd model && uv run pytest tests/exercise_corpus/ -v` — must pass 100% against committed fixtures with no live network access, no Aria weights required.
- **Harness:** No Task Group 0 harness needed. Test fixtures (small MusicXML files representing 3-exercise Hanon snippets and equivalent stubs for Czerny/Burgmüller) are committed as part of the implementation tasks. Tests are self-contained, rely on no external data, and use synthetic embeddings where Aria weights would be required.

The purity >= 0.70 threshold and >= 11/15 neighbor-pair human acceptance gate are **post-ship manual acceptance criteria** applied by the user on their real ~125-primitive run. They are not automated build gates.

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/exercise_corpus/__init__.py` | Package init | New |
| `model/src/exercise_corpus/sources.toml` | Hand-listed manifest (source name, license, local MusicXML path) | New |
| `model/src/exercise_corpus/segment.py` | Deep segmentation module (partitura, per-source dispatch, MusicXML+MIDI export) | New |
| `model/src/exercise_corpus/embed.py` | Thin adapter over aria_embeddings.extract_all_embeddings | New |
| `model/src/exercise_corpus/catalog.py` | SQLite read/write with embedding blob serialization | New |
| `model/src/exercise_corpus/validate.py` | k-NN purity, UMAP plot, 15-pair review artifact | New |
| `model/src/exercise_corpus/run.py` | CLI orchestrator | New |
| `model/tests/exercise_corpus/__init__.py` | Test package init | New |
| `model/tests/exercise_corpus/fixtures/hanon_3ex.xml` | 3-exercise Hanon MusicXML fixture | New |
| `model/tests/exercise_corpus/fixtures/czerny_3ex.xml` | 3-etude Czerny MusicXML fixture | New |
| `model/tests/exercise_corpus/fixtures/burgmuller_3ex.xml` | 3-piece Burgmüller MusicXML fixture | New |
| `model/tests/exercise_corpus/test_segment.py` | segment_source behavior tests | New |
| `model/tests/exercise_corpus/test_catalog.py` | catalog read/write round-trip tests | New |
| `model/tests/exercise_corpus/test_validate.py` | purity metric and review artifact tests | New |
| `model/pyproject.toml` | Add partitura>=1.8.0 to dependencies; add exercise_corpus to hatch packages | Modify |

## Open Questions

- Q: What is the exact MusicXML structure Hanon op.599 uses for exercise boundaries — one part per exercise, or one movement, or measure-range markers? Default: assume each Hanon exercise is a separate `<part>` element; segment.py raises `SegmentationError` if this assumption fails, surfacing the real structure for the user to correct.
- Q: Should `embed.py` accept a list of MIDI paths (one per primitive) rather than a directory? Default: accept a directory matching the existing `extract_all_embeddings` interface; the orchestrator writes all primitive MIDIs to `midi/exercise_primitives/` before calling embed.
