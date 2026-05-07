# CPT Pipeline Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Build a deterministic, re-runnable pipeline that turns the raw 9K-file `apps/evals/teacher_model/data/corpus/` directory into a private HuggingFace Hub dataset (`Jai-D/Crescendai-piano-pedagogy-cpt-v1`).
**Spec:** docs/specs/2026-05-06-cpt-pipeline-design.md
**Style:** Python, uv for deps, pytest, explicit exception handling, no fallbacks. Match existing patterns in `apps/evals/teacher_model/`.

---

## Task Groups

- **Group 0 (sequential, must complete first):** Task 1 — setup
- **Group 1 (parallel after Group 0):** Tasks 2-6 (`source_resolver`), Tasks 7-11 (`structural_filter`), Tasks 12-16 (`dedup`), Tasks 17-20 (`split`), Tasks 21-25 (`hf_publish`) — five independent tracks; tasks within a track are sequential, tracks run in parallel
- **Group 2 (sequential, depends on `source_resolver` track complete):** Tasks 26-30 (`ingest`)
- **Group 3 (sequential, depends on all):** Task 31 (`pipeline` driver + E2E)

---

## Test command convention

All tests run from `apps/evals/` working directory:
```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_<module>.py::<test_name> -xvs
```

All commits use the format `feat(cpt-pipeline): <description>` matching repo conventions.

---

## Task 1: Setup — package scaffolding, deps, fixture

**Group:** 0 (must complete before any other task)

**Behavior being verified:** the `cpt_pipeline` package imports cleanly and the `tiny_corpus` fixture builds a directory with the expected file count.
**Interface under test:** package import + pytest fixture instantiation.

**Files:**
- Create: `apps/evals/teacher_model/cpt_pipeline/__init__.py`
- Create: `apps/evals/teacher_model/cpt_pipeline/tests/__init__.py`
- Create: `apps/evals/teacher_model/cpt_pipeline/tests/conftest.py`
- Create: `apps/evals/teacher_model/cpt_pipeline/tests/test_setup.py`
- Modify: `apps/evals/pyproject.toml`

- [ ] **Step 1: Write the failing test**

`apps/evals/teacher_model/cpt_pipeline/tests/test_setup.py`:
```python
"""Smoke test: package imports and tiny_corpus fixture builds correctly."""
from pathlib import Path


def test_package_imports():
    import teacher_model.cpt_pipeline  # noqa: F401


def test_tiny_corpus_fixture_has_expected_files(tiny_corpus):
    corpus_dir, provenance_dir = tiny_corpus
    txt_files = sorted(Path(corpus_dir).glob("*.txt"))
    jsonl_files = sorted(Path(provenance_dir).glob("provenance_*.jsonl"))
    assert len(txt_files) >= 14, f"expected >=14 fixture .txt files, got {len(txt_files)}"
    assert len(jsonl_files) >= 3, f"expected >=3 fixture provenance JSONLs, got {len(jsonl_files)}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_setup.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.cpt_pipeline'`

- [ ] **Step 3: Implement the minimum to make the test pass**

`apps/evals/teacher_model/cpt_pipeline/__init__.py`:
```python
"""CPT corpus preprocessing pipeline."""
```

`apps/evals/teacher_model/cpt_pipeline/tests/__init__.py`:
```python
```

`apps/evals/teacher_model/cpt_pipeline/tests/conftest.py`:
```python
"""Shared fixtures for cpt_pipeline tests."""
import hashlib
import json
from pathlib import Path

import pytest


def _sha12(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


@pytest.fixture
def tiny_corpus(tmp_path):
    """Hand-crafted 14-file corpus + 3 provenance JSONLs covering every observable behavior.

    Returns (corpus_dir, provenance_dir) as Paths.
    """
    corpus_dir = tmp_path / "corpus"
    provenance_dir = tmp_path / "provenance"
    corpus_dir.mkdir()
    provenance_dir.mkdir()

    # --- Normal docs from three different sources ---
    yt_id_1 = "abcdefghijk"
    (corpus_dir / f"{yt_id_1}.txt").write_text(
        "Practicing scales builds finger independence.\n\n"
        "Start slow with a metronome at 60 bpm. Increase by 4 bpm only after\n"
        "you can play four consecutive perfect repetitions. The Russian school\n"
        "emphasizes weight transfer through the keys, while the Taubman approach\n"
        "focuses on rotation and alignment of the forearm.\n",
        encoding="utf-8",
    )

    yt_id_2 = "lmnopqrstuv"
    (corpus_dir / f"{yt_id_2}.txt").write_text(
        "Voicing the melody in a Chopin nocturne requires careful balance.\n\n"
        "The right hand top voice should sing above the inner accompaniment\n"
        "voices. Practice each voice separately, then layer them with awareness\n"
        "of dynamic hierarchy.\n",
        encoding="utf-8",
    )

    pdf_url_1 = "https://openalex.example.org/W123/paper.pdf"
    pdf_h_1 = _sha12(pdf_url_1)
    (corpus_dir / f"pdf_{pdf_h_1}.txt").write_text(
        "Pedagogical approaches to early-stage technique acquisition.\n\n"
        "This paper examines how teachers structure the first six months of\n"
        "instruction, with particular attention to posture, hand position,\n"
        "and tone production. The authors interviewed forty conservatory\n"
        "professors across three continents.\n\n"
        "References\n"
        "Smith, J. (2019). Foundations of piano pedagogy. Journal of Piano X.\n"
        "Lee, A. (2020). Tone production studies. Pedagogy Quarterly.\n",
        encoding="utf-8",
    )

    # --- Doc with legal disclaimer boilerplate (corpus-wide line-freq target) ---
    web_url_disc = "https://music.example.org/disclaimer/article-1"
    web_h_disc = _sha12(web_url_disc)
    boilerplate_line = (
        "The author and publisher disclaim all such representations and warranties for a particular purpose."
    )
    (corpus_dir / f"web_{web_h_disc}.txt").write_text(
        f"Real content about phrasing.\n\n"
        f"{boilerplate_line}\n"
        f"Phrasing means shaping a musical line so its peaks and resolutions\n"
        f"feel inevitable to the listener.\n",
        encoding="utf-8",
    )

    # 24 more web docs containing the same boilerplate line — pushes >20 threshold
    for i in range(24):
        url = f"https://music.example.org/disclaimer/article-{i + 2}"
        h = _sha12(url)
        (corpus_dir / f"web_{h}.txt").write_text(
            f"Doc {i} body content here. Talking about ornamentation in baroque music\n"
            f"requires understanding the conventions of the period.\n\n"
            f"{boilerplate_line}\n"
            f"More body content about appoggiaturas and trills.\n",
            encoding="utf-8",
        )

    # --- Doc with within-doc repeated lines (within-doc strip target) ---
    yt_id_repeats = "wxyz12345AB"
    repeat_block = "Newly formed bands\nAlbums\nDisbandments\nEvents\n"
    (corpus_dir / f"{yt_id_repeats}.txt").write_text(
        "Year-by-year history of music ensembles.\n\n"
        f"1972\n{repeat_block}1973\n{repeat_block}1974\n{repeat_block}"
        f"1975\n{repeat_block}1976\n{repeat_block}\n"
        "End of the historical survey section.\n",
        encoding="utf-8",
    )

    # --- Two academic-paper docs with References section ---
    pdf_url_refs = "https://openalex.example.org/W456/paper.pdf"
    pdf_h_refs = _sha12(pdf_url_refs)
    (corpus_dir / f"pdf_{pdf_h_refs}.txt").write_text(
        "Body content about pedaling techniques in Debussy.\n\n"
        "The half-pedal allows partial damper engagement, useful for impressionist textures.\n\n"
        "References\n"
        "Debussy, C. (1905). Estampes. Durand.\n"
        "Howat, R. (1983). Debussy in proportion. Cambridge University Press.\n",
        encoding="utf-8",
    )

    # --- Two near-duplicate docs (Jaccard ~0.85) ---
    yt_id_dup_a = "DUP1234567x"
    yt_id_dup_b = "DUP1234567y"
    dup_text_base = (
        "Slow practice is the foundation of all technique work. Begin every\n"
        "session with five minutes of metronome scales at quarter = 60. The\n"
        "objective is not speed but evenness, articulation, and tonal control.\n"
        "Listen for unevenness in the weaker fingers and isolate problem groups.\n"
    )
    (corpus_dir / f"{yt_id_dup_a}.txt").write_text(dup_text_base, encoding="utf-8")
    (corpus_dir / f"{yt_id_dup_b}.txt").write_text(
        dup_text_base + "Additional minor tail content here.\n", encoding="utf-8"
    )

    # --- Doc <100 chars (length-floor drop) ---
    yt_id_short = "SHORTABCDEF"
    (corpus_dir / f"{yt_id_short}.txt").write_text("Too short.", encoding="utf-8")

    # --- Doc with 60% non-ASCII ---
    yt_id_nonascii = "NONASCIIABC"
    (corpus_dir / f"{yt_id_nonascii}.txt").write_text(
        "Mostly content here. " + ("中文内容" * 30),
        encoding="utf-8",
    )

    # --- Doc in French (language-filter drop) ---
    yt_id_fr = "FRENCHABCDE"
    (corpus_dir / f"{yt_id_fr}.txt").write_text(
        "La pratique du piano est un art exigeant qui demande de la patience.\n"
        "Les exercices de Hanon developpent la force et l'independance des doigts.\n"
        "Chaque journee de pratique doit commencer par un echauffement progressif.\n",
        encoding="utf-8",
    )

    # --- Corrupt UTF-8 file ---
    yt_id_corrupt = "CORRUPTABCD"
    (corpus_dir / f"{yt_id_corrupt}.txt").write_bytes(
        b"Some valid prefix text.\n\xff\xfe\xfd\xfc invalid UTF-8 here \xc3\x28 broken.\n"
    )

    # --- Provenance JSONLs ---
    yt_jsonl = provenance_dir / "provenance_tonebase.jsonl"
    yt_jsonl.write_text(
        json.dumps({
            "url": f"https://www.youtube.com/watch?v={yt_id_1}",
            "title": "Scale Practice Masterclass",
            "channel_or_publisher": "tonebase",
            "download_timestamp": "2026-04-01T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 50,
            "source_tier": "tier1_youtube",
        }) + "\n" + json.dumps({
            "url": f"https://www.youtube.com/watch?v={yt_id_2}",
            "title": "Chopin Voicing",
            "channel_or_publisher": "tonebase",
            "download_timestamp": "2026-04-02T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 40,
            "source_tier": "tier1_youtube",
        }) + "\n" + json.dumps({
            "url": f"https://www.youtube.com/watch?v={yt_id_repeats}",
            "title": "Music History Survey",
            "channel_or_publisher": "tonebase",
            "download_timestamp": "2026-04-03T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 30,
            "source_tier": "tier1_youtube",
        }) + "\n" + json.dumps({
            "url": f"https://www.youtube.com/watch?v={yt_id_dup_a}",
            "title": "Slow Practice 1",
            "channel_or_publisher": "tonebase",
            "download_timestamp": "2026-04-04T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 50,
            "source_tier": "tier1_youtube",
        }) + "\n" + json.dumps({
            "url": f"https://www.youtube.com/watch?v={yt_id_dup_b}",
            "title": "Slow Practice 2",
            "channel_or_publisher": "tonebase",
            "download_timestamp": "2026-04-05T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 55,
            "source_tier": "tier1_youtube",
        }) + "\n" + json.dumps({
            "url": f"https://www.youtube.com/watch?v={yt_id_fr}",
            "title": "Pratique Francaise",
            "channel_or_publisher": "tonebase",
            "download_timestamp": "2026-04-06T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 30,
            "source_tier": "tier1_youtube",
        }) + "\n",
        encoding="utf-8",
    )

    pdf_jsonl = provenance_dir / "provenance_openalex.jsonl"
    pdf_jsonl.write_text(
        json.dumps({
            "url": pdf_url_1,
            "title": "Pedagogy Paper 1",
            "channel_or_publisher": "OpenAlex",
            "download_timestamp": "2026-04-07T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 60,
            "source_tier": "tier3_musicology",
        }) + "\n" + json.dumps({
            "url": pdf_url_refs,
            "title": "Debussy Pedaling",
            "channel_or_publisher": "OpenAlex",
            "download_timestamp": "2026-04-08T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 80,
            "source_tier": "tier3_musicology",
        }) + "\n",
        encoding="utf-8",
    )

    web_jsonl = provenance_dir / "provenance_disclaimer.jsonl"
    web_jsonl.write_text(
        json.dumps({
            "url": web_url_disc,
            "title": "Phrasing article",
            "channel_or_publisher": "music.example.org",
            "download_timestamp": "2026-04-09T00:00:00Z",
            "license_claimed": "unknown",
            "word_count": 30,
            "source_tier": "tier2_literature",
        }) + "\n",
        encoding="utf-8",
    )

    return corpus_dir, provenance_dir


@pytest.fixture
def fixture_ids(tiny_corpus):
    """Stable handle on the doc_ids the fixture produces, for assertions."""
    corpus_dir, _ = tiny_corpus
    return {
        "yt_normal_1": "abcdefghijk",
        "yt_normal_2": "lmnopqrstuv",
        "yt_repeats": "wxyz12345AB",
        "yt_short": "SHORTABCDEF",
        "yt_nonascii": "NONASCIIABC",
        "yt_french": "FRENCHABCDE",
        "yt_corrupt": "CORRUPTABCD",
        "yt_dup_a": "DUP1234567x",
        "yt_dup_b": "DUP1234567y",
        "pdf_h_1": _sha12("https://openalex.example.org/W123/paper.pdf"),
        "pdf_h_refs": _sha12("https://openalex.example.org/W456/paper.pdf"),
        "web_h_disc": _sha12("https://music.example.org/disclaimer/article-1"),
    }
```

`apps/evals/pyproject.toml` — add four lines to `[project.dependencies]` between `datasketch>=1.9.0` and `httpx>=0.28.0`:
```toml
    "datasets>=3.0.0",
    "ftfy>=6.2.0",
    "huggingface_hub>=0.27.0",
    "langdetect>=1.0.9",
```

Then run `cd apps/evals && uv sync` to install.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv sync && uv run pytest teacher_model/cpt_pipeline/tests/test_setup.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/__init__.py apps/evals/teacher_model/cpt_pipeline/tests/__init__.py apps/evals/teacher_model/cpt_pipeline/tests/conftest.py apps/evals/teacher_model/cpt_pipeline/tests/test_setup.py apps/evals/pyproject.toml apps/evals/uv.lock && git commit -m "feat(cpt-pipeline): scaffold package, fixture, and deps"
```

---

# Track A: source_resolver (Tasks 2-6, sequential within track)

## Task 2: source_resolver — youtube coarse classification

**Group:** 1 (parallel with Tracks B/C/D/E)

**Behavior being verified:** an 11-char alphanumeric filename stem is classified as `youtube` source.
**Interface under test:** `resolve_source(filename, provenance_index)`.

**Files:**
- Create: `apps/evals/teacher_model/cpt_pipeline/source_resolver.py`
- Create: `apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py`

- [ ] **Step 1: Write the failing test**

`apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py`:
```python
"""Source resolution behavior tests."""
from teacher_model.cpt_pipeline.source_resolver import resolve_source


def test_youtube_filename_resolves_to_youtube_coarse():
    result = resolve_source("abcdefghijk.txt", {})
    assert result.startswith("youtube:"), f"expected youtube: prefix, got {result!r}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_source_resolver.py::test_youtube_filename_resolves_to_youtube_coarse -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.cpt_pipeline.source_resolver'`

- [ ] **Step 3: Implement the minimum to make the test pass**

`apps/evals/teacher_model/cpt_pipeline/source_resolver.py`:
```python
"""Source resolution for cpt_pipeline.

Classifies corpus files into a `<coarse>:<fine>` source string.
Coarse source comes from filename pattern; fine source from URL-hash / video-id
lookup against provenance JSONLs.
"""
from __future__ import annotations

import re
from pathlib import Path

YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def resolve_source(filename: str, provenance_index: dict[str, str]) -> str:
    """Return `<coarse>:<fine>` source for a corpus filename.

    Coarse classification:
      - 11-char alphanumeric stem -> "youtube"
      - else -> "unknown"
    Fine source comes from `provenance_index[stem]`; "unknown" if absent.
    """
    stem = Path(filename).stem
    if YOUTUBE_ID_PATTERN.match(stem):
        coarse = "youtube"
    else:
        coarse = "unknown"
    fine = provenance_index.get(stem, "unknown")
    return f"{coarse}:{fine}"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_source_resolver.py::test_youtube_filename_resolves_to_youtube_coarse -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/source_resolver.py apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py && git commit -m "feat(cpt-pipeline): resolve youtube coarse source from filename"
```

---

## Task 3: source_resolver — academic_pdf coarse classification

**Group:** 1 (parallel with other tracks; sequential within Track A — depends on Task 2)

**Behavior being verified:** a `pdf_<12hex>` filename stem is classified as `academic_pdf` source.
**Interface under test:** `resolve_source(filename, provenance_index)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/source_resolver.py`
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py`:
```python
def test_pdf_filename_resolves_to_academic_pdf_coarse():
    result = resolve_source("pdf_0123456789ab.txt", {})
    assert result.startswith("academic_pdf:"), f"expected academic_pdf: prefix, got {result!r}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_source_resolver.py::test_pdf_filename_resolves_to_academic_pdf_coarse -xvs
```
Expected: FAIL — `AssertionError: expected academic_pdf: prefix, got 'unknown:unknown'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/evals/teacher_model/cpt_pipeline/source_resolver.py` with:
```python
"""Source resolution for cpt_pipeline."""
from __future__ import annotations

import re
from pathlib import Path

YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
PDF_PATTERN = re.compile(r"^pdf_[a-f0-9]{12}$")


def resolve_source(filename: str, provenance_index: dict[str, str]) -> str:
    """Return `<coarse>:<fine>` source for a corpus filename."""
    stem = Path(filename).stem
    if PDF_PATTERN.match(stem):
        coarse = "academic_pdf"
    elif YOUTUBE_ID_PATTERN.match(stem):
        coarse = "youtube"
    else:
        coarse = "unknown"
    fine = provenance_index.get(stem, "unknown")
    return f"{coarse}:{fine}"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_source_resolver.py -xvs
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/source_resolver.py apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py && git commit -m "feat(cpt-pipeline): resolve academic_pdf coarse source from filename"
```

---

## Task 4: source_resolver — web_scrape coarse classification

**Group:** 1 (sequential within Track A — depends on Task 3)

**Behavior being verified:** a `web_<12hex>` filename stem is classified as `web_scrape` source.
**Interface under test:** `resolve_source(filename, provenance_index)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/source_resolver.py`
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py`:
```python
def test_web_filename_resolves_to_web_scrape_coarse():
    result = resolve_source("web_0123456789ab.txt", {})
    assert result.startswith("web_scrape:"), f"expected web_scrape: prefix, got {result!r}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_source_resolver.py::test_web_filename_resolves_to_web_scrape_coarse -xvs
```
Expected: FAIL — `AssertionError: expected web_scrape: prefix, got 'unknown:unknown'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/evals/teacher_model/cpt_pipeline/source_resolver.py` with:
```python
"""Source resolution for cpt_pipeline."""
from __future__ import annotations

import re
from pathlib import Path

YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
PDF_PATTERN = re.compile(r"^pdf_[a-f0-9]{12}$")
WEB_PATTERN = re.compile(r"^web_[a-f0-9]{12}$")


def resolve_source(filename: str, provenance_index: dict[str, str]) -> str:
    """Return `<coarse>:<fine>` source for a corpus filename."""
    stem = Path(filename).stem
    if PDF_PATTERN.match(stem):
        coarse = "academic_pdf"
    elif WEB_PATTERN.match(stem):
        coarse = "web_scrape"
    elif YOUTUBE_ID_PATTERN.match(stem):
        coarse = "youtube"
    else:
        coarse = "unknown"
    fine = provenance_index.get(stem, "unknown")
    return f"{coarse}:{fine}"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_source_resolver.py -xvs
```
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/source_resolver.py apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py && git commit -m "feat(cpt-pipeline): resolve web_scrape coarse source from filename"
```

---

## Task 5: source_resolver — provenance index builds and enriches fine source

**Group:** 1 (sequential within Track A — depends on Task 4)

**Behavior being verified:** `build_provenance_index` walks JSONLs and produces an index mapping each expected filename stem to its harvester name; `resolve_source` uses this index for fine source.
**Interface under test:** `build_provenance_index(provenance_dir)` and `resolve_source(filename, provenance_index)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/source_resolver.py`
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py`:
```python
from teacher_model.cpt_pipeline.source_resolver import build_provenance_index


def test_provenance_index_enriches_fine_source(tiny_corpus):
    _, provenance_dir = tiny_corpus
    index = build_provenance_index(provenance_dir)
    # Youtube doc id from fixture should be in index, mapped to "tonebase"
    assert index.get("abcdefghijk") == "tonebase", \
        f"expected tonebase mapping for youtube id, got {index.get('abcdefghijk')!r}"
    # OpenAlex pdf url hash should be in index, mapped to "openalex"
    import hashlib
    pdf_h = hashlib.sha256(b"https://openalex.example.org/W123/paper.pdf").hexdigest()[:12]
    assert index.get(f"pdf_{pdf_h}") == "openalex", \
        f"expected openalex mapping for pdf hash, got {index.get(f'pdf_{pdf_h}')!r}"
    # resolve_source uses the index for fine source
    assert resolve_source("abcdefghijk.txt", index) == "youtube:tonebase"
    assert resolve_source(f"pdf_{pdf_h}.txt", index) == "academic_pdf:openalex"
    # Unknown filename gets coarse classification with :unknown fine
    assert resolve_source("ZZZZZZZZZZZ.txt", index) == "youtube:unknown"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_source_resolver.py::test_provenance_index_enriches_fine_source -xvs
```
Expected: FAIL — `ImportError: cannot import name 'build_provenance_index' from 'teacher_model.cpt_pipeline.source_resolver'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/evals/teacher_model/cpt_pipeline/source_resolver.py` with:
```python
"""Source resolution for cpt_pipeline."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
PDF_PATTERN = re.compile(r"^pdf_[a-f0-9]{12}$")
WEB_PATTERN = re.compile(r"^web_[a-f0-9]{12}$")


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _extract_youtube_id(url: str) -> str | None:
    if "youtu.be/" in url:
        tail = url.split("youtu.be/", 1)[1]
        return tail.split("?", 1)[0].split("/", 1)[0]
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    candidates = qs.get("v", [])
    return candidates[0] if candidates else None


def build_provenance_index(provenance_dir: Path) -> dict[str, str]:
    """Walk `provenance_*.jsonl` files in `provenance_dir` and return a
    mapping of expected filename stems to harvester names.

    For YouTube rows: extracts `v=` (or `youtu.be/`) id from the url.
    For non-YouTube rows: produces both `pdf_{sha256(url)[:12]}` and
    `web_{sha256(url)[:12]}` index entries (the harvester used one or the other,
    we cannot tell from the JSONL alone).
    """
    index: dict[str, str] = {}
    for path in sorted(Path(provenance_dir).glob("provenance_*.jsonl")):
        harvester = path.stem.replace("provenance_", "")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                url = row.get("url", "")
                vid = _extract_youtube_id(url)
                if vid and YOUTUBE_ID_PATTERN.match(vid):
                    index[vid] = harvester
                else:
                    h = _hash_url(url)
                    index.setdefault(f"pdf_{h}", harvester)
                    index.setdefault(f"web_{h}", harvester)
    return index


def resolve_source(filename: str, provenance_index: dict[str, str]) -> str:
    """Return `<coarse>:<fine>` source for a corpus filename."""
    stem = Path(filename).stem
    if PDF_PATTERN.match(stem):
        coarse = "academic_pdf"
    elif WEB_PATTERN.match(stem):
        coarse = "web_scrape"
    elif YOUTUBE_ID_PATTERN.match(stem):
        coarse = "youtube"
    else:
        coarse = "unknown"
    fine = provenance_index.get(stem, "unknown")
    return f"{coarse}:{fine}"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_source_resolver.py -xvs
```
Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/source_resolver.py apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py && git commit -m "feat(cpt-pipeline): build provenance index and enrich fine source"
```

---

## Task 6: source_resolver — unknown coarse fallback for malformed filenames

**Group:** 1 (sequential within Track A — depends on Task 5)

**Behavior being verified:** filenames matching none of the patterns return `unknown:unknown`.
**Interface under test:** `resolve_source(filename, provenance_index)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py`:
```python
def test_malformed_filename_returns_unknown_coarse_and_fine():
    # 5-char stem, doesn't match any pattern
    assert resolve_source("short.txt", {}) == "unknown:unknown"
    # 12-char stem (one off from youtube)
    assert resolve_source("twelvecharsxx.txt", {}) == "unknown:unknown"
    # pdf_ prefix but wrong hash length
    assert resolve_source("pdf_short.txt", {}) == "unknown:unknown"
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_source_resolver.py::test_malformed_filename_returns_unknown_coarse_and_fine -xvs
```
Expected: PASS — Task 5's impl already handles this case (the pattern matchers reject malformed inputs and fall through to `unknown`). If the test PASSES without changes, the test is documenting an invariant; commit it as a guard against regression.

If FAIL: investigate the pattern that's incorrectly matching. The most likely cause is a regex not anchored — adjust the regex to require `^...$` exactly.

- [ ] **Step 3: No implementation needed**

The regex patterns from Task 5 are anchored (`^...$`), so malformed inputs fall through to `unknown`. No code changes.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_source_resolver.py -xvs
```
Expected: PASS (all five tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_source_resolver.py && git commit -m "test(cpt-pipeline): guard unknown coarse fallback for malformed filenames"
```

---

# Track B: structural_filter (Tasks 7-11, sequential within track)

## Task 7: structural_filter — drop docs below min char floor

**Group:** 1 (parallel with Tracks A/C/D/E)

**Behavior being verified:** docs shorter than 100 characters are dropped with reason `too_short` and emitted to `drops.jsonl`.
**Interface under test:** `run_filter(manifest_in, out_dir)`.

**Files:**
- Create: `apps/evals/teacher_model/cpt_pipeline/structural_filter.py`
- Create: `apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py`

- [ ] **Step 1: Write the failing test**

`apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py`:
```python
"""Structural filter behavior tests."""
import json
from pathlib import Path

from teacher_model.cpt_pipeline.structural_filter import run_filter


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_drops_doc_below_min_chars(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_manifest(manifest_in, [
        {"doc_id": "long", "source": "youtube:tonebase", "text": "x" * 200, "word_count": 1},
        {"doc_id": "short", "source": "youtube:tonebase", "text": "Too short.", "word_count": 2},
    ])

    manifest_out = run_filter(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert [r["doc_id"] for r in surviving] == ["long"]
    drops = _read_jsonl(out_dir / "drops.jsonl")
    assert any(d["doc_id"] == "short" and d["drop_reason"] == "too_short" for d in drops), \
        f"expected short doc dropped with reason too_short, got {drops}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_structural_filter.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.cpt_pipeline.structural_filter'`

- [ ] **Step 3: Implement the minimum to make the test pass**

`apps/evals/teacher_model/cpt_pipeline/structural_filter.py`:
```python
"""Stage 2: structural filter for cpt_pipeline."""
from __future__ import annotations

import json
from pathlib import Path

MIN_CHARS = 100


def run_filter(manifest_in: Path, out_dir: Path) -> Path:
    """Drop docs failing structural gates; emit surviving docs and drops sidecar.

    Returns path to the output manifest.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"
    drops_path = out_dir / "drops.jsonl"

    with Path(manifest_in).open(encoding="utf-8") as fh, \
         manifest_out.open("w", encoding="utf-8") as out_fh, \
         drops_path.open("w", encoding="utf-8") as drops_fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if len(text) < MIN_CHARS:
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "too_short"}) + "\n")
                continue
            out_fh.write(json.dumps(row) + "\n")
    return manifest_out
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_structural_filter.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/structural_filter.py apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py && git commit -m "feat(cpt-pipeline): drop docs below min-chars in structural filter"
```

---

## Task 8: structural_filter — drop docs above non-ASCII ratio threshold

**Group:** 1 (sequential within Track B — depends on Task 7)

**Behavior being verified:** docs with >50% non-ASCII characters are dropped with reason `non_ascii_ratio`.
**Interface under test:** `run_filter(manifest_in, out_dir)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/structural_filter.py`
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py`:
```python
def test_drops_doc_above_non_ascii_ratio(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    text_high_nonascii = "ascii prefix " + ("中文" * 100)  # heavily Chinese
    _write_manifest(manifest_in, [
        {"doc_id": "ok", "source": "youtube:tonebase", "text": "x" * 200, "word_count": 1},
        {"doc_id": "nonascii", "source": "youtube:tonebase", "text": text_high_nonascii, "word_count": 100},
    ])

    manifest_out = run_filter(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert [r["doc_id"] for r in surviving] == ["ok"]
    drops = _read_jsonl(out_dir / "drops.jsonl")
    assert any(d["doc_id"] == "nonascii" and d["drop_reason"] == "non_ascii_ratio" for d in drops)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_structural_filter.py::test_drops_doc_above_non_ascii_ratio -xvs
```
Expected: FAIL — non-ASCII doc passes through (filter not yet implemented)

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/evals/teacher_model/cpt_pipeline/structural_filter.py` with:
```python
"""Stage 2: structural filter for cpt_pipeline."""
from __future__ import annotations

import json
from pathlib import Path

MIN_CHARS = 100
MAX_NON_ASCII_RATIO = 0.5


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def run_filter(manifest_in: Path, out_dir: Path) -> Path:
    """Drop docs failing structural gates; emit surviving docs and drops sidecar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"
    drops_path = out_dir / "drops.jsonl"

    with Path(manifest_in).open(encoding="utf-8") as fh, \
         manifest_out.open("w", encoding="utf-8") as out_fh, \
         drops_path.open("w", encoding="utf-8") as drops_fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if len(text) < MIN_CHARS:
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "too_short"}) + "\n")
                continue
            if _non_ascii_ratio(text) > MAX_NON_ASCII_RATIO:
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "non_ascii_ratio"}) + "\n")
                continue
            out_fh.write(json.dumps(row) + "\n")
    return manifest_out
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_structural_filter.py -xvs
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/structural_filter.py apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py && git commit -m "feat(cpt-pipeline): drop docs above non-ASCII ratio threshold"
```

---

## Task 9: structural_filter — drop non-English docs

**Group:** 1 (sequential within Track B — depends on Task 8)

**Behavior being verified:** docs detected as non-English by `langdetect` are dropped with reason `non_english`.
**Interface under test:** `run_filter(manifest_in, out_dir)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/structural_filter.py`
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py`:
```python
def test_drops_non_english_doc(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    french_text = (
        "La pratique du piano est un art exigeant qui demande de la patience. "
        "Les exercices de Hanon developpent la force des doigts. "
        "Chaque journee de pratique doit commencer par un echauffement progressif."
    )
    english_text = (
        "Slow practice is the foundation of all technique work. Begin every "
        "session with five minutes of metronome scales. The objective is evenness."
    )
    _write_manifest(manifest_in, [
        {"doc_id": "en", "source": "youtube:tonebase", "text": english_text, "word_count": 25},
        {"doc_id": "fr", "source": "youtube:tonebase", "text": french_text, "word_count": 30},
    ])

    manifest_out = run_filter(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert [r["doc_id"] for r in surviving] == ["en"]
    drops = _read_jsonl(out_dir / "drops.jsonl")
    assert any(d["doc_id"] == "fr" and d["drop_reason"] == "non_english" for d in drops)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_structural_filter.py::test_drops_non_english_doc -xvs
```
Expected: FAIL — French doc passes through

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/evals/teacher_model/cpt_pipeline/structural_filter.py` with:
```python
"""Stage 2: structural filter for cpt_pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

# Make langdetect deterministic across runs
DetectorFactory.seed = 0

MIN_CHARS = 100
MAX_NON_ASCII_RATIO = 0.5


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def run_filter(manifest_in: Path, out_dir: Path) -> Path:
    """Drop docs failing structural gates; emit surviving docs and drops sidecar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"
    drops_path = out_dir / "drops.jsonl"

    with Path(manifest_in).open(encoding="utf-8") as fh, \
         manifest_out.open("w", encoding="utf-8") as out_fh, \
         drops_path.open("w", encoding="utf-8") as drops_fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if len(text) < MIN_CHARS:
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "too_short"}) + "\n")
                continue
            if _non_ascii_ratio(text) > MAX_NON_ASCII_RATIO:
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "non_ascii_ratio"}) + "\n")
                continue
            if not _is_english(text):
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "non_english"}) + "\n")
                continue
            out_fh.write(json.dumps(row) + "\n")
    return manifest_out
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_structural_filter.py -xvs
```
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/structural_filter.py apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py && git commit -m "feat(cpt-pipeline): drop non-English docs via langdetect"
```

---

## Task 10: structural_filter — strip references on PDF-derived sources

**Group:** 1 (sequential within Track B — depends on Task 9)

**Behavior being verified:** for docs whose `source` starts with `academic_pdf:`, the text is truncated at the last line matching `^(References|Bibliography|Works Cited|REFERENCES)\s*$`.
**Interface under test:** `run_filter(manifest_in, out_dir)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/structural_filter.py`
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py`:
```python
def test_strips_references_on_academic_pdf_source(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    text = (
        "Body content about pedaling techniques in Debussy. "
        "The half-pedal allows partial damper engagement.\n\n"
        "More body content about impressionist textures and the use of\n"
        "blurred sonorities to create atmospheric effects.\n\n"
        "References\n"
        "Debussy, C. (1905). Estampes. Durand.\n"
        "Howat, R. (1983). Debussy in proportion.\n"
    )
    _write_manifest(manifest_in, [
        {"doc_id": "paper", "source": "academic_pdf:openalex", "text": text, "word_count": 50},
    ])

    manifest_out = run_filter(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert len(surviving) == 1
    out_text = surviving[0]["text"]
    assert "Debussy, C. (1905)" not in out_text, f"references not stripped: {out_text!r}"
    assert "blurred sonorities" in out_text, f"prefix lost during strip: {out_text!r}"
    assert "References" not in out_text, f"References header retained: {out_text!r}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_structural_filter.py::test_strips_references_on_academic_pdf_source -xvs
```
Expected: FAIL — `assert "Debussy, C. (1905)" not in out_text` fails (refs not stripped yet)

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/evals/teacher_model/cpt_pipeline/structural_filter.py` with:
```python
"""Stage 2: structural filter for cpt_pipeline."""
from __future__ import annotations

import json
import re
from pathlib import Path

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

MIN_CHARS = 100
MAX_NON_ASCII_RATIO = 0.5

REFS_PATTERN = re.compile(r"^(References|Bibliography|Works Cited|REFERENCES)\s*$")


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def _strip_references(text: str) -> str:
    """Truncate at the last line matching REFS_PATTERN, drop that line and everything after."""
    lines = text.splitlines(keepends=True)
    last_idx = -1
    for i, line in enumerate(lines):
        if REFS_PATTERN.match(line.strip()):
            last_idx = i
    if last_idx == -1:
        return text
    return "".join(lines[:last_idx]).rstrip() + "\n"


def run_filter(manifest_in: Path, out_dir: Path) -> Path:
    """Drop docs failing structural gates; emit surviving docs and drops sidecar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"
    drops_path = out_dir / "drops.jsonl"

    with Path(manifest_in).open(encoding="utf-8") as fh, \
         manifest_out.open("w", encoding="utf-8") as out_fh, \
         drops_path.open("w", encoding="utf-8") as drops_fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if len(text) < MIN_CHARS:
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "too_short"}) + "\n")
                continue
            if _non_ascii_ratio(text) > MAX_NON_ASCII_RATIO:
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "non_ascii_ratio"}) + "\n")
                continue
            if not _is_english(text):
                drops_fh.write(json.dumps({"doc_id": row["doc_id"], "drop_reason": "non_english"}) + "\n")
                continue
            source = row.get("source", "")
            if source.startswith("academic_pdf:"):
                text = _strip_references(text)
            row["text"] = text
            out_fh.write(json.dumps(row) + "\n")
    return manifest_out
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_structural_filter.py -xvs
```
Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/structural_filter.py apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py && git commit -m "feat(cpt-pipeline): strip references on academic_pdf sources"
```

---

## Task 11: structural_filter — preserve references on non-PDF sources

**Group:** 1 (sequential within Track B — depends on Task 10)

**Behavior being verified:** docs whose `source` does NOT start with `academic_pdf:` (e.g., `youtube:`, `web_scrape:`) retain text containing the literal word "References".
**Interface under test:** `run_filter(manifest_in, out_dir)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py`:
```python
def test_does_not_strip_references_on_youtube_source(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    text = (
        "In this masterclass we discuss several musical references.\n\n"
        "References\n"
        "to Beethoven's Hammerklavier are common in late Romantic literature.\n"
        "We continue with examples from Brahms.\n"
    )
    _write_manifest(manifest_in, [
        {"doc_id": "yt", "source": "youtube:tonebase", "text": text, "word_count": 30},
    ])

    manifest_out = run_filter(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert len(surviving) == 1
    out_text = surviving[0]["text"]
    assert "Beethoven's Hammerklavier" in out_text, \
        f"refs strip incorrectly fired on youtube source: {out_text!r}"
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_structural_filter.py::test_does_not_strip_references_on_youtube_source -xvs
```
Expected: PASS — Task 10's impl already conditions on `source.startswith("academic_pdf:")`. This test is a regression guard.

- [ ] **Step 3: No implementation needed**

Task 10's source-conditional check already enforces this behavior.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_structural_filter.py -xvs
```
Expected: PASS (all five tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_structural_filter.py && git commit -m "test(cpt-pipeline): guard refs strip is source-conditional"
```

---

# Track C: dedup (Tasks 12-16, sequential within track)

## Task 12: dedup — stage 3a removes whole-doc near-duplicates

**Group:** 1 (parallel with Tracks A/B/D/E)

**Behavior being verified:** when stage 3a runs, near-duplicate docs (Jaccard ≥ 0.8) collapse to one (alphabetically-first `doc_id` survives).
**Interface under test:** `run_dedup(manifest_in, out_dir)`.

**Files:**
- Create: `apps/evals/teacher_model/cpt_pipeline/dedup.py`
- Create: `apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py`

- [ ] **Step 1: Write the failing test**

`apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py`:
```python
"""Dedup behavior tests."""
import json
from pathlib import Path

from teacher_model.cpt_pipeline.dedup import run_dedup


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_3a_collapses_doc_level_near_dups(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    base = (
        "Slow practice is the foundation of all technique work. Begin every "
        "session with five minutes of metronome scales at quarter equals 60. The "
        "objective is not speed but evenness, articulation, and tonal control. "
        "Listen for unevenness in the weaker fingers and isolate problem groups."
    )
    _write_manifest(manifest_in, [
        {"doc_id": "DUP1234567x", "source": "youtube:tonebase", "text": base, "word_count": 60},
        {"doc_id": "DUP1234567y", "source": "youtube:tonebase", "text": base + " Additional tail.", "word_count": 62},
        {"doc_id": "OTHER12345A", "source": "youtube:tonebase", "text": "Wholly different content here about voicing in chamber music with eight to ten distinct sentences.", "word_count": 20},
    ])

    manifest_out = run_dedup(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    surviving_ids = sorted(r["doc_id"] for r in surviving)
    assert "DUP1234567x" in surviving_ids, "alphabetically-first dup should survive"
    assert "DUP1234567y" not in surviving_ids, "alphabetically-later dup should be removed"
    assert "OTHER12345A" in surviving_ids, "non-dup unrelated doc must remain"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_dedup.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.cpt_pipeline.dedup'`

- [ ] **Step 3: Implement the minimum to make the test pass**

`apps/evals/teacher_model/cpt_pipeline/dedup.py`:
```python
"""Stage 3: dedup orchestrator (3a doc-level + 3b within-doc + 3c corpus-wide)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from teacher_model.dedup import find_duplicates


def _write_corpus_for_3a(rows: list[dict], scratch: Path) -> dict[str, dict]:
    """Materialize each row's text to a .txt file under scratch named by doc_id.
    Returns {doc_id: row} index for downstream lookup."""
    index: dict[str, dict] = {}
    for row in rows:
        doc_id = row["doc_id"]
        (scratch / f"{doc_id}.txt").write_text(row["text"], encoding="utf-8")
        index[doc_id] = row
    return index


def _stage_3a_remove_doc_dups(rows: list[dict]) -> tuple[list[dict], list[tuple[str, str, float]]]:
    """Run existing teacher_model.dedup.find_duplicates on a temp corpus dir.
    Returns (surviving_rows, dup_pairs)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scratch = Path(tmpdir)
        index = _write_corpus_for_3a(rows, scratch)
        pairs = find_duplicates(scratch, threshold=0.8)
        to_remove: set[str] = set()
        for file1, file2, _sim in pairs:
            id1 = Path(file1).stem
            id2 = Path(file2).stem
            keeper, dup = sorted([id1, id2])
            if keeper not in to_remove:
                to_remove.add(dup)
        surviving = [index[doc_id] for doc_id in index if doc_id not in to_remove]
        return surviving, pairs


def run_dedup(manifest_in: Path, out_dir: Path) -> Path:
    """Three-pass dedup. Returns path to output manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"

    with Path(manifest_in).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    rows, _pairs = _stage_3a_remove_doc_dups(rows)

    with manifest_out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return manifest_out
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_dedup.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/dedup.py apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py && git commit -m "feat(cpt-pipeline): stage 3a doc-level dedup via existing dedup.py"
```

---

## Task 13: dedup — stage 3b strips lines repeating >=3x within a doc

**Group:** 1 (sequential within Track C — depends on Task 12)

**Behavior being verified:** lines (length >=30 chars after normalization) appearing >=3x within the same doc are stripped to first occurrence; surrounding text untouched.
**Interface under test:** `run_dedup(manifest_in, out_dir)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/dedup.py`
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py`:
```python
def test_3b_strips_within_doc_repeated_lines(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    repeat_line = "These are some lengthy disclaimer words that repeat across pages."
    text = (
        "Real article content begins here in the first paragraph. "
        "Discussing pedagogy in detail.\n"
        f"{repeat_line}\n"
        "Section 1 body content goes here with several distinct sentences.\n"
        f"{repeat_line}\n"
        "Section 2 body content with distinct words.\n"
        f"{repeat_line}\n"
        "Section 3 body content with distinct words.\n"
        f"{repeat_line}\n"
        "Conclusion of the article in this final paragraph.\n"
    )
    _write_manifest(manifest_in, [
        {"doc_id": "REPEATSABCDX", "source": "youtube:tonebase", "text": text, "word_count": 80},
    ])

    manifest_out = run_dedup(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert len(surviving) == 1
    out_text = surviving[0]["text"]
    assert out_text.count(repeat_line) == 1, \
        f"expected exactly 1 occurrence of repeated line, got {out_text.count(repeat_line)}: {out_text!r}"
    assert "Section 1 body content" in out_text and "Conclusion" in out_text, \
        "surrounding text was incorrectly stripped"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_dedup.py::test_3b_strips_within_doc_repeated_lines -xvs
```
Expected: FAIL — repeated line still appears 4 times

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/evals/teacher_model/cpt_pipeline/dedup.py` with:
```python
"""Stage 3: dedup orchestrator (3a doc-level + 3b within-doc + 3c corpus-wide)."""
from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from teacher_model.dedup import find_duplicates

WITHIN_DOC_REPEAT_THRESHOLD = 3
MIN_LINE_LEN_FOR_STRIP = 30
PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")


def _normalize_line(line: str) -> str:
    """Lowercase + collapse internal whitespace + strip leading/trailing whitespace."""
    return " ".join(line.lower().split())


def _is_strippable(normalized: str) -> bool:
    """Lines short enough to legitimately repeat (e.g., 'C major') are exempt."""
    return len(normalized) >= MIN_LINE_LEN_FOR_STRIP and not PAGE_NUMBER_RE.match(normalized)


def _strip_within_doc(text: str) -> str:
    """Drop strippable lines that repeat >= WITHIN_DOC_REPEAT_THRESHOLD times within `text`,
    keeping only the first occurrence."""
    lines = text.splitlines(keepends=True)
    counts: Counter[str] = Counter()
    for line in lines:
        norm = _normalize_line(line)
        if _is_strippable(norm):
            counts[norm] += 1
    seen: set[str] = set()
    out_lines: list[str] = []
    for line in lines:
        norm = _normalize_line(line)
        if _is_strippable(norm) and counts[norm] >= WITHIN_DOC_REPEAT_THRESHOLD:
            if norm in seen:
                continue
            seen.add(norm)
        out_lines.append(line)
    return "".join(out_lines)


def _write_corpus_for_3a(rows: list[dict], scratch: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in rows:
        doc_id = row["doc_id"]
        (scratch / f"{doc_id}.txt").write_text(row["text"], encoding="utf-8")
        index[doc_id] = row
    return index


def _stage_3a_remove_doc_dups(rows: list[dict]) -> tuple[list[dict], list[tuple[str, str, float]]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        scratch = Path(tmpdir)
        index = _write_corpus_for_3a(rows, scratch)
        pairs = find_duplicates(scratch, threshold=0.8)
        to_remove: set[str] = set()
        for file1, file2, _sim in pairs:
            id1 = Path(file1).stem
            id2 = Path(file2).stem
            keeper, dup = sorted([id1, id2])
            if keeper not in to_remove:
                to_remove.add(dup)
        surviving = [index[doc_id] for doc_id in index if doc_id not in to_remove]
        return surviving, pairs


def run_dedup(manifest_in: Path, out_dir: Path) -> Path:
    """Three-pass dedup."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"

    with Path(manifest_in).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    rows, _pairs = _stage_3a_remove_doc_dups(rows)
    for row in rows:
        row["text"] = _strip_within_doc(row["text"])

    with manifest_out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return manifest_out
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_dedup.py -xvs
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/dedup.py apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py && git commit -m "feat(cpt-pipeline): stage 3b within-doc line-frequency strip"
```

---

## Task 14: dedup — stage 3c drops lines appearing in >20 distinct docs

**Group:** 1 (sequential within Track C — depends on Task 13)

**Behavior being verified:** strippable lines appearing in >20 distinct surviving docs corpus-wide are dropped from every doc.
**Interface under test:** `run_dedup(manifest_in, out_dir)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/dedup.py`
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py`:
```python
def test_3c_drops_lines_in_more_than_20_docs(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    boilerplate = "This is a sufficiently long boilerplate disclaimer line for stripping."
    rows = []
    for i in range(25):
        text = (
            f"Doc {i} unique body content here with distinct words and pedagogy detail. "
            f"More distinct content for doc {i}.\n"
            f"{boilerplate}\n"
            f"Final unique body content for doc {i} with more distinct words.\n"
        )
        rows.append({"doc_id": f"D{i:010d}A", "source": "youtube:tonebase", "text": text, "word_count": 30})
    _write_manifest(manifest_in, rows)

    manifest_out = run_dedup(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert len(surviving) == 25, "no doc-level dups expected; all 25 should survive"
    assert all(boilerplate not in r["text"] for r in surviving), \
        "boilerplate line should be stripped from every doc"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_dedup.py::test_3c_drops_lines_in_more_than_20_docs -xvs
```
Expected: FAIL — boilerplate still present in surviving docs

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/evals/teacher_model/cpt_pipeline/dedup.py` with:
```python
"""Stage 3: dedup orchestrator (3a doc-level + 3b within-doc + 3c corpus-wide)."""
from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from teacher_model.dedup import find_duplicates

WITHIN_DOC_REPEAT_THRESHOLD = 3
CORPUS_WIDE_LINE_DOC_THRESHOLD = 20
MIN_LINE_LEN_FOR_STRIP = 30
PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")


def _normalize_line(line: str) -> str:
    return " ".join(line.lower().split())


def _is_strippable(normalized: str) -> bool:
    return len(normalized) >= MIN_LINE_LEN_FOR_STRIP and not PAGE_NUMBER_RE.match(normalized)


def _strip_within_doc(text: str) -> str:
    lines = text.splitlines(keepends=True)
    counts: Counter[str] = Counter()
    for line in lines:
        norm = _normalize_line(line)
        if _is_strippable(norm):
            counts[norm] += 1
    seen: set[str] = set()
    out_lines: list[str] = []
    for line in lines:
        norm = _normalize_line(line)
        if _is_strippable(norm) and counts[norm] >= WITHIN_DOC_REPEAT_THRESHOLD:
            if norm in seen:
                continue
            seen.add(norm)
        out_lines.append(line)
    return "".join(out_lines)


def _build_global_line_doc_counts(rows: list[dict]) -> Counter[str]:
    """Count how many distinct docs contain each strippable normalized line."""
    counts: Counter[str] = Counter()
    for row in rows:
        seen: set[str] = set()
        for line in row["text"].splitlines():
            norm = _normalize_line(line)
            if _is_strippable(norm):
                seen.add(norm)
        for norm in seen:
            counts[norm] += 1
    return counts


def _strip_corpus_wide(text: str, banned: set[str]) -> str:
    out_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        norm = _normalize_line(line)
        if norm in banned:
            continue
        out_lines.append(line)
    return "".join(out_lines)


def _write_corpus_for_3a(rows: list[dict], scratch: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in rows:
        doc_id = row["doc_id"]
        (scratch / f"{doc_id}.txt").write_text(row["text"], encoding="utf-8")
        index[doc_id] = row
    return index


def _stage_3a_remove_doc_dups(rows: list[dict]) -> tuple[list[dict], list[tuple[str, str, float]]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        scratch = Path(tmpdir)
        index = _write_corpus_for_3a(rows, scratch)
        pairs = find_duplicates(scratch, threshold=0.8)
        to_remove: set[str] = set()
        for file1, file2, _sim in pairs:
            id1 = Path(file1).stem
            id2 = Path(file2).stem
            keeper, dup = sorted([id1, id2])
            if keeper not in to_remove:
                to_remove.add(dup)
        surviving = [index[doc_id] for doc_id in index if doc_id not in to_remove]
        return surviving, pairs


def run_dedup(manifest_in: Path, out_dir: Path) -> Path:
    """Three-pass dedup."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"

    with Path(manifest_in).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    rows, _pairs = _stage_3a_remove_doc_dups(rows)
    for row in rows:
        row["text"] = _strip_within_doc(row["text"])

    line_doc_counts = _build_global_line_doc_counts(rows)
    banned = {norm for norm, c in line_doc_counts.items() if c > CORPUS_WIDE_LINE_DOC_THRESHOLD}
    for row in rows:
        row["text"] = _strip_corpus_wide(row["text"], banned)

    with manifest_out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return manifest_out
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_dedup.py -xvs
```
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/dedup.py apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py && git commit -m "feat(cpt-pipeline): stage 3c corpus-wide line-frequency strip"
```

---

## Task 15: dedup — 3c respects threshold (line in 19 docs is kept)

**Group:** 1 (sequential within Track C — depends on Task 14)

**Behavior being verified:** the 20-distinct-doc threshold is strictly `>` 20; lines appearing in exactly 19 docs are kept.
**Interface under test:** `run_dedup(manifest_in, out_dir)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py`:
```python
def test_3c_respects_threshold_for_19_docs(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    line = "This is a sufficiently long line that appears in exactly 19 distinct docs."
    rows = []
    for i in range(19):
        text = (
            f"Doc {i} unique body content with distinct words for doc {i}.\n"
            f"{line}\n"
            f"More unique words for doc {i} appearing only once.\n"
        )
        rows.append({"doc_id": f"E{i:010d}B", "source": "youtube:tonebase", "text": text, "word_count": 25})
    _write_manifest(manifest_in, rows)

    manifest_out = run_dedup(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    assert len(surviving) == 19
    kept = sum(1 for r in surviving if line in r["text"])
    assert kept == 19, f"line at threshold (19 docs) should be kept, only {kept}/19 retained"
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_dedup.py::test_3c_respects_threshold_for_19_docs -xvs
```
Expected: PASS — the threshold check is `c > CORPUS_WIDE_LINE_DOC_THRESHOLD` (20), so 19 < 20 passes through.

- [ ] **Step 3: No implementation needed**

Task 14's `> CORPUS_WIDE_LINE_DOC_THRESHOLD` check enforces this. Test is regression guard.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_dedup.py -xvs
```
Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py && git commit -m "test(cpt-pipeline): guard 3c threshold strictly greater than 20"
```

---

## Task 16: dedup — line normalization handles whitespace and case

**Group:** 1 (sequential within Track C — depends on Task 15)

**Behavior being verified:** lines differing only in case or trailing/leading/internal whitespace are recognized as the same line for dedup purposes in 3b.
**Interface under test:** `run_dedup(manifest_in, out_dir)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py`:
```python
def test_normalization_collapses_whitespace_and_case_in_3b(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    text = (
        "Body content begins here in this section with several distinct words.\n"
        "This is a Sufficiently Long Disclaimer line that repeats with variations.\n"
        "Section 1 body with distinct content for the first part.\n"
        "this is a sufficiently long disclaimer line that repeats with variations.\n"
        "Section 2 body with distinct content for the second part.\n"
        "  This  is a   Sufficiently long  disclaimer  line that repeats with variations.  \n"
        "Section 3 body with distinct content for the third part.\n"
    )
    _write_manifest(manifest_in, [
        {"doc_id": "NORMABCDEFG", "source": "youtube:tonebase", "text": text, "word_count": 60},
    ])

    manifest_out = run_dedup(manifest_in, out_dir)

    surviving = _read_jsonl(manifest_out)
    out_text = surviving[0]["text"].lower()
    occurrences = out_text.count("disclaimer line that repeats with variations")
    assert occurrences == 1, \
        f"expected normalization to recognize 3 case/whitespace variants as one line; got {occurrences}"
```

- [ ] **Step 2: Run test — verify it FAILS or PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_dedup.py::test_normalization_collapses_whitespace_and_case_in_3b -xvs
```
Expected: PASS — `_normalize_line` from Task 13 lowercases and collapses whitespace via `" ".join(line.lower().split())`. Test is regression guard.

- [ ] **Step 3: No implementation needed**

Existing `_normalize_line` already handles this.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_dedup.py -xvs
```
Expected: PASS (all five tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_dedup.py && git commit -m "test(cpt-pipeline): guard line normalization for case and whitespace"
```

---

# Track D: split (Tasks 17-20, sequential within track)

## Task 17: split — stratifies 1 doc per source per 100 to validation

**Group:** 1 (parallel with Tracks A/B/C/E)

**Behavior being verified:** sources with ≥100 docs contribute floor(N/100) docs to validation.
**Interface under test:** `run_split(manifest_in, out_dir, seed)`.

**Files:**
- Create: `apps/evals/teacher_model/cpt_pipeline/split.py`
- Create: `apps/evals/teacher_model/cpt_pipeline/tests/test_split.py`

- [ ] **Step 1: Write the failing test**

`apps/evals/teacher_model/cpt_pipeline/tests/test_split.py`:
```python
"""Split behavior tests."""
import json
from pathlib import Path

from teacher_model.cpt_pipeline.split import run_split


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_stratifies_one_per_source_per_100(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rows = []
    for i in range(200):
        rows.append({"doc_id": f"A{i:010d}", "source": "youtube:tonebase", "text": "x", "word_count": 1})
    for i in range(150):
        rows.append({"doc_id": f"B{i:010d}", "source": "academic_pdf:openalex", "text": "x", "word_count": 1})
    _write_manifest(manifest_in, rows)

    train_path, val_path = run_split(manifest_in, out_dir, seed=42)

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    val_yt = [r for r in val_rows if r["source"] == "youtube:tonebase"]
    val_pdf = [r for r in val_rows if r["source"] == "academic_pdf:openalex"]
    assert len(val_yt) == 2, f"200 yt docs -> 2 in val, got {len(val_yt)}"
    assert len(val_pdf) == 1, f"150 pdf docs -> 1 in val, got {len(val_pdf)}"
    assert len(train_rows) + len(val_rows) == 350
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_split.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.cpt_pipeline.split'`

- [ ] **Step 3: Implement the minimum to make the test pass**

`apps/evals/teacher_model/cpt_pipeline/split.py`:
```python
"""Stage 4: stratified 1% per-source held-out split."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

SMALL_SOURCE_THRESHOLD = 100


def run_split(manifest_in: Path, out_dir: Path, seed: int = 42) -> tuple[Path, Path]:
    """Stratified 1%-per-source split. Sources <100 docs -> all to train.

    Returns (train_path, validation_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(manifest_in).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for source in sorted(by_source):
        group = by_source[source]
        group_sorted = sorted(group, key=lambda r: r["doc_id"])
        if len(group_sorted) < SMALL_SOURCE_THRESHOLD:
            train_rows.extend(group_sorted)
            continue
        n_val = len(group_sorted) // 100
        rng = random.Random(seed + hash(source) % 10000)
        idxs = list(range(len(group_sorted)))
        rng.shuffle(idxs)
        val_idx_set = set(idxs[:n_val])
        for i, row in enumerate(group_sorted):
            (val_rows if i in val_idx_set else train_rows).append(row)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"
    with train_path.open("w", encoding="utf-8") as fh:
        for row in train_rows:
            fh.write(json.dumps(row) + "\n")
    with val_path.open("w", encoding="utf-8") as fh:
        for row in val_rows:
            fh.write(json.dumps(row) + "\n")
    return train_path, val_path
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_split.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/split.py apps/evals/teacher_model/cpt_pipeline/tests/test_split.py && git commit -m "feat(cpt-pipeline): stratified 1%-per-source split"
```

---

## Task 18: split — small sources (<100 docs) route entirely to train

**Group:** 1 (sequential within Track D — depends on Task 17)

**Behavior being verified:** sources with <100 docs send 0 docs to validation.
**Interface under test:** `run_split(manifest_in, out_dir, seed)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_split.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_split.py`:
```python
def test_small_source_skips_validation(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rows = []
    for i in range(99):
        rows.append({"doc_id": f"S{i:010d}", "source": "web_scrape:henle", "text": "x", "word_count": 1})
    _write_manifest(manifest_in, rows)

    _train_path, val_path = run_split(manifest_in, out_dir, seed=42)

    val_rows = _read_jsonl(val_path)
    assert len(val_rows) == 0, f"99-doc source should produce 0 val rows, got {len(val_rows)}"
```

- [ ] **Step 2: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_split.py::test_small_source_skips_validation -xvs
```
Expected: PASS — Task 17's `< SMALL_SOURCE_THRESHOLD` branch routes everything to train.

- [ ] **Step 3: No implementation needed**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_split.py -xvs
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_split.py && git commit -m "test(cpt-pipeline): guard small-source skip routes all to train"
```

---

## Task 19: split — same seed produces byte-identical output

**Group:** 1 (sequential within Track D — depends on Task 18)

**Behavior being verified:** running `run_split` twice with the same input and same seed produces byte-identical train/validation files.
**Interface under test:** `run_split(manifest_in, out_dir, seed)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_split.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_split.py`:
```python
def test_same_seed_produces_identical_output(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir_a = tmp_path / "a"
    out_dir_b = tmp_path / "b"
    out_dir_a.mkdir()
    out_dir_b.mkdir()
    rows = []
    for i in range(200):
        rows.append({"doc_id": f"D{i:010d}", "source": "youtube:tonebase", "text": "x", "word_count": 1})
    _write_manifest(manifest_in, rows)

    train_a, val_a = run_split(manifest_in, out_dir_a, seed=42)
    train_b, val_b = run_split(manifest_in, out_dir_b, seed=42)

    assert train_a.read_bytes() == train_b.read_bytes(), "train output not deterministic with same seed"
    assert val_a.read_bytes() == val_b.read_bytes(), "validation output not deterministic with same seed"
```

- [ ] **Step 2: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_split.py::test_same_seed_produces_identical_output -xvs
```
Expected: PASS — `random.Random(seed + hash(source) % 10000)` is deterministic, group sort is stable.

- [ ] **Step 3: No implementation needed**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_split.py -xvs
```
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_split.py && git commit -m "test(cpt-pipeline): guard split determinism with fixed seed"
```

---

## Task 20: split — every doc_id in exactly one of train/validation

**Group:** 1 (sequential within Track D — depends on Task 19)

**Behavior being verified:** no doc_id leaks into both splits or vanishes from both.
**Interface under test:** `run_split(manifest_in, out_dir, seed)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_split.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_split.py`:
```python
def test_every_doc_id_in_exactly_one_split(tmp_path):
    manifest_in = tmp_path / "in.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rows = []
    for i in range(250):
        rows.append({"doc_id": f"X{i:010d}", "source": "youtube:tonebase", "text": "x", "word_count": 1})
    for i in range(50):
        rows.append({"doc_id": f"Y{i:010d}", "source": "web_scrape:small", "text": "x", "word_count": 1})
    _write_manifest(manifest_in, rows)

    train_path, val_path = run_split(manifest_in, out_dir, seed=42)

    train_ids = {r["doc_id"] for r in _read_jsonl(train_path)}
    val_ids = {r["doc_id"] for r in _read_jsonl(val_path)}
    all_ids = {r["doc_id"] for r in rows}
    assert train_ids.isdisjoint(val_ids), "doc_id appears in both splits"
    assert train_ids | val_ids == all_ids, "some doc_id missing from both splits"
```

- [ ] **Step 2: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_split.py::test_every_doc_id_in_exactly_one_split -xvs
```
Expected: PASS — split logic uses `val_idx_set` and `else train`, ensuring exactly-one routing.

- [ ] **Step 3: No implementation needed**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_split.py -xvs
```
Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_split.py && git commit -m "test(cpt-pipeline): guard no doc_id leakage between splits"
```

---

# Track E: hf_publish (Tasks 21-25, sequential within track)

## Task 21: hf_publish — missing HF_TOKEN raises clear exception before network call

**Group:** 1 (parallel with Tracks A/B/C/D)

**Behavior being verified:** when `HF_TOKEN` env var is unset, `run_publish` raises `RuntimeError` with a clearly named message before attempting any Hub API call.
**Interface under test:** `run_publish(train, val, repo_id, private)`.

**Files:**
- Create: `apps/evals/teacher_model/cpt_pipeline/hf_publish.py`
- Create: `apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py`

- [ ] **Step 1: Write the failing test**

`apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py`:
```python
"""HF publish behavior tests."""
import json
from pathlib import Path

import pytest

from teacher_model.cpt_pipeline.hf_publish import run_publish


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_missing_hf_token_raises_runtime_error(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_manifest(train, [{"doc_id": "a", "source": "youtube:tonebase", "text": "Some content here that is long enough.", "word_count": 7}])
    _write_manifest(val, [])
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        run_publish(train, val, repo_id="Jai-D/test-repo", private=True)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_hf_publish.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.cpt_pipeline.hf_publish'`

- [ ] **Step 3: Implement the minimum to make the test pass**

`apps/evals/teacher_model/cpt_pipeline/hf_publish.py`:
```python
"""Stage 5: publish train + validation manifests as a private HF Hub dataset."""
from __future__ import annotations

import os
from pathlib import Path


def run_publish(
    train_manifest: Path,
    val_manifest: Path,
    repo_id: str,
    private: bool = True,
) -> str:
    """Build a HF DatasetDict from the manifests and push to Hub.

    Returns the published Hub URL.
    Raises RuntimeError if HF_TOKEN env var is missing.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Set it to a token with 'write' scope before running stage 5."
        )
    raise NotImplementedError("subsequent tasks add the actual push")
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_hf_publish.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/hf_publish.py apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py && git commit -m "feat(cpt-pipeline): hf_publish fails loud on missing HF_TOKEN"
```

---

## Task 22: hf_publish — push uses correct repo_id, private flag, dataset repo_type

**Group:** 1 (sequential within Track E — depends on Task 21)

**Behavior being verified:** when `HF_TOKEN` is set, `run_publish` calls `HfApi.create_repo` and `push_to_hub` with the configured `repo_id`, `private=True`, `repo_type="dataset"`.
**Interface under test:** `run_publish(train, val, repo_id, private)`, with `huggingface_hub` mocked.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/hf_publish.py`
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py`:
```python
def test_push_uses_correct_repo_args(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_manifest(train, [
        {"doc_id": "a", "source": "youtube:tonebase", "text": "Some content here that is long enough.", "word_count": 7},
    ])
    _write_manifest(val, [
        {"doc_id": "b", "source": "youtube:tonebase", "text": "Validation content here that is long enough.", "word_count": 7},
    ])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    captured = {}

    class FakeHfApi:
        def create_repo(self, repo_id, private, repo_type, exist_ok, token):
            captured["create"] = {
                "repo_id": repo_id, "private": private, "repo_type": repo_type,
                "exist_ok": exist_ok, "token": token,
            }
        def repo_info(self, repo_id, repo_type, token):
            captured["info"] = {"repo_id": repo_id, "repo_type": repo_type}
            class _R:
                pass
            return _R()

    captured_push = {}
    def fake_push_to_hub(self, repo_id, private, token):
        captured_push["repo_id"] = repo_id
        captured_push["private"] = private
        captured_push["token"] = token

    import teacher_model.cpt_pipeline.hf_publish as mod
    monkeypatch.setattr(mod, "HfApi", FakeHfApi)
    from datasets import DatasetDict
    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push_to_hub)

    url = run_publish(train, val, repo_id="Jai-D/test-repo", private=True)

    assert captured["create"]["repo_id"] == "Jai-D/test-repo"
    assert captured["create"]["private"] is True
    assert captured["create"]["repo_type"] == "dataset"
    assert captured["create"]["exist_ok"] is True
    assert captured_push["repo_id"] == "Jai-D/test-repo"
    assert captured_push["private"] is True
    assert "Jai-D/test-repo" in url
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_hf_publish.py::test_push_uses_correct_repo_args -xvs
```
Expected: FAIL — `NotImplementedError: subsequent tasks add the actual push`

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/evals/teacher_model/cpt_pipeline/hf_publish.py` with:
```python
"""Stage 5: publish train + validation manifests as a private HF Hub dataset."""
from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi


_FEATURES = Features({
    "text": Value("string"),
    "source": Value("string"),
    "doc_id": Value("string"),
})


def _read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"text": r["text"], "source": r["source"], "doc_id": r["doc_id"]})
    return rows


def run_publish(
    train_manifest: Path,
    val_manifest: Path,
    repo_id: str,
    private: bool = True,
) -> str:
    """Build a HF DatasetDict from the manifests and push to Hub."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Set it to a token with 'write' scope before running stage 5."
        )

    train_rows = _read_manifest(train_manifest)
    val_rows = _read_manifest(val_manifest)
    train_ds = Dataset.from_list(train_rows, features=_FEATURES)
    val_ds = Dataset.from_list(val_rows, features=_FEATURES)
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})

    api = HfApi()
    api.create_repo(
        repo_id=repo_id, private=private, repo_type="dataset", exist_ok=True, token=token,
    )
    dataset.push_to_hub(repo_id, private=private, token=token)

    return f"https://huggingface.co/datasets/{repo_id}"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_hf_publish.py -xvs
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/hf_publish.py apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py && git commit -m "feat(cpt-pipeline): publish DatasetDict to private Hub dataset repo"
```

---

## Task 23: hf_publish — schema is exactly {text, source, doc_id}

**Group:** 1 (sequential within Track E — depends on Task 22)

**Behavior being verified:** the published dataset's features schema contains exactly `text`, `source`, `doc_id` (no other columns).
**Interface under test:** `run_publish(train, val, repo_id, private)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py`:
```python
def test_published_schema_is_exactly_three_columns(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_manifest(train, [
        {"doc_id": "a", "source": "youtube:tonebase", "text": "Long enough text content here for ingestion.", "word_count": 8, "extra_internal_field": "should_not_appear"},
    ])
    _write_manifest(val, [])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    captured_dataset = {}

    class FakeHfApi:
        def create_repo(self, **kwargs): pass
    def fake_push_to_hub(self, repo_id, private, token):
        captured_dataset["features"] = {split: ds.features for split, ds in self.items()}

    import teacher_model.cpt_pipeline.hf_publish as mod
    monkeypatch.setattr(mod, "HfApi", FakeHfApi)
    from datasets import DatasetDict
    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push_to_hub)

    run_publish(train, val, repo_id="Jai-D/test-repo", private=True)

    train_features = captured_dataset["features"]["train"]
    assert sorted(train_features.keys()) == ["doc_id", "source", "text"], \
        f"unexpected schema columns: {sorted(train_features.keys())}"
```

- [ ] **Step 2: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_hf_publish.py::test_published_schema_is_exactly_three_columns -xvs
```
Expected: PASS — `_read_manifest` only copies the three fields into the row dict, so extra fields are dropped before `Dataset.from_list`.

- [ ] **Step 3: No implementation needed**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_hf_publish.py -xvs
```
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py && git commit -m "test(cpt-pipeline): guard schema is exactly text/source/doc_id"
```

---

## Task 24: hf_publish — DatasetDict has both train and validation splits

**Group:** 1 (sequential within Track E — depends on Task 23)

**Behavior being verified:** the published artifact is a `DatasetDict` with keys exactly `{train, validation}` (no `test`).
**Interface under test:** `run_publish(train, val, repo_id, private)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py`:
```python
def test_dataset_dict_has_train_and_validation_only(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_manifest(train, [
        {"doc_id": "a", "source": "youtube:tonebase", "text": "Long enough text content here for ingestion.", "word_count": 8},
    ])
    _write_manifest(val, [
        {"doc_id": "b", "source": "youtube:tonebase", "text": "Validation content here for ingestion.", "word_count": 6},
    ])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    captured = {}

    class FakeHfApi:
        def create_repo(self, **kwargs): pass
    def fake_push(self, repo_id, private, token):
        captured["splits"] = sorted(self.keys())

    import teacher_model.cpt_pipeline.hf_publish as mod
    monkeypatch.setattr(mod, "HfApi", FakeHfApi)
    from datasets import DatasetDict
    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push)

    run_publish(train, val, repo_id="Jai-D/test-repo", private=True)

    assert captured["splits"] == ["train", "validation"], f"unexpected splits: {captured['splits']}"
```

- [ ] **Step 2: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_hf_publish.py::test_dataset_dict_has_train_and_validation_only -xvs
```
Expected: PASS — Task 22's impl uses exactly `DatasetDict({"train": ..., "validation": ...})`.

- [ ] **Step 3: No implementation needed**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_hf_publish.py -xvs
```
Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py && git commit -m "test(cpt-pipeline): guard DatasetDict has only train and validation splits"
```

---

## Task 25: hf_publish — dataset card includes counts and per-source breakdown

**Group:** 1 (sequential within Track E — depends on Task 24)

**Behavior being verified:** `run_publish` writes a `README.md` dataset card containing total train word count, validation word count, and a per-source row count breakdown.
**Interface under test:** `run_publish(train, val, repo_id, private)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/hf_publish.py`
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py`:
```python
def test_dataset_card_includes_counts_and_per_source(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_manifest(train, [
        {"doc_id": "a", "source": "youtube:tonebase", "text": "ten words here ten words here ten words here ten words here", "word_count": 12},
        {"doc_id": "c", "source": "academic_pdf:openalex", "text": "twenty words here twenty words here twenty words here twenty words here twenty words here", "word_count": 20},
    ])
    _write_manifest(val, [
        {"doc_id": "b", "source": "youtube:tonebase", "text": "five words here ok content for validation set", "word_count": 8},
    ])
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    class FakeHfApi:
        def create_repo(self, **kwargs): pass
    def fake_push(self, repo_id, private, token): pass

    import teacher_model.cpt_pipeline.hf_publish as mod
    monkeypatch.setattr(mod, "HfApi", FakeHfApi)
    from datasets import DatasetDict
    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push)

    run_publish(train, val, repo_id="Jai-D/test-repo", private=True, card_out_dir=tmp_path)

    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "train" in card.lower()
    assert "validation" in card.lower()
    assert "youtube:tonebase" in card
    assert "academic_pdf:openalex" in card
    # train rows = 2, val rows = 1, total = 3
    assert "3" in card or "2" in card
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_hf_publish.py::test_dataset_card_includes_counts_and_per_source -xvs
```
Expected: FAIL — `TypeError: run_publish() got an unexpected keyword argument 'card_out_dir'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/evals/teacher_model/cpt_pipeline/hf_publish.py` with:
```python
"""Stage 5: publish train + validation manifests as a private HF Hub dataset."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi


_FEATURES = Features({
    "text": Value("string"),
    "source": Value("string"),
    "doc_id": Value("string"),
})


def _read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"text": r["text"], "source": r["source"], "doc_id": r["doc_id"]})
    return rows


def _word_count(rows: list[dict]) -> int:
    return sum(len(r["text"].split()) for r in rows)


def _source_breakdown(rows: list[dict]) -> Counter:
    return Counter(r["source"] for r in rows)


def _build_card(repo_id: str, train_rows: list[dict], val_rows: list[dict]) -> str:
    train_words = _word_count(train_rows)
    val_words = _word_count(val_rows)
    src_breakdown = _source_breakdown(train_rows) + _source_breakdown(val_rows)
    src_lines = "\n".join(f"- `{src}`: {count}" for src, count in sorted(src_breakdown.items()))
    return (
        f"# {repo_id}\n\n"
        f"Private intermediate corpus for piano-pedagogy CPT / SFT data synthesis.\n\n"
        f"## Splits\n\n"
        f"- `train`: {len(train_rows)} docs, {train_words} words\n"
        f"- `validation`: {len(val_rows)} docs, {val_words} words\n\n"
        f"## Per-source breakdown (train + validation)\n\n"
        f"{src_lines}\n\n"
        f"## Provenance\n\n"
        f"Mixed sources (YouTube transcripts, OpenAlex / Semantic Scholar / Internet Archive PDFs, "
        f"scraped pedagogy web pages). See per-source `provenance_*.jsonl` in the harvest pipeline "
        f"for license claims. Private dataset, internal use only; do not redistribute.\n"
    )


def run_publish(
    train_manifest: Path,
    val_manifest: Path,
    repo_id: str,
    private: bool = True,
    card_out_dir: Path | None = None,
) -> str:
    """Build a HF DatasetDict from the manifests, write a card, and push to Hub."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Set it to a token with 'write' scope before running stage 5."
        )

    train_rows = _read_manifest(train_manifest)
    val_rows = _read_manifest(val_manifest)

    if card_out_dir is not None:
        Path(card_out_dir).mkdir(parents=True, exist_ok=True)
        (Path(card_out_dir) / "README.md").write_text(
            _build_card(repo_id, train_rows, val_rows), encoding="utf-8",
        )

    train_ds = Dataset.from_list(train_rows, features=_FEATURES)
    val_ds = Dataset.from_list(val_rows, features=_FEATURES)
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})

    api = HfApi()
    api.create_repo(
        repo_id=repo_id, private=private, repo_type="dataset", exist_ok=True, token=token,
    )
    dataset.push_to_hub(repo_id, private=private, token=token)

    return f"https://huggingface.co/datasets/{repo_id}"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_hf_publish.py -xvs
```
Expected: PASS (all five tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/hf_publish.py apps/evals/teacher_model/cpt_pipeline/tests/test_hf_publish.py && git commit -m "feat(cpt-pipeline): generate dataset card with counts and per-source breakdown"
```

---

# Track F: ingest (Tasks 26-30, sequential within track, depends on Track A complete)

## Task 26: ingest — produces stable doc_id from filename stem

**Group:** 2 (depends on Track A: source_resolver complete)

**Behavior being verified:** every output manifest row has `doc_id == Path(filename).stem`.
**Interface under test:** `run_ingest(corpus_dir, provenance_dir, out_dir)`.

**Files:**
- Create: `apps/evals/teacher_model/cpt_pipeline/ingest.py`
- Create: `apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

`apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py`:
```python
"""Ingest behavior tests."""
import json
from pathlib import Path

from teacher_model.cpt_pipeline.ingest import run_ingest


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_doc_id_is_filename_stem(tiny_corpus, tmp_path):
    corpus_dir, provenance_dir = tiny_corpus
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest_path = run_ingest(corpus_dir, provenance_dir, out_dir)

    rows = _read_jsonl(manifest_path)
    doc_ids = {r["doc_id"] for r in rows}
    assert "abcdefghijk" in doc_ids
    assert "lmnopqrstuv" in doc_ids
    # No row has a doc_id ending in .txt
    assert all(not r["doc_id"].endswith(".txt") for r in rows)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_ingest.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.cpt_pipeline.ingest'`

- [ ] **Step 3: Implement the minimum to make the test pass**

`apps/evals/teacher_model/cpt_pipeline/ingest.py`:
```python
"""Stage 1: ingest corpus + provenance into unified manifest."""
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.cpt_pipeline.source_resolver import build_provenance_index, resolve_source


def run_ingest(corpus_dir: Path, provenance_dir: Path, out_dir: Path) -> Path:
    """Walk corpus_dir for .txt files, join to provenance_dir JSONLs, emit manifest."""
    corpus_dir = Path(corpus_dir)
    provenance_dir = Path(provenance_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest.jsonl"
    drops_path = out_dir / "drops.jsonl"

    index = build_provenance_index(provenance_dir)

    with manifest_out.open("w", encoding="utf-8") as out_fh, \
         drops_path.open("w", encoding="utf-8") as drops_fh:
        for path in sorted(corpus_dir.glob("*.txt")):
            doc_id = path.stem
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                drops_fh.write(json.dumps({"doc_id": doc_id, "drop_reason": "decode_error"}) + "\n")
                continue
            source = resolve_source(path.name, index)
            row = {
                "doc_id": doc_id,
                "source": source,
                "text": text,
                "word_count": len(text.split()),
            }
            out_fh.write(json.dumps(row) + "\n")
    return manifest_out
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_ingest.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/ingest.py apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py && git commit -m "feat(cpt-pipeline): ingest corpus to manifest with stable doc_ids"
```

---

## Task 27: ingest — source field reflects coarse + fine resolution

**Group:** 2 (sequential within Track F — depends on Task 26)

**Behavior being verified:** known YouTube docs get `source = "youtube:tonebase"`; known OpenAlex pdf docs get `source = "academic_pdf:openalex"`.
**Interface under test:** `run_ingest(corpus_dir, provenance_dir, out_dir)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py`:
```python
def test_source_field_resolves_coarse_and_fine(tiny_corpus, fixture_ids, tmp_path):
    corpus_dir, provenance_dir = tiny_corpus
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest_path = run_ingest(corpus_dir, provenance_dir, out_dir)

    rows = {r["doc_id"]: r for r in _read_jsonl(manifest_path)}
    assert rows["abcdefghijk"]["source"] == "youtube:tonebase"
    assert rows[f"pdf_{fixture_ids['pdf_h_1']}"]["source"] == "academic_pdf:openalex"
```

- [ ] **Step 2: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_ingest.py::test_source_field_resolves_coarse_and_fine -xvs
```
Expected: PASS — Task 26 already calls `resolve_source` with the built index.

- [ ] **Step 3: No implementation needed**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_ingest.py -xvs
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py && git commit -m "test(cpt-pipeline): guard source field resolution in ingest"
```

---

## Task 28: ingest — corrupt UTF-8 file logged to drops.jsonl

**Group:** 2 (sequential within Track F — depends on Task 27)

**Behavior being verified:** a file that fails UTF-8 decode is recorded in `drops.jsonl` with `drop_reason="decode_error"`, and the pipeline continues.
**Interface under test:** `run_ingest(corpus_dir, provenance_dir, out_dir)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py`:
```python
def test_corrupt_file_logged_to_drops(tiny_corpus, fixture_ids, tmp_path):
    corpus_dir, provenance_dir = tiny_corpus
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest_path = run_ingest(corpus_dir, provenance_dir, out_dir)

    rows = {r["doc_id"] for r in _read_jsonl(manifest_path)}
    drops = _read_jsonl(out_dir / "drops.jsonl")
    assert fixture_ids["yt_corrupt"] not in rows, "corrupt doc should not be in manifest"
    assert any(d["doc_id"] == fixture_ids["yt_corrupt"] and d["drop_reason"] == "decode_error" for d in drops)
    # Other docs survived (pipeline continued)
    assert "abcdefghijk" in rows
```

- [ ] **Step 2: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_ingest.py::test_corrupt_file_logged_to_drops -xvs
```
Expected: PASS — Task 26 catches `UnicodeDecodeError` and writes to `drops.jsonl`.

- [ ] **Step 3: No implementation needed**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_ingest.py -xvs
```
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py && git commit -m "test(cpt-pipeline): guard decode error logged to drops"
```

---

## Task 29: ingest — word count reflects whitespace-split token count

**Group:** 2 (sequential within Track F — depends on Task 28)

**Behavior being verified:** `word_count` for a known fixture doc equals `len(text.split())`.
**Interface under test:** `run_ingest(corpus_dir, provenance_dir, out_dir)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py`:
```python
def test_word_count_matches_whitespace_split(tiny_corpus, tmp_path):
    corpus_dir, provenance_dir = tiny_corpus
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest_path = run_ingest(corpus_dir, provenance_dir, out_dir)

    rows = {r["doc_id"]: r for r in _read_jsonl(manifest_path)}
    row = rows["abcdefghijk"]
    expected = len(row["text"].split())
    assert row["word_count"] == expected, f"expected {expected}, got {row['word_count']}"
```

- [ ] **Step 2: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_ingest.py::test_word_count_matches_whitespace_split -xvs
```
Expected: PASS — Task 26's impl already sets `word_count = len(text.split())`.

- [ ] **Step 3: No implementation needed**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_ingest.py -xvs
```
Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py && git commit -m "test(cpt-pipeline): guard word_count is whitespace-split tokens"
```

---

## Task 30: ingest — orphan file (no provenance row) gets `:unknown` fine source

**Group:** 2 (sequential within Track F — depends on Task 29)

**Behavior being verified:** a corpus .txt file with no matching provenance row is included in the manifest with `source` ending in `:unknown`.
**Interface under test:** `run_ingest(corpus_dir, provenance_dir, out_dir)`.

**Files:**
- Modify: `apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

Append to `apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py`:
```python
def test_orphan_file_gets_unknown_fine_source(tiny_corpus, tmp_path):
    corpus_dir, provenance_dir = tiny_corpus
    # The fixture's "DUP1234567x" doc has no provenance row tied to its hash for web/pdf prefix,
    # but it's a YouTube-pattern stem. Use the corrupt or non-ascii doc which also has no provenance.
    # Add a fresh orphan to be explicit.
    (corpus_dir / "ORPHAN12345.txt").write_text(
        "Solid pedagogical content about practice strategy here for orphan test purposes.",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest_path = run_ingest(corpus_dir, provenance_dir, out_dir)

    rows = {r["doc_id"]: r for r in _read_jsonl(manifest_path)}
    assert rows["ORPHAN12345"]["source"] == "youtube:unknown"
```

- [ ] **Step 2: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_ingest.py::test_orphan_file_gets_unknown_fine_source -xvs
```
Expected: PASS — Task 5's `resolve_source` returns `<coarse>:unknown` when stem isn't in index.

- [ ] **Step 3: No implementation needed**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_ingest.py -xvs
```
Expected: PASS (all five tests)

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/tests/test_ingest.py && git commit -m "test(cpt-pipeline): guard orphan file gets unknown fine source"
```

---

# Track G: pipeline driver + E2E (Task 31)

## Task 31: pipeline — end-to-end run on tiny_corpus produces dataset

**Group:** 3 (depends on all prior tracks complete)

**Behavior being verified:** `run_pipeline` with `--push-disabled` (test mode) executes stages 1-4 against tiny_corpus and produces a final train/validation manifest pair where boilerplate is stripped, dups collapsed, French/short/non-ASCII docs dropped, and refs stripped from PDF docs.
**Interface under test:** `run_pipeline(argv)`.

**Files:**
- Create: `apps/evals/teacher_model/cpt_pipeline/pipeline.py`
- Create: `apps/evals/teacher_model/cpt_pipeline/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`apps/evals/teacher_model/cpt_pipeline/tests/test_pipeline.py`:
```python
"""End-to-end pipeline test."""
import json
from pathlib import Path

from teacher_model.cpt_pipeline.pipeline import run_pipeline


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_end_to_end_produces_clean_dataset(tiny_corpus, fixture_ids, tmp_path):
    corpus_dir, provenance_dir = tiny_corpus

    exit_code = run_pipeline([
        "run",
        "--corpus-dir", str(corpus_dir),
        "--provenance-dir", str(provenance_dir),
        "--out-dir", str(tmp_path / "pipeline_out"),
        "--repo-id", "Jai-D/test-repo",
        "--push-disabled",
    ])
    assert exit_code == 0, f"pipeline exited with {exit_code}"

    train = _read_jsonl(tmp_path / "pipeline_out" / "4_split" / "train.jsonl")
    val = _read_jsonl(tmp_path / "pipeline_out" / "4_split" / "validation.jsonl")
    all_rows = train + val
    surviving_ids = {r["doc_id"] for r in all_rows}

    # Dropped expected:
    assert fixture_ids["yt_short"] not in surviving_ids, "short doc not dropped"
    assert fixture_ids["yt_french"] not in surviving_ids, "french doc not dropped"
    assert fixture_ids["yt_nonascii"] not in surviving_ids, "non-ASCII doc not dropped"
    assert fixture_ids["yt_corrupt"] not in surviving_ids, "corrupt doc not dropped"

    # Dedup of two near-dup docs: only one should survive
    dup_ids = {fixture_ids["yt_dup_a"], fixture_ids["yt_dup_b"]}
    surviving_dups = surviving_ids & dup_ids
    assert len(surviving_dups) == 1, f"expected exactly one dup to survive, got {surviving_dups}"

    # 3c boilerplate (legal disclaimer, present in 25 web docs) should be stripped
    boilerplate = "The author and publisher disclaim all such representations and warranties for a particular purpose."
    assert all(boilerplate not in r["text"] for r in all_rows), "boilerplate not stripped corpus-wide"

    # Refs stripped on PDF docs
    pdf_doc = next((r for r in all_rows if r["doc_id"] == f"pdf_{fixture_ids['pdf_h_refs']}"), None)
    if pdf_doc is not None:
        assert "Debussy, C. (1905)" not in pdf_doc["text"], "refs not stripped on academic_pdf doc"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_pipeline.py -xvs
```
Expected: FAIL — `ModuleNotFoundError: No module named 'teacher_model.cpt_pipeline.pipeline'`

- [ ] **Step 3: Implement the minimum to make the test pass**

`apps/evals/teacher_model/cpt_pipeline/pipeline.py`:
```python
"""CLI driver for the cpt_pipeline.

Usage:
  uv run python -m teacher_model.cpt_pipeline.pipeline run \\
    --corpus-dir <path> --provenance-dir <path> --out-dir <path> \\
    --repo-id <hf_repo_id> [--push-disabled]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from teacher_model.cpt_pipeline.dedup import run_dedup
from teacher_model.cpt_pipeline.hf_publish import run_publish
from teacher_model.cpt_pipeline.ingest import run_ingest
from teacher_model.cpt_pipeline.split import run_split
from teacher_model.cpt_pipeline.structural_filter import run_filter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPT corpus preprocessing pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Run all pipeline stages.")
    run_p.add_argument("--corpus-dir", type=Path, required=True)
    run_p.add_argument("--provenance-dir", type=Path, required=True)
    run_p.add_argument("--out-dir", type=Path, required=True)
    run_p.add_argument("--repo-id", type=str, required=True)
    run_p.add_argument("--push-disabled", action="store_true",
                       help="Skip stage 5 (HF Hub push). Useful for tests and dry runs.")
    run_p.add_argument("--seed", type=int, default=42)
    return parser


def run_pipeline(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[stage 1/5] ingest: {args.corpus_dir} -> {out}/1_ingest")
    ingest_manifest = run_ingest(args.corpus_dir, args.provenance_dir, out / "1_ingest")
    print(f"[stage 2/5] structural_filter: {ingest_manifest} -> {out}/2_filter")
    filter_manifest = run_filter(ingest_manifest, out / "2_filter")
    print(f"[stage 3/5] dedup: {filter_manifest} -> {out}/3_dedup")
    dedup_manifest = run_dedup(filter_manifest, out / "3_dedup")
    print(f"[stage 4/5] split: {dedup_manifest} -> {out}/4_split")
    train_path, val_path = run_split(dedup_manifest, out / "4_split", seed=args.seed)

    if args.push_disabled:
        print("[stage 5/5] push DISABLED via --push-disabled")
        return 0

    print(f"[stage 5/5] hf_publish: {args.repo_id} (private)")
    url = run_publish(train_path, val_path, args.repo_id, private=True, card_out_dir=out / "5_publish")
    print(f"published: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline(sys.argv[1:]))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/test_pipeline.py -xvs
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/evals/teacher_model/cpt_pipeline/pipeline.py apps/evals/teacher_model/cpt_pipeline/tests/test_pipeline.py && git commit -m "feat(cpt-pipeline): CLI driver and end-to-end integration"
```

---

## Final verification

After Task 31 commits, run the full test suite to confirm no cross-task regressions:

```bash
cd apps/evals && uv run pytest teacher_model/cpt_pipeline/tests/ -xvs
```
Expected: all 30 tests across 7 test files PASS.
