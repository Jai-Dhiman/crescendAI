"""Source resolution behavior tests."""
from teacher_model.cpt_pipeline.source_resolver import build_provenance_index, resolve_source


def test_youtube_filename_resolves_to_youtube_coarse():
    result = resolve_source("abcdefghijk.txt", {})
    assert result.startswith("youtube:"), f"expected youtube: prefix, got {result!r}"


def test_pdf_filename_resolves_to_academic_pdf_coarse():
    result = resolve_source("pdf_0123456789ab.txt", {})
    assert result.startswith("academic_pdf:"), f"expected academic_pdf: prefix, got {result!r}"


def test_web_filename_resolves_to_web_scrape_coarse():
    result = resolve_source("web_0123456789ab.txt", {})
    assert result.startswith("web_scrape:"), f"expected web_scrape: prefix, got {result!r}"


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


def test_malformed_filename_returns_unknown_coarse_and_fine():
    # 5-char stem, doesn't match any pattern
    assert resolve_source("short.txt", {}) == "unknown:unknown"
    # 12-char stem (one off from youtube)
    assert resolve_source("twelvecharsxx.txt", {}) == "unknown:unknown"
    # pdf_ prefix but wrong hash length
    assert resolve_source("pdf_short.txt", {}) == "unknown:unknown"
